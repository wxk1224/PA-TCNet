"""
PA-TCNet model implementation.

This single file follows the paper-level naming:

1. Local sensorimotor pattern encoding
2. Temporal token position encoding
3. Pathology-aware Rhythmic State Mamba (PRSM)
4. Motor-imagery classification head
"""

import math

import torch
import torch.nn.functional as F
from einops import einsum, repeat
from einops.layers.torch import Rearrange
from torch import nn


class DepthwisePointwiseConv2d(nn.Module):
    """Depthwise temporal filtering followed by pointwise channel fusion."""

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=padding,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)

    def forward(self, features):
        return self.pointwise(self.depthwise(features))


class LocalSensorimotorPatternEncoder(nn.Module):
    """PA-TCNet module: local sensorimotor pattern encoding into temporal tokens."""

    def __init__(
        self,
        temporal_filters_per_branch=10,
        spatial_multiplier=3,
        first_pool_size=8,
        second_pool_size=8,
        dropout_rate=0.3,
        num_channels=22,
        embedding_dim=40,
        temporal_kernel_sizes=(36, 24, 18),
        fusion_kernel_size=16,
    ):
        super().__init__()
        if len(temporal_kernel_sizes) != 3:
            raise ValueError("temporal_kernel_sizes must contain exactly 3 kernel sizes.")

        temporal_feature_channels = temporal_filters_per_branch * len(temporal_kernel_sizes)
        spatial_feature_channels = temporal_feature_channels * spatial_multiplier

        self.multi_scale_temporal_filters = nn.ModuleList(
            [
                nn.Conv2d(
                    1,
                    temporal_filters_per_branch,
                    kernel_size=(1, kernel_size),
                    padding="same",
                    bias=False,
                )
                for kernel_size in temporal_kernel_sizes
            ]
        )
        self.activation = nn.ELU()
        self.temporal_batch_norm = nn.BatchNorm2d(temporal_feature_channels)

        self.sensorimotor_spatial_filter = nn.Conv2d(
            temporal_feature_channels,
            spatial_feature_channels,
            kernel_size=(num_channels, 1),
            groups=temporal_feature_channels,
            padding="valid",
            bias=False,
        )
        self.spatial_batch_norm = nn.BatchNorm2d(spatial_feature_channels)
        self.spatial_temporal_pool = nn.MaxPool2d(
            kernel_size=(1, first_pool_size),
            stride=(1, first_pool_size),
        )
        self.spatial_dropout = nn.Dropout(dropout_rate)

        self.compact_fusion_filter = DepthwisePointwiseConv2d(
            spatial_feature_channels,
            spatial_feature_channels,
            kernel_size=(1, fusion_kernel_size),
            padding="same",
        )
        self.fusion_batch_norm = nn.BatchNorm2d(spatial_feature_channels)
        self.token_pool = nn.MaxPool2d(
            kernel_size=(1, second_pool_size),
            stride=(1, second_pool_size),
        )
        self.token_dropout = nn.Dropout(dropout_rate)

        self.token_sequence_projection = Rearrange(
            "batch channel height width -> batch (height width) channel"
        )
        self.token_embedding_projection = (
            nn.Identity()
            if spatial_feature_channels == embedding_dim
            else nn.Linear(spatial_feature_channels, embedding_dim)
        )

    def forward(self, eeg_trial):
        temporal_responses = torch.cat(
            [temporal_filter(eeg_trial) for temporal_filter in self.multi_scale_temporal_filters],
            dim=1,
        )
        temporal_features = self.temporal_batch_norm(self.activation(temporal_responses))

        spatial_features = self.sensorimotor_spatial_filter(temporal_features)
        spatial_features = self.spatial_batch_norm(spatial_features)
        spatial_features = self.activation(spatial_features)
        spatial_features = self.spatial_temporal_pool(spatial_features)
        spatial_features = self.spatial_dropout(spatial_features)

        fused_features = self.compact_fusion_filter(spatial_features)
        fused_features = self.fusion_batch_norm(fused_features)
        fused_features = self.activation(fused_features)
        fused_features = self.token_pool(fused_features)
        fused_features = self.token_dropout(fused_features)

        sensorimotor_tokens = self.token_sequence_projection(fused_features)
        sensorimotor_tokens = self.token_embedding_projection(sensorimotor_tokens)
        return sensorimotor_tokens


class TemporalTokenPositionEncoding(nn.Module):
    """PA-TCNet module: position injection for sensorimotor temporal tokens."""

    def __init__(self, embedding_dim, max_token_length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, max_token_length, embedding_dim))

    def forward(self, sensorimotor_tokens):
        positioned_tokens = sensorimotor_tokens + self.encoding[:, : sensorimotor_tokens.shape[1], :].to(
            sensorimotor_tokens.device
        )
        return self.dropout(positioned_tokens)


def _init_average_lowpass_filter(depthwise_conv):
    kernel_size = depthwise_conv.kernel_size[0]
    with torch.no_grad():
        depthwise_conv.weight.zero_()
        depthwise_conv.weight[:, :, :] = 1.0 / kernel_size


class RhythmicTokenDecomposition(nn.Module):
    """PRSM submodule: split normalized tokens into slow and fast temporal branches."""

    def __init__(self, embedding_dim, lowpass_kernel_size=9):
        super().__init__()
        self.slow_rhythm_filter = nn.Conv1d(
            embedding_dim,
            embedding_dim,
            kernel_size=lowpass_kernel_size,
            padding=lowpass_kernel_size // 2,
            groups=embedding_dim,
            bias=False,
        )
        _init_average_lowpass_filter(self.slow_rhythm_filter)

    def forward(self, normalized_tokens):
        channel_first_tokens = normalized_tokens.transpose(1, 2)
        slow_rhythm_tokens = self.slow_rhythm_filter(channel_first_tokens)
        fast_transient_tokens = channel_first_tokens - slow_rhythm_tokens
        return slow_rhythm_tokens, fast_transient_tokens


class RhythmicContextConstructor(nn.Module):
    """PRSM submodule: fuse slow rhythmic context and fast transient details."""

    def __init__(
        self,
        embedding_dim,
        slow_kernels=(5, 9, 13),
        fast_kernels=(3, 5, 7),
        rhythmic_branch_mode="full",
    ):
        super().__init__()
        self.rhythmic_branch_mode = rhythmic_branch_mode
        self.rhythmic_decomposition = RhythmicTokenDecomposition(embedding_dim, lowpass_kernel_size=9)

        self.slow_context_filters = nn.ModuleList(
            [
                nn.Conv1d(
                    embedding_dim,
                    embedding_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=embedding_dim,
                    bias=False,
                )
                for kernel_size in slow_kernels
            ]
        )
        self.fast_detail_filters = nn.ModuleList(
            [
                nn.Conv1d(
                    embedding_dim,
                    embedding_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=embedding_dim,
                    bias=False,
                )
                for kernel_size in fast_kernels
            ]
        )

        self.slow_branch_fusion = nn.Sequential(
            nn.Conv1d(embedding_dim * len(slow_kernels), embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
        )
        self.fast_branch_fusion = nn.Sequential(
            nn.Conv1d(embedding_dim * len(fast_kernels), embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
        )
        self.pathology_context_projection = nn.Sequential(
            nn.Conv1d(embedding_dim * 2, embedding_dim * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.SiLU(),
            nn.Conv1d(embedding_dim * 2, embedding_dim, kernel_size=1, bias=False),
        )
        self.context_normalization = nn.LayerNorm(embedding_dim)

    def forward(self, normalized_tokens):
        slow_rhythm_tokens, fast_transient_tokens = self.rhythmic_decomposition(normalized_tokens)

        slow_multiscale_features = torch.cat(
            [slow_filter(slow_rhythm_tokens) for slow_filter in self.slow_context_filters],
            dim=1,
        )
        fast_multiscale_features = torch.cat(
            [fast_filter(fast_transient_tokens) for fast_filter in self.fast_detail_filters],
            dim=1,
        )

        slow_rhythm_context = self.slow_branch_fusion(slow_multiscale_features) + slow_rhythm_tokens
        fast_transient_context = self.fast_branch_fusion(fast_multiscale_features) + fast_transient_tokens

        if self.rhythmic_branch_mode == "slow_only":
            selected_slow_context = slow_rhythm_context
            selected_fast_context = torch.zeros_like(fast_transient_context)
            residual_context = slow_rhythm_context
        elif self.rhythmic_branch_mode == "fast_only":
            selected_slow_context = torch.zeros_like(slow_rhythm_context)
            selected_fast_context = fast_transient_context
            residual_context = fast_transient_context
        else:
            selected_slow_context = slow_rhythm_context
            selected_fast_context = fast_transient_context
            residual_context = 0.5 * (slow_rhythm_context + fast_transient_context)

        fused_pathology_context = self.pathology_context_projection(
            torch.cat([selected_slow_context, selected_fast_context], dim=1)
        )
        fused_pathology_context = fused_pathology_context + residual_context
        rhythmic_context = self.context_normalization(fused_pathology_context.transpose(1, 2))

        return (
            slow_rhythm_context.transpose(1, 2),
            fast_transient_context.transpose(1, 2),
            rhythmic_context,
        )


class ContextGuidedStateModulation(nn.Module):
    """PRSM submodule: produce adaptive scaling and residual bias from rhythmic context."""

    def __init__(self, embedding_dim, state_inner_dim):
        super().__init__()
        self.context_normalization = nn.LayerNorm(embedding_dim)
        self.input_drive_scale = nn.Linear(embedding_dim, state_inner_dim)
        self.residual_gate_bias = nn.Linear(embedding_dim, state_inner_dim)

    def forward(self, rhythmic_context):
        normalized_context = self.context_normalization(rhythmic_context)
        input_drive_scale = torch.sigmoid(self.input_drive_scale(normalized_context))
        residual_gate_bias = self.residual_gate_bias(normalized_context)
        return input_drive_scale, residual_gate_bias


class ContextGuidedSelectiveStateModel(nn.Module):
    """PRSM submodule: context-guided selective state propagation."""

    def __init__(self, embedding_dim, use_context_modulation=True, rhythmic_branch_mode="full"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.state_inner_dim = embedding_dim * 2
        self.delta_rank = math.ceil(embedding_dim / 16)
        self.state_dim = 16
        self.use_context_modulation = use_context_modulation
        self.rhythmic_branch_mode = rhythmic_branch_mode

        self.input_projection = nn.Linear(self.embedding_dim, self.state_inner_dim * 2)
        self.local_temporal_filter = nn.Conv1d(
            in_channels=self.state_inner_dim,
            out_channels=self.state_inner_dim,
            kernel_size=3,
            groups=self.state_inner_dim,
            padding=2,
        )
        self.state_parameter_projection = nn.Linear(
            self.state_inner_dim,
            self.delta_rank + self.state_dim * 2,
            bias=False,
        )
        self.delta_projection = nn.Linear(self.delta_rank, self.state_inner_dim, bias=True)

        state_index = repeat(
            torch.arange(1, self.state_dim + 1),
            "state -> channel state",
            channel=self.state_inner_dim,
        )
        self.log_state_transition = nn.Parameter(torch.log(state_index))
        self.skip_connection = nn.Parameter(torch.ones(self.state_inner_dim))
        self.output_projection = nn.Linear(self.state_inner_dim, self.embedding_dim)
        self.context_modulation = (
            ContextGuidedStateModulation(self.embedding_dim, self.state_inner_dim)
            if use_context_modulation
            else None
        )

    def _run_selective_state_space(self, state_input_tokens):
        _, state_dim = self.log_state_transition.shape
        state_transition = -torch.exp(self.log_state_transition.float())
        skip_connection = self.skip_connection.float()

        projected_state_params = self.state_parameter_projection(state_input_tokens)
        delta_params, input_coefficients, output_coefficients = projected_state_params.split(
            split_size=[self.delta_rank, state_dim, state_dim],
            dim=-1,
        )
        discretized_delta = F.softplus(self.delta_projection(delta_params))
        return self._selective_state_scan(
            state_input_tokens,
            discretized_delta,
            state_transition,
            input_coefficients,
            output_coefficients,
            skip_connection,
        )

    def _selective_state_scan(
        self,
        state_input_tokens,
        discretized_delta,
        state_transition,
        input_coefficients,
        output_coefficients,
        skip_connection,
    ):
        batch_size, sequence_length, inner_dim = state_input_tokens.shape
        state_dim = state_transition.shape[1]

        discretized_transition = torch.exp(
            einsum(
                discretized_delta,
                state_transition,
                "batch token channel, channel state -> batch token channel state",
            )
        )
        discretized_input = einsum(
            discretized_delta,
            input_coefficients,
            state_input_tokens,
            "batch token channel, batch token state, batch token channel -> batch token channel state",
        )

        recurrent_state = torch.zeros(
            (batch_size, inner_dim, state_dim),
            device=discretized_transition.device,
            dtype=discretized_transition.dtype,
        )
        state_outputs = []
        for token_index in range(sequence_length):
            recurrent_state = (
                discretized_transition[:, token_index] * recurrent_state
                + discretized_input[:, token_index]
            )
            token_output = einsum(
                recurrent_state,
                output_coefficients[:, token_index, :],
                "batch channel state, batch state -> batch channel",
            )
            state_outputs.append(token_output)

        state_output_tokens = torch.stack(state_outputs, dim=1)
        return state_output_tokens + state_input_tokens * skip_connection

    def forward(self, normalized_tokens, rhythmic_context=None, return_details=False):
        _, sequence_length, _ = normalized_tokens.shape

        projected_branches = self.input_projection(normalized_tokens)
        state_input_tokens, residual_gate_tokens = projected_branches.split(
            split_size=[self.state_inner_dim, self.state_inner_dim],
            dim=-1,
        )

        input_drive_scale = None
        residual_gate_bias = None
        if (
            self.use_context_modulation
            and self.rhythmic_branch_mode != "none"
            and rhythmic_context is not None
        ):
            input_drive_scale, residual_gate_bias = self.context_modulation(rhythmic_context)
            state_input_tokens = state_input_tokens * (1.0 + input_drive_scale)
            residual_gate_tokens = residual_gate_tokens + residual_gate_bias

        local_temporal_tokens = state_input_tokens.transpose(1, 2)
        local_temporal_tokens = self.local_temporal_filter(local_temporal_tokens)[:, :, :sequence_length]
        local_temporal_tokens = local_temporal_tokens.transpose(1, 2)

        activated_state_input = F.silu(local_temporal_tokens)
        state_output_tokens = self._run_selective_state_space(activated_state_input)
        gated_state_output = state_output_tokens * F.silu(residual_gate_tokens)
        output_tokens = self.output_projection(gated_state_output)

        if not return_details:
            return output_tokens
        return output_tokens, {
            "input_drive_scale": input_drive_scale,
            "residual_gate_bias": residual_gate_bias,
        }


class PathologyAwareRhythmicStateMamba(nn.Module):
    """Pathology-aware Rhythmic State Mamba (PRSM) module."""

    def __init__(
        self,
        embedding_dim,
        dropout_rate=0.3,
        use_context_modulation=True,
        rhythmic_branch_mode="full",
    ):
        super().__init__()
        self.token_normalization = nn.LayerNorm(embedding_dim)
        self.use_context_modulation = use_context_modulation
        self.rhythmic_branch_mode = rhythmic_branch_mode
        self.rhythmic_context_constructor = (
            RhythmicContextConstructor(embedding_dim, rhythmic_branch_mode=rhythmic_branch_mode)
            if use_context_modulation
            else None
        )
        self.selective_state_model = ContextGuidedSelectiveStateModel(
            embedding_dim=embedding_dim,
            use_context_modulation=use_context_modulation,
            rhythmic_branch_mode=rhythmic_branch_mode,
        )
        self.residual_dropout = nn.Dropout(dropout_rate)

    def forward(self, input_tokens, return_details=False):
        normalized_tokens = self.token_normalization(input_tokens)
        rhythmic_context = None
        slow_rhythm_context = None
        fast_transient_context = None

        if self.use_context_modulation and self.rhythmic_branch_mode != "none":
            slow_rhythm_context, fast_transient_context, rhythmic_context = self.rhythmic_context_constructor(
                normalized_tokens
            )

        if return_details:
            calibrated_tokens, state_model_details = self.selective_state_model(
                normalized_tokens,
                rhythmic_context,
                return_details=True,
            )
        else:
            calibrated_tokens = self.selective_state_model(normalized_tokens, rhythmic_context)
            state_model_details = None

        output_tokens = input_tokens + self.residual_dropout(calibrated_tokens)
        if not return_details:
            return output_tokens
        return output_tokens, {
            "block_input": input_tokens,
            "normalized_tokens": normalized_tokens,
            "slow_rhythm_context": slow_rhythm_context,
            "fast_transient_context": fast_transient_context,
            "rhythmic_context": rhythmic_context,
            "input_drive_scale": None if state_model_details is None else state_model_details["input_drive_scale"],
            "residual_gate_bias": None if state_model_details is None else state_model_details["residual_gate_bias"],
            "block_output": output_tokens,
        }


class MotorImageryClassificationHead(nn.Module):
    """PA-TCNet module: final motor-imagery decision head."""

    def __init__(self, flattened_feature_dim, num_classes):
        super().__init__()
        self.decision_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flattened_feature_dim, num_classes),
        )

    def forward(self, flattened_features):
        return self.decision_layer(flattened_features)


class PATCNet(nn.Module):
    """Pathology-Aware Temporal Calibration Network (PA-TCNet)."""

    def __init__(
        self,
        embedding_dim=64,
        depth=2,
        temporal_filters_per_branch=10,
        spatial_multiplier=3,
        first_pool_size=8,
        second_pool_size=8,
        dropout_rate=0.3,
        num_channels=22,
        num_classes=2,
        flattened_feature_dim=600,
        use_context_modulation=True,
        use_prsm_backbone=True,
        rhythmic_branch_mode="full",
        temporal_kernel_sizes=(36, 24, 18),
        fusion_kernel_size=16,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_prsm_backbone = use_prsm_backbone
        self.rhythmic_branch_mode = rhythmic_branch_mode

        self.sensorimotor_encoder = LocalSensorimotorPatternEncoder(
            temporal_filters_per_branch=temporal_filters_per_branch,
            spatial_multiplier=spatial_multiplier,
            first_pool_size=first_pool_size,
            second_pool_size=second_pool_size,
            dropout_rate=dropout_rate,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            temporal_kernel_sizes=temporal_kernel_sizes,
            fusion_kernel_size=fusion_kernel_size,
        )
        self.position_encoding = (
            TemporalTokenPositionEncoding(embedding_dim, dropout=0.1)
            if use_prsm_backbone
            else nn.Identity()
        )
        self.prsm_backbone = (
            nn.Sequential(
                *[
                    PathologyAwareRhythmicStateMamba(
                        embedding_dim=embedding_dim,
                        dropout_rate=dropout_rate,
                        use_context_modulation=use_context_modulation,
                        rhythmic_branch_mode=rhythmic_branch_mode,
                    )
                    for _ in range(depth)
                ]
            )
            if use_prsm_backbone
            else nn.Identity()
        )
        self.final_token_norm = nn.LayerNorm(embedding_dim)
        self.flatten_tokens = nn.Flatten()
        self.classification_head = MotorImageryClassificationHead(flattened_feature_dim, num_classes)

    def forward_features(self, eeg_trial, return_block_details=False):
        if eeg_trial.dim() == 3:
            eeg_trial = eeg_trial.unsqueeze(1)

        sensorimotor_tokens = self.sensorimotor_encoder(eeg_trial)
        export = {"sensorimotor_tokens": sensorimotor_tokens}

        if not self.use_prsm_backbone:
            normalized_tokens = self.final_token_norm(sensorimotor_tokens)
            flattened_features = self.flatten_tokens(normalized_tokens)
            class_logits = self.classification_head(flattened_features)
            export.update(
                {
                    "positioned_tokens": sensorimotor_tokens,
                    "encoder_features": normalized_tokens,
                    "flatten_features": flattened_features,
                    "logits": class_logits,
                }
            )
            return export

        scaled_sensorimotor_tokens = sensorimotor_tokens * math.sqrt(self.embedding_dim)
        positioned_tokens = self.position_encoding(scaled_sensorimotor_tokens)
        temporal_tokens = positioned_tokens

        prsm_block_details = []
        for block_index, prsm_block in enumerate(self.prsm_backbone):
            if return_block_details:
                temporal_tokens, block_details = prsm_block(temporal_tokens, return_details=True)
                block_details["block_index"] = block_index
                prsm_block_details.append(block_details)
            else:
                temporal_tokens = prsm_block(temporal_tokens)

        normalized_tokens = self.final_token_norm(temporal_tokens)
        flattened_features = self.flatten_tokens(normalized_tokens)
        class_logits = self.classification_head(flattened_features)

        export.update(
            {
                "positioned_tokens": positioned_tokens,
                "encoder_features": normalized_tokens,
                "flatten_features": flattened_features,
                "logits": class_logits,
            }
        )

        if return_block_details and prsm_block_details:
            final_block = prsm_block_details[-1]
            export["encoder_block_details"] = prsm_block_details
            export.update(
                {
                    "slow_rhythm_context": final_block["slow_rhythm_context"],
                    "fast_transient_context": final_block["fast_transient_context"],
                    "rhythmic_context": final_block["rhythmic_context"],
                    "input_drive_scale": final_block["input_drive_scale"],
                    "residual_gate_bias": final_block["residual_gate_bias"],
                }
            )
        return export

    def forward(self, eeg_trial, return_features=False, return_block_details=False):
        export = self.forward_features(eeg_trial, return_block_details=return_block_details)
        if return_features:
            return export
        return export["logits"]


def get_model(args):
    return PATCNet(
        embedding_dim=args.emb_size,
        depth=args.depth,
        temporal_filters_per_branch=args.temporal_filters_per_branch,
        spatial_multiplier=args.spatial_multiplier,
        first_pool_size=args.pooling_size1,
        second_pool_size=args.pooling_size2,
        dropout_rate=args.dropout,
        num_channels=args.n_channels,
        num_classes=args.n_classes,
        flattened_feature_dim=args.flatten,
        use_context_modulation=getattr(args, "use_context_modulation", True),
        use_prsm_backbone=getattr(args, "use_prsm_backbone", True),
        rhythmic_branch_mode=getattr(args, "rhythmic_branch_mode", "full"),
    )

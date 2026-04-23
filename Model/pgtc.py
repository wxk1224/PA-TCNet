import torch
import torch.nn.functional as F


def _summarize_calibration(probabilities, accepted_mask):
    confidences, _ = torch.max(probabilities, dim=1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
    rejected_mask = ~accepted_mask

    def masked_mean(values, mask):
        if int(mask.sum().item()) == 0:
            return 0.0
        return float(values[mask].mean().item())

    return {
        "accepted_count": int(accepted_mask.sum().item()),
        "rejected_count": int(rejected_mask.sum().item()),
        "accepted_confidence": masked_mean(confidences, accepted_mask),
        "rejected_confidence": masked_mean(confidences, rejected_mask),
        "accepted_entropy": masked_mean(entropy, accepted_mask),
        "rejected_entropy": masked_mean(entropy, rejected_mask),
    }


def apply_confidence_calibration(probabilities, prob_threshold=0.90, existing_targets=None):
    if probabilities.ndim != 2:
        raise ValueError(f"Expected probabilities with shape [B, C], got {probabilities.shape}")

    confidences, predictions = torch.max(probabilities, dim=1)
    accepted_mask = confidences >= prob_threshold

    calibrated_targets = torch.zeros_like(probabilities)
    calibrated_targets[accepted_mask, predictions[accepted_mask]] = 1.0

    if existing_targets is not None:
        updated_targets = existing_targets.clone()
        updated_targets[accepted_mask] = calibrated_targets[accepted_mask]
    else:
        updated_targets = calibrated_targets

    summary = _summarize_calibration(probabilities, accepted_mask)
    summary.update(
        {
            "updated_targets": updated_targets,
            "accepted_mask": accepted_mask,
            "predictions": predictions,
            "confidences": confidences,
            "similarities": None,
        }
    )
    return summary


def apply_pgtc_calibration(
    probabilities,
    roi_vectors,
    templates,
    class_thresholds,
    prob_threshold=0.90,
    existing_targets=None,
):
    if probabilities.ndim != 2:
        raise ValueError(f"Expected probabilities with shape [B, C], got {probabilities.shape}")
    if roi_vectors.ndim != 2:
        raise ValueError(f"Expected roi_vectors with shape [B, D], got {roi_vectors.shape}")

    roi_vectors = F.normalize(roi_vectors, p=2, dim=1)
    templates = F.normalize(templates, p=2, dim=1)

    confidences, predictions = torch.max(probabilities, dim=1)
    predicted_templates = templates[predictions]
    predicted_thresholds = class_thresholds[predictions]
    similarities = torch.sum(roi_vectors * predicted_templates, dim=1)

    accepted_mask = (confidences >= prob_threshold) & (similarities >= predicted_thresholds)

    calibrated_targets = torch.zeros_like(probabilities)
    calibrated_targets[accepted_mask, predictions[accepted_mask]] = 1.0

    if existing_targets is not None:
        updated_targets = existing_targets.clone()
        updated_targets[accepted_mask] = calibrated_targets[accepted_mask]
    else:
        updated_targets = calibrated_targets

    summary = _summarize_calibration(probabilities, accepted_mask)
    summary.update(
        {
            "updated_targets": updated_targets,
            "accepted_mask": accepted_mask,
            "predictions": predictions,
            "confidences": confidences,
            "similarities": similarities,
        }
    )
    return summary

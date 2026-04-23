import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset

EXPORT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPORT_ROOT.parent
for path in (EXPORT_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import data_config
from dataloader.TyStrokeLoader import get_dataset as get_ty_dataset
from dataloader.XwStrokeLoader import get_dataset as get_xw_dataset
from model import get_model
from pgtc import apply_pgtc_calibration
from physiology import build_class_templates, compute_roi_features, read_channel_info
from utils import BestModelSaver, EarlyStopping, Logger, compute_metrics, make_dirs, set_device


DEFAULT_XW_SUB_LIST = [2, 5, 8, 9, 11, 12, 14, 17, 21, 23, 24, 26, 27, 28, 30, 32, 33, 37, 38, 43, 44, 47, 49, 50]


class SourceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TargetCalibrationDataset(Dataset):
    def __init__(self, x, roi_vectors, num_classes):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.roi_vectors = torch.tensor(roi_vectors, dtype=torch.float32)
        self.calibrated_targets = torch.zeros((len(x), num_classes), dtype=torch.float32)

    def update_calibrated_targets(self, indices, new_targets):
        self.calibrated_targets[indices] = new_targets

    def active_ratio(self):
        return float((self.calibrated_targets.sum(dim=1) > 0).float().mean().item())

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.calibrated_targets[idx], self.roi_vectors[idx], idx


class TargetEvalDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_dataset_by_name(dataset_name, subject_indices):
    if dataset_name.startswith("XW"):
        return get_xw_dataset(dataset_name, subject_indices)
    if dataset_name.startswith("TY"):
        return get_ty_dataset(dataset_name, subject_indices)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def default_subject_list(dataset_name, n_subjects):
    if dataset_name.startswith("XW"):
        return DEFAULT_XW_SUB_LIST.copy()
    return list(range(1, n_subjects + 1))


def build_dynamic_dataloaders(source_dataset, target_train_dataset, target_eval_dataset, batch_size, num_workers):
    source_batch = min(batch_size, len(source_dataset))
    target_batch = min(batch_size, len(target_train_dataset))

    source_drop_last = len(source_dataset) > 1 and (len(source_dataset) % source_batch == 1)
    target_drop_last = len(target_train_dataset) > 1 and (len(target_train_dataset) % target_batch == 1)

    source_loader = DataLoader(
        source_dataset,
        batch_size=source_batch,
        shuffle=True,
        drop_last=source_drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    target_train_loader = DataLoader(
        target_train_dataset,
        batch_size=target_batch,
        shuffle=True,
        drop_last=target_drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    target_eval_loader = DataLoader(
        target_eval_dataset,
        batch_size=target_batch,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    target_init_loader = DataLoader(
        target_train_dataset,
        batch_size=target_batch,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return source_loader, target_train_loader, target_eval_loader, target_init_loader


def get_loso_task_data(dataset_name, test_subject_id, sub_list):
    x_train_list, y_train_list = [], []
    x_test = y_test = None

    for subject_id in sub_list:
        dataset_index = subject_id - 1
        x_sub, y_sub = load_dataset_by_name(dataset_name, [dataset_index])
        x_sub = x_sub.astype(np.float32)
        y_sub = y_sub.astype(np.int64)

        if subject_id == test_subject_id:
            x_test = x_sub
            y_test = y_sub
        else:
            x_train_list.append(x_sub)
            y_train_list.append(y_sub)

    if x_test is None or y_test is None:
        raise ValueError(f"Test subject {test_subject_id} not found in subject list.")

    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    return x_train, y_train, x_test, y_test


def parameter_count(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def save_final_report(args, total_time, all_metrics):
    report_path = args.result_dir / f"{args.exp_name}_{args.model}_result.txt"
    total_seconds = int(total_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    summary_order = ["Acc", "Kappa", "F1-Score", "Precision", "Recall", "AUC", "Latency(ms)"]

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("Experiment Report\n")
        handle.write(f"Model: {args.model}\n")
        handle.write(f"Dataset: {args.dataset}\n")
        handle.write("Protocol: LOSO PA-TCNet + PGTC\n")
        handle.write("Task Space: left/right\n")
        handle.write("Target Calibration: prediction confidence + ROI-template consistency\n")
        handle.write("Entropy Weighting: disabled\n")
        handle.write(f"PGTC Confidence Threshold: {args.pgtc_confidence_threshold:.2f}\n")
        handle.write(f"PGTC Warmup Epochs: {args.pgtc_warmup_epochs}\n")
        handle.write(f"ROI Threshold Floor: {args.roi_threshold_floor:.2f}\n")
        handle.write(f"Date: {args.exp_time}\n")
        handle.write(f"Total Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}\n")
        handle.write("=" * 80 + "\n")
        for metric_name in summary_order:
            values = np.asarray(all_metrics[metric_name], dtype=np.float64)
            handle.write(f"{metric_name:<12}: {values.mean():.4f} +/- {values.std():.4f}\n")
        handle.write("=" * 80 + "\n")
    return report_path


def initialize_target_calibration(model, target_init_loader, templates, thresholds, device, prob_threshold):
    model.eval()
    total_accepted = 0
    total_samples = 0

    with torch.no_grad():
        for x_t, calibrated_targets, roi_t, idx_t in target_init_loader:
            x_t = x_t.to(device, non_blocking=True)
            calibrated_targets = calibrated_targets.to(device, non_blocking=True)
            roi_t = roi_t.to(device, non_blocking=True)

            logits = model(x_t)
            probs = torch.softmax(logits, dim=1)
            calibration_result = apply_pgtc_calibration(
                probabilities=probs,
                roi_vectors=roi_t,
                templates=templates,
                class_thresholds=thresholds,
                prob_threshold=prob_threshold,
                existing_targets=calibrated_targets,
            )
            target_init_loader.dataset.update_calibrated_targets(
                idx_t,
                calibration_result["updated_targets"].cpu(),
            )
            total_accepted += int(calibration_result["accepted_mask"].sum().item())
            total_samples += len(idx_t)

    return total_accepted / max(total_samples, 1)


def train_source_only_epoch(source_loader, model, optimizer, device):
    model.train()
    total_loss_epoch = 0.0
    total_batches = len(source_loader)
    if total_batches == 0:
        return 0.0

    for x_s, y_s in source_loader:
        x_s = x_s.to(device, non_blocking=True)
        y_s = y_s.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        src_logits = model(x_s)
        loss_source = F.cross_entropy(src_logits, y_s)
        loss_source.backward()
        optimizer.step()
        total_loss_epoch += loss_source.item()

    return total_loss_epoch / total_batches


def train_pgtc_epoch(source_loader, target_loader, model, optimizer, device, alpha, templates, thresholds, prob_threshold):
    model.train()
    total_loss_epoch = 0.0
    total_source_loss = 0.0
    total_target_loss = 0.0
    total_batches = len(source_loader)
    total_accepted = 0
    total_target_samples = 0

    if total_batches == 0:
        return {
            "train_loss": 0.0,
            "source_loss": 0.0,
            "target_loss": 0.0,
            "accepted_ratio": 0.0,
            "active_ratio": 0.0,
        }

    target_iter = iter(target_loader)

    for x_s, y_s in source_loader:
        try:
            x_t, target_calibration, roi_t, tgt_idx = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            x_t, target_calibration, roi_t, tgt_idx = next(target_iter)

        x_s = x_s.to(device, non_blocking=True)
        y_s = y_s.to(device, non_blocking=True)
        x_t = x_t.to(device, non_blocking=True)
        target_calibration = target_calibration.to(device, non_blocking=True)
        roi_t = roi_t.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        src_logits = model(x_s)
        loss_source = F.cross_entropy(src_logits, y_s)

        tgt_logits = model(x_t)
        log_probs = F.log_softmax(tgt_logits, dim=1)
        loss_target_sample = -torch.sum(target_calibration * log_probs, dim=1)
        loss_target = torch.mean(loss_target_sample)

        total_loss = alpha * loss_source + (1.0 - alpha) * loss_target
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            updated_logits = model(x_t)
            updated_probs = torch.softmax(updated_logits, dim=1)
            calibration_result = apply_pgtc_calibration(
                probabilities=updated_probs,
                roi_vectors=roi_t,
                templates=templates,
                class_thresholds=thresholds,
                prob_threshold=prob_threshold,
                existing_targets=target_calibration,
            )
            target_loader.dataset.update_calibrated_targets(
                tgt_idx,
                calibration_result["updated_targets"].cpu(),
            )
            total_accepted += int(calibration_result["accepted_mask"].sum().item())
            total_target_samples += len(tgt_idx)

        total_loss_epoch += total_loss.item()
        total_source_loss += loss_source.item()
        total_target_loss += loss_target.item()

    return {
        "train_loss": total_loss_epoch / total_batches,
        "source_loss": total_source_loss / total_batches,
        "target_loss": total_target_loss / total_batches,
        "accepted_ratio": total_accepted / max(total_target_samples, 1),
        "active_ratio": target_loader.dataset.active_ratio(),
    }


def evaluate_target(model, target_loader, device, n_classes):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    with torch.no_grad():
        for x_t, y_t in target_loader:
            x_t = x_t.to(device, non_blocking=True)
            y_t = y_t.to(device, non_blocking=True)
            logits = model(x_t)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            loss = F.cross_entropy(logits, y_t, reduction="sum")

            total_loss += loss.item()
            total_samples += y_t.size(0)
            y_true_list.append(y_t.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())
            y_prob_list.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, n_classes=n_classes)
    return {
        "val_loss": total_loss / max(total_samples, 1),
        "val_acc": metrics["Acc"],
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def train_loso(args):
    logger = Logger(args)
    logger.log_init()
    all_metrics = {}
    channel_info = read_channel_info(args.channel_info_path)

    for fold_idx, test_subject_id in enumerate(args.sub_list, start=1):
        logger.print(f"\n{'=' * 18} LOSO Fold {fold_idx}/{len(args.sub_list)} | Test Subject {test_subject_id} {'=' * 18}")
        x_train, y_train, x_val, y_val = get_loso_task_data(
            args.dataset,
            test_subject_id=test_subject_id,
            sub_list=args.sub_list,
        )
        logger.print(f"Source {x_train.shape} | Target {x_val.shape}")

        train_roi = compute_roi_features(x_train, channel_info)["roi_vector"]
        val_roi = compute_roi_features(x_val, channel_info)["roi_vector"]
        templates_np, thresholds_np = build_class_templates(
            roi_vectors=train_roi,
            labels=y_train,
            num_classes=args.n_classes,
            floor=args.roi_threshold_floor,
        )
        templates = torch.tensor(templates_np, dtype=torch.float32, device=args.device)
        thresholds = torch.tensor(thresholds_np, dtype=torch.float32, device=args.device)

        source_dataset = SourceDataset(x_train, y_train)
        target_train_dataset = TargetCalibrationDataset(x_val, val_roi, args.n_classes)
        target_eval_dataset = TargetEvalDataset(x_val, y_val)
        source_loader, target_train_loader, target_eval_loader, target_init_loader = build_dynamic_dataloaders(
            source_dataset=source_dataset,
            target_train_dataset=target_train_dataset,
            target_eval_dataset=target_eval_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        model = get_model(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, logger=logger)
        saver = BestModelSaver(
            save_dir=args.ckpt_dir,
            sub_idx=test_subject_id,
            fold_idx=fold_idx,
            logger=logger,
            monitor="acc",
        )

        logger.print(f"Trainable parameters: {parameter_count(model):,}")
        logger.print(
            f"PGTC warmup epochs {args.pgtc_warmup_epochs} | "
            f"ROI thresholds {thresholds_np.round(4).tolist()}"
        )
        pgtc_initialized = False

        for epoch in range(1, args.epochs + 1):
            if epoch <= args.pgtc_warmup_epochs:
                train_loss = train_source_only_epoch(
                    source_loader=source_loader,
                    model=model,
                    optimizer=optimizer,
                    device=args.device,
                )
                train_stats = {
                    "train_loss": train_loss,
                    "source_loss": train_loss,
                    "target_loss": 0.0,
                    "accepted_ratio": 0.0,
                    "active_ratio": target_train_dataset.active_ratio(),
                }
            else:
                if not pgtc_initialized:
                    init_ratio = initialize_target_calibration(
                        model=model,
                        target_init_loader=target_init_loader,
                        templates=templates,
                        thresholds=thresholds,
                        device=args.device,
                        prob_threshold=args.pgtc_confidence_threshold,
                    )
                    logger.print(
                        f"PGTC branch enabled at epoch {epoch} | "
                        f"Initial calibrated target active {target_train_dataset.active_ratio():.4f} | "
                        f"Initial accepted {init_ratio:.4f}"
                    )
                    pgtc_initialized = True

                train_stats = train_pgtc_epoch(
                    source_loader=source_loader,
                    target_loader=target_train_loader,
                    model=model,
                    optimizer=optimizer,
                    device=args.device,
                    alpha=args.alpha,
                    templates=templates,
                    thresholds=thresholds,
                    prob_threshold=args.pgtc_confidence_threshold,
                )
            val_results = evaluate_target(
                model=model,
                target_loader=target_eval_loader,
                device=args.device,
                n_classes=args.n_classes,
            )
            saver.save_if_best(model, epoch, val_results["val_acc"], val_results["val_loss"])
            early_stopping(val_results["val_loss"])

            logger.print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Train {train_stats['train_loss']:.4f} | Src {train_stats['source_loss']:.4f} | "
                f"Tgt {train_stats['target_loss']:.4f} | Val {val_results['val_loss']:.4f}/{val_results['val_acc']:.4f} | "
                f"PGTCAccept {train_stats['accepted_ratio']:.4f} | PGTCActive {train_stats['active_ratio']:.4f}"
            )

            if early_stopping.early_stop:
                logger.print(f"Early stopping at epoch {epoch}, best val loss {early_stopping.val_loss_min:.4f}")
                break

        if saver.best_path and os.path.exists(saver.best_path):
            logger.print(f"Reloading Best Model from {Path(saver.best_path).name}")
            model.load_state_dict(torch.load(saver.best_path, map_location=args.device, weights_only=True))

        model.eval()
        t0 = datetime.now()
        with torch.no_grad():
            for batch_x, _ in target_eval_loader:
                batch_x = batch_x.to(args.device, non_blocking=True)
                _ = model(batch_x)
        t1 = datetime.now()
        latency = ((t1 - t0).total_seconds() / max(len(target_eval_loader.dataset), 1)) * 1000.0

        final_res = evaluate_target(
            model=model,
            target_loader=target_eval_loader,
            device=args.device,
            n_classes=args.n_classes,
        )
        subject_metrics = dict(final_res["metrics"])
        subject_metrics["Latency(ms)"] = latency

        logger.print("Subject Result: " + " | ".join([f"{key} {value:.4f}" for key, value in subject_metrics.items()]))
        for key, value in subject_metrics.items():
            all_metrics.setdefault(key, []).append(value)

    logger.print(f"\n{'=' * 18} LOSO Grand Average {'=' * 18}")
    logger.print(" | ".join([f"{key}: {np.mean(values):.4f} +/- {np.std(values):.4f}" for key, values in all_metrics.items()]))
    return all_metrics


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="XW_30Chs")
    parser.add_argument("--exp-name", default="pgtc_adaptation")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--pgtc-confidence-threshold", type=float, default=0.90)
    parser.add_argument("--roi-threshold-floor", type=float, default=0.70)
    parser.add_argument("--pgtc-warmup-epochs", type=int, default=10)
    parser.add_argument("--channel-info-path", required=True)
    parser.add_argument("--emb-size", type=int, default=30)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--temporal-filters-per-branch", type=int, default=10)
    parser.add_argument("--spatial-multiplier", type=int, default=3)
    parser.add_argument("--pooling-size1", type=int, default=8)
    parser.add_argument("--pooling-size2", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--subset-subjects", type=int, default=0)
    parser.add_argument("--override-sub-list", nargs="*", type=int, default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()

    vars(args).update(data_config.params[args.dataset])

    if args.override_sub_list:
        args.sub_list = args.override_sub_list
    else:
        args.sub_list = default_subject_list(args.dataset, args.n_subjects)

    if args.subset_subjects > 0:
        args.sub_list = args.sub_list[: args.subset_subjects]

    if args.fast_dev_run:
        args.epochs = min(args.epochs, 8)
        args.sub_list = args.sub_list[:2]
        args.pgtc_warmup_epochs = min(args.pgtc_warmup_epochs, 2)

    if args.eval_batch_size <= 0:
        args.eval_batch_size = args.batch_size

    args.n_subjects = len(args.sub_list)
    args.flatten = (args.n_samples // args.pooling_size1 // args.pooling_size2) * args.emb_size
    args.model = "PATCNet_PGTC"
    args.device = set_device(device="cuda", gpu_id=args.gpu_id)
    seed_everything(args.seed)
    return args

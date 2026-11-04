"""Train/evaluate a frozen custom NationalCSL subset, preserving the old 20-class experiment.

--stage tune reads only train/validation feature arrays. --stage final is an
explicit, separate evaluation step. Prior use of test signers is recorded.
Only load checkpoints produced by your own trusted training run.
"""

import argparse
import csv
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config_v2 import DEFAULT_CONFIG, config_from_mapping
from dataset_v2 import SignDatasetV2, indices_for_signers
from feature_io import verified_sequence
from feature_processing import MaskAwareStandardizer, augment_sequence
from features_v2 import PRESENCE_SLICE
from model_v2 import SinusoidalPositionalEncoding, count_trainable_parameters
from training_support import (
    TEST_SIGNERS,
    TRAIN_SIGNERS,
    VAL_SIGNERS,
    classification_metrics,
    compact_metrics,
    load_checkpoint,
    load_manifest_definition,
    resolve_path,
    select_device,
    set_random_seed,
    write_csv,
    write_evaluation_reports,
)


from protocol import dataset_signature, read_protocol

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune or finally evaluate the regularized custom NationalCSL Transformer."
    )
    parser.add_argument("--data-root", type=Path, default=Path("nationalcsl200_v2"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--stage", choices=("tune", "final"), default="tune")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints_v2/nationalcsl200_transformer_optimized.pt"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports_v2/nationalcsl200_transformer_optimized"),
    )
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace a tune checkpoint/report; old 20-class paths are separate.")
    parser.add_argument("--allow-repeat-final", action="store_true", help="Explicitly allow re-evaluating an already reported final test.")
    return parser.parse_args()


class PreparedSplitDataset(Dataset):
    def __init__(self, source_dataset, indices, normalizer, training=False):
        self.source_dataset = source_dataset
        self.indices = list(indices)
        self.normalizer = normalizer
        self.training = bool(training)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        dataset_index = self.indices[item]
        record = self.source_dataset.samples[dataset_index]
        sequence = verified_sequence(self.source_dataset, record.path)
        if self.training:
            sequence = augment_sequence(sequence)
        sequence = self.normalizer.transform(sequence)
        return torch.from_numpy(sequence), torch.tensor(record.label, dtype=torch.long)


class MaskAwareSmallTransformer(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.position_encoding = SinusoidalPositionalEncoding(
            config.d_model, config.sequence_length
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, num_classes),
        )

    def forward(self, inputs):
        expected = (self.config.sequence_length, self.config.input_dim)
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(f"Expected [B, T, F] with (T, F)={expected}, got {tuple(inputs.shape)}")

        frame_valid = inputs[:, :, PRESENCE_SLICE].sum(dim=-1) > 0.5
        # A defensive fallback avoids an all-masked Transformer sequence.
        all_missing = ~frame_valid.any(dim=1)
        if all_missing.any():
            frame_valid = frame_valid.clone()
            frame_valid[all_missing] = True

        hidden = torch.nn.functional.gelu(self.input_norm(self.input_projection(inputs)))
        hidden = self.position_encoding(hidden)
        hidden = self.transformer(hidden, src_key_padding_mask=~frame_valid)
        weights = frame_valid.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.output(pooled)


def run_loader(model, loader, criterion, device, num_classes, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += float(loss.item()) * inputs.shape[0]
            total_samples += inputs.shape[0]
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    metrics = classification_metrics(y_true, y_pred, num_classes)
    metrics["loss"] = total_loss / total_samples
    return metrics, y_true, y_pred


def build_components(args):
    data_root = resolve_path(args.data_root)
    manifest = resolve_path(args.manifest) if args.manifest else data_root / "manifest.csv"
    checkpoint = resolve_path(args.checkpoint)
    report_dir = resolve_path(args.report_dir)
    manifest_rows, class_names = load_manifest_definition(manifest)
    config = replace(
        DEFAULT_CONFIG,
        data_root=str(data_root),
        checkpoint_path=str(checkpoint),
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.35,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
        random_seed=args.seed,
    )
    config.validate()
    dataset = SignDatasetV2(
        data_root=data_root,
        class_names=class_names,
        metadata_csv=manifest,
        config=config,
    )
    dataset.manifest_path = manifest
    dataset.data_signature = dataset_signature(manifest)
    dataset.feature_checksums = {str((data_root / row["path"]).resolve()): row["feature_sha256"] for row in manifest_rows}
    dataset.verified_features = set()
    split_indices = {
        "train": indices_for_signers(dataset, TRAIN_SIGNERS),
        "val": indices_for_signers(dataset, VAL_SIGNERS),
        "test": indices_for_signers(dataset, TEST_SIGNERS),
    }
    return dataset, class_names, config, checkpoint, report_dir, split_indices


def make_loaders(dataset, split_indices, normalizer, config, seed, include_test=False):
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            PreparedSplitDataset(dataset, split_indices["train"], normalizer, training=True),
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        ),
        "val": DataLoader(
            PreparedSplitDataset(dataset, split_indices["val"], normalizer),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }
    if include_test:
        loaders["test"] = DataLoader(
            PreparedSplitDataset(dataset, split_indices["test"], normalizer),
            batch_size=config.batch_size, shuffle=False, num_workers=0,
        )
    return loaders


def save_tuning_checkpoint(
    path, epoch, model, optimizer, scheduler, class_names, config, normalizer,
    best_validation, history, label_smoothing, data_signature,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "data_signature": data_signature,
            "epoch": int(epoch),
            "model_kind": "mask_aware_small_transformer",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_names": list(class_names),
            "config": config.to_dict(),
            "normalizer_mean": normalizer.mean,
            "normalizer_scale": normalizer.scale,
            "best_validation": compact_metrics(best_validation),
            "history": history,
            "label_smoothing": float(label_smoothing),
            "train_signers": list(TRAIN_SIGNERS),
            "val_signers": list(VAL_SIGNERS),
            "test_signers": list(TEST_SIGNERS),
        },
        temporary,
    )
    temporary.replace(path)


def write_validation_reports(
    report_dir, class_names, dataset, val_indices, metrics, y_true, y_pred
):
    report_dir.mkdir(parents=True, exist_ok=True)
    per_class_rows = []
    for class_id, class_name in enumerate(class_names):
        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": f"{metrics['per_class_precision'][class_id]:.6f}",
                "recall": f"{metrics['per_class_recall'][class_id]:.6f}",
                "f1": f"{metrics['per_class_f1'][class_id]:.6f}",
                "support": metrics["per_class_support"][class_id],
            }
        )
    write_csv(
        report_dir / "val_per_class_metrics.csv",
        ("class_id", "class_name", "precision", "recall", "f1", "support"),
        per_class_rows,
    )

    with (report_dir / "val_confusion_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for class_name, values in zip(class_names, metrics["confusion_matrix"]):
            writer.writerow([class_name, *values])

    prediction_rows = []
    for dataset_index, true_id, predicted_id in zip(val_indices, y_true, y_pred):
        record = dataset.samples[dataset_index]
        prediction_rows.append(
            {
                "sample_id": record.path.stem,
                "signer_id": record.signer_id,
                "true_class_id": true_id,
                "true_class_name": class_names[true_id],
                "predicted_class_id": predicted_id,
                "predicted_class_name": class_names[predicted_id],
                "correct": str(true_id == predicted_id).lower(),
            }
        )
    write_csv(
        report_dir / "val_predictions.csv",
        (
            "sample_id", "signer_id", "true_class_id", "true_class_name",
            "predicted_class_id", "predicted_class_name", "correct",
        ),
        prediction_rows,
    )


def train_tune(args, dataset, class_names, config, checkpoint, report_dir, split_indices, device):
    if (checkpoint.exists() or (report_dir.exists() and any(report_dir.iterdir()))) and not args.overwrite:
        raise FileExistsError("Tune output already exists. Keep it and choose NEW --checkpoint/--report-dir paths, or explicitly pass --overwrite.")
    if (report_dir / "metrics.json").exists():
        raise FileExistsError("Final results already exist here. Choose NEW checkpoint/report paths for another experiment.")
    normalizer = MaskAwareStandardizer.fit(dataset, split_indices["train"])
    loaders = make_loaders(dataset, split_indices, normalizer, config, args.seed)
    model = MaskAwareSmallTransformer(len(class_names), config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    print(f"Device: {device}")
    print("Stage: tune (Participant_09/10 FEATURE ARRAYS will not be read)")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print("Optimizations: train-only normalization, augmentation, invalid-frame mask, label smoothing")

    best_key = (-math.inf, -math.inf, -math.inf)
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        train_metrics, _, _ = run_loader(
            model, loaders["train"], criterion, device, len(class_names), optimizer
        )
        val_metrics, _, _ = run_loader(
            model, loaders["val"], criterion, device, len(class_names)
        )
        scheduler.step(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": compact_metrics(train_metrics),
                "val": compact_metrics(val_metrics),
            }
        )
        print(
            f"Epoch {epoch:02d}/{config.max_epochs} "
            f"| Train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} "
            f"F1={train_metrics['macro_f1']:.4f} "
            f"| Val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"F1={val_metrics['macro_f1']:.4f}"
        )
        candidate_key = (
            val_metrics["accuracy"], val_metrics["macro_f1"], -val_metrics["loss"]
        )
        if candidate_key > best_key:
            best_key = candidate_key
            best_epoch = epoch
            stale_epochs = 0
            save_tuning_checkpoint(
                checkpoint, epoch, model, optimizer, scheduler, class_names, config,
                normalizer, val_metrics, history, args.label_smoothing, dataset.data_signature,
            )
            print(f"  Saved best validation checkpoint: {checkpoint}")
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stopping_patience:
            print(f"Early stopping after {stale_epochs} epochs without validation improvement.")
            break

    saved = load_checkpoint(checkpoint, device)
    model.load_state_dict(saved["model_state_dict"])
    val_metrics, val_true, val_pred = run_loader(
        model, loaders["val"], criterion, device, len(class_names)
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    tuning_summary = {
        "data_signature": dataset.data_signature,
        "class_count": len(class_names),
        "stage": "tune",
        "test_evaluated": False,
        "model": "mask_aware_small_transformer",
        "best_epoch": best_epoch,
        "best_validation": compact_metrics(val_metrics),
        "checkpoint": str(checkpoint),
        "history": history,
    }
    with (report_dir / "tuning_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(tuning_summary, handle, ensure_ascii=False, indent=2)
    write_validation_reports(
        report_dir,
        class_names,
        dataset,
        split_indices["val"],
        val_metrics,
        val_true,
        val_pred,
    )

    print("=" * 76)
    print(f"Best epoch: {best_epoch}")
    print(
        f"Validation only: accuracy={val_metrics['accuracy']:.4f}, "
        f"Macro-F1={val_metrics['macro_f1']:.4f}, loss={val_metrics['loss']:.4f}"
    )
    print("Held-out test: NOT EVALUATED")
    print(f"Checkpoint: {checkpoint}")
    print(f"Tuning report: {report_dir / 'tuning_metrics.json'}")
    print("=" * 76)


def evaluate_final(args, dataset, class_names, checkpoint, report_dir, split_indices, device):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Optimized checkpoint not found: {checkpoint}")
    if (report_dir / "metrics.json").exists() and not args.allow_repeat_final:
        raise FileExistsError("Final test report already exists. Do not tune to it. Use --allow-repeat-final only for deliberate reproduction.")
    saved = load_checkpoint(checkpoint, device)
    if saved.get("data_signature") != dataset.data_signature:
        raise ValueError("Dataset/protocol fingerprint differs from the tuning checkpoint.")
    if saved.get("model_kind") != "mask_aware_small_transformer":
        raise ValueError("Checkpoint is not from this optimized training script.")
    if list(saved["class_names"]) != list(class_names):
        raise ValueError("Checkpoint class order does not match the manifest.")
    config = config_from_mapping(saved["config"])
    normalizer = MaskAwareStandardizer(
        saved["normalizer_mean"], saved["normalizer_scale"]
    )
    loaders = make_loaders(dataset, split_indices, normalizer, config, config.random_seed, include_test=True)
    model = MaskAwareSmallTransformer(len(class_names), config).to(device)
    model.load_state_dict(saved["model_state_dict"])
    criterion = nn.CrossEntropyLoss(label_smoothing=float(saved["label_smoothing"]))

    val_metrics, val_true, val_pred = run_loader(
        model, loaders["val"], criterion, device, len(class_names)
    )
    print("Validation checkpoint restored. Evaluating held-out Participant_09/10 once.")
    test_metrics, test_true, test_pred = run_loader(
        model, loaders["test"], criterion, device, len(class_names)
    )
    summary = {
        "data_signature": dataset.data_signature,
        "protocol": read_protocol(dataset.manifest_path.parent),
        "model": "mask_aware_small_transformer",
        "device": str(device),
        "class_names": class_names,
        "split": {
            "train_signers": list(TRAIN_SIGNERS),
            "val_signers": list(VAL_SIGNERS),
            "test_signers": list(TEST_SIGNERS),
            "train_samples": len(split_indices["train"]),
            "val_samples": len(split_indices["val"]),
            "test_samples": len(split_indices["test"]),
        },
        "best_epoch": int(saved["epoch"]),
        "best_validation": compact_metrics(val_metrics),
        "held_out_test": compact_metrics(test_metrics),
        "test_use": "Stage benchmark: final command after validation selection. Participant_09/10 were used in earlier 20-class development; not a fresh blind test.",
        "checkpoint": str(checkpoint),
        "history": saved["history"],
        "optimization": {
            "train_only_mask_aware_standardization": True,
            "training_only_augmentation": True,
            "invalid_frame_attention_mask": True,
            "d_model": config.d_model,
            "num_layers": config.num_layers,
            "label_smoothing": float(saved["label_smoothing"]),
        },
    }
    write_evaluation_reports(
        report_dir,
        class_names,
        dataset,
        split_indices,
        {"val": (val_metrics, val_true, val_pred), "test": (test_metrics, test_true, test_pred)},
        summary,
    )
    print("=" * 76)
    print(
        f"Validation: accuracy={val_metrics['accuracy']:.4f}, "
        f"Macro-F1={val_metrics['macro_f1']:.4f}"
    )
    print(
        f"Held-out test: accuracy={test_metrics['accuracy']:.4f}, "
        f"Macro-F1={test_metrics['macro_f1']:.4f}"
    )
    print(f"Reports: {report_dir}")
    print("=" * 76)


def main():
    args = parse_args()
    if args.max_epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("Epochs, patience and batch size must be positive.")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1).")
    if args.threads <= 0:
        raise ValueError("--threads must be positive.")
    torch.set_num_threads(args.threads)
    set_random_seed(args.seed)
    device = select_device(args.device)
    dataset, class_names, config, checkpoint, report_dir, split_indices = build_components(args)
    expected = (7 * len(class_names), len(class_names), 2 * len(class_names))
    actual = tuple(len(split_indices[name]) for name in ("train", "val", "test"))
    if actual != expected:
        raise RuntimeError(f"Expected split sizes {expected}, found {actual}.")
    print(f"Classes: {len(class_names)} (custom subset, not official NationalCSL200)")
    print(f"Samples: train={actual[0]}, val={actual[1]}, test={actual[2]}")
    print("Split: train=01..07, val=08, test=09..10")
    print(f"Dataset signature: {dataset.data_signature}")
    if args.stage == "tune":
        train_tune(
            args, dataset, class_names, config, checkpoint, report_dir, split_indices, device
        )
    else:
        evaluate_final(
            args, dataset, class_names, checkpoint, report_dir, split_indices, device
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

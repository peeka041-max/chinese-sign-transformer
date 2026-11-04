"""Transfer-tune the fixed realtime MVP daily-20 vocabulary.

This entry point intentionally has no test-evaluation mode. It reads test rows as
manifest metadata for split-integrity checks, but only train/validation arrays are
ever opened. Only checkpoints produced by trusted local training runs should be
loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFORMER_ROOT = SCRIPT_DIR.parent
SEED = 42
TRAIN_SIGNERS = tuple(f"Participant_{index:02d}" for index in range(1, 8))
VAL_SIGNERS = ("Participant_08",)
TEST_SIGNERS = ("Participant_09", "Participant_10")
CLASSIFIER_KEYS = {"output.2.weight", "output.2.bias"}

PHASE1_EPOCHS = 10
PHASE1_LEARNING_RATE = 1e-3
PHASE2_MAX_EPOCHS = 100
PHASE2_LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 20
BATCH_SIZE = 16
WEIGHT_DECAY = 3e-3
LABEL_SMOOTHING = 0.08

SELECTION_RULE = (
    "validation_accuracy_desc, validation_macro_f1_desc, validation_loss_asc"
)
FEATURE_DEFINITION = {
    "shape": [30, 266],
    "dtype": "float32",
    "left_local_landmarks": [0, 63],
    "right_local_landmarks": [63, 126],
    "left_wrist_global_xyz": [126, 129],
    "right_wrist_global_xyz": [129, 132],
    "temporal_difference": [132, 264],
    "left_present": 264,
    "right_present": 265,
    "temporal_difference_note": (
        "difference between adjacent normalized time steps; not divided by seconds"
    ),
}


def load_base_module():
    root_value = str(TRANSFORMER_ROOT)
    if root_value not in sys.path:
        sys.path.insert(0, root_value)
    from domain_mvp_v1 import train_domain_mvp20_transfer as base

    return base


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the fixed seed-42 daily20 transfer-tuning experiment."
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("realtime_mvp_v2/data")
    )
    parser.add_argument(
        "--class-order",
        type=Path,
        default=Path("realtime_mvp_v2/class_order.csv"),
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=Path(
            "nationalcsl200_stage_v1/checkpoints_v2/"
            "nationalcsl200_transformer_optimized.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("realtime_mvp_v2/checkpoints/daily20_transfer_seed42.pt"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("realtime_mvp_v2/reports/daily20_transfer_seed42"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace only this experiment's checkpoint and report.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (TRANSFORMER_ROOT / path).resolve()


def selection_key(metrics: Mapping[str, object]) -> Tuple[float, float, float]:
    """Fixed lexicographic rule: accuracy, Macro-F1, then lower loss."""
    return (
        float(metrics["accuracy"]),
        float(metrics["macro_f1"]),
        -float(metrics["loss"]),
    )


def training_records(records: Sequence[object]) -> List[object]:
    """Return records whose arrays may be read during this tuning run."""
    return [record for record in records if getattr(record, "split") in {"train", "val"}]


def tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def classifier_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(CLASSIFIER_KEYS):
        digest.update(key.encode("utf-8"))
        digest.update(tensor_hash(state[key]).encode("ascii"))
    return digest.hexdigest()


def strict_load_transfer_weights(
    model: nn.Module, pretrained_payload: Mapping[str, object], seed: int
) -> Dict[str, object]:
    """Load every compatible non-head tensor and reject any structural mismatch."""
    source_state = pretrained_payload["model_state_dict"]
    target_state = model.state_dict()
    head_before = classifier_hash(model)
    compatible = {}
    unexpected = []
    shape_mismatches = []
    skipped = {}

    for key, value in source_state.items():
        if key in CLASSIFIER_KEYS:
            skipped[key] = {
                "pretrained_shape": list(value.shape),
                "new_shape": list(target_state[key].shape) if key in target_state else None,
                "reason": (
                    "fresh target head required because daily20 class semantics and order "
                    "differ from every prior task"
                ),
            }
            continue
        if key not in target_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "pretrained_shape": list(value.shape),
                    "new_shape": list(target_state[key].shape),
                }
            )
            continue
        compatible[key] = value

    missing_source = sorted(
        key
        for key in target_state
        if key not in CLASSIFIER_KEYS and key not in compatible
    )
    if unexpected or shape_mismatches or missing_source:
        raise RuntimeError(
            "Non-classifier transfer mismatch: "
            + json.dumps(
                {
                    "unexpected_source_keys": unexpected,
                    "shape_mismatches": shape_mismatches,
                    "missing_source_keys": missing_source,
                },
                ensure_ascii=False,
            )
        )

    incompatible = model.load_state_dict(compatible, strict=False)
    missing_after = sorted(incompatible.missing_keys)
    unexpected_after = sorted(incompatible.unexpected_keys)
    if set(missing_after) != CLASSIFIER_KEYS or unexpected_after:
        raise RuntimeError(
            "Transfer load did not leave exactly the classifier head unmatched: "
            + json.dumps(
                {"missing_keys": missing_after, "unexpected_keys": unexpected_after},
                ensure_ascii=False,
            )
        )
    head_after = classifier_hash(model)
    if head_before != head_after:
        raise RuntimeError("Classifier changed during transfer load; fresh-head rule violated.")

    loaded_names = sorted(compatible)
    return {
        "source": "nationalcsl200_transformer_optimized.pt",
        "seed_for_fresh_head": int(seed),
        "loaded_tensor_count": len(loaded_names),
        "loaded_parameter_count": int(
            sum(value.numel() for value in compatible.values() if torch.is_tensor(value))
        ),
        "loaded_parameter_names": loaded_names,
        "skipped_parameter_names": sorted(skipped),
        "skipped_parameters": skipped,
        "classifier_initialization": "fresh_random_initialization",
        "classifier_hash_before_transfer": head_before,
        "classifier_hash_after_transfer": head_after,
        "non_classifier_mismatch_count": 0,
        "missing_keys_after_load": missing_after,
        "unexpected_keys_after_load": unexpected_after,
        "pretrained_class_count": len(pretrained_payload.get("class_names", [])),
        "new_class_count": int(model.output[2].out_features),
    }


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def array_hash(value: np.ndarray) -> str:
    array = np.asarray(value, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def build_normalization_report(
    normalizer, pretrained_payload: Mapping[str, object], preparation_summary
) -> Dict[str, object]:
    source_mean = np.asarray(pretrained_payload["normalizer_mean"], dtype=np.float32)
    source_scale = np.asarray(pretrained_payload["normalizer_scale"], dtype=np.float32)
    target_mean = np.asarray(normalizer.mean, dtype=np.float32)
    target_scale = np.asarray(normalizer.scale, dtype=np.float32)
    if source_mean.shape != (264,) or source_scale.shape != (264,):
        raise ValueError("Pretrained normalizer must contain 264-channel mean and scale.")

    contract = preparation_summary.get("feature_contract", {})
    if contract.get("shape") != [30, 266] or contract.get("dtype") != "float32":
        raise ValueError(f"Unexpected prepared feature contract: {contract}")
    if contract.get("standardization") != "none; raw V2 features":
        raise ValueError("Prepared arrays must be raw and unstandardized.")

    return {
        "raw_feature_contract": contract,
        "target_fit_split": "train",
        "target_fit_signers": list(TRAIN_SIGNERS),
        "target_fit_samples": 140,
        "target_statistics_scope": "current daily20 training split only",
        "continuous_channels_standardized": 264,
        "presence_channels_standardized": False,
        "normalizer_algorithm": "mask-aware per-channel standardization",
        "training_path": (
            "raw V2 feature -> training-only augmentation -> target train-fitted "
            "MaskAwareStandardizer -> model"
        ),
        "validation_path": (
            "raw V2 feature -> same target train-fitted MaskAwareStandardizer -> model"
        ),
        "fit_before_augmentation": True,
        "pretrained_statistics_reused": False,
        "pretrained_algorithm_difference": "none",
        "pretrained_population_difference": (
            "source statistics were fitted on the NationalCSL200 training task; "
            "target statistics are refitted on the 140 daily20 train samples"
        ),
        "pretrained_train_signers": list(pretrained_payload.get("train_signers", [])),
        "pretrained_normalizer_declared_source": pretrained_payload.get(
            "normalizer_source"
        ),
        "pretrained_mean_sha256": array_hash(source_mean),
        "pretrained_scale_sha256": array_hash(source_scale),
        "target_mean_sha256": array_hash(target_mean),
        "target_scale_sha256": array_hash(target_scale),
        "mean_absolute_statistic_difference": float(
            np.mean(np.abs(target_mean - source_mean))
        ),
        "scale_absolute_statistic_difference": float(
            np.mean(np.abs(target_scale - source_scale))
        ),
    }


def access_audit(dataset, split_indices: Mapping[str, Sequence[int]]) -> Dict[str, object]:
    paths_by_split = {
        split: {
            str(dataset.samples[index].path.resolve()) for index in split_indices[split]
        }
        for split in ("train", "val", "test")
    }
    accessed = set(dataset.verified_features)
    test_accessed = accessed & paths_by_split["test"]
    if test_accessed:
        raise RuntimeError(
            "Stage-test feature array access detected: " + ", ".join(sorted(test_accessed))
        )
    unknown = accessed - paths_by_split["train"] - paths_by_split["val"]
    if unknown:
        raise RuntimeError("Unknown feature array access detected: " + ", ".join(sorted(unknown)))
    return {
        "train_unique_arrays_read": len(accessed & paths_by_split["train"]),
        "validation_unique_arrays_read": len(accessed & paths_by_split["val"]),
        "stage_test_unique_arrays_read": 0,
        "stage_test_evaluated": False,
    }


def validate_output_paths(checkpoint: Path, report_dir: Path, overwrite: bool) -> None:
    if overwrite:
        return
    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}")
    if report_dir.exists() and any(report_dir.iterdir()):
        raise FileExistsError(f"Report directory is not empty: {report_dir}")


def build_components(args, base, device):
    data_root = resolve_project_path(args.data_root)
    class_order_path = resolve_project_path(args.class_order)
    manifest_path = data_root / "manifest.csv"
    checkpoint_path = resolve_project_path(args.checkpoint)
    report_dir = resolve_project_path(args.report_dir)
    pretrained_path = resolve_project_path(args.pretrained_checkpoint)
    preparation_summary_path = data_root / "preparation_summary.json"

    for path in (
        class_order_path,
        manifest_path,
        preparation_summary_path,
        pretrained_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    pretrained = base.load_checkpoint(pretrained_path, device)
    config_args = argparse.Namespace(
        batch_size=BATCH_SIZE,
        phase2_learning_rate=PHASE2_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        phase2_max_epochs=PHASE2_MAX_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        seed=SEED,
    )
    config = base.build_config_from_pretrained(
        config_args, data_root, checkpoint_path, pretrained
    )
    classes = base.load_class_order(class_order_path)
    records = base.load_manifest(data_root, manifest_path, classes)
    dataset = base.DomainMvpDataset(data_root, records, classes, config)
    dataset.data_signature = base.canonical_hash(
        {
            "manifest_sha256": base.file_hash(manifest_path),
            "class_order_sha256": base.file_hash(class_order_path),
            "split": {
                "train": list(TRAIN_SIGNERS),
                "val": list(VAL_SIGNERS),
                "test": list(TEST_SIGNERS),
            },
        }
    )
    split_indices = {
        "train": base.indices_for_signers(dataset, TRAIN_SIGNERS),
        "val": base.indices_for_signers(dataset, VAL_SIGNERS),
        "test": base.indices_for_signers(dataset, TEST_SIGNERS),
    }
    sizes = tuple(len(split_indices[name]) for name in ("train", "val", "test"))
    if sizes != (140, 20, 40):
        raise RuntimeError(f"Expected split sizes (140, 20, 40), found {sizes}.")

    paths = {
        "data_root": data_root,
        "class_order": class_order_path,
        "manifest": manifest_path,
        "preparation_summary": preparation_summary_path,
        "checkpoint": checkpoint_path,
        "report_dir": report_dir,
        "pretrained": pretrained_path,
    }
    return dataset, config, pretrained, split_indices, paths


def one_batch_gate(base, dataset, config, pretrained, split_indices, normalizer, device):
    base.set_random_seed(SEED)
    loaders = base.make_loaders(
        dataset, split_indices, normalizer, config, SEED, include_test=False
    )
    model = base.MaskAwareSmallTransformer(len(dataset.class_info), config).to(device)
    transfer = strict_load_transfer_weights(model, pretrained, SEED)
    base.set_head_only_trainable(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=PHASE1_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    inputs, labels = next(iter(loaders["train"]))
    if tuple(inputs.shape[1:]) != (30, 266):
        raise RuntimeError(f"Smoke input has invalid shape: {tuple(inputs.shape)}")
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, labels)
    loss.backward()
    grad_norm = float(model.output[2].weight.grad.detach().norm().cpu())
    if tuple(logits.shape) != (inputs.shape[0], 20):
        raise RuntimeError(f"Smoke logits have invalid shape: {tuple(logits.shape)}")
    if not math.isfinite(float(loss.detach().cpu())) or not math.isfinite(grad_norm):
        raise RuntimeError("Smoke loss or classifier gradient is not finite.")
    optimizer.step()
    audit = access_audit(dataset, split_indices)
    return {
        "status": "passed",
        "input_shape": list(inputs.shape),
        "output_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "classifier_gradient_norm": grad_norm,
        "transfer_loaded_tensor_count": transfer["loaded_tensor_count"],
        "stage_test_unique_arrays_read": audit["stage_test_unique_arrays_read"],
    }


def save_checkpoint(
    path: Path,
    epoch_item,
    model,
    optimizer,
    scheduler,
    dataset,
    config,
    normalizer,
    normalization_report,
    transfer_report,
    smoke_gate,
    history,
):
    access = {
        "train_unique_arrays_read": 140,
        "validation_unique_arrays_read": 20,
        "stage_test_unique_arrays_read": 0,
    }
    payload = {
        "experiment": "daily20_transfer_seed42",
        "seed": SEED,
        "data_signature": dataset.data_signature,
        "epoch": int(epoch_item["global_epoch"]),
        "best_phase": epoch_item["phase"],
        "best_phase_epoch": int(epoch_item["phase_epoch"]),
        "selection_rule": SELECTION_RULE,
        "model_kind": "mask_aware_small_transformer",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "class_names": list(dataset.class_names),
        "class_order": [asdict(item) for item in dataset.class_info],
        "vocabulary_status": "daily vocabulary experiment v1; provisional",
        "feature_definition": FEATURE_DEFINITION,
        "config": config.to_dict(),
        "normalizer_mean": normalizer.mean,
        "normalizer_scale": normalizer.scale,
        "normalizer_source": "daily20 train split only (140 samples)",
        "normalization": normalization_report,
        "best_train": dict(epoch_item["train"]),
        "best_validation": dict(epoch_item["val"]),
        "history": list(history),
        "label_smoothing": LABEL_SMOOTHING,
        "pretrained_checkpoint": "nationalcsl200_transformer_optimized.pt",
        "transfer_load": transfer_report,
        "preflight": smoke_gate,
        "feature_access": access,
        "train_signers": list(TRAIN_SIGNERS),
        "val_signers": list(VAL_SIGNERS),
        "test_signers": list(TEST_SIGNERS),
        "test_evaluated": False,
        "stage_test_status": "NOT EVALUATED",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_validation_with_probabilities(base, model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true = []
    probabilities = []
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            total_loss += float(loss.item()) * inputs.shape[0]
            total += inputs.shape[0]
            y_true.extend(labels.cpu().tolist())
            probabilities.append(probs.cpu().numpy())
    probability_array = np.concatenate(probabilities, axis=0)
    y_pred = probability_array.argmax(axis=1).astype(int).tolist()
    metrics = base.classification_metrics(y_true, y_pred, num_classes)
    metrics["loss"] = total_loss / total
    return metrics, y_true, y_pred, probability_array


def write_training_history(base, report_dir: Path, history):
    rows = []
    for item in history:
        rows.append(
            {
                "global_epoch": item["global_epoch"],
                "phase": item["phase"],
                "phase_epoch": item["phase_epoch"],
                "learning_rate": f"{float(item['learning_rate']):.10f}",
                "train_loss": f"{item['train']['loss']:.8f}",
                "train_accuracy": f"{item['train']['accuracy']:.8f}",
                "train_macro_f1": f"{item['train']['macro_f1']:.8f}",
                "validation_loss": f"{item['val']['loss']:.8f}",
                "validation_accuracy": f"{item['val']['accuracy']:.8f}",
                "validation_macro_f1": f"{item['val']['macro_f1']:.8f}",
                "selection_key": json.dumps(item["selection_key"]),
                "is_best": str(bool(item["is_best"])).lower(),
            }
        )
    base.write_csv(
        report_dir / "training_history.csv",
        (
            "global_epoch",
            "phase",
            "phase_epoch",
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "train_macro_f1",
            "validation_loss",
            "validation_accuracy",
            "validation_macro_f1",
            "selection_key",
            "is_best",
        ),
        rows,
    )


def write_validation_reports(
    base,
    report_dir: Path,
    dataset,
    val_indices,
    metrics,
    y_true,
    y_pred,
    probabilities,
):
    per_class = []
    for class_id, info in enumerate(dataset.class_info):
        per_class.append(
            {
                "class_id": class_id,
                "gloss_id": info.gloss_id,
                "dataset_gloss": info.dataset_gloss,
                "display_word": info.display_word,
                "precision": f"{metrics['per_class_precision'][class_id]:.8f}",
                "recall": f"{metrics['per_class_recall'][class_id]:.8f}",
                "f1": f"{metrics['per_class_f1'][class_id]:.8f}",
                "support": metrics["per_class_support"][class_id],
            }
        )
    base.write_csv(
        report_dir / "validation_per_class_metrics.csv",
        (
            "class_id",
            "gloss_id",
            "dataset_gloss",
            "display_word",
            "precision",
            "recall",
            "f1",
            "support",
        ),
        per_class,
    )

    rows = []
    for row_index, (dataset_index, true_id, predicted_id) in enumerate(
        zip(val_indices, y_true, y_pred)
    ):
        record = dataset.samples[dataset_index]
        true_info = dataset.class_info[true_id]
        predicted_info = dataset.class_info[predicted_id]
        top_ids = np.argsort(-probabilities[row_index])[:3].astype(int).tolist()
        row = {
            "sample_id": record.sample_id,
            "signer_id": record.signer_id,
            "true_class_id": true_id,
            "true_gloss_id": true_info.gloss_id,
            "true_word": true_info.display_word,
            "predicted_class_id": predicted_id,
            "predicted_gloss_id": predicted_info.gloss_id,
            "predicted_word": predicted_info.display_word,
            "correct": str(true_id == predicted_id).lower(),
        }
        for rank, class_id in enumerate(top_ids, start=1):
            info = dataset.class_info[class_id]
            row[f"top{rank}_class_id"] = class_id
            row[f"top{rank}_gloss_id"] = info.gloss_id
            row[f"top{rank}_word"] = info.display_word
            row[f"top{rank}_probability"] = f"{probabilities[row_index, class_id]:.10f}"
        rows.append(row)
    fields = [
        "sample_id",
        "signer_id",
        "true_class_id",
        "true_gloss_id",
        "true_word",
        "predicted_class_id",
        "predicted_gloss_id",
        "predicted_word",
        "correct",
    ]
    for rank in range(1, 4):
        fields.extend(
            [
                f"top{rank}_class_id",
                f"top{rank}_gloss_id",
                f"top{rank}_word",
                f"top{rank}_probability",
            ]
        )
    base.write_csv(report_dir / "validation_predictions.csv", fields, rows)


def major_confusions(dataset, y_true, y_pred) -> List[Dict[str, object]]:
    counts = Counter(
        (int(true_id), int(predicted_id))
        for true_id, predicted_id in zip(y_true, y_pred)
        if true_id != predicted_id
    )
    rows = []
    for (true_id, predicted_id), count in sorted(
        counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    ):
        rows.append(
            {
                "true_class_id": true_id,
                "true_gloss_id": dataset.class_info[true_id].gloss_id,
                "true_word": dataset.class_info[true_id].display_word,
                "predicted_class_id": predicted_id,
                "predicted_gloss_id": dataset.class_info[predicted_id].gloss_id,
                "predicted_word": dataset.class_info[predicted_id].display_word,
                "count": count,
            }
        )
    return rows


def print_epoch(item) -> None:
    train = item["train"]
    val = item["val"]
    print(
        f"Epoch {item['global_epoch']:03d} {item['phase']}:{item['phase_epoch']:03d} "
        f"| train loss={train['loss']:.4f} acc={train['accuracy']:.4f} "
        f"F1={train['macro_f1']:.4f} | val loss={val['loss']:.4f} "
        f"acc={val['accuracy']:.4f} F1={val['macro_f1']:.4f}"
        f"{' *' if item['is_best'] else ''}"
    )


def train(args) -> Dict[str, object]:
    if args.threads <= 0:
        raise ValueError("--threads must be positive.")
    base = load_base_module()
    torch.set_num_threads(args.threads)
    device = base.select_device(args.device)
    (
        dataset,
        config,
        pretrained,
        split_indices,
        paths,
    ) = build_components(args, base, device)
    validate_output_paths(paths["checkpoint"], paths["report_dir"], args.overwrite)
    paths["report_dir"].mkdir(parents=True, exist_ok=True)

    preparation_summary = read_json(paths["preparation_summary"])
    normalizer = base.MaskAwareStandardizer.fit(dataset, split_indices["train"])
    normalization_report = build_normalization_report(
        normalizer, pretrained, preparation_summary
    )
    smoke = one_batch_gate(
        base, dataset, config, pretrained, split_indices, normalizer, device
    )
    print("Preflight gate: " + json.dumps(smoke, ensure_ascii=False))

    # The smoke update must not perturb the single fixed-seed experiment.
    base.set_random_seed(SEED)
    loaders = base.make_loaders(
        dataset, split_indices, normalizer, config, SEED, include_test=False
    )
    model = base.MaskAwareSmallTransformer(len(dataset.class_info), config).to(device)
    transfer_report = strict_load_transfer_weights(model, pretrained, SEED)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print(f"Device: {device}")
    print("Experiment: daily20 transfer, seed=42, one run")
    print("Stage test arrays: WILL NOT BE READ")
    print(
        "Transfer: "
        f"loaded {transfer_report['loaded_tensor_count']} tensors / "
        f"{transfer_report['loaded_parameter_count']} values; skipped "
        f"{transfer_report['skipped_parameter_names']} for fresh daily20 head"
    )
    print(
        "Model config: "
        f"T={config.sequence_length}, F={config.input_dim}, d_model={config.d_model}, "
        f"nhead={config.nhead}, layers={config.num_layers}, "
        f"ff={config.dim_feedforward}, dropout={config.dropout}"
    )
    print(f"Selection rule: {SELECTION_RULE}")

    history = []
    best_key: Optional[Tuple[float, float, float]] = None
    best_epoch_item = None
    global_epoch = 0

    base.set_head_only_trainable(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=PHASE1_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    print(f"Phase 1: classifier head only, epochs={PHASE1_EPOCHS}")
    for phase_epoch in range(1, PHASE1_EPOCHS + 1):
        global_epoch += 1
        train_metrics, _, _ = base.run_loader(
            model, loaders["train"], criterion, device, len(dataset.class_info), optimizer
        )
        val_metrics, _, _ = base.run_loader(
            model, loaders["val"], criterion, device, len(dataset.class_info)
        )
        candidate = selection_key(val_metrics)
        improved = best_key is None or candidate > best_key
        if improved:
            best_key = candidate
        item = {
            "global_epoch": global_epoch,
            "phase": "phase1_head",
            "phase_epoch": phase_epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": base.compact_metrics(train_metrics),
            "val": base.compact_metrics(val_metrics),
            "selection_key": list(candidate),
            "is_best": improved,
        }
        history.append(item)
        print_epoch(item)
        if improved:
            best_epoch_item = item
            save_checkpoint(
                paths["checkpoint"],
                item,
                model,
                optimizer,
                None,
                dataset,
                config,
                normalizer,
                normalization_report,
                transfer_report,
                smoke,
                history,
            )

    base.set_all_trainable(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PHASE2_LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    stale_epochs = 0
    print(
        f"Phase 2: full fine-tuning, max_epochs={PHASE2_MAX_EPOCHS}, "
        f"patience={EARLY_STOPPING_PATIENCE}"
    )
    for phase_epoch in range(1, PHASE2_MAX_EPOCHS + 1):
        global_epoch += 1
        train_metrics, _, _ = base.run_loader(
            model, loaders["train"], criterion, device, len(dataset.class_info), optimizer
        )
        val_metrics, _, _ = base.run_loader(
            model, loaders["val"], criterion, device, len(dataset.class_info)
        )
        scheduler.step(val_metrics["macro_f1"])
        candidate = selection_key(val_metrics)
        improved = best_key is None or candidate > best_key
        if improved:
            best_key = candidate
            stale_epochs = 0
        else:
            stale_epochs += 1
        item = {
            "global_epoch": global_epoch,
            "phase": "phase2_finetune",
            "phase_epoch": phase_epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": base.compact_metrics(train_metrics),
            "val": base.compact_metrics(val_metrics),
            "selection_key": list(candidate),
            "is_best": improved,
        }
        history.append(item)
        print_epoch(item)
        if improved:
            best_epoch_item = item
            save_checkpoint(
                paths["checkpoint"],
                item,
                model,
                optimizer,
                scheduler,
                dataset,
                config,
                normalizer,
                normalization_report,
                transfer_report,
                smoke,
                history,
            )
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping: no lexicographic validation improvement for "
                f"{stale_epochs} phase-2 epochs."
            )
            break

    if best_epoch_item is None or not paths["checkpoint"].is_file():
        raise RuntimeError("No best checkpoint was saved.")

    saved = base.load_checkpoint(paths["checkpoint"], device)
    best_model = base.MaskAwareSmallTransformer(len(dataset.class_info), config).to(device)
    best_model.load_state_dict(saved["model_state_dict"])
    val_metrics, y_true, y_pred, probabilities = run_validation_with_probabilities(
        base,
        best_model,
        loaders["val"],
        criterion,
        device,
        len(dataset.class_info),
    )
    if selection_key(val_metrics) != selection_key(saved["best_validation"]):
        raise RuntimeError("Reloaded best checkpoint validation metrics do not match.")

    feature_access = access_audit(dataset, split_indices)
    if feature_access != {
        "train_unique_arrays_read": 140,
        "validation_unique_arrays_read": 20,
        "stage_test_unique_arrays_read": 0,
        "stage_test_evaluated": False,
    }:
        raise RuntimeError(f"Unexpected feature access audit: {feature_access}")

    report_dir = paths["report_dir"]
    write_training_history(base, report_dir, history)
    write_validation_reports(
        base,
        report_dir,
        dataset,
        split_indices["val"],
        val_metrics,
        y_true,
        y_pred,
        probabilities,
    )
    shutil.copyfile(paths["class_order"], report_dir / "class_order.csv")

    config_report = {
        "experiment": "daily20_transfer_seed42",
        "seed": SEED,
        "test_evaluated": False,
        "stage_test": "NOT EVALUATED",
        "paths": {key: str(value) for key, value in paths.items()},
        "split": {
            "train_signers": list(TRAIN_SIGNERS),
            "val_signers": list(VAL_SIGNERS),
            "stage_test_signers": list(TEST_SIGNERS),
            "train_samples": 140,
            "val_samples": 20,
            "stage_test_samples": 40,
        },
        "model_config_from_pretrained_checkpoint": config.to_dict(),
        "phase1": {
            "trainable": "output.2 classification head only",
            "epochs": PHASE1_EPOCHS,
            "learning_rate": PHASE1_LEARNING_RATE,
        },
        "phase2": {
            "trainable": "all parameters",
            "max_epochs": PHASE2_MAX_EPOCHS,
            "learning_rate": PHASE2_LEARNING_RATE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "scheduler": "ReduceLROnPlateau(macro_f1, factor=0.5, patience=5)",
        },
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "training_only_augmentation": "existing augment_sequence; no new strategy",
        "selection_rule": SELECTION_RULE,
        "class_order": [asdict(item) for item in dataset.class_info],
        "feature_definition": FEATURE_DEFINITION,
        "normalization": normalization_report,
        "transfer_load": transfer_report,
        "preflight": smoke,
        "feature_access": feature_access,
        "data_signature": dataset.data_signature,
    }
    write_json(report_dir / "training_config.json", config_report)

    confusions = major_confusions(dataset, y_true, y_pred)
    tuning = {
        "experiment": "daily20_transfer_seed42",
        "seed": SEED,
        "test_evaluated": False,
        "stage_test": "NOT EVALUATED",
        "selection_rule": SELECTION_RULE,
        "best_epoch": int(saved["epoch"]),
        "best_phase": saved["best_phase"],
        "best_phase_epoch": int(saved["best_phase_epoch"]),
        "best_train": saved["best_train"],
        "best_validation": base.compact_metrics(val_metrics),
        "major_validation_confusions": confusions,
        "epochs_ran": len(history),
        "transfer_load": transfer_report,
        "normalization": normalization_report,
        "feature_access": feature_access,
        "checkpoint": str(paths["checkpoint"]),
    }
    write_json(report_dir / "tuning_metrics.json", tuning)

    saved["history"] = history
    saved["training_complete"] = True
    saved["epochs_ran"] = len(history)
    saved["feature_access"] = feature_access
    saved["best_validation"] = base.compact_metrics(val_metrics)
    temporary = paths["checkpoint"].with_name(paths["checkpoint"].name + ".final.tmp")
    torch.save(saved, temporary)
    temporary.replace(paths["checkpoint"])

    required = [
        paths["checkpoint"],
        report_dir / "training_config.json",
        report_dir / "training_history.csv",
        report_dir / "tuning_metrics.json",
        report_dir / "validation_predictions.csv",
        report_dir / "validation_per_class_metrics.csv",
    ]
    empty = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if empty:
        raise RuntimeError(f"Required outputs missing or empty: {empty}")

    print("=" * 76)
    print(
        f"Best epoch {saved['epoch']}: train acc={saved['best_train']['accuracy']:.4f}, "
        f"F1={saved['best_train']['macro_f1']:.4f}, "
        f"loss={saved['best_train']['loss']:.4f}"
    )
    print(
        f"Validation: acc={val_metrics['accuracy']:.4f}, "
        f"F1={val_metrics['macro_f1']:.4f}, loss={val_metrics['loss']:.4f}"
    )
    print("Stage test: NOT EVALUATED")
    print(f"Checkpoint: {paths['checkpoint']}")
    print(f"Reports: {report_dir}")
    print("=" * 76)
    return tuning


def main() -> int:
    args = parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

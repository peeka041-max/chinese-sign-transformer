"""Controlled daily20 initialization experiment.

All model parameters are randomly initialized with seed 42. The data, model
shape, normalizer, augmentation, optimizer, two-phase schedule, early stopping,
and validation selection rule are inherited from daily20_transfer_seed42.
This is an initialization control under the transfer schedule, not an optimized
from-scratch training recipe. There is deliberately no test-evaluation mode.
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
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFORMER_ROOT = SCRIPT_DIR.parent
SEED = 42
EXPERIMENT = "daily20_scratch_seed42"
REFERENCE_EXPERIMENT = "daily20_transfer_seed42"
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
STRUCTURAL_CONFIG_KEYS = (
    "sequence_length",
    "input_dim",
    "d_model",
    "nhead",
    "num_layers",
    "dim_feedforward",
    "dropout",
)
LOSS_CONTRACT = (
    "torch.nn.CrossEntropyLoss(label_smoothing=0.08, reduction='mean'); "
    "epoch loss is sample-count-weighted mean of batch losses"
)


def load_reference_modules():
    for path in (TRANSFORMER_ROOT, SCRIPT_DIR):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    import train_daily20_transfer as reference

    return reference, reference.load_base_module()


def resolve_dataset_api(reference, base):
    """The transfer entry owns orchestration; its domain base owns data/model APIs."""
    required = (
        "load_class_order",
        "load_manifest",
        "DomainMvpDataset",
        "indices_for_signers",
        "make_loaders",
        "MaskAwareSmallTransformer",
        "run_loader",
    )
    missing = [name for name in required if not hasattr(base, name)]
    if missing:
        raise AttributeError(f"Domain base is missing required APIs: {missing}")
    return base


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the single seed-42 random-initialization daily20 control."
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
        "--reference-checkpoint",
        type=Path,
        default=Path("realtime_mvp_v2/checkpoints/daily20_transfer_seed42.pt"),
    )
    parser.add_argument(
        "--reference-report-dir",
        type=Path,
        default=Path("realtime_mvp_v2/reports/daily20_transfer_seed42"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("realtime_mvp_v2/checkpoints/daily20_scratch_seed42.pt"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("realtime_mvp_v2/reports/daily20_scratch_seed42"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace only this scratch-control output.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (TRANSFORMER_ROOT / path).resolve()


def array_hash(value: np.ndarray) -> str:
    array = np.asarray(value, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def model_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def create_seeded_random_model(base, model_factory, seed: int):
    """Reset every RNG immediately before constructing a fully random model."""
    base.set_random_seed(seed)
    return model_factory()


def set_phase_trainability(base, model: nn.Module, phase: str) -> List[str]:
    if phase == "phase1_head":
        base.set_head_only_trainable(model)
    elif phase == "phase2_finetune":
        base.set_all_trainable(model)
    else:
        raise ValueError(f"Unknown training phase: {phase}")
    return sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def optimizer_parameter_names(model: nn.Module, optimizer) -> List[str]:
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    names = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in names_by_id:
                raise RuntimeError("Optimizer contains a parameter outside the model.")
            names.append(names_by_id[id(parameter)])
    return sorted(names)


def reuse_reference_normalizer(
    base,
    reference_payload: Mapping[str, object],
    expected_data_signature: str,
    expected_class_names: Sequence[str],
    config,
):
    """Reconstruct, but never refit, the exact transfer-run normalizer."""
    if reference_payload.get("data_signature") != expected_data_signature:
        raise ValueError("Reference data signature does not match current prepared data.")
    if list(reference_payload.get("class_names", [])) != list(expected_class_names):
        raise ValueError("Reference class order does not match current class_order.csv.")
    saved_config = reference_payload.get("config", {})
    mismatches = {
        key: {"reference": saved_config.get(key), "current": getattr(config, key)}
        for key in STRUCTURAL_CONFIG_KEYS
        if saved_config.get(key) != getattr(config, key)
    }
    if mismatches:
        raise ValueError("Reference model config mismatch: " + json.dumps(mismatches))

    mean = np.asarray(reference_payload["normalizer_mean"], dtype=np.float32).copy()
    scale = np.asarray(reference_payload["normalizer_scale"], dtype=np.float32).copy()
    if mean.shape != (264,) or scale.shape != (264,):
        raise ValueError("Reference normalizer must have 264 mean and scale values.")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Reference normalizer contains invalid values.")

    expected_mean_hash = reference_payload.get("normalization", {}).get(
        "target_mean_sha256"
    )
    expected_scale_hash = reference_payload.get("normalization", {}).get(
        "target_scale_sha256"
    )
    actual_mean_hash = array_hash(mean)
    actual_scale_hash = array_hash(scale)
    if expected_mean_hash and actual_mean_hash != expected_mean_hash:
        raise ValueError("Reference normalizer mean hash mismatch.")
    if expected_scale_hash and actual_scale_hash != expected_scale_hash:
        raise ValueError("Reference normalizer scale hash mismatch.")

    normalizer = base.MaskAwareStandardizer(mean, scale)
    exact = np.array_equal(normalizer.mean, mean) and np.array_equal(normalizer.scale, scale)
    if not exact:
        raise RuntimeError("Normalizer changed while being reconstructed.")
    return normalizer, {
        "source_experiment": REFERENCE_EXPERIMENT,
        "source_checkpoint_normalizer": True,
        "refitted": False,
        "fit_samples_in_reference": 140,
        "fit_split_in_reference": "train",
        "exact_value_match": True,
        "mean_sha256": actual_mean_hash,
        "scale_sha256": actual_scale_hash,
        "standardization_passes_per_sample": 1,
        "training_path": (
            "raw V2 feature -> existing training-only augmentation -> reused daily20 "
            "train normalizer -> model"
        ),
        "validation_path": (
            "raw V2 feature -> reused daily20 train normalizer -> model"
        ),
    }


def compare_predictions(transfer_rows, scratch_rows) -> List[Dict[str, object]]:
    transfer_by_id = {row["sample_id"]: row for row in transfer_rows}
    scratch_by_id = {row["sample_id"]: row for row in scratch_rows}
    if set(transfer_by_id) != set(scratch_by_id):
        raise ValueError("Validation sample IDs differ between experiments.")
    results = []
    for sample_id in sorted(transfer_by_id):
        transfer = transfer_by_id[sample_id]
        scratch = scratch_by_id[sample_id]
        if transfer["true_word"] != scratch["true_word"]:
            raise ValueError(f"True label mismatch for {sample_id}.")
        transfer_correct = str(transfer["correct"]).lower() == "true"
        scratch_correct = str(scratch["correct"]).lower() == "true"
        if transfer_correct and not scratch_correct:
            change_type = "correct_to_incorrect"
        elif not transfer_correct and scratch_correct:
            change_type = "incorrect_to_correct"
        elif transfer_correct and scratch_correct:
            change_type = "both_correct"
        elif transfer["predicted_word"] == scratch["predicted_word"]:
            change_type = "both_incorrect_same_prediction"
        else:
            change_type = "both_incorrect_changed_prediction"
        results.append(
            {
                "sample_id": sample_id,
                "signer_id": scratch.get("signer_id", ""),
                "true_word": scratch["true_word"],
                "true_gloss_id": scratch.get("true_gloss_id", ""),
                "transfer_prediction": transfer["predicted_word"],
                "scratch_prediction": scratch["predicted_word"],
                "transfer_correct": transfer_correct,
                "scratch_correct": scratch_correct,
                "change_type": change_type,
            }
        )
    return results


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_reference_protocol(reference_config, reference_payload) -> None:
    expected = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "selection_rule": SELECTION_RULE,
    }
    mismatches = {
        key: {"expected": value, "reference": reference_config.get(key)}
        for key, value in expected.items()
        if reference_config.get(key) != value
    }
    phase1 = reference_config.get("phase1", {})
    phase2 = reference_config.get("phase2", {})
    phase_checks = {
        "phase1_epochs": (phase1.get("epochs"), PHASE1_EPOCHS),
        "phase1_learning_rate": (
            phase1.get("learning_rate"),
            PHASE1_LEARNING_RATE,
        ),
        "phase2_max_epochs": (phase2.get("max_epochs"), PHASE2_MAX_EPOCHS),
        "phase2_learning_rate": (
            phase2.get("learning_rate"),
            PHASE2_LEARNING_RATE,
        ),
        "phase2_patience": (
            phase2.get("early_stopping_patience"),
            EARLY_STOPPING_PATIENCE,
        ),
    }
    mismatches.update(
        {
            name: {"reference": actual, "expected": wanted}
            for name, (actual, wanted) in phase_checks.items()
            if actual != wanted
        }
    )
    if reference_payload.get("label_smoothing") != LABEL_SMOOTHING:
        mismatches["checkpoint_label_smoothing"] = {
            "reference": reference_payload.get("label_smoothing"),
            "expected": LABEL_SMOOTHING,
        }
    if mismatches:
        raise ValueError("Reference training protocol mismatch: " + json.dumps(mismatches))


def build_components(args, reference, base, device):
    data_api = resolve_dataset_api(reference, base)
    paths = {
        "data_root": resolve_project_path(args.data_root),
        "class_order": resolve_project_path(args.class_order),
        "reference_checkpoint": resolve_project_path(args.reference_checkpoint),
        "reference_report_dir": resolve_project_path(args.reference_report_dir),
        "checkpoint": resolve_project_path(args.checkpoint),
        "report_dir": resolve_project_path(args.report_dir),
    }
    paths["manifest"] = paths["data_root"] / "manifest.csv"
    paths["reference_training_config"] = (
        paths["reference_report_dir"] / "training_config.json"
    )
    paths["reference_tuning_metrics"] = (
        paths["reference_report_dir"] / "tuning_metrics.json"
    )
    paths["reference_predictions"] = (
        paths["reference_report_dir"] / "validation_predictions.csv"
    )
    for key in (
        "class_order",
        "manifest",
        "reference_checkpoint",
        "reference_training_config",
        "reference_tuning_metrics",
        "reference_predictions",
    ):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])

    payload = base.load_checkpoint(paths["reference_checkpoint"], device)
    reference_config = read_json(paths["reference_training_config"])
    validate_reference_protocol(reference_config, payload)
    config = replace(
        base.config_from_mapping(payload["config"]),
        data_root=str(paths["data_root"]),
        checkpoint_path=str(paths["checkpoint"]),
    )
    config.validate()
    classes = data_api.load_class_order(paths["class_order"])
    records = data_api.load_manifest(
        paths["data_root"], paths["manifest"], classes
    )
    dataset = data_api.DomainMvpDataset(
        paths["data_root"], records, classes, config
    )
    dataset.data_signature = data_api.canonical_hash(
        {
            "manifest_sha256": data_api.file_hash(paths["manifest"]),
            "class_order_sha256": data_api.file_hash(paths["class_order"]),
            "split": {
                "train": list(reference.TRAIN_SIGNERS),
                "val": list(reference.VAL_SIGNERS),
                "test": list(reference.TEST_SIGNERS),
            },
        }
    )
    split_indices = {
        "train": data_api.indices_for_signers(dataset, reference.TRAIN_SIGNERS),
        "val": data_api.indices_for_signers(dataset, reference.VAL_SIGNERS),
        "test": data_api.indices_for_signers(dataset, reference.TEST_SIGNERS),
    }
    sizes = tuple(len(split_indices[name]) for name in ("train", "val", "test"))
    if sizes != (140, 20, 40):
        raise RuntimeError(f"Expected split sizes (140, 20, 40), found {sizes}.")
    normalizer, normalizer_report = reuse_reference_normalizer(
        base,
        payload,
        dataset.data_signature,
        dataset.class_names,
        config,
    )
    return (
        dataset,
        config,
        payload,
        reference_config,
        split_indices,
        normalizer,
        normalizer_report,
        paths,
    )


def validate_output_paths(paths, overwrite: bool) -> None:
    if overwrite:
        return
    if paths["checkpoint"].exists():
        raise FileExistsError(f"Checkpoint already exists: {paths['checkpoint']}")
    if paths["report_dir"].exists() and any(paths["report_dir"].iterdir()):
        raise FileExistsError(f"Report directory is not empty: {paths['report_dir']}")


def make_seeded_model_and_loaders(
    reference, base, dataset, config, split_indices, normalizer, device
):
    # Required isolation point: metadata and normalizer have already been read.
    base.set_random_seed(SEED)
    loaders = base.make_loaders(
        dataset, split_indices, normalizer, config, SEED, include_test=False
    )
    model = base.MaskAwareSmallTransformer(
        len(dataset.class_info), config
    ).to(device)
    return model, loaders


def one_batch_gate(
    reference, base, dataset, config, split_indices, normalizer, device
):
    model, loaders = make_seeded_model_and_loaders(
        reference, base, dataset, config, split_indices, normalizer, device
    )
    initial_hash = model_state_hash(model)
    trainable = set_phase_trainability(base, model, "phase1_head")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=PHASE1_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer_names = optimizer_parameter_names(model, optimizer)
    if optimizer_names != trainable:
        raise RuntimeError("Phase-1 optimizer parameters differ from trainable parameters.")
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    inputs, labels = next(iter(loaders["train"]))
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, labels)
    loss.backward()
    gradient_norm = float(model.output[2].weight.grad.detach().norm().cpu())
    if tuple(inputs.shape[1:]) != (30, 266) or tuple(logits.shape) != (inputs.shape[0], 20):
        raise RuntimeError(
            f"Scratch smoke shape mismatch: input={tuple(inputs.shape)}, output={tuple(logits.shape)}"
        )
    if not math.isfinite(float(loss.detach().cpu())) or not math.isfinite(gradient_norm):
        raise RuntimeError("Scratch smoke loss or gradient is not finite.")
    optimizer.step()
    access = reference.access_audit(dataset, split_indices)
    return {
        "status": "passed",
        "input_shape": list(inputs.shape),
        "output_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "classifier_gradient_norm": gradient_norm,
        "initial_model_state_sha256": initial_hash,
        "phase1_trainable_parameter_names": trainable,
        "phase1_optimizer_parameter_names": optimizer_names,
        "model_state_dict_loaded": False,
        "stage_test_unique_arrays_read": access["stage_test_unique_arrays_read"],
    }


def compact_metrics(base, metrics):
    return base.compact_metrics(metrics)


def save_scratch_checkpoint(
    path,
    epoch_item,
    model,
    optimizer,
    scheduler,
    dataset,
    config,
    normalizer,
    normalizer_report,
    initialization_report,
    protocol_report,
    smoke,
    history,
):
    payload = {
        "experiment": EXPERIMENT,
        "experiment_role": "initialization_control_under_transfer_schedule",
        "not_an_optimized_scratch_recipe": True,
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
        "feature_definition": protocol_report["feature_definition"],
        "config": config.to_dict(),
        "normalizer_mean": normalizer.mean,
        "normalizer_scale": normalizer.scale,
        "normalizer_source": "exactly reused from daily20_transfer_seed42",
        "normalizer_reuse": normalizer_report,
        "initialization": initialization_report,
        "training_protocol": protocol_report,
        "best_train": dict(epoch_item["train"]),
        "best_validation": dict(epoch_item["val"]),
        "history": list(history),
        "label_smoothing": LABEL_SMOOTHING,
        "loss_contract": LOSS_CONTRACT,
        "preflight": smoke,
        "train_signers": list(reference_signers("train")),
        "val_signers": list(reference_signers("val")),
        "test_signers": list(reference_signers("test")),
        "test_evaluated": False,
        "stage_test_status": "NOT EVALUATED",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def reference_signers(split: str):
    if split == "train":
        return tuple(f"Participant_{index:02d}" for index in range(1, 8))
    if split == "val":
        return ("Participant_08",)
    if split == "test":
        return ("Participant_09", "Participant_10")
    raise ValueError(split)


def write_validation_predictions(
    base, report_dir, dataset, val_indices, y_true, y_pred, probabilities
):
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
    return rows


def comparison_rows(transfer_metrics, scratch_metrics):
    rows = []
    for initialization, metrics in (
        ("nationalcsl200_pretrained", transfer_metrics),
        ("random_seed42", scratch_metrics),
    ):
        rows.append(
            {
                "experiment": metrics["experiment"],
                "initialization": initialization,
                "best_epoch": metrics["best_epoch"],
                "best_phase": metrics["best_phase"],
                "best_phase_epoch": metrics["best_phase_epoch"],
                "train_accuracy": f"{metrics['best_train']['accuracy']:.8f}",
                "train_macro_f1": f"{metrics['best_train']['macro_f1']:.8f}",
                "train_loss": f"{metrics['best_train']['loss']:.8f}",
                "val_accuracy": f"{metrics['best_validation']['accuracy']:.8f}",
                "val_macro_f1": f"{metrics['best_validation']['macro_f1']:.8f}",
                "val_loss": f"{metrics['best_validation']['loss']:.8f}",
                "loss_contract": LOSS_CONTRACT,
                "stage_test": "NOT EVALUATED",
            }
        )
    return rows


def train(args):
    if args.threads <= 0:
        raise ValueError("--threads must be positive.")
    reference, base = load_reference_modules()
    torch.set_num_threads(args.threads)
    device = base.select_device(args.device)
    (
        dataset,
        config,
        reference_payload,
        reference_config,
        split_indices,
        normalizer,
        normalizer_report,
        paths,
    ) = build_components(args, reference, base, device)
    validate_output_paths(paths, args.overwrite)
    paths["report_dir"].mkdir(parents=True, exist_ok=True)

    smoke = one_batch_gate(
        reference, base, dataset, config, split_indices, normalizer, device
    )
    print("Preflight gate: " + json.dumps(smoke, ensure_ascii=False))

    # Reset after all metadata/normalizer reads and after the disposable smoke run.
    model, loaders = make_seeded_model_and_loaders(
        reference, base, dataset, config, split_indices, normalizer, device
    )
    formal_initial_hash = model_state_hash(model)
    if formal_initial_hash != smoke["initial_model_state_sha256"]:
        raise RuntimeError("Formal scratch initialization differs from smoke initialization.")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    initialization_report = {
        "mode": "all_parameters_random",
        "seed": SEED,
        "seed_reset_after_metadata_and_normalizer_load": True,
        "model_state_dict_loaded_before_training": False,
        "pretrained_model_parameters_loaded": 0,
        "initial_model_state_sha256": formal_initial_hash,
        "control_interpretation": (
            "initialization control under the identical transfer schedule; "
            "not an optimized from-scratch configuration"
        ),
    }
    protocol_report = {
        "reference_experiment": REFERENCE_EXPERIMENT,
        "same_data": True,
        "same_class_order": True,
        "same_model_structure": True,
        "same_normalizer_values": True,
        "same_augmentation": True,
        "same_optimizer": "AdamW",
        "same_learning_rates": True,
        "same_freeze_unfreeze_schedule": True,
        "same_max_epochs": True,
        "same_early_stopping": True,
        "same_selection_rule": True,
        "same_loss_contract": True,
        "only_experimental_variable": "model parameter initialization",
        "phase1_epochs": PHASE1_EPOCHS,
        "phase1_learning_rate": PHASE1_LEARNING_RATE,
        "phase2_max_epochs": PHASE2_MAX_EPOCHS,
        "phase2_learning_rate": PHASE2_LEARNING_RATE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "selection_rule": SELECTION_RULE,
        "loss_contract": LOSS_CONTRACT,
        "feature_definition": reference_payload["feature_definition"],
    }

    print(f"Device: {device}")
    print("Experiment: random initialization control, seed=42, one run")
    print("No model state_dict loaded; stage test arrays WILL NOT BE READ")
    print(f"Selection rule: {SELECTION_RULE}")

    history = []
    best_key: Optional[Tuple[float, float, float]] = None
    best_epoch_item = None
    global_epoch = 0

    phase1_names = set_phase_trainability(base, model, "phase1_head")
    phase1_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=PHASE1_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    phase1_optimizer_names = optimizer_parameter_names(model, phase1_optimizer)
    if phase1_optimizer_names != phase1_names:
        raise RuntimeError("Phase-1 optimizer group mismatch.")
    protocol_report["phase1_trainable_parameter_names"] = phase1_names
    protocol_report["phase1_optimizer_parameter_names"] = phase1_optimizer_names
    protocol_report["random_encoder_frozen_during_phase1"] = True
    protocol_report["unfreeze_before_global_epoch"] = PHASE1_EPOCHS + 1

    print(f"Phase 1: random encoder frozen, classifier head only, epochs={PHASE1_EPOCHS}")
    for phase_epoch in range(1, PHASE1_EPOCHS + 1):
        global_epoch += 1
        train_metrics, _, _ = base.run_loader(
            model,
            loaders["train"],
            criterion,
            device,
            len(dataset.class_info),
            phase1_optimizer,
        )
        val_metrics, _, _ = base.run_loader(
            model, loaders["val"], criterion, device, len(dataset.class_info)
        )
        candidate = reference.selection_key(val_metrics)
        improved = best_key is None or candidate > best_key
        if improved:
            best_key = candidate
        item = {
            "global_epoch": global_epoch,
            "phase": "phase1_head",
            "phase_epoch": phase_epoch,
            "learning_rate": phase1_optimizer.param_groups[0]["lr"],
            "train": compact_metrics(base, train_metrics),
            "val": compact_metrics(base, val_metrics),
            "selection_key": list(candidate),
            "is_best": improved,
        }
        history.append(item)
        reference.print_epoch(item)
        if improved:
            best_epoch_item = item
            save_scratch_checkpoint(
                paths["checkpoint"],
                item,
                model,
                phase1_optimizer,
                None,
                dataset,
                config,
                normalizer,
                normalizer_report,
                initialization_report,
                protocol_report,
                smoke,
                history,
            )

    phase2_names = set_phase_trainability(base, model, "phase2_finetune")
    phase2_optimizer = torch.optim.AdamW(
        model.parameters(), lr=PHASE2_LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    phase2_optimizer_names = optimizer_parameter_names(model, phase2_optimizer)
    if phase2_optimizer_names != phase2_names:
        raise RuntimeError("Phase-2 optimizer group mismatch.")
    protocol_report["phase2_trainable_parameter_names"] = phase2_names
    protocol_report["phase2_optimizer_parameter_names"] = phase2_optimizer_names
    protocol_report["all_parameters_unfrozen_for_phase2"] = True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        phase2_optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    stale_epochs = 0
    print(
        f"Phase 2: all parameters unfrozen, max_epochs={PHASE2_MAX_EPOCHS}, "
        f"patience={EARLY_STOPPING_PATIENCE}"
    )
    for phase_epoch in range(1, PHASE2_MAX_EPOCHS + 1):
        global_epoch += 1
        train_metrics, _, _ = base.run_loader(
            model,
            loaders["train"],
            criterion,
            device,
            len(dataset.class_info),
            phase2_optimizer,
        )
        val_metrics, _, _ = base.run_loader(
            model, loaders["val"], criterion, device, len(dataset.class_info)
        )
        scheduler.step(val_metrics["macro_f1"])
        candidate = reference.selection_key(val_metrics)
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
            "learning_rate": phase2_optimizer.param_groups[0]["lr"],
            "train": compact_metrics(base, train_metrics),
            "val": compact_metrics(base, val_metrics),
            "selection_key": list(candidate),
            "is_best": improved,
        }
        history.append(item)
        reference.print_epoch(item)
        if improved:
            best_epoch_item = item
            save_scratch_checkpoint(
                paths["checkpoint"],
                item,
                model,
                phase2_optimizer,
                scheduler,
                dataset,
                config,
                normalizer,
                normalizer_report,
                initialization_report,
                protocol_report,
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
        raise RuntimeError("No scratch checkpoint was saved.")

    saved = base.load_checkpoint(paths["checkpoint"], device)
    best_model = base.MaskAwareSmallTransformer(
        len(dataset.class_info), config
    ).to(device)
    best_model.load_state_dict(saved["model_state_dict"])
    val_metrics, y_true, y_pred, probabilities = (
        reference.run_validation_with_probabilities(
            base,
            best_model,
            loaders["val"],
            criterion,
            device,
            len(dataset.class_info),
        )
    )
    if reference.selection_key(val_metrics) != reference.selection_key(
        saved["best_validation"]
    ):
        raise RuntimeError("Reloaded scratch checkpoint metrics differ from saved best.")
    feature_access = reference.access_audit(dataset, split_indices)
    expected_access = {
        "train_unique_arrays_read": 140,
        "validation_unique_arrays_read": 20,
        "stage_test_unique_arrays_read": 0,
        "stage_test_evaluated": False,
    }
    if feature_access != expected_access:
        raise RuntimeError(f"Unexpected feature access: {feature_access}")

    report_dir = paths["report_dir"]
    reference.write_training_history(base, report_dir, history)
    scratch_prediction_rows = write_validation_predictions(
        base,
        report_dir,
        dataset,
        split_indices["val"],
        y_true,
        y_pred,
        probabilities,
    )

    transfer_metrics = read_json(paths["reference_tuning_metrics"])
    transfer_predictions = read_csv(paths["reference_predictions"])
    scratch_metrics = {
        "experiment": EXPERIMENT,
        "best_epoch": int(saved["epoch"]),
        "best_phase": saved["best_phase"],
        "best_phase_epoch": int(saved["best_phase_epoch"]),
        "best_train": saved["best_train"],
        "best_validation": compact_metrics(base, val_metrics),
    }
    changes = compare_predictions(transfer_predictions, scratch_prediction_rows)
    improved_samples = [
        item for item in changes if item["change_type"] == "incorrect_to_correct"
    ]
    regressed_samples = [
        item for item in changes if item["change_type"] == "correct_to_incorrect"
    ]

    rows = comparison_rows(transfer_metrics, scratch_metrics)
    base.write_csv(
        report_dir / "initialization_comparison.csv",
        (
            "experiment",
            "initialization",
            "best_epoch",
            "best_phase",
            "best_phase_epoch",
            "train_accuracy",
            "train_macro_f1",
            "train_loss",
            "val_accuracy",
            "val_macro_f1",
            "val_loss",
            "loss_contract",
            "stage_test",
        ),
        rows,
    )

    training_config = {
        "experiment": EXPERIMENT,
        "experiment_role": "initialization_control_under_transfer_schedule",
        "not_an_optimized_scratch_recipe": True,
        "seed": SEED,
        "test_evaluated": False,
        "stage_test": "NOT EVALUATED",
        "paths": {key: str(value) for key, value in paths.items()},
        "model_config": config.to_dict(),
        "initialization": initialization_report,
        "normalizer_reuse": normalizer_report,
        "training_protocol": protocol_report,
        "preflight": smoke,
        "feature_access": feature_access,
    }
    write_json(report_dir / "training_config.json", training_config)
    tuning = {
        **scratch_metrics,
        "seed": SEED,
        "test_evaluated": False,
        "stage_test": "NOT EVALUATED",
        "epochs_ran": len(history),
        "selection_rule": SELECTION_RULE,
        "loss_contract": LOSS_CONTRACT,
        "feature_access": feature_access,
        "initialization_comparison_csv": str(
            report_dir / "initialization_comparison.csv"
        ),
        "prediction_comparison": changes,
        "incorrect_to_correct": improved_samples,
        "correct_to_incorrect": regressed_samples,
    }
    write_json(report_dir / "tuning_metrics.json", tuning)

    saved["history"] = history
    saved["epochs_ran"] = len(history)
    saved["training_complete"] = True
    saved["best_validation"] = compact_metrics(base, val_metrics)
    saved["feature_access"] = feature_access
    saved["prediction_comparison"] = changes
    temporary = paths["checkpoint"].with_name(
        paths["checkpoint"].name + ".final.tmp"
    )
    torch.save(saved, temporary)
    temporary.replace(paths["checkpoint"])

    required = [
        paths["checkpoint"],
        report_dir / "training_config.json",
        report_dir / "training_history.csv",
        report_dir / "tuning_metrics.json",
        report_dir / "validation_predictions.csv",
        report_dir / "initialization_comparison.csv",
    ]
    missing = [
        str(path) for path in required if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Required outputs missing or empty: {missing}")

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
    print(f"Incorrect -> correct: {len(improved_samples)}")
    print(f"Correct -> incorrect: {len(regressed_samples)}")
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

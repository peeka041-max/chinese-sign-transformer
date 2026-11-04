"""Transfer-tune the domain MVP 20-class Transformer.

The tune stage reads train/validation feature arrays only. Participant_09 and
Participant_10 are reserved for an explicit final stage and are not loaded or
evaluated during tune or preflight checks.

Only load checkpoints produced by trusted local training runs.
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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFORMER_ROOT = SCRIPT_DIR.parent
NATIONALCSL200_DIR = TRANSFORMER_ROOT / "nationalcsl200_stage_v1"

for import_root in (TRANSFORMER_ROOT, NATIONALCSL200_DIR):
    import_value = str(import_root)
    if import_value not in sys.path:
        sys.path.insert(0, import_value)

from config_v2 import DEFAULT_CONFIG, config_from_mapping  # noqa: E402
from feature_io import verified_sequence  # noqa: E402
from feature_processing import MaskAwareStandardizer, augment_sequence  # noqa: E402
from features_v2 import PRESENCE_SLICE  # noqa: E402
from model_v2 import SinusoidalPositionalEncoding, count_trainable_parameters  # noqa: E402
from training_support import (  # noqa: E402
    classification_metrics,
    compact_metrics,
    load_checkpoint,
    select_device,
    set_random_seed,
    write_csv,
)


TRAIN_SIGNERS = tuple(f"Participant_{index:02d}" for index in range(1, 8))
VAL_SIGNERS = ("Participant_08",)
TEST_SIGNERS = ("Participant_09", "Participant_10")
ALL_SIGNERS = TRAIN_SIGNERS + VAL_SIGNERS + TEST_SIGNERS
CLASSIFIER_KEYS = {"output.2.weight", "output.2.bias"}
REQUIRED_CLASS_COLUMNS = (
    "class_id",
    "gloss_id",
    "dataset_gloss",
    "display_word",
    "match_type",
)
REQUIRED_MANIFEST_COLUMNS = (
    "sample_id",
    "path",
    "feature_path",
    *REQUIRED_CLASS_COLUMNS,
    "signer_id",
    "split",
    "quality_status",
    "processing_status",
    "feature_sha256",
)


@dataclass(frozen=True)
class ClassInfo:
    class_id: int
    gloss_id: str
    dataset_gloss: str
    display_word: str
    match_type: str

    def to_row(self) -> Dict[str, str]:
        return {
            "class_id": str(self.class_id),
            "gloss_id": self.gloss_id,
            "dataset_gloss": self.dataset_gloss,
            "display_word": self.display_word,
            "match_type": self.match_type,
        }


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    path: Path
    label: int
    signer_id: str
    split: str
    gloss_id: str
    dataset_gloss: str
    display_word: str
    match_type: str
    quality_status: str
    feature_sha256: str


class DomainMvpDataset(Dataset):
    """Manifest-backed dataset preserving domain label metadata."""

    def __init__(self, data_root: Path, records: Sequence[SampleRecord], class_info: Sequence[ClassInfo], config):
        self.data_root = Path(data_root)
        self.samples = list(records)
        self.class_info = list(class_info)
        self.class_names = [item.display_word for item in self.class_info]
        self.config = config
        self.config.validate()
        self.feature_checksums = {
            str(record.path.resolve()): record.feature_sha256 for record in self.samples
        }
        self.verified_features = set()
        self.data_signature = ""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        sequence = verified_sequence(self, record.path)
        return torch.from_numpy(sequence), torch.tensor(record.label, dtype=torch.long)


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
            raise ValueError(
                f"Expected [B, T, F] with (T, F)={expected}, got {tuple(inputs.shape)}"
            )

        frame_valid = inputs[:, :, PRESENCE_SLICE].sum(dim=-1) > 0.5
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transfer-tune a domain MVP 20-class Transformer from the NationalCSL200 checkpoint."
    )
    parser.add_argument("--data-root", type=Path, default=Path("domain_mvp_v1/domain20_v2"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--class-order", type=Path, default=Path("domain_mvp_v1/class_order.csv"))
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=Path("nationalcsl200_stage_v1/checkpoints_v2/nationalcsl200_transformer_optimized.pt"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("domain_mvp_v1/checkpoints/domain_mvp20_transfer.pt"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("domain_mvp_v1/reports/domain_mvp20_transfer"),
    )
    parser.add_argument("--stage", choices=("check-transfer", "check-batch", "tune", "final"), default="tune")
    parser.add_argument("--phase1-epochs", type=int, default=10)
    parser.add_argument("--phase1-learning-rate", type=float, default=1e-3)
    parser.add_argument("--phase2-max-epochs", type=int, default=100)
    parser.add_argument("--phase2-learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace this domain MVP tune checkpoint/report.")
    parser.add_argument("--allow-repeat-final", action="store_true", help="Explicitly allow final held-out test evaluation.")
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (TRANSFORMER_ROOT / path).resolve()


def read_csv_rows(path: Path, required_columns: Iterable[str]) -> Tuple[List[Dict[str, str]], List[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = set(required_columns) - set(fieldnames)
        if missing:
            raise ValueError(f"Missing CSV columns {sorted(missing)} in {path}")
        return list(reader), fieldnames


def normalize_gloss_id(value) -> str:
    text = str(value).strip()
    if not text.isascii() or not text.isdigit() or len(text) > 4:
        raise ValueError(f"Expected numeric gloss_id with at most four digits, got {text!r}")
    return text.zfill(4)


def split_for_signer(signer_id: str) -> str:
    if signer_id in TRAIN_SIGNERS:
        return "train"
    if signer_id in VAL_SIGNERS:
        return "val"
    if signer_id in TEST_SIGNERS:
        return "test"
    raise ValueError(f"Unexpected signer_id: {signer_id}")


def canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_class_order(path: Path) -> List[ClassInfo]:
    rows, _ = read_csv_rows(path, REQUIRED_CLASS_COLUMNS)
    classes = []
    for row_number, row in enumerate(rows, start=2):
        try:
            class_id = int(str(row["class_id"]).strip())
        except ValueError as exc:
            raise ValueError(f"Invalid class_id on row {row_number}: {row['class_id']!r}") from exc
        classes.append(
            ClassInfo(
                class_id=class_id,
                gloss_id=normalize_gloss_id(row["gloss_id"]),
                dataset_gloss=str(row["dataset_gloss"]).strip(),
                display_word=str(row["display_word"]).strip(),
                match_type=str(row["match_type"]).strip(),
            )
        )
    if len(classes) != 20:
        raise ValueError(f"Expected exactly 20 classes, found {len(classes)}.")
    if [item.class_id for item in classes] != list(range(20)):
        raise ValueError("class_id must be contiguous from 0 to 19.")
    if len({item.gloss_id for item in classes}) != len(classes):
        raise ValueError("Duplicate gloss_id in class_order.csv.")
    if len({item.display_word for item in classes}) != len(classes):
        raise ValueError("Duplicate display_word in class_order.csv.")
    for item in classes:
        if not item.dataset_gloss or not item.display_word or not item.match_type:
            raise ValueError(f"Empty class metadata for class_id={item.class_id}.")
    return classes


def load_manifest(data_root: Path, manifest_path: Path, classes: Sequence[ClassInfo]) -> List[SampleRecord]:
    rows, _ = read_csv_rows(manifest_path, REQUIRED_MANIFEST_COLUMNS)
    class_by_id = {str(item.class_id): item for item in classes}
    records = []
    for row_number, row in enumerate(rows, start=2):
        status = str(row["processing_status"]).strip()
        if status != "success":
            raise ValueError(f"Manifest row {row_number} is not successful: {row.get('sample_id')}")
        class_id_text = str(row["class_id"]).strip()
        if class_id_text not in class_by_id:
            raise ValueError(f"Manifest row {row_number} has unknown class_id={class_id_text!r}")
        class_info = class_by_id[class_id_text]
        gloss_id = normalize_gloss_id(row["gloss_id"])
        signer_id = str(row["signer_id"]).strip()
        split = str(row["split"]).strip()
        if split != split_for_signer(signer_id):
            raise ValueError(f"Manifest row {row_number} split/signature mismatch for {signer_id}.")
        expected_values = {
            "gloss_id": class_info.gloss_id,
            "dataset_gloss": class_info.dataset_gloss,
            "display_word": class_info.display_word,
            "match_type": class_info.match_type,
        }
        for key, expected in expected_values.items():
            actual = gloss_id if key == "gloss_id" else str(row[key]).strip()
            if actual != expected:
                raise ValueError(
                    f"Manifest row {row_number} {key}={actual!r}, expected {expected!r}."
                )
        feature_path = Path(str(row["feature_path"]).strip() or str(row["path"]).strip())
        if not feature_path.is_absolute():
            feature_path = data_root / feature_path
        feature_sha256 = str(row["feature_sha256"]).strip()
        if len(feature_sha256) != 64 or any(char not in "0123456789abcdef" for char in feature_sha256):
            raise ValueError(f"Manifest row {row_number} has invalid feature_sha256.")
        if not feature_path.is_file():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        records.append(
            SampleRecord(
                sample_id=str(row["sample_id"]).strip(),
                path=feature_path.resolve(),
                label=class_info.class_id,
                signer_id=signer_id,
                split=split,
                gloss_id=class_info.gloss_id,
                dataset_gloss=class_info.dataset_gloss,
                display_word=class_info.display_word,
                match_type=class_info.match_type,
                quality_status=str(row["quality_status"]).strip(),
                feature_sha256=feature_sha256,
            )
        )
    validate_records(records, classes)
    return records


def validate_records(records: Sequence[SampleRecord], classes: Sequence[ClassInfo]) -> None:
    if len(records) != 200:
        raise ValueError(f"Expected 200 manifest rows, found {len(records)}.")
    seen = set()
    for record in records:
        if record.sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {record.sample_id}")
        seen.add(record.sample_id)
    for split_name, expected_count in (("train", 140), ("val", 20), ("test", 40)):
        count = sum(1 for record in records if record.split == split_name)
        if count != expected_count:
            raise ValueError(f"Expected {expected_count} {split_name} samples, found {count}.")
    for class_info in classes:
        class_records = [record for record in records if record.label == class_info.class_id]
        if len(class_records) != 10:
            raise ValueError(f"Expected 10 samples for class_id={class_info.class_id}, found {len(class_records)}.")
    for signer_id in ALL_SIGNERS:
        count = sum(1 for record in records if record.signer_id == signer_id)
        if count != 20:
            raise ValueError(f"Expected 20 samples for {signer_id}, found {count}.")


def indices_for_signers(dataset: DomainMvpDataset, signer_ids: Sequence[str]) -> List[int]:
    signer_set = set(signer_ids)
    return [index for index, record in enumerate(dataset.samples) if record.signer_id in signer_set]


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
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
    return loaders


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


def build_config_from_pretrained(args, data_root: Path, checkpoint_path: Path, pretrained_payload: Mapping):
    if pretrained_payload.get("model_kind") != "mask_aware_small_transformer":
        raise ValueError(f"Unsupported pretrained model_kind: {pretrained_payload.get('model_kind')!r}")
    pretrained_config = config_from_mapping(pretrained_payload["config"])
    if pretrained_config.input_dim != 266:
        raise ValueError(f"Pretrained input_dim must be 266, got {pretrained_config.input_dim}.")
    if pretrained_config.sequence_length != 30:
        raise ValueError(
            f"Pretrained sequence_length must be 30, got {pretrained_config.sequence_length}."
        )
    config = replace(
        pretrained_config,
        data_root=str(data_root),
        checkpoint_path=str(checkpoint_path),
        batch_size=args.batch_size,
        learning_rate=args.phase2_learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.phase2_max_epochs,
        early_stopping_patience=args.patience,
        random_seed=args.seed,
        num_workers=0,
    )
    config.validate()
    return config


def is_classifier_key(key: str) -> bool:
    return key in CLASSIFIER_KEYS


def load_transfer_weights(model: nn.Module, pretrained_payload: Mapping) -> Dict[str, object]:
    pretrained_state = pretrained_payload["model_state_dict"]
    model_state = model.state_dict()
    compatible = {}
    skipped_classifier = {}
    unexpected_keys = []
    shape_mismatches = []
    for key, value in pretrained_state.items():
        if is_classifier_key(key):
            skipped_classifier[key] = {
                "pretrained_shape": list(value.shape),
                "new_shape": list(model_state[key].shape) if key in model_state else None,
            }
            continue
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "pretrained_shape": list(value.shape),
                    "new_shape": list(model_state[key].shape),
                }
            )
            continue
        compatible[key] = value

    if unexpected_keys or shape_mismatches:
        raise RuntimeError(
            "Non-classifier transfer mismatch: "
            + json.dumps(
                {"unexpected_keys": unexpected_keys, "shape_mismatches": shape_mismatches},
                ensure_ascii=False,
            )
        )

    incompatible = model.load_state_dict(compatible, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_after_load = list(incompatible.unexpected_keys)
    non_classifier_missing = [key for key in missing_keys if not is_classifier_key(key)]
    non_classifier_unexpected = [key for key in unexpected_after_load if not is_classifier_key(key)]
    if non_classifier_missing or non_classifier_unexpected:
        raise RuntimeError(
            "Transfer load left non-classifier keys unmatched: "
            + json.dumps(
                {
                    "missing_keys": non_classifier_missing,
                    "unexpected_keys": non_classifier_unexpected,
                },
                ensure_ascii=False,
            )
        )

    return {
        "loaded_tensor_count": len(compatible),
        "loaded_parameter_count": int(
            sum(int(value.numel()) for value in compatible.values() if torch.is_tensor(value))
        ),
        "skipped_classifier_parameters": skipped_classifier,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_after_load,
        "pretrained_class_count": len(pretrained_payload.get("class_names", [])),
        "new_class_count": int(model.output[2].out_features),
    }


def set_head_only_trainable(model: nn.Module) -> None:
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False
    for parameter in model.output[2].parameters():
        parameter.requires_grad = True


def set_all_trainable(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def class_rows(classes: Sequence[ClassInfo]) -> List[Dict[str, str]]:
    return [item.to_row() for item in classes]


def class_metadata_by_id(dataset: DomainMvpDataset, class_id: int) -> ClassInfo:
    return dataset.class_info[int(class_id)]


def write_training_history(report_dir: Path, history: Sequence[Mapping[str, object]]) -> None:
    rows = []
    for item in history:
        train = item["train"]
        val = item["val"]
        rows.append(
            {
                "global_epoch": item["global_epoch"],
                "phase": item["phase"],
                "phase_epoch": item["phase_epoch"],
                "learning_rate": f"{float(item['learning_rate']):.10f}",
                "train_loss": f"{train['loss']:.6f}",
                "train_accuracy": f"{train['accuracy']:.6f}",
                "train_macro_precision": f"{train['macro_precision']:.6f}",
                "train_macro_recall": f"{train['macro_recall']:.6f}",
                "train_macro_f1": f"{train['macro_f1']:.6f}",
                "validation_loss": f"{val['loss']:.6f}",
                "validation_accuracy": f"{val['accuracy']:.6f}",
                "validation_macro_precision": f"{val['macro_precision']:.6f}",
                "validation_macro_recall": f"{val['macro_recall']:.6f}",
                "validation_macro_f1": f"{val['macro_f1']:.6f}",
                "is_best": str(bool(item["is_best"])).lower(),
            }
        )
    write_csv(
        report_dir / "training_history.csv",
        (
            "global_epoch",
            "phase",
            "phase_epoch",
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "train_macro_precision",
            "train_macro_recall",
            "train_macro_f1",
            "validation_loss",
            "validation_accuracy",
            "validation_macro_precision",
            "validation_macro_recall",
            "validation_macro_f1",
            "is_best",
        ),
        rows,
    )


def write_validation_reports(
    report_dir: Path,
    dataset: DomainMvpDataset,
    val_indices: Sequence[int],
    metrics: Mapping[str, object],
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> None:
    per_class_rows = []
    for class_id, item in enumerate(dataset.class_info):
        per_class_rows.append(
            {
                "class_id": class_id,
                "gloss_id": item.gloss_id,
                "dataset_gloss": item.dataset_gloss,
                "display_word": item.display_word,
                "match_type": item.match_type,
                "precision": f"{metrics['per_class_precision'][class_id]:.6f}",
                "recall": f"{metrics['per_class_recall'][class_id]:.6f}",
                "f1": f"{metrics['per_class_f1'][class_id]:.6f}",
                "support": metrics["per_class_support"][class_id],
            }
        )
    write_csv(
        report_dir / "validation_per_class_metrics.csv",
        (
            "class_id",
            "gloss_id",
            "dataset_gloss",
            "display_word",
            "match_type",
            "precision",
            "recall",
            "f1",
            "support",
        ),
        per_class_rows,
    )

    prediction_rows = []
    for dataset_index, true_id, predicted_id in zip(val_indices, y_true, y_pred):
        record = dataset.samples[dataset_index]
        true_info = class_metadata_by_id(dataset, true_id)
        predicted_info = class_metadata_by_id(dataset, predicted_id)
        prediction_rows.append(
            {
                "sample_id": record.sample_id,
                "signer_id": record.signer_id,
                "true_class_id": true_id,
                "true_gloss_id": true_info.gloss_id,
                "true_dataset_gloss": true_info.dataset_gloss,
                "true_display_word": true_info.display_word,
                "predicted_class_id": predicted_id,
                "predicted_gloss_id": predicted_info.gloss_id,
                "predicted_dataset_gloss": predicted_info.dataset_gloss,
                "predicted_display_word": predicted_info.display_word,
                "correct": str(true_id == predicted_id).lower(),
            }
        )
    write_csv(
        report_dir / "validation_predictions.csv",
        (
            "sample_id",
            "signer_id",
            "true_class_id",
            "true_gloss_id",
            "true_dataset_gloss",
            "true_display_word",
            "predicted_class_id",
            "predicted_gloss_id",
            "predicted_dataset_gloss",
            "predicted_display_word",
            "correct",
        ),
        prediction_rows,
    )

    with (report_dir / "validation_confusion_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *[item.display_word for item in dataset.class_info]])
        for item, values in zip(dataset.class_info, metrics["confusion_matrix"]):
            writer.writerow([item.display_word, *values])


def save_checkpoint(
    path: Path,
    epoch_info: Mapping[str, object],
    model: nn.Module,
    optimizer,
    scheduler,
    dataset: DomainMvpDataset,
    config,
    normalizer: MaskAwareStandardizer,
    best_validation: Mapping[str, object],
    history: Sequence[Mapping[str, object]],
    args,
    transfer_report: Mapping[str, object],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "data_signature": dataset.data_signature,
            "epoch": int(epoch_info["global_epoch"]),
            "best_phase": epoch_info["phase"],
            "best_phase_epoch": int(epoch_info["phase_epoch"]),
            "model_kind": "mask_aware_small_transformer",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "class_names": list(dataset.class_names),
            "class_order": [asdict(item) for item in dataset.class_info],
            "config": config.to_dict(),
            "normalizer_mean": normalizer.mean,
            "normalizer_scale": normalizer.scale,
            "normalizer_source": "domain_mvp20 train split only",
            "best_validation": compact_metrics(best_validation),
            "history": list(history),
            "label_smoothing": float(args.label_smoothing),
            "pretrained_checkpoint": str(args.pretrained_checkpoint),
            "transfer_load": transfer_report,
            "train_signers": list(TRAIN_SIGNERS),
            "val_signers": list(VAL_SIGNERS),
            "test_signers": list(TEST_SIGNERS),
            "test_evaluated": False,
        },
        temporary,
    )
    temporary.replace(path)


def write_training_config(
    report_dir: Path,
    args,
    dataset: DomainMvpDataset,
    config,
    class_order_path: Path,
    manifest_path: Path,
    transfer_report: Mapping[str, object],
) -> None:
    payload = {
        "stage": args.stage,
        "test_evaluated": False,
        "data_root": str(args.data_root),
        "manifest": str(manifest_path),
        "class_order": str(class_order_path),
        "checkpoint": str(args.checkpoint),
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "seed": args.seed,
        "split": {
            "train_signers": list(TRAIN_SIGNERS),
            "val_signers": list(VAL_SIGNERS),
            "test_signers": list(TEST_SIGNERS),
            "train_samples": 140,
            "val_samples": 20,
            "test_samples": 40,
        },
        "phase1": {
            "frozen_backbone": True,
            "trainable": "output.2 classifier head only",
            "epochs": args.phase1_epochs,
            "learning_rate": args.phase1_learning_rate,
        },
        "phase2": {
            "frozen_backbone": False,
            "trainable": "all parameters",
            "max_epochs": args.phase2_max_epochs,
            "learning_rate": args.phase2_learning_rate,
            "early_stopping_patience": args.patience,
            "best_epoch_metric": "validation_macro_f1",
        },
        "model_config": config.to_dict(),
        "normalizer_source": "domain_mvp20 train split only",
        "training_only_augmentation": True,
        "invalid_frame_attention_mask": True,
        "label_smoothing": args.label_smoothing,
        "weight_decay": args.weight_decay,
        "class_order_rows": [asdict(item) for item in dataset.class_info],
        "transfer_load": transfer_report,
        "data_signature": dataset.data_signature,
    }
    with (report_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_components(args, device):
    data_root = resolve_project_path(args.data_root)
    manifest_path = resolve_project_path(args.manifest) if args.manifest else data_root / "manifest.csv"
    class_order_path = resolve_project_path(args.class_order)
    checkpoint_path = resolve_project_path(args.checkpoint)
    report_dir = resolve_project_path(args.report_dir)
    pretrained_path = resolve_project_path(args.pretrained_checkpoint)
    args.data_root = data_root
    args.manifest = manifest_path
    args.class_order = class_order_path
    args.checkpoint = checkpoint_path
    args.report_dir = report_dir
    args.pretrained_checkpoint = pretrained_path

    if not pretrained_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
    pretrained_payload = load_checkpoint(pretrained_path, device)
    config = build_config_from_pretrained(args, data_root, checkpoint_path, pretrained_payload)
    classes = load_class_order(class_order_path)
    records = load_manifest(data_root, manifest_path, classes)
    dataset = DomainMvpDataset(data_root, records, classes, config)
    dataset.data_signature = canonical_hash(
        {
            "manifest_sha256": file_hash(manifest_path),
            "class_order_sha256": file_hash(class_order_path),
            "split": {
                "train": list(TRAIN_SIGNERS),
                "val": list(VAL_SIGNERS),
                "test": list(TEST_SIGNERS),
            },
        }
    )
    split_indices = {
        "train": indices_for_signers(dataset, TRAIN_SIGNERS),
        "val": indices_for_signers(dataset, VAL_SIGNERS),
        "test": indices_for_signers(dataset, TEST_SIGNERS),
    }
    actual = tuple(len(split_indices[name]) for name in ("train", "val", "test"))
    if actual != (140, 20, 40):
        raise RuntimeError(f"Expected split sizes (140, 20, 40), found {actual}.")
    return dataset, config, pretrained_payload, checkpoint_path, report_dir, class_order_path, manifest_path, split_indices


def validate_output_paths(args) -> None:
    if args.stage != "tune":
        return
    checkpoint = Path(args.checkpoint)
    report_dir = Path(args.report_dir)
    if not args.overwrite:
        if checkpoint.exists():
            raise FileExistsError(
                f"Checkpoint already exists: {checkpoint}. Use --overwrite only for an intentional rerun."
            )
        if report_dir.exists() and any(report_dir.iterdir()):
            raise FileExistsError(
                f"Report directory is not empty: {report_dir}. Use --overwrite only for an intentional rerun."
            )


def evaluate_validation_from_checkpoint(
    checkpoint_path: Path,
    dataset: DomainMvpDataset,
    config,
    normalizer: MaskAwareStandardizer,
    loaders: Mapping[str, DataLoader],
    criterion,
    device,
):
    saved = load_checkpoint(checkpoint_path, device)
    model = MaskAwareSmallTransformer(len(dataset.class_info), config).to(device)
    model.load_state_dict(saved["model_state_dict"])
    return run_loader(model, loaders["val"], criterion, device, len(dataset.class_info))


def train_tune(args, dataset, config, pretrained_payload, checkpoint_path, report_dir, class_order_path, manifest_path, split_indices, device):
    validate_output_paths(args)
    report_dir.mkdir(parents=True, exist_ok=True)

    normalizer = MaskAwareStandardizer.fit(dataset, split_indices["train"])
    loaders = make_loaders(dataset, split_indices, normalizer, config, args.seed, include_test=False)
    model = MaskAwareSmallTransformer(len(dataset.class_info), config).to(device)
    transfer_report = load_transfer_weights(model, pretrained_payload)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    print(f"Device: {device}")
    print("Stage: tune (Participant_09/10 FEATURE ARRAYS will not be read)")
    print(
        "Pretrained config: "
        f"input_dim={config.input_dim}, sequence_length={config.sequence_length}, "
        f"d_model={config.d_model}, nhead={config.nhead}, "
        f"num_layers={config.num_layers}, dim_feedforward={config.dim_feedforward}, "
        f"dropout={config.dropout}"
    )
    print(
        "Transfer load: "
        f"loaded_tensors={transfer_report['loaded_tensor_count']}, "
        f"loaded_parameters={transfer_report['loaded_parameter_count']}, "
        f"missing_keys={transfer_report['missing_keys']}, "
        f"unexpected_keys={transfer_report['unexpected_keys']}"
    )

    best_macro_f1 = -math.inf
    best_epoch_info = None
    stale_phase2_epochs = 0
    history = []
    global_epoch = 0

    set_head_only_trainable(model)
    phase1_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.phase1_learning_rate,
        weight_decay=args.weight_decay,
    )
    phase1_scheduler = None
    print(f"Phase 1: head-only training for {args.phase1_epochs} epochs")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    for phase_epoch in range(1, args.phase1_epochs + 1):
        global_epoch += 1
        train_metrics, _, _ = run_loader(
            model, loaders["train"], criterion, device, len(dataset.class_info), phase1_optimizer
        )
        val_metrics, _, _ = run_loader(
            model, loaders["val"], criterion, device, len(dataset.class_info)
        )
        improved = val_metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
        history_item = {
            "global_epoch": global_epoch,
            "phase": "phase1_head",
            "phase_epoch": phase_epoch,
            "learning_rate": phase1_optimizer.param_groups[0]["lr"],
            "train": compact_metrics(train_metrics),
            "val": compact_metrics(val_metrics),
            "is_best": improved,
        }
        history.append(history_item)
        print_epoch(history_item)
        if improved:
            best_epoch_info = history_item
            save_checkpoint(
                checkpoint_path,
                best_epoch_info,
                model,
                phase1_optimizer,
                phase1_scheduler,
                dataset,
                config,
                normalizer,
                val_metrics,
                history,
                args,
                transfer_report,
            )
            print(f"  Saved best validation checkpoint: {checkpoint_path}")

    set_all_trainable(model)
    phase2_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.phase2_learning_rate, weight_decay=args.weight_decay
    )
    phase2_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        phase2_optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    print(f"Phase 2: full fine-tuning for up to {args.phase2_max_epochs} epochs")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    for phase_epoch in range(1, args.phase2_max_epochs + 1):
        global_epoch += 1
        train_metrics, _, _ = run_loader(
            model, loaders["train"], criterion, device, len(dataset.class_info), phase2_optimizer
        )
        val_metrics, _, _ = run_loader(
            model, loaders["val"], criterion, device, len(dataset.class_info)
        )
        phase2_scheduler.step(val_metrics["macro_f1"])
        improved = val_metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            stale_phase2_epochs = 0
        else:
            stale_phase2_epochs += 1
        history_item = {
            "global_epoch": global_epoch,
            "phase": "phase2_finetune",
            "phase_epoch": phase_epoch,
            "learning_rate": phase2_optimizer.param_groups[0]["lr"],
            "train": compact_metrics(train_metrics),
            "val": compact_metrics(val_metrics),
            "is_best": improved,
        }
        history.append(history_item)
        print_epoch(history_item)
        if improved:
            best_epoch_info = history_item
            save_checkpoint(
                checkpoint_path,
                best_epoch_info,
                model,
                phase2_optimizer,
                phase2_scheduler,
                dataset,
                config,
                normalizer,
                val_metrics,
                history,
                args,
                transfer_report,
            )
            print(f"  Saved best validation checkpoint: {checkpoint_path}")
        if stale_phase2_epochs >= args.patience:
            print(
                f"Early stopping after {stale_phase2_epochs} phase-2 epochs without validation Macro-F1 improvement."
            )
            break

    if best_epoch_info is None:
        raise RuntimeError("No validation checkpoint was saved.")

    val_metrics, val_true, val_pred = evaluate_validation_from_checkpoint(
        checkpoint_path, dataset, config, normalizer, loaders, criterion, device
    )
    best_checkpoint = load_checkpoint(checkpoint_path, device)
    write_training_history(report_dir, history)
    write_validation_reports(report_dir, dataset, split_indices["val"], val_metrics, val_true, val_pred)
    shutil.copyfile(class_order_path, report_dir / "class_order.csv")
    write_training_config(report_dir, args, dataset, config, class_order_path, manifest_path, transfer_report)
    tuning_summary = {
        "stage": "tune",
        "test_evaluated": False,
        "held_out_test": "NOT EVALUATED",
        "class_count": len(dataset.class_info),
        "data_signature": dataset.data_signature,
        "model": "mask_aware_small_transformer",
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_phase": best_checkpoint["best_phase"],
        "best_phase_epoch": int(best_checkpoint["best_phase_epoch"]),
        "best_validation": compact_metrics(val_metrics),
        "transfer_load": transfer_report,
        "split": {
            "train_signers": list(TRAIN_SIGNERS),
            "val_signers": list(VAL_SIGNERS),
            "test_signers": list(TEST_SIGNERS),
            "train_samples": len(split_indices["train"]),
            "val_samples": len(split_indices["val"]),
            "test_samples": len(split_indices["test"]),
        },
        "normalizer_source": "domain_mvp20 train split only",
        "history": history,
    }
    with (report_dir / "tuning_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(tuning_summary, handle, ensure_ascii=False, indent=2)

    print("=" * 76)
    print(
        f"Best epoch: {best_checkpoint['epoch']} "
        f"({best_checkpoint['best_phase']} epoch {best_checkpoint['best_phase_epoch']})"
    )
    print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation Macro-F1: {val_metrics['macro_f1']:.4f}")
    print(f"Validation loss: {val_metrics['loss']:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Held-out test: NOT EVALUATED")
    print(f"Reports: {report_dir}")
    print("=" * 76)


def print_epoch(history_item: Mapping[str, object]) -> None:
    train = history_item["train"]
    val = history_item["val"]
    print(
        f"Epoch {int(history_item['global_epoch']):03d} "
        f"{history_item['phase']}:{int(history_item['phase_epoch']):03d} "
        f"| Train loss={train['loss']:.4f} acc={train['accuracy']:.4f} F1={train['macro_f1']:.4f} "
        f"| Val loss={val['loss']:.4f} acc={val['accuracy']:.4f} F1={val['macro_f1']:.4f}"
        f"{' *' if history_item['is_best'] else ''}"
    )


def check_transfer(args, dataset, config, pretrained_payload, device) -> Dict[str, object]:
    model = MaskAwareSmallTransformer(len(dataset.class_info), config).to(device)
    report = load_transfer_weights(model, pretrained_payload)
    payload = {
        "check": "transfer_load",
        "input_dim": config.input_dim,
        "sequence_length": config.sequence_length,
        "d_model": config.d_model,
        "nhead": config.nhead,
        "num_layers": config.num_layers,
        "dim_feedforward": config.dim_feedforward,
        "dropout": config.dropout,
        **report,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def check_batch(args, dataset, config, pretrained_payload, split_indices, device) -> Dict[str, object]:
    normalizer = MaskAwareStandardizer.fit(dataset, split_indices["train"])
    loaders = make_loaders(dataset, split_indices, normalizer, config, args.seed, include_test=False)
    model = MaskAwareSmallTransformer(len(dataset.class_info), config).to(device)
    transfer_report = load_transfer_weights(model, pretrained_payload)
    set_head_only_trainable(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.phase1_learning_rate,
        weight_decay=args.weight_decay,
    )
    model.train(True)
    inputs, labels = next(iter(loaders["train"]))
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, labels)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    head_grad_norm = float(model.output[2].weight.grad.detach().norm().cpu())
    payload = {
        "check": "single_batch_forward_backward",
        "batch_size": int(inputs.shape[0]),
        "input_shape": list(inputs.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "head_grad_norm": head_grad_norm,
        "trainable_parameters": count_trainable_parameters(model),
        "test_evaluated": False,
        "transfer_loaded_parameters": transfer_report["loaded_parameter_count"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def evaluate_final(args, dataset, config, checkpoint_path, split_indices, device) -> None:
    if not args.allow_repeat_final:
        raise RuntimeError(
            "Final held-out test evaluation is disabled unless --allow-repeat-final is explicitly provided."
        )
    saved = load_checkpoint(checkpoint_path, device)
    if saved.get("test_evaluated") is not False:
        raise ValueError("Checkpoint does not declare test_evaluated=False.")
    if saved.get("data_signature") != dataset.data_signature:
        raise ValueError("Dataset signature differs from checkpoint.")
    if list(saved["class_names"]) != list(dataset.class_names):
        raise ValueError("Class order differs from checkpoint.")
    normalizer = MaskAwareStandardizer(saved["normalizer_mean"], saved["normalizer_scale"])
    loaders = make_loaders(dataset, split_indices, normalizer, config, args.seed, include_test=True)
    model = MaskAwareSmallTransformer(len(dataset.class_info), config).to(device)
    model.load_state_dict(saved["model_state_dict"])
    criterion = nn.CrossEntropyLoss(label_smoothing=float(saved["label_smoothing"]))
    test_metrics, _, _ = run_loader(
        model, loaders["test"], criterion, device, len(dataset.class_info)
    )
    print(json.dumps({"held_out_test": compact_metrics(test_metrics)}, ensure_ascii=False, indent=2))


def validate_args(args) -> None:
    if args.phase1_epochs <= 0:
        raise ValueError("--phase1-epochs must be positive.")
    if args.phase2_max_epochs <= 0:
        raise ValueError("--phase2-max-epochs must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.phase1_learning_rate <= 0.0 or args.phase2_learning_rate <= 0.0:
        raise ValueError("Learning rates must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative.")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1).")
    if args.threads <= 0:
        raise ValueError("--threads must be positive.")


def main():
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.threads)
    set_random_seed(args.seed)
    device = select_device(args.device)
    dataset, config, pretrained_payload, checkpoint_path, report_dir, class_order_path, manifest_path, split_indices = build_components(args, device)
    print(f"Classes: {len(dataset.class_info)}")
    print(
        f"Samples: train={len(split_indices['train'])}, "
        f"val={len(split_indices['val'])}, test={len(split_indices['test'])}"
    )
    print("Split: train=Participant_01..07, val=Participant_08, test=Participant_09..10")
    print(f"Dataset signature: {dataset.data_signature}")
    if args.stage == "check-transfer":
        check_transfer(args, dataset, config, pretrained_payload, device)
    elif args.stage == "check-batch":
        check_batch(args, dataset, config, pretrained_payload, split_indices, device)
    elif args.stage == "tune":
        train_tune(
            args,
            dataset,
            config,
            pretrained_payload,
            checkpoint_path,
            report_dir,
            class_order_path,
            manifest_path,
            split_indices,
            device,
        )
    else:
        evaluate_final(args, dataset, config, checkpoint_path, split_indices, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

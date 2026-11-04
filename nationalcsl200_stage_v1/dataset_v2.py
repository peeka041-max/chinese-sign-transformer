import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from config_v2 import DEFAULT_CONFIG


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    class_name: str
    label: int
    signer_id: Optional[str] = None


def discover_class_names(data_root):
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"V2 data directory not found: {data_root}")
    class_names = sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not class_names:
        raise ValueError(f"No class directories found under {data_root}.")
    return class_names


def _validate_class_names(class_names):
    normalized = [str(name).strip() for name in class_names]
    if not normalized or any(not name for name in normalized):
        raise ValueError("class_names must contain non-empty names.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("class_names must be unique.")
    return normalized


from feature_io import load_v2_sequence


class SignDatasetV2(Dataset):
    """V2 dataset with optional signer metadata from a CSV manifest.

    Directory mode expects ``data_root/class_name/*.npy`` and has no signer
    metadata. Manifest mode expects columns ``path`` and ``class_name`` plus an
    optional ``signer_id`` column. Relative manifest paths resolve from
    ``data_root``. Signer IDs remain attached to records for leakage-safe splits.
    """

    def __init__(
        self,
        data_root=DEFAULT_CONFIG.data_root,
        class_names=None,
        metadata_csv=None,
        config=DEFAULT_CONFIG,
    ):
        self.data_root = Path(data_root)
        self.config = config
        self.config.validate()

        if metadata_csv is None:
            if class_names is None:
                class_names = discover_class_names(self.data_root)
            self.class_names = _validate_class_names(class_names)
            self.class_to_index = {
                name: index for index, name in enumerate(self.class_names)
            }
            self.samples = self._scan_directories()
        else:
            metadata_rows = self._read_metadata(metadata_csv)
            if class_names is None:
                class_names = sorted(
                    {row["class_name"] for row in metadata_rows}
                )
            self.class_names = _validate_class_names(class_names)
            self.class_to_index = {
                name: index for index, name in enumerate(self.class_names)
            }
            self.samples = self._records_from_metadata(metadata_rows)

        if not self.samples:
            raise ValueError("No V2 .npy samples were found.")

    def _scan_directories(self):
        records = []
        for class_name in self.class_names:
            class_dir = self.data_root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            label = self.class_to_index[class_name]
            for file_path in sorted(class_dir.glob("*.npy")):
                records.append(
                    SampleRecord(
                        path=file_path,
                        class_name=class_name,
                        label=label,
                        signer_id=None,
                    )
                )
        return records

    @staticmethod
    def _read_metadata(metadata_csv):
        metadata_csv = Path(metadata_csv)
        if not metadata_csv.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

        with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"path", "class_name"}
            if reader.fieldnames is None or not required_columns.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    "Metadata CSV must contain path and class_name columns."
                )
            rows = []
            for row_number, row in enumerate(reader, start=2):
                path_value = str(row.get("path", "")).strip()
                class_name = str(row.get("class_name", "")).strip()
                signer_value = str(row.get("signer_id", "")).strip()
                if not path_value or not class_name:
                    raise ValueError(
                        f"Metadata row {row_number} has an empty path/class_name."
                    )
                rows.append(
                    {
                        "path": path_value,
                        "class_name": class_name,
                        "signer_id": signer_value or None,
                    }
                )
        if not rows:
            raise ValueError("Metadata CSV contains no samples.")
        return rows

    def _records_from_metadata(self, rows):
        records = []
        for row in rows:
            class_name = row["class_name"]
            if class_name not in self.class_to_index:
                raise ValueError(
                    f"Metadata class {class_name!r} is absent from class_names."
                )
            file_path = Path(row["path"])
            if not file_path.is_absolute():
                file_path = self.data_root / file_path
            if not file_path.is_file():
                raise FileNotFoundError(f"V2 sample not found: {file_path}")
            records.append(
                SampleRecord(
                    path=file_path,
                    class_name=class_name,
                    label=self.class_to_index[class_name],
                    signer_id=row["signer_id"],
                )
            )
        return records

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        sequence = load_v2_sequence(record.path, config=self.config)
        return (
            torch.from_numpy(sequence),
            torch.tensor(record.label, dtype=torch.long),
        )

    def get_signer_id(self, index):
        return self.samples[index].signer_id


def indices_for_signers(dataset, signer_ids: Sequence[str]):
    signer_ids = {str(value) for value in signer_ids}
    if not signer_ids:
        raise ValueError("signer_ids cannot be empty.")

    available = {
        record.signer_id
        for record in dataset.samples
        if record.signer_id is not None
    }
    unknown = signer_ids - available
    if unknown:
        raise ValueError(f"Unknown signer IDs: {sorted(unknown)}")

    indices = [
        index
        for index, record in enumerate(dataset.samples)
        if record.signer_id in signer_ids
    ]
    if not indices:
        raise ValueError("Signer selection produced no samples.")
    return indices


def make_train_val_split(
    dataset,
    validation_ratio=DEFAULT_CONFIG.validation_ratio,
    seed=DEFAULT_CONFIG.random_seed,
    val_signer_ids=None,
):
    """Use an explicit signer split when available, otherwise stratify labels."""
    if val_signer_ids:
        if any(record.signer_id is None for record in dataset.samples):
            raise ValueError(
                "Signer split requested, but at least one sample lacks signer_id."
            )
        val_indices = indices_for_signers(dataset, val_signer_ids)
        val_set = set(val_indices)
        train_indices = [
            index for index in range(len(dataset)) if index not in val_set
        ]
        if not train_indices:
            raise ValueError("Signer split produced an empty training set.")
        return train_indices, val_indices, "signer"

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1).")

    by_label = defaultdict(list)
    for index, record in enumerate(dataset.samples):
        by_label[record.label].append(index)

    random_generator = random.Random(seed)
    train_indices = []
    val_indices = []
    for label, indices in sorted(by_label.items()):
        if len(indices) < 2:
            raise ValueError(
                f"Class index {label} needs at least two samples for a split."
            )
        shuffled = indices.copy()
        random_generator.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * validation_ratio)))
        val_count = min(val_count, len(shuffled) - 1)
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    random_generator.shuffle(train_indices)
    random_generator.shuffle(val_indices)
    return train_indices, val_indices, "stratified_random"

"""Prepare the fixed 20-word realtime MVP V2 dataset.

The script selects classes only from the completed vocabulary audit, reuses
compatible raw V2 arrays by ``(signer_id, gloss_id)``, and extracts only missing
sequences from the NationalCSL-DP ZIP archives. It never trains or evaluates a
model and never modifies prior datasets or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
STAGE_ROOT = PROJECT_ROOT / "nationalcsl200_stage_v1"

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "local_data" / "NationalCSL-DP"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "data"

AUDIT_PATH = SCRIPT_DIR / "common_vocab_audit.csv"
PROPOSAL_PATH = SCRIPT_DIR / "vocabulary_proposal.json"
PROTOCOL_PATH = SCRIPT_DIR / "evaluation_protocol_draft.json"
CLASS_ORDER_PATH = SCRIPT_DIR / "class_order.csv"

PREFERRED_WORDS = (
    "我", "你", "他", "我们", "好", "感谢", "再见", "请", "是", "不是",
    "有", "可以", "要", "帮助", "什么", "谁", "喝", "家", "学校", "朋友",
)
BACKUP_WORDS = (
    "厕所", "今天", "明天", "现在", "老师", "左", "右", "停止", "医院", "医生",
)

TRAIN_SIGNERS = tuple(f"Participant_{index:02d}" for index in range(1, 8))
VAL_SIGNERS = ("Participant_08",)
TEST_SIGNERS = ("Participant_09", "Participant_10")
ALL_SIGNERS = TRAIN_SIGNERS + VAL_SIGNERS + TEST_SIGNERS

FEATURE_SHAPE = (30, 266)
GOOD_THRESHOLD = 0.80
WARNING_THRESHOLD = 0.50
VOCABULARY_STATUS = "日常词实验v1词表，尚不代表全组最终统一词表"

CLASS_COLUMNS = (
    "class_id",
    "gloss_id",
    "dataset_gloss",
    "display_word",
    "match_type",
    "vocabulary_status",
)

MANIFEST_COLUMNS = (
    "sample_id",
    "path",
    "feature_path",
    "class_id",
    "gloss_id",
    "dataset_gloss",
    "display_word",
    "match_type",
    "signer_id",
    "split",
    "evaluation_role",
    "source_archive",
    "source_sequence",
    "original_frames",
    "left_detection_ratio",
    "right_detection_ratio",
    "any_hand_detection_ratio",
    "quality_status",
    "preparation_method",
    "reused_from",
    "reuse_source_feature_path",
    "processing_status",
    "feature_sha256",
    "error_message",
)

SOURCE_SPECS = (
    {
        "name": "nationalcsl20_v2",
        "root": PROJECT_ROOT / "nationalcsl20_v2",
        "manifest": PROJECT_ROOT / "nationalcsl20_v2" / "manifest.csv",
    },
    {
        "name": "nationalcsl200_v2",
        "root": STAGE_ROOT / "nationalcsl200_v2",
        "manifest": STAGE_ROOT / "nationalcsl200_v2" / "manifest.csv",
    },
    {
        "name": "domain_mvp_v1",
        "root": PROJECT_ROOT / "domain_mvp_v1" / "domain20_v2",
        "manifest": PROJECT_ROOT / "domain_mvp_v1" / "domain20_v2" / "manifest.csv",
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the fixed realtime MVP V2 20-word dataset."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smoke-count", type=int, default=5)
    return parser.parse_args()


def normalize_gloss_id(value):
    text = str(value).strip()
    if not re.fullmatch(r"\d{1,4}", text):
        raise ValueError(f"Invalid textual gloss_id: {text!r}")
    return text.zfill(4)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_required_metadata():
    for path in (AUDIT_PATH, PROPOSAL_PATH, PROTOCOL_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required audit input is missing: {path}")

    proposal = json.loads(PROPOSAL_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if proposal.get("proposal_status") != "provisional":
        raise ValueError("vocabulary_proposal.json must remain provisional.")
    if protocol.get("task") != "isolated_sign_recognition_v1":
        raise ValueError("Unexpected task in evaluation_protocol_draft.json.")
    expected_split = {
        "train": list(TRAIN_SIGNERS),
        "val": list(VAL_SIGNERS),
        "stage_test": list(TEST_SIGNERS),
    }
    if protocol.get("nationalcsl_split") != expected_split:
        raise ValueError("Evaluation protocol signer split does not match this run.")
    if protocol.get("stage_test_is_fully_blind") is not False:
        raise ValueError("Protocol must identify stage_test as not fully blind.")
    return proposal, protocol


def audit_row_is_covered(row):
    if str(row.get("available_signers", "")).strip() != "10":
        return False
    for signer in ALL_SIGNERS:
        if str(row.get(f"{signer}_front_sequence_count", "")).strip() not in {"", "1"}:
            return False
        frame_count = str(row.get(f"{signer}_front_frame_count", "")).strip()
        if frame_count and int(frame_count) <= 0:
            return False
    return True


def select_classes(
    audit_path,
    preferred_words=PREFERRED_WORDS,
    backup_words=BACKUP_WORDS,
    required_count=20,
):
    audit_rows = read_csv(audit_path)
    by_candidate = defaultdict(list)
    for row in audit_rows:
        by_candidate[str(row.get("candidate_word", "")).strip()].append(row)

    selected = []
    rejected = {}

    def eligible(word):
        exact_rows = [
            row for row in by_candidate.get(word, [])
            if str(row.get("match_type", "")).strip() == "exact"
        ]
        if len(exact_rows) != 1:
            return None, f"exact匹配数量为{len(exact_rows)}，要求唯一"
        row = exact_rows[0]
        if not audit_row_is_covered(row):
            return None, "未达到10/10手语者front覆盖"
        gloss_id = normalize_gloss_id(row.get("gloss_id", ""))
        dataset_gloss = str(row.get("matched_gloss", "")).strip()
        if dataset_gloss != word:
            return None, f"exact标签不一致: {dataset_gloss!r}"
        return {
            "gloss_id": gloss_id,
            "dataset_gloss": dataset_gloss,
            "display_word": word,
            "match_type": "exact",
            "vocabulary_status": VOCABULARY_STATUS,
        }, ""

    for word in preferred_words:
        item, reason = eligible(word)
        if item is None:
            rejected[word] = reason
        else:
            selected.append(item)

    if len(selected) < required_count:
        for word in backup_words:
            if len(selected) >= required_count:
                break
            item, reason = eligible(word)
            if item is None:
                rejected[word] = reason
            elif item["gloss_id"] not in {row["gloss_id"] for row in selected}:
                selected.append(item)

    if len(selected) < required_count:
        missing_count = required_count - len(selected)
        details = "; ".join(f"{word}: {reason}" for word, reason in rejected.items())
        raise RuntimeError(
            f"仅有{len(selected)}个合格词，仍缺{missing_count}个。{details}"
        )
    if len(selected) > required_count:
        selected = selected[:required_count]
    if len({row["gloss_id"] for row in selected}) != required_count:
        raise RuntimeError("Selected gloss IDs are not unique.")

    for class_id, row in enumerate(selected):
        row["class_id"] = class_id
    return selected, rejected


def split_for_signer(signer_id):
    if signer_id in TRAIN_SIGNERS:
        return "train"
    if signer_id in VAL_SIGNERS:
        return "val"
    if signer_id in TEST_SIGNERS:
        return "test"
    raise ValueError(f"Unknown signer: {signer_id}")


def build_planned_rows(classes):
    rows = []
    for signer_id in ALL_SIGNERS:
        split = split_for_signer(signer_id)
        for item in classes:
            gloss_id = normalize_gloss_id(item["gloss_id"])
            sample_id = f"{signer_id}_{gloss_id}"
            feature_path = f"features/{gloss_id}/{sample_id}.npy"
            rows.append(
                {
                    "sample_id": sample_id,
                    "path": feature_path,
                    "feature_path": feature_path,
                    "class_id": item["class_id"],
                    "gloss_id": gloss_id,
                    "dataset_gloss": item["dataset_gloss"],
                    "display_word": item["display_word"],
                    "match_type": item["match_type"],
                    "signer_id": signer_id,
                    "split": split,
                    "evaluation_role": (
                        "阶段评测集" if split == "test" else split
                    ),
                    "source_archive": f"{signer_id}.zip",
                    "source_sequence": f"{signer_id}/front/{gloss_id}/",
                    "original_frames": "",
                    "left_detection_ratio": "",
                    "right_detection_ratio": "",
                    "any_hand_detection_ratio": "",
                    "quality_status": "",
                    "preparation_method": "",
                    "reused_from": "",
                    "reuse_source_feature_path": "",
                    "processing_status": "planned",
                    "feature_sha256": "",
                    "error_message": "",
                }
            )
    return rows


def validate_feature_array(array):
    if not isinstance(array, np.ndarray):
        raise ValueError("Feature must be a NumPy array.")
    if array.shape != FEATURE_SHAPE:
        raise ValueError(f"Expected feature shape {FEATURE_SHAPE}, got {array.shape}.")
    if array.dtype != np.float32:
        raise ValueError(f"Expected float32 features, got {array.dtype}.")
    if not np.isfinite(array).all():
        raise ValueError("Feature contains NaN or Inf.")
    presence = array[:, 264:266]
    if not np.all(np.logical_or(presence == 0.0, presence == 1.0)):
        raise ValueError(
            "Presence columns are not binary; feature may be standardized or incompatible."
        )


def safe_source_feature_path(source_root, relative_path):
    source_root = Path(source_root).resolve()
    candidate = (source_root / str(relative_path)).resolve()
    try:
        candidate.relative_to(source_root)
    except ValueError as error:
        raise ValueError(f"Feature path escapes source root: {relative_path}") from error
    return candidate


def parse_ratio(row, field):
    value = float(str(row.get(field, "")).strip())
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Invalid {field}: {value}")
    return value


def quality_status(any_hand_ratio):
    if any_hand_ratio >= GOOD_THRESHOLD:
        return "good"
    if any_hand_ratio >= WARNING_THRESHOLD:
        return "warning"
    return "bad"


def load_reuse_candidates(selected_ids):
    candidates = defaultdict(list)
    queried = {}
    for priority, spec in enumerate(SOURCE_SPECS):
        manifest_path = spec["manifest"]
        if not manifest_path.is_file():
            queried[spec["name"]] = {"manifest_found": False, "matching_rows": 0}
            continue
        rows = read_csv(manifest_path)
        matching = 0
        for row in rows:
            raw_id = str(row.get("gloss_id", "")).strip()
            if not raw_id:
                continue
            gloss_id = normalize_gloss_id(raw_id)
            signer_id = str(row.get("signer_id", "")).strip()
            if gloss_id not in selected_ids or signer_id not in ALL_SIGNERS:
                continue
            if str(row.get("processing_status", "success")).strip() != "success":
                continue
            feature_path = str(
                row.get("feature_path") or row.get("path") or ""
            ).strip()
            if not feature_path:
                continue
            matching += 1
            candidates[(signer_id, gloss_id)].append(
                {
                    "priority": priority,
                    "source_name": spec["name"],
                    "source_root": spec["root"],
                    "source_manifest": manifest_path,
                    "feature_path": feature_path,
                    "manifest_row": row,
                }
            )
        queried[spec["name"]] = {
            "manifest_found": True,
            "manifest_rows": len(rows),
            "matching_rows": matching,
        }
    return candidates, queried


def validate_reuse_candidate(candidate):
    row = candidate["manifest_row"]
    path = safe_source_feature_path(candidate["source_root"], candidate["feature_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Reusable feature not found: {path}")
    array = np.load(path, allow_pickle=False)
    validate_feature_array(array)
    actual_hash = file_sha256(path)
    recorded_hash = str(row.get("feature_sha256", "")).strip().lower()
    if recorded_hash and actual_hash != recorded_hash:
        raise ValueError(f"Feature hash mismatch: {path}")
    ratios = (
        parse_ratio(row, "left_detection_ratio"),
        parse_ratio(row, "right_detection_ratio"),
        parse_ratio(row, "any_hand_detection_ratio"),
    )
    original_frames = int(str(row.get("original_frames", "")).strip())
    if original_frames <= 0:
        raise ValueError(f"Invalid original_frames in {candidate['source_manifest']}")
    return {
        **candidate,
        "source_path": path,
        "array": array,
        "feature_sha256": actual_hash,
        "ratios": ratios,
        "original_frames": original_frames,
        "quality_status": quality_status(ratios[2]),
    }


def resolve_reuse_index(raw_candidates):
    resolved = {}
    diagnostics = {
        "keys_with_candidates": len(raw_candidates),
        "valid_candidates": 0,
        "invalid_candidates": 0,
        "conflicting_keys_reextracted": 0,
        "invalid_reasons": [],
    }
    for key, candidates in raw_candidates.items():
        valid = []
        for candidate in sorted(candidates, key=lambda item: item["priority"]):
            try:
                valid.append(validate_reuse_candidate(candidate))
                diagnostics["valid_candidates"] += 1
            except Exception as error:
                diagnostics["invalid_candidates"] += 1
                diagnostics["invalid_reasons"].append(
                    {
                        "sample_key": list(key),
                        "source": candidate["source_name"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        if not valid:
            continue
        reference = valid[0]["array"]
        if any(not np.array_equal(reference, item["array"]) for item in valid[1:]):
            diagnostics["conflicting_keys_reextracted"] += 1
            continue
        chosen = valid[0].copy()
        chosen.pop("array")
        resolved[key] = chosen
    return resolved, diagnostics


def source_compatibility_report(model_path):
    root_features = PROJECT_ROOT / "features_v2.py"
    stage_features = STAGE_ROOT / "features_v2.py"
    root_config = PROJECT_ROOT / "config_v2.py"
    stage_config = STAGE_ROOT / "config_v2.py"
    paths = (root_features, stage_features, root_config, stage_config, model_path)
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Compatibility input missing: {path}")
    report = {
        "feature_definition_sha256": {
            "root": file_sha256(root_features),
            "stage": file_sha256(stage_features),
        },
        "config_sha256": {
            "root": file_sha256(root_config),
            "stage": file_sha256(stage_config),
        },
        "hand_landmarker_model_sha256": file_sha256(model_path),
    }
    report["feature_definition_match"] = (
        report["feature_definition_sha256"]["root"]
        == report["feature_definition_sha256"]["stage"]
    )
    report["config_match"] = (
        report["config_sha256"]["root"] == report["config_sha256"]["stage"]
    )
    if not report["feature_definition_match"] or not report["config_match"]:
        raise RuntimeError("Root and stage V2 feature contracts differ.")

    stage_protocol_path = STAGE_ROOT / "nationalcsl200_v2" / "protocol.json"
    if stage_protocol_path.is_file():
        stage_protocol = json.loads(stage_protocol_path.read_text(encoding="utf-8"))
        recorded = (
            stage_protocol.get("definition", {})
            .get("features", {})
            .get("model_sha256", "")
        )
        report["stage_protocol_model_sha256"] = recorded
        report["stage_protocol_model_match"] = recorded == report["hand_landmarker_model_sha256"]
        if not report["stage_protocol_model_match"]:
            raise RuntimeError("Current hand landmarker model differs from stage protocol.")
    return report


def index_archive(archive, signer_id, selected_ids):
    pattern = re.compile(
        rf"^{re.escape(signer_id)}/front/(\d{{4}})/(\d+)\.(jpg|jpeg|png)$",
        re.IGNORECASE,
    )
    indexed = {gloss_id: [] for gloss_id in selected_ids}
    for member in archive.infolist():
        normalized = member.filename.replace("\\", "/")
        match = pattern.fullmatch(normalized)
        if match and match.group(1) in indexed:
            indexed[match.group(1)].append((int(match.group(2)), member.filename))

    output = {}
    for gloss_id, entries in indexed.items():
        ordered = sorted(entries, key=lambda pair: pair[0])
        if not ordered:
            raise FileNotFoundError(f"No front frames for {signer_id}/front/{gloss_id}/")
        numbers = [number for number, _ in ordered]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f"Non-contiguous frames for {signer_id}/front/{gloss_id}/")
        output[gloss_id] = [name for _, name in ordered]
    return output


def load_existing_extractor():
    stage_path = str(STAGE_ROOT)
    if stage_path not in sys.path:
        sys.path.insert(0, stage_path)
    from prep_support import extract_sequence_features, save_feature_array

    return extract_sequence_features, save_feature_array


def populate_reused_row(row, chosen, staging_root):
    destination = staging_root / row["feature_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(chosen["source_path"], destination)
    array = np.load(destination, allow_pickle=False)
    validate_feature_array(array)
    copied_hash = file_sha256(destination)
    if copied_hash != chosen["feature_sha256"]:
        raise RuntimeError(f"Copied feature hash changed for {row['sample_id']}")
    ratios = chosen["ratios"]
    row.update(
        original_frames=chosen["original_frames"],
        left_detection_ratio=f"{ratios[0]:.6f}",
        right_detection_ratio=f"{ratios[1]:.6f}",
        any_hand_detection_ratio=f"{ratios[2]:.6f}",
        quality_status=chosen["quality_status"],
        preparation_method="reused",
        reused_from=chosen["source_name"],
        reuse_source_feature_path=(
            chosen["source_path"].relative_to(PROJECT_ROOT).as_posix()
        ),
        processing_status="success",
        feature_sha256=copied_hash,
        error_message="",
    )


def populate_extracted_row(
    row,
    archive,
    entries,
    model_path,
    staging_root,
    extract_sequence_features,
    save_feature_array,
):
    features, ratios = extract_sequence_features(
        archive,
        entries=entries,
        model_path=model_path,
    )
    validate_feature_array(features)
    destination = staging_root / row["feature_path"]
    save_feature_array(destination, features, overwrite=False)
    saved = np.load(destination, allow_pickle=False)
    validate_feature_array(saved)
    row.update(
        original_frames=len(entries),
        left_detection_ratio=f"{ratios[0]:.6f}",
        right_detection_ratio=f"{ratios[1]:.6f}",
        any_hand_detection_ratio=f"{ratios[2]:.6f}",
        quality_status=quality_status(ratios[2]),
        preparation_method="extracted",
        reused_from="",
        reuse_source_feature_path="",
        processing_status="success",
        feature_sha256=file_sha256(destination),
        error_message="",
    )


def mark_failure(row, error):
    row.update(
        quality_status="failed",
        processing_status="failed",
        error_message=f"{type(error).__name__}: {error}".replace("\r", " ").replace("\n", " "),
        feature_sha256="",
    )


def validate_rows_and_features(rows, classes, data_root, expected_count=None):
    if expected_count is not None and len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} rows, found {len(rows)}.")
    class_by_id = {int(item["class_id"]): item for item in classes}
    sample_ids = set()
    feature_paths = set()
    for row in rows:
        if row["processing_status"] != "success":
            raise RuntimeError(f"Unsuccessful sample: {row['sample_id']}: {row['error_message']}")
        if row["sample_id"] in sample_ids:
            raise RuntimeError(f"Duplicate sample_id: {row['sample_id']}")
        if row["feature_path"] in feature_paths:
            raise RuntimeError(f"Duplicate feature_path: {row['feature_path']}")
        sample_ids.add(row["sample_id"])
        feature_paths.add(row["feature_path"])

        class_id = int(row["class_id"])
        expected_class = class_by_id[class_id]
        for field in ("gloss_id", "dataset_gloss", "display_word", "match_type"):
            if str(row[field]) != str(expected_class[field]):
                raise RuntimeError(
                    f"Label mapping mismatch for {row['sample_id']} field {field}."
                )
        if row["split"] != split_for_signer(row["signer_id"]):
            raise RuntimeError(f"Signer split mismatch for {row['sample_id']}")

        feature_path = data_root / row["feature_path"]
        array = np.load(feature_path, allow_pickle=False)
        validate_feature_array(array)
        if file_sha256(feature_path) != row["feature_sha256"]:
            raise RuntimeError(f"Feature hash mismatch after save: {feature_path}")


def validate_completed_dataset(rows, classes, data_root):
    validate_rows_and_features(rows, classes, data_root, expected_count=200)
    split_counts = Counter(row["split"] for row in rows)
    if split_counts != Counter({"train": 140, "val": 20, "test": 40}):
        raise RuntimeError(f"Unexpected split counts: {dict(split_counts)}")
    per_class = Counter(str(row["class_id"]) for row in rows)
    if set(per_class.values()) != {10} or len(per_class) != 20:
        raise RuntimeError(f"Unexpected per-class counts: {dict(per_class)}")
    per_signer = Counter(row["signer_id"] for row in rows)
    if set(per_signer.values()) != {20} or len(per_signer) != 10:
        raise RuntimeError(f"Unexpected per-signer counts: {dict(per_signer)}")
    signer_splits = defaultdict(set)
    for row in rows:
        signer_splits[row["signer_id"]].add(row["split"])
    if any(len(splits) != 1 for splits in signer_splits.values()):
        raise RuntimeError("At least one signer appears in multiple splits.")


def prepare_output_paths(output_root):
    output_root = Path(output_root).resolve()
    if output_root != DEFAULT_OUTPUT_ROOT.resolve():
        raise ValueError(
            f"This fixed run must write to {DEFAULT_OUTPUT_ROOT}; got {output_root}."
        )
    staging_root = output_root.with_name(output_root.name + ".__preparing__")
    if output_root.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_root}")
    if staging_root.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_root}")
    staging_root.mkdir(parents=True)
    return output_root, staging_root


def build_summary(
    rows,
    classes,
    elapsed_seconds,
    queried_sources,
    reuse_diagnostics,
    compatibility,
    archive_open_counts,
    smoke_count,
):
    successful = [row for row in rows if row["processing_status"] == "success"]
    failed = [row for row in rows if row["processing_status"] == "failed"]
    method_counts = Counter(row["preparation_method"] for row in successful)
    reuse_by_source = Counter(row["reused_from"] for row in successful if row["reused_from"])
    split_counts = Counter(row["split"] for row in successful)
    quality_counts = Counter(row["quality_status"] for row in successful)
    means = {}
    for field in (
        "left_detection_ratio",
        "right_detection_ratio",
        "any_hand_detection_ratio",
    ):
        values = [float(row[field]) for row in successful]
        means[field] = float(np.mean(values)) if values else None
    return {
        "dataset": "realtime_mvp_v2_daily20",
        "vocabulary_status": VOCABULARY_STATUS,
        "classes": classes,
        "class_count": len(classes),
        "planned_total": 200,
        "successful": len(successful),
        "failed": len(failed),
        "reused": method_counts["reused"],
        "newly_extracted": method_counts["extracted"],
        "reused_by_source": dict(reuse_by_source),
        "splits": dict(split_counts),
        "split_note": {
            "test": "阶段评测集；Participant_09/10并非全新盲测",
        },
        "quality": dict(quality_counts),
        "mean_detection_ratio": means,
        "feature_contract": {
            "shape": list(FEATURE_SHAPE),
            "dtype": "float32",
            "standardization": "none; raw V2 features",
            "finite_required": True,
            "presence_columns": [264, 265],
        },
        "smoke_gate": {
            "requested_train_samples": smoke_count,
            "status": "passed",
        },
        "validation": {
            "total_200": len(successful) == 200 and not failed,
            "per_class_10": True,
            "per_signer_20": True,
            "split_140_20_40": dict(split_counts) == {"train": 140, "val": 20, "test": 40},
            "no_duplicate_samples": True,
            "no_signer_cross_split": True,
            "all_features_valid": True,
            "labels_match_class_order": True,
        },
        "source_manifests_queried": queried_sources,
        "reuse_diagnostics": reuse_diagnostics,
        "compatibility": compatibility,
        "zip_policy": {
            "whole_archive_extracted": False,
            "central_directory_reads_per_archive": archive_open_counts,
            "maximum_per_archive": max(archive_open_counts.values(), default=0),
        },
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def main():
    args = parse_args()
    if args.smoke_count != 5:
        raise ValueError("This fixed preparation run requires --smoke-count 5.")
    dataset_root = args.dataset_root.resolve()
    model_path = args.model.resolve()
    pics_root = dataset_root / "Pics"
    if not dataset_root.is_dir() or not pics_root.is_dir():
        raise FileNotFoundError(f"Invalid NationalCSL-DP root: {dataset_root}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Hand landmarker model not found: {model_path}")

    _, protocol = load_required_metadata()
    classes, rejected = select_classes(AUDIT_PATH)
    if len(classes) != 20:
        raise RuntimeError("Class selection did not produce exactly 20 classes.")
    planned_rows = build_planned_rows(classes)
    selected_ids = {item["gloss_id"] for item in classes}
    compatibility = source_compatibility_report(model_path)
    raw_reuse, queried_sources = load_reuse_candidates(selected_ids)
    reuse_index, reuse_diagnostics = resolve_reuse_index(raw_reuse)
    output_root, staging_root = prepare_output_paths(args.output)
    extract_sequence_features, save_feature_array = load_existing_extractor()

    print("Fixed class order:", flush=True)
    for item in classes:
        print(
            f"  {item['class_id']:02d} {item['gloss_id']} {item['display_word']}",
            flush=True,
        )
    print(f"Eligible preferred words rejected: {rejected or 'none'}", flush=True)
    print(f"Reusable sample keys: {len(reuse_index)}", flush=True)

    started = time.perf_counter()
    completed_rows = []
    archive_open_counts = Counter()
    rows_by_signer = defaultdict(list)
    for row in planned_rows:
        rows_by_signer[row["signer_id"]].append(row)

    for signer_id in ALL_SIGNERS:
        archive_path = pics_root / f"{signer_id}.zip"
        if not archive_path.is_file():
            raise FileNotFoundError(f"Participant ZIP missing: {archive_path}")
        print(f"Opening {archive_path.name} once", flush=True)
        archive_open_counts[archive_path.name] += 1
        if archive_open_counts[archive_path.name] > 1:
            raise RuntimeError(f"Archive opened more than once: {archive_path.name}")
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive_index = index_archive(archive, signer_id, selected_ids)
            for row in rows_by_signer[signer_id]:
                key = (signer_id, row["gloss_id"])
                try:
                    if key in reuse_index:
                        populate_reused_row(row, reuse_index[key], staging_root)
                    else:
                        populate_extracted_row(
                            row,
                            archive=archive,
                            entries=archive_index[row["gloss_id"]],
                            model_path=model_path,
                            staging_root=staging_root,
                            extract_sequence_features=extract_sequence_features,
                            save_feature_array=save_feature_array,
                        )
                    print(
                        f"[{len(completed_rows) + 1:03d}/200] {row['sample_id']} "
                        f"{row['preparation_method']} any={row['any_hand_detection_ratio']} "
                        f"quality={row['quality_status']}",
                        flush=True,
                    )
                except Exception as error:
                    mark_failure(row, error)
                    print(
                        f"[{len(completed_rows) + 1:03d}/200] {row['sample_id']} "
                        f"FAILED: {row['error_message']}",
                        flush=True,
                    )
                completed_rows.append(row)

                if len(completed_rows) == args.smoke_count:
                    smoke_rows = completed_rows[: args.smoke_count]
                    if any(row["split"] != "train" for row in smoke_rows):
                        raise RuntimeError("Smoke gate must use five training samples.")
                    validate_rows_and_features(
                        smoke_rows,
                        classes,
                        staging_root,
                        expected_count=args.smoke_count,
                    )
                    print(
                        "SMOKE GATE PASS: 5 train samples, shape=(30, 266), "
                        "dtype=float32, finite values, labels mapped.",
                        flush=True,
                    )

    write_csv(staging_root / "manifest.csv", MANIFEST_COLUMNS, completed_rows)
    validation_error = None
    try:
        validate_completed_dataset(completed_rows, classes, staging_root)
    except Exception as error:
        validation_error = error

    summary = build_summary(
        completed_rows,
        classes,
        time.perf_counter() - started,
        queried_sources,
        reuse_diagnostics,
        compatibility,
        dict(archive_open_counts),
        args.smoke_count,
    )
    if validation_error is not None:
        summary["validation_error"] = f"{type(validation_error).__name__}: {validation_error}"
        summary["smoke_gate"]["status"] = (
            "passed" if len(completed_rows) >= args.smoke_count else "not_reached"
        )
        write_json(staging_root / "preparation_summary.json", summary)
        raise validation_error

    write_json(staging_root / "preparation_summary.json", summary)
    staging_root.replace(output_root)
    write_csv(CLASS_ORDER_PATH, CLASS_COLUMNS, classes)

    print("FULL VALIDATION PASS", flush=True)
    print(
        f"Prepared=200 reused={summary['reused']} "
        f"newly_extracted={summary['newly_extracted']} failed={summary['failed']}",
        flush=True,
    )
    print(f"Splits={summary['splits']} quality={summary['quality']}", flush=True)
    print(f"Manifest={output_root / 'manifest.csv'}", flush=True)
    print(f"Summary={output_root / 'preparation_summary.json'}", flush=True)
    print(f"Class order={CLASS_ORDER_PATH}", flush=True)
    print(
        "Test split is a stage evaluation set, not a fresh blind test.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

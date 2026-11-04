"""Prepare a frozen, team-defined 200-class front-view NationalCSL-DP subset.

Relative output/model paths are relative to THIS script, not the shell CWD.
Existing successful arrays are verified and reused automatically.
"""
import argparse
from collections import Counter
from importlib import metadata
import json
from pathlib import Path
import re
import time
import zipfile

from protocol import (
    ALL_SIGNERS, TRAIN_SIGNERS, VAL_SIGNERS, TEST_SIGNERS, SCRIPT_DIR,
    CLASS_COLUMNS, MANIFEST_COLUMNS, canonical_hash, expected_rows, file_hash,
    read_csv, read_protocol, resolve_path, select_classes, validate_manifest_rows,
    write_csv, write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("nationalcsl200_v2"))
    parser.add_argument("--model", type=Path, default=Path("../models/hand_landmarker.task"))
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--class-seed", type=int, default=42)
    parser.add_argument("--class-list", type=Path, help="Optional CSV with gloss_id, in agreed class order.")
    parser.add_argument("--plan", action="store_true", help="Freeze word list and check ZIP indexes; no decoding or training.")
    parser.add_argument("--max-sequences", type=int, help="Smoke test: prepare only the first N planned sequences.")
    return parser.parse_args()


def index_archive(archive, signer, classes):
    selected = {row["gloss_id"] for row in classes}
    indexed = {gid: [] for gid in selected}
    pattern = re.compile(re.escape(signer) + r"/front/(\d{4})/(\d+)\.(jpg|jpeg|png)$", re.I)
    for item in archive.infolist():
        match = pattern.fullmatch(item.filename)
        if match and match[1] in selected:
            indexed[match[1]].append((int(match[2]), item))
    output = {}
    checksums = []
    for gid in sorted(selected):
        items = sorted(indexed[gid], key=lambda pair: pair[0])
        numbers = [number for number, _ in items]
        if not items or numbers != list(range(1, len(items) + 1)):
            raise ValueError(f"Missing, duplicate or non-contiguous frames: {signer}/front/{gid}")
        output[gid] = [item.filename for _, item in items]
        checksums.extend((item.filename, item.CRC, item.file_size) for _, item in items)
    return output, canonical_hash(checksums)


def runtime_versions():
    result = {}
    for name in ("numpy", "mediapipe", "opencv-python", "opencv-contrib-python"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def freeze_plan(args):
    root, output, model = map(resolve_path, (args.dataset_root, args.output, args.model))
    if not model.is_file():
        raise FileNotFoundError(f"Hand landmarker model not found: {model}")
    if args.max_sequences is not None and args.max_sequences <= 0:
        raise ValueError("--max-sequences must be positive.")
    class_list = resolve_path(args.class_list) if args.class_list else None
    classes = select_classes(root / "gloss.csv", args.num_classes, args.class_seed, class_list)
    indexes, archives = {}, {}
    for signer in ALL_SIGNERS:
        archive_path = root / "Pics" / f"{signer}.zip"
        with zipfile.ZipFile(archive_path) as archive:
            indexes[signer], archives[signer] = index_archive(archive, signer, classes)
        print(f"Checked {signer}.zip: {len(classes)} front-view sequences", flush=True)
    definition = {
        "dataset": "NationalCSL-DP", "subset": "team-defined (not official NationalCSL200)",
        "classes": classes, "view": "front",
        "selection": {"method": "supplied-list" if class_list else "sha256-rank-v1",
                      "seed": None if class_list else args.class_seed},
        "signers": {"train": list(TRAIN_SIGNERS), "val": list(VAL_SIGNERS), "test": list(TEST_SIGNERS)},
        "features": {"shape": [30, 266], "version": "V2-all-frames-time-nearest",
                     "model_sha256": file_hash(model), "runtime_versions": runtime_versions(),
                     "source_sha256": {name: file_hash(SCRIPT_DIR / name) for name in
                                       ("prepare_nationalcsl200.py", "prep_support.py", "features_v2.py", "config_v2.py")}},
        "selected_archive_indexes": archives,
        "quality_policy": "Report all detection rates; never discard by quality or accuracy.",
        "test_history": "Participant_09/10 were used in a prior 20-class experiment. This is a stage benchmark, not a fresh blind test.",
    }
    payload = {"schema": "nationalcsl-custom-stage-v1", "definition": definition,
               "protocol_sha256": canonical_hash(definition)}
    if (output / "protocol.json").exists():
        existing = read_protocol(output)
        if existing != payload:
            raise ValueError("Frozen protocol differs (classes/model/runtime/source/index). Use a NEW --output directory.")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to use an unrecognized non-empty output directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / "class_order.csv", CLASS_COLUMNS, classes)
        write_json(output / "protocol.json", payload)
    planned = expected_rows(classes)
    for row in planned:
        row["original_frames"] = len(indexes[row["signer_id"]][row["gloss_id"]])
    manifest = output / "manifest.csv"
    if manifest.is_file():
        previous = read_csv(manifest, MANIFEST_COLUMNS)
        validate_manifest_rows(previous, classes, require_complete=False)
        old_by_id = {row["sample_id"]: row for row in previous}
        planned = [old_by_id.get(row["sample_id"], row) for row in planned]
    write_csv(manifest, MANIFEST_COLUMNS, planned)
    return root, output, model, classes, indexes, planned


def reusable_feature(row, output):
    import numpy as np
    if row["processing_status"] != "success":
        return False
    path = output / row["path"]
    try:
        if file_hash(path) != row["feature_sha256"]:
            return False
        array = np.load(path, allow_pickle=False)
        return array.shape == (30, 266) and array.dtype == np.float32 and bool(np.isfinite(array).all())
    except (OSError, ValueError, EOFError):
        return False


def prepare(args, extractor=None):
    started = time.perf_counter()
    root, output, model, classes, indexes, rows = freeze_plan(args)
    counts = {"train": len(classes) * 7, "val": len(classes), "test": len(classes) * 2}
    print(f"Classes: {len(classes)} (team-defined subset)")
    print(f"Planned sequences: {len(rows)}; splits={counts}")
    print(f"Class order: {output / 'class_order.csv'}")
    if args.plan:
        print("Plan frozen. No images decoded; no model trained.")
        return 0
    if extractor is None:
        # Lazy import keeps --help/--plan usable without loading MediaPipe.
        from prep_support import extract_sequence_features
        extractor = extract_sequence_features
    from prep_support import save_feature_array, quality_status
    limit = len(rows) if args.max_sequences is None else min(args.max_sequences, len(rows))
    selected = rows[:limit]
    reused = new = failed = 0
    for signer in ALL_SIGNERS:
        signer_rows = [row for row in selected if row["signer_id"] == signer]
        if not signer_rows:
            continue
        with zipfile.ZipFile(root / "Pics" / f"{signer}.zip") as archive:
            for row in signer_rows:
                if reusable_feature(row, output):
                    reused += 1
                    print(f"Reused {row['sample_id']}", flush=True)
                    continue
                try:
                    entries = indexes[signer][row["gloss_id"]]
                    sequence, ratios = extractor(archive, entries, model)
                    path = output / row["path"]
                    save_feature_array(path, sequence, overwrite=True)
                    row.update(processing_status="success", error_message="",
                               feature_sha256=file_hash(path), original_frames=len(entries),
                               left_detection_ratio=f"{ratios[0]:.6f}", right_detection_ratio=f"{ratios[1]:.6f}",
                               any_hand_detection_ratio=f"{ratios[2]:.6f}", quality_status=quality_status(ratios[2], .8, .5))
                    new += 1
                    print(f"[{new + reused + failed}/{limit}] {row['sample_id']} {row['chinese_gloss']}: "
                          f"shape={sequence.shape}, any={ratios[2]:.2%}, quality={row['quality_status']}", flush=True)
                except Exception as error:
                    failed += 1
                    row.update(processing_status="failed", error_message=f"{type(error).__name__}: {error}", feature_sha256="")
                    print(f"FAILED {row['sample_id']}: {error}", flush=True)
                finally:
                    # Atomic progress checkpoint after EVERY sequence.
                    write_csv(output / "manifest.csv", MANIFEST_COLUMNS, rows)
    success = sum(row["processing_status"] == "success" for row in selected)
    complete = sum(row["processing_status"] == "success" for row in rows)
    if args.max_sequences is None and failed == 0:
        validate_manifest_rows(rows, classes)
    summary = {"classes": len(classes), "requested_this_run": limit, "successful_this_run": success,
               "new": new, "reused": reused, "failed_this_run": failed,
               "completed_total": complete, "planned_total": len(rows),
               "completed_splits": dict(Counter(row["split"] for row in rows if row["processing_status"] == "success")),
               "quality": dict(Counter(row["quality_status"] for row in rows if row["processing_status"] == "success")),
               "elapsed_seconds": round(time.perf_counter() - started, 2)}
    write_json(output / "preparation_summary.json", summary)
    print("=" * 70)
    print(f"Requested: {limit}\nSuccessful: {success}\nFailed: {failed}\nReused: {reused}")
    print(f"Completed total: {complete}/{len(rows)}")
    print(f"Splits: {summary['completed_splits']}")
    print("Feature shape: (30, 266)")
    print(f"Elapsed seconds: {summary['elapsed_seconds']}\nManifest: {output / 'manifest.csv'}")
    print("No ZIP archives were extracted. Bad-quality samples were retained.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(prepare(parse_args()))

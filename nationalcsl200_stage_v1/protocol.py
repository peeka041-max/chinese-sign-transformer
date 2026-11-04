"""Frozen word list and signer split; no ML dependencies or quality selection."""
import csv
import hashlib
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SIGNERS = tuple(f"Participant_{i:02d}" for i in range(1, 8))
VAL_SIGNERS = ("Participant_08",)
TEST_SIGNERS = ("Participant_09", "Participant_10")
ALL_SIGNERS = TRAIN_SIGNERS + VAL_SIGNERS + TEST_SIGNERS
CLASS_COLUMNS = ("class_id", "gloss_id", "class_name", "chinese_gloss")
MANIFEST_COLUMNS = (
    "sample_id", "path", "feature_path", *CLASS_COLUMNS, "signer_id", "split",
    "source_archive", "source_sequence", "original_frames",
    "left_detection_ratio", "right_detection_ratio", "any_hand_detection_ratio",
    "quality_status", "processing_status", "feature_sha256", "error_message",
)


def resolve_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def split_for_signer(signer):
    if signer in TRAIN_SIGNERS:
        return "train"
    if signer in VAL_SIGNERS:
        return "val"
    if signer in TEST_SIGNERS:
        return "test"
    raise ValueError(f"Unknown signer: {signer}")


def canonical_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_csv(path, required):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required CSV fields {required}: {path}")
        return list(reader)


def normalize_gloss_id(value):
    value = str(value).strip()
    if not value.isascii() or not value.isdigit() or len(value) > 4:
        raise ValueError(f"Expected a numeric gloss ID with at most four digits, got {value!r}")
    return value.zfill(4)


def select_classes(gloss_path, count=200, seed=42, class_list=None):
    glossary = {}
    for row in read_csv(gloss_path, ("#ID", "Chinese Sign Language Word")):
        gloss_id = normalize_gloss_id(row["#ID"])
        label = row["Chinese Sign Language Word"].strip()
        if not label or gloss_id in glossary:
            raise ValueError(f"Empty label or duplicate gloss ID: {gloss_id}")
        glossary[gloss_id] = label
    if count <= 0 or count > len(glossary):
        raise ValueError(f"Requested {count} classes, but glossary has {len(glossary)}.")
    if class_list is None:
        # Stable across Python versions and glossary row order. No images or
        # model scores are involved. This is NOT the authors' NationalCSL200.
        ids = sorted(glossary, key=lambda gid: (
            hashlib.sha256(f"NationalCSL-stage:{seed}:{gid}".encode("ascii")).hexdigest(), gid
        ))[:count]
        ids.sort()
    else:
        ids = [normalize_gloss_id(row["gloss_id"]) for row in read_csv(class_list, ("gloss_id",))]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f"Class-list CSV must contain exactly {count} unique gloss_id rows.")
        if set(ids) - set(glossary):
            raise ValueError("Class list contains IDs absent from gloss.csv.")
    return [{"class_id": i, "gloss_id": gid,
             "class_name": f"{gid}_{glossary[gid]}", "chinese_gloss": glossary[gid]}
            for i, gid in enumerate(ids)]


def expected_rows(classes):
    rows = []
    for signer in ALL_SIGNERS:
        for item in classes:
            gid = item["gloss_id"]
            sample_id = f"{signer}_{gid}"
            path = f"features/{gid}/{sample_id}.npy"
            row = dict.fromkeys(MANIFEST_COLUMNS, "")
            row.update(item)
            row.update(sample_id=sample_id, path=path, feature_path=path,
                       signer_id=signer, split=split_for_signer(signer),
                       source_archive=f"{signer}.zip", source_sequence=f"{signer}/front/{gid}/",
                       processing_status="pending")
            rows.append(row)
    return rows


def read_protocol(data_root):
    path = Path(data_root) / "protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "nationalcsl-custom-stage-v1":
        raise ValueError("This directory is not a custom NationalCSL stage experiment.")
    if payload.get("protocol_sha256") != canonical_hash(payload["definition"]):
        raise ValueError("Protocol fingerprint is invalid; do not hand-edit protocol.json.")
    definition = payload["definition"]
    if definition["signers"] != {"train": list(TRAIN_SIGNERS), "val": list(VAL_SIGNERS), "test": list(TEST_SIGNERS)}:
        raise ValueError("Protocol signer split does not match this experiment.")
    classes = definition["classes"]
    if not classes or [row["class_id"] for row in classes] != list(range(len(classes))):
        raise ValueError("Class IDs must be contiguous from zero.")
    if len({row["gloss_id"] for row in classes}) != len(classes):
        raise ValueError("Protocol contains duplicate gloss IDs.")
    if len({row["class_name"] for row in classes}) != len(classes):
        raise ValueError("Protocol contains duplicate class names.")
    exported = read_csv(Path(data_root) / "class_order.csv", CLASS_COLUMNS)
    expected = [{k: str(c[k]) for k in CLASS_COLUMNS} for c in classes]
    if exported != expected:
        raise ValueError("class_order.csv differs from the frozen protocol.")
    return payload


def validate_manifest_rows(rows, classes, require_complete=True):
    expected = {row["sample_id"]: row for row in expected_rows(classes)}
    seen = set()
    seen_paths = set()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen or sample_id not in expected:
            raise ValueError(f"Duplicate or unexpected sample: {sample_id}")
        seen.add(sample_id)
        for key in ("sample_id", "path", "feature_path", *CLASS_COLUMNS, "signer_id", "split", "source_archive", "source_sequence"):
            if str(row.get(key, "")) != str(expected[sample_id][key]):
                raise ValueError(f"Manifest mismatch for {sample_id}: {key}")
        if row["path"] in seen_paths:
            raise ValueError(f"Duplicate feature path: {row['path']}")
        seen_paths.add(row["path"])
        if require_complete:
            if row["processing_status"] != "success":
                raise ValueError(f"Incomplete preparation: {sample_id}. Re-run the preparation command.")
            checksum = row.get("feature_sha256", "")
            if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
                raise ValueError(f"Missing/invalid feature checksum: {sample_id}")
    if require_complete and seen != set(expected):
        raise ValueError(f"Expected {len(expected)} samples, found {len(seen)}.")


def load_manifest_definition(manifest_path):
    manifest_path = Path(manifest_path)
    payload = read_protocol(manifest_path.parent)
    classes = payload["definition"]["classes"]
    rows = read_csv(manifest_path, MANIFEST_COLUMNS)
    validate_manifest_rows(rows, classes)
    return rows, [row["class_name"] for row in classes]


def dataset_signature(manifest_path):
    manifest_path = Path(manifest_path)
    payload = read_protocol(manifest_path.parent)
    # Hashes manifest metadata/checksums, not test feature contents.
    return canonical_hash({"manifest_sha256": file_hash(manifest_path),
                           "protocol_sha256": payload["protocol_sha256"]})

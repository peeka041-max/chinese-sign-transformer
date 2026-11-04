"""One-sample inference for the domain MVP 20-class Transformer.

This script does not train, update checkpoints, or tune thresholds. In medical
mode, logits outside the locked medical class set are masked before softmax.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFORMER_ROOT = SCRIPT_DIR.parent
NATIONALCSL200_DIR = TRANSFORMER_ROOT / "nationalcsl200_stage_v1"

for import_root in reversed((SCRIPT_DIR, NATIONALCSL200_DIR, TRANSFORMER_ROOT)):
    import_value = str(import_root)
    if import_value not in sys.path:
        sys.path.insert(0, import_value)

from config_v2 import config_from_mapping  # noqa: E402
from feature_io import load_v2_sequence  # noqa: E402
from feature_processing import MaskAwareStandardizer  # noqa: E402
from prep_support import (  # noqa: E402
    calculate_detection_ratios,
    create_hand_landmarker,
    observations_to_v2_features,
    result_to_observation,
    save_feature_array,
)
from train_domain_mvp20_transfer import MaskAwareSmallTransformer  # noqa: E402
from training_support import load_checkpoint, select_device  # noqa: E402


DEFAULT_CHECKPOINT = Path("domain_mvp_v1/checkpoints/domain_mvp20_transfer.pt")
DEFAULT_CLASS_ORDER = Path("domain_mvp_v1/class_order.csv")
DEFAULT_MEDICAL_CONFIG = Path("domain_mvp_v1/medical_mode_config.json")
DEFAULT_MEDIAPIPE_MODEL = Path("models/hand_landmarker.task")
LOW_CONFIDENCE_DEFAULT = "未识别，请重试"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict one sign with the domain MVP 20-class Transformer."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--feature", type=Path, help="Prepared V2 NumPy feature array with shape (30, 266).")
    source.add_argument("--video", type=Path, help="One video file to decode and process with MediaPipe.")
    source.add_argument("--pics-zip", type=Path, help="NationalCSL-DP Pics ZIP containing signer/front/gloss_id frames.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--class-order", type=Path, default=DEFAULT_CLASS_ORDER)
    parser.add_argument("--medical-config", type=Path, default=DEFAULT_MEDICAL_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MEDIAPIPE_MODEL, help="MediaPipe hand model for --video or --pics-zip.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--signer", help="Signer folder inside --pics-zip, for example Participant_08.")
    parser.add_argument("--gloss-id", help="Four-digit gloss folder inside signer/front, for example 0811.")
    parser.add_argument("--mode", choices=("medical", "all"), default="medical")
    parser.add_argument("--threshold", type=float, default=None, help="Acceptance threshold. Medical mode defaults to medical_mode_config.json.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--save-feature", type=Path, help="Optionally save features extracted from --video.")
    parser.add_argument("--json-output", type=Path, help="Optionally save prediction JSON.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (TRANSFORMER_ROOT / path).resolve()


def require_file(path: Path, description: str) -> Path:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def read_csv(path: Path, required: Iterable[str]) -> List[Mapping[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns {sorted(missing)} in {path}")
        return list(reader)


def load_class_order(path: Path) -> List[Mapping[str, object]]:
    rows = read_csv(path, ("class_id", "gloss_id", "dataset_gloss", "display_word", "match_type"))
    classes = []
    for row in rows:
        classes.append(
            {
                "class_id": int(str(row["class_id"]).strip()),
                "gloss_id": str(row["gloss_id"]).strip().zfill(4),
                "dataset_gloss": str(row["dataset_gloss"]).strip(),
                "display_word": str(row["display_word"]).strip(),
                "match_type": str(row["match_type"]).strip(),
            }
        )
    if len(classes) != 20:
        raise ValueError(f"Expected 20 classes, found {len(classes)}")
    if [item["class_id"] for item in classes] != list(range(20)):
        raise ValueError("class_id must be contiguous from 0 to 19.")
    if len({item["display_word"] for item in classes}) != len(classes):
        raise ValueError("display_word values must be unique.")
    return classes


def load_medical_config(path: Path, classes: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required_ids = list(range(9, 20))
    expected_words = [str(classes[index]["display_word"]) for index in required_ids]
    if config.get("mode_name") != "medical":
        raise ValueError("medical_mode_config.json must declare mode_name=medical.")
    if list(config.get("class_ids", [])) != required_ids:
        raise ValueError("medical_mode_config.json class_ids must be 9..19.")
    if list(config.get("display_words", [])) != expected_words:
        raise ValueError("medical_mode_config.json display_words do not match class_order.csv.")
    return config


def load_model_bundle(checkpoint_path: Path, class_order_path: Path, medical_config_path: Path, device: torch.device):
    classes = load_class_order(class_order_path)
    medical_config = load_medical_config(medical_config_path, classes)
    checkpoint = load_checkpoint(checkpoint_path, device)
    required = {
        "model_kind",
        "model_state_dict",
        "class_names",
        "config",
        "normalizer_mean",
        "normalizer_scale",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {missing}")
    if checkpoint["model_kind"] != "mask_aware_small_transformer":
        raise ValueError(f"Unexpected model_kind: {checkpoint['model_kind']!r}")
    class_names = [str(item["display_word"]) for item in classes]
    if list(checkpoint["class_names"]) != class_names:
        raise ValueError("Checkpoint class_names do not match class_order.csv display_word.")
    config = config_from_mapping(checkpoint["config"])
    if config.input_dim != 266 or config.sequence_length != 30:
        raise ValueError(
            f"Expected checkpoint input_dim=266 and sequence_length=30, got {config.input_dim}/{config.sequence_length}."
        )
    normalizer = MaskAwareStandardizer(checkpoint["normalizer_mean"], checkpoint["normalizer_scale"])
    model = MaskAwareSmallTransformer(len(classes), config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, classes, config, normalizer, checkpoint, medical_config


def extract_video_features(video_path: Path, model_path: Path, sequence_length: int):
    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError("Video inference requires opencv-python and mediapipe.") from error

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    observations = []
    previous_timestamp_ms = -1
    try:
        with create_hand_landmarker(model_path) as landmarker:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb_frame),
                )
                timestamp_ms = int(round(frame_index * 1000.0 / fps))
                timestamp_ms = max(timestamp_ms, previous_timestamp_ms + 1)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                observations.append(
                    result_to_observation(
                        result,
                        frame_index=frame_index,
                        timestamp_seconds=frame_index / fps,
                    )
                )
                previous_timestamp_ms = timestamp_ms
                frame_index += 1
    finally:
        capture.release()

    if not observations:
        raise ValueError(f"Video contains no decodable frames: {video_path}")
    features, _, timestamp_source = observations_to_v2_features(
        observations, fps=fps, target_length=sequence_length
    )
    left_ratio, right_ratio, any_ratio = calculate_detection_ratios(observations)
    return features, {
        "decoded_frames": len(observations),
        "fps": fps,
        "timestamp_source": timestamp_source,
        "left_hand_detection": left_ratio,
        "right_hand_detection": right_ratio,
        "any_hand_detection": any_ratio,
    }


def normalize_gloss_id(value: str) -> str:
    value = str(value).strip()
    if not re.fullmatch(r"\d{1,4}", value):
        raise ValueError(f"--gloss-id must be numeric with at most four digits: {value!r}")
    return value.zfill(4)


def find_pic_zip_entries(archive: zipfile.ZipFile, signer: str, gloss_id: str):
    signer = str(signer).strip()
    gloss_id = normalize_gloss_id(gloss_id)
    if not signer:
        raise ValueError("--signer must not be empty.")
    pattern = re.compile(
        re.escape(signer)
        + r"/front/"
        + re.escape(gloss_id)
        + r"/(\d+)\.(jpg|jpeg|png)$",
        re.I,
    )
    matches = []
    for item in archive.infolist():
        if item.is_dir():
            continue
        normalized_name = item.filename.replace("\\", "/")
        match = pattern.fullmatch(normalized_name)
        if match:
            matches.append((int(match[1]), item.filename))
    if not matches:
        raise ValueError(f"No frames found in ZIP for {signer}/front/{gloss_id}/")
    matches.sort(key=lambda pair: pair[0])
    frame_numbers = [number for number, _ in matches]
    if frame_numbers != list(range(1, len(matches) + 1)):
        raise ValueError(f"Non-contiguous frame sequence in {signer}/front/{gloss_id}/")
    return [entry for _, entry in matches]


def extract_pics_zip_features(archive_path: Path, signer: str, gloss_id: str, model_path: Path, sequence_length: int):
    from prep_support import extract_sequence_features

    with zipfile.ZipFile(archive_path) as archive:
        entries = find_pic_zip_entries(archive, signer, gloss_id)
        features, ratios = extract_sequence_features(archive, entries, model_path)
    expected_shape = (sequence_length, 266)
    if features.shape != expected_shape:
        raise RuntimeError(f"Expected feature shape {expected_shape}, got {features.shape}.")
    return features, {
        "decoded_frames": len(entries),
        "fps": 30.0,
        "timestamp_source": "zip_frame_order_30fps",
        "zip_sequence": f"{signer}/front/{normalize_gloss_id(gloss_id)}/",
        "left_hand_detection": ratios[0],
        "right_hand_detection": ratios[1],
        "any_hand_detection": ratios[2],
    }


def allowed_class_ids(mode: str, medical_config: Mapping[str, object], class_count: int):
    if mode == "all":
        return list(range(class_count))
    return [int(value) for value in medical_config["class_ids"]]


def masked_probabilities(logits: torch.Tensor, allowed_ids: Sequence[int]) -> torch.Tensor:
    masked = torch.full_like(logits, -torch.inf)
    masked[list(allowed_ids)] = logits[list(allowed_ids)]
    return torch.softmax(masked, dim=0)


def class_payload(class_item: Mapping[str, object]):
    return {
        "class_id": int(class_item["class_id"]),
        "gloss_id": str(class_item["gloss_id"]),
        "dataset_gloss": str(class_item["dataset_gloss"]),
        "display_word": str(class_item["display_word"]),
    }


def predict(model, normalizer, classes, medical_config, features, device, mode: str, top_k: int, threshold):
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")
    effective_threshold = threshold
    if effective_threshold is None and mode == "medical":
        effective_threshold = float(medical_config["confidence_threshold"])
    if effective_threshold is not None and not 0.0 <= float(effective_threshold) <= 1.0:
        raise ValueError("--threshold must be in [0, 1].")

    normalized = normalizer.transform(features)
    inputs = torch.from_numpy(normalized).unsqueeze(0).to(device)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(inputs)[0]
        probabilities = masked_probabilities(logits, allowed_class_ids(mode, medical_config, len(classes)))
    inference_time_ms = (time.perf_counter() - start) * 1000.0

    allowed = allowed_class_ids(mode, medical_config, len(classes))
    top_count = min(max(int(top_k), 1), len(allowed))
    scores, indices = probabilities.topk(top_count)
    top_predictions = []
    for rank, (score, class_id) in enumerate(
        zip(scores.detach().cpu().tolist(), indices.detach().cpu().tolist()),
        start=1,
    ):
        payload = class_payload(classes[int(class_id)])
        payload.update(rank=rank, confidence=float(score))
        top_predictions.append(payload)

    best = top_predictions[0]
    accepted = True
    message = None
    prediction = best
    if mode == "medical" and float(best["confidence"]) < float(effective_threshold):
        accepted = False
        prediction = None
        message = str(medical_config.get("low_confidence_message", LOW_CONFIDENCE_DEFAULT))

    return {
        "mode": mode,
        "accepted": accepted,
        "prediction": prediction,
        "confidence": float(best["confidence"]),
        "threshold": effective_threshold,
        "message": message,
        "top_predictions": top_predictions,
        "feature_shape": list(features.shape),
        "inference_time_ms": float(inference_time_ms),
    }


def write_json(path: Path, payload):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def main():
    args = parse_args()
    if args.save_feature and not args.video:
        raise ValueError("--save-feature can only be used with --video.")
    if args.pics_zip and (not args.signer or not args.gloss_id):
        raise ValueError("--pics-zip requires --signer and --gloss-id.")
    if not args.pics_zip and (args.signer or args.gloss_id):
        raise ValueError("--signer and --gloss-id can only be used with --pics-zip.")

    checkpoint_path = require_file(args.checkpoint, "Checkpoint")
    class_order_path = require_file(args.class_order, "class_order.csv")
    medical_config_path = require_file(args.medical_config, "medical_mode_config.json")
    device = select_device(args.device)
    model, classes, config, normalizer, checkpoint, medical_config = load_model_bundle(
        checkpoint_path, class_order_path, medical_config_path, device
    )

    extraction = None
    if args.feature:
        source_path = require_file(args.feature, "Feature array")
        features = load_v2_sequence(source_path, config=config)
        source_type = "feature"
    else:
        model_path = require_file(args.model, "MediaPipe hand model")
        if args.video:
            source_path = require_file(args.video, "Video")
            features, extraction = extract_video_features(source_path, model_path, config.sequence_length)
            source_type = "video"
            if args.save_feature:
                feature_path = resolve_path(args.save_feature)
                save_feature_array(feature_path, features, overwrite=False)
                print(f"Saved extracted feature: {feature_path}")
        else:
            source_path = require_file(args.pics_zip, "Pics ZIP")
            features, extraction = extract_pics_zip_features(
                source_path, args.signer, args.gloss_id, model_path, config.sequence_length
            )
            source_type = "pics_zip"

    result = predict(
        model,
        normalizer,
        classes,
        medical_config,
        features,
        device,
        args.mode,
        args.top_k,
        args.threshold,
    )
    result.update(
        {
            "model": "mask_aware_small_transformer",
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "source_type": source_type,
            "source": str(source_path),
            "extraction": extraction,
        }
    )

    print("=" * 72)
    print("Stage: inference only (checkpoint unchanged; no training)")
    print(f"Mode: {result['mode']}")
    print(f"Device: {device}")
    print(f"Input: {source_path}")
    print(f"Feature shape: {tuple(features.shape)}")
    if extraction:
        print(
            "Hand detection: "
            f"any={extraction['any_hand_detection']:.2%}, "
            f"left={extraction['left_hand_detection']:.2%}, "
            f"right={extraction['right_hand_detection']:.2%}"
        )
    if result["accepted"]:
        pred = result["prediction"]
        print(
            f"Prediction: {pred['display_word']} "
            f"(class_id={pred['class_id']}, gloss_id={pred['gloss_id']}, "
            f"confidence={result['confidence']:.2%})"
        )
    else:
        print(f"Prediction: {result['message']} (confidence={result['confidence']:.2%})")
    print(f"Threshold: {result['threshold']}")
    print("Top predictions:")
    for item in result["top_predictions"]:
        print(
            f"  {item['rank']}. {item['display_word']} "
            f"[{item['gloss_id']}] {item['confidence']:.2%}"
        )
    print(f"Inference time: {result['inference_time_ms']:.2f} ms")
    print("=" * 72)

    if args.json_output:
        output_path = write_json(args.json_output, result)
        print(f"JSON output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

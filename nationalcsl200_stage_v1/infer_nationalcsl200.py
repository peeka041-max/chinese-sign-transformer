"""Run one-sample inference with the frozen NationalCSL custom 200-class model.

This script never trains or changes the checkpoint.  It accepts either an
already prepared V2 ``(30, 266)`` NumPy feature array, one video file, or one
NationalCSL-DP Pics ZIP frame sequence.  Only load a checkpoint produced by
your own trusted training run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import torch

from config_v2 import config_from_mapping
from feature_io import load_v2_sequence
from feature_processing import MaskAwareStandardizer
from prep_support import (
    calculate_detection_ratios,
    create_hand_landmarker,
    observations_to_v2_features,
    result_to_observation,
    save_feature_array,
)
from train_nationalcsl200 import MaskAwareSmallTransformer
from training_support import load_checkpoint, select_device


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Predict one Chinese sign with the frozen custom 200-class "
            "Transformer. No training is performed."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--feature",
        type=Path,
        help="Prepared V2 NumPy array with shape (30, 266).",
    )
    source.add_argument(
        "--video",
        type=Path,
        help="One video to decode and process with MediaPipe.",
    )
    source.add_argument(
        "--pics-zip",
        type=Path,
        help="NationalCSL-DP Pics ZIP containing signer/front/gloss_id frames.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints_v2/nationalcsl200_transformer_optimized.pt"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("../models/hand_landmarker.task"),
        help="MediaPipe model; required only with --video or --pics-zip.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--signer",
        help="Signer folder inside --pics-zip, for example Participant_09.",
    )
    parser.add_argument(
        "--gloss-id",
        help="Four-digit gloss folder inside signer/front, for example 0021.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--save-feature",
        type=Path,
        help="Optionally save the (30, 266) feature extracted from --video.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optionally save a machine-readable prediction JSON file.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def require_file(path: Path, description: str) -> Path:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def load_model_bundle(checkpoint_path: Path, device: torch.device):
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
        raise ValueError(
            "Checkpoint model_kind is not mask_aware_small_transformer: "
            f"{checkpoint['model_kind']!r}"
        )

    class_names = list(checkpoint["class_names"])
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("Checkpoint class_names are empty or duplicated.")
    config = config_from_mapping(checkpoint["config"])
    normalizer = MaskAwareStandardizer(
        checkpoint["normalizer_mean"], checkpoint["normalizer_scale"]
    )
    model = MaskAwareSmallTransformer(len(class_names), config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, class_names, config, normalizer, checkpoint


def extract_video_features(video_path: Path, model_path: Path, sequence_length: int):
    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError(
            "Video inference requires opencv-python and mediapipe in the "
            "sign_transformer environment."
        ) from error

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
    metadata = {
        "decoded_frames": len(observations),
        "fps": fps,
        "timestamp_source": timestamp_source,
        "left_hand_detection": left_ratio,
        "right_hand_detection": right_ratio,
        "any_hand_detection": any_ratio,
    }
    return features, metadata


def normalize_gloss_id(value: str) -> str:
    value = str(value).strip()
    if not re.fullmatch(r"\d{1,4}", value):
        raise ValueError(
            f"--gloss-id must be a numeric ID with at most four digits: {value!r}"
        )
    return value.zfill(4)


def find_pic_zip_entries(archive: zipfile.ZipFile, signer: str, gloss_id: str):
    signer = str(signer).strip()
    if not signer:
        raise ValueError("--signer must not be empty.")
    gloss_id = normalize_gloss_id(gloss_id)

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
        raise ValueError(
            "No image frames found in ZIP for "
            f"{signer}/front/{gloss_id}/"
        )

    matches.sort(key=lambda pair: pair[0])
    frame_numbers = [number for number, _ in matches]
    expected = list(range(1, len(matches) + 1))
    if frame_numbers != expected:
        raise ValueError(
            "Missing, duplicate or non-contiguous frames in ZIP sequence: "
            f"{signer}/front/{gloss_id}/"
        )
    return [entry_name for _, entry_name in matches]


def extract_pics_zip_features(
    archive_path: Path,
    signer: str,
    gloss_id: str,
    model_path: Path,
    sequence_length: int,
):
    from prep_support import extract_sequence_features

    with zipfile.ZipFile(archive_path) as archive:
        entries = find_pic_zip_entries(archive, signer, gloss_id)
        features, ratios = extract_sequence_features(archive, entries, model_path)

    expected_shape = (sequence_length, 266)
    if features.shape != expected_shape:
        raise RuntimeError(
            f"Expected feature shape {expected_shape}, got {features.shape}."
        )
    metadata = {
        "decoded_frames": len(entries),
        "fps": 30.0,
        "timestamp_source": "zip_frame_order_30fps",
        "zip_sequence": f"{signer}/front/{normalize_gloss_id(gloss_id)}/",
        "left_hand_detection": ratios[0],
        "right_hand_detection": ratios[1],
        "any_hand_detection": ratios[2],
    }
    return features, metadata


def split_class_name(class_name: str):
    gloss_id, separator, chinese_gloss = str(class_name).partition("_")
    if not separator:
        return "", str(class_name)
    return gloss_id, chinese_gloss


def predict(model, normalizer, class_names, features, device, top_k):
    normalized = normalizer.transform(features)
    inputs = torch.from_numpy(normalized).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(inputs), dim=1)[0]
    top_k = min(max(int(top_k), 1), len(class_names))
    scores, indices = probabilities.topk(top_k)
    candidates = []
    for rank, (score, class_id) in enumerate(
        zip(scores.detach().cpu().tolist(), indices.detach().cpu().tolist()),
        start=1,
    ):
        class_name = class_names[class_id]
        gloss_id, chinese_gloss = split_class_name(class_name)
        candidates.append(
            {
                "rank": rank,
                "class_id": int(class_id),
                "class_name": class_name,
                "gloss_id": gloss_id,
                "chinese_gloss": chinese_gloss,
                "confidence": float(score),
            }
        )
    return candidates


def write_json(path: Path, payload):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def main():
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.save_feature and not args.video:
        raise ValueError("--save-feature can only be used together with --video.")
    if args.pics_zip and (not args.signer or not args.gloss_id):
        raise ValueError("--pics-zip requires both --signer and --gloss-id.")
    if not args.pics_zip and (args.signer or args.gloss_id):
        raise ValueError("--signer and --gloss-id can only be used with --pics-zip.")

    device = select_device(args.device)
    checkpoint_path = require_file(args.checkpoint, "Checkpoint")
    model, class_names, config, normalizer, checkpoint = load_model_bundle(
        checkpoint_path, device
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
            features, extraction = extract_video_features(
                source_path, model_path, config.sequence_length
            )
            source_type = "video"
            if args.save_feature:
                feature_path = resolve_path(args.save_feature)
                save_feature_array(feature_path, features, overwrite=False)
                print(f"Saved extracted feature: {feature_path}")
        else:
            source_path = require_file(args.pics_zip, "Pics ZIP")
            features, extraction = extract_pics_zip_features(
                source_path,
                args.signer,
                args.gloss_id,
                model_path,
                config.sequence_length,
            )
            source_type = "pics_zip"

    candidates = predict(
        model, normalizer, class_names, features, device, args.top_k
    )
    result = {
        "model": "mask_aware_small_transformer",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "class_count": len(class_names),
        "source_type": source_type,
        "source": str(source_path),
        "feature_shape": list(features.shape),
        "extraction": extraction,
        "prediction": candidates[0],
        "top_k": candidates,
    }

    best = candidates[0]
    print("=" * 72)
    print("Stage: inference only (checkpoint unchanged; no training)")
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
    print(
        f"Prediction: {best['chinese_gloss']} "
        f"(gloss_id={best['gloss_id']}, class_id={best['class_id']}, "
        f"confidence={best['confidence']:.2%})"
    )
    print("Top predictions:")
    for item in candidates:
        print(
            f"  {item['rank']}. {item['chinese_gloss']} "
            f"[{item['gloss_id']}] {item['confidence']:.2%}"
        )
    print("=" * 72)

    if args.json_output:
        output_path = write_json(args.json_output, result)
        print(f"JSON output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""V2 feature extraction reused from the previously successful 20-class pipeline."""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from config_v2 import DEFAULT_CONFIG
from features_v2 import (COORDINATES_PER_LANDMARK, FRAME_FEATURE_DIM,
    LANDMARKS_PER_HAND, build_base_frame_features, compose_v2_features, hand_landmarks_to_array)

@dataclass(frozen=True)
class FrameObservation:
    frame_index: int
    timestamp_seconds: Optional[float]
    left_landmarks: np.ndarray
    right_landmarks: np.ndarray
    left_present: int
    right_present: int



def _optional_positive_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value



def validate_quality_thresholds(good_threshold, warning_threshold):
    if not 0.0 <= warning_threshold <= good_threshold <= 1.0:
        raise ValueError(
            "Quality thresholds must satisfy "
            "0 <= warning_threshold <= good_threshold <= 1."
        )



def _category_score(categories):
    if not categories:
        raise ValueError("MediaPipe returned an empty handedness entry.")
    category = categories[0]
    score = getattr(category, "score", 0.0)
    score = float(score) if score is not None else 0.0
    return str(category.category_name).lower(), score



def result_to_observation(result, frame_index, timestamp_seconds):
    """Retain both 21x3 hands, explicit presence, and frame timestamp."""
    empty_hand = np.zeros(
        (LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK),
        dtype=np.float32,
    )
    left_landmarks = empty_hand.copy()
    right_landmarks = empty_hand.copy()
    left_present = 0
    right_present = 0
    best_scores = {"left": -math.inf, "right": -math.inf}

    hand_landmarks = list(getattr(result, "hand_landmarks", None) or [])
    handedness = list(getattr(result, "handedness", None) or [])
    if len(hand_landmarks) != len(handedness):
        raise ValueError(
            "MediaPipe hand_landmarks and handedness lengths differ."
        )

    for landmarks, categories in zip(hand_landmarks, handedness):
        hand_name, score = _category_score(categories)
        if hand_name not in best_scores:
            raise ValueError(f"Unknown handedness label: {hand_name}")
        if score < best_scores[hand_name]:
            continue

        landmark_array = hand_landmarks_to_array(landmarks)
        best_scores[hand_name] = score
        if hand_name == "left":
            left_landmarks = landmark_array
            left_present = 1
        else:
            right_landmarks = landmark_array
            right_present = 1

    return FrameObservation(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        left_landmarks=left_landmarks,
        right_landmarks=right_landmarks,
        left_present=left_present,
        right_present=right_present,
    )



def choose_time_axis(observations, fps=None):
    """Prefer container timestamps, then FPS time, then frame indices."""
    if not observations:
        raise ValueError("Cannot construct a time axis for an empty video.")

    timestamps = np.asarray(
        [
            np.nan
            if observation.timestamp_seconds is None
            else observation.timestamp_seconds
            for observation in observations
        ],
        dtype=np.float64,
    )
    if (
        np.isfinite(timestamps).all()
        and np.all(np.diff(timestamps) >= 0.0)
        and (len(timestamps) == 1 or timestamps[-1] > timestamps[0])
    ):
        return timestamps, "container_timestamp"

    fps = _optional_positive_float(fps)
    if fps is not None:
        return (
            np.arange(len(observations), dtype=np.float64) / fps,
            "fps",
        )

    return np.arange(len(observations), dtype=np.float64), "frame_index"



def nearest_time_indices(time_axis, target_length):
    """Select observed frames nearest to uniform positions over the full time."""
    time_axis = np.asarray(time_axis, dtype=np.float64)
    if time_axis.ndim != 1 or time_axis.size == 0:
        raise ValueError("time_axis must be a non-empty vector.")
    if not np.isfinite(time_axis).all() or np.any(np.diff(time_axis) < 0.0):
        raise ValueError("time_axis must be finite and non-decreasing.")
    if target_length <= 0:
        raise ValueError("target_length must be positive.")
    if time_axis.size == 1:
        return np.zeros(target_length, dtype=np.int64)

    target_times = np.linspace(
        time_axis[0],
        time_axis[-1],
        target_length,
    )
    right_indices = np.searchsorted(time_axis, target_times, side="left")
    right_indices = np.clip(right_indices, 0, time_axis.size - 1)
    left_indices = np.maximum(right_indices - 1, 0)

    left_distances = np.abs(target_times - time_axis[left_indices])
    right_distances = np.abs(time_axis[right_indices] - target_times)
    use_right = right_distances < left_distances
    return np.where(use_right, right_indices, left_indices).astype(np.int64)



def observations_to_v2_features(
    observations,
    fps=None,
    target_length=DEFAULT_CONFIG.sequence_length,
):
    """Time-resample all-frame observations, then compose V2 features.

    The velocity block remains a difference between adjacent normalized time
    steps after resampling. It is deliberately not divided by elapsed seconds,
    preserving the V2 feature definition that passed the existing smoke tests.
    """
    if not observations:
        raise ValueError("Observation sequence cannot be empty.")

    base_frames = []
    presence_masks = []
    for observation in observations:
        left_hand = (
            observation.left_landmarks if observation.left_present else None
        )
        right_hand = (
            observation.right_landmarks if observation.right_present else None
        )
        base_feature, presence = build_base_frame_features(
            left_hand,
            right_hand,
        )
        base_frames.append(base_feature)
        presence_masks.append(presence)

    time_axis, timestamp_source = choose_time_axis(observations, fps=fps)
    indices = nearest_time_indices(time_axis, target_length=target_length)
    resampled_base = np.asarray(base_frames, dtype=np.float32)[indices]
    resampled_presence = np.asarray(
        presence_masks,
        dtype=np.float32,
    )[indices]

    features = compose_v2_features(resampled_base, resampled_presence)
    features = np.asarray(features, dtype=np.float32)
    expected_shape = (target_length, FRAME_FEATURE_DIM)
    if features.shape != expected_shape:
        raise RuntimeError(
            f"Expected preprocessed feature shape {expected_shape}, "
            f"got {features.shape}."
        )
    if not np.isfinite(features).all():
        raise RuntimeError("Preprocessed features contain NaN or Inf.")
    return features, indices, timestamp_source



def calculate_detection_ratios(observations):
    """Calculate quality ratios over all original decoded frames."""
    if not observations:
        raise ValueError("Cannot calculate detection ratios for no frames.")
    left = np.asarray(
        [observation.left_present for observation in observations],
        dtype=np.float64,
    )
    right = np.asarray(
        [observation.right_present for observation in observations],
        dtype=np.float64,
    )
    any_hand = np.maximum(left, right)
    return float(left.mean()), float(right.mean()), float(any_hand.mean())



def quality_status(any_hand_ratio, good_threshold, warning_threshold):
    validate_quality_thresholds(good_threshold, warning_threshold)
    if any_hand_ratio >= good_threshold:
        return "good"
    if any_hand_ratio >= warning_threshold:
        return "warning"
    return "bad"



def create_hand_landmarker(model_path):
    import mediapipe as mp
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)



def decode_zip_image(archive, entry_name):
    import cv2
    encoded = np.frombuffer(archive.read(entry_name), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"OpenCV could not decode ZIP entry: {entry_name}")
    return frame



def extract_sequence_features(archive, entries, model_path):
    import cv2
    import mediapipe as mp
    observations = []
    with create_hand_landmarker(model_path) as landmarker:
        for frame_index, entry_name in enumerate(entries):
            frame = decode_zip_image(archive, entry_name)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_frame),
            )
            timestamp_ms = frame_index * 33
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            observations.append(
                result_to_observation(
                    result,
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / 30.0,
                )
            )

    features, _, _ = observations_to_v2_features(
        observations,
        fps=30.0,
        target_length=DEFAULT_CONFIG.sequence_length,
    )
    ratios = calculate_detection_ratios(observations)
    expected_shape = (DEFAULT_CONFIG.sequence_length, FRAME_FEATURE_DIM)
    if features.shape != expected_shape:
        raise RuntimeError(
            f"Expected feature shape {expected_shape}, got {features.shape}."
        )
    if features.dtype != np.float32 or not np.isfinite(features).all():
        raise RuntimeError("Feature array must be finite float32 data.")
    return features, ratios



def save_feature_array(path, features, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Feature already exists: {path}. Use --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.save(handle, features, allow_pickle=False)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


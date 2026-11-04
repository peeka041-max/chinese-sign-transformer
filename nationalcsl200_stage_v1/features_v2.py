"""Transformer V2 feature construction.

V2 features must be created while left/right hand presence is explicit, ideally
from MediaPipe Hand Landmarker results. A legacy ``(30, 126)`` array must not be
silently treated as V2 data because its presence semantics are not declared.
"""

from typing import Optional, Sequence

import numpy as np

from config_v2 import DEFAULT_CONFIG


LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
HAND_FEATURE_DIM = LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK

LOCAL_FEATURE_DIM = HAND_FEATURE_DIM * 2
WRIST_FEATURE_DIM = COORDINATES_PER_LANDMARK * 2
BASE_FEATURE_DIM = LOCAL_FEATURE_DIM + WRIST_FEATURE_DIM
VELOCITY_FEATURE_DIM = BASE_FEATURE_DIM
PRESENCE_FEATURE_DIM = 2
FRAME_FEATURE_DIM = (
    BASE_FEATURE_DIM + VELOCITY_FEATURE_DIM + PRESENCE_FEATURE_DIM
)

LEFT_LOCAL_SLICE = slice(0, HAND_FEATURE_DIM)
RIGHT_LOCAL_SLICE = slice(HAND_FEATURE_DIM, LOCAL_FEATURE_DIM)
LEFT_WRIST_SLICE = slice(LOCAL_FEATURE_DIM, LOCAL_FEATURE_DIM + 3)
RIGHT_WRIST_SLICE = slice(LOCAL_FEATURE_DIM + 3, BASE_FEATURE_DIM)

BASE_SLICE = slice(0, BASE_FEATURE_DIM)
VELOCITY_SLICE = slice(BASE_FEATURE_DIM, BASE_FEATURE_DIM * 2)
PRESENCE_SLICE = slice(BASE_FEATURE_DIM * 2, FRAME_FEATURE_DIM)


if FRAME_FEATURE_DIM != DEFAULT_CONFIG.input_dim:
    raise RuntimeError(
        "features_v2 FRAME_FEATURE_DIM does not match config_v2 input_dim."
    )


def hand_landmarks_to_array(hand_landmarks):
    """Convert one MediaPipe-like hand or a numeric array to ``(21, 3)``."""
    if hand_landmarks is None:
        raise ValueError("hand_landmarks cannot be None in this function.")

    if isinstance(hand_landmarks, np.ndarray):
        array = np.asarray(hand_landmarks, dtype=np.float32)
    else:
        points = list(hand_landmarks)
        if points and all(
            hasattr(points[0], coordinate)
            for coordinate in ("x", "y", "z")
        ):
            array = np.asarray(
                [[point.x, point.y, point.z] for point in points],
                dtype=np.float32,
            )
        else:
            array = np.asarray(points, dtype=np.float32)

    expected_shape = (LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK)
    if array.shape != expected_shape:
        raise ValueError(
            f"Expected hand landmark shape {expected_shape}, got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError("Hand landmarks contain NaN or infinite values.")
    return array


def build_base_frame_features(
    left_landmarks=None,
    right_landmarks=None,
):
    """Return one 132-D base feature and its explicit 2-D presence mask."""
    base_feature = np.zeros(BASE_FEATURE_DIM, dtype=np.float32)
    presence = np.zeros(PRESENCE_FEATURE_DIM, dtype=np.float32)

    if left_landmarks is not None:
        left = hand_landmarks_to_array(left_landmarks)
        left_wrist = left[0].copy()
        left_local = left - left_wrist
        left_local[0] = 0.0
        base_feature[LEFT_LOCAL_SLICE] = left_local.reshape(-1)
        base_feature[LEFT_WRIST_SLICE] = left_wrist
        presence[0] = 1.0

    if right_landmarks is not None:
        right = hand_landmarks_to_array(right_landmarks)
        right_wrist = right[0].copy()
        right_local = right - right_wrist
        right_local[0] = 0.0
        base_feature[RIGHT_LOCAL_SLICE] = right_local.reshape(-1)
        base_feature[RIGHT_WRIST_SLICE] = right_wrist
        presence[1] = 1.0

    return base_feature, presence


def extract_base_frame_features(result):
    """Convert one MediaPipe Tasks HandLandmarkerResult to base/presence."""
    hand_landmarks = list(getattr(result, "hand_landmarks", None) or [])
    handedness = list(getattr(result, "handedness", None) or [])

    if len(hand_landmarks) != len(handedness):
        raise ValueError(
            "MediaPipe result hand_landmarks and handedness lengths differ."
        )

    left_hand = None
    right_hand = None

    for landmarks, categories in zip(hand_landmarks, handedness):
        if not categories:
            raise ValueError("MediaPipe handedness entry is empty.")
        hand_name = str(categories[0].category_name).lower()
        if hand_name == "left":
            left_hand = landmarks
        elif hand_name == "right":
            right_hand = landmarks
        else:
            raise ValueError(f"Unknown MediaPipe handedness label: {hand_name}")

    return build_base_frame_features(left_hand, right_hand)


def _validate_base_and_presence(base_sequence, presence_mask):
    base_sequence = np.asarray(base_sequence, dtype=np.float32)
    presence_mask = np.asarray(presence_mask, dtype=np.float32)

    if base_sequence.ndim != 2 or base_sequence.shape[1] != BASE_FEATURE_DIM:
        raise ValueError(
            f"base_sequence must have shape (T, {BASE_FEATURE_DIM}), "
            f"got {base_sequence.shape}."
        )
    if presence_mask.shape != (base_sequence.shape[0], PRESENCE_FEATURE_DIM):
        raise ValueError(
            "presence_mask must have shape "
            f"(T, {PRESENCE_FEATURE_DIM}), got {presence_mask.shape}."
        )
    if base_sequence.shape[0] == 0:
        raise ValueError("Feature sequence cannot be empty.")
    if not np.isfinite(base_sequence).all():
        raise ValueError("base_sequence contains NaN or infinite values.")
    if not np.isfinite(presence_mask).all():
        raise ValueError("presence_mask contains NaN or infinite values.")
    if not np.all(np.logical_or(presence_mask == 0.0, presence_mask == 1.0)):
        raise ValueError("presence_mask values must be exactly 0 or 1.")

    return base_sequence, presence_mask


def resample_base_and_presence(
    base_sequence,
    presence_mask,
    target_length=DEFAULT_CONFIG.sequence_length,
):
    """Uniformly resample base features and masks before velocity is computed."""
    base_sequence, presence_mask = _validate_base_and_presence(
        base_sequence,
        presence_mask,
    )
    if target_length <= 0:
        raise ValueError("target_length must be positive.")

    indices = np.linspace(
        0,
        base_sequence.shape[0] - 1,
        target_length,
    )
    indices = np.round(indices).astype(np.int64)
    return base_sequence[indices], presence_mask[indices]


def compose_v2_features(base_sequence, presence_mask):
    """Compose base, masked temporal velocity, and presence into 266-D frames."""
    base_sequence, presence_mask = _validate_base_and_presence(
        base_sequence,
        presence_mask,
    )

    velocity = np.zeros_like(base_sequence, dtype=np.float32)
    if base_sequence.shape[0] > 1:
        delta = base_sequence[1:] - base_sequence[:-1]
        left_valid = (
            presence_mask[1:, 0] * presence_mask[:-1, 0]
        ).astype(np.float32)
        right_valid = (
            presence_mask[1:, 1] * presence_mask[:-1, 1]
        ).astype(np.float32)

        velocity[1:, LEFT_LOCAL_SLICE] = (
            delta[:, LEFT_LOCAL_SLICE] * left_valid[:, None]
        )
        velocity[1:, LEFT_WRIST_SLICE] = (
            delta[:, LEFT_WRIST_SLICE] * left_valid[:, None]
        )
        velocity[1:, RIGHT_LOCAL_SLICE] = (
            delta[:, RIGHT_LOCAL_SLICE] * right_valid[:, None]
        )
        velocity[1:, RIGHT_WRIST_SLICE] = (
            delta[:, RIGHT_WRIST_SLICE] * right_valid[:, None]
        )

    features = np.concatenate(
        [base_sequence, velocity, presence_mask],
        axis=1,
    ).astype(np.float32, copy=False)

    expected_shape = (base_sequence.shape[0], FRAME_FEATURE_DIM)
    if features.shape != expected_shape:
        raise RuntimeError(
            f"Expected V2 feature shape {expected_shape}, got {features.shape}."
        )
    return features


def build_v2_sequence_from_landmarks(
    left_sequence: Optional[Sequence],
    right_sequence: Optional[Sequence],
    target_length=None,
):
    """Build V2 features from frame-aligned hands where ``None`` means absent."""
    if left_sequence is None and right_sequence is None:
        raise ValueError("At least one hand sequence container is required.")

    frame_count = len(
        left_sequence if left_sequence is not None else right_sequence
    )
    if left_sequence is not None and len(left_sequence) != frame_count:
        raise ValueError("left_sequence length is inconsistent.")
    if right_sequence is not None and len(right_sequence) != frame_count:
        raise ValueError("right_sequence length is inconsistent.")
    if frame_count == 0:
        raise ValueError("Landmark sequence cannot be empty.")

    base_frames = []
    masks = []
    for index in range(frame_count):
        left_hand = None if left_sequence is None else left_sequence[index]
        right_hand = None if right_sequence is None else right_sequence[index]
        base_feature, presence = build_base_frame_features(
            left_hand,
            right_hand,
        )
        base_frames.append(base_feature)
        masks.append(presence)

    base_sequence = np.asarray(base_frames, dtype=np.float32)
    presence_mask = np.asarray(masks, dtype=np.float32)
    if target_length is not None:
        base_sequence, presence_mask = resample_base_and_presence(
            base_sequence,
            presence_mask,
            target_length=target_length,
        )
    return compose_v2_features(base_sequence, presence_mask)


def extract_v2_sequence_from_results(
    results: Sequence,
    target_length=DEFAULT_CONFIG.sequence_length,
):
    """Create one V2 action from raw MediaPipe Results, then resample to 30."""
    if len(results) == 0:
        raise ValueError("MediaPipe result sequence cannot be empty.")

    base_frames = []
    masks = []
    for result in results:
        base_feature, presence = extract_base_frame_features(result)
        base_frames.append(base_feature)
        masks.append(presence)

    base_sequence, presence_mask = resample_base_and_presence(
        base_frames,
        masks,
        target_length=target_length,
    )
    return compose_v2_features(base_sequence, presence_mask)

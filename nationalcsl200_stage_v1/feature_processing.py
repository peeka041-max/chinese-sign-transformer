"""Mask-aware normalization and conservative training augmentation."""
import numpy as np
from feature_io import verified_sequence
from features_v2 import (BASE_FEATURE_DIM, LEFT_LOCAL_SLICE, LEFT_WRIST_SLICE,
    PRESENCE_SLICE, RIGHT_LOCAL_SLICE, RIGHT_WRIST_SLICE, compose_v2_features)
CONTINUOUS_DIM = BASE_FEATURE_DIM * 2

def continuous_valid_mask(sequence):
    """Return the validity mask for 264 continuous channels."""
    presence = sequence[:, PRESENCE_SLICE]
    left = presence[:, 0] > 0.5
    right = presence[:, 1] > 0.5
    left_velocity = left & np.concatenate(([False], left[:-1]))
    right_velocity = right & np.concatenate(([False], right[:-1]))

    mask = np.zeros((sequence.shape[0], CONTINUOUS_DIM), dtype=bool)
    mask[left, LEFT_LOCAL_SLICE] = True
    mask[left, LEFT_WRIST_SLICE] = True
    mask[right, RIGHT_LOCAL_SLICE] = True
    mask[right, RIGHT_WRIST_SLICE] = True

    velocity_offset = BASE_FEATURE_DIM
    mask[left_velocity, velocity_offset + LEFT_LOCAL_SLICE.start : velocity_offset + LEFT_LOCAL_SLICE.stop] = True
    mask[left_velocity, velocity_offset + LEFT_WRIST_SLICE.start : velocity_offset + LEFT_WRIST_SLICE.stop] = True
    mask[right_velocity, velocity_offset + RIGHT_LOCAL_SLICE.start : velocity_offset + RIGHT_LOCAL_SLICE.stop] = True
    mask[right_velocity, velocity_offset + RIGHT_WRIST_SLICE.start : velocity_offset + RIGHT_WRIST_SLICE.stop] = True
    return mask



class MaskAwareStandardizer:
    """Per-channel statistics computed only from valid training landmarks."""

    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        if self.mean.shape != (CONTINUOUS_DIM,) or self.scale.shape != (CONTINUOUS_DIM,):
            raise ValueError("Normalizer mean/scale must have 264 values.")

    @classmethod
    def fit(cls, dataset, train_indices):
        sums = np.zeros(CONTINUOUS_DIM, dtype=np.float64)
        squared_sums = np.zeros(CONTINUOUS_DIM, dtype=np.float64)
        counts = np.zeros(CONTINUOUS_DIM, dtype=np.float64)
        for index in train_indices:
            sequence = verified_sequence(dataset, dataset.samples[index].path)
            values = sequence[:, :CONTINUOUS_DIM].astype(np.float64)
            valid = continuous_valid_mask(sequence)
            sums += np.where(valid, values, 0.0).sum(axis=0)
            squared_sums += np.where(valid, values * values, 0.0).sum(axis=0)
            counts += valid.sum(axis=0)

        safe_counts = np.maximum(counts, 1.0)
        mean = sums / safe_counts
        variance = np.maximum(squared_sums / safe_counts - mean * mean, 0.0)
        scale = np.sqrt(variance)
        mean[counts == 0] = 0.0
        scale[(counts < 2) | (scale < 1e-4)] = 1.0
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, sequence):
        output = np.asarray(sequence, dtype=np.float32).copy()
        valid = continuous_valid_mask(output)
        normalized = (output[:, :CONTINUOUS_DIM] - self.mean) / self.scale
        output[:, :CONTINUOUS_DIM] = np.where(valid, normalized, 0.0)
        output[:, PRESENCE_SLICE] = (output[:, PRESENCE_SLICE] > 0.5).astype(np.float32)
        return output



def augment_sequence(sequence):
    """Apply conservative signer/tempo variation, then recompute velocity."""
    base = np.asarray(sequence[:, :BASE_FEATURE_DIM], dtype=np.float32).copy()
    presence = np.asarray(sequence[:, PRESENCE_SLICE], dtype=np.float32).copy()
    frame_count = base.shape[0]

    # Smooth monotonic time warp. Nearest sampling preserves explicit absence.
    gamma = float(np.random.uniform(0.86, 1.16))
    positions = np.linspace(0.0, 1.0, frame_count) ** gamma
    indices = np.rint(positions * (frame_count - 1)).astype(np.int64)
    base = base[indices]
    presence = presence[indices]

    # Hand-size variation affects wrist-local landmark geometry only.
    for hand_index, local_slice in enumerate((LEFT_LOCAL_SLICE, RIGHT_LOCAL_SLICE)):
        hand_scale = float(np.random.uniform(0.92, 1.08))
        valid = presence[:, hand_index] > 0.5
        base[valid, local_slice] *= hand_scale

    # Camera/signer position jitter: translate both wrists together.
    shift_x, shift_y = np.random.normal(0.0, 0.012, size=2).astype(np.float32)
    for hand_index, wrist_slice in enumerate((LEFT_WRIST_SLICE, RIGHT_WRIST_SLICE)):
        valid = presence[:, hand_index] > 0.5
        base[valid, wrist_slice.start] += shift_x
        base[valid, wrist_slice.start + 1] += shift_y

    # Small coordinate noise; absent hands stay exactly zero.
    base_mask = continuous_valid_mask(
        compose_v2_features(base, presence)
    )[:, :BASE_FEATURE_DIM]
    noise = np.random.normal(0.0, 0.0015, size=base.shape).astype(np.float32)
    base = np.where(base_mask, base + noise, 0.0).astype(np.float32)
    # The wrist landmark is the local-coordinate origin for each hand.
    base[:, LEFT_LOCAL_SLICE.start : LEFT_LOCAL_SLICE.start + 3] = 0.0
    base[:, RIGHT_LOCAL_SLICE.start : RIGHT_LOCAL_SLICE.start + 3] = 0.0
    return compose_v2_features(base, presence)


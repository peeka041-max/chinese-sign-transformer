"""Strict V2 array loading and per-array integrity checks."""
from pathlib import Path
import numpy as np
from config_v2 import DEFAULT_CONFIG

def load_v2_sequence(file_path, config=DEFAULT_CONFIG):
    """Load only an explicitly prepared V2 ``(30, 266)`` array."""
    file_path = Path(file_path)
    sequence = np.load(file_path, allow_pickle=False)

    legacy_shape = (config.sequence_length, 126)
    expected_shape = (config.sequence_length, config.input_dim)
    if sequence.shape == legacy_shape:
        raise ValueError(
            f"{file_path} contains legacy V1 shape {legacy_shape}. "
            "Do not infer V2 wrist, velocity, or presence fields silently. "
            "Create V2 arrays explicitly with features_v2 from frame-aligned "
            "landmarks or MediaPipe HandLandmarkerResult objects."
        )
    if sequence.shape != expected_shape:
        raise ValueError(
            f"{file_path} must have V2 shape {expected_shape}, "
            f"got {sequence.shape}."
        )

    sequence = np.asarray(sequence, dtype=np.float32)
    if not np.isfinite(sequence).all():
        raise ValueError(f"{file_path} contains NaN or infinite values.")
    return sequence



def verified_sequence(dataset, path):
    from protocol import file_hash
    key = str(Path(path).resolve())
    if key not in dataset.verified_features:
        if file_hash(path) != dataset.feature_checksums[key]:
            raise ValueError(f"Feature checksum mismatch: {path}. Re-run preparation.")
        dataset.verified_features.add(key)
    return load_v2_sequence(path, dataset.config)


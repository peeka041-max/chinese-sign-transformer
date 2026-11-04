"""Shared real-time frame pipeline for the 200-class and medical models."""

from __future__ import annotations

import csv
import math
import sys
import threading
import time
from itertools import chain
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFORMER_ROOT = SCRIPT_DIR.parent
NATIONALCSL200_DIR = TRANSFORMER_ROOT / "nationalcsl200_stage_v1"

for import_root in reversed((SCRIPT_DIR, NATIONALCSL200_DIR, TRANSFORMER_ROOT)):
    import_value = str(import_root)
    if import_value not in sys.path:
        sys.path.insert(0, import_value)

import infer_nationalcsl200 as national200_infer  # noqa: E402
from medical_inference_engine import MedicalInferenceEngine  # noqa: E402
from prep_support import (  # noqa: E402
    calculate_detection_ratios,
    create_hand_landmarker,
    observations_to_v2_features,
    result_to_observation,
)


MODE_NATIONAL200 = "nationalcsl200"
MODE_MEDICAL = "medical"
MODEL_LABELS = {
    MODE_NATIONAL200: "NationalCSL 200类实验模型",
    MODE_MEDICAL: "医疗11类场景模型",
}
DEFAULT_MEDIAPIPE_MODEL = TRANSFORMER_ROOT / "models" / "hand_landmarker.task"
DEFAULT_NATIONAL200_CHECKPOINT = (
    NATIONALCSL200_DIR
    / "checkpoints_v2"
    / "nationalcsl200_transformer_optimized.pt"
)
DEFAULT_NATIONAL200_CLASS_ORDER = (
    NATIONALCSL200_DIR / "nationalcsl200_v2" / "class_order.csv"
)


def require_file(path: Path, description: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


class SharedFrameFeatureProcessor:
    """Own one MediaPipe task and construct the established 30x266 features."""

    def __init__(self, model_path: Path = DEFAULT_MEDIAPIPE_MODEL):
        started = time.perf_counter()
        try:
            import cv2
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "Real-time inference requires opencv-python and mediapipe."
            ) from error

        self.cv2 = cv2
        self.mp = mp
        self.model_path = require_file(model_path, "MediaPipe hand model")
        self.landmarker = create_hand_landmarker(self.model_path)
        self._lock = threading.RLock()
        self._closed = False
        self._next_timestamp_ms = 0
        self._sequences_processed = 0
        self._stream_observations = None
        self._stream_base_ms = None
        self.load_count = 1
        self.load_time_ms = (time.perf_counter() - started) * 1000.0

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SharedFrameFeatureProcessor is closed.")

    def _mp_image(self, frame: np.ndarray):
        rgb_frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        return self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )

    def _clear_previous_tracking(self, frame: np.ndarray) -> None:
        if not self._sequences_processed:
            return
        empty_frame = np.zeros_like(frame)
        for _ in range(3):
            self.landmarker.detect_for_video(
                self._mp_image(empty_frame), self._next_timestamp_ms
            )
            self._next_timestamp_ms += 33

    def _detect_observation(
        self,
        frame: np.ndarray,
        frame_index: int,
        local_timestamp_seconds: float,
        sequence_base_ms: int,
    ):
        timestamp_ms = max(
            sequence_base_ms + int(round(local_timestamp_seconds * 1000.0)),
            self._next_timestamp_ms,
        )
        result = self.landmarker.detect_for_video(
            self._mp_image(frame), timestamp_ms
        )
        self._next_timestamp_ms = timestamp_ms + 1
        return result_to_observation(
            result,
            frame_index=frame_index,
            timestamp_seconds=local_timestamp_seconds,
        )

    @staticmethod
    def _measured_fps(observations: Sequence, fallback_fps: float) -> float:
        if len(observations) > 1:
            duration = (
                float(observations[-1].timestamp_seconds)
                - float(observations[0].timestamp_seconds)
            )
            if duration > 0.0:
                return (len(observations) - 1) / duration
        return float(fallback_fps)

    @staticmethod
    def _validate_features(features: np.ndarray, sequence_length: int) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        expected_shape = (sequence_length, 266)
        if features.shape != expected_shape:
            raise RuntimeError(
                f"Expected feature shape {expected_shape}, got {features.shape}."
            )
        if not np.isfinite(features).all():
            raise RuntimeError("Features contain NaN or Inf.")
        return features

    def _finalize(
        self,
        observations: Sequence,
        sequence_length: int,
        fallback_fps: float,
        source: str,
    ):
        if not observations:
            raise ValueError("No frames were captured.")
        measured_fps = self._measured_fps(observations, fallback_fps)
        features, _, timestamp_source = observations_to_v2_features(
            observations,
            fps=measured_fps,
            target_length=sequence_length,
        )
        features = self._validate_features(features, sequence_length)
        left, right, any_hand = calculate_detection_ratios(observations)
        metadata = {
            "captured_frames": len(observations),
            "actual_fps": float(measured_fps),
            "timestamp_source": timestamp_source,
            "source": source,
            "left_hand_detection": left,
            "right_hand_detection": right,
            "any_hand_detection": any_hand,
        }
        return features, metadata

    def features_from_frames(
        self,
        frames: Iterable[np.ndarray],
        fps: float,
        sequence_length: int = 30,
        source: str = "frame_list",
    ):
        if not math.isfinite(float(fps)) or float(fps) <= 0.0:
            raise ValueError("fps must be a positive finite value.")
        with self._lock:
            self._ensure_open()
            if self._stream_observations is not None:
                raise RuntimeError("Cannot process a frame list during recording.")
            iterator = iter(frames)
            try:
                first_frame = next(iterator)
            except StopIteration as error:
                raise ValueError("Frame list is empty.") from error
            self._clear_previous_tracking(first_frame)
            sequence_base_ms = self._next_timestamp_ms
            observations = [
                self._detect_observation(
                    frame,
                    frame_index=index,
                    local_timestamp_seconds=index / float(fps),
                    sequence_base_ms=sequence_base_ms,
                )
                for index, frame in enumerate(chain((first_frame,), iterator))
            ]
            self._sequences_processed += 1
            return self._finalize(
                observations, sequence_length, float(fps), source
            )

    def features_from_video(self, video_path: Path, sequence_length: int = 30):
        path = require_file(video_path, "Video")
        capture = self.cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {path}")
        fps = float(capture.get(self.cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0.0:
            fps = 30.0

        def decoded_frames():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame

        try:
            return self.features_from_frames(
                decoded_frames(), fps, sequence_length, source="video"
            )
        finally:
            capture.release()

    def begin_stream(self) -> None:
        with self._lock:
            self._ensure_open()
            if self._stream_observations is not None:
                raise RuntimeError("A recording is already active.")
            self._stream_observations = []
            self._stream_base_ms = None

    def process_stream_frame(
        self, frame: np.ndarray, timestamp_seconds: float
    ) -> None:
        with self._lock:
            self._ensure_open()
            if self._stream_observations is None:
                raise RuntimeError("No recording is active.")
            local_timestamp = max(float(timestamp_seconds), 0.0)
            if not self._stream_observations:
                self._clear_previous_tracking(frame)
                self._stream_base_ms = self._next_timestamp_ms
            observation = self._detect_observation(
                frame,
                frame_index=len(self._stream_observations),
                local_timestamp_seconds=local_timestamp,
                sequence_base_ms=int(self._stream_base_ms),
            )
            self._stream_observations.append(observation)

    def finish_stream(self, sequence_length: int = 30):
        with self._lock:
            self._ensure_open()
            if self._stream_observations is None:
                raise RuntimeError("No recording is active.")
            observations = self._stream_observations
            self._stream_observations = None
            self._stream_base_ms = None
            if observations:
                self._sequences_processed += 1
            return self._finalize(
                observations, sequence_length, 30.0, source="camera"
            )

    def cancel_stream(self) -> None:
        with self._lock:
            if self._stream_observations:
                self._sequences_processed += 1
            self._stream_observations = None
            self._stream_base_ms = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._stream_observations = None
            self.landmarker.close()
            self._closed = True


class NationalCSL200Backend:
    def __init__(
        self,
        device,
        top_k: int,
        checkpoint_path: Path = DEFAULT_NATIONAL200_CHECKPOINT,
        class_order_path: Path = DEFAULT_NATIONAL200_CLASS_ORDER,
    ):
        self.checkpoint_path = require_file(checkpoint_path, "200-class checkpoint")
        self.class_order_path = require_file(class_order_path, "200-class class_order.csv")
        self.device = device
        self.top_k = int(top_k)
        self.model, self.class_names, self.config, self.normalizer, self.checkpoint = (
            national200_infer.load_model_bundle(self.checkpoint_path, self.device)
        )
        with self.class_order_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            self.class_rows = list(csv.DictReader(handle))
        if len(self.class_rows) != 200:
            raise ValueError(
                f"Expected 200 class_order rows, found {len(self.class_rows)}."
            )
        if [int(row["class_id"]) for row in self.class_rows] != list(range(200)):
            raise ValueError("200-class class_id values must be contiguous from 0 to 199.")
        if [row["class_name"] for row in self.class_rows] != self.class_names:
            raise ValueError("200-class checkpoint labels do not match class_order.csv.")
        if self.config.input_dim != 266 or self.config.sequence_length != 30:
            raise ValueError("200-class checkpoint must use feature shape (30, 266).")

    def predict(self, features: np.ndarray) -> dict:
        started = time.perf_counter()
        candidates = national200_infer.predict(
            self.model,
            self.normalizer,
            self.class_names,
            features,
            self.device,
            self.top_k,
        )
        inference_time_ms = (time.perf_counter() - started) * 1000.0
        top_predictions = []
        for item in candidates:
            row = self.class_rows[int(item["class_id"])]
            top_predictions.append(
                {
                    "rank": int(item["rank"]),
                    "class_id": int(item["class_id"]),
                    "gloss_id": str(row["gloss_id"]).strip().zfill(4),
                    "display_word": str(row["chinese_gloss"]).strip(),
                    "confidence": float(item["confidence"]),
                }
            )
        best = top_predictions[0]
        return {
            "mode": MODE_NATIONAL200,
            "accepted": True,
            "prediction": best,
            "confidence": float(best["confidence"]),
            "threshold": None,
            "message": None,
            "top_predictions": top_predictions,
            "feature_shape": list(features.shape),
            "inference_time_ms": inference_time_ms,
        }

    def close(self) -> None:
        return None


class MedicalBackend:
    def __init__(self, device: str, top_k: int):
        self.engine = MedicalInferenceEngine(
            device=str(device),
            top_k=top_k,
            load_mediapipe=False,
        )
        self.config = self.engine.config

    def predict(self, features: np.ndarray) -> dict:
        result = self.engine.predict_features(features)
        result["mode"] = MODE_MEDICAL
        return result

    def close(self) -> None:
        self.engine.close()


class RealtimeSignEngine:
    """Cache model backends and share one MediaPipe processor across modes."""

    def __init__(self, default_mode: str = MODE_NATIONAL200, device: str = "cpu", top_k: int = 3):
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self.device_name = device
        self.device = national200_infer.select_device(device)
        self.top_k = int(top_k)
        self.processor = SharedFrameFeatureProcessor()
        self.backends = {}
        self.backend_load_counts = {MODE_NATIONAL200: 0, MODE_MEDICAL: 0}
        self.current_mode = None
        self.recording_mode = None
        self._lock = threading.RLock()
        self._closed = False
        self.switch_model(default_mode)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RealtimeSignEngine is closed.")

    def _load_backend(self, mode: str):
        if mode == MODE_NATIONAL200:
            backend = NationalCSL200Backend(self.device, self.top_k)
        elif mode == MODE_MEDICAL:
            backend = MedicalBackend(self.device_name, self.top_k)
        else:
            raise ValueError(f"Unknown model mode: {mode}")
        self.backends[mode] = backend
        self.backend_load_counts[mode] += 1
        return backend

    def switch_model(self, mode: str) -> dict:
        with self._lock:
            self._ensure_open()
            if self.recording_mode is not None:
                raise RuntimeError("Cannot switch models while recording.")
            started = time.perf_counter()
            loaded_now = mode not in self.backends
            if loaded_now:
                self._load_backend(mode)
            self.current_mode = mode
            return {
                "mode": mode,
                "label": MODEL_LABELS[mode],
                "loaded_now": loaded_now,
                "load_time_ms": (time.perf_counter() - started) * 1000.0,
                "load_counts": self.load_counts(),
            }

    def load_counts(self) -> dict:
        return {
            "mediapipe": int(self.processor.load_count),
            MODE_NATIONAL200: int(self.backend_load_counts[MODE_NATIONAL200]),
            MODE_MEDICAL: int(self.backend_load_counts[MODE_MEDICAL]),
        }

    def _backend(self, mode: str | None = None):
        selected = mode or self.current_mode
        if selected not in self.backends:
            self.switch_model(selected)
        return selected, self.backends[selected]

    @staticmethod
    def _complete_result(result: dict, extraction: dict, processing_time_ms: float, load_counts: dict):
        result["extraction"] = extraction
        result["processing_time_ms"] = float(processing_time_ms)
        result["model_load_counts"] = load_counts
        return result

    def predict_frame_list(self, frames: Iterable[np.ndarray], fps: float, mode: str | None = None) -> dict:
        started = time.perf_counter()
        with self._lock:
            self._ensure_open()
            if mode is not None and mode != self.current_mode:
                self.switch_model(mode)
            selected, backend = self._backend()
            features, extraction = self.processor.features_from_frames(
                frames,
                fps,
                sequence_length=backend.config.sequence_length,
            )
            result = backend.predict(features)
        return self._complete_result(
            result,
            extraction,
            (time.perf_counter() - started) * 1000.0,
            self.load_counts(),
        )

    def predict_video(self, video_path: Path, mode: str | None = None) -> dict:
        started = time.perf_counter()
        with self._lock:
            self._ensure_open()
            if mode is not None and mode != self.current_mode:
                self.switch_model(mode)
            selected, backend = self._backend()
            features, extraction = self.processor.features_from_video(
                video_path,
                sequence_length=backend.config.sequence_length,
            )
            result = backend.predict(features)
        return self._complete_result(
            result,
            extraction,
            (time.perf_counter() - started) * 1000.0,
            self.load_counts(),
        )

    def begin_recording(self) -> None:
        with self._lock:
            self._ensure_open()
            if self.recording_mode is not None:
                raise RuntimeError("A recording is already active.")
            self._backend()
            self.processor.begin_stream()
            self.recording_mode = self.current_mode

    def add_recording_frame(self, frame: np.ndarray, timestamp_seconds: float) -> None:
        with self._lock:
            self._ensure_open()
            if self.recording_mode is None:
                return
            self.processor.process_stream_frame(frame, timestamp_seconds)

    def finish_recording(self) -> dict:
        started = time.perf_counter()
        with self._lock:
            self._ensure_open()
            if self.recording_mode is None:
                raise RuntimeError("No recording is active.")
            mode = self.recording_mode
            backend = self.backends[mode]
            try:
                features, extraction = self.processor.finish_stream(
                    sequence_length=backend.config.sequence_length
                )
                result = backend.predict(features)
            finally:
                self.recording_mode = None
        return self._complete_result(
            result,
            extraction,
            (time.perf_counter() - started) * 1000.0,
            self.load_counts(),
        )

    def cancel_recording(self) -> None:
        with self._lock:
            self.processor.cancel_stream()
            self.recording_mode = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.processor.close()
            for backend in self.backends.values():
                backend.close()
            self._closed = True


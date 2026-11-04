"""Resident inference engine for the locked domain MVP medical mode."""

from __future__ import annotations

import math
import threading
import time
import zipfile
from itertools import chain
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

import infer_domain_mvp20 as infer
from prep_support import decode_zip_image


LOCKED_THRESHOLD = 0.40


class MedicalInferenceEngine:
    """Load all models once and reuse them for sequential medical inference."""

    def __init__(
        self,
        checkpoint: Path = infer.DEFAULT_CHECKPOINT,
        class_order: Path = infer.DEFAULT_CLASS_ORDER,
        medical_config: Path = infer.DEFAULT_MEDICAL_CONFIG,
        mediapipe_model: Path = infer.DEFAULT_MEDIAPIPE_MODEL,
        device: str = "cpu",
        top_k: int = 3,
        load_mediapipe: bool = True,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        started = time.perf_counter()
        self.checkpoint_path = infer.require_file(checkpoint, "Checkpoint")
        self.class_order_path = infer.require_file(class_order, "class_order.csv")
        self.medical_config_path = infer.require_file(
            medical_config, "medical_mode_config.json"
        )
        self.mediapipe_model_path = infer.require_file(
            mediapipe_model, "MediaPipe hand model"
        )
        self.device = infer.select_device(device)
        self.top_k = int(top_k)
        self._lock = threading.Lock()
        self._closed = False
        self._next_timestamp_ms = 0
        self._sequences_processed = 0

        self.model, self.classes, self.config, self.normalizer, self.checkpoint, self.medical_config = (
            infer.load_model_bundle(
                self.checkpoint_path,
                self.class_order_path,
                self.medical_config_path,
                self.device,
            )
        )
        self.threshold = float(self.medical_config["confidence_threshold"])
        if not math.isclose(self.threshold, LOCKED_THRESHOLD, abs_tol=1e-12):
            raise ValueError(
                f"Locked medical threshold must be {LOCKED_THRESHOLD:.2f}, got {self.threshold}."
            )

        self.cv2 = None
        self.mp = None
        self.landmarker = None
        if load_mediapipe:
            try:
                import cv2
                import mediapipe as mp
            except ImportError as error:
                raise RuntimeError(
                    "Medical inference requires opencv-python and mediapipe."
                ) from error
            self.cv2 = cv2
            self.mp = mp
            self.landmarker = infer.create_hand_landmarker(
                self.mediapipe_model_path
            )
        self.load_time_ms = (time.perf_counter() - started) * 1000.0

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MedicalInferenceEngine is closed.")

    def _require_landmarker(self) -> None:
        if self.landmarker is None:
            raise RuntimeError(
                "This engine was created without MediaPipe; use predict_features()."
            )

    def _observe_frames(
        self,
        frames: Iterable[np.ndarray],
        fps: float,
    ) -> list:
        frame_iterator = iter(frames)
        try:
            first_frame = next(frame_iterator)
        except StopIteration:
            return []

        # VIDEO mode keeps a tracker between calls. Empty boundary frames clear
        # the previous clip's ROI without recreating or reloading the landmarker.
        if self._sequences_processed:
            empty_frame = np.zeros_like(first_frame)
            for _ in range(3):
                rgb_frame = self.cv2.cvtColor(empty_frame, self.cv2.COLOR_BGR2RGB)
                mp_image = self.mp.Image(
                    image_format=self.mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb_frame),
                )
                self.landmarker.detect_for_video(mp_image, self._next_timestamp_ms)
                self._next_timestamp_ms += 33

        observations = []
        sequence_base_ms = self._next_timestamp_ms
        previous_timestamp_ms = sequence_base_ms - 1
        for frame_index, frame in enumerate(chain((first_frame,), frame_iterator)):
            rgb_frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(
                image_format=self.mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_frame),
            )
            local_timestamp_ms = int(round(frame_index * 1000.0 / fps))
            timestamp_ms = max(
                sequence_base_ms + local_timestamp_ms,
                previous_timestamp_ms + 1,
            )
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            observations.append(
                infer.result_to_observation(
                    result,
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                )
            )
            previous_timestamp_ms = timestamp_ms
            self._next_timestamp_ms = timestamp_ms + 1
        self._sequences_processed += 1
        return observations

    def _finalize_features(self, observations: list, fps: float) -> Tuple[np.ndarray, dict]:
        if not observations:
            raise ValueError("Input contains no decodable frames.")
        features, _, timestamp_source = infer.observations_to_v2_features(
            observations,
            fps=fps,
            target_length=self.config.sequence_length,
        )
        expected_shape = (self.config.sequence_length, self.config.input_dim)
        if features.shape != expected_shape:
            raise RuntimeError(
                f"Expected feature shape {expected_shape}, got {features.shape}."
            )
        if features.dtype != np.float32 or not np.isfinite(features).all():
            raise RuntimeError("Feature array must be finite float32 data.")
        left_ratio, right_ratio, any_ratio = infer.calculate_detection_ratios(
            observations
        )
        return features, {
            "decoded_frames": len(observations),
            "fps": float(fps),
            "timestamp_source": timestamp_source,
            "left_hand_detection": left_ratio,
            "right_hand_detection": right_ratio,
            "any_hand_detection": any_ratio,
        }

    def _extract_video_features(self, video_path: Path) -> Tuple[np.ndarray, dict]:
        capture = self.cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {video_path}")
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
            observations = self._observe_frames(decoded_frames(), fps)
        finally:
            capture.release()
        features, extraction = self._finalize_features(observations, fps)
        extraction["timestamp_source"] = "video_fps"
        return features, extraction

    def _extract_pics_zip_features(
        self,
        archive_path: Path,
        signer: str,
        gloss_id: str,
    ) -> Tuple[np.ndarray, dict]:
        gloss_id = infer.normalize_gloss_id(gloss_id)
        with zipfile.ZipFile(archive_path) as archive:
            entries = infer.find_pic_zip_entries(archive, signer, gloss_id)

            def decoded_frames():
                for entry_name in entries:
                    yield decode_zip_image(archive, entry_name)

            observations = self._observe_frames(decoded_frames(), 30.0)
        features, extraction = self._finalize_features(observations, 30.0)
        extraction.update(
            timestamp_source="zip_frame_order_30fps",
            zip_sequence=f"{signer}/front/{gloss_id}/",
        )
        return features, extraction

    def _predict(self, features: np.ndarray, source_type: str, source: Path, extraction=None) -> dict:
        result = infer.predict(
            self.model,
            self.normalizer,
            self.classes,
            self.medical_config,
            features,
            self.device,
            mode="medical",
            top_k=self.top_k,
            threshold=None,
        )
        result.update(
            source_type=source_type,
            source=str(source),
            extraction=extraction,
            engine_load_time_ms=float(self.load_time_ms),
        )
        return result

    def predict_video(self, video_path: Path) -> dict:
        source = infer.require_file(video_path, "Video")
        started = time.perf_counter()
        with self._lock:
            self._ensure_open()
            self._require_landmarker()
            features, extraction = self._extract_video_features(source)
            result = self._predict(features, "video", source, extraction)
        result["processing_time_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    def predict_pics_zip(self, archive_path: Path, signer: str, gloss_id: str) -> dict:
        source = infer.require_file(archive_path, "Pics ZIP")
        started = time.perf_counter()
        with self._lock:
            self._ensure_open()
            self._require_landmarker()
            features, extraction = self._extract_pics_zip_features(
                source, signer, gloss_id
            )
            result = self._predict(features, "pics_zip", source, extraction)
        result["processing_time_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    def predict_features(self, features: np.ndarray) -> dict:
        started = time.perf_counter()
        expected_shape = (self.config.sequence_length, self.config.input_dim)
        features = np.asarray(features, dtype=np.float32)
        if features.shape != expected_shape:
            raise ValueError(
                f"Expected feature shape {expected_shape}, got {features.shape}."
            )
        if not np.isfinite(features).all():
            raise ValueError("Features contain NaN or Inf.")
        with self._lock:
            self._ensure_open()
            result = self._predict(features, "memory", Path("<memory>"))
        result["processing_time_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    def predict_feature(self, feature_path: Path) -> dict:
        source = infer.require_file(feature_path, "Feature array")
        started = time.perf_counter()
        with self._lock:
            self._ensure_open()
            features = infer.load_v2_sequence(source, config=self.config)
            result = self._predict(features, "feature", source)
        result["processing_time_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self.landmarker is not None:
                self.landmarker.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

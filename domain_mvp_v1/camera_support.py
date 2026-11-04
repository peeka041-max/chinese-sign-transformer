"""OpenCV camera discovery and validated frame capture for the Tk demo."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


BACKENDS = (
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("ANY", cv2.CAP_ANY),
)
WARMUP_FRAMES = 10


def validate_bgr_frame(frame: np.ndarray) -> None:
    if frame is None:
        raise ValueError("Camera returned no frame.")
    if frame.ndim != 3:
        raise ValueError(f"Expected frame.ndim == 3, got {frame.ndim}.")
    if frame.shape[2] != 3:
        raise ValueError(
            f"Expected frame.shape[2] == 3, got {frame.shape[2]}."
        )
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected frame.dtype == uint8, got {frame.dtype}.")


def frame_statistics(frame: np.ndarray) -> dict:
    validate_bgr_frame(frame)
    return {
        "frame_shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "min": float(frame.min()),
        "max": float(frame.max()),
        "mean": float(frame.mean()),
        "std": float(frame.std()),
        "resolution": [int(frame.shape[1]), int(frame.shape[0])],
    }


def diagnose_frame_sequence(frames: list[np.ndarray], read_success: bool) -> dict:
    frame = frames[-1]
    stats = frame_statistics(frame)
    temporal_diff = (
        float(
            np.mean(
                [
                    cv2.absdiff(previous, current).mean()
                    for previous, current in zip(frames[:-1], frames[1:])
                ]
            )
        )
        if len(frames) > 1
        else 0.0
    )
    near_black = stats["max"] < 24.0 or (
        stats["mean"] < 8.0 and stats["std"] < 5.0
    )
    static_placeholder = temporal_diff < 0.15
    suspected_infrared = stats["mean"] < 30.0 and stats["std"] < 12.0
    valid_frame = bool(read_success and not near_black)
    return {
        **stats,
        "near_black": near_black,
        "temporal_mean_abs_diff": temporal_diff,
        "static_placeholder": static_placeholder,
        "suspected_infrared": suspected_infrared,
        "valid_frame": valid_frame,
        "recommended_color_preview": bool(
            valid_frame and not static_placeholder and not suspected_infrared
        ),
    }


def diagnosis_warning(diagnosis: dict) -> str:
    if diagnosis.get("static_placeholder"):
        return "静态/虚拟占位"
    if diagnosis.get("suspected_infrared"):
        return "疑似红外/过暗"
    if diagnosis.get("recommended_color_preview_raw") and not diagnosis.get(
        "recommended_color_preview"
    ):
        return "画面尚未稳定"
    return ""


def _open_capture(index: int, backend_id: int):
    capture = cv2.VideoCapture(int(index), int(backend_id))
    if capture.isOpened():
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 30)
    return capture


def probe_camera(index: int, backend_name: str, backend_id: int) -> dict:
    started = time.perf_counter()
    capture = _open_capture(index, backend_id)
    opened = bool(capture.isOpened())
    frames = []
    read_results = []
    reported_backend = ""
    if opened:
        try:
            reported_backend = capture.getBackendName()
        except cv2.error:
            reported_backend = ""
        for _ in range(WARMUP_FRAMES):
            ok, frame = capture.read()
            frame_ok = bool(ok and frame is not None)
            read_results.append(frame_ok)
            if frame_ok:
                try:
                    validate_bgr_frame(frame)
                except ValueError:
                    frame_ok = False
                    read_results[-1] = False
                else:
                    frames.append(frame)
    capture.release()

    result = {
        "index": int(index),
        "backend": backend_name,
        "backend_id": int(backend_id),
        "reported_backend": reported_backend,
        "opened": opened,
        "read_success": len(read_results) == WARMUP_FRAMES
        and all(read_results),
        "successful_reads": sum(read_results),
        "frame_shape": None,
        "dtype": None,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "resolution": None,
        "near_black": None,
        "temporal_mean_abs_diff": None,
        "static_placeholder": None,
        "suspected_infrared": None,
        "valid_frame": False,
        "recommended_color_preview": False,
        "diagnostic_image": None,
        "elapsed_ms": None,
    }
    if frames:
        frame = frames[-1]
        diagnosis = diagnose_frame_sequence(frames, result["read_success"])
        diagnostic_dir = (
            Path(tempfile.gettempdir()) / "realtime_sign_camera_diagnostics"
        )
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        diagnostic_path = (
            diagnostic_dir / f"camera_{index}_{backend_name.lower()}.jpg"
        )
        if result["read_success"] and cv2.imwrite(str(diagnostic_path), frame):
            result["diagnostic_image"] = str(diagnostic_path)
        result.update(diagnosis)
    result["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
    print(json.dumps({"event": "camera_probe", **result}, ensure_ascii=False), flush=True)
    return result


@dataclass(frozen=True)
class CameraCandidate:
    index: int
    backend_name: str
    backend_id: int
    diagnosis: dict

    @property
    def warning(self) -> str:
        return diagnosis_warning(self.diagnosis)

    @property
    def label(self) -> str:
        suffix = f"（{self.warning}）" if self.warning else ""
        return f"摄像头 {self.index} · {self.backend_name}{suffix}"

    @property
    def recommended(self) -> bool:
        return bool(self.diagnosis.get("recommended_color_preview"))


def discover_cameras(max_index: int = 4):
    attempts = []
    candidates = []
    for index in range(max_index + 1):
        for backend_name, backend_id in BACKENDS:
            result = probe_camera(index, backend_name, backend_id)
            attempts.append(result)
            if result["valid_frame"]:
                candidates.append(
                    CameraCandidate(index, backend_name, backend_id, result)
                )
    return candidates, attempts


class CameraStream:
    def __init__(self, engine, candidate: CameraCandidate):
        self.engine = engine
        self.candidate = candidate
        self.capture = None
        self.thread = None
        self._running = threading.Event()
        self._recording = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame = None
        self._error = None
        self._capture_times = deque(maxlen=120)
        self._diagnostic_frames = deque(maxlen=WARMUP_FRAMES)
        self._frames_since_diagnosis = 0
        self._recommended_streak = 0
        self._recording_started = None
        self.initial_statistics = None
        self.runtime_diagnosis = None

    @property
    def runtime_warning(self) -> str:
        with self._lock:
            diagnosis = dict(self.runtime_diagnosis or {})
        return diagnosis_warning(diagnosis)

    @property
    def runtime_recommended(self) -> bool:
        with self._lock:
            diagnosis = dict(self.runtime_diagnosis or {})
        return bool(diagnosis.get("recommended_color_preview"))

    def runtime_diagnosis_snapshot(self) -> dict:
        with self._lock:
            return dict(self.runtime_diagnosis or {})

    def start(self) -> None:
        capture = _open_capture(
            self.candidate.index, self.candidate.backend_id
        )
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("摄像头读取失败，请切换设备")
        warm_frames = []
        for _ in range(WARMUP_FRAMES):
            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                raise RuntimeError("摄像头读取失败，请切换设备")
            validate_bgr_frame(frame)
            warm_frames.append(frame)
        last_frame = warm_frames[-1]
        diagnosis = diagnose_frame_sequence(warm_frames, True)
        raw_recommended = bool(diagnosis["recommended_color_preview"])
        self._recommended_streak = 1 if raw_recommended else 0
        diagnosis["recommended_color_preview_raw"] = raw_recommended
        diagnosis["recommended_streak"] = self._recommended_streak
        diagnosis["recommended_color_preview"] = False
        if diagnosis["near_black"]:
            capture.release()
            raise RuntimeError("摄像头读取失败，请切换设备")
        self.capture = capture
        self.initial_statistics = frame_statistics(last_frame)
        with self._lock:
            self.runtime_diagnosis = diagnosis
            self._diagnostic_frames.extend(frame.copy() for frame in warm_frames)
            self._latest_frame = last_frame.copy()
        self._running.set()
        self.thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture",
            daemon=True,
        )
        self.thread.start()

    def _capture_loop(self) -> None:
        while self._running.is_set():
            ok, frame = self.capture.read()
            now = time.perf_counter()
            if not ok or frame is None:
                with self._lock:
                    self._latest_frame = None
                    self._error = "摄像头读取失败，请切换设备"
                break
            try:
                validate_bgr_frame(frame)
            except ValueError:
                with self._lock:
                    self._latest_frame = None
                    self._error = "摄像头读取失败，请切换设备"
                break
            inference_frame = frame
            diagnosis_event = None
            diagnosis_near_black = False
            with self._lock:
                self._latest_frame = frame.copy()
                self._capture_times.append(now)
                self._diagnostic_frames.append(frame.copy())
                self._frames_since_diagnosis += 1
                if self._frames_since_diagnosis >= WARMUP_FRAMES:
                    previous_signature = (
                        diagnosis_warning(self.runtime_diagnosis or {}),
                        bool(
                            (self.runtime_diagnosis or {}).get(
                                "recommended_color_preview"
                            )
                        ),
                    )
                    diagnosis = diagnose_frame_sequence(
                        list(self._diagnostic_frames), True
                    )
                    raw_recommended = bool(
                        diagnosis["recommended_color_preview"]
                    )
                    self._recommended_streak = (
                        self._recommended_streak + 1
                        if raw_recommended
                        else 0
                    )
                    diagnosis["recommended_color_preview_raw"] = raw_recommended
                    diagnosis["recommended_streak"] = self._recommended_streak
                    diagnosis["recommended_color_preview"] = bool(
                        raw_recommended and self._recommended_streak >= 2
                    )
                    self.runtime_diagnosis = diagnosis
                    self._frames_since_diagnosis = 0
                    current_signature = (
                        diagnosis_warning(diagnosis),
                        bool(diagnosis["recommended_color_preview"]),
                    )
                    if current_signature != previous_signature:
                        diagnosis_event = dict(diagnosis)
                    if diagnosis["near_black"]:
                        self._latest_frame = None
                        self._error = "摄像头读取失败，请切换设备"
                        diagnosis_near_black = True
            if diagnosis_event is not None:
                print(
                    json.dumps(
                        {
                            "event": "camera_stream_diagnosis",
                            "index": self.candidate.index,
                            "backend": self.candidate.backend_name,
                            "warning": diagnosis_warning(diagnosis_event) or None,
                            **diagnosis_event,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if diagnosis_near_black:
                break
            if self._recording.is_set() and self.engine is not None:
                try:
                    self.engine.add_recording_frame(
                        inference_frame,
                        now - float(self._recording_started),
                    )
                except Exception as error:
                    with self._lock:
                        self._latest_frame = None
                        self._error = str(error)
                    self._recording.clear()
                    break
        self._running.clear()

    def begin_recording(self) -> None:
        if self.engine is None:
            raise RuntimeError("预览模式未加载Transformer。")
        if not self._running.is_set():
            raise RuntimeError("摄像头读取失败，请切换设备")
        self.engine.begin_recording()
        self._recording_started = time.perf_counter()
        self._recording.set()

    def end_recording(self) -> None:
        self._recording.clear()

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def actual_fps(self) -> float:
        with self._lock:
            if len(self._capture_times) < 2:
                return 0.0
            duration = self._capture_times[-1] - self._capture_times[0]
            if duration <= 0.0:
                return 0.0
            return (len(self._capture_times) - 1) / duration

    def pop_error(self):
        with self._lock:
            error = self._error
            self._error = None
            return error

    def close(self) -> None:
        self._recording.clear()
        self._running.clear()
        if self.capture is not None:
            self.capture.release()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        with self._lock:
            self._latest_frame = None

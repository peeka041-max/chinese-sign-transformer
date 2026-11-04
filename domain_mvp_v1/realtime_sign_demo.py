"""Interactive isolated-sign recognition from a live camera or video file."""

from __future__ import annotations

import argparse
import json
import time
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from camera_support import (
    CameraCandidate,
    CameraStream,
    discover_cameras,
    validate_bgr_frame,
)
from realtime_sign_engine import (
    MODE_MEDICAL,
    MODE_NATIONAL200,
    MODEL_LABELS,
    RealtimeSignEngine,
)


BG = "#F7F9F8"
SURFACE = "#FFFFFF"
TEXT = "#20302C"
MUTED = "#687771"
PRIMARY = "#2F6B62"
PRIMARY_ACTIVE = "#275C54"
SOFT = "#E8F1EE"
BORDER = "#D7E0DD"
ERROR = "#A34848"


class RealtimeSignDemo:
    def __init__(self, root: tk.Tk, preview_only: bool = False):
        self.root = root
        self.root.title("Transformer手语实时识别")
        self.root.geometry("900x790")
        self.root.minsize(760, 690)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sign-model"
        )
        self.preview_only = bool(preview_only)
        self.engine = None
        self.camera = None
        self.camera_candidates: list[CameraCandidate] = []
        self.camera_candidate_by_label: dict[str, CameraCandidate] = {}
        self.camera_attempts: list[dict] = []
        self.selected_camera_candidate: CameraCandidate | None = None
        self.pending_future: Future | None = None
        self.pending_kind = None
        self.stop_clicked_at = None
        self.closed = False

        self.model_label = tk.StringVar(value=MODEL_LABELS[MODE_NATIONAL200])
        self.camera_label = tk.StringVar(value="正在探测摄像头...")
        self.status = tk.StringVar(value="当前状态：等待（正在探测摄像头）")
        self.video_path = tk.StringVar(value="未选择已有视频")
        self.result = tk.StringVar(value="等待输入")
        self.confidence = tk.StringVar(value="置信度：--")
        self.top3 = tk.StringVar(value="Top-3：--")
        self.processing_time = tk.StringVar(value="处理耗时：--")
        self.camera_fps = tk.StringVar(value="摄像头实际 FPS：--")

        self._configure_styles()
        self._build_layout()
        self.pending_kind = "camera_probe_startup"
        self.pending_future = self.executor.submit(discover_cameras, 4)
        self.root.after(80, self._poll_future)
        self.root.after(33, self._refresh_preview)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=BG)
        style.configure("Surface.TFrame", background=SURFACE)
        style.configure(
            "Title.TLabel",
            background=BG,
            foreground=TEXT,
            font=("Microsoft YaHei UI", 19, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background=BG,
            foreground=TEXT,
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "Muted.TLabel",
            background=BG,
            foreground=MUTED,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Result.TLabel",
            background=SOFT,
            foreground=TEXT,
            font=("Microsoft YaHei UI", 17, "bold"),
            padding=(18, 18),
        )
        style.configure(
            "Primary.TButton",
            background=PRIMARY,
            foreground="#FFFFFF",
            borderwidth=0,
            padding=(18, 11),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", PRIMARY_ACTIVE), ("disabled", "#ADB9B5")],
        )
        style.configure(
            "Secondary.TButton",
            background=SURFACE,
            foreground=TEXT,
            bordercolor=BORDER,
            borderwidth=1,
            padding=(15, 10),
            font=("Microsoft YaHei UI", 10),
        )
        style.map("Secondary.TButton", background=[("active", SOFT)])
        style.configure(
            "Model.TCombobox",
            fieldbackground=SURFACE,
            background=SURFACE,
            foreground=TEXT,
            bordercolor=BORDER,
            padding=8,
            font=("Microsoft YaHei UI", 10),
        )

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, style="App.TFrame", padding=(28, 22))
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)

        header = ttk.Frame(container, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header, text="Transformer手语实时识别", style="Title.TLabel"
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status, style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(3, 0)
        )
        self.model_combo = ttk.Combobox(
            header,
            textvariable=self.model_label,
            values=list(MODEL_LABELS.values()),
            state="disabled",
            width=30,
            style="Model.TCombobox",
        )
        self.model_combo.grid(row=0, column=1, rowspan=2, sticky="e")
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)

        camera_row = ttk.Frame(container, style="App.TFrame")
        camera_row.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        camera_row.columnconfigure(1, weight=1)
        ttk.Label(camera_row, text="摄像头选择", style="Body.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )
        self.camera_combo = ttk.Combobox(
            camera_row,
            textvariable=self.camera_label,
            values=(),
            state="disabled",
            style="Model.TCombobox",
        )
        self.camera_combo.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.camera_combo.bind(
            "<<ComboboxSelected>>", self._on_camera_selected
        )
        self.refresh_camera_button = ttk.Button(
            camera_row,
            text="刷新摄像头",
            style="Secondary.TButton",
            command=self.refresh_cameras,
            state="disabled",
        )
        self.refresh_camera_button.grid(row=0, column=2, sticky="e")

        preview_shell = tk.Frame(
            container,
            bg="#E4E9E7",
            highlightbackground=BORDER,
            highlightthickness=+1,
            width=800,
            height=450,
        )
        preview_shell.grid(row=2, column=0, sticky="nsew", pady=(12, 14))
        preview_shell.grid_propagate(False)
        preview_shell.rowconfigure(0, weight=1)
        preview_shell.columnconfigure(0, weight=1)
        self.preview = tk.Label(
            preview_shell,
            text="正在打开摄像头...",
            bg="#E4E9E7",
            fg=MUTED,
            font=("Microsoft YaHei UI", 11),
        )
        self.preview.grid(row=0, column=0, sticky="nsew")

        actions = ttk.Frame(container, style="App.TFrame")
        actions.grid(row=3, column=0, sticky="ew")
        actions.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(
            actions,
            text="开始录制手语",
            style="Primary.TButton",
            command=self.start_recording,
            state="disabled",
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.finish_button = ttk.Button(
            actions,
            text="结束并识别",
            style="Secondary.TButton",
            command=self.finish_recording,
            state="disabled",
        )
        self.finish_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        file_row = ttk.Frame(container, style="App.TFrame")
        file_row.grid(row=4, column=0, sticky="ew", pady=(10, 16))
        file_row.columnconfigure(1, weight=1)
        self.file_button = ttk.Button(
            file_row,
            text="选择已有视频",
            style="Secondary.TButton",
            command=self.choose_video,
            state="disabled",
        )
        self.file_button.grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Label(
            file_row,
            textvariable=self.video_path,
            style="Muted.TLabel",
            anchor="w",
        ).grid(row=0, column=1, sticky="ew")

        ttk.Label(
            container,
            textvariable=self.result,
            style="Result.TLabel",
            anchor="center",
        ).grid(row=5, column=0, sticky="ew")

        metrics = ttk.Frame(container, style="App.TFrame")
        metrics.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        metrics.columnconfigure(0, weight=1)
        ttk.Label(metrics, textvariable=self.confidence, style="Body.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(metrics, textvariable=self.top3, style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )
        ttk.Label(
            metrics, textvariable=self.processing_time, style="Muted.TLabel"
        ).grid(row=0, column=1, sticky="e")
        ttk.Label(metrics, textvariable=self.camera_fps, style="Muted.TLabel").grid(
            row=1, column=1, sticky="e", pady=(4, 0)
        )

    @staticmethod
    def _mode_from_label(label: str) -> str:
        for mode, value in MODEL_LABELS.items():
            if value == label:
                return mode
        raise ValueError(f"未知模型：{label}")

    def _set_idle_controls(self, enabled: bool) -> None:
        model_ready = enabled and self.engine is not None and not self.preview_only
        camera_ready = (
            model_ready
            and self.camera is not None
            and self.camera.runtime_recommended
        )
        self.start_button.configure(state="normal" if camera_ready else "disabled")
        self.file_button.configure(state="normal" if model_ready else "disabled")
        self.model_combo.configure(
            state="readonly" if model_ready else "disabled"
        )
        self.camera_combo.configure(
            state="readonly"
            if enabled and self.camera_candidates
            else "disabled"
        )
        self.refresh_camera_button.configure(
            state="normal" if enabled else "disabled"
        )

    def _clear_preview(self, message: str) -> None:
        self.preview.configure(image="", text=message)
        self.preview.image = None

    def refresh_cameras(self) -> None:
        if self.pending_future is not None:
            return
        if self.camera is not None:
            self.camera.close()
            self.camera = None
        self.selected_camera_candidate = None
        self.camera_candidates = []
        self.camera_candidate_by_label = {}
        self.camera_combo.configure(values=(), state="disabled")
        self.camera_label.set("正在探测摄像头...")
        self.camera_fps.set("摄像头实际 FPS：--")
        self.status.set("当前状态：等待（正在探测摄像头）")
        self._clear_preview("正在探测摄像头...")
        self._set_idle_controls(False)
        self.pending_kind = "camera_probe_refresh"
        self.pending_future = self.executor.submit(discover_cameras, 4)

    def _on_camera_selected(self, _event=None) -> None:
        if self.pending_future is not None:
            return
        candidate = self.camera_candidate_by_label.get(self.camera_label.get())
        if candidate is None:
            return
        self._start_camera(candidate)

    def _handle_camera_probe(self, value) -> None:
        candidates, attempts = value
        self.camera_candidates = list(candidates)
        self.camera_attempts = list(attempts)
        self.camera_candidate_by_label = {
            candidate.label: candidate for candidate in self.camera_candidates
        }
        labels = list(self.camera_candidate_by_label)
        self.camera_combo.configure(values=labels)
        if not candidates:
            self.camera_label.set("未检测到可读取摄像头")
            self.status.set("当前状态：等待")
            self.result.set("摄像头不可用，可选择已有视频")
            self._clear_preview("摄像头读取失败，请切换设备")
            self.selected_camera_candidate = None
        else:
            candidate = next(
                (item for item in candidates if item.recommended), candidates[0]
            )
            self.selected_camera_candidate = candidate
            self.camera_label.set(candidate.label)

        if self.preview_only:
            if self.selected_camera_candidate is not None:
                self._start_camera(self.selected_camera_candidate)
            else:
                self._set_idle_controls(True)
            return
        if self.engine is None:
            self.status.set("当前状态：等待（正在加载模型）")
            self.pending_kind = "startup"
            self.pending_future = self.executor.submit(
                RealtimeSignEngine,
                default_mode=MODE_NATIONAL200,
                device="cpu",
                top_k=3,
            )
        elif self.selected_camera_candidate is not None:
            self._start_camera(self.selected_camera_candidate)
        else:
            self._set_idle_controls(True)

    def _start_busy(self, text: str = "当前状态：识别中") -> None:
        self.status.set(text)
        self._set_idle_controls(False)
        self.finish_button.configure(state="disabled")

    def _on_model_selected(self, _event=None) -> None:
        if self.engine is None or self.pending_future is not None:
            return
        mode = self._mode_from_label(self.model_label.get())
        self._set_idle_controls(False)
        self.status.set("当前状态：等待（正在加载模型）")
        self.pending_kind = "switch_model"
        self.pending_future = self.executor.submit(self.engine.switch_model, mode)

    def start_recording(self) -> None:
        if self.camera is None or self.pending_future is not None:
            return
        if (
            self.camera is None
            or not self.camera.runtime_recommended
        ):
            messagebox.showwarning(
                "摄像头不可用", "请先切换到正常彩色摄像头。"
            )
            return
        try:
            self.camera.begin_recording()
        except Exception as error:
            messagebox.showerror("无法录制", str(error))
            return
        self.status.set("当前状态：录制中")
        self.result.set("正在录制手语")
        self._set_idle_controls(False)
        self.finish_button.configure(state="normal")

    def finish_recording(self) -> None:
        if self.camera is None:
            return
        self.camera.end_recording()
        self.stop_clicked_at = time.perf_counter()
        self._show_processing_state()
        self.pending_kind = "camera_inference"
        self.pending_future = self.executor.submit(self.engine.finish_recording)

    def _show_processing_state(self) -> None:
        self._set_idle_controls(False)
        self.finish_button.configure(state="disabled")
        self.status.set("当前状态：识别中")
        self.result.set("正在识别")
        self.confidence.set("置信度：--")
        self.top3.set("Top-3：--")
        self.processing_time.set("处理耗时：--")

    def choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="选择已有手语视频",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        self.video_path.set(path)
        self.stop_clicked_at = time.perf_counter()
        self._show_processing_state()
        self.pending_kind = "video_inference"
        self.pending_future = self.executor.submit(
            self.engine.predict_video, Path(path)
        )

    def _poll_future(self) -> None:
        if self.closed:
            return
        if self.pending_future is None or not self.pending_future.done():
            self.root.after(80, self._poll_future)
            return
        future = self.pending_future
        kind = self.pending_kind
        self.pending_future = None
        self.pending_kind = None
        try:
            value = future.result()
        except Exception as error:
            self.status.set("当前状态：等待")
            self.result.set("处理失败")
            if kind and kind.startswith("camera_probe"):
                self.camera_label.set("摄像头探测失败")
                self._clear_preview("摄像头读取失败，请切换设备")
            self._set_idle_controls(self.engine is not None)
            messagebox.showerror("处理失败", str(error))
        else:
            if kind and kind.startswith("camera_probe"):
                self._handle_camera_probe(value)
            elif kind == "startup":
                self.engine = value
                if self.selected_camera_candidate is not None:
                    self._start_camera(self.selected_camera_candidate)
                else:
                    self.status.set("当前状态：等待")
                    self.result.set("摄像头不可用，可选择已有视频")
                    self._set_idle_controls(True)
            elif kind == "switch_model":
                self.status.set("当前状态：等待")
                self.result.set("等待输入")
                self._set_idle_controls(True)
            else:
                display_latency_ms = (
                    (time.perf_counter() - float(self.stop_clicked_at)) * 1000.0
                    if self.stop_clicked_at is not None
                    else float(value["processing_time_ms"])
                )
                self._show_result(value, display_latency_ms, kind)
        self.root.after(80, self._poll_future)

    def _start_camera(self, candidate: CameraCandidate) -> None:
        if self.camera is not None:
            self.camera.close()
            self.camera = None
        self.selected_camera_candidate = candidate
        self.camera_label.set(candidate.label)
        self.camera_fps.set("摄像头实际 FPS：--")
        self._clear_preview("正在打开摄像头...")
        try:
            self.camera = CameraStream(self.engine, candidate)
            self.camera.start()
        except Exception as error:
            self.camera = None
            self.status.set("当前状态：等待")
            self.result.set("摄像头不可用，可选择已有视频")
            self._clear_preview("摄像头读取失败，请切换设备")
            self._set_idle_controls(True)
            messagebox.showwarning("摄像头不可用", str(error))
            return
        self.status.set("当前状态：等待")
        runtime_warning = self.camera.runtime_warning
        if self.camera.runtime_recommended:
            self.result.set("等待输入")
        else:
            self.result.set(f"当前设备{runtime_warning}，请切换彩色摄像头")
        self._set_idle_controls(True)
        print(
            json.dumps(
                {
                    "event": "camera_selected",
                    "index": candidate.index,
                    "backend": candidate.backend_name,
                    "probe_warning": candidate.warning or None,
                    "runtime_warning": runtime_warning or None,
                    "recommended_color_preview": self.camera.runtime_recommended,
                    "runtime_diagnosis": self.camera.runtime_diagnosis_snapshot(),
                    **(self.camera.initial_statistics or {}),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def _show_result(self, value: dict, display_latency_ms: float, source_kind: str) -> None:
        if value["mode"] == MODE_MEDICAL and not value["accepted"]:
            self.result.set("未识别，请重试")
        else:
            self.result.set(f"识别结果：{value['prediction']['display_word']}")
        self.confidence.set(f"置信度：{float(value['confidence']):.2%}")
        top_text = "  ·  ".join(
            f"{item['rank']}. {item['display_word']} {float(item['confidence']):.2%}"
            for item in value["top_predictions"]
        )
        self.top3.set(f"Top-3：{top_text}")
        self.processing_time.set(f"处理耗时：{display_latency_ms / 1000.0:.2f} 秒")
        extraction = value.get("extraction") or {}
        actual_fps = float(extraction.get("actual_fps", 0.0))
        if actual_fps > 0.0:
            self.camera_fps.set(f"摄像头实际 FPS：{actual_fps:.2f}")
        self.status.set("当前状态：完成")
        self._set_idle_controls(True)
        print(
            json.dumps(
                {
                    "event": "recognition_complete",
                    "source": source_kind,
                    "mode": value["mode"],
                    "top1": value["top_predictions"][0]["display_word"],
                    "confidence": float(value["confidence"]),
                    "accepted": bool(value["accepted"]),
                    "feature_shape": value["feature_shape"],
                    "actual_fps": actual_fps,
                    "stop_to_display_ms": display_latency_ms,
                    "model_load_counts": value["model_load_counts"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def _refresh_preview(self) -> None:
        if self.closed:
            return
        if self.camera is not None:
            error = self.camera.pop_error()
            if error:
                self.camera.close()
                self.camera = None
                self.result.set("摄像头读取失败，请切换设备")
                self.status.set("当前状态：等待")
                self.camera_fps.set("摄像头实际 FPS：--")
                self._clear_preview("摄像头读取失败，请切换设备")
                self._set_idle_controls(True)
            frame = self.camera.latest_frame() if self.camera is not None else None
            if frame is not None:
                try:
                    validate_bgr_frame(frame)
                except ValueError:
                    self.camera.close()
                    self.camera = None
                    self.result.set("摄像头读取失败，请切换设备")
                    self.status.set("当前状态：等待")
                    self.camera_fps.set("摄像头实际 FPS：--")
                    self._clear_preview("摄像头读取失败，请切换设备")
                    self._set_idle_controls(True)
                    self.root.after(33, self._refresh_preview)
                    return
                display_frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                image.thumbnail((800, 450), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image=image)
                self.preview.configure(image=photo, text="")
                self.preview.image = photo
                preview_fps = self.camera.actual_fps()
                if preview_fps > 0.0 and "完成" not in self.status.get():
                    self.camera_fps.set(f"摄像头实际 FPS：{preview_fps:.2f}")
                if (
                    self.pending_future is None
                    and "等待" in self.status.get()
                    and not self.camera.runtime_recommended
                ):
                    warning = self.camera.runtime_warning
                    self.result.set(f"当前设备{warning}，请切换彩色摄像头")
                    self._set_idle_controls(True)
        self.root.after(33, self._refresh_preview)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.camera is not None:
            self.camera.close()
        if self.engine is not None:
            try:
                self.engine.cancel_recording()
            except Exception:
                pass
            self.executor.submit(self.engine.close)
        self.executor.shutdown(wait=False, cancel_futures=False)
        self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transformer手语实时识别摄像头演示"
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="只验证原始摄像头预览，不加载Transformer模型",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    RealtimeSignDemo(root, preview_only=args.preview_only)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Main window — 3-panel QSplitter layout with menu bar.

Layout:
    ┌──────────┬──────────────────────┬──────────────┐
    │  Clips   │    Preview           │  Parameters  │
    │  Browser │    Viewport          │    Panel     │
    │ (220px)  │    (fills)           │  (280px)     │
    ├──────────┴──────────────────────┴──────────────┤
    │  Queue Panel (collapsible, per-job progress)   │
    ├────────────────────────────────────────────────┤
    │  Status Bar (progress, VRAM, GPU, run/stop)    │
    └────────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
import logging
import os
import tempfile

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QSplitter, QVBoxLayout, QWidget,
    QLabel, QHBoxLayout, QMessageBox,
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QShortcut, QKeySequence, QAction, QImage

from backend import (
    CorridorKeyService, ClipEntry, ClipState, InferenceParams,
    OutputConfig, JobType, JobStatus,
)
from backend.errors import CorridorKeyError
from backend.job_queue import GPUJob

from ui.models.clip_model import ClipListModel
from ui.widgets.clip_browser import ClipBrowser
from ui.widgets.preview_viewport import PreviewViewport
from ui.widgets.parameter_panel import ParameterPanel
from ui.widgets.status_bar import StatusBar
from ui.widgets.queue_panel import QueuePanel
from ui.workers.gpu_job_worker import GPUJobWorker, create_job_snapshot
from ui.workers.gpu_monitor import GPUMonitor
from ui.workers.thumbnail_worker import ThumbnailGenerator

logger = logging.getLogger(__name__)

# Session file stored in clips dir (Codex: JSON sidecar)
_SESSION_FILENAME = ".corridorkey_session.json"
_SESSION_VERSION = 1


class MainWindow(QMainWindow):
    """CorridorKey main application window."""

    def __init__(self, service: CorridorKeyService | None = None):
        super().__init__()
        self.setWindowTitle("CORRIDORKEY")
        self.setMinimumSize(1100, 650)
        self.resize(1400, 800)

        self._service = service or CorridorKeyService()
        self._current_clip: ClipEntry | None = None
        self._clips_dir: str | None = None
        # Track active job ID — set only once when job starts, not on every progress
        self._active_job_id: str | None = None

        # Data model
        self._clip_model = ClipListModel()

        # Thumbnail generator (background, Codex: no QPixmap off main thread)
        self._thumb_gen = ThumbnailGenerator(self)
        self._thumb_gen.thumbnail_ready.connect(self._clip_model.set_thumbnail)

        # Reprocess debounce timer (200ms, Codex: coalesce stale requests)
        self._reprocess_timer = QTimer(self)
        self._reprocess_timer.setSingleShot(True)
        self._reprocess_timer.setInterval(200)
        self._reprocess_timer.timeout.connect(self._do_reprocess)

        # Build UI
        self._build_menu_bar()
        self._build_central()
        self._build_status_bar()
        self._setup_shortcuts()

        # Workers
        self._gpu_worker = GPUJobWorker(self._service, parent=self)
        self._gpu_monitor = GPUMonitor(interval_ms=2000, parent=self)

        # Connect signals
        self._connect_signals()

        # Start GPU monitoring
        self._gpu_monitor.start()

        # Detect device
        device = self._service.detect_device()
        logger.info(f"Compute device: {device}")

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction("Open Clips Folder...", self._on_open_folder)
        file_menu.addSeparator()

        # Session save/load
        save_action = file_menu.addAction("Save Session", self._on_save_session)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        load_action = file_menu.addAction("Load Session...", self._on_load_session)
        load_action.setShortcut(QKeySequence("Ctrl+O"))

        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # View menu
        view_menu = menu_bar.addMenu("View")
        view_menu.addAction("Reset Layout", self._reset_layout)
        view_menu.addAction("Toggle Queue Panel", self._toggle_queue_panel)

        # Split view toggle (checkable)
        self._split_action = QAction("Split View", self)
        self._split_action.setCheckable(True)
        self._split_action.setShortcut(QKeySequence("Ctrl+D"))
        self._split_action.triggered.connect(self._on_toggle_split)
        view_menu.addAction(self._split_action)

        view_menu.addAction("Reset Zoom", self._on_reset_zoom)

        # Help menu
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("About", self._show_about)

    def _build_central(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top bar with brand mark
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(12, 6, 12, 6)

        brand = QLabel("CORRIDORKEY")
        brand.setObjectName("brandMark")
        top_bar.addWidget(brand)
        top_bar.addStretch()

        main_layout.addLayout(top_bar)

        # 3-panel splitter
        self._splitter = QSplitter(Qt.Horizontal)

        # Left — Clip Browser
        self._clip_browser = ClipBrowser(self._clip_model)
        self._splitter.addWidget(self._clip_browser)

        # Center — Preview Viewport
        self._preview = PreviewViewport()
        self._splitter.addWidget(self._preview)

        # Right — Parameter Panel
        self._param_panel = ParameterPanel()
        self._splitter.addWidget(self._param_panel)

        # Set initial sizes (220, fill, 280)
        self._splitter.setSizes([220, 700, 280])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)

        main_layout.addWidget(self._splitter, 1)

        # Queue panel (collapsible, above status bar)
        self._queue_panel = QueuePanel(self._service.job_queue)
        self._queue_panel.hide()  # hidden by default, shown when jobs are queued
        main_layout.addWidget(self._queue_panel)

    def _build_status_bar(self) -> None:
        self._status_bar = StatusBar()
        self.centralWidget().layout().addWidget(self._status_bar)

    def _setup_shortcuts(self) -> None:
        """Keyboard shortcuts."""
        # Escape — stop/cancel
        QShortcut(QKeySequence(Qt.Key_Escape), self, self._on_stop_inference)
        # Ctrl+R — run inference on selected clip
        QShortcut(QKeySequence("Ctrl+R"), self, self._on_run_inference)
        # Ctrl+Shift+R — run all ready clips
        QShortcut(QKeySequence("Ctrl+Shift+R"), self, self._on_run_all_ready)

    def _connect_signals(self) -> None:
        # Clip browser
        self._clip_browser.clip_selected.connect(self._on_clip_selected)
        self._clip_browser.clips_dir_changed.connect(self._on_clips_dir_changed)

        # Status bar buttons
        self._status_bar.run_clicked.connect(self._on_run_inference)
        self._status_bar.stop_clicked.connect(self._on_stop_inference)

        # GPU worker signals
        self._gpu_worker.progress.connect(self._on_worker_progress)
        self._gpu_worker.preview_ready.connect(self._on_worker_preview)
        self._gpu_worker.clip_finished.connect(self._on_worker_clip_finished)
        self._gpu_worker.warning.connect(self._on_worker_warning)
        self._gpu_worker.error.connect(self._on_worker_error)
        self._gpu_worker.queue_empty.connect(self._on_queue_empty)
        self._gpu_worker.reprocess_result.connect(self._on_reprocess_result)

        # GPU monitor
        self._gpu_monitor.vram_updated.connect(self._status_bar.update_vram)
        self._gpu_monitor.gpu_name.connect(self._status_bar.set_gpu_name)

        # Queue panel cancel signals
        self._queue_panel.cancel_job_requested.connect(self._on_cancel_job)

        # Parameter panel — wire GVM/VideoMaMa buttons
        self._param_panel.gvm_requested.connect(self._on_run_gvm)
        self._param_panel.videomama_requested.connect(self._on_run_videomama)

        # Parameter panel — live reprocess (debounced, Codex: coalesce stale)
        self._param_panel.params_changed.connect(self._on_params_changed)

    # ── Clip Selection ──

    @Slot(ClipEntry)
    def _on_clip_selected(self, clip: ClipEntry) -> None:
        self._current_clip = clip

        # Load clip into preview (builds FrameIndex, configures scrubber + modes)
        self._preview.set_clip(clip)

        # Enable run button only for READY or COMPLETE (reprocess) clips
        can_run = clip.state in (ClipState.READY, ClipState.COMPLETE)
        self._status_bar.set_run_enabled(can_run)

        # Enable GVM/VideoMaMa buttons based on state
        self._param_panel.set_gvm_enabled(clip.state == ClipState.RAW)
        self._param_panel.set_videomama_enabled(clip.state == ClipState.MASKED)

    @Slot(str)
    def _on_clips_dir_changed(self, dir_path: str) -> None:
        logger.info(f"Scanning clips directory: {dir_path}")
        self._clips_dir = dir_path
        try:
            clips = self._service.scan_clips(dir_path)
            self._clip_model.set_clips(clips)

            # Generate thumbnails for all clips (background)
            for clip in clips:
                if clip.input_asset:
                    self._thumb_gen.generate(
                        clip.name, clip.root_path,
                        clip.input_asset.path, clip.input_asset.asset_type,
                    )

            if clips:
                self._clip_browser.select_first()
            logger.info(f"Found {len(clips)} clips")

            # Auto-load session if exists (Codex: block signals during restore)
            self._try_auto_load_session(dir_path)

        except Exception as e:
            logger.error(f"Failed to scan clips: {e}")
            QMessageBox.critical(self, "Scan Error", f"Failed to scan clips directory:\n{e}")

    def _on_open_folder(self) -> None:
        self._clip_browser._on_add_clicked()

    # ── View Controls ──

    def _on_toggle_split(self, checked: bool) -> None:
        """Toggle split view in preview."""
        self._preview.set_split_mode(checked)

    def _on_reset_zoom(self) -> None:
        """Reset preview zoom to fit."""
        self._preview.reset_zoom()

    # ── Live Reprocess (Codex: through GPU queue, not parallel) ──

    @Slot()
    def _on_params_changed(self) -> None:
        """Handle parameter change — debounce before reprocess."""
        if self._param_panel.live_preview_enabled and self._service.is_engine_loaded():
            self._reprocess_timer.start()

    def _do_reprocess(self) -> None:
        """Submit a PREVIEW_REPROCESS job through the GPU queue (Codex: no bypass)."""
        if self._current_clip is None:
            return
        clip = self._current_clip
        if clip.state not in (ClipState.READY, ClipState.COMPLETE):
            return
        if not self._service.is_engine_loaded():
            return

        frame_idx = max(0, self._preview.current_stem_index)
        params = self._param_panel.get_params()
        job = create_job_snapshot(clip, params, job_type=JobType.PREVIEW_REPROCESS)
        job.params["_frame_index"] = frame_idx

        self._service.job_queue.submit(job)
        self._start_worker_if_needed()

    @Slot(str, object)
    def _on_reprocess_result(self, job_id: str, result: object) -> None:
        """Handle live reprocess result — display comp preview."""
        if not isinstance(result, dict) or 'comp' not in result:
            return
        comp = result['comp']
        rgb = (np.clip(comp, 0.0, 1.0) * 255.0).astype(np.uint8)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._preview.show_reprocess_preview(qimg)

    # ── Inference Control ──

    @Slot()
    def _on_run_inference(self) -> None:
        if self._current_clip is None:
            return

        clip = self._current_clip
        if clip.state not in (ClipState.READY, ClipState.COMPLETE):
            QMessageBox.warning(
                self, "Not Ready",
                f"Clip '{clip.name}' is in {clip.state.value} state.\n"
                "Only READY or COMPLETE clips can be processed.",
            )
            return

        # Check for resume (partial outputs exist)
        resume = False
        if clip.state == ClipState.COMPLETE or clip.completed_frame_count() > 0:
            existing = clip.completed_frame_count()
            total = clip.input_asset.frame_count if clip.input_asset else 0
            if 0 < existing < total:
                reply = QMessageBox.question(
                    self, "Resume?",
                    f"Clip '{clip.name}' has {existing}/{total} frames already processed.\n\n"
                    "Resume from where you left off?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                )
                if reply == QMessageBox.Cancel:
                    return
                resume = (reply == QMessageBox.Yes)

        # For COMPLETE clips wanting reprocess, transition back to READY
        if clip.state == ClipState.COMPLETE:
            clip.transition_to(ClipState.READY)

        params = self._param_panel.get_params()
        job = create_job_snapshot(clip, params, resume=resume)

        # Store output config in job params
        output_config = self._param_panel.get_output_config()
        job.params["_output_config"] = output_config

        if not self._service.job_queue.submit(job):
            QMessageBox.information(self, "Duplicate", f"'{clip.name}' is already queued.")
            return

        self._start_worker_if_needed(job.id)

    @Slot()
    def _on_run_all_ready(self) -> None:
        """Queue all READY clips for inference."""
        ready_clips = self._clip_model.clips_by_state(ClipState.READY)
        if not ready_clips:
            QMessageBox.information(self, "No Clips", "No READY clips to process.")
            return

        params = self._param_panel.get_params()
        output_config = self._param_panel.get_output_config()
        queued = 0
        for clip in ready_clips:
            job = create_job_snapshot(clip, params)
            job.params["_output_config"] = output_config
            if self._service.job_queue.submit(job):
                queued += 1

        if queued > 0:
            first_job = self._service.job_queue.next_job()
            self._start_worker_if_needed(first_job.id if first_job else None)
            logger.info(f"Batch queued: {queued} clips")

    @Slot()
    def _on_run_gvm(self) -> None:
        """Run GVM alpha generation on the selected clip."""
        if self._current_clip is None or self._current_clip.state != ClipState.RAW:
            return

        job = create_job_snapshot(self._current_clip, job_type=JobType.GVM_ALPHA)
        if not self._service.job_queue.submit(job):
            return

        self._current_clip.set_processing(True)
        self._start_worker_if_needed(job.id)

    @Slot()
    def _on_run_videomama(self) -> None:
        """Run VideoMaMa alpha generation on the selected clip."""
        if self._current_clip is None or self._current_clip.state != ClipState.MASKED:
            return

        job = create_job_snapshot(self._current_clip, job_type=JobType.VIDEOMAMA_ALPHA)
        if not self._service.job_queue.submit(job):
            return

        self._current_clip.set_processing(True)
        self._start_worker_if_needed(job.id)

    def _start_worker_if_needed(self, first_job_id: str | None = None) -> None:
        """Ensure GPU worker is running and show queue panel."""
        if first_job_id and self._active_job_id is None:
            self._active_job_id = first_job_id

        if not self._gpu_worker.isRunning():
            self._gpu_worker.start()
        else:
            self._gpu_worker.wake()

        self._status_bar.set_running(True)
        self._status_bar.reset_progress()
        self._queue_panel.refresh()
        self._queue_panel.show()

    @Slot()
    def _on_stop_inference(self) -> None:
        self._service.job_queue.cancel_all()
        self._status_bar.set_running(False)
        self._queue_panel.refresh()
        logger.info("Inference cancelled by user")

    @Slot(str)
    def _on_cancel_job(self, job_id: str) -> None:
        """Cancel a specific job from the queue panel."""
        job = self._service.job_queue.find_job_by_id(job_id)
        if job:
            self._service.job_queue.cancel_job(job)
            self._queue_panel.refresh()

    # ── Worker Signal Handlers ──

    @Slot(str, str, int, int)
    def _on_worker_progress(self, job_id: str, clip_name: str, current: int, total: int) -> None:
        # Set active_job_id only on first progress of a new job (not every event)
        if self._active_job_id != job_id:
            # Only update if this is genuinely a new running job
            current_job = self._service.job_queue.current_job
            if current_job and current_job.id == job_id:
                self._active_job_id = job_id

        if job_id == self._active_job_id:
            self._status_bar.update_progress(current, total)

        self._queue_panel.refresh()

    @Slot(str, str, int, str)
    def _on_worker_preview(self, job_id: str, clip_name: str, frame_index: int, path: str) -> None:
        # Only update preview if this is the active job
        if job_id == self._active_job_id:
            self._preview.load_preview_from_file(path, clip_name, frame_index)

    @Slot(str, str)
    def _on_worker_clip_finished(self, job_id: str, clip_name: str) -> None:
        self._clip_model.update_clip_state(clip_name, ClipState.COMPLETE)
        # Clear processing lock
        for clip in self._clip_model.clips:
            if clip.name == clip_name:
                clip.set_processing(False)
                break
        self._queue_panel.refresh()
        logger.info(f"Clip completed: {clip_name}")

    @Slot(str, str)
    def _on_worker_warning(self, job_id: str, message: str) -> None:
        self._status_bar.add_warning()
        logger.warning(f"Worker warning: {message}")

    @Slot(str, str, str)
    def _on_worker_error(self, job_id: str, clip_name: str, error_msg: str) -> None:
        self._clip_model.update_clip_state(clip_name, ClipState.ERROR)
        # Clear processing lock
        for clip in self._clip_model.clips:
            if clip.name == clip_name:
                clip.set_processing(False)
                break
        self._queue_panel.refresh()
        logger.error(f"Worker error for {clip_name}: {error_msg}")
        QMessageBox.critical(self, "Processing Error", f"Clip: {clip_name}\n\n{error_msg}")

    @Slot()
    def _on_queue_empty(self) -> None:
        self._status_bar.set_running(False)
        self._active_job_id = None
        self._queue_panel.refresh()
        logger.info("All jobs completed")

    # ── Session Save/Load (Codex: JSON sidecar, atomic write, version) ──

    def _session_path(self) -> str | None:
        """Return session file path, or None if no clips dir."""
        if not self._clips_dir:
            return None
        return os.path.join(self._clips_dir, _SESSION_FILENAME)

    def _build_session_data(self) -> dict:
        """Build session data dict from current UI state."""
        data: dict = {
            "version": _SESSION_VERSION,
            "params": self._param_panel.get_params().to_dict(),
            "output_config": self._param_panel.get_output_config().to_dict(),
            "live_preview": self._param_panel.live_preview_enabled,
            "split_view": self._split_action.isChecked(),
        }

        # Window geometry
        geo = self.geometry()
        data["geometry"] = {
            "x": geo.x(), "y": geo.y(),
            "width": geo.width(), "height": geo.height(),
        }

        # Splitter sizes
        data["splitter_sizes"] = self._splitter.sizes()

        # Selected clip
        if self._current_clip:
            data["selected_clip"] = self._current_clip.name

        return data

    def _apply_session_data(self, data: dict) -> None:
        """Apply session data to UI widgets.

        Codex: block widget signals during restore to prevent event storms.
        Ignores unknown keys for forward compatibility.
        """
        version = data.get("version", 0)
        if version > _SESSION_VERSION:
            logger.warning(f"Session version {version} is newer than supported {_SESSION_VERSION}")

        # Restore params
        if "params" in data:
            try:
                params = InferenceParams.from_dict(data["params"])
                self._param_panel.set_params(params)
            except Exception as e:
                logger.warning(f"Failed to restore params: {e}")

        # Restore output config
        if "output_config" in data:
            try:
                config = OutputConfig.from_dict(data["output_config"])
                self._param_panel.set_output_config(config)
            except Exception as e:
                logger.warning(f"Failed to restore output config: {e}")

        # Restore split view
        if "split_view" in data:
            checked = bool(data["split_view"])
            self._split_action.setChecked(checked)
            self._preview.set_split_mode(checked)

        # Restore splitter sizes
        if "splitter_sizes" in data:
            try:
                sizes = [int(s) for s in data["splitter_sizes"]]
                self._splitter.setSizes(sizes)
            except Exception:
                pass

        # Restore window geometry (clamped to current screen)
        if "geometry" in data:
            try:
                g = data["geometry"]
                self.setGeometry(g["x"], g["y"], g["width"], g["height"])
            except Exception:
                pass

        # Restore selected clip
        if "selected_clip" in data:
            clip_name = data["selected_clip"]
            for i, clip in enumerate(self._clip_model.clips):
                if clip.name == clip_name:
                    self._clip_browser.select_by_index(i)
                    break

    @Slot()
    def _on_save_session(self) -> None:
        """Save session to JSON sidecar in clips directory."""
        path = self._session_path()
        if not path:
            QMessageBox.information(self, "No Folder", "Open a clips folder first.")
            return

        data = self._build_session_data()
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
            # Atomic rename (Windows: need to remove target first)
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp_path, path)
            logger.info(f"Session saved: {path}")
        except OSError as e:
            logger.warning(f"Failed to save session: {e}")
            # Clean up tmp if it exists
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @Slot()
    def _on_load_session(self) -> None:
        """Load session from JSON sidecar."""
        path = self._session_path()
        if not path or not os.path.isfile(path):
            QMessageBox.information(self, "No Session", "No saved session found in current folder.")
            return
        self._load_session_from(path)

    def _try_auto_load_session(self, clips_dir: str) -> None:
        """Auto-load session if .corridorkey_session.json exists in clips dir."""
        path = os.path.join(clips_dir, _SESSION_FILENAME)
        if os.path.isfile(path):
            self._load_session_from(path)

    def _load_session_from(self, path: str) -> None:
        """Load session data from a file path."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self._apply_session_data(data)
            logger.info(f"Session loaded: {path}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load session: {e}")

    # ── Layout & Dialogs ──

    def _reset_layout(self) -> None:
        self._splitter.setSizes([220, 700, 280])

    def _toggle_queue_panel(self) -> None:
        self._queue_panel.setVisible(not self._queue_panel.isVisible())

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About CorridorKey",
            "CorridorKey — AI Green Screen Keyer\n\n"
            "Based on GreenFormer by Corridor Crew\n"
            "CC BY-NC-SA 4.0 License\n\n"
            "PySide6 Desktop Application",
        )

    def closeEvent(self, event) -> None:
        """Clean shutdown — auto-save session, stop worker, unload engines."""
        # Auto-save session on close
        if self._clips_dir:
            try:
                self._on_save_session()
            except Exception:
                pass

        self._gpu_monitor.stop()
        if self._gpu_worker.isRunning():
            self._gpu_worker.stop()
            self._gpu_worker.wait(5000)
        self._service.unload_engines()
        super().closeEvent(event)

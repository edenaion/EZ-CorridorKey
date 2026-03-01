"""Main window — dual viewer layout with I/O tray and menu bar.

Layout:
    ┌─[CORRIDORKEY]──────────────────[GPU | VRAM ██ X/YGB]─┐
    ├──────────┬──────────┬──────────┬──────────────────────┤
    │  Clips   │ INPUT    │ OUTPUT   │  Parameters          │
    │  Browser │ Viewer   │ Viewer   │    Panel             │
    │ (220px)  │ (fills)  │ (fills)  │  (280px)             │
    ├──────────┴──────────┴──────────┴──────────────────────┤
    │  INPUT (N)               │  EXPORTS (N)               │
    ├──────────────────────────────────────────────────────┤
    │  Queue Panel (collapsible, per-job progress)         │
    ├──────────────────────────────────────────────────────┤
    │  [progress]  frame counter  warnings  [RUN/STOP]     │
    └──────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
import logging
import os

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QSplitter, QVBoxLayout, QWidget,
    QLabel, QHBoxLayout, QMessageBox, QStackedWidget,
    QProgressBar, QFileDialog,
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
from ui.widgets.dual_viewer import DualViewerPanel
from ui.widgets.parameter_panel import ParameterPanel
from ui.widgets.status_bar import StatusBar
from ui.widgets.queue_panel import QueuePanel
from ui.widgets.io_tray_panel import IOTrayPanel
from ui.widgets.welcome_screen import WelcomeScreen
from ui.workers.gpu_job_worker import GPUJobWorker, create_job_snapshot
from ui.workers.gpu_monitor import GPUMonitor
from ui.workers.thumbnail_worker import ThumbnailGenerator
from ui.workers.extract_worker import ExtractWorker
from ui.recent_sessions import RecentSessionsStore

logger = logging.getLogger(__name__)

# Session file stored in clips dir (Codex: JSON sidecar)
_SESSION_FILENAME = ".corridorkey_session.json"
_SESSION_VERSION = 1


class MainWindow(QMainWindow):
    """CorridorKey main application window."""

    def __init__(self, service: CorridorKeyService | None = None,
                 store: RecentSessionsStore | None = None):
        super().__init__()
        self.setWindowTitle("CORRIDORKEY")
        self.setMinimumSize(1100, 650)
        self.resize(1400, 800)

        self._service = service or CorridorKeyService()
        self._recent_store = store or RecentSessionsStore()
        self._current_clip: ClipEntry | None = None
        self._clips_dir: str | None = None
        # Track active job ID — set only once when job starts, not on every progress
        self._active_job_id: str | None = None
        # Display name for next workspace registration (set by _on_welcome_files)
        self._pending_display_name: str | None = None

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
        self._extract_worker = ExtractWorker(parent=self)

        # Connect signals
        self._connect_signals()

        # Start GPU monitoring
        self._gpu_monitor.start()

        # Periodic auto-save for crash recovery (every 60s)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(60_000)
        self._autosave_timer.timeout.connect(self._auto_save_session)
        self._autosave_timer.start()

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
        file_menu.addAction("Export Video...", self._on_export_video)
        file_menu.addSeparator()
        file_menu.addAction("Return to Welcome", self._return_to_welcome)
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

        # Top bar with brand mark (left) + GPU/VRAM info (right)
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(12, 6, 12, 6)

        brand = QLabel('<span style="color:#FFF203;">CORRIDOR</span><span style="color:#2CC350;">KEY</span>')
        brand.setObjectName("brandMark")
        top_bar.addWidget(brand)
        top_bar.addStretch()

        # GPU info (right side of brand bar)
        self._gpu_label = QLabel("")
        self._gpu_label.setObjectName("gpuName")
        top_bar.addWidget(self._gpu_label)

        self._vram_label = QLabel("VRAM")
        self._vram_label.setStyleSheet("color: #808070; font-size: 10px;")
        top_bar.addWidget(self._vram_label)

        self._vram_bar = QProgressBar()
        self._vram_bar.setObjectName("vramMeter")
        self._vram_bar.setFixedWidth(80)
        self._vram_bar.setFixedHeight(8)
        self._vram_bar.setTextVisible(False)
        self._vram_bar.setRange(0, 100)
        self._vram_bar.setValue(0)
        top_bar.addWidget(self._vram_bar)

        self._vram_text = QLabel("")
        self._vram_text.setObjectName("vramText")
        self._vram_text.setMinimumWidth(70)
        top_bar.addWidget(self._vram_text)

        main_layout.addLayout(top_bar)

        # Stacked widget: page 0 = welcome, page 1 = workspace
        self._stack = QStackedWidget()

        # Page 0 — Welcome/drop screen
        self._welcome = WelcomeScreen(self._recent_store)
        self._welcome.folder_selected.connect(self._on_welcome_folder)
        self._welcome.files_selected.connect(self._on_welcome_files)
        self._welcome.recent_project_opened.connect(self._on_recent_project_opened)
        self._stack.addWidget(self._welcome)

        # Page 1 — Workspace (vertical splitter: top panels + I/O tray)
        workspace = QWidget()
        ws_layout = QVBoxLayout(workspace)
        ws_layout.setContentsMargins(0, 0, 0, 0)
        ws_layout.setSpacing(0)

        # Vertical splitter: top = 3-panel horizontal splitter, bottom = I/O tray
        self._vsplitter = QSplitter(Qt.Vertical)

        # Horizontal splitter: clip browser | dual viewer | param panel
        self._splitter = QSplitter(Qt.Horizontal)

        # Left — Clip Browser
        self._clip_browser = ClipBrowser(self._clip_model)
        self._splitter.addWidget(self._clip_browser)

        # Center — Dual Viewer (input + output side by side)
        self._dual_viewer = DualViewerPanel()
        self._splitter.addWidget(self._dual_viewer)

        # Right — Parameter Panel
        self._param_panel = ParameterPanel()
        self._splitter.addWidget(self._param_panel)

        # Set initial sizes (220, fill, 280)
        self._splitter.setSizes([220, 700, 280])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)

        self._vsplitter.addWidget(self._splitter)

        # Bottom — I/O Tray Panel
        self._io_tray = IOTrayPanel(self._clip_model)
        self._vsplitter.addWidget(self._io_tray)

        # Top fills, tray is fixed height
        self._vsplitter.setStretchFactor(0, 1)
        self._vsplitter.setStretchFactor(1, 0)
        self._vsplitter.setSizes([600, 140])

        ws_layout.addWidget(self._vsplitter, 1)

        # Queue panel (collapsible, above status bar)
        self._queue_panel = QueuePanel(self._service.job_queue)
        self._queue_panel.hide()
        ws_layout.addWidget(self._queue_panel)

        self._stack.addWidget(workspace)

        main_layout.addWidget(self._stack, 1)

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

        # GPU monitor → top bar widgets (not status bar)
        self._gpu_monitor.vram_updated.connect(self._update_vram)
        self._gpu_monitor.gpu_name.connect(self._set_gpu_name)

        # Queue panel cancel signals
        self._queue_panel.cancel_job_requested.connect(self._on_cancel_job)

        # Parameter panel — wire GVM/VideoMaMa buttons
        self._param_panel.gvm_requested.connect(self._on_run_gvm)
        self._param_panel.videomama_requested.connect(self._on_run_videomama)

        # Parameter panel — live reprocess (debounced, Codex: coalesce stale)
        self._param_panel.params_changed.connect(self._on_params_changed)

        # I/O tray click → select clip
        self._io_tray.clip_clicked.connect(self._on_tray_clip_clicked)

        # Extract worker signals
        self._extract_worker.progress.connect(self._on_extract_progress)
        self._extract_worker.finished.connect(self._on_extract_finished)
        self._extract_worker.error.connect(self._on_extract_error)

    # ── GPU Header ──

    @Slot(dict)
    def _update_vram(self, info: dict) -> None:
        """Update VRAM meter in the top bar."""
        if not info.get("available"):
            self._vram_text.setText("No GPU")
            self._vram_bar.setValue(0)
            return
        pct = info.get("usage_pct", 0)
        used = info.get("used_gb", 0)
        total = info.get("total_gb", 0)
        self._vram_bar.setValue(int(pct))
        self._vram_text.setText(f"{used:.1f}/{total:.1f}GB")

    @Slot(str)
    def _set_gpu_name(self, name: str) -> None:
        """Display GPU name in the top bar."""
        short = name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        self._gpu_label.setText(short)

    @Slot(object)
    def _on_tray_clip_clicked(self, clip: ClipEntry) -> None:
        """Handle clip clicked in I/O tray — select it in browser and preview."""
        for i, c in enumerate(self._clip_model.clips):
            if c.name == clip.name:
                self._clip_browser.select_by_index(i)
                break

    # ── Clip Selection ──

    @Slot(ClipEntry)
    def _on_clip_selected(self, clip: ClipEntry) -> None:
        self._current_clip = clip

        # Load clip into dual viewer (both input + output viewports)
        self._dual_viewer.set_clip(clip)

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
        # Ensure workspace is visible (may come from welcome screen or menu)
        self._switch_to_workspace()
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

            # Auto-submit EXTRACTING clips to extract worker
            self._auto_extract_clips(clips)

            if clips:
                self._clip_browser.select_first()
            logger.info(f"Found {len(clips)} clips")

            # Register in recent sessions store
            display_name = self._pending_display_name or os.path.basename(dir_path)
            self._pending_display_name = None
            self._recent_store.add_or_update(dir_path, display_name, len(clips))

            # Auto-load session if exists (Codex: block signals during restore)
            self._try_auto_load_session(dir_path)

        except Exception as e:
            logger.error(f"Failed to scan clips: {e}")
            QMessageBox.critical(self, "Scan Error", f"Failed to scan clips directory:\n{e}")

    def _switch_to_workspace(self) -> None:
        """Switch from welcome screen to the 3-panel workspace."""
        self._stack.setCurrentIndex(1)

    @Slot(str)
    def _on_welcome_folder(self, dir_path: str) -> None:
        """Handle folder selected from welcome screen."""
        self._switch_to_workspace()
        self._on_clips_dir_changed(dir_path)

    @Slot(str)
    def _on_recent_project_opened(self, workspace_path: str) -> None:
        """Open a workspace from the recent projects list."""
        if not os.path.isdir(workspace_path):
            QMessageBox.warning(self, "Missing", f"Workspace no longer exists:\n{workspace_path}")
            self._recent_store.remove(workspace_path)
            self._welcome.refresh_recents()
            return
        self._switch_to_workspace()
        self._on_clips_dir_changed(workspace_path)

    def _return_to_welcome(self) -> None:
        """Save session and return to the welcome screen."""
        if self._clips_dir:
            try:
                self._on_save_session()
            except Exception:
                pass
        self._stack.setCurrentIndex(0)
        self._welcome.refresh_recents()
        self._clips_dir = None
        self._current_clip = None

    @Slot(list)
    def _on_welcome_files(self, file_paths: list) -> None:
        """Handle files selected from welcome screen.

        Creates a dedicated workspace directory next to the first file,
        restructures each selected video into its own clip subdirectory,
        then scans ONLY that workspace — not the parent dir (which would
        pick up every video in e.g. Downloads).
        """
        if not file_paths:
            return

        import shutil

        # Create a workspace dir adjacent to the source files
        parent = os.path.dirname(file_paths[0])
        workspace = os.path.join(parent, "CorridorKey_Workspace")
        os.makedirs(workspace, exist_ok=True)

        _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v"}
        for fpath in file_paths:
            ext = os.path.splitext(fpath)[1].lower()
            if ext in _VIDEO_EXTS:
                stem = os.path.splitext(os.path.basename(fpath))[0]
                clip_dir = os.path.join(workspace, stem)
                target = os.path.join(clip_dir, f"Input{ext}")
                if not os.path.isfile(target):
                    os.makedirs(clip_dir, exist_ok=True)
                    shutil.copy2(fpath, target)
                    logger.info(f"Imported video: {fpath} → {target}")

        # Derive display name from source files for the recents list
        if len(file_paths) == 1:
            self._pending_display_name = os.path.splitext(os.path.basename(file_paths[0]))[0]
        else:
            first = os.path.splitext(os.path.basename(file_paths[0]))[0]
            self._pending_display_name = f"{first} + {len(file_paths) - 1} more"

        self._switch_to_workspace()
        self._on_clips_dir_changed(workspace)

    def _on_open_folder(self) -> None:
        self._clip_browser._on_add_clicked()

    # ── View Controls ──

    def _on_toggle_split(self, checked: bool) -> None:
        """Toggle split view in preview."""
        self._dual_viewer.set_split_mode(checked)

    def _on_reset_zoom(self) -> None:
        """Reset preview zoom to fit."""
        self._dual_viewer.reset_zoom()

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

        frame_idx = max(0, self._dual_viewer.current_stem_index)
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
        self._dual_viewer.show_reprocess_preview(qimg)

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
        self._start_worker_if_needed(job.id, job_label="GVM Auto", indeterminate=False)

    @Slot()
    def _on_run_videomama(self) -> None:
        """Run VideoMaMa alpha generation on the selected clip."""
        if self._current_clip is None or self._current_clip.state != ClipState.MASKED:
            return

        job = create_job_snapshot(self._current_clip, job_type=JobType.VIDEOMAMA_ALPHA)
        if not self._service.job_queue.submit(job):
            return

        self._current_clip.set_processing(True)
        self._start_worker_if_needed(job.id, job_label="VideoMaMa", indeterminate=True)

    def _start_worker_if_needed(
        self,
        first_job_id: str | None = None,
        job_label: str = "Inference",
        indeterminate: bool = False,
    ) -> None:
        """Ensure GPU worker is running and show queue panel."""
        if first_job_id and self._active_job_id is None:
            self._active_job_id = first_job_id

        if not self._gpu_worker.isRunning():
            self._gpu_worker.start()
        else:
            self._gpu_worker.wake()

        self._status_bar.set_running(True)
        self._status_bar.reset_progress()
        self._status_bar.start_job_timer(label=job_label, indeterminate=indeterminate)
        self._queue_panel.refresh()
        self._queue_panel.show()

    @Slot()
    def _on_stop_inference(self) -> None:
        # Guard: only cancel if a job is actually running
        if not self._status_bar._stop_btn.isVisible():
            return
        queue = self._service.job_queue
        if not queue.current_job and not queue.has_pending:
            return

        reply = QMessageBox.question(
            self, "Cancel Processing",
            "Are you sure you want to cancel?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        queue.cancel_all()
        self._status_bar.set_message("Cancelling...")
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
            self._dual_viewer.load_preview_from_file(path, clip_name, frame_index)

    @Slot(str, str)
    def _on_worker_clip_finished(self, job_id: str, clip_name: str) -> None:
        self._clip_model.update_clip_state(clip_name, ClipState.COMPLETE)
        # Clear processing lock
        for clip in self._clip_model.clips:
            if clip.name == clip_name:
                clip.set_processing(False)
                break
        self._queue_panel.refresh()
        self._io_tray.refresh()
        logger.info(f"Clip completed: {clip_name}")

    @Slot(str, str)
    def _on_worker_warning(self, job_id: str, message: str) -> None:
        if message.startswith("Cancelled:"):
            # Job was cancelled — clear processing lock on the clip
            clip_name = message.removeprefix("Cancelled:").strip()
            for clip in self._clip_model.clips:
                if clip.name == clip_name:
                    clip.set_processing(False)
                    break
            self._status_bar.stop_job_timer()
            self._status_bar.set_running(False)
            self._status_bar.set_message(f"Cancelled: {clip_name}")
            self._queue_panel.refresh()
            logger.info(f"Job cancelled: {clip_name}")
        else:
            self._status_bar.add_warning()
            logger.warning(f"Worker warning: {message}")

    @Slot(str, str, str)
    def _on_worker_error(self, job_id: str, clip_name: str, error_msg: str) -> None:
        self._status_bar.stop_job_timer()
        self._status_bar.set_running(False)
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
        self._status_bar.stop_job_timer()
        self._active_job_id = None
        self._queue_panel.refresh()
        logger.info("All jobs completed")

    # ── Video Extraction ──

    def _auto_extract_clips(self, clips: list[ClipEntry]) -> None:
        """Auto-submit EXTRACTING clips to the extract worker.

        For standalone videos (root_path == clips_dir), restructure into
        a proper clip directory first so extraction writes to clip/Input/.
        """
        import shutil

        extracting = [c for c in clips if c.state == ClipState.EXTRACTING]
        if not extracting:
            return

        if not self._extract_worker.isRunning():
            self._extract_worker.start()

        for clip in extracting:
            if not (clip.input_asset and clip.input_asset.asset_type == "video"):
                continue

            video_path = clip.input_asset.path

            # Standalone video: root_path is the parent dir, not a clip dir.
            # Restructure: create clip_name/ dir and copy video as Input.ext
            if clip.root_path == self._clips_dir:
                ext = os.path.splitext(video_path)[1]
                clip_dir = os.path.join(self._clips_dir, clip.name)
                target = os.path.join(clip_dir, f"Input{ext}")
                if not os.path.isfile(target):
                    os.makedirs(clip_dir, exist_ok=True)
                    shutil.copy2(video_path, target)
                    logger.info(f"Restructured standalone video: {video_path} → {target}")
                clip.root_path = clip_dir
                clip.input_asset.path = target

            self._extract_worker.submit(
                clip.name, clip.input_asset.path, clip.root_path,
            )
        logger.info(f"Auto-extraction queued: {len(extracting)} clip(s)")

    @Slot(str, int, int)
    def _on_extract_progress(self, clip_name: str, current: int, total: int) -> None:
        """Update status bar with extraction progress."""
        pct = int(current / total * 100) if total > 0 else 0
        self._status_bar.set_message(f"Extracting {clip_name}: {pct}%")

    @Slot(str, int)
    def _on_extract_finished(self, clip_name: str, frame_count: int) -> None:
        """Handle extraction complete — update clip to RAW with image sequence."""
        from backend.clip_state import ClipAsset
        for clip in self._clip_model.clips:
            if clip.name == clip_name:
                # Update input asset to point to extracted sequence
                input_dir = os.path.join(clip.root_path, "Input")
                if os.path.isdir(input_dir):
                    clip.input_asset = ClipAsset(input_dir, "sequence")

                # Transition EXTRACTING → RAW
                clip.state = ClipState.RAW
                self._clip_model.update_clip_state(clip_name, ClipState.RAW)

                # Regenerate thumbnail from sequence
                if clip.input_asset:
                    self._thumb_gen.generate(
                        clip.name, clip.root_path,
                        clip.input_asset.path, clip.input_asset.asset_type,
                    )

                # If this is the selected clip, reload preview
                if self._current_clip and self._current_clip.name == clip_name:
                    self._dual_viewer.set_clip(clip)
                    self._status_bar.set_run_enabled(
                        clip.state in (ClipState.READY, ClipState.COMPLETE)
                    )

                logger.info(f"Extraction complete: {clip_name} ({frame_count} frames)")
                break

        self._io_tray.refresh()
        self._status_bar.set_message("")

    @Slot(str, str)
    def _on_extract_error(self, clip_name: str, error_msg: str) -> None:
        """Handle extraction failure."""
        self._clip_model.update_clip_state(clip_name, ClipState.ERROR)
        for clip in self._clip_model.clips:
            if clip.name == clip_name:
                clip.error_message = error_msg
                break
        self._status_bar.set_message("")
        logger.error(f"Extraction failed for {clip_name}: {error_msg}")

    # ── Export Video ──

    def _on_export_video(self) -> None:
        """Export output image sequence as video file."""
        if self._current_clip is None:
            QMessageBox.information(self, "No Clip", "Select a clip first.")
            return

        clip = self._current_clip
        if clip.state != ClipState.COMPLETE:
            QMessageBox.warning(
                self, "Not Complete",
                f"Clip '{clip.name}' must be COMPLETE to export video.",
            )
            return

        # Find output directory with frames (prefer Comp, fall back to FG)
        comp_dir = os.path.join(clip.output_dir, "Comp")
        fg_dir = os.path.join(clip.output_dir, "FG")
        if os.path.isdir(comp_dir) and os.listdir(comp_dir):
            source_dir = comp_dir
        elif os.path.isdir(fg_dir) and os.listdir(fg_dir):
            source_dir = fg_dir
        else:
            QMessageBox.warning(self, "No Output", "No output frames found to export.")
            return

        # Read video metadata for fps
        from backend.ffmpeg_tools import read_video_metadata, stitch_video, find_ffmpeg
        if not find_ffmpeg():
            QMessageBox.critical(
                self, "FFmpeg Not Found",
                "FFmpeg is required for video export.\n"
                "Install FFmpeg and add it to your PATH.",
            )
            return

        metadata = read_video_metadata(clip.root_path)
        fps = metadata.get("fps", 24.0) if metadata else 24.0

        # Ask for output path
        default_name = f"{clip.name}_export.mp4"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", default_name,
            "MP4 Video (*.mp4);;All Files (*)",
        )
        if not out_path:
            return

        # Determine frame pattern from first file
        frames = sorted(os.listdir(source_dir))
        if not frames:
            return

        # Detect pattern (frame_000000.png → frame_%06d.png)
        first = frames[0]
        ext = os.path.splitext(first)[1]
        pattern = f"frame_%06d{ext}"

        self._status_bar.set_message(f"Exporting {clip.name}...")

        try:
            stitch_video(
                in_dir=source_dir,
                out_path=out_path,
                fps=fps,
                pattern=pattern,
            )
            self._status_bar.set_message("")
            QMessageBox.information(
                self, "Export Complete",
                f"Video exported:\n{out_path}",
            )
        except Exception as e:
            self._status_bar.set_message("")
            QMessageBox.critical(
                self, "Export Failed",
                f"Failed to export video:\n{e}",
            )

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
        data["vsplitter_sizes"] = self._vsplitter.sizes()

        # Workspace path (for absolute reference)
        if self._clips_dir:
            data["workspace_path"] = self._clips_dir

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
            self._dual_viewer.set_split_mode(checked)

        # Restore splitter sizes (validate: must have 3 panels, none at 0)
        if "splitter_sizes" in data:
            try:
                sizes = [int(s) for s in data["splitter_sizes"]]
                if len(sizes) == 3 and all(s > 0 for s in sizes):
                    self._splitter.setSizes(sizes)
                else:
                    logger.info("Ignoring invalid splitter_sizes, using defaults")
            except Exception:
                pass

        # Restore vertical splitter sizes (validate: must have 2 panels)
        if "vsplitter_sizes" in data:
            try:
                sizes = [int(s) for s in data["vsplitter_sizes"]]
                if len(sizes) == 2 and all(s > 0 for s in sizes):
                    self._vsplitter.setSizes(sizes)
                else:
                    logger.info("Ignoring invalid vsplitter_sizes, using defaults")
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

    def _auto_save_session(self) -> None:
        """Periodic auto-save for crash recovery (called by timer)."""
        if self._clips_dir and self._stack.currentIndex() == 1:
            path = self._session_path()
            if not path:
                return
            data = self._build_session_data()
            tmp_path = path + ".tmp"
            try:
                with open(tmp_path, 'w') as f:
                    json.dump(data, f, indent=2)
                if os.path.exists(path):
                    os.remove(path)
                os.rename(tmp_path, path)
            except OSError:
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
        self._vsplitter.setSizes([600, 140])

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
        """Clean shutdown — auto-save session, stop workers, unload engines."""
        # Auto-save session on close
        if self._clips_dir:
            try:
                self._on_save_session()
            except Exception:
                pass

        self._gpu_monitor.stop()
        if self._extract_worker.isRunning():
            self._extract_worker.stop()
            self._extract_worker.wait(5000)
        if self._gpu_worker.isRunning():
            self._gpu_worker.stop()
            self._gpu_worker.wait(5000)
        self._service.unload_engines()
        super().closeEvent(event)

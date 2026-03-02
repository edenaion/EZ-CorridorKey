"""Main window — dual viewer layout with I/O tray and menu bar.

Layout:
    ┌─[CORRIDORKEY]──────────────────[GPU | VRAM ██ X/YGB]─┐
    ├──────────────────────┬───────────────────────────────┤
    │  INPUT    │ OUTPUT   │  Parameters                    │
    │  Viewer   │ Viewer   │    Panel                       │
    │  (fills)  │ (fills)  │  (280px)                       │
    ├───────────┴──────────┴───────────────────────────────┤
    │  INPUT (N)  [+ADD]       │  EXPORTS (N)               │
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
    QProgressBar, QFileDialog, QInputDialog,
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import (
    QShortcut, QKeySequence, QAction, QImage,
    QPainter, QLinearGradient, QColor,
)

from backend import (
    CorridorKeyService, ClipEntry, ClipState, InferenceParams,
    InOutRange, OutputConfig, JobType, JobStatus,
)
from backend.errors import CorridorKeyError
from backend.job_queue import GPUJob

from ui.models.clip_model import ClipListModel
from ui.widgets.dual_viewer import DualViewerPanel
from ui.widgets.parameter_panel import ParameterPanel
from ui.widgets.status_bar import StatusBar
from ui.widgets.queue_panel import QueuePanel
from ui.widgets.io_tray_panel import IOTrayPanel
from ui.widgets.welcome_screen import WelcomeScreen
from ui.widgets.preferences_dialog import PreferencesDialog, KEY_SHOW_TOOLTIPS, DEFAULT_SHOW_TOOLTIPS, get_setting_bool
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
        self._bg_cache: QImage | None = None
        # Batch pipeline: clip names queued for GVM→inference auto-chain
        self._batch_clips: set[str] = set()

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

        # Always start on welcome screen — user picks a project from recents or imports
        # Deferred sync of IO tray divider with viewer splitter
        QTimer.singleShot(0, self._sync_io_divider)

        # Apply persisted preferences (e.g. tooltip visibility)
        self._apply_tooltip_setting()

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

        # Edit menu
        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addAction("Preferences...", self._show_preferences)
        edit_menu.addSeparator()
        edit_menu.addAction("Export Annotation Masks", self._on_export_masks)
        edit_menu.addAction("Clear Annotations", self._on_clear_annotations)

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
        self._gpu_label.setToolTip("Detected GPU used for inference")
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
        self._vram_bar.setToolTip("GPU video memory usage — updates during inference")
        top_bar.addWidget(self._vram_bar)

        self._vram_text = QLabel("")
        self._vram_text.setObjectName("vramText")
        self._vram_text.setMinimumWidth(70)
        self._vram_text.setToolTip("Current VRAM used / total available")
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

        # Vertical splitter: top = viewer+params, bottom = I/O tray
        self._vsplitter = QSplitter(Qt.Vertical)

        # Horizontal splitter: dual viewer | param panel
        self._splitter = QSplitter(Qt.Horizontal)

        # Left — Dual Viewer (input + output side by side)
        self._dual_viewer = DualViewerPanel()
        self._splitter.addWidget(self._dual_viewer)

        # Right — Parameter Panel
        self._param_panel = ParameterPanel()
        self._splitter.addWidget(self._param_panel)

        # Viewer fills, param panel fixed width
        self._splitter.setSizes([920, 280])
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(1, False)

        self._vsplitter.addWidget(self._splitter)

        # Bottom — I/O Tray Panel
        self._io_tray = IOTrayPanel(self._clip_model)
        self._vsplitter.addWidget(self._io_tray)

        # Top fills by default, tray can be dragged up freely
        self._vsplitter.setStretchFactor(0, 1)
        self._vsplitter.setStretchFactor(1, 0)
        self._vsplitter.setCollapsible(0, False)
        self._vsplitter.setCollapsible(1, False)
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
        # Hidden until user opens a project (welcome screen has no use for it)
        self._status_bar.hide()

    def _setup_shortcuts(self) -> None:
        """Keyboard shortcuts."""
        # Escape — stop/cancel + exit annotation mode
        QShortcut(QKeySequence(Qt.Key_Escape), self, self._on_escape)
        # Ctrl+R — run inference on selected clip
        QShortcut(QKeySequence("Ctrl+R"), self, self._on_run_inference)
        # Ctrl+Shift+R — run all ready clips
        QShortcut(QKeySequence("Ctrl+Shift+R"), self, self._on_run_all_ready)
        # I/O markers — set in/out points on timeline
        QShortcut(QKeySequence(Qt.Key_I), self, self._set_in_point)
        QShortcut(QKeySequence(Qt.Key_O), self, self._set_out_point)
        QShortcut(QKeySequence("Alt+I"), self, self._clear_in_out)
        # Space — play/pause
        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_playback)
        # Annotation: 1 = green (foreground), 2 = red (background)
        QShortcut(QKeySequence(Qt.Key_1), self, self._toggle_annotation_fg)
        QShortcut(QKeySequence(Qt.Key_2), self, self._toggle_annotation_bg)
        # Ctrl+Z — undo annotation stroke
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo_annotation)

    def _toggle_playback(self) -> None:
        """Forward Space key to the scrubber's play/pause toggle."""
        self._dual_viewer.toggle_playback()

    def _on_escape(self) -> None:
        """Escape: exit annotation mode first, then fall through to stop."""
        iv = self._dual_viewer.input_viewer
        if iv.annotation_mode:
            iv.set_annotation_mode(None)
            return
        self._on_stop_inference()

    def _toggle_annotation_fg(self) -> None:
        """Hotkey 1: toggle green (foreground) annotation brush."""
        iv = self._dual_viewer.input_viewer
        if iv.annotation_mode == "fg":
            iv.set_annotation_mode(None)
        else:
            iv.set_annotation_mode("fg")

    def _toggle_annotation_bg(self) -> None:
        """Hotkey 2: toggle red (background) annotation brush."""
        iv = self._dual_viewer.input_viewer
        if iv.annotation_mode == "bg":
            iv.set_annotation_mode(None)
        else:
            iv.set_annotation_mode("bg")

    def _undo_annotation(self) -> None:
        """Ctrl+Z: undo last annotation stroke on current frame."""
        iv = self._dual_viewer.input_viewer
        if iv.annotation_mode and iv.current_stem_index >= 0:
            if iv.annotation_model.undo(iv.current_stem_index):
                iv._split_view.update()

    def _on_export_masks(self) -> None:
        """Export annotation strokes as VideoMamaMaskHint PNGs and refresh clip."""
        clip = self._current_clip
        if clip is None:
            return
        iv = self._dual_viewer.input_viewer
        model = iv.annotation_model
        if not model.has_annotations():
            QMessageBox.information(
                self, "No Annotations",
                "Paint green (1) and red (2) strokes on frames first.",
            )
            return

        fi = iv._frame_index
        if fi is None:
            return

        # Get input dimensions from the first frame
        from backend.frame_io import read_image_frame
        sample_path = fi.get_path(
            next(iter(fi.availability.keys())), 0
        )
        if sample_path is None:
            return
        sample = read_image_frame(sample_path)
        if sample is None:
            return
        h, w = sample.shape[:2]

        # Export masks
        mask_dir = model.export_masks(clip.root_path, fi.stems, w, h)
        logger.info(f"Exported annotation masks to {mask_dir}")

        # If AlphaHint already exists, it must be removed so the clip drops
        # to MASKED state (otherwise READY takes priority and VideoMaMa
        # button stays disabled). Ask the user first since this is destructive.
        import shutil
        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir):
            reply = QMessageBox.question(
                self, "Replace Existing Alpha?",
                "This clip already has an AlphaHint (from GVM or a previous run).\n\n"
                "To use your annotations with VideoMaMa, the existing AlphaHint "
                "must be removed so it can be regenerated.\n\n"
                "Remove existing AlphaHint and proceed?",
            )
            if reply != QMessageBox.Yes:
                return
            shutil.rmtree(alpha_dir)
            logger.info(f"Removed existing AlphaHint/ to allow re-generation")

        # Re-scan clip assets to detect the new VideoMamaMaskHint directory
        clip.find_assets()

        # Update button states — VideoMaMa should now be enabled
        self._param_panel.set_videomama_enabled(
            clip.state == ClipState.MASKED
        )
        self._param_panel.set_gvm_enabled(clip.state == ClipState.RAW)

        # Update annotation info on param panel
        self._param_panel.set_annotation_info(
            model.annotated_frame_count(), fi.frame_count
        )

        QMessageBox.information(
            self, "Masks Exported",
            f"Exported {fi.frame_count} mask frames "
            f"({model.annotated_frame_count()} annotated).\n"
            f"Click VIDEOMAMA to generate the alpha hint.",
        )

    def _on_clear_annotations(self) -> None:
        """Clear all annotations on the current clip."""
        iv = self._dual_viewer.input_viewer
        iv.clear_annotations()
        self._param_panel.set_annotation_info(0, 0)

    def _update_annotation_info(self) -> None:
        """Update parameter panel with current annotation count."""
        iv = self._dual_viewer.input_viewer
        model = iv.annotation_model
        fi = iv._frame_index
        total = fi.frame_count if fi else 0
        self._param_panel.set_annotation_info(
            model.annotated_frame_count(), total
        )

    def _connect_signals(self) -> None:
        # I/O tray — clip selection, import, drag-and-drop
        self._io_tray.clip_clicked.connect(self._on_tray_clip_clicked)
        self._io_tray.selection_changed.connect(self._on_selection_changed)
        self._io_tray.clips_dir_changed.connect(self._on_clips_dir_changed)
        self._io_tray.files_imported.connect(self._on_welcome_files)

        # Status bar buttons
        self._status_bar.run_clicked.connect(self._on_run_inference)
        self._status_bar.resume_clicked.connect(self._on_resume_inference)
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

        # Parameter panel — wire GVM/VideoMaMa buttons + annotation export
        self._param_panel.gvm_requested.connect(self._on_run_gvm)
        self._param_panel.videomama_requested.connect(self._on_run_videomama)
        self._param_panel.export_masks_requested.connect(self._on_export_masks)

        # Annotation stroke finished → update annotation counter
        self._dual_viewer.input_viewer._split_view.stroke_finished.connect(
            self._update_annotation_info
        )

        # Parameter panel — live reprocess (debounced, Codex: coalesce stale)
        self._param_panel.params_changed.connect(self._on_params_changed)

        # Sync IO tray divider with dual viewer splitter
        self._dual_viewer._viewer_splitter.splitterMoved.connect(self._sync_io_divider)

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
        """Handle clip clicked in I/O tray — select it and load preview."""
        self._on_clip_selected(clip)

    @Slot(list)
    def _on_selection_changed(self, clips: list) -> None:
        """Handle multi-select change in I/O tray — update button state."""
        batch_count = len(clips)
        if batch_count > 1:
            # Multi-select: enable batch run if at least one clip is processable
            self._status_bar.update_button_state(
                can_run=True,
                has_partial=False,
                has_in_out=False,
                batch_count=batch_count,
            )
        elif batch_count == 1:
            # Single clip: use normal button state
            self._refresh_button_state()
        else:
            self._status_bar.update_button_state(
                can_run=False, has_partial=False, has_in_out=False,
            )

    # ── Clip Selection ──

    @Slot(ClipEntry)
    def _on_clip_selected(self, clip: ClipEntry) -> None:
        self._current_clip = clip
        logger.debug(f"Clip selected: '{clip.name}' state={clip.state.value}")

        # Highlight in I/O tray (single-select unless multi-select is active)
        batch_count = self._io_tray.selected_count()
        if batch_count <= 1:
            self._io_tray.set_selected(clip.name)

        # Load clip into dual viewer (both input + output viewports)
        self._dual_viewer.set_clip(clip)

        # Ensure run/stop buttons are in correct visibility state
        # (guards against stale running state from crashed jobs)
        if not self._gpu_worker.isRunning():
            self._status_bar.set_running(False)

        # Enable run button only for READY or COMPLETE (reprocess) clips
        can_run = clip.state in (ClipState.READY, ClipState.COMPLETE)
        logger.debug(f"Run button enabled: {can_run} (state={clip.state.value})")
        self._status_bar.update_button_state(
            can_run=can_run,
            has_partial=clip.completed_frame_count() > 0,
            has_in_out=clip.in_out_range is not None,
            batch_count=batch_count if batch_count > 1 else 0,
        )

        # Enable GVM/VideoMaMa buttons based on state
        self._param_panel.set_gvm_enabled(clip.state == ClipState.RAW)
        # VideoMaMa: enable if MASKED, or if mask_asset exists (even if READY/COMPLETE)
        self._param_panel.set_videomama_enabled(
            clip.state == ClipState.MASKED or clip.mask_asset is not None
        )

    @Slot(str)
    def _on_clips_dir_changed(
        self, dir_path: str, *,
        skip_session_restore: bool = False,
        select_clip: str | None = None,
    ) -> None:
        logger.info(f"Scanning clips directory: {dir_path}")
        self._clips_dir = dir_path
        # Reset status bar state on project load (no active job)
        self._status_bar.set_running(False)
        self._status_bar.update_button_state(
            can_run=False, has_partial=False, has_in_out=False,
        )
        # Ensure workspace is visible (may come from welcome screen or menu)
        self._switch_to_workspace()
        try:
            # Detect if this is the Projects root (no standalone videos there)
            from backend.project import projects_root as _projects_root, get_display_name
            is_projects = os.path.normcase(os.path.abspath(dir_path)) == os.path.normcase(
                os.path.abspath(_projects_root())
            )
            clips = self._service.scan_clips(
                dir_path, allow_standalone_videos=not is_projects,
            )
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
                # Select the requested clip (e.g. newly imported), fall back to last
                target = None
                if select_clip:
                    for clip in clips:
                        if clip.name == select_clip:
                            target = clip
                            break
                if target is None:
                    target = clips[-1]  # newest (timestamps sort ascending)
                self._on_clip_selected(target)
            logger.info(f"Found {len(clips)} clips")

            # Register in recent sessions store — per-project, not per-clip
            if is_projects:
                from backend.project import is_v2_project, get_clip_dirs
                # Group clips by their project container
                registered: set[str] = set()
                for clip in clips:
                    # Find the project dir this clip belongs to
                    # v2: clip.root_path is .../project_dir/clips/clip_name
                    # v1: clip.root_path IS the project dir
                    clip_parent = os.path.dirname(clip.root_path)
                    if os.path.basename(clip_parent) == "clips":
                        project_path = os.path.dirname(clip_parent)
                    else:
                        project_path = clip.root_path
                    norm_proj = os.path.normcase(os.path.abspath(project_path))
                    if norm_proj not in registered:
                        registered.add(norm_proj)
                        proj_name = get_display_name(project_path)
                        clip_count = len(get_clip_dirs(project_path))
                        self._recent_store.add_or_update(
                            project_path, proj_name, clip_count,
                        )
            else:
                display_name = os.path.basename(dir_path)
                self._recent_store.add_or_update(dir_path, display_name, len(clips))

            # Auto-load session if exists (Codex: block signals during restore)
            # Skip restore when creating new projects — prevents old session
            # from overriding the newly-added clip selection.
            if not skip_session_restore:
                self._try_auto_load_session(dir_path)

        except Exception as e:
            logger.error(f"Failed to scan clips: {e}")
            QMessageBox.critical(self, "Scan Error", f"Failed to scan clips directory:\n{e}")

    def _switch_to_workspace(self) -> None:
        """Switch from welcome screen to the 3-panel workspace."""
        self._stack.setCurrentIndex(1)
        self._status_bar.show()

    def _try_auto_open_projects(self) -> None:
        """Auto-open the Projects root if it has project folders."""
        from backend.project import projects_root as _projects_root
        try:
            root = _projects_root()
        except Exception:
            return
        if not os.path.isdir(root):
            return
        # Check for at least one subdirectory (project folder)
        has_projects = any(
            os.path.isdir(os.path.join(root, d))
            for d in os.listdir(root)
            if not d.startswith('.') and not d.startswith('_')
        )
        if has_projects:
            self._on_clips_dir_changed(root)

    @Slot(str)
    def _on_welcome_folder(self, dir_path: str) -> None:
        """Handle folder selected from welcome screen."""
        self._switch_to_workspace()
        self._on_clips_dir_changed(dir_path)

    @Slot(str)
    def _on_recent_project_opened(self, workspace_path: str) -> None:
        """Open a workspace from the recent projects list.

        If the path is an individual project folder inside Projects/,
        scan the Projects root so all clips appear in the browser.
        """
        if not os.path.isdir(workspace_path):
            QMessageBox.warning(self, "Missing", f"Workspace no longer exists:\n{workspace_path}")
            self._recent_store.remove(workspace_path)
            self._welcome.refresh_recents()
            return
        self._switch_to_workspace()

        # If this is a project subfolder inside Projects/, scan the root
        from backend.project import projects_root as _projects_root
        try:
            root = _projects_root()
            parent = os.path.normcase(os.path.abspath(os.path.dirname(workspace_path)))
            root_norm = os.path.normcase(os.path.abspath(root))
            if parent == root_norm:
                # Individual project inside Projects/ — scan the root
                self._on_clips_dir_changed(root)
                return
        except Exception:
            pass
        self._on_clips_dir_changed(workspace_path)

    def _return_to_welcome(self) -> None:
        """Save session and return to the welcome screen."""
        if self._clips_dir:
            try:
                self._on_save_session()
            except Exception:
                pass
        self._stack.setCurrentIndex(0)
        self._status_bar.hide()
        self._welcome.refresh_recents()
        self._clips_dir = None
        self._current_clip = None

    @Slot(list)
    def _on_welcome_files(self, file_paths: list) -> None:
        """Handle files selected from welcome screen or +ADD import.

        Creates ONE project folder for all selected videos, with each
        video as a separate clip nested inside clips/.
        """
        if not file_paths:
            return

        from backend.project import projects_root, create_project, is_video_file
        from ui.widgets.preferences_dialog import KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE, get_setting_bool

        root = projects_root()
        copy_source = get_setting_bool(KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE)

        video_paths = [f for f in file_paths if is_video_file(f)]
        if not video_paths:
            return

        # Multi-video: ask user to name the project
        display_name = None
        if len(video_paths) > 1:
            name, ok = QInputDialog.getText(
                self, "Name Your Project",
                "Give your project a name:",
            )
            if not ok:
                return  # user cancelled
            display_name = name.strip() or None

        # Create ONE project with all videos as clips
        project_dir = create_project(
            video_paths, copy_source=copy_source, display_name=display_name,
        )
        logger.info(
            f"Created project with {len(video_paths)} clip(s): "
            f"{os.path.basename(project_dir)} (copy={copy_source})"
        )

        # Scan Projects root — select the last clip from the new project
        self._switch_to_workspace()
        self._on_clips_dir_changed(
            root, skip_session_restore=True, select_clip=None,
        )

    def _on_open_folder(self) -> None:
        self._io_tray.show_add_menu()

    # ── View Controls ──

    def _on_toggle_split(self, checked: bool) -> None:
        """Toggle split view in preview."""
        self._dual_viewer.set_split_mode(checked)

    def _on_reset_zoom(self) -> None:
        """Reset preview zoom to fit."""
        self._dual_viewer.reset_zoom()

    @Slot()
    def _sync_io_divider(self, *_args) -> None:
        """Keep the IO tray divider aligned with the dual viewer splitter.

        Sets the IO tray divider to match the viewer's internal split
        position (between INPUT and OUTPUT viewports).
        """
        viewer_sizes = self._dual_viewer._viewer_splitter.sizes()
        if len(viewer_sizes) < 2:
            return
        self._io_tray.sync_divider(viewer_sizes[0])

    # ── In/Out Markers ──

    def _set_in_point(self) -> None:
        """Set in-point at the current scrubber position."""
        if not self._current_clip:
            return
        idx = self._dual_viewer._scrubber.current_frame()
        _, out = self._dual_viewer.get_in_out()
        # Clamp: in-point cannot exceed out-point
        if out is not None and idx > out:
            idx = out
        self._dual_viewer._scrubber.set_in_point(idx)
        self._persist_in_out()

    def _set_out_point(self) -> None:
        """Set out-point at the current scrubber position."""
        if not self._current_clip:
            return
        idx = self._dual_viewer._scrubber.current_frame()
        in_pt, _ = self._dual_viewer.get_in_out()
        # Clamp: out-point cannot precede in-point
        if in_pt is not None and idx < in_pt:
            idx = in_pt
        self._dual_viewer._scrubber.set_out_point(idx)
        self._persist_in_out()

    def _clear_in_out(self) -> None:
        """Clear in/out markers (Alt+I)."""
        if not self._current_clip:
            return
        self._dual_viewer._scrubber.clear_in_out()
        self._current_clip.in_out_range = None
        from backend.project import save_in_out_range
        save_in_out_range(self._current_clip.root_path, None)
        self._refresh_button_state()

    def _persist_in_out(self) -> None:
        """Save current in/out markers to the clip and project.json."""
        if not self._current_clip:
            return
        in_pt, out_pt = self._dual_viewer.get_in_out()
        if in_pt is not None and out_pt is not None:
            rng = InOutRange(in_point=in_pt, out_point=out_pt)
            self._current_clip.in_out_range = rng
            from backend.project import save_in_out_range
            save_in_out_range(self._current_clip.root_path, rng)
        self._refresh_button_state()

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
        # If multiple clips are selected, dispatch to batch pipeline
        if self._io_tray.selected_count() > 1:
            self._on_run_batch()
            return

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

        # Warn if alpha doesn't cover all input frames
        if clip.alpha_asset and clip.input_asset:
            alpha_count = clip.alpha_asset.frame_count
            input_count = clip.input_asset.frame_count
            if 0 < alpha_count < input_count:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Incomplete Alpha")
                msg.setText(
                    f"Alpha hints cover {alpha_count} of {input_count} frames.\n\n"
                    "You can process the available range, re-run GVM to\n"
                    "regenerate all alpha frames, or cancel."
                )
                btn_process = msg.addButton("Process Available", QMessageBox.AcceptRole)
                btn_rerun = msg.addButton("Re-run GVM", QMessageBox.ActionRole)
                msg.addButton(QMessageBox.Cancel)
                msg.exec()
                clicked = msg.clickedButton()
                if clicked == btn_rerun:
                    # Transition back to RAW and submit GVM job
                    clip.transition_to(ClipState.RAW)
                    self._clip_model.update_clip_state(clip.name, ClipState.RAW)
                    gvm_job = create_job_snapshot(clip, job_type=JobType.GVM_ALPHA)
                    if self._service.job_queue.submit(gvm_job):
                        clip.set_processing(True)
                        self._start_worker_if_needed(
                            gvm_job.id, job_label="GVM Auto", indeterminate=False,
                        )
                    return
                elif clicked != btn_process:
                    return  # Cancel

        # For COMPLETE clips wanting reprocess, transition back to READY
        if clip.state == ClipState.COMPLETE:
            clip.transition_to(ClipState.READY)

        params = self._param_panel.get_params()
        job = create_job_snapshot(clip, params, resume=False)

        # Store output config in job params
        output_config = self._param_panel.get_output_config()
        job.params["_output_config"] = output_config

        # Pass in/out frame range (GVM always processes full clip)
        if clip.in_out_range:
            job.params["_frame_range"] = (
                clip.in_out_range.in_point,
                clip.in_out_range.out_point,
            )

        if not self._service.job_queue.submit(job):
            QMessageBox.information(self, "Duplicate", f"'{clip.name}' is already queued.")
            return

        self._start_worker_if_needed(job.id)

    @Slot()
    def _on_resume_inference(self) -> None:
        """Resume inference — skip already-processed frames, process full clip."""
        if self._current_clip is None:
            return

        clip = self._current_clip
        if clip.state not in (ClipState.READY, ClipState.COMPLETE):
            return

        # For COMPLETE clips, transition back to READY
        if clip.state == ClipState.COMPLETE:
            clip.transition_to(ClipState.READY)

        params = self._param_panel.get_params()
        job = create_job_snapshot(clip, params, resume=True)

        output_config = self._param_panel.get_output_config()
        job.params["_output_config"] = output_config
        # Resume always processes full clip — no in/out range

        if not self._service.job_queue.submit(job):
            QMessageBox.information(self, "Duplicate", f"'{clip.name}' is already queued.")
            return

        self._start_worker_if_needed(job.id)

    def _refresh_button_state(self) -> None:
        """Update run/resume button state based on current clip."""
        clip = self._current_clip
        batch_count = self._io_tray.selected_count()
        if clip is None:
            self._status_bar.update_button_state(
                can_run=False, has_partial=False, has_in_out=False,
            )
            return
        can_run = clip.state in (ClipState.READY, ClipState.COMPLETE)
        self._status_bar.update_button_state(
            can_run=can_run,
            has_partial=clip.completed_frame_count() > 0,
            has_in_out=clip.in_out_range is not None,
            batch_count=batch_count if batch_count > 1 else 0,
        )

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

    def _on_run_batch(self) -> None:
        """Batch pipeline: queue GVM for RAW clips, then inference for READY/COMPLETE.

        For RAW clips, GVM runs first; on completion, inference is auto-chained
        via _batch_clips tracking in _on_worker_clip_finished.
        """
        selected = self._io_tray.get_selected_clips()
        if not selected:
            return

        params = self._param_panel.get_params()
        output_config = self._param_panel.get_output_config()

        # Separate clips by state
        raw_clips = [c for c in selected if c.state == ClipState.RAW]
        ready_clips = [c for c in selected
                       if c.state in (ClipState.READY, ClipState.COMPLETE)]

        # Track RAW clips for GVM→inference auto-chain
        self._batch_clips = {c.name for c in raw_clips}

        queued = 0
        first_job_id = None

        # Phase 1: queue all GVM jobs first
        for clip in raw_clips:
            job = create_job_snapshot(clip, job_type=JobType.GVM_ALPHA)
            if self._service.job_queue.submit(job):
                clip.set_processing(True)
                if first_job_id is None:
                    first_job_id = job.id
                queued += 1

        # Phase 2: queue inference jobs for already-READY clips
        for clip in ready_clips:
            if clip.state == ClipState.COMPLETE:
                clip.transition_to(ClipState.READY)
            job = create_job_snapshot(clip, params)
            job.params["_output_config"] = output_config
            if self._service.job_queue.submit(job):
                if first_job_id is None:
                    first_job_id = job.id
                queued += 1

        if queued > 0:
            label = "Batch" if raw_clips else "Inference"
            self._start_worker_if_needed(first_job_id, job_label=label)
            logger.info(
                f"Batch pipeline queued: {len(raw_clips)} GVM + "
                f"{len(ready_clips)} inference = {queued} jobs"
            )

    @Slot()
    def _on_run_gvm(self) -> None:
        """Run GVM alpha generation on the selected clip."""
        if self._current_clip is None or self._current_clip.state != ClipState.RAW:
            return

        # Detect partial alpha from a previous interrupted run
        alpha_dir = os.path.join(self._current_clip.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir):
            existing = [f for f in os.listdir(alpha_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if existing:
                total = (self._current_clip.input_asset.frame_count
                         if self._current_clip.input_asset else 0)
                reply = QMessageBox.question(
                    self, "Partial Alpha Found",
                    f"Found {len(existing)}/{total} alpha frames from a previous run.\n\n"
                    "GVM will regenerate all frames from scratch.\nContinue?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
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
        self._batch_clips.clear()
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

    @Slot(str, str, str)
    def _on_worker_clip_finished(self, job_id: str, clip_name: str, job_type: str) -> None:
        # Map job type to correct next state
        # GVM/VideoMaMa produce AlphaHint -> clip becomes READY for inference
        if job_type in (JobType.GVM_ALPHA.value, JobType.VIDEOMAMA_ALPHA.value):
            target_state = ClipState.READY
        else:
            target_state = ClipState.COMPLETE

        self._clip_model.update_clip_state(clip_name, target_state)

        # Clear processing lock and rescan assets for pipeline steps
        for clip in self._clip_model.clips:
            if clip.name == clip_name:
                clip.set_processing(False)
                # Rescan assets so alpha hints are discovered
                if target_state == ClipState.READY:
                    try:
                        clip.find_assets()
                    except Exception:
                        pass  # find_assets may update state further
                break

        # Batch auto-chain: GVM completed on a batch clip → queue inference
        if target_state == ClipState.READY and clip_name in self._batch_clips:
            self._batch_clips.discard(clip_name)
            for clip in self._clip_model.clips:
                if clip.name == clip_name:
                    params = self._param_panel.get_params()
                    job = create_job_snapshot(clip, params)
                    job.params["_output_config"] = self._param_panel.get_output_config()
                    if self._service.job_queue.submit(job):
                        logger.info(f"Batch auto-chain: queued inference for {clip_name}")
                    break

        # Stop timer; only exit running state if no more jobs are pending
        has_more = self._service.job_queue.has_pending or self._batch_clips
        self._status_bar.stop_job_timer()
        if not has_more:
            self._status_bar.set_running(False)
        else:
            # Reset progress for next job in the queue
            self._status_bar.reset_progress()
            self._status_bar.start_job_timer(label="Batch")

        if target_state == ClipState.READY:
            type_label = "GVM Auto" if job_type == JobType.GVM_ALPHA.value else "VideoMaMa"
            # Show alpha coverage count
            alpha_info = ""
            for c in self._clip_model.clips:
                if c.name == clip_name and c.alpha_asset and c.input_asset:
                    alpha_info = f" ({c.alpha_asset.frame_count}/{c.input_asset.frame_count} alpha frames)"
                    break
            self._status_bar.set_message(
                f"{type_label} complete for {clip_name}{alpha_info} -- Ready to Run Inference"
            )
        elif target_state == ClipState.COMPLETE:
            self._status_bar.set_message(f"Inference complete: {clip_name}")

        # Refresh views
        self._queue_panel.refresh()
        self._io_tray.refresh()

        # If selected clip, reload preview to show new assets
        if self._current_clip and self._current_clip.name == clip_name:
            self._dual_viewer.set_clip(self._current_clip)
            self._refresh_button_state()
            self._param_panel.set_videomama_enabled(
                self._current_clip.state == ClipState.MASKED
                or self._current_clip.mask_asset is not None
            )

        logger.info(f"Clip finished ({job_type}): {clip_name} -> {target_state.value}")

    @Slot(str, str)
    def _on_worker_warning(self, job_id: str, message: str) -> None:
        if message.startswith("Cancelled:"):
            # Job was cancelled — clear processing lock on the clip
            clip_name = message.removeprefix("Cancelled:").strip()
            for clip in self._clip_model.clips:
                if clip.name == clip_name:
                    clip.set_processing(False)
                    # Refresh viewer to show any partial output from before cancel
                    if self._current_clip and self._current_clip.name == clip_name:
                        clip.find_assets()
                        self._dual_viewer.set_clip(clip)
                        self._refresh_button_state()
                    break
            self._status_bar.stop_job_timer()
            self._status_bar.set_running(False)
            self._status_bar.set_message(f"Cancelled: {clip_name}")
            self._queue_panel.refresh()
            logger.info(f"Job cancelled: {clip_name}")
        else:
            self._status_bar.add_warning(message)
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
                # Refresh viewer to show any partial output
                if self._current_clip and self._current_clip.name == clip_name:
                    clip.find_assets()
                    self._dual_viewer.set_clip(clip)
                break
        self._queue_panel.refresh()
        logger.error(f"Worker error for {clip_name}: {error_msg}")
        QMessageBox.critical(self, "Processing Error", f"Clip: {clip_name}\n\n{error_msg}")

    @Slot()
    def _on_queue_empty(self) -> None:
        self._status_bar.set_running(False)
        self._status_bar.stop_job_timer()
        self._active_job_id = None
        self._batch_clips.clear()
        self._queue_panel.refresh()
        logger.info("All jobs completed")

    # ── Video Extraction ──

    def _auto_extract_clips(self, clips: list[ClipEntry]) -> None:
        """Auto-submit EXTRACTING clips to the extract worker.

        New-format projects (Source/ subdir) already have video in the right
        place — extraction writes to Frames/. Legacy standalone videos
        (root_path == clips_dir) are restructured into a clip subdirectory.
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

            # New format: video already lives in Source/ — no restructuring needed.
            # Legacy standalone video: root_path is the parent dir, not a clip dir.
            # Restructure: create clip_name/ dir and copy video as Input.ext
            source_dir = os.path.join(clip.root_path, "Source")
            if not os.path.isdir(source_dir) and clip.root_path == self._clips_dir:
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
        """Update status bar, clip card, and input viewer progress."""
        pct = int(current / total * 100) if total > 0 else 0
        self._status_bar.set_message(f"Extracting {clip_name}: {pct}%")
        progress = current / total if total > 0 else 0.0
        # Update clip for per-card progress bar
        for i, clip in enumerate(self._clip_model.clips):
            if clip.name == clip_name:
                clip.extraction_progress = progress
                clip.extraction_total = total
                idx = self._clip_model.index(i)
                self._clip_model.dataChanged.emit(idx, idx)
                break
        # Update input viewer overlay if this is the selected clip
        if self._current_clip and self._current_clip.name == clip_name:
            self._dual_viewer.set_extraction_progress(progress, total)

    @Slot(str, int)
    def _on_extract_finished(self, clip_name: str, frame_count: int) -> None:
        """Handle extraction complete — update clip to RAW with image sequence."""
        from backend.clip_state import ClipAsset
        for clip in self._clip_model.clips:
            if clip.name == clip_name:
                # Clear extraction progress
                clip.extraction_progress = 0.0
                clip.extraction_total = 0

                # Clear input viewer overlay
                if self._current_clip and self._current_clip.name == clip_name:
                    self._dual_viewer.set_extraction_progress(0.0, 0)

                # Update input asset to point to extracted sequence
                # Check Frames/ (new format) then Input/ (legacy)
                frames_dir = os.path.join(clip.root_path, "Frames")
                input_dir = os.path.join(clip.root_path, "Input")
                actual_dir = frames_dir if os.path.isdir(frames_dir) else input_dir
                if os.path.isdir(actual_dir):
                    clip.input_asset = ClipAsset(actual_dir, "sequence")

                # Transition EXTRACTING -> RAW
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
                    self._refresh_button_state()

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
                clip.extraction_progress = 0.0
                clip.extraction_total = 0
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

        # Restore splitter sizes (validate: must have 2 panels, none at 0)
        if "splitter_sizes" in data:
            try:
                sizes = [int(s) for s in data["splitter_sizes"]]
                if len(sizes) == 2 and all(s > 0 for s in sizes):
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
            for clip in self._clip_model.clips:
                if clip.name == clip_name:
                    self._on_clip_selected(clip)
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

    def _show_preferences(self) -> None:
        """Open the Preferences dialog and apply changes."""
        dlg = PreferencesDialog(self)
        if dlg.exec() == PreferencesDialog.Accepted:
            self._apply_tooltip_setting()

    def _apply_tooltip_setting(self) -> None:
        """Enable or disable tooltips globally based on saved preference."""
        show = get_setting_bool(KEY_SHOW_TOOLTIPS, DEFAULT_SHOW_TOOLTIPS)
        if not show:
            # Disable: clear all tooltips on the main window tree
            for w in self.findChildren(QWidget):
                w.setToolTip("")
        # If enabled, tooltips stay as set during widget construction.
        # A full re-enable would require rebuilding tooltip strings, which
        # is unnecessary — the setting takes full effect on next app launch.

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About CorridorKey",
            "CorridorKey — AI Green Screen Keyer\n\n"
            "Based on GreenFormer by Corridor Crew\n"
            "CC BY-NC-SA 4.0 License\n\n"
            "PySide6 Desktop Application",
        )

    def paintEvent(self, event) -> None:
        """Paint dithered diagonal gradient background.

        Uses a cached QImage with per-pixel noise dithering to eliminate
        banding on subtle dark gradients. Diagonal: lower-left (darker)
        to upper-right (lighter).
        """
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return

        # Cache the gradient image, regenerate only on resize
        if (not hasattr(self, '_bg_cache')
                or self._bg_cache is None
                or self._bg_cache.width() != w
                or self._bg_cache.height() != h):
            self._bg_cache = self._render_dithered_gradient(w, h)

        painter = QPainter(self)
        painter.drawImage(0, 0, self._bg_cache)
        painter.end()

    def _render_dithered_gradient(self, w: int, h: int) -> QImage:
        """Render a dithered diagonal gradient to a QImage."""
        img = np.empty((h, w, 3), dtype=np.float32)

        # Diagonal parameter: 0 at lower-left, 1 at upper-right
        ys = np.linspace(1, 0, h).reshape(-1, 1)  # top=0 → 1, bottom=1 → 0
        xs = np.linspace(0, 1, w).reshape(1, -1)
        diag = (xs + ys) * 0.5  # 0..1 diagonal

        # Color range: dark edge (10, 9, 0) to lighter center (22, 21, 2)
        r = 10.0 + diag * 12.0
        g = 9.0 + diag * 12.0
        b = 0.0 + diag * 2.0

        img[..., 0] = r
        img[..., 1] = g
        img[..., 2] = b

        # Add triangular dithering noise (±0.5) to break banding
        rng = np.random.default_rng(42)  # fixed seed for stability
        noise = rng.uniform(-0.5, 0.5, (h, w, 3)).astype(np.float32)
        img += noise

        # Clamp and convert to uint8
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)

        # numpy RGB → QImage
        bytes_per_line = w * 3
        qimg = QImage(img_u8.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return qimg.copy()  # deep copy so numpy buffer can be freed

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

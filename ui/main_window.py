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
import re
import shutil
import sys

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QSplitter, QVBoxLayout, QWidget,
    QLabel, QHBoxLayout, QMessageBox, QStackedWidget,
    QProgressBar, QFileDialog, QInputDialog, QGraphicsOpacityEffect,
    QPushButton,
)
from PySide6.QtCore import Qt, Slot, QTimer, QPropertyAnimation, QEasingCurve, QSettings, QThread, Signal
from PySide6.QtGui import QKeySequence, QAction, QImage, QPainter

from backend import (
    CorridorKeyService, ClipEntry, ClipState, InferenceParams,
    InOutRange, OutputConfig, JobType,
    PipelineRoute, classify_pipeline_route, mask_sequence_is_videomama_ready,
)
from backend.project import VIDEO_FILE_FILTER

from ui.models.clip_model import ClipListModel
from ui.preview.frame_index import ViewMode
from ui.preview.display_transform import processed_rgba_to_qimage
from ui.widgets.dual_viewer import DualViewerPanel
from ui.widgets.parameter_panel import ParameterPanel
from ui.widgets.status_bar import StatusBar
from ui.widgets.queue_panel import QueuePanel
from ui.widgets.io_tray_panel import IOTrayPanel
from ui.widgets.welcome_screen import WelcomeScreen
from ui.widgets.preferences_dialog import (
    PreferencesDialog, KEY_SHOW_TOOLTIPS, DEFAULT_SHOW_TOOLTIPS,
    KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE,
    KEY_COPY_SEQUENCES, DEFAULT_COPY_SEQUENCES,
    KEY_TRACKER_MODEL, DEFAULT_TRACKER_MODEL,
    get_setting_bool, get_setting_str,
)
from ui.workers.gpu_job_worker import GPUJobWorker, create_job_snapshot
from ui.workers.gpu_monitor import GPUMonitor
from ui.workers.thumbnail_worker import ThumbnailGenerator
from ui.workers.extract_worker import ExtractWorker
from ui.recent_sessions import RecentSessionsStore
from ui.shortcut_registry import ShortcutRegistry

logger = logging.getLogger(__name__)

# Session file stored in clips dir (Codex: JSON sidecar)
_SESSION_FILENAME = ".corridorkey_session.json"
_SESSION_VERSION = 1


class _Toast(QLabel):
    """Non-blocking notification that auto-fades after a duration. Click to dismiss."""

    def __init__(self, parent: QWidget, text: str, duration_ms: int = 4000,
                 center: bool = False):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background: #1A1900; color: #E0E0D0; border: 1px solid #454430;"
            "border-radius: 6px; padding: 12px 20px; font-size: 13px;"
        )
        self.setFixedWidth(380)
        self.adjustSize()
        if center:
            # Position center of window
            self.move(
                (parent.width() - self.width()) // 2,
                (parent.height() - self.height()) // 2,
            )
        else:
            # Position bottom-center, above the status bar
            self.move(
                (parent.width() - self.width()) // 2,
                parent.height() - self.height() - 60,
            )
        # Fade-out animation
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity)
        self._fade = QPropertyAnimation(self._opacity, b"opacity")
        self._fade.setDuration(800)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.setEasingCurve(QEasingCurve.InQuad)
        self._fade.finished.connect(self.deleteLater)
        # Start fade after hold duration
        QTimer.singleShot(duration_ms, self._fade.start)
        self.show()
        self.raise_()

    def mousePressEvent(self, event):
        self.deleteLater()


class _MuteOverlay(QLabel):
    """Brief overlay inside the brand bar strip, fades after 1.5s."""

    _FIXED_W = 160
    _FIXED_H = 22

    def __init__(self, parent: QWidget, text: str):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(self._FIXED_W, self._FIXED_H)
        self.setStyleSheet(
            "background: rgba(26, 25, 0, 220); color: #E0E0D0;"
            "border: 1px solid #454430; border-radius: 4px;"
            "font-family: 'Open Sans'; font-size: 11px; font-weight: 600;"
        )
        # Position inside the brand bar (top strip, ~24px tall)
        menu_h = parent.menuBar().height() if hasattr(parent, 'menuBar') else 22
        self.move((parent.width() - self._FIXED_W) // 2, menu_h + 2)
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity)
        self._fade = QPropertyAnimation(self._opacity, b"opacity")
        self._fade.setDuration(600)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.setEasingCurve(QEasingCurve.InQuad)
        self._fade.finished.connect(self.deleteLater)
        QTimer.singleShot(1500, self._fade.start)
        self.raise_()


class _UpdateChecker(QThread):
    """Background thread that checks GitHub for a newer version."""
    update_available = Signal(str)  # emits the remote version string

    def __init__(self, local_version: str):
        super().__init__()
        self._local = local_version

    def run(self):
        try:
            import urllib.request
            url = (
                "https://raw.githubusercontent.com/edenaion/EZ-CorridorKey"
                "/main/pyproject.toml"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "CorridorKey"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                text = resp.read().decode("utf-8")
            for line in text.splitlines():
                if line.strip().startswith("version"):
                    remote = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if self._is_newer(remote, self._local):
                        self.update_available.emit(remote)
                    return
        except Exception:
            pass  # silently fail — no internet, no button

    @staticmethod
    def _is_newer(remote: str, local: str) -> bool:
        """Simple semver comparison: 1.4.0 > 1.3.1."""
        try:
            r = tuple(int(x) for x in remote.split("."))
            l = tuple(int(x) for x in local.split("."))
            return r > l
        except (ValueError, AttributeError):
            return False


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
        self._cancel_requested_job_id: str | None = None
        self._force_stop_armed = False
        self._skip_shutdown_cleanup = False
        self._bg_cache: QImage | None = None
        # Batch pipeline: clip_name -> remaining queued steps after the current one.
        self._pipeline_steps: dict[str, list[JobType]] = {}
        # Debug console — created eagerly so log handler captures from startup
        from ui.widgets.debug_console import DebugConsoleWidget
        self._debug_console = DebugConsoleWidget()

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

        # Live selected-clip refresh: coalesce worker progress into a cheap UI-only
        # asset rescan so the scrubber and mode buttons update while frames are written.
        self._live_asset_refresh_timer = QTimer(self)
        self._live_asset_refresh_timer.setSingleShot(True)
        self._live_asset_refresh_timer.setInterval(150)
        self._live_asset_refresh_timer.timeout.connect(self._refresh_selected_clip_live_assets)
        self._pending_live_asset_refresh_clip: str | None = None

        # Shortcut registry — single source of truth for key bindings
        self._shortcut_registry = ShortcutRegistry()

        # Build UI
        self._build_menu_bar()
        self._build_central()
        self._build_status_bar()
        self._setup_shortcuts()

        # Workers
        self._gpu_worker = GPUJobWorker(self._service, parent=self)
        self._gpu_monitor = GPUMonitor(interval_ms=2000, parent=self)
        self._extract_worker = ExtractWorker(parent=self)
        self._extract_progress: dict[str, tuple[int, int]] = {}  # clip_name -> (current, total)

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

        # Run startup diagnostics (deferred so the window is visible first)
        if os.environ.get("CORRIDORKEY_SKIP_STARTUP_DIAGNOSTICS") != "1":
            QTimer.singleShot(500, lambda: self._run_startup_diagnostics(device))

        # Always start on welcome screen — user picks a project from recents or imports
        # Deferred sync of IO tray divider with viewer splitter
        QTimer.singleShot(0, self._sync_io_divider)

        # Apply persisted preferences (e.g. tooltip visibility, sound mute)
        self._apply_tooltip_setting()
        self._apply_sound_setting()
        self._apply_tracker_model_setting()

        # Check for updates (non-blocking background thread)
        if os.environ.get("CORRIDORKEY_SKIP_UPDATE_CHECK") != "1":
            self._check_for_updates()

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")
        import_menu = file_menu.addMenu("Import Clips")
        import_menu.addAction("Import Folder...", self._on_import_folder)
        import_menu.addAction("Import Video(s)...", self._on_import_videos)
        import_menu.addAction("Import Image Sequence...", self._on_import_image_sequence)
        file_menu.addSeparator()

        # Session save/load (shortcuts managed by registry)
        self._save_action = file_menu.addAction("Save Session", self._on_save_session)
        self._open_action = file_menu.addAction("Open Project...", self._on_open_project)

        file_menu.addSeparator()
        file_menu.addAction("Export Video...", self._on_export_video)
        file_menu.addSeparator()
        file_menu.addAction("Return to Home", self._return_to_welcome)
        file_menu.addAction("Exit", self.close)

        # Edit menu
        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addAction("Preferences...", self._show_preferences)
        edit_menu.addAction("Hotkeys...", self._show_hotkeys)
        edit_menu.addSeparator()
        edit_menu.addAction("Track Annotation Masks", self._on_track_masks)
        edit_menu.addAction("Clear Annotations", self._on_clear_annotations)

        # View menu
        view_menu = menu_bar.addMenu("View")
        view_menu.addAction("Reset Layout", self._reset_layout)
        view_menu.addAction("Toggle Queue Panel", self._toggle_queue_panel)

        view_menu.addAction("Reset Zoom", self._on_reset_zoom)

        # Help menu
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("Console", self._toggle_debug_console)
        help_menu.addSeparator()
        help_menu.addAction("Report Issue...", self._show_report_issue)
        help_menu.addSeparator()
        help_menu.addAction("About", self._show_about)

        # Click sound on any menu action
        menu_bar.triggered.connect(lambda _: self._menu_click_sound())

        # Right corner: update button (hidden) + volume control
        self._corner_widget = QWidget()
        corner_layout = QHBoxLayout(self._corner_widget)
        corner_layout.setContentsMargins(0, 0, 4, 0)
        corner_layout.setSpacing(8)

        self._update_btn = QPushButton("Update Available")
        self._update_btn.setVisible(False)
        self._update_btn.setCursor(Qt.PointingHandCursor)
        self._update_btn.setStyleSheet(
            "QPushButton {"
            "  background: #FFF203; color: #141300; border: none;"
            "  border-radius: 3px; padding: 2px 10px;"
            "  font-family: 'Open Sans'; font-size: 11px; font-weight: 700;"
            "}"
            "QPushButton:hover { background: #E0D600; }"
        )
        self._update_btn.clicked.connect(self._run_update)
        corner_layout.addWidget(self._update_btn)

        from ui.widgets.volume_control import VolumeControl
        self._volume_control = VolumeControl(self._corner_widget)
        corner_layout.addWidget(self._volume_control)

        menu_bar.setCornerWidget(self._corner_widget)

    def _menu_click_sound(self) -> None:
        from ui.sounds.audio_manager import UIAudio
        UIAudio.click()

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

        # Queue panel — floating overlay on the left edge of the workspace
        self._queue_panel = QueuePanel(self._service.job_queue, parent=workspace)
        self._queue_panel.raise_()

        self._stack.addWidget(workspace)

        # Keep the queue panel sized to the workspace height
        self._workspace = workspace

        main_layout.addWidget(self._stack, 1)

    def _build_status_bar(self) -> None:
        self._status_bar = StatusBar()
        self.centralWidget().layout().addWidget(self._status_bar)
        # Hidden until user opens a project (welcome screen has no use for it)
        self._status_bar.hide()

    def _setup_shortcuts(self) -> None:
        """Wire keyboard shortcuts from the centralized registry."""
        self._shortcut_registry.create_shortcuts(self)
        # Sync menu-bar QAction shortcuts (display text + activation)
        reg = self._shortcut_registry
        self._save_action.setShortcut(QKeySequence(reg.get_key("save_session")))
        self._open_action.setShortcut(QKeySequence(reg.get_key("open_project")))

    def _toggle_mute(self) -> None:
        """Toggle UI sounds on/off and show a brief overlay indicator."""
        from ui.sounds.audio_manager import UIAudio
        from ui.widgets.preferences_dialog import KEY_UI_SOUNDS
        from PySide6.QtCore import QSettings
        muted = not UIAudio.is_muted()
        UIAudio.set_muted(muted)
        QSettings().setValue(KEY_UI_SOUNDS, not muted)
        # Sync the menu-bar volume control
        if hasattr(self, "_volume_control"):
            self._volume_control.sync_mute_state()
        # Show overlay top-right
        icon = "\U0001F507" if muted else "\U0001F50A"  # muted vs speaker
        text = f"{icon}  Sound {'OFF' if muted else 'ON'}"
        overlay = _MuteOverlay(self, text)
        overlay.show()

    def _toggle_playback(self) -> None:
        """Forward Space key to the scrubber's play/pause toggle."""
        self._dual_viewer.toggle_playback()

    def _on_escape(self) -> None:
        """Escape: cancel the current action — auto-detects what's running."""
        # 1. Exit annotation mode (no confirmation needed)
        iv = self._dual_viewer.input_viewer
        if iv.annotation_mode:
            iv.set_annotation_mode(None)
            return

        # 2. Detect active process and ask to cancel
        process_name = self._detect_active_process()
        if not process_name:
            return

        reply = QMessageBox.question(
            self, "Cancel",
            f"Cancel {process_name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        from ui.sounds.audio_manager import UIAudio
        UIAudio.user_cancel()

        if process_name == "frame extraction":
            self._cancel_extraction()
        else:
            self._cancel_inference()

    def _detect_active_process(self) -> str | None:
        """Return a human-readable name for the currently active process, or None."""
        # Check extraction — is_busy means actively extracting or has pending jobs
        if self._extract_worker.is_busy:
            return "frame extraction"
        # Check inference / GPU jobs
        queue = self._service.job_queue
        if queue.current_job or queue.has_pending:
            return "processing"
        return None

    def _cancel_extraction(self) -> None:
        """Stop frame extraction and reset the worker."""
        old = self._extract_worker
        # Disconnect old signals to prevent stale deliveries
        try:
            old.progress.disconnect(self._on_extract_progress)
            old.finished.disconnect(self._on_extract_finished)
            old.error.disconnect(self._on_extract_error)
        except RuntimeError:
            pass  # already disconnected
        old.stop()
        if not old.wait(5000):
            logger.warning("Extract worker did not stop in 5s — terminating")
            old.terminate()
            old.wait(2000)
        # Restart the worker thread (stop kills the thread loop)
        self._extract_worker = ExtractWorker(parent=self)
        self._extract_worker.progress.connect(self._on_extract_progress)
        self._extract_worker.finished.connect(self._on_extract_finished)
        self._extract_worker.error.connect(self._on_extract_error)
        self._extract_progress.clear()
        self._status_bar.reset_progress()
        # Clear extraction overlay on the viewer
        self._dual_viewer.set_extraction_progress(0.0, 0)
        # Reset any EXTRACTING clips back to their original state
        for clip in self._clip_model.clips:
            if clip.state == ClipState.EXTRACTING:
                clip.extraction_progress = 0.0
                clip.extraction_total = 0
                # Check if frames already exist → RAW, else back to VIDEO
                frames_dir = os.path.join(clip.root_path, "Frames")
                input_dir = os.path.join(clip.root_path, "Input")
                has_frames = (os.path.isdir(frames_dir) and os.listdir(frames_dir)) or \
                             (os.path.isdir(input_dir) and os.listdir(input_dir))
                new_state = ClipState.RAW if has_frames else ClipState.ERROR
                self._clip_model.update_clip_state(clip.name, new_state)
        self._io_tray.refresh()
        self._refresh_button_state()
        logger.info("Frame extraction cancelled by user")

    def _cancel_inference(self) -> None:
        """Cancel all inference/GPU jobs."""
        queue = self._service.job_queue
        current_job = queue.current_job
        is_videomama = (queue.current_job
                        and queue.current_job.job_type == JobType.VIDEOMAMA_ALPHA)
        self._cancel_requested_job_id = current_job.id if current_job is not None else None
        queue.cancel_all()
        self._pipeline_steps.clear()
        self._status_bar.stop_job_timer()
        if current_job is not None:
            self._force_stop_armed = True
            self._status_bar.set_running(True)
            self._status_bar.set_stop_button_mode(force=True)
            self._status_bar.set_message(
                "Stop requested — waiting for current GPU step. "
                "Press FORCE STOP to relaunch if it stays stuck."
            )
        else:
            self._force_stop_armed = False
            self._status_bar.set_running(False)
            self._status_bar.set_message("Cancelled queued work.")
        self._queue_panel.refresh()
        logger.info("Processing cancelled by user")
        if is_videomama:
            _Toast(self, "GPU is finishing the current chunk.\n"
                         "VideoMaMa will stop after it completes.",
                   center=True)

    def _force_restart_app(self) -> None:
        """Hard-stop a blocked GPU phase by relaunching the app process."""
        import subprocess
        import sys
        from PySide6.QtWidgets import QApplication

        try:
            self._auto_save_annotations()
        except Exception:
            logger.exception("Force stop: failed to auto-save annotations")

        try:
            self._auto_save_session()
        except Exception:
            logger.exception("Force stop: failed to auto-save session")

        if getattr(sys, "frozen", False):
            cmd = [sys.executable, *sys.argv[1:]]
            cwd = os.path.dirname(sys.executable)
        else:
            cmd = [sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]]
            cwd = os.path.dirname(os.path.abspath(sys.argv[0]))

        kwargs: dict = {"cwd": cwd}
        if os.name == "nt":
            kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0)
            )
        else:
            kwargs["start_new_session"] = True

        logger.warning("Force stop: relaunching application to break blocked GPU job")
        try:
            subprocess.Popen(cmd, **kwargs)
        except Exception as e:
            logger.exception(f"Force stop relaunch failed: {e}")
            QMessageBox.critical(
                self,
                "Force Stop Failed",
                "Could not relaunch the app automatically.\n\n"
                "Please close and reopen CorridorKey manually.",
            )
            return

        self._skip_shutdown_cleanup = True
        self._status_bar.set_message("Force restarting...")
        QApplication.instance().quit()

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

    def _cycle_fg_color(self) -> None:
        """Hotkey C: cycle foreground annotation color (green/blue)."""
        from ui.widgets.annotation_overlay import cycle_fg_color
        name = cycle_fg_color()
        self._dual_viewer.input_viewer.update()
        self._show_toast(f"Foreground color: {name}")

    def _auto_save_annotations(self) -> None:
        """Auto-save annotation strokes to disk after changes."""
        if self._current_clip is not None:
            iv = self._dual_viewer.input_viewer
            iv.annotation_model.save(self._current_clip.root_path)
            manifest_path = os.path.join(
                self._current_clip.root_path,
                ".corridorkey_mask_manifest.json",
            )
            if os.path.isfile(manifest_path):
                os.remove(manifest_path)
            _has_mask = self._clip_has_videomama_ready_mask(self._current_clip)
            self._param_panel.set_videomama_enabled(_has_mask)
            self._param_panel.set_matanyone2_enabled(_has_mask)

    def _clip_has_videomama_ready_mask(self, clip: ClipEntry | None) -> bool:
        """True when the clip has a dense mask track that alpha generators can consume."""
        if clip is None or clip.mask_asset is None:
            return False
        # mask_asset is set only when VideoMamaMaskHint/ has actual frames —
        # that's sufficient to enable alpha generators regardless of manifest.
        return True

    def _undo_annotation(self) -> None:
        """Ctrl+Z: undo last annotation stroke on current frame."""
        iv = self._dual_viewer.input_viewer
        if iv.annotation_mode and iv.current_stem_index >= 0:
            if iv.annotation_model.undo(iv.current_stem_index):
                iv._split_view.update()
                self._auto_save_annotations()

    def _on_track_masks(self) -> None:
        """Preview SAM2 on the annotated frame, then confirm full tracking."""
        clip = self._current_clip
        if clip is None:
            return
        self._auto_save_annotations()
        iv = self._dual_viewer.input_viewer
        model = iv.annotation_model
        if not model.has_annotations():
            QMessageBox.information(
                self, "No Annotations",
                "Paint green (1) and red (2) strokes on frames first.",
            )
            return

        job = create_job_snapshot(clip, self._param_panel.get_params(), job_type=JobType.SAM2_PREVIEW)
        job.params["_frame_index"] = max(0, self._dual_viewer.current_stem_index)
        if not self._service.job_queue.submit(job):
            return

        self._start_worker_if_needed(job.id, job_label="Track Preview")

    def _submit_sam2_track_job(self, clip: ClipEntry) -> bool:
        """Queue the full SAM2 tracking job after preview confirmation."""
        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir):
            reply = QMessageBox.question(
                self, "Replace Existing Alpha?",
                "This clip already has an AlphaHint (from GVM or a previous run).\n\n"
                "Tracking a new mask sequence will replace that alpha hint.\n\n"
                "Remove existing AlphaHint and proceed?",
            )
            if reply != QMessageBox.Yes:
                return False
            shutil.rmtree(alpha_dir, ignore_errors=True)
            clip.alpha_asset = None
            clip.find_assets()
            self._refresh_button_state()
            logger.info("Removed existing AlphaHint/ before SAM2 tracking")

        job = create_job_snapshot(clip, self._param_panel.get_params(), job_type=JobType.SAM2_TRACK)
        if not self._service.job_queue.submit(job):
            return False

        clip.set_processing(True)
        self._start_worker_if_needed(job.id, job_label="Track Mask")
        return True

    @staticmethod
    def _sam2_preview_qimage(frame_rgb: np.ndarray, mask: np.ndarray) -> QImage:
        """Render a contour-only SAM2 preview for the output viewer."""
        overlay = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (0, 230, 255), 4)
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()

    def _confirm_clear_annotations(self) -> None:
        """Ctrl+C: choose to clear annotations on this frame, entire clip, or cancel."""
        iv = self._dual_viewer.input_viewer
        model = iv.annotation_model
        if not model or not model.has_annotations():
            return

        stem_idx = iv._split_view._annotation_stem_idx
        has_frame = model.has_annotations(stem_idx)

        box = QMessageBox(self)
        box.setWindowTitle("Clear Annotations")
        box.setText("What would you like to clear?")
        frame_btn = box.addButton("This Frame", QMessageBox.AcceptRole)
        clip_btn = box.addButton("Entire Clip", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)

        # Disable "This Frame" if current frame has no annotations
        if not has_frame:
            frame_btn.setEnabled(False)

        box.exec()
        clicked = box.clickedButton()

        if clicked == frame_btn:
            model.clear(stem_idx)
            iv._split_view.update()
            self._update_annotation_info()
        elif clicked == clip_btn:
            self._on_clear_annotations()

    def _on_clear_annotations(self) -> None:
        """Clear all annotations on the current clip and remove tracked masks."""
        iv = self._dual_viewer.input_viewer
        iv.clear_annotations()
        self._update_annotation_info()

        # Remove exported mask directory so VideoMaMa button disables
        clip = self._current_clip
        if clip is not None:
            mask_dir = os.path.join(clip.root_path, "VideoMamaMaskHint")
            if os.path.isdir(mask_dir):
                shutil.rmtree(mask_dir, ignore_errors=True)
                logger.info(f"Removed mask hints: {mask_dir}")
            manifest_path = os.path.join(clip.root_path, ".corridorkey_mask_manifest.json")
            if os.path.isfile(manifest_path):
                os.remove(manifest_path)
            clip.mask_asset = None
            clip.find_assets()
            _has_mask = self._clip_has_videomama_ready_mask(clip)
            self._param_panel.set_videomama_enabled(_has_mask)
            self._param_panel.set_matanyone2_enabled(_has_mask)

    def _update_annotation_info(self) -> None:
        """Update parameter panel with current annotation count and scrubber markers."""
        iv = self._dual_viewer.input_viewer
        model = iv.annotation_model
        fi = iv._frame_index
        total = fi.frame_count if fi else 0
        self._param_panel.set_annotation_info(
            model.annotated_frame_count(), total
        )
        # Push annotation coverage to the scrubber timeline
        if fi and total > 0:
            annotated = [model.has_annotations(i) for i in range(total)]
            self._dual_viewer._scrubber.set_annotation_markers(annotated)
        else:
            self._dual_viewer._scrubber.set_annotation_markers([])

    def _connect_signals(self) -> None:
        # I/O tray — clip selection, import, drag-and-drop
        self._io_tray.clip_clicked.connect(self._on_tray_clip_clicked)
        self._io_tray.selection_changed.connect(self._on_selection_changed)
        self._io_tray.clips_dir_changed.connect(self._on_tray_folder_imported)
        self._io_tray.files_imported.connect(self._on_tray_files_imported)
        self._io_tray.sequence_folder_imported.connect(self._on_sequence_folder_imported)
        self._io_tray.image_files_dropped.connect(self._on_image_files_dropped)
        self._io_tray.extract_requested.connect(self._on_extract_requested)
        self._io_tray.export_video_requested.connect(self._on_export_video)
        self._io_tray.reset_in_out_requested.connect(self._on_reset_all_in_out)

        # Status bar buttons
        self._status_bar.run_clicked.connect(self._on_run_inference)
        self._status_bar.extract_clicked.connect(self._on_extract_current_clip)
        self._status_bar.resume_clicked.connect(self._on_resume_inference)
        self._status_bar.stop_clicked.connect(self._on_stop_inference)

        # GPU worker signals
        self._gpu_worker.progress.connect(self._on_worker_progress)
        self._gpu_worker.preview_ready.connect(self._on_worker_preview)
        self._gpu_worker.clip_finished.connect(self._on_worker_clip_finished)
        self._gpu_worker.warning.connect(self._on_worker_warning)
        self._gpu_worker.status_update.connect(self._on_worker_status)
        self._gpu_worker.error.connect(self._on_worker_error)
        self._gpu_worker.queue_empty.connect(self._on_queue_empty)
        self._gpu_worker.reprocess_result.connect(self._on_reprocess_result)

        # GPU monitor → top bar widgets (not status bar)
        self._gpu_monitor.vram_updated.connect(self._update_vram)
        self._gpu_monitor.gpu_name.connect(self._set_gpu_name)

        # Queue panel cancel signals
        self._queue_panel.cancel_job_requested.connect(self._on_cancel_job)

        # Parameter panel — wire GVM / Track Mask / VideoMaMa
        self._param_panel.gvm_requested.connect(self._on_run_gvm)
        self._param_panel.videomama_requested.connect(self._on_run_videomama)
        self._param_panel.matanyone2_requested.connect(self._on_run_matanyone2)
        self._param_panel.track_masks_requested.connect(self._on_track_masks)
        self._param_panel.import_alpha_requested.connect(self._on_import_alpha)

        # Annotation stroke finished → update annotation counter + auto-save
        self._dual_viewer.input_viewer._split_view.stroke_finished.connect(
            self._update_annotation_info
        )
        self._dual_viewer.input_viewer._split_view.stroke_finished.connect(
            self._auto_save_annotations
        )

        # Parameter panel — live reprocess (debounced, Codex: coalesce stale)
        self._param_panel.params_changed.connect(self._on_params_changed)
        self._param_panel._live_preview.toggled.connect(self._on_live_preview_toggled)

        # Sync IO tray divider with dual viewer splitter
        self._dual_viewer._viewer_splitter.splitterMoved.connect(self._sync_io_divider)

        # Reposition queue overlay when vertical splitter is dragged
        self._vsplitter.splitterMoved.connect(self._position_queue_panel)

        # Scrubber in/out marker drags → persist + refresh button state
        scrubber = self._dual_viewer._scrubber
        scrubber.in_point_changed.connect(lambda _: self._persist_in_out())
        scrubber.out_point_changed.connect(lambda _: self._persist_in_out())
        scrubber.range_cleared.connect(self._clear_in_out)

        # Extract worker signals
        self._extract_worker.progress.connect(self._on_extract_progress)
        self._extract_worker.finished.connect(self._on_extract_finished)
        self._extract_worker.error.connect(self._on_extract_error)

    # ── GPU Header ──

    @Slot(dict)
    def _update_vram(self, info: dict) -> None:
        """Update VRAM meter in the top bar."""
        self._last_vram_info = info  # stash for Report Issue dialog
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
            # Check if any clip needs alpha generation (pipeline mode)
            needs_pipeline = any(
                classify_pipeline_route(c) not in (
                    PipelineRoute.INFERENCE_ONLY, PipelineRoute.SKIP)
                for c in clips
            )
            self._status_bar.update_button_state(
                can_run=True,
                has_partial=False,
                has_in_out=False,
                batch_count=batch_count,
                needs_pipeline=needs_pipeline,
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

        # Refresh annotation coverage bar (annotations loaded from disk above)
        self._update_annotation_info()

        # Ensure run/stop buttons are in correct visibility state
        # (guards against stale running state from crashed jobs)
        if not self._gpu_worker.isRunning():
            self._status_bar.set_running(False)

        # Enable run button only for READY or COMPLETE (reprocess) clips
        can_run = clip.state in (ClipState.READY, ClipState.COMPLETE)
        needs_extraction = clip.state == ClipState.EXTRACTING
        logger.debug(f"Run button enabled: {can_run} (state={clip.state.value})")
        self._status_bar.update_button_state(
            can_run=can_run,
            has_partial=clip.completed_frame_count() > 0,
            has_in_out=clip.in_out_range is not None,
            batch_count=batch_count if batch_count > 1 else 0,
            needs_extraction=needs_extraction,
        )

        # Default standalone EXR sequences to Linear, but keep extracted video
        # EXRs in sRGB-range mode because FFmpeg writes video values as-is.
        if clip.input_asset is not None:
            self._param_panel.auto_detect_color_space(clip.should_default_input_linear())

        # Enable GVM/VideoMaMa/MatAnyone2/Import Alpha buttons based on state
        self._param_panel.set_gvm_enabled(clip.state in (ClipState.RAW, ClipState.MASKED))
        has_mask = self._clip_has_videomama_ready_mask(clip)
        self._param_panel.set_videomama_enabled(has_mask)
        self._param_panel.set_matanyone2_enabled(has_mask)
        self._param_panel.set_import_alpha_enabled(
            clip.state in (ClipState.RAW, ClipState.MASKED, ClipState.READY)
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
                from backend.project import is_v2_project as _is_v2
                if _is_v2(dir_path):
                    display_name = get_display_name(dir_path)
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
            from ui.sounds.audio_manager import UIAudio
            UIAudio.error()
            QMessageBox.critical(self, "Scan Error", f"Failed to scan clips directory:\n{e}")

    def _switch_to_workspace(self) -> None:
        """Switch from welcome screen to the 3-panel workspace."""
        self._stack.setCurrentIndex(1)
        self._status_bar.show()
        # Sync IO tray divider and position queue overlay after layout settles
        QTimer.singleShot(0, self._sync_io_divider)
        QTimer.singleShot(0, self._position_queue_panel)

    @Slot(str)
    def _on_welcome_folder(self, dir_path: str) -> None:
        """Handle folder selected from welcome screen."""
        self._switch_to_workspace()
        self._on_clips_dir_changed(dir_path)

    @Slot(str)
    def _on_recent_project_opened(self, workspace_path: str) -> None:
        """Open a workspace from the recent projects list.

        Opens the specific project folder directly — NOT the Projects root —
        so only that project's clips appear in the browser (project isolation).
        """
        if not os.path.isdir(workspace_path):
            QMessageBox.warning(self, "Missing", f"Workspace no longer exists:\n{workspace_path}")
            self._recent_store.remove(workspace_path)
            self._welcome.refresh_recents()
            return
        self._switch_to_workspace()
        self._on_clips_dir_changed(workspace_path)

    def _on_delete_selected_clips(self) -> None:
        """Delete key — open remove dialog for selected clips."""
        if not self._clips_dir:
            return
        selected = self._io_tray.get_selected_clips()
        if not selected:
            return
        self._io_tray._remove_dialog(selected)

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
    def _on_tray_folder_imported(self, dir_path: str) -> None:
        """Handle folder from I/O tray +ADD button — context-aware."""
        if self._clips_dir:
            self._add_folder_to_project(dir_path)
        else:
            self._on_clips_dir_changed(dir_path)

    def _on_tray_files_imported(self, file_paths: list) -> None:
        """Handle files from I/O tray +ADD button — context-aware."""
        if self._clips_dir:
            self._add_videos_to_project(file_paths)
        else:
            self._on_welcome_files(file_paths)

    def _on_welcome_files(self, file_paths: list) -> None:
        """Handle files selected from welcome screen — creates a new project.

        Creates ONE project folder for all selected media (videos and/or images),
        with each video as a separate clip nested inside clips/.
        Image files are handled via the image_files_dropped flow.
        """
        if not file_paths:
            return

        from backend.project import create_project, is_video_file, is_image_file

        # Separate videos from images
        video_paths = [f for f in file_paths if is_video_file(f)]
        image_paths = [f for f in file_paths if is_image_file(f)]

        # If only images were selected, route to image handler
        if not video_paths and image_paths:
            self._on_image_files_dropped(image_paths)
            return

        if not video_paths:
            return

        copy_source = get_setting_bool(KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE)

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

        # Open the new project (not the Projects root — each project is isolated)
        self._switch_to_workspace()
        self._on_clips_dir_changed(
            project_dir, skip_session_restore=True, select_clip=None,
        )

    def _on_import_folder(self) -> None:
        """File → Import Clips → Import Folder — context-aware."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Clips Directory", "",
            QFileDialog.ShowDirsOnly,
        )
        if not dir_path:
            return
        if self._clips_dir:
            # Already in a project — add folder contents as clips
            self._add_folder_to_project(dir_path)
        else:
            # Welcome screen — create a project named after the folder
            self._create_project_from_folder(dir_path)

    def _on_import_videos(self) -> None:
        """File → Import Clips → Import Video(s) — context-aware."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "",
            VIDEO_FILE_FILTER,
        )
        if not paths:
            return
        if self._clips_dir:
            # Already in a project — add videos to current project
            self._add_videos_to_project(paths)
        else:
            # Welcome screen — create a new project
            self._on_welcome_files(paths)

    def _on_import_image_sequence(self) -> None:
        """File → Import Clips → Import Image Sequence — context-aware."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Image Sequence Folder", "",
            QFileDialog.ShowDirsOnly,
        )
        if not dir_path:
            return
        # When no project is open, prompt for a name (pre-filled with folder name)
        display_name = None
        if not self._clips_dir:
            folder_name = os.path.basename(dir_path.rstrip("/\\"))
            name, ok = QInputDialog.getText(
                self, "Name Your Project",
                "Give your project a name:",
                text=folder_name,
            )
            if not ok:
                return
            display_name = name.strip() or folder_name
        self._on_sequence_folder_imported(dir_path, display_name=display_name)

    def _add_folder_to_project(self, dir_path: str) -> None:
        """Import all videos and image sequences from a folder into the current project."""
        from backend.project import (
            is_video_file, add_clips_to_project,
            folder_has_image_sequence, add_sequences_to_project,
        )
        videos = [
            os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if is_video_file(os.path.join(dir_path, f))
        ]
        is_seq = folder_has_image_sequence(dir_path)

        if not videos and not is_seq:
            QMessageBox.information(
                self, "No Media",
                "No video files or image sequences found in that folder."
            )
            return

        if videos:
            copy_source = get_setting_bool(KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE)
            add_clips_to_project(self._clips_dir, videos, copy_source=copy_source)
            logger.info(f"Added {len(videos)} video clip(s) from folder to project")

        if is_seq:
            copy_seq = get_setting_bool(KEY_COPY_SEQUENCES, DEFAULT_COPY_SEQUENCES)
            add_sequences_to_project(
                self._clips_dir, [dir_path], copy_source=copy_seq,
            )
            logger.info(f"Added image sequence from folder to project")

        self._on_clips_dir_changed(self._clips_dir, skip_session_restore=True)

    def _add_videos_to_project(self, file_paths: list) -> None:
        """Import selected video files into the current project."""
        from backend.project import (
            is_video_file, add_clips_to_project, find_clip_by_source,
            find_removed_clip_by_source, clear_removed_clip,
        )
        videos = [f for f in file_paths if is_video_file(f)]
        if not videos:
            return

        # Categorise: already active, removed (restore), or genuinely new
        new_videos = []
        skipped = []
        restored = []
        project_dir = self._clips_dir  # project root (contains clips/ subdir)
        for v in videos:
            existing = find_clip_by_source(self._clips_dir, v)
            if existing:
                skipped.append(existing)
                continue
            # Check if this was previously removed — restore instead of
            # creating a duplicate folder
            removed_folder = find_removed_clip_by_source(project_dir, v)
            if removed_folder:
                clear_removed_clip(project_dir, removed_folder)
                restored.append(removed_folder)
                continue
            new_videos.append(v)

        if skipped and not new_videos and not restored:
            names = ", ".join(f'"{s}"' for s in skipped[:3])
            QMessageBox.information(
                self, "Already Imported",
                f"All selected videos are already in the project ({names})."
            )
            return

        if new_videos:
            copy_source = get_setting_bool(KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE)
            add_clips_to_project(self._clips_dir, new_videos, copy_source=copy_source)

        parts = []
        if new_videos:
            parts.append(f"{len(new_videos)} added")
        if restored:
            parts.append(f"{len(restored)} restored")
        if skipped:
            parts.append(f"{len(skipped)} duplicate(s) skipped")
        logger.info(f"Import: {', '.join(parts)}")

        if new_videos or restored:
            self._on_clips_dir_changed(self._clips_dir, skip_session_restore=True)

    @Slot(str)
    def _on_sequence_folder_imported(
        self, folder_path: str, display_name: str | None = None,
    ) -> None:
        """Handle image sequence folder from +ADD menu or drag-drop."""
        from backend.project import (
            folder_has_image_sequence, validate_sequence_stems,
            count_sequence_frames, add_sequences_to_project,
            create_project_from_media, find_clip_by_source,
        )

        if not folder_has_image_sequence(folder_path):
            QMessageBox.information(
                self, "No Images",
                "No image files found in that folder.\n\n"
                "Supported formats: PNG, JPG, EXR, TIF, TIFF, BMP, DPX"
            )
            return

        # Check if this source is already in the project (skip removed clips)
        if self._clips_dir:
            existing = find_clip_by_source(self._clips_dir, folder_path)
            if existing:
                QMessageBox.information(
                    self, "Already Imported",
                    f"This sequence is already in the project as \"{existing}\"."
                )
                return
            # Restore if it was previously removed
            from backend.project import find_removed_clip_by_source, clear_removed_clip
            removed_folder = find_removed_clip_by_source(self._clips_dir, folder_path)
            if removed_folder:
                clear_removed_clip(self._clips_dir, removed_folder)
                logger.info(f"Restored removed sequence: {removed_folder}")
                self._on_clips_dir_changed(self._clips_dir, skip_session_restore=True)
                return

        # Check for duplicate stems (e.g. frame.png + frame.exr)
        dupes = validate_sequence_stems(folder_path)
        if dupes:
            sample = ", ".join(dupes[:5])
            if len(dupes) > 5:
                sample += f" ... ({len(dupes)} total)"
            QMessageBox.warning(
                self, "Duplicate Filenames",
                f"Found files with the same name but different extensions:\n"
                f"{sample}\n\n"
                f"This would cause output file conflicts. Please use one format "
                f"per sequence folder."
            )
            return

        n_frames = count_sequence_frames(folder_path)
        logger.info(f"Importing image sequence: {folder_path} ({n_frames} frames)")

        copy_seq = get_setting_bool(KEY_COPY_SEQUENCES, DEFAULT_COPY_SEQUENCES)

        if self._clips_dir:
            add_sequences_to_project(
                self._clips_dir, [folder_path], copy_source=copy_seq,
            )
            logger.info(f"Added image sequence to project: {folder_path}")
            self._on_clips_dir_changed(self._clips_dir, skip_session_restore=True)
        else:
            proj_name = display_name or os.path.basename(folder_path.rstrip("/\\"))
            project_dir = create_project_from_media(
                sequence_folders=[folder_path],
                copy_sequences=copy_seq,
                display_name=proj_name,
            )
            logger.info(f"Created project from image sequence: {folder_path}")
            self._switch_to_workspace()
            self._on_clips_dir_changed(project_dir, skip_session_restore=True)

    @Slot(list)
    def _on_image_files_dropped(self, file_paths: list) -> None:
        """Handle individual image files dropped — popup for <5 or auto-detect."""
        if not file_paths:
            return

        n = len(file_paths)
        parent_folder = os.path.dirname(file_paths[0])

        # When no project is open, prompt for a project name (loose files
        # may come from generic folders like "Downloads").
        display_name = None
        if not self._clips_dir:
            folder_name = os.path.basename(parent_folder.rstrip("/\\"))
            name, ok = QInputDialog.getText(
                self, "Name Your Project",
                "Give your project a name:",
                text=folder_name,
            )
            if not ok:
                return  # user cancelled
            display_name = name.strip() or folder_name

        if n < 5:
            # Show popup: "Just these N frames" or "Scan folder for full sequence?"
            from backend.project import count_sequence_frames
            folder_count = count_sequence_frames(parent_folder)

            msg = QMessageBox(self)
            msg.setWindowTitle("Import Image Frames")
            msg.setText(
                f"You dropped {n} image file(s).\n"
                f"The source folder contains {folder_count} image(s) total."
            )
            msg.setInformativeText("How would you like to import?")

            btn_just_these = msg.addButton(
                f"Copy Just These {n}", QMessageBox.AcceptRole,
            )
            btn_full_seq = msg.addButton(
                "Import Full Sequence", QMessageBox.ActionRole,
            )
            msg.addButton(QMessageBox.Cancel)
            msg.setDefaultButton(btn_full_seq)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked == btn_just_these:
                self._import_specific_frames(
                    parent_folder, file_paths, display_name=display_name,
                )
            elif clicked == btn_full_seq:
                self._on_sequence_folder_imported(
                    parent_folder, display_name=display_name,
                )
            # else: Cancel — do nothing
        else:
            # >= 5 files: auto-detect as full sequence from parent folder
            self._on_sequence_folder_imported(
                parent_folder, display_name=display_name,
            )

    def _import_specific_frames(
        self, source_folder: str, file_paths: list[str],
        display_name: str | None = None,
    ) -> None:
        """Import specific frames (always copies into Frames/)."""
        from backend.project import (
            create_clip_from_sequence, projects_root,
            write_project_json,
        )

        filenames = [os.path.basename(f) for f in file_paths]

        if self._clips_dir:
            clips_dir = os.path.join(self._clips_dir, "clips")
            os.makedirs(clips_dir, exist_ok=True)
            create_clip_from_sequence(
                clips_dir, source_folder,
                copy_source=True, specific_files=filenames,
            )
            logger.info(f"Imported {len(filenames)} specific frame(s)")
            self._on_clips_dir_changed(self._clips_dir, skip_session_restore=True)
        else:
            # No project open — create one manually with specific files
            from datetime import datetime
            proj_name = display_name or os.path.basename(source_folder.rstrip("/\\"))
            name_stem = re.sub(r"[^\w\-]", "_", proj_name)
            name_stem = re.sub(r"_+", "_", name_stem).strip("_")[:60]
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            project_dir = os.path.join(projects_root(), f"{timestamp}_{name_stem}")
            clips_dir = os.path.join(project_dir, "clips")
            os.makedirs(clips_dir, exist_ok=True)
            clip_name = create_clip_from_sequence(
                clips_dir, source_folder,
                copy_source=True, specific_files=filenames,
            )
            write_project_json(project_dir, {
                "version": 2,
                "created": datetime.now().isoformat(),
                "display_name": proj_name,
                "clips": [clip_name],
            })
            logger.info(f"Created project with {len(filenames)} specific frame(s)")
            self._switch_to_workspace()
            self._on_clips_dir_changed(project_dir, skip_session_restore=True)

    def _create_project_from_folder(self, dir_path: str) -> None:
        """Create a new project from a folder of videos/sequences (welcome screen path).

        Scans the folder for video files and image sequences, creates a project
        named after the folder, and opens it.
        """
        from backend.project import (
            is_video_file, create_project, folder_has_image_sequence,
            create_project_from_media,
        )
        videos = sorted(
            os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if is_video_file(os.path.join(dir_path, f))
        )

        # Check if folder itself is an image sequence (no subdirectories)
        is_seq = folder_has_image_sequence(dir_path)

        if not videos and not is_seq:
            QMessageBox.information(
                self, "No Media",
                "No video files or image sequences found in that folder."
            )
            return

        folder_name = os.path.basename(dir_path.rstrip("/\\"))
        copy_video = get_setting_bool(KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE)
        copy_seq = get_setting_bool(KEY_COPY_SEQUENCES, DEFAULT_COPY_SEQUENCES)

        if videos and not is_seq:
            # Videos only — use existing path
            project_dir = create_project(
                videos, copy_source=copy_video, display_name=folder_name,
            )
        elif is_seq and not videos:
            # Image sequence only
            project_dir = create_project_from_media(
                sequence_folders=[dir_path],
                copy_sequences=copy_seq,
                display_name=folder_name,
            )
        else:
            # Mixed: videos + image sequence
            project_dir = create_project_from_media(
                video_paths=videos,
                sequence_folders=[dir_path],
                copy_video=copy_video,
                copy_sequences=copy_seq,
                display_name=folder_name,
            )

        media_count = len(videos) + (1 if is_seq else 0)
        logger.info(
            f"Created project '{folder_name}' with {media_count} source(s) from folder"
        )
        self._switch_to_workspace()
        self._on_clips_dir_changed(project_dir, skip_session_restore=True)

    # ── View Controls ──

    def _on_reset_zoom(self) -> None:
        """Reset preview zoom to fit."""
        self._dual_viewer.reset_zoom()

    @Slot()
    def _sync_io_divider(self, *_args) -> None:
        """Keep the IO tray divider aligned with the dual viewer splitter."""
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
        # set_in_point emits in_point_changed → _persist_in_out
        self._dual_viewer._scrubber.set_in_point(idx)

    def _set_out_point(self) -> None:
        """Set out-point at the current scrubber position."""
        if not self._current_clip:
            return
        idx = self._dual_viewer._scrubber.current_frame()
        in_pt, _ = self._dual_viewer.get_in_out()
        # Clamp: out-point cannot precede in-point
        if in_pt is not None and idx < in_pt:
            idx = in_pt
        # set_out_point emits out_point_changed → _persist_in_out
        self._dual_viewer._scrubber.set_out_point(idx)

    def _clear_in_out(self) -> None:
        """Clear in/out markers (Alt+I)."""
        if not self._current_clip:
            return
        self._dual_viewer._scrubber.clear_in_out()
        self._current_clip.in_out_range = None
        from backend.project import save_in_out_range
        save_in_out_range(self._current_clip.root_path, None)
        # Re-resolve state: removing in/out may drop READY → RAW
        # if alpha only partially covers the full clip
        self._current_clip._resolve_state()
        self._clip_model.update_clip_state(
            self._current_clip.name, self._current_clip.state)
        self._io_tray.refresh()
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
            # Re-resolve state: partial alpha + new in/out range may
            # promote RAW → READY (v1.2.1 partial alpha logic)
            self._current_clip._resolve_state()
            self._clip_model.update_clip_state(
                self._current_clip.name, self._current_clip.state)
            self._io_tray.refresh()
        self._refresh_button_state()

    def _on_reset_all_in_out(self) -> None:
        """Clear in/out markers on all clips (called from IO tray button)."""
        from backend.project import save_in_out_range
        count = 0
        for clip in self._clip_model.clips:
            if clip.in_out_range is not None:
                clip.in_out_range = None
                save_in_out_range(clip.root_path, None)
                count += 1
        # Clear the scrubber if current clip was affected
        if self._current_clip:
            self._dual_viewer._scrubber.clear_in_out()
        self._refresh_button_state()
        logger.info(f"Reset in/out markers on {count} clips")

    # ── Live Reprocess (Codex: through GPU queue, not parallel) ──

    @Slot()
    def _on_params_changed(self) -> None:
        """Handle parameter change — debounce before reprocess."""
        if self._param_panel.live_preview_enabled and self._service.is_engine_loaded():
            self._reprocess_timer.start()

    @Slot(bool)
    def _on_live_preview_toggled(self, checked: bool) -> None:
        """When live preview is re-enabled, immediately reprocess current frame."""
        if checked and self._service.is_engine_loaded():
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
        """Handle live reprocess result — display preview matching current view mode."""
        if not isinstance(result, dict):
            return

        if result.get("kind") == "sam2_preview":
            clip_name = result.get("clip_name")
            if not isinstance(clip_name, str):
                return
            clip = next((c for c in self._clip_model.clips if c.name == clip_name), None)
            if clip is None or self._current_clip is None or self._current_clip.name != clip_name:
                return

            frame_rgb = result.get("frame_rgb")
            mask = result.get("mask")
            if not isinstance(frame_rgb, np.ndarray) or not isinstance(mask, np.ndarray):
                return

            qimg = self._sam2_preview_qimage(frame_rgb, mask)
            self._dual_viewer.show_reprocess_preview(qimg)

            frame_number = int(result.get("frame_index", 0)) + 1
            fill_pct = float(result.get("fill", 0.0)) * 100.0
            reply = QMessageBox.question(
                self,
                "Track Mask Preview",
                f"SAM2 preview on frame {frame_number} covers {fill_pct:.1f}% of the frame.\n\n"
                "If this looks right, continue with full Track Mask.\n"
                "If not, keep painting corrections on this frame and run Track Mask again.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._submit_sam2_track_job(clip)
            else:
                self._status_bar.set_message("Track preview ready. Refine annotations and run Track Mask again.")
            return

        if 'comp' not in result:
            return

        mode = self._dual_viewer._output_viewer._current_mode

        # Pick the array that matches current view mode
        if mode == ViewMode.MATTE and 'alpha' in result:
            arr = result['alpha']
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            # Matte display: clamp, light gamma lift, grayscale → RGB
            display = np.power(np.clip(arr, 0.0, 1.0), 0.85)
            gray8 = (display * 255.0).astype(np.uint8)
            rgb = np.stack([gray8, gray8, gray8], axis=2)
        elif mode == ViewMode.FG and 'fg' in result:
            # FG is already sRGB float [H,W,3]
            rgb = (np.clip(result['fg'], 0.0, 1.0) * 255.0).astype(np.uint8)
        elif mode == ViewMode.PROCESSED and 'processed' in result:
            qimg = processed_rgba_to_qimage(result['processed'])
            self._dual_viewer.show_reprocess_preview(qimg)
            return
        else:
            # Default: COMP (also for INPUT/MASK/ALPHA which don't change on reprocess)
            rgb = (np.clip(result['comp'], 0.0, 1.0) * 255.0).astype(np.uint8)

        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._dual_viewer.show_reprocess_preview(qimg)

    # ── Inference Control ──

    @Slot()
    def _on_run_inference(self) -> None:
        # If multiple clips are selected, dispatch to pipeline
        if self._io_tray.selected_count() > 1:
            self._on_run_pipeline()
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
                            gvm_job.id, job_label="GVM Auto",
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
            if clip.in_out_range:
                job.params["_frame_range"] = (
                    clip.in_out_range.in_point,
                    clip.in_out_range.out_point,
                )
            if self._service.job_queue.submit(job):
                queued += 1

        if queued > 0:
            first_job = self._service.job_queue.next_job()
            self._start_worker_if_needed(first_job.id if first_job else None)
            logger.info(f"Batch queued: {queued} clips")

    def _on_run_pipeline(self) -> None:
        """Full pipeline: classify each selected clip and queue appropriate jobs.

        Routes per clip (see PipelineRoute):
        - INFERENCE_ONLY: queue inference directly
        - GVM_PIPELINE: queue GVM, auto-chain inference on completion
        - VIDEOMAMA_PIPELINE: track dense masks, queue VideoMaMa, auto-chain inference
        - VIDEOMAMA_INFERENCE: queue VideoMaMa, auto-chain inference
        - SKIP: skip clips in EXTRACTING/ERROR state
        """
        selected = self._io_tray.get_selected_clips()
        if not selected:
            return
        self._auto_save_annotations()

        params = self._param_panel.get_params()
        output_config = self._param_panel.get_output_config()

        # Classify all clips
        routes: dict[str, PipelineRoute] = {}
        for clip in selected:
            route = classify_pipeline_route(clip)
            if route != PipelineRoute.SKIP:
                routes[clip.name] = route

        if not routes:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Nothing to Process",
                "No selected clips are in a processable state.",
            )
            return

        self._pipeline_steps.clear()

        queued = 0
        first_job_id = None

        for clip in selected:
            route = routes.get(clip.name)
            if route is None:
                continue
            job = None
            next_steps: list[JobType] = []

            if route == PipelineRoute.GVM_PIPELINE:
                job = create_job_snapshot(clip, job_type=JobType.GVM_ALPHA)
                next_steps = [JobType.INFERENCE]
            elif route == PipelineRoute.VIDEOMAMA_PIPELINE:
                job = create_job_snapshot(clip, params, job_type=JobType.SAM2_TRACK)
                next_steps = [JobType.VIDEOMAMA_ALPHA, JobType.INFERENCE]
            elif route == PipelineRoute.VIDEOMAMA_INFERENCE:
                job = create_job_snapshot(clip, job_type=JobType.VIDEOMAMA_ALPHA)
                next_steps = [JobType.INFERENCE]
            elif route == PipelineRoute.INFERENCE_ONLY:
                if clip.state == ClipState.COMPLETE:
                    clip.transition_to(ClipState.READY)
                job = create_job_snapshot(clip, params)
                job.params["_output_config"] = output_config
                if clip.in_out_range:
                    job.params["_frame_range"] = (
                        clip.in_out_range.in_point,
                        clip.in_out_range.out_point,
                    )

            if job is None:
                continue
            if self._service.job_queue.submit(job):
                clip.set_processing(True)
                if next_steps:
                    self._pipeline_steps[clip.name] = next_steps
                if first_job_id is None:
                    first_job_id = job.id
                queued += 1

        if queued > 0:
            self._start_worker_if_needed(first_job_id, job_label="Pipeline")
            gvm_n = sum(1 for r in routes.values() if r == PipelineRoute.GVM_PIPELINE)
            track_n = sum(1 for r in routes.values() if r == PipelineRoute.VIDEOMAMA_PIPELINE)
            vm_n = sum(1 for r in routes.values() if r == PipelineRoute.VIDEOMAMA_INFERENCE)
            inf_n = sum(1 for r in routes.values() if r == PipelineRoute.INFERENCE_ONLY)
            logger.info(
                f"Pipeline queued: {gvm_n} GVM + {track_n} Track Mask + {vm_n} VideoMaMa + "
                f"{inf_n} inference = {queued} initial jobs "
                f"(+{sum(len(v) for v in self._pipeline_steps.values())} auto-chain pending)"
            )

    @Slot()
    def _on_import_alpha(self) -> None:
        """Import user-provided alpha hint images into the clip's AlphaHint/ folder.

        Files are renamed to match input frame stems so index-based matching
        in the inference loop works correctly (frame 0 → frame 0, etc.).
        """
        clip = self._current_clip
        if clip is None or clip.state not in (ClipState.RAW, ClipState.MASKED, ClipState.READY):
            return
        if clip.input_asset is None:
            return

        # If AlphaHint already exists, ask before replacing
        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir) and os.listdir(alpha_dir):
            result = QMessageBox.question(
                self, "Replace Alpha Hints?",
                f"Clip '{clip.name}' already has alpha hint images.\n\n"
                "Do you want to replace them with new ones?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                return

        src_dir = QFileDialog.getExistingDirectory(
            self, "Select Alpha Hint Folder",
            "",
            QFileDialog.ShowDirsOnly,
        )
        if not src_dir:
            return

        # Find image files in the selected folder (natural/numeric sort)
        import glob as glob_module
        import re as re_module
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.exr")
        src_files = []
        for pat in patterns:
            src_files.extend(glob_module.glob(os.path.join(src_dir, pat)))

        def _natural_key(path: str):
            """Sort key that handles any zero-padding scheme correctly."""
            name = os.path.basename(path)
            return [int(c) if c.isdigit() else c.lower()
                    for c in re_module.split(r'(\d+)', name)]

        src_files.sort(key=_natural_key)

        if not src_files:
            QMessageBox.warning(
                self, "No Images",
                "No image files found in the selected folder.\n"
                "Expected grayscale images (white=foreground, black=background).",
            )
            return

        # Get input frame stems for renaming
        input_files = clip.input_asset.get_frame_files()
        n_input = len(input_files)
        n_src = len(src_files)

        if n_src != n_input:
            result = QMessageBox.warning(
                self, "Frame Count Mismatch",
                f"Clip '{clip.name}' has {n_input} input frames but you "
                f"selected {n_src} alpha images.\n\n"
                f"Each input frame needs a matching alpha hint.\n"
                f"Only {min(n_src, n_input)} frames will be paired.",
                QMessageBox.Ok | QMessageBox.Cancel,
            )
            if result == QMessageBox.Cancel:
                return

        # Confirm import
        n_paired = min(n_src, n_input)
        msg = f"Import {n_paired} alpha hint images into '{clip.name}'?"
        if n_src != n_input:
            msg += f"\n({abs(n_src - n_input)} frames will have no alpha hint)"
        if QMessageBox.question(self, "Import Alpha", msg) != QMessageBox.Yes:
            return

        # Copy + rename to match input frame stems
        import shutil
        import cv2
        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir):
            shutil.rmtree(alpha_dir)
        os.makedirs(alpha_dir)

        for i in range(n_paired):
            src_path = src_files[i]
            # Use the input frame's stem with .png extension
            input_stem = os.path.splitext(input_files[i])[0]
            dst_path = os.path.join(alpha_dir, f"{input_stem}.png")

            src_ext = os.path.splitext(src_path)[1].lower()
            if src_ext == '.png':
                shutil.copy2(src_path, dst_path)
            else:
                # Convert non-PNG to PNG (grayscale)
                img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    cv2.imwrite(dst_path, img)
                else:
                    logger.warning(f"Failed to read alpha image: {src_path}")

        logger.info(f"Imported {n_paired} alpha hints into {alpha_dir} "
                     f"(renamed to match input stems)")

        # Refresh clip state
        clip.find_assets()
        self._io_tray.refresh()

        # Reload preview and button states
        if self._current_clip and self._current_clip.name == clip.name:
            self._dual_viewer.set_clip(clip)
            self._refresh_button_state()
            self._param_panel.set_import_alpha_enabled(
                clip.state in (ClipState.RAW, ClipState.MASKED, ClipState.READY)
            )

        _Toast(self, f"Imported {n_src} alpha hints.\nClip is now {clip.state.value}.")

    def _on_run_gvm(self) -> None:
        """Run GVM alpha generation on the selected clip."""
        if self._current_clip is None or self._current_clip.state not in (ClipState.RAW, ClipState.MASKED):
            return

        # Detect partial alpha from a previous interrupted run
        alpha_dir = os.path.join(self._current_clip.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir):
            existing = [f for f in os.listdir(alpha_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if existing:
                total = (self._current_clip.input_asset.frame_count
                         if self._current_clip.input_asset else 0)
                msg = QMessageBox(self)
                msg.setWindowTitle("Partial Alpha Found")
                msg.setText(
                    f"Found {len(existing)}/{total} alpha frames from a previous run."
                )
                msg.setInformativeText(
                    "Resume will skip completed frames.\n"
                    "Regenerate will redo all frames from scratch."
                )
                resume_btn = msg.addButton("Resume", QMessageBox.AcceptRole)
                regen_btn = msg.addButton("Regenerate", QMessageBox.DestructiveRole)
                msg.addButton(QMessageBox.Cancel)
                msg.setDefaultButton(resume_btn)
                msg.exec()
                clicked = msg.clickedButton()
                if clicked == regen_btn:
                    shutil.rmtree(alpha_dir, ignore_errors=True)
                elif clicked != resume_btn:
                    return  # cancelled

        job = create_job_snapshot(self._current_clip, job_type=JobType.GVM_ALPHA)
        if not self._service.job_queue.submit(job):
            return

        self._current_clip.set_processing(True)
        self._start_worker_if_needed(job.id, job_label="GVM Auto")

    @Slot()
    def _on_run_videomama(self) -> None:
        """Run VideoMaMa alpha generation on the selected clip."""
        if self._current_clip is None:
            return
        if not self._clip_has_videomama_ready_mask(self._current_clip):
            QMessageBox.information(
                self,
                "Track Mask First",
                "Paint prompts and run Track Mask before using VideoMaMa.",
            )
            return

        job = create_job_snapshot(self._current_clip, job_type=JobType.VIDEOMAMA_ALPHA)
        if not self._service.job_queue.submit(job):
            return

        self._current_clip.set_processing(True)
        self._start_worker_if_needed(job.id, job_label="VideoMaMa")

    @Slot()
    def _on_run_matanyone2(self) -> None:
        """Run MatAnyone2 video matting alpha generation on the selected clip."""
        if self._current_clip is None:
            return
        if not self._clip_has_videomama_ready_mask(self._current_clip):
            QMessageBox.information(
                self,
                "Track Mask First",
                "MatAnyone2 requires a tracked mask on frame 0.\n\n"
                "Paint prompts and run Track Mask before using MatAnyone2.",
            )
            return

        job = create_job_snapshot(self._current_clip, job_type=JobType.MATANYONE2_ALPHA)
        if not self._service.job_queue.submit(job):
            return

        self._current_clip.set_processing(True)
        self._start_worker_if_needed(job.id, job_label="MatAnyone2")

    def _start_worker_if_needed(
        self,
        first_job_id: str | None = None,
        job_label: str = "Inference",
    ) -> None:
        """Ensure GPU worker is running and show queue panel."""
        if first_job_id and self._active_job_id is None:
            self._active_job_id = first_job_id

        if not self._gpu_worker.isRunning():
            if sys.platform == "win32":
                self._gpu_worker.start(QThread.LowPriority)
            else:
                self._gpu_worker.start()
        else:
            self._gpu_worker.wake()

        self._force_stop_armed = False
        self._status_bar.set_running(True)
        self._status_bar.set_stop_button_mode(force=False)
        self._status_bar.reset_progress()
        self._status_bar.start_job_timer(label=job_label)
        self._queue_panel.refresh()
        if self._queue_panel._collapsed:
            self._queue_panel.toggle_collapsed()  # auto-expand when job starts

    @Slot()
    def _on_stop_inference(self) -> None:
        """STOP button handler — confirms and cancels inference."""
        if not self._status_bar._stop_btn.isVisible():
            return
        queue = self._service.job_queue
        if not queue.current_job and not queue.has_pending:
            return

        if self._force_stop_armed:
            reply = QMessageBox.question(
                self, "Force Stop",
                "The current GPU step has not returned to Python.\n\n"
                "Force Stop will auto-save the session and relaunch the app "
                "to break the stuck job immediately.\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            self._force_restart_app()
            return

        reply = QMessageBox.question(
            self, "Cancel",
            "Cancel processing?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        from ui.sounds.audio_manager import UIAudio
        UIAudio.user_cancel()
        self._cancel_inference()

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
        if self._cancel_requested_job_id == job_id:
            self._queue_panel.refresh()
            return

        # Set active_job_id only on first progress of a new job (not every event)
        if self._active_job_id != job_id:
            # Only update if this is genuinely a new running job
            current_job = self._service.job_queue.current_job
            if current_job and current_job.id == job_id:
                self._active_job_id = job_id

        if job_id == self._active_job_id:
            self._status_bar.update_progress(current, total)

        if self._current_clip and self._current_clip.name == clip_name:
            self._schedule_live_asset_refresh(clip_name, current, total)

        self._queue_panel.refresh()

    def _schedule_live_asset_refresh(self, clip_name: str, current: int, total: int) -> None:
        """Coalesce progress-driven asset refreshes for the selected clip.

        This keeps the feedback live without rescanning on every frame tick or
        touching any GPU-side work.
        """
        if current <= 0 or total <= 0:
            return

        step = max(1, total // 50)
        should_refresh = current <= 3 or current >= total or total <= 30 or current % step == 0
        if not should_refresh:
            return

        self._pending_live_asset_refresh_clip = clip_name
        if not self._live_asset_refresh_timer.isActive():
            self._live_asset_refresh_timer.start()

    @Slot()
    def _refresh_selected_clip_live_assets(self) -> None:
        """Refresh the selected clip's coverage/modes without resetting navigation."""
        clip_name = self._pending_live_asset_refresh_clip
        self._pending_live_asset_refresh_clip = None

        if clip_name is None or self._current_clip is None:
            return
        if self._current_clip.name != clip_name:
            return

        self._dual_viewer.refresh_generated_assets()

    @Slot(str, str)
    def _on_worker_status(self, job_id: str, message: str) -> None:
        """Phase status from long-running jobs (e.g. VideoMaMa loading phases).

        Always update — status signals fire during loading phases before
        progress signals arrive, so _active_job_id may not match yet.
        Also set _active_job_id if not already set.
        """
        if self._cancel_requested_job_id == job_id:
            return
        if self._active_job_id is None:
            self._active_job_id = job_id
        self._status_bar.set_phase(message)

    @Slot(str, str, int, str)
    def _on_worker_preview(self, job_id: str, clip_name: str, frame_index: int, path: str) -> None:
        if self._cancel_requested_job_id == job_id:
            return
        # Only update preview if this is the active job
        if job_id == self._active_job_id:
            self._dual_viewer.load_preview_from_file(path, clip_name, frame_index)

    @Slot(str, str, str)
    def _on_worker_clip_finished(self, job_id: str, clip_name: str, job_type: str) -> None:
        self._force_stop_armed = False
        self._status_bar.set_stop_button_mode(force=False)
        # Map job type to correct next state.
        if job_type == JobType.SAM2_TRACK.value:
            target_state = ClipState.MASKED
        elif job_type in (JobType.GVM_ALPHA.value, JobType.VIDEOMAMA_ALPHA.value,
                          JobType.MATANYONE2_ALPHA.value):
            target_state = ClipState.READY
        else:
            target_state = ClipState.COMPLETE

        self._clip_model.update_clip_state(clip_name, target_state)

        # Clear processing lock and rescan assets for pipeline steps
        for clip in self._clip_model.clips:
            if clip.name == clip_name:
                clip.set_processing(False)
                if target_state in (ClipState.MASKED, ClipState.READY):
                    try:
                        clip.find_assets()
                    except Exception:
                        pass
                    clip.state = target_state
                    self._clip_model.update_clip_state(clip_name, target_state)
                break

        # Pipeline auto-chain: queue the next stage, if any.
        if clip_name in self._pipeline_steps and self._pipeline_steps[clip_name]:
            next_step = self._pipeline_steps[clip_name].pop(0)
            queued_next = False
            for clip in self._clip_model.clips:
                if clip.name != clip_name:
                    continue
                if next_step == JobType.INFERENCE:
                    params = self._param_panel.get_params()
                    job = create_job_snapshot(clip, params)
                    job.params["_output_config"] = self._param_panel.get_output_config()
                    if clip.in_out_range:
                        job.params["_frame_range"] = (
                            clip.in_out_range.in_point,
                            clip.in_out_range.out_point,
                        )
                else:
                    job = create_job_snapshot(clip, job_type=next_step)
                clip.set_processing(True)
                if self._service.job_queue.submit(job):
                    queued_next = True
                    logger.info(
                        "Pipeline auto-chain: queued %s for %s",
                        next_step.value,
                        clip_name,
                    )
                else:
                    clip.set_processing(False)
                break
            if not self._pipeline_steps[clip_name]:
                self._pipeline_steps.pop(clip_name, None)
            elif not queued_next:
                logger.warning("Pipeline auto-chain failed to queue next step for %s", clip_name)

        # Stop timer; only exit running state if no more jobs are pending
        has_more = self._service.job_queue.has_pending or self._pipeline_steps
        self._status_bar.stop_job_timer()
        if not has_more:
            self._status_bar.set_running(False)
        else:
            # Reset progress for next job — show descriptive label
            self._status_bar.reset_progress()
            next_job = self._service.job_queue.next_job()
            if next_job:
                _label_map = {
                    JobType.GVM_ALPHA: "GVM Auto",
                    JobType.SAM2_PREVIEW: "Track Preview",
                    JobType.SAM2_TRACK: "Track Mask",
                    JobType.VIDEOMAMA_ALPHA: "VideoMaMa",
                    JobType.MATANYONE2_ALPHA: "MatAnyone2",
                    JobType.INFERENCE: "Inference",
                }
                next_label = _label_map.get(next_job.job_type, "Pipeline")
            else:
                next_label = "Pipeline"
            self._status_bar.start_job_timer(label=next_label)

        from ui.sounds.audio_manager import UIAudio
        if target_state == ClipState.MASKED:
            UIAudio.mask_done()
            self._status_bar.set_message(f"Track Mask complete for {clip_name}")
        elif target_state == ClipState.READY:
            UIAudio.mask_done()
            type_label = {
                JobType.GVM_ALPHA.value: "GVM Auto",
                JobType.VIDEOMAMA_ALPHA.value: "VideoMaMa",
                JobType.MATANYONE2_ALPHA.value: "MatAnyone2",
            }.get(job_type, "Alpha")
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
            UIAudio.inference_done()
            self._status_bar.set_message(f"Inference complete: {clip_name}")

        # Refresh views
        self._pending_live_asset_refresh_clip = None
        self._live_asset_refresh_timer.stop()
        self._queue_panel.refresh()
        self._io_tray.refresh()

        # If selected clip, reload preview to show new assets
        if self._current_clip and self._current_clip.name == clip_name:
            self._dual_viewer.set_clip(self._current_clip)
            self._refresh_button_state()
            _has_mask = self._clip_has_videomama_ready_mask(self._current_clip)
            self._param_panel.set_videomama_enabled(_has_mask)
            self._param_panel.set_matanyone2_enabled(_has_mask)
            self._param_panel.set_import_alpha_enabled(
                self._current_clip.state in (ClipState.RAW, ClipState.MASKED, ClipState.READY)
            )

        logger.info(f"Clip finished ({job_type}): {clip_name} -> {target_state.value}")

    @Slot(str, str)
    def _on_worker_warning(self, job_id: str, message: str) -> None:
        if message.startswith("Cancelled:"):
            self._cancel_requested_job_id = None
            self._active_job_id = None
            self._force_stop_armed = False
            self._status_bar.set_stop_button_mode(force=False)
            # Job was cancelled — clear processing lock on the clip
            clip_name = message.removeprefix("Cancelled:").strip()
            self._pipeline_steps.pop(clip_name, None)
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
            self._pending_live_asset_refresh_clip = None
            self._live_asset_refresh_timer.stop()
            self._queue_panel.refresh()
            logger.info(f"Job cancelled: {clip_name}")
        else:
            self._status_bar.add_warning(message)
            logger.warning(f"Worker warning: {message}")

    @Slot(str, str, str)
    def _on_worker_error(self, job_id: str, clip_name: str, error_msg: str) -> None:
        if self._cancel_requested_job_id == job_id:
            self._cancel_requested_job_id = None
            self._active_job_id = None
        self._force_stop_armed = False
        self._status_bar.set_stop_button_mode(force=False)
        self._status_bar.stop_job_timer()
        self._status_bar.set_running(False)
        self._pending_live_asset_refresh_clip = None
        self._live_asset_refresh_timer.stop()
        self._pipeline_steps.pop(clip_name, None)
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
        from ui.sounds.audio_manager import UIAudio
        UIAudio.error()

        # Try to match a known error pattern for actionable diagnostics
        from ui.widgets.diagnostic_dialog import match_diagnostic, DiagnosticDialog
        diag = match_diagnostic(error_msg)
        if diag:
            dlg = DiagnosticDialog(
                diag, error_msg,
                gpu_info=self._service.get_vram_info(),
                recent_errors=self._debug_console.recent_errors()
                if hasattr(self, '_debug_console') else None,
                parent=self,
            )
            dlg.exec()
        else:
            QMessageBox.critical(self, "Processing Error", f"Clip: {clip_name}\n\n{error_msg}")

    @Slot()
    def _on_queue_empty(self) -> None:
        if self._service.job_queue.has_pending:
            return
        self._cancel_requested_job_id = None
        self._force_stop_armed = False
        self._status_bar.set_stop_button_mode(force=False)
        self._status_bar.set_running(False)
        self._status_bar.stop_job_timer()
        self._active_job_id = None
        self._pending_live_asset_refresh_clip = None
        self._live_asset_refresh_timer.stop()
        self._pipeline_steps.clear()
        self._queue_panel.refresh()
        logger.info("All jobs completed")

    # ── Video Extraction ──

    def _auto_extract_clips(self, clips: list[ClipEntry]) -> None:
        """Auto-submit EXTRACTING clips to the extract worker.

        New-format projects (Source/ subdir) already have video in the right
        place — extraction writes to Frames/. Legacy standalone videos
        (root_path == clips_dir) are restructured into a clip subdirectory.
        """
        extracting = [c for c in clips if c.state == ClipState.EXTRACTING]
        if not extracting:
            return

        if not self._extract_worker.isRunning():
            if sys.platform == "win32":
                self._extract_worker.start(QThread.LowPriority)
            else:
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
        self._status_bar.start_job_timer(label="Extracting")
        logger.info(f"Auto-extraction queued: {len(extracting)} clip(s)")

    def _on_extract_current_clip(self) -> None:
        """Handle RUN EXTRACTION button — extract the currently selected clip."""
        clip = self._current_clip
        if not clip or clip.state != ClipState.EXTRACTING:
            return
        self._on_extract_requested([clip])

    @Slot(list)
    def _on_extract_requested(self, clips: list) -> None:
        """Handle right-click → Run Extraction on selected clips."""
        if not clips:
            return
        if not self._extract_worker.isRunning():
            if sys.platform == "win32":
                self._extract_worker.start(QThread.LowPriority)
            else:
                self._extract_worker.start()
        count = 0
        for clip in clips:
            if clip.input_asset and clip.input_asset.asset_type == "video":
                # If retrying after error, wipe partial frames so
                # extract_frames() starts fresh instead of resuming
                if clip.state == ClipState.ERROR:
                    for subdir in ("Frames", "Input"):
                        target = os.path.join(clip.root_path, subdir)
                        if os.path.isdir(target):
                            shutil.rmtree(target)
                            os.makedirs(target, exist_ok=True)
                            logger.info(f"Cleared {subdir}/ for retry: {clip.name}")
                    clip.error_message = None
                    self._clip_model.update_clip_state(
                        clip.name, ClipState.EXTRACTING)
                self._extract_worker.submit(
                    clip.name, clip.input_asset.path, clip.root_path,
                )
                count += 1
        if count:
            self._status_bar.start_job_timer(label="Extracting")
            logger.info(f"Manual extraction queued: {count} clip(s)")

    @Slot(str, int, int)
    def _on_extract_progress(self, clip_name: str, current: int, total: int) -> None:
        """Update status bar, clip card, and input viewer progress."""
        # Track per-clip progress for aggregate status bar
        self._extract_progress[clip_name] = (current, total)
        agg_current = sum(c for c, _ in self._extract_progress.values())
        agg_total = sum(t for _, t in self._extract_progress.values())
        self._status_bar.update_progress(agg_current, agg_total)
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

                # If this is the selected clip, fully re-select to update
                # viewer, param panel buttons (GVM/VideoMaMa), and status bar
                if self._current_clip and self._current_clip.name == clip_name:
                    self._on_clip_selected(clip)

                logger.info(f"Extraction complete: {clip_name} ({frame_count} frames)")
                break

        self._io_tray.refresh()
        # Remove from aggregate tracker; reset status bar when all done
        self._extract_progress.pop(clip_name, None)
        if not self._extract_worker.is_busy:
            self._status_bar.reset_progress()
            from ui.sounds.audio_manager import UIAudio
            UIAudio.frame_extract_done()

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
        self._extract_progress.pop(clip_name, None)
        # Clear the extraction overlay on the viewer
        if self._current_clip and self._current_clip.name == clip_name:
            self._dual_viewer.set_extraction_progress(0.0, 0)
        if not self._extract_worker.is_busy:
            self._status_bar.reset_progress()
        from ui.sounds.audio_manager import UIAudio
        UIAudio.error()
        logger.error(f"Extraction failed for {clip_name}: {error_msg}")

    # ── Export Video ──

    def _on_export_video(self, clip: ClipEntry | None = None,
                         source_dir: str | None = None) -> None:
        """Export output image sequence as video file.

        Args:
            clip: Specific clip to export. If None, uses current selection.
            source_dir: Specific source directory. If None, auto-detect.
        """
        if clip is None:
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

        # Use provided source_dir or auto-detect
        if source_dir and os.path.isdir(source_dir) and os.listdir(source_dir):
            pass  # use as-is
        else:
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
        from backend.ffmpeg_tools import read_video_metadata, stitch_video, require_ffmpeg_install
        try:
            require_ffmpeg_install(require_probe=True)
        except RuntimeError as exc:
            QMessageBox.critical(
                self, "FFmpeg Unavailable",
                str(exc),
            )
            return

        metadata = read_video_metadata(clip.root_path)
        fps = metadata.get("fps", 24.0) if metadata else 24.0

        # Default export to _EXPORTS in the clip's project folder
        subdir_name = os.path.basename(source_dir)
        exports_dir = os.path.join(clip.root_path, "_EXPORTS")
        os.makedirs(exports_dir, exist_ok=True)
        default_name = f"{clip.name}_{subdir_name}_export.mp4"
        default_path = os.path.join(exports_dir, default_name)
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", default_path,
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
            from ui.sounds.audio_manager import UIAudio
            UIAudio.error()
            QMessageBox.critical(
                self, "Export Failed",
                f"Failed to export video:\n{e}",
            )

    # ── Session Save/Load (Codex: JSON sidecar, atomic write, version) ──

    def _session_path(self) -> str | None:
        """Return session file path, or None if no clips dir or Projects root.

        The Projects root should never have a session file — sessions are
        scoped to individual project folders to prevent cross-contamination.
        """
        if not self._clips_dir:
            return None
        from backend.project import projects_root as _projects_root
        try:
            if os.path.normcase(os.path.abspath(self._clips_dir)) == os.path.normcase(
                os.path.abspath(_projects_root())
            ):
                return None
        except Exception:
            pass
        return os.path.join(self._clips_dir, _SESSION_FILENAME)

    def _build_session_data(self) -> dict:
        """Build session data dict from current UI state."""
        data: dict = {
            "version": _SESSION_VERSION,
            "params": self._param_panel.get_params().to_dict(),
            "output_config": self._param_panel.get_output_config().to_dict(),
            "live_preview": self._param_panel.live_preview_enabled,
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

        # Restore live preview toggle
        if "live_preview" in data:
            self._param_panel._live_preview.setChecked(bool(data["live_preview"]))

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
    def _on_open_project(self) -> None:
        """Open a project folder via directory picker (Ctrl+O)."""
        from backend.project import projects_root
        start_dir = projects_root()
        folder = QFileDialog.getExistingDirectory(
            self, "Open Project", start_dir,
        )
        if not folder:
            return
        self._switch_to_workspace()
        self._on_clips_dir_changed(folder, skip_session_restore=False)

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
        self._splitter.setSizes([920, 280])
        self._vsplitter.setSizes([600, 140])

    def _toggle_queue_panel(self) -> None:
        self._queue_panel.toggle_collapsed()

    def _toggle_debug_console(self) -> None:
        """Toggle the in-app debug console (F12)."""
        if self._debug_console.isVisible():
            self._debug_console.hide()
        else:
            self._debug_console.show()

    def _run_startup_diagnostics(self, device: str) -> None:
        """Check environment for known issues and show a diagnostic dialog."""
        from ui.widgets.diagnostic_dialog import run_startup_diagnostics, StartupDiagnosticDialog
        issues = run_startup_diagnostics(device)
        if issues:
            dlg = StartupDiagnosticDialog(issues, parent=self)
            dlg.exec()

    def _show_preferences(self) -> None:
        """Open the Preferences dialog and apply changes."""
        dlg = PreferencesDialog(self)
        if dlg.exec() == PreferencesDialog.Accepted:
            self._apply_tooltip_setting()
            self._apply_sound_setting()
            self._apply_tracker_model_setting()

    def _show_hotkeys(self) -> None:
        """Open the Hotkeys configuration dialog and apply changes."""
        from ui.widgets.hotkeys_dialog import HotkeysDialog
        dlg = HotkeysDialog(self._shortcut_registry, self)
        if dlg.exec() == HotkeysDialog.Accepted:
            self._setup_shortcuts()

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

    def _apply_sound_setting(self) -> None:
        """Apply UI sounds on/off and volume from saved preferences."""
        from ui.widgets.preferences_dialog import KEY_UI_SOUNDS, DEFAULT_UI_SOUNDS
        from ui.sounds.audio_manager import UIAudio
        UIAudio.set_muted(not get_setting_bool(KEY_UI_SOUNDS, DEFAULT_UI_SOUNDS))
        # Restore volume level
        vol = QSettings().value("ui/sounds_volume", 1.0, type=float)
        UIAudio.set_volume(vol)
        if hasattr(self, "_volume_control"):
            self._volume_control.sync_mute_state()

    def _apply_tracker_model_setting(self) -> None:
        """Apply saved SAM2 tracker model preference to the backend service."""
        model_id = get_setting_str(KEY_TRACKER_MODEL, DEFAULT_TRACKER_MODEL)
        self._service.set_sam2_model(model_id)

    def _show_report_issue(self) -> None:
        import logging as _logging

        from ui.widgets.report_issue_dialog import ReportIssueDialog

        # Gather GPU info from last monitor update
        gpu_info = {}
        if hasattr(self, "_last_vram_info"):
            gpu_info = self._last_vram_info

        # Gather recent WARNING/ERROR log lines from debug console buffer
        import re
        recent_errors: list[str] = []
        if hasattr(self, "_debug_console"):
            for html, levelno in self._debug_console._log_buffer:
                if levelno >= _logging.WARNING:
                    plain = re.sub(r"<[^>]+>", "", html).strip()
                    if plain:
                        recent_errors.append(plain)
                if len(recent_errors) >= 20:
                    break

        dlg = ReportIssueDialog(
            gpu_info=gpu_info,
            recent_errors=recent_errors,
            parent=self,
        )
        dlg.exec()

    def _show_about(self) -> None:
        try:
            from importlib.metadata import version
            app_version = version("corridorkey")
        except Exception:
            # Running from source — read version from pyproject.toml
            try:
                import tomllib
                pyproject = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyproject.toml")
                with open(pyproject, "rb") as f:
                    app_version = tomllib.load(f)["project"]["version"]
            except Exception:
                app_version = "unknown"

        box = QMessageBox(self)
        box.setWindowTitle("About CorridorKey")
        box.setTextFormat(Qt.RichText)
        box.setText(
            f"<h2>CorridorKey v{app_version}</h2>"
            "<p>AI Green Screen Keyer<br>"
            '<a href="https://github.com/nikopueringer/CorridorKey#corridorkey-licensing-and-permissions">'
            "CC BY-NC-SA 4.0 License</a></p>"
            "<p><b>Special Thanks</b></p>"
            "<p>"
            '<a href="https://github.com/nikopueringer/">Niko Pueringer</a> — OG CorridorKey Creator<br>'
            '<a href="https://www.edzisk.com">Ed Zisk</a> — GUI, workflow, SFX, QA<br>'
            '<a href="https://www.clade.design/">Sara Ann Stewart</a> — Logo<br>'
            '<a href="https://github.com/Raiden129">Jhe Kim</a> — Hiera optimization<br>'
            '<a href="https://github.com/MarcelLieb">MarcelLieb</a> — Tiling optimization'
            "</p>"
        )
        # QMessageBox uses an internal QLabel — find it and enable clickable links
        for label in box.findChildren(QLabel):
            label.setOpenExternalLinks(True)
        box.exec()

    # ── Update Check ──────────────────────────────────────────

    def _get_local_version(self) -> str:
        try:
            from importlib.metadata import version
            return version("corridorkey")
        except Exception:
            try:
                import tomllib
                pyproject = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "pyproject.toml"
                )
                with open(pyproject, "rb") as f:
                    return tomllib.load(f)["project"]["version"]
            except Exception:
                return "0.0.0"

    def _check_for_updates(self) -> None:
        self._update_thread = _UpdateChecker(self._get_local_version())
        self._update_thread.update_available.connect(self._on_update_available)
        self._update_thread.start()

    @Slot(str)
    def _on_update_available(self, remote_version: str) -> None:
        self._update_btn.setText(f"Update Available (v{remote_version})")
        self._update_btn.setToolTip(
            f"A new version (v{remote_version}) is available.\n"
            "Click to save your session and run the updater."
        )
        self._update_btn.setVisible(True)

    def _run_update(self) -> None:
        reply = QMessageBox.question(
            self, "Update CorridorKey",
            "This will save your session, close the app, and run the updater.\n"
            "The app will relaunch automatically after updating.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return
        # Save session before closing
        self._auto_save_session()
        # Launch update script --relaunch detached, then quit
        import subprocess
        root = os.path.dirname(os.path.dirname(__file__))
        if os.name == "nt":
            bat = os.path.join(root, "3-update.bat")
            subprocess.Popen(
                ["cmd", "/c", "start", "", bat, "--relaunch"],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            sh = os.path.join(root, "3-update.sh")
            subprocess.Popen(
                [sh, "--relaunch"],
                start_new_session=True,
            )
        from PySide6.QtWidgets import QApplication
        QApplication.instance().quit()

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
        if (self._bg_cache is None
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


    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_queue_panel()

    def _position_queue_panel(self) -> None:
        """Keep the floating queue panel sized to the viewer area height."""
        if hasattr(self, '_workspace') and hasattr(self, '_queue_panel'):
            # Use the top section height (viewer+params) not the full workspace
            sizes = self._vsplitter.sizes()
            h = sizes[0] if sizes else self._workspace.height()
            self._queue_panel.setFixedHeight(h)
            self._queue_panel.move(0, 0)
            self._queue_panel.raise_()

    def closeEvent(self, event) -> None:
        """Clean shutdown — auto-save session, stop workers, unload engines."""
        if self._skip_shutdown_cleanup:
            super().closeEvent(event)
            return
        # Save annotation strokes before closing
        self._auto_save_annotations()
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
        if self._debug_console is not None:
            self._debug_console.close_permanently()
        super().closeEvent(event)

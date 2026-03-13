"""Preferences dialog — Edit > Preferences.

Provides user-configurable settings that persist across sessions via QSettings.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QLabel,
    QComboBox, QGroupBox, QProgressBar, QMessageBox, QApplication, QSpinBox,
)
from PySide6.QtCore import QSettings, Qt, QUrl, QThread, Signal


# QSettings keys
KEY_SHOW_TOOLTIPS = "ui/show_tooltips"
KEY_UI_SOUNDS = "ui/sounds_enabled"
KEY_COPY_SOURCE = "project/copy_source_videos"
KEY_LOOP_PLAYBACK = "playback/loop"
KEY_COPY_SEQUENCES = "project/copy_image_sequences"
KEY_EXR_COMPRESSION = "output/exr_compression"
KEY_TRACKER_MODEL = "tracking/sam2_model"
KEY_PARALLEL_CLIPS = "gpu/parallel_clips"

# Defaults
DEFAULT_SHOW_TOOLTIPS = True
DEFAULT_UI_SOUNDS = True
DEFAULT_COPY_SOURCE = True
DEFAULT_COPY_SEQUENCES = False
DEFAULT_LOOP_PLAYBACK = True
DEFAULT_EXR_COMPRESSION = "dwab"
DEFAULT_TRACKER_MODEL = "facebook/sam2.1-hiera-base-plus"
DEFAULT_PARALLEL_CLIPS = 1

EXR_COMPRESSION_OPTIONS = [
    ("DWAB — Lossy, Smallest Files", "dwab"),
    ("PIZ — Lossless, VFX Standard", "piz"),
    ("ZIP — Lossless, Scanline", "zip"),
    ("None — Uncompressed", "none"),
]

TRACKER_MODEL_OPTIONS = [
    ("Fast", "184 MB", "facebook/sam2.1-hiera-small"),
    ("Base+ (Default)", "324 MB", "facebook/sam2.1-hiera-base-plus"),
    ("Highest Quality", "898 MB", "facebook/sam2.1-hiera-large"),
]


def get_setting_bool(key: str, default: bool) -> bool:
    """Read a boolean setting from QSettings."""
    s = QSettings()
    return s.value(key, default, type=bool)


def get_setting_int(key: str, default: int) -> int:
    """Read an integer setting from QSettings."""
    s = QSettings()
    return s.value(key, default, type=int)


def get_setting_str(key: str, default: str) -> str:
    """Read a string setting from QSettings."""
    s = QSettings()
    value = s.value(key, default, type=str)
    return value or default


def get_tracker_model_cache_dir() -> Path:
    """Return the local Hugging Face cache directory used for SAM2 models."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def get_local_ffmpeg_dir() -> Path:
    """Return the repo-local bundled FFmpeg folder."""
    return Path(__file__).resolve().parents[2] / "tools" / "ffmpeg"


class _FFmpegRepairWorker(QThread):
    """Background repair worker so the Preferences dialog stays responsive."""

    progress = Signal(str, int, int)
    succeeded = Signal(str)
    failed = Signal(str)

    def run(self) -> None:
        from backend.ffmpeg_tools import repair_ffmpeg_install

        try:
            result = repair_ffmpeg_install(
                progress_callback=lambda phase, current, total: self.progress.emit(
                    phase, current, total
                )
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(result.message)


class PreferencesDialog(QDialog):
    """Application preferences dialog.

    Currently supports:
    - Toggle tooltips on/off globally
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(520)
        self.setModal(True)
        self._ffmpeg_repair_worker: _FFmpegRepairWorker | None = None
        self._local_ffmpeg_dir = get_local_ffmpeg_dir()

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # UI section
        ui_group = QGroupBox("User Interface")
        ui_layout = QVBoxLayout(ui_group)

        self._tooltips_cb = QCheckBox("Show tooltips on controls")
        self._tooltips_cb.setChecked(
            get_setting_bool(KEY_SHOW_TOOLTIPS, DEFAULT_SHOW_TOOLTIPS)
        )
        ui_layout.addWidget(self._tooltips_cb)

        self._sounds_cb = QCheckBox("UI sounds")
        self._sounds_cb.setChecked(
            get_setting_bool(KEY_UI_SOUNDS, DEFAULT_UI_SOUNDS)
        )
        ui_layout.addWidget(self._sounds_cb)

        layout.addWidget(ui_group)

        # Project section
        proj_group = QGroupBox("Project")
        proj_layout = QVBoxLayout(proj_group)

        self._copy_source_cb = QCheckBox("Copy source videos into project folder")
        self._copy_source_cb.setToolTip(
            "When enabled, imported videos are copied into the project folder.\n"
            "When disabled, the project references the original file in place.\n\n"
            "Note: Deleting a project never touches the original source file."
        )
        self._copy_source_cb.setChecked(
            get_setting_bool(KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE)
        )
        proj_layout.addWidget(self._copy_source_cb)

        self._copy_sequences_cb = QCheckBox("Copy imported image sequences into project folder")
        self._copy_sequences_cb.setToolTip(
            "When enabled, imported image sequence files are copied into the project.\n"
            "When disabled (default), the project references the original files in place.\n\n"
            "Referencing saves disk space for large EXR/TIF sequences.\n"
            "Original files are never modified regardless of this setting."
        )
        self._copy_sequences_cb.setChecked(
            get_setting_bool(KEY_COPY_SEQUENCES, DEFAULT_COPY_SEQUENCES)
        )
        proj_layout.addWidget(self._copy_sequences_cb)

        layout.addWidget(proj_group)

        # Output section
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)

        exr_label = QLabel("EXR compression")
        output_layout.addWidget(exr_label)

        self._exr_compression_combo = QComboBox()
        saved_compression = get_setting_str(KEY_EXR_COMPRESSION, DEFAULT_EXR_COMPRESSION)
        for label, value in EXR_COMPRESSION_OPTIONS:
            self._exr_compression_combo.addItem(label, value)
        idx = self._exr_compression_combo.findData(saved_compression)
        self._exr_compression_combo.setCurrentIndex(max(0, idx))
        self._exr_compression_combo.setToolTip(
            "Compression used when writing EXR output files.\n\n"
            "DWAB: Lossy wavelet, smallest files. Default.\n"
            "PIZ: Lossless wavelet, preferred by compositors.\n"
            "ZIP: Lossless deflate, good for clean renders.\n"
            "None: No compression, fastest write, largest files."
        )
        output_layout.addWidget(self._exr_compression_combo)

        layout.addWidget(output_group)

        # Playback section
        play_group = QGroupBox("Playback")
        play_layout = QVBoxLayout(play_group)

        self._loop_cb = QCheckBox("Loop playback within in/out range")
        self._loop_cb.setToolTip(
            "When enabled, playback loops back to the in-point\n"
            "after reaching the out-point (or start/end if no range)."
        )
        self._loop_cb.setChecked(
            get_setting_bool(KEY_LOOP_PLAYBACK, DEFAULT_LOOP_PLAYBACK)
        )
        play_layout.addWidget(self._loop_cb)

        layout.addWidget(play_group)

        # Tracking section
        tracking_group = QGroupBox("Tracking")
        tracking_layout = QVBoxLayout(tracking_group)

        tracking_label = QLabel("SAM2 model")
        tracking_layout.addWidget(tracking_label)

        self._tracker_model_combo = QComboBox()
        saved_model = get_setting_str(KEY_TRACKER_MODEL, DEFAULT_TRACKER_MODEL)
        for label, size, model_id in TRACKER_MODEL_OPTIONS:
            self._tracker_model_combo.addItem(f"{label}  ({size})", model_id)
        idx = self._tracker_model_combo.findData(saved_model)
        self._tracker_model_combo.setCurrentIndex(max(0, idx))
        self._tracker_model_combo.setToolTip(
            "Fast: lower VRAM, lower quality.\n"
            "Base+: best default tradeoff for this app.\n"
            "Highest Quality: slowest, heaviest tracker."
        )
        tracking_layout.addWidget(self._tracker_model_combo)

        tracking_info = QLabel(
            "Models download automatically on first use. "
            "Download progress appears in the status bar."
        )
        tracking_info.setWordWrap(True)
        tracking_info.setStyleSheet("color: #999980; font-size: 11px;")
        tracking_layout.addWidget(tracking_info)

        manage_label = QLabel("Manage models")
        tracking_layout.addWidget(manage_label)

        self._tracker_cache_dir = get_tracker_model_cache_dir()
        cache_row = QHBoxLayout()
        cache_row.setSpacing(8)
        self._cache_path_label = QLabel(str(self._tracker_cache_dir))
        self._cache_path_label.setWordWrap(True)
        self._cache_path_label.setStyleSheet("color: #999980; font-size: 11px;")
        self._cache_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        cache_row.addWidget(self._cache_path_label, 1)

        open_cache_btn = QPushButton("Open Cache Folder")
        open_cache_btn.clicked.connect(self._open_tracker_cache_dir)
        cache_row.addWidget(open_cache_btn)
        tracking_layout.addLayout(cache_row)

        layout.addWidget(tracking_group)

        ffmpeg_group = QGroupBox("Video Tools")
        ffmpeg_layout = QVBoxLayout(ffmpeg_group)

        ffmpeg_label = QLabel("FFmpeg status")
        ffmpeg_layout.addWidget(ffmpeg_label)

        self._ffmpeg_status_label = QLabel("")
        self._ffmpeg_status_label.setWordWrap(True)
        self._ffmpeg_status_label.setStyleSheet("font-size: 11px;")
        self._ffmpeg_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ffmpeg_layout.addWidget(self._ffmpeg_status_label)

        ffmpeg_info = QLabel(
            "Windows: Repair downloads a bundled full FFmpeg build into tools/ffmpeg "
            "without changing your system install.\n"
            "macOS: Repair installs FFmpeg via Homebrew.\n"
            "Linux: Repair copies the install command to your clipboard."
        )
        ffmpeg_info.setWordWrap(True)
        ffmpeg_info.setStyleSheet("color: #999980; font-size: 11px;")
        ffmpeg_layout.addWidget(ffmpeg_info)

        self._ffmpeg_progress = QProgressBar()
        self._ffmpeg_progress.setTextVisible(True)
        self._ffmpeg_progress.hide()
        ffmpeg_layout.addWidget(self._ffmpeg_progress)

        ffmpeg_btn_row = QHBoxLayout()
        ffmpeg_btn_row.setSpacing(8)

        self._repair_ffmpeg_btn = QPushButton("Repair FFmpeg")
        self._repair_ffmpeg_btn.setToolTip(
            "Windows: download and install a full bundled FFmpeg build into "
            "tools/ffmpeg, validate ffmpeg + ffprobe 7+, and switch CorridorKey "
            "to that local copy immediately.\n\n"
            "macOS: install FFmpeg via Homebrew and validate ffmpeg + ffprobe 7+.\n\n"
            "Linux: do not change system packages. CorridorKey shows the exact "
            "install commands and copies them to your clipboard instead."
        )
        self._repair_ffmpeg_btn.clicked.connect(self._on_repair_ffmpeg)
        ffmpeg_btn_row.addWidget(self._repair_ffmpeg_btn)

        self._open_ffmpeg_btn = QPushButton("Open FFmpeg Folder")
        self._open_ffmpeg_btn.setToolTip(
            "Open CorridorKey's bundled FFmpeg folder.\n"
            "If Repair FFmpeg has been run on Windows, this is where the local "
            "full build is stored."
        )
        self._open_ffmpeg_btn.clicked.connect(self._open_local_ffmpeg_dir)
        ffmpeg_btn_row.addWidget(self._open_ffmpeg_btn)
        ffmpeg_btn_row.addStretch(1)

        ffmpeg_layout.addLayout(ffmpeg_btn_row)

        layout.addWidget(ffmpeg_group)

        # Performance section
        perf_group = QGroupBox("Performance")
        perf_layout = QVBoxLayout(perf_group)

        parallel_row = QHBoxLayout()
        parallel_label = QLabel("Parallel frames")
        parallel_row.addWidget(parallel_label)

        self._parallel_clips_spin = QSpinBox()
        self._parallel_clips_spin.setRange(1, 8)
        self._parallel_clips_spin.setValue(
            get_setting_int(KEY_PARALLEL_CLIPS, DEFAULT_PARALLEL_CLIPS)
        )
        self._parallel_clips_spin.setToolTip(
            "Number of frames to process simultaneously.\n\n"
            "Uses multiple inference engine instances to process\n"
            "N frames at once within a single clip. Higher values\n"
            "use more GPU memory but increase throughput.\n\n"
            "1 = Sequential (default, safest)\n"
            "2 = ~70% faster, uses ~3 GB extra VRAM\n"
            "3-4 = Diminishing returns, needs 16+ GB VRAM\n\n"
            "Changes take effect on the next inference run."
        )
        self._parallel_clips_spin.setFixedWidth(60)
        parallel_row.addWidget(self._parallel_clips_spin)
        parallel_row.addStretch(1)
        perf_layout.addLayout(parallel_row)

        perf_info = QLabel(
            "Multiple engine instances process frames concurrently.\n"
            "Each extra engine adds ~3 GB VRAM but increases throughput."
        )
        perf_info.setWordWrap(True)
        perf_info.setStyleSheet("color: #999980; font-size: 11px;")
        perf_layout.addWidget(perf_info)

        layout.addWidget(perf_group)

        layout.addStretch(1)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self._cancel_btn)

        self._ok_btn = QPushButton("OK")
        self._ok_btn.setDefault(True)
        self._ok_btn.clicked.connect(self._save_and_accept)
        btn_layout.addWidget(self._ok_btn)

        layout.addLayout(btn_layout)
        self._refresh_ffmpeg_status()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Prevent closing the dialog while FFmpeg repair is active."""
        if self._ffmpeg_repair_worker is not None and self._ffmpeg_repair_worker.isRunning():
            event.ignore()
            return
        super().closeEvent(event)

    def _save_and_accept(self) -> None:
        """Persist settings and close."""
        s = QSettings()
        s.setValue(KEY_SHOW_TOOLTIPS, self._tooltips_cb.isChecked())
        s.setValue(KEY_UI_SOUNDS, self._sounds_cb.isChecked())
        s.setValue(KEY_COPY_SOURCE, self._copy_source_cb.isChecked())
        s.setValue(KEY_COPY_SEQUENCES, self._copy_sequences_cb.isChecked())
        s.setValue(KEY_LOOP_PLAYBACK, self._loop_cb.isChecked())
        s.setValue(KEY_EXR_COMPRESSION, self._exr_compression_combo.currentData())
        s.setValue(KEY_TRACKER_MODEL, self._tracker_model_combo.currentData())
        s.setValue(KEY_PARALLEL_CLIPS, self._parallel_clips_spin.value())
        # Apply sound mute immediately
        from ui.sounds.audio_manager import UIAudio
        UIAudio.set_muted(not self._sounds_cb.isChecked())
        self.accept()

    def _open_tracker_cache_dir(self) -> None:
        """Open the local cache folder where SAM2 checkpoints are stored."""
        self._tracker_cache_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._tracker_cache_dir)))

    def _open_local_ffmpeg_dir(self) -> None:
        """Open the bundled FFmpeg folder if it exists."""
        self._local_ffmpeg_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._local_ffmpeg_dir)))

    def _refresh_ffmpeg_status(self) -> None:
        """Refresh the FFmpeg health label and button state."""
        from backend.ffmpeg_tools import validate_ffmpeg_install

        result = validate_ffmpeg_install(require_probe=True)
        self._ffmpeg_status_label.setText(result.message)
        if result.ok:
            self._ffmpeg_status_label.setStyleSheet("color: #22C55E; font-size: 11px;")
        else:
            self._ffmpeg_status_label.setStyleSheet("color: #FFA500; font-size: 11px;")
        self._open_ffmpeg_btn.setEnabled(self._local_ffmpeg_dir.exists())

    def _set_ffmpeg_repair_busy(self, busy: bool) -> None:
        """Disable controls while a repair is running."""
        self._repair_ffmpeg_btn.setEnabled(not busy)
        self._open_ffmpeg_btn.setEnabled(not busy and self._local_ffmpeg_dir.exists())
        self._cancel_btn.setEnabled(not busy)
        self._ok_btn.setEnabled(not busy)

    def _set_parent_status_message(self, message: str) -> None:
        """Mirror long-running repair status to the main window status bar when available."""
        parent = self.parent()
        if parent is not None and hasattr(parent, "_status_bar"):
            parent._status_bar.set_message(message)

    def _on_repair_ffmpeg(self) -> None:
        """Repair FFmpeg on Windows or show manual install guidance elsewhere."""
        from backend.ffmpeg_tools import get_ffmpeg_install_help, validate_ffmpeg_install
        import sys

        current = validate_ffmpeg_install(require_probe=True)
        if current.ok:
            QMessageBox.information(
                self,
                "FFmpeg OK",
                f"{current.message}\n\nNo repair is needed.",
            )
            return

        # Linux: needs sudo, can't run from GUI — copy instructions to clipboard
        if sys.platform not in ("win32", "darwin"):
            help_text = get_ffmpeg_install_help()
            QApplication.clipboard().setText(help_text)
            QMessageBox.information(
                self,
                "Repair FFmpeg",
                help_text + "\n\nThe install command has been copied to your clipboard.\n"
                "Paste it into a terminal to install.",
            )
            return

        if sys.platform == "win32":
            confirm_msg = (
                "CorridorKey will download and install a full bundled FFmpeg build into:\n\n"
                f"{self._local_ffmpeg_dir}\n\n"
                "This does not modify your system-wide FFmpeg.\n\nContinue?"
            )
        else:
            confirm_msg = (
                "CorridorKey will install FFmpeg via Homebrew:\n\n"
                "    brew install ffmpeg\n\n"
                "Continue?"
            )

        reply = QMessageBox.question(
            self,
            "Repair FFmpeg",
            confirm_msg,
        )
        if reply != QMessageBox.Yes:
            return

        self._ffmpeg_progress.setRange(0, 0)
        self._ffmpeg_progress.setFormat("Preparing repair...")
        self._ffmpeg_progress.show()
        self._set_ffmpeg_repair_busy(True)
        self._set_parent_status_message("Repairing FFmpeg...")

        self._ffmpeg_repair_worker = _FFmpegRepairWorker(self)
        self._ffmpeg_repair_worker.progress.connect(self._on_ffmpeg_repair_progress)
        self._ffmpeg_repair_worker.succeeded.connect(self._on_ffmpeg_repair_succeeded)
        self._ffmpeg_repair_worker.failed.connect(self._on_ffmpeg_repair_failed)
        self._ffmpeg_repair_worker.start()

    def _on_ffmpeg_repair_progress(self, phase: str, current: int, total: int) -> None:
        """Update progress text while a repair download/extract is running."""
        self._ffmpeg_status_label.setText(phase)
        self._set_parent_status_message(phase)
        if total > 0:
            self._ffmpeg_progress.setRange(0, total)
            self._ffmpeg_progress.setValue(min(current, total))
        else:
            self._ffmpeg_progress.setRange(0, 0)
        self._ffmpeg_progress.setFormat(phase)

    def _finish_ffmpeg_repair(self) -> None:
        """Clear worker/progress state after repair completes."""
        self._ffmpeg_progress.hide()
        self._set_ffmpeg_repair_busy(False)
        self._set_parent_status_message("")
        if self._ffmpeg_repair_worker is not None:
            self._ffmpeg_repair_worker.deleteLater()
            self._ffmpeg_repair_worker = None

    def _on_ffmpeg_repair_succeeded(self, message: str) -> None:
        """Handle a successful FFmpeg repair."""
        self._finish_ffmpeg_repair()
        self._refresh_ffmpeg_status()
        QMessageBox.information(
            self,
            "FFmpeg Repaired",
            message + "\n\nCorridorKey will use FFmpeg immediately.",
        )

    def _on_ffmpeg_repair_failed(self, message: str) -> None:
        """Handle a failed FFmpeg repair."""
        self._finish_ffmpeg_repair()
        self._refresh_ffmpeg_status()
        QMessageBox.critical(self, "FFmpeg Repair Failed", message)

    @property
    def show_tooltips(self) -> bool:
        return self._tooltips_cb.isChecked()

    @property
    def copy_source(self) -> bool:
        return self._copy_source_cb.isChecked()

    @property
    def tracker_model(self) -> str:
        data = self._tracker_model_combo.currentData()
        return str(data or DEFAULT_TRACKER_MODEL)

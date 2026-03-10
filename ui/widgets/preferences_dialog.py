"""Preferences dialog — Edit > Preferences.

Provides user-configurable settings that persist across sessions via QSettings.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QLabel,
    QComboBox, QGroupBox,
)
from PySide6.QtCore import QSettings, Qt, QUrl


# QSettings keys
KEY_SHOW_TOOLTIPS = "ui/show_tooltips"
KEY_UI_SOUNDS = "ui/sounds_enabled"
KEY_COPY_SOURCE = "project/copy_source_videos"
KEY_LOOP_PLAYBACK = "playback/loop"
KEY_COPY_SEQUENCES = "project/copy_image_sequences"
KEY_EXR_COMPRESSION = "output/exr_compression"
KEY_TRACKER_MODEL = "tracking/sam2_model"

# Defaults
DEFAULT_SHOW_TOOLTIPS = True
DEFAULT_UI_SOUNDS = True
DEFAULT_COPY_SOURCE = True
DEFAULT_COPY_SEQUENCES = False
DEFAULT_LOOP_PLAYBACK = True
DEFAULT_EXR_COMPRESSION = "dwab"
DEFAULT_TRACKER_MODEL = "facebook/sam2.1-hiera-base-plus"

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


class PreferencesDialog(QDialog):
    """Application preferences dialog.

    Currently supports:
    - Toggle tooltips on/off globally
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(460)
        self.setModal(True)

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
        layout.addStretch(1)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._save_and_accept)
        btn_layout.addWidget(ok_btn)

        layout.addLayout(btn_layout)

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
        # Apply sound mute immediately
        from ui.sounds.audio_manager import UIAudio
        UIAudio.set_muted(not self._sounds_cb.isChecked())
        self.accept()

    def _open_tracker_cache_dir(self) -> None:
        """Open the local cache folder where SAM2 checkpoints are stored."""
        self._tracker_cache_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._tracker_cache_dir)))

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

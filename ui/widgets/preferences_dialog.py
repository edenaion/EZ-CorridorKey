"""Preferences dialog — Edit > Preferences.

Provides user-configurable settings that persist across sessions via QSettings.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QLabel,
    QComboBox, QGroupBox,
)
from PySide6.QtCore import QSettings, Qt


# QSettings keys
KEY_SHOW_TOOLTIPS = "ui/show_tooltips"
KEY_UI_SOUNDS = "ui/sounds_enabled"
KEY_COPY_SOURCE = "project/copy_source_videos"
KEY_LOOP_PLAYBACK = "playback/loop"
KEY_COPY_SEQUENCES = "project/copy_image_sequences"
KEY_TRACKER_MODEL = "tracking/sam2_model"

# Defaults
DEFAULT_SHOW_TOOLTIPS = True
DEFAULT_UI_SOUNDS = True
DEFAULT_COPY_SOURCE = True
DEFAULT_COPY_SEQUENCES = False
DEFAULT_LOOP_PLAYBACK = True
DEFAULT_TRACKER_MODEL = "facebook/sam2.1-hiera-base-plus"

TRACKER_MODEL_OPTIONS = [
    ("Fast", "facebook/sam2.1-hiera-small"),
    ("Base+ (Default)", "facebook/sam2.1-hiera-base-plus"),
    ("Highest Quality", "facebook/sam2.1-hiera-large"),
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


class PreferencesDialog(QDialog):
    """Application preferences dialog.

    Currently supports:
    - Toggle tooltips on/off globally
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(360)
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
        for label, model_id in TRACKER_MODEL_OPTIONS:
            self._tracker_model_combo.addItem(label, model_id)
        idx = self._tracker_model_combo.findData(saved_model)
        self._tracker_model_combo.setCurrentIndex(max(0, idx))
        self._tracker_model_combo.setToolTip(
            "Fast: lower VRAM, lower quality.\n"
            "Base+: best default tradeoff for this app.\n"
            "Highest Quality: slowest, heaviest tracker."
        )
        tracking_layout.addWidget(self._tracker_model_combo)

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
        s.setValue(KEY_TRACKER_MODEL, self._tracker_model_combo.currentData())
        # Apply sound mute immediately
        from ui.sounds.audio_manager import UIAudio
        UIAudio.set_muted(not self._sounds_cb.isChecked())
        self.accept()

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

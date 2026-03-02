"""Preferences dialog — Edit > Preferences.

Provides user-configurable settings that persist across sessions via QSettings.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QLabel,
    QGroupBox,
)
from PySide6.QtCore import QSettings, Qt


# QSettings keys
KEY_SHOW_TOOLTIPS = "ui/show_tooltips"
KEY_COPY_SOURCE = "project/copy_source_videos"
KEY_LOOP_PLAYBACK = "playback/loop"

# Defaults
DEFAULT_SHOW_TOOLTIPS = True
DEFAULT_COPY_SOURCE = True
DEFAULT_LOOP_PLAYBACK = True


def get_setting_bool(key: str, default: bool) -> bool:
    """Read a boolean setting from QSettings."""
    s = QSettings()
    return s.value(key, default, type=bool)


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
        s.setValue(KEY_COPY_SOURCE, self._copy_source_cb.isChecked())
        s.setValue(KEY_LOOP_PLAYBACK, self._loop_cb.isChecked())
        self.accept()

    @property
    def show_tooltips(self) -> bool:
        return self._tooltips_cb.isChecked()

    @property
    def copy_source(self) -> bool:
        return self._copy_source_cb.isChecked()

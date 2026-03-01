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

# Defaults
DEFAULT_SHOW_TOOLTIPS = True


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
        self.accept()

    @property
    def show_tooltips(self) -> bool:
        return self._tooltips_cb.isChecked()

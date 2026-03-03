"""Compact volume control for the menu bar — speaker icon + slider."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSlider, QWidget


# Speaker icons by volume tier
_ICON_MUTED = "\U0001f507"  # 🔇
_ICON_LOW = "\U0001f508"  # 🔈
_ICON_MED = "\U0001f509"  # 🔉
_ICON_HIGH = "\U0001f50a"  # 🔊


class VolumeControl(QWidget):
    """Speaker icon + horizontal slider.  Lives in the menu bar corner."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from ui.sounds.audio_manager import UIAudio

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(4)

        # Speaker icon — click toggles mute
        self._icon = QPushButton(_ICON_HIGH)
        self._icon.setFixedSize(22, 22)
        self._icon.setFlat(True)
        self._icon.setCursor(Qt.PointingHandCursor)
        self._icon.setToolTip("Click to mute / unmute")
        self._icon.setStyleSheet(
            "QPushButton { font-size: 13px; border: none; background: transparent;"
            "padding: 0; color: #808070; }"
            "QPushButton:hover { color: #FFF203; }"
        )
        self._icon.clicked.connect(self._toggle_mute)
        layout.addWidget(self._icon)

        # Volume slider
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setObjectName("volumeSlider")
        self._slider.setRange(0, 100)
        self._slider.setFixedWidth(60)
        self._slider.setToolTip("Volume")
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider)

        # Restore persisted state
        self._pre_mute_volume = 100
        vol = int(round(UIAudio.get_volume() * 100))
        muted = UIAudio.is_muted()
        self._slider.setValue(vol)
        if muted:
            self._pre_mute_volume = vol or 100
            self._slider.setEnabled(False)
        self._update_icon()

    # -- Public API (called from MainWindow when Ctrl+M fires) --

    def sync_mute_state(self) -> None:
        """Re-read UIAudio mute state and update icon + slider."""
        from ui.sounds.audio_manager import UIAudio

        muted = UIAudio.is_muted()
        self._slider.setEnabled(not muted)
        self._update_icon()

    # -- Internals --

    def _toggle_mute(self) -> None:
        from ui.sounds.audio_manager import UIAudio
        from ui.widgets.preferences_dialog import KEY_UI_SOUNDS
        from PySide6.QtCore import QSettings

        muted = UIAudio.is_muted()
        if muted:
            # Unmute — restore previous volume
            UIAudio.set_muted(False)
            QSettings().setValue(KEY_UI_SOUNDS, True)
            self._slider.setEnabled(True)
            self._slider.setValue(self._pre_mute_volume)
        else:
            # Mute — remember current volume
            self._pre_mute_volume = self._slider.value() or 100
            UIAudio.set_muted(True)
            QSettings().setValue(KEY_UI_SOUNDS, False)
            self._slider.setEnabled(False)
        self._update_icon()

    def _on_slider_changed(self, value: int) -> None:
        from ui.sounds.audio_manager import UIAudio

        UIAudio.set_volume(value / 100.0)
        self._update_icon()

    def _update_icon(self) -> None:
        from ui.sounds.audio_manager import UIAudio

        if UIAudio.is_muted():
            self._icon.setText(_ICON_MUTED)
            return
        v = self._slider.value()
        if v == 0:
            self._icon.setText(_ICON_MUTED)
        elif v <= 33:
            self._icon.setText(_ICON_LOW)
        elif v <= 66:
            self._icon.setText(_ICON_MED)
        else:
            self._icon.setText(_ICON_HIGH)

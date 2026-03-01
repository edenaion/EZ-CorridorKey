"""Individual clip card widget for the clip browser list.

Displays clip name + state badge with brand colors.
State badge colors:
    RAW=#808070, MASKED=#009ADA, READY=#FFF203, COMPLETE=#22C55E, ERROR=#D10000
"""
from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout
from PySide6.QtCore import Qt

from backend import ClipEntry, ClipState

_STATE_COLORS: dict[ClipState, str] = {
    ClipState.RAW: "#808070",
    ClipState.MASKED: "#009ADA",
    ClipState.READY: "#FFF203",
    ClipState.COMPLETE: "#22C55E",
    ClipState.ERROR: "#D10000",
}


class ClipCard(QFrame):
    """A single clip card showing name, state badge, and frame count."""

    def __init__(self, clip: ClipEntry, parent=None):
        super().__init__(parent)
        self.setObjectName("clipCard")
        self._clip = clip
        self._selected = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        # State badge (colored dot + state text)
        self._badge = QLabel(clip.state.value)
        self._badge.setObjectName(f"stateBadge_{clip.state.value}")
        self._badge.setFixedWidth(70)
        self._badge.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._badge)

        # Clip info
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(1)

        self._name_label = QLabel(clip.name)
        self._name_label.setObjectName("clipName")
        info_layout.addWidget(self._name_label)

        # Frame count subtitle
        frame_text = ""
        if clip.input_asset:
            frame_text = f"{clip.input_asset.frame_count} frames"
            if clip.input_asset.asset_type == "video":
                frame_text += " (video)"
        self._detail_label = QLabel(frame_text)
        self._detail_label.setStyleSheet("font-size: 10px; color: #808070;")
        info_layout.addWidget(self._detail_label)

        layout.addLayout(info_layout, 1)

        # Warning indicator
        if clip.warnings:
            warn_label = QLabel(f"({len(clip.warnings)})")
            warn_label.setStyleSheet("color: #FFA500; font-size: 10px;")
            warn_label.setToolTip("\n".join(clip.warnings))
            layout.addWidget(warn_label)

    @property
    def clip(self) -> ClipEntry:
        return self._clip

    @property
    def selected(self) -> bool:
        return self._selected

    @selected.setter
    def selected(self, value: bool) -> None:
        self._selected = value
        self.setProperty("selected", value)
        self.style().unpolish(self)
        self.style().polish(self)

    def update_state(self, new_state: ClipState) -> None:
        """Update the displayed state badge."""
        self._clip.state = new_state
        self._badge.setText(new_state.value)
        self._badge.setObjectName(f"stateBadge_{new_state.value}")
        self._badge.setStyleSheet(f"color: {_STATE_COLORS.get(new_state, '#808070')};")

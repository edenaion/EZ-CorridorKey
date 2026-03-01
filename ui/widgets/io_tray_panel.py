"""Bottom I/O tray panel — Topaz-style Input/Exports thumbnail strips.

Shows two horizontal-scrolling rows:
- INPUT (N): All loaded clips with thumbnails
- EXPORTS (N): Only COMPLETE clips with output thumbnails

Clicking a card selects the clip and loads it in the preview viewport.
"""
from __future__ import annotations

import logging

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, Signal, QRect, QSize
from PySide6.QtGui import QPainter, QColor, QImage, QMouseEvent

from backend import ClipEntry, ClipState
from ui.models.clip_model import ClipListModel

logger = logging.getLogger(__name__)

# State → color mapping (matches brand palette)
_STATE_COLORS: dict[ClipState, str] = {
    ClipState.EXTRACTING: "#FF8C00",
    ClipState.RAW: "#808070",
    ClipState.MASKED: "#009ADA",
    ClipState.READY: "#FFF203",
    ClipState.COMPLETE: "#22C55E",
    ClipState.ERROR: "#D10000",
}


class ThumbnailCanvas(QWidget):
    """Custom-painted horizontal strip of clip thumbnail cards.

    Each card is CARD_WIDTH wide and shows: thumbnail, clip name, state badge,
    and frame count. Mouse clicks emit card_clicked with the ClipEntry.
    """

    card_clicked = Signal(object)  # ClipEntry

    CARD_WIDTH = 130
    CARD_SPACING = 4
    CARD_PADDING = 6
    THUMB_W = 110
    THUMB_H = 62

    def __init__(self, parent=None):
        super().__init__(parent)
        self._clips: list[ClipEntry] = []
        self._model: ClipListModel | None = None
        self.setMinimumHeight(100)

    def set_clips(self, clips: list[ClipEntry], model: ClipListModel) -> None:
        """Update the displayed clips and trigger repaint."""
        self._clips = list(clips)
        self._model = model
        total_w = max(1, len(clips) * (self.CARD_WIDTH + self.CARD_SPACING))
        self.setFixedWidth(total_w)
        self.update()

    def paintEvent(self, event) -> None:
        if not self._clips:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)

        for i, clip in enumerate(self._clips):
            x = i * (self.CARD_WIDTH + self.CARD_SPACING)
            card_rect = QRect(x, 0, self.CARD_WIDTH, self.height())

            # Skip cards not in the visible region
            if not card_rect.intersects(event.rect()):
                continue

            self._paint_card(p, card_rect, clip)

        p.end()

    def _paint_card(self, p: QPainter, rect: QRect, clip: ClipEntry) -> None:
        pad = self.CARD_PADDING

        # Card background
        p.fillRect(rect, QColor("#1A1900"))

        # Border
        p.setPen(QColor("#2A2910"))
        p.drawRect(rect.adjusted(0, 0, -1, -1))

        # Thumbnail
        thumb_rect = QRect(
            rect.x() + (self.CARD_WIDTH - self.THUMB_W) // 2,
            rect.y() + pad,
            self.THUMB_W,
            self.THUMB_H,
        )
        thumb = self._model.get_thumbnail(clip.name) if self._model else None
        if isinstance(thumb, QImage) and not thumb.isNull():
            scaled = thumb.scaled(
                self.THUMB_W, self.THUMB_H,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            dx = thumb_rect.x() + (self.THUMB_W - scaled.width()) // 2
            dy = thumb_rect.y() + (self.THUMB_H - scaled.height()) // 2
            p.drawImage(dx, dy, scaled)
        else:
            p.fillRect(thumb_rect, QColor("#0A0A00"))
            p.setPen(QColor("#3A3A30"))
            p.drawRect(thumb_rect.adjusted(0, 0, -1, -1))

        # State badge (top-right of card)
        badge_color = QColor(_STATE_COLORS.get(clip.state, "#808070"))
        p.setPen(badge_color)
        font = p.font()
        font.setPointSize(8)
        font.setBold(True)
        p.setFont(font)
        badge_rect = QRect(rect.x() + pad, rect.y() + pad, self.CARD_WIDTH - pad * 2, 14)
        p.drawText(badge_rect, Qt.AlignRight | Qt.AlignTop, clip.state.value)

        # Clip name (below thumbnail)
        text_y = rect.y() + pad + self.THUMB_H + 4
        p.setPen(QColor("#E0E0E0"))
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)
        name_rect = QRect(rect.x() + pad, text_y, self.CARD_WIDTH - pad * 2, 16)
        metrics = p.fontMetrics()
        elided = metrics.elidedText(clip.name, Qt.ElideRight, name_rect.width())
        p.drawText(name_rect, Qt.AlignLeft | Qt.AlignVCenter, elided)

        # Frame count (below name)
        if clip.input_asset:
            p.setPen(QColor("#808070"))
            font.setPointSize(8)
            font.setBold(False)
            p.setFont(font)
            info_rect = QRect(rect.x() + pad, text_y + 14, self.CARD_WIDTH - pad * 2, 14)
            info_text = f"{clip.input_asset.frame_count} frames"
            if clip.input_asset.asset_type == "video":
                info_text += " (video)"
            p.drawText(info_rect, Qt.AlignLeft | Qt.AlignVCenter, info_text)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._clips:
            x = event.position().x()
            idx = int(x // (self.CARD_WIDTH + self.CARD_SPACING))
            if 0 <= idx < len(self._clips):
                self.card_clicked.emit(self._clips[idx])

    def sizeHint(self) -> QSize:
        w = max(1, len(self._clips) * (self.CARD_WIDTH + self.CARD_SPACING))
        return QSize(w, 100)


class IOTrayPanel(QWidget):
    """Bottom panel with Input and Exports thumbnail strips.

    Input section shows all loaded clips. Exports section shows only
    COMPLETE clips. Clicking a card emits clip_clicked.
    """

    clip_clicked = Signal(object)  # ClipEntry

    def __init__(self, model: ClipListModel, parent=None):
        super().__init__(parent)
        self.setObjectName("ioTrayPanel")
        self._model = model
        self.setMinimumHeight(80)
        self.setMaximumHeight(250)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Content: two strips side by side
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)

        # Input section
        input_section = QVBoxLayout()
        input_section.setContentsMargins(0, 0, 0, 0)
        input_section.setSpacing(0)

        self._input_header = QLabel("INPUT (0)")
        self._input_header.setObjectName("trayHeader")
        input_section.addWidget(self._input_header)

        self._input_scroll = QScrollArea()
        self._input_scroll.setObjectName("trayScroll")
        self._input_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._input_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._input_scroll.setWidgetResizable(False)

        self._input_canvas = ThumbnailCanvas()
        self._input_canvas.card_clicked.connect(self.clip_clicked.emit)
        self._input_scroll.setWidget(self._input_canvas)

        input_section.addWidget(self._input_scroll, 1)

        input_widget = QWidget()
        input_widget.setLayout(input_section)
        content.addWidget(input_widget, 1)

        # Vertical divider
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFixedWidth(1)
        divider.setStyleSheet("background-color: #2A2910;")
        content.addWidget(divider)

        # Exports section
        export_section = QVBoxLayout()
        export_section.setContentsMargins(0, 0, 0, 0)
        export_section.setSpacing(0)

        self._export_header = QLabel("EXPORTS (0)")
        self._export_header.setObjectName("trayHeader")
        export_section.addWidget(self._export_header)

        self._export_scroll = QScrollArea()
        self._export_scroll.setObjectName("trayScroll")
        self._export_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._export_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._export_scroll.setWidgetResizable(False)

        self._export_canvas = ThumbnailCanvas()
        self._export_canvas.card_clicked.connect(self.clip_clicked.emit)
        self._export_scroll.setWidget(self._export_canvas)

        export_section.addWidget(self._export_scroll, 1)

        export_widget = QWidget()
        export_widget.setLayout(export_section)
        content.addWidget(export_widget, 1)

        layout.addLayout(content)

        # Connect to model signals for auto-rebuild
        self._model.modelReset.connect(self._rebuild)
        self._model.dataChanged.connect(self._on_data_changed)
        self._model.clip_count_changed.connect(lambda _: self._rebuild())

    def _rebuild(self) -> None:
        """Rebuild both strips from current model data."""
        all_clips = self._model.clips
        complete_clips = [c for c in all_clips if c.state == ClipState.COMPLETE]

        self._input_canvas.set_clips(all_clips, self._model)
        self._export_canvas.set_clips(complete_clips, self._model)

        self._input_header.setText(f"INPUT ({len(all_clips)})")
        self._export_header.setText(f"EXPORTS ({len(complete_clips)})")

    def _on_data_changed(self, top_left, bottom_right, roles) -> None:
        """Handle model data changes — rebuild to catch state transitions."""
        self._rebuild()

    def refresh(self) -> None:
        """Force rebuild (called after worker completes a clip)."""
        self._rebuild()

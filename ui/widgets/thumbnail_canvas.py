"""Thumbnail canvas widget — wrapping grid of clip cards with vertical scroll.

Extracted from io_tray_panel.py for maintainability.
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QToolTip
from PySide6.QtCore import Qt, Signal, QRect, QSize, QEvent
from PySide6.QtGui import QPainter, QColor, QImage, QMouseEvent

from backend import ClipEntry, ClipState
from ui.models.clip_model import ClipListModel

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
    """Wrapping grid of clip thumbnail cards with vertical scroll.

    Cards flow left-to-right, top-to-bottom, wrapping into rows based on
    available width. A vertical scrollbar appears when rows exceed the
    visible height.
    """

    card_clicked = Signal(object)  # ClipEntry (single left-click)
    card_double_clicked = Signal(object)  # ClipEntry (double-click)
    multi_select_toggled = Signal(object)  # ClipEntry (Ctrl+click toggle)
    shift_select_requested = Signal(object)  # ClipEntry (Shift+click range)
    context_menu_requested = Signal(object)  # ClipEntry (right-click)
    folder_icon_clicked = Signal(object)  # ClipEntry (folder icon click)

    CARD_WIDTH = 130
    CARD_HEIGHT = 110
    CARD_SPACING = 4
    CARD_PADDING = 6
    THUMB_W = 110
    THUMB_H = 62

    def __init__(
        self,
        parent=None,
        show_manifest_tooltip: bool = False,
        thumbnail_kind: str = "input",
    ):
        super().__init__(parent)
        self._clips: list[ClipEntry] = []
        self._model: ClipListModel | None = None
        self._show_manifest_tooltip = show_manifest_tooltip
        self._thumbnail_kind = thumbnail_kind
        self._selected_names: set[str] = set()
        self._hovered_name: str | None = None
        self._thumb_cache: dict[str, QImage] = {}  # name → scaled thumbnail
        self._cols = 1  # current number of columns in the grid
        self.setMouseTracking(True)
        self.setMinimumHeight(self.CARD_HEIGHT)

    def set_clips(self, clips: list[ClipEntry], model: ClipListModel) -> None:
        """Update the displayed clips and trigger repaint."""
        # Invalidate thumbnail cache when clip list changes
        new_names = {c.name for c in clips}
        old_names = {c.name for c in self._clips}
        if new_names != old_names:
            self._thumb_cache = {k: v for k, v in self._thumb_cache.items()
                                 if k in new_names}
        self._clips = list(clips)
        self._model = model
        self._reflow()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reflow()

    def _reflow(self) -> None:
        """Recompute grid layout: wrap cards into rows, set min height for vertical scroll."""
        parent_scroll = self.parent()
        if parent_scroll and hasattr(parent_scroll, 'viewport'):
            avail_w = parent_scroll.viewport().width()
        else:
            avail_w = self.width()
        if avail_w < self.CARD_WIDTH:
            avail_w = self.CARD_WIDTH + self.CARD_SPACING
        self._cols = max(1, avail_w // (self.CARD_WIDTH + self.CARD_SPACING))
        rows = max(1, (len(self._clips) + self._cols - 1) // self._cols) if self._clips else 1
        row_h = self.CARD_HEIGHT + self.CARD_SPACING
        total_h = max(self.CARD_HEIGHT, rows * row_h)
        self.setMinimumHeight(total_h)
        self.update()

    def set_selected(self, name: str | None) -> None:
        """Set single-selected clip (clears multi-select, draws highlight border)."""
        new_set = {name} if name else set()
        if self._selected_names != new_set:
            self._selected_names = new_set
            self.update()

    def set_multi_selected(self, names: set[str]) -> None:
        """Set the multi-selected clip names."""
        if self._selected_names != names:
            self._selected_names = set(names)
            self.update()

    def _card_rect_for(self, index: int) -> QRect:
        """Return the QRect for card at the given index in the grid."""
        col = index % self._cols
        row = index // self._cols
        x = col * (self.CARD_WIDTH + self.CARD_SPACING)
        y = row * (self.CARD_HEIGHT + self.CARD_SPACING)
        return QRect(x, y, self.CARD_WIDTH, self.CARD_HEIGHT)

    def paintEvent(self, event) -> None:
        if not self._clips:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)

        for i, clip in enumerate(self._clips):
            card_rect = self._card_rect_for(i)

            # Skip cards not in the visible region
            if not card_rect.intersects(event.rect()):
                continue

            self._paint_card(p, card_rect, clip)

        p.end()

    def _paint_card(self, p: QPainter, rect: QRect, clip: ClipEntry) -> None:
        pad = self.CARD_PADDING
        is_selected = clip.name in self._selected_names
        is_hovered = clip.name == self._hovered_name and not is_selected

        # Card background
        if is_selected:
            bg = QColor("#252413")
        elif is_hovered:
            bg = QColor("#1E1D0A")
        else:
            bg = QColor("#1A1900")
        p.fillRect(rect, bg)

        # Border — yellow for selected, subtle yellow for hover, default otherwise
        if is_selected:
            p.setPen(QColor("#FFF203"))
            p.drawRect(rect.adjusted(0, 0, -1, -1))
            p.drawRect(rect.adjusted(1, 1, -2, -2))  # 2px border
        elif is_hovered:
            p.setPen(QColor(255, 242, 3, 100))  # subtle yellow glow
            p.drawRect(rect.adjusted(0, 0, -1, -1))
        else:
            p.setPen(QColor("#2A2910"))
            p.drawRect(rect.adjusted(0, 0, -1, -1))

        # Thumbnail
        thumb_rect = QRect(
            rect.x() + (self.CARD_WIDTH - self.THUMB_W) // 2,
            rect.y() + pad,
            self.THUMB_W,
            self.THUMB_H,
        )
        # Use cached scaled thumbnail to avoid expensive SmoothTransformation
        # on every repaint.
        scaled = self._thumb_cache.get(clip.name)
        if scaled is None and self._model:
            thumb = self._model.get_thumbnail(clip.name, kind=self._thumbnail_kind)
            if isinstance(thumb, QImage) and not thumb.isNull():
                scaled = thumb.scaled(
                    self.THUMB_W, self.THUMB_H,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
                self._thumb_cache[clip.name] = scaled
        if scaled is not None:
            dx = thumb_rect.x() + (self.THUMB_W - scaled.width()) // 2
            dy = thumb_rect.y() + (self.THUMB_H - scaled.height()) // 2
            p.drawImage(dx, dy, scaled)
        else:
            p.fillRect(thumb_rect, QColor("#0A0A00"))
            p.setPen(QColor("#3A3A30"))
            p.drawRect(thumb_rect.adjusted(0, 0, -1, -1))

        # State badge (top-right over thumbnail, with background pill)
        badge_color = QColor(_STATE_COLORS.get(clip.state, "#808070"))
        font = p.font()
        font.setPointSize(8)
        font.setBold(True)
        p.setFont(font)
        badge_text = clip.state.value
        metrics = p.fontMetrics()
        text_w = metrics.horizontalAdvance(badge_text)
        bg_rect = QRect(
            rect.x() + self.CARD_WIDTH - pad - text_w - 6,
            rect.y() + pad,
            text_w + 6, 14,
        )
        p.fillRect(bg_rect, QColor(0, 0, 0, 128))
        p.setPen(badge_color)
        p.drawText(bg_rect, Qt.AlignCenter, badge_text)

        # Clip name (below thumbnail, with background)
        text_y = rect.y() + pad + self.THUMB_H + 4
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)
        name_rect = QRect(rect.x() + pad, text_y, self.CARD_WIDTH - pad * 2, 16)
        metrics = p.fontMetrics()
        elided = metrics.elidedText(clip.name, Qt.ElideRight, name_rect.width())
        p.fillRect(name_rect, QColor(0, 0, 0, 128))
        p.setPen(QColor("#E0E0E0"))
        p.drawText(name_rect, Qt.AlignLeft | Qt.AlignVCenter, elided)

        # Source type badge (top-left over thumbnail, input cards only)
        if not self._show_manifest_tooltip and clip.source_type != "unknown":
            src_icon = "\U0001F39E" if clip.source_type == "video" else "\U0001F4F7"  # film frames / camera
            src_font = p.font()
            src_font.setPointSize(9)
            src_font.setBold(False)
            p.setFont(src_font)
            src_rect = QRect(rect.x() + pad, rect.y() + pad, 18, 18)
            p.fillRect(src_rect, QColor(0, 0, 0, 140))
            p.setPen(QColor("#C0C0A0"))
            p.drawText(src_rect, Qt.AlignCenter, src_icon)

        # Frame count (below name, with background)
        if clip.input_asset:
            font.setPointSize(8)
            font.setBold(False)
            p.setFont(font)
            info_rect = QRect(rect.x() + pad, text_y + 14, self.CARD_WIDTH - pad * 2, 14)
            info_text = f"{clip.input_asset.frame_count} frames"
            if clip.input_asset.asset_type == "video":
                info_text += " (video)"
            elif clip.source_type == "sequence":
                info_text += " (imported)"
            p.fillRect(info_rect, QColor(0, 0, 0, 128))
            p.setPen(QColor("#808070"))
            p.drawText(info_rect, Qt.AlignLeft | Qt.AlignVCenter, info_text)

        # Folder icon (top-left of thumbnail, export cards only)
        if self._show_manifest_tooltip:
            icon_size = 18
            icon_rect = QRect(rect.x() + pad, rect.y() + pad, icon_size, icon_size)
            is_icon_hovered = (clip.name == self._hovered_name
                               and hasattr(self, '_hover_pos')
                               and icon_rect.contains(self._hover_pos))
            bg_alpha = 200 if is_icon_hovered else 140
            p.fillRect(icon_rect, QColor(0, 0, 0, bg_alpha))
            folder_font = p.font()
            folder_font.setPointSize(10)
            folder_font.setBold(False)
            p.setFont(folder_font)
            p.setPen(QColor("#FFF203") if is_icon_hovered else QColor("#C0C0A0"))
            p.drawText(icon_rect, Qt.AlignCenter, "\U0001F4C2")

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position().toPoint()
        clip, _ = self._card_at(pos)
        name = clip.name if clip else None
        self._hover_pos = pos
        if name != self._hovered_name:
            self._hovered_name = name
        self.update()

    def leaveEvent(self, event) -> None:
        if self._hovered_name is not None:
            self._hovered_name = None
            self.update()

    def _folder_icon_rect(self, clip_index: int) -> QRect:
        """Return the folder icon QRect for a given card index."""
        r = self._card_rect_for(clip_index)
        return QRect(r.x() + self.CARD_PADDING, r.y() + self.CARD_PADDING, 18, 18)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position().toPoint()
        if event.button() == Qt.LeftButton and self._clips:
            clip, idx = self._card_at(pos)
            if clip:
                # Check folder icon click (export cards only)
                if self._show_manifest_tooltip and idx is not None:
                    if self._folder_icon_rect(idx).contains(pos):
                        self.folder_icon_clicked.emit(clip)
                        return
                from ui.sounds.audio_manager import UIAudio
                UIAudio.click()
                if event.modifiers() & Qt.ShiftModifier:
                    self.shift_select_requested.emit(clip)
                elif event.modifiers() & Qt.ControlModifier:
                    self.multi_select_toggled.emit(clip)
                else:
                    self.card_clicked.emit(clip)
        elif event.button() == Qt.RightButton and self._clips:
            clip, _ = self._card_at(pos)
            if clip:
                self.context_menu_requested.emit(clip)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._clips:
            clip, _ = self._card_at(event.position().toPoint())
            if clip:
                self.card_double_clicked.emit(clip)

    def sizeHint(self) -> QSize:
        return QSize(self.CARD_WIDTH, self.CARD_HEIGHT)

    def _card_at(self, pos) -> tuple[ClipEntry | None, int | None]:
        """Return (ClipEntry, index) under the given point, or (None, None)."""
        if not self._clips:
            return None, None
        for i in range(len(self._clips)):
            if self._card_rect_for(i).contains(pos):
                return self._clips[i], i
        return None, None

    def event(self, ev: QEvent) -> bool:
        if ev.type() == QEvent.ToolTip and self._show_manifest_tooltip:
            # QHelpEvent: older PySide6 uses pos()/globalPos(),
            # newer versions use position()/globalPosition().
            local_pos = (ev.position().toPoint()
                         if hasattr(ev, 'position') else ev.pos())
            global_pos = (ev.globalPosition().toPoint()
                          if hasattr(ev, 'globalPosition') else ev.globalPos())
            clip, _ = self._card_at(local_pos)
            tip = _format_manifest_tooltip(clip) if clip else ""
            if tip:
                QToolTip.showText(global_pos, tip, self)
            else:
                QToolTip.hideText()
            return True
        return super().event(ev)


def _format_manifest_tooltip(clip: ClipEntry) -> str:
    """Build a tooltip string from the clip's .corridorkey_manifest.json."""
    manifest = clip._read_manifest()
    if manifest is None:
        return ""

    lines: list[str] = [f"<b>{clip.name}</b> — Export Settings"]

    # Outputs + formats
    enabled = manifest.get("enabled_outputs", [])
    formats = manifest.get("formats", {})
    if enabled:
        out_parts = []
        for name in enabled:
            fmt = formats.get(name, "?").upper()
            out_parts.append(f"{name.upper()} ({fmt})")
        lines.append(f"<b>Outputs:</b> {', '.join(out_parts)}")

    # Params
    params = manifest.get("params", {})
    if params:
        cs = "Linear" if params.get("input_is_linear") else "sRGB"
        lines.append(f"<b>Color Space:</b> {cs}")
        ds = params.get("despill_strength", 1.0)
        lines.append(f"<b>Despill:</b> {ds:.0%}")
        rs = params.get("refiner_scale", 1.0)
        lines.append(f"<b>Refiner:</b> {rs:.0%}")
        if params.get("auto_despeckle"):
            sz = params.get("despeckle_size", 400)
            lines.append(f"<b>Despeckle:</b> On (size {sz})")
        else:
            lines.append(f"<b>Despeckle:</b> Off")

    return "<br>".join(lines)

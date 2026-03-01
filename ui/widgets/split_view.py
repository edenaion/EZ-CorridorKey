"""Split view widget with draggable yellow divider, zoom, and pan.

Renders two QImages (left=before, right=after) with a split divider.
When split is disabled, renders a single image full-width.

Uses QImage as internal currency (Codex finding: QPixmap can use
platform-native GPU backing, QImage is guaranteed CPU-only).

Zoom/pan: Ctrl+wheel zooms, middle-click pans, double-click resets.
Hit-test precedence: divider drag > pan > zoom (Codex finding).
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QPen, QColor, QImage, QMouseEvent, QWheelEvent,
    QCursor,
)


class SplitViewWidget(QWidget):
    """Image display with optional split view, zoom, and pan."""

    zoom_changed = Signal(float)  # current zoom level

    # Divider hit zone (pixels from divider line)
    _DIVIDER_HIT_ZONE = 8
    _DIVIDER_WIDTH = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Images
        self._left_image: QImage | None = None
        self._right_image: QImage | None = None
        self._single_image: QImage | None = None  # used when split disabled

        # Split state
        self._split_enabled = False
        self._divider_pos = 0.5  # normalized 0.0-1.0

        # Drag state
        self._dragging_divider = False

        # Zoom/pan
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_start = QPointF()
        self._pan_start_offset = QPointF()

        # Zoom limits
        self._zoom_min = 0.25
        self._zoom_max = 8.0

        # Placeholder text
        self._placeholder = "No clip selected"

    # ── Public API ──

    def set_image(self, image: QImage | None) -> None:
        """Set the single (non-split) image."""
        self._single_image = image
        self.update()

    def set_left_image(self, image: QImage | None) -> None:
        """Set the left (before/input) image for split view."""
        self._left_image = image
        self.update()

    def set_right_image(self, image: QImage | None) -> None:
        """Set the right (after/output) image for split view."""
        self._right_image = image
        self.update()

    def set_split_enabled(self, enabled: bool) -> None:
        """Toggle split view on/off."""
        self._split_enabled = enabled
        self.update()

    @property
    def split_enabled(self) -> bool:
        return self._split_enabled

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self._single_image = None
        self._left_image = None
        self._right_image = None
        self.update()

    def reset_zoom(self) -> None:
        """Reset zoom to fit and center pan."""
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.zoom_changed.emit(self._zoom)
        self.update()

    @property
    def zoom_level(self) -> float:
        return self._zoom

    # ── Paint ──

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Background
        painter.fillRect(self.rect(), QColor("#0A0A00"))

        if self._split_enabled and (self._left_image or self._right_image):
            self._paint_split(painter)
        elif self._single_image:
            self._paint_single(painter)
        else:
            self._paint_placeholder(painter)

        painter.end()

    def _paint_single(self, painter: QPainter) -> None:
        """Draw single image with zoom/pan."""
        img = self._single_image
        if img is None:
            return

        dest = self._image_rect(img)
        painter.drawImage(dest, img)

    def _paint_split(self, painter: QPainter) -> None:
        """Draw split view with left/right images and divider."""
        w = self.width()
        divider_x = int(w * self._divider_pos)

        # Left side
        if self._left_image:
            dest = self._image_rect(self._left_image)
            painter.setClipRect(0, 0, divider_x, self.height())
            painter.drawImage(dest, self._left_image)

        # Right side
        if self._right_image:
            dest = self._image_rect(self._right_image)
            painter.setClipRect(divider_x, 0, w - divider_x, self.height())
            painter.drawImage(dest, self._right_image)

        # Remove clip for divider drawing
        painter.setClipping(False)

        # Divider line
        pen = QPen(QColor("#FFF203"), self._DIVIDER_WIDTH)
        painter.setPen(pen)
        painter.drawLine(divider_x, 0, divider_x, self.height())

        # Handle triangles at top and bottom
        handle_size = 8
        painter.setBrush(QColor("#FFF203"))
        painter.setPen(Qt.NoPen)

        # Top triangle
        top_points = [
            (divider_x - handle_size, 0),
            (divider_x + handle_size, 0),
            (divider_x, handle_size),
        ]
        from PySide6.QtGui import QPolygon
        from PySide6.QtCore import QPoint
        painter.drawPolygon(QPolygon([QPoint(x, y) for x, y in top_points]))

        # Bottom triangle
        h = self.height()
        bot_points = [
            (divider_x - handle_size, h),
            (divider_x + handle_size, h),
            (divider_x, h - handle_size),
        ]
        painter.drawPolygon(QPolygon([QPoint(x, y) for x, y in bot_points]))

    def _paint_placeholder(self, painter: QPainter) -> None:
        """Draw placeholder text."""
        painter.setPen(QColor("#808070"))
        font = painter.font()
        font.setPointSize(16)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)

    def _image_rect(self, img: QImage) -> QRectF:
        """Calculate the destination rect for an image with zoom/pan."""
        iw, ih = img.width(), img.height()
        vw, vh = self.width(), self.height()

        # Fit to viewport at zoom=1.0
        scale_fit = min(vw / iw, vh / ih)
        display_w = iw * scale_fit * self._zoom
        display_h = ih * scale_fit * self._zoom

        # Center + pan offset
        x = (vw - display_w) / 2 + self._pan.x()
        y = (vh - display_h) / 2 + self._pan.y()

        return QRectF(x, y, display_w, display_h)

    # ── Mouse Events ──

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._split_enabled:
            # Check divider hit
            divider_x = self.width() * self._divider_pos
            if abs(event.position().x() - divider_x) < self._DIVIDER_HIT_ZONE:
                self._dragging_divider = True
                return

        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self._pan_start_offset = QPointF(self._pan)
            self.setCursor(Qt.ClosedHandCursor)
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging_divider:
            self._divider_pos = max(0.05, min(0.95,
                event.position().x() / self.width()))
            self.update()
            return

        if self._panning:
            delta = event.position() - self._pan_start
            self._pan = QPointF(
                self._pan_start_offset.x() + delta.x(),
                self._pan_start_offset.y() + delta.y(),
            )
            self.update()
            return

        # Cursor feedback for divider hover
        if self._split_enabled:
            divider_x = self.width() * self._divider_pos
            if abs(event.position().x() - divider_x) < self._DIVIDER_HIT_ZONE:
                self.setCursor(Qt.SplitHCursor)
            else:
                self.unsetCursor()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging_divider:
            self._dragging_divider = False
            return

        if self._panning:
            self._panning = False
            self.unsetCursor()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Double-click to reset zoom/pan."""
        if event.button() == Qt.LeftButton:
            self.reset_zoom()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Ctrl+Wheel to zoom."""
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.1 if delta > 0 else 1.0 / 1.1
            new_zoom = self._zoom * factor
            new_zoom = max(self._zoom_min, min(self._zoom_max, new_zoom))
            self._zoom = new_zoom
            self.zoom_changed.emit(self._zoom)
            self.update()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event) -> None:
        """Keyboard zoom: +/- keys, 0 to reset."""
        key = event.key()
        if key == Qt.Key_Plus or key == Qt.Key_Equal:
            self._zoom = min(self._zoom_max, self._zoom * 1.2)
            self.zoom_changed.emit(self._zoom)
            self.update()
        elif key == Qt.Key_Minus:
            self._zoom = max(self._zoom_min, self._zoom / 1.2)
            self.zoom_changed.emit(self._zoom)
            self.update()
        elif key == Qt.Key_0:
            self.reset_zoom()
        else:
            super().keyPressEvent(event)

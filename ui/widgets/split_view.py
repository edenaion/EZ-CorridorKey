"""Split view widget with draggable yellow divider, zoom, and pan.

Renders two QImages (left=before, right=after) with a split divider.
When split is disabled, renders a single image full-width.

Uses QImage as internal currency (Codex finding: QPixmap can use
platform-native GPU backing, QImage is guaranteed CPU-only).

Zoom/pan: Ctrl+wheel zooms, middle-click pans, double-click resets.
Hit-test precedence: divider drag > pan > annotation > zoom (Codex finding).

Annotation mode: hotkey 1/2 activates green/red brush. Left-click draws
strokes in image-pixel coordinates. Shift+drag resizes brush.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QPainter,
    QPen,
    QColor,
    QImage,
    QMouseEvent,
    QWheelEvent,
)

from ui.widgets.annotation_overlay import (
    AnnotationModel,
    paint_annotations,
    paint_brush_cursor,
    paint_resize_indicator,
    paint_annotation_hud,
)
from ui.widgets.wipe_controller import (
    wipe_line_endpoints,
    wipe_handle_rect,
    wipe_distance_to_line,
    paint_wipe,
    handle_wipe_press,
    handle_wipe_drag,
    wipe_cursor_for_pos,
    handle_wipe_scroll,
)


class SplitViewWidget(QWidget):
    """Image display with optional split view, zoom, and pan."""

    zoom_changed = Signal(float)  # current zoom level
    stroke_finished = Signal()  # emitted when an annotation stroke completes

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

        # Wipe mode (A/B comparison — diagonal divider, A=left, B=right)
        self._wipe_mode = False
        self._wipe_angle = -45.0  # degrees, default: bottom-left to top-right. Range -90 to 90
        self._wipe_offset = 0.0  # perpendicular offset from center (-0.5 to 0.5)
        self._wipe_dragging: str | None = None  # "handle" or "line" or None
        self._wipe_drag_start = QPointF()
        self._wipe_drag_start_offset = 0.0
        self._wipe_drag_start_angle = 45.0

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

        # Extraction progress overlay
        self._extraction_progress = 0.0  # 0.0 to 1.0
        self._extraction_total = 0

        # Annotation state
        self._annotation_mode: str | None = None  # "fg", "bg", or None
        self._annotation_model: AnnotationModel | None = None
        self._annotation_stem_idx: int = -1
        self._brush_radius: float = 15.0  # image pixels
        self._drawing: bool = False
        self._resizing_brush: bool = False
        self._resize_start_y: float = 0.0
        self._resize_start_radius: float = 0.0
        self._mouse_pos: QPointF = QPointF()  # last known mouse position (display)
        self._straight_line: bool = False  # Alt+click straight-line mode
        self._line_anchor: tuple[float, float] | None = None  # anchor in image-pixel coords

    # ── Public API ──

    @property
    def is_annotating(self) -> bool:
        """True when annotation mode is active (brush tool enabled)."""
        return self._annotation_mode is not None

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

    def set_wipe_mode(self, enabled: bool) -> None:
        """Toggle A/B wipe comparison (diagonal divider, A above/left, B below/right)."""
        self._wipe_mode = enabled
        self._wipe_angle = -45.0
        self._wipe_offset = 0.0
        self._wipe_dragging = None
        self.update()

    @property
    def wipe_mode(self) -> bool:
        return self._wipe_mode

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text

    def set_extraction_progress(self, progress: float, total: int) -> None:
        """Update extraction progress overlay. Set total=0 to clear."""
        self._extraction_progress = progress
        self._extraction_total = total
        self.update()
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

    # ── Annotation API ──

    def set_annotation_mode(self, mode: str | None) -> None:
        """Set annotation mode: 'fg', 'bg', or None to disable."""
        self._annotation_mode = mode
        if mode:
            self.setCursor(Qt.CrossCursor)
        else:
            self.unsetCursor()
        self.update()

    def set_annotation_model(self, model: AnnotationModel | None) -> None:
        self._annotation_model = model

    def set_annotation_stem_index(self, idx: int) -> None:
        self._annotation_stem_idx = idx
        self.update()

    @property
    def annotation_mode(self) -> str | None:
        return self._annotation_mode

    # ── Paint ──

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Background
        painter.fillRect(self.rect(), QColor("#0A0A00"))

        if self._wipe_mode and self._left_image and self._right_image:
            self._paint_wipe(painter)
        elif self._split_enabled and (self._left_image or self._right_image):
            self._paint_split(painter)
        elif self._single_image:
            self._paint_single(painter)
        elif self._extraction_total <= 0:
            self._paint_placeholder(painter)

        # Annotation overlay (after image, before extraction)
        if self._annotation_model is not None and self._annotation_mode:
            self._paint_annotations(painter)

        # Extraction progress overlay (replaces placeholder during extraction)
        if self._extraction_total > 0:
            self._paint_extraction_overlay(painter)

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
        h = self.height()
        divider_x = int(w * self._divider_pos)

        # Left side — fit image into left panel
        if self._left_image:
            left_vp = QRectF(0, 0, divider_x, h)
            dest = self._image_rect(self._left_image, viewport=left_vp)
            painter.setClipRect(0, 0, divider_x, h)
            painter.drawImage(dest, self._left_image)

        # Right side — fit image into right panel
        if self._right_image:
            right_vp = QRectF(divider_x, 0, w - divider_x, h)
            dest = self._image_rect(self._right_image, viewport=right_vp)
            painter.setClipRect(divider_x, 0, w - divider_x, h)
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

    # ── Wipe mode rendering (delegated to wipe_controller) ──

    def _wipe_line_endpoints(self):
        """Compute the wipe line endpoints from angle + offset."""
        return wipe_line_endpoints(self.width(), self.height(), self._wipe_angle, self._wipe_offset)

    def _wipe_handle_rect(self, center: QPointF, hit=False) -> QRectF:
        """Return the center square handle rect. hit=True returns 2x hitbox."""
        return wipe_handle_rect(center, hit)

    def _paint_wipe(self, painter: QPainter) -> None:
        """Draw A/B wipe comparison with diagonal divider."""
        paint_wipe(
            painter,
            self.width(),
            self.height(),
            self._wipe_angle,
            self._wipe_offset,
            self._left_image,
            self._right_image,
            self._image_rect,
            self._DIVIDER_WIDTH,
        )

    def _paint_placeholder(self, painter: QPainter) -> None:
        """Draw placeholder text."""
        painter.setPen(QColor("#808070"))
        font = painter.font()
        font.setPointSize(16)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)

    def _paint_extraction_overlay(self, painter: QPainter) -> None:
        """Draw extraction progress bar and percentage centered on the viewer."""
        w, h = self.width(), self.height()
        pct = int(self._extraction_progress * 100)
        current = int(self._extraction_progress * self._extraction_total)

        bar_w = int(w * 0.6)
        bar_h = 8
        bar_x = (w - bar_w) // 2

        # "Extracting frames..." header above the bar
        font = painter.font()
        font.setPointSize(14)
        painter.setFont(font)
        header_rect = QRectF(bar_x, h // 2 - 30, bar_w, 28)
        # Semi-transparent background for readability
        painter.fillRect(header_rect.adjusted(-8, -2, 8, 2), QColor(0, 0, 0, 140))
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(header_rect, Qt.AlignCenter, "Extracting frames...")

        # Progress bar below header
        bar_y = h // 2

        # Track background
        painter.fillRect(bar_x, bar_y, bar_w, bar_h, QColor(30, 29, 0))

        # Fill
        fill_w = int(bar_w * self._extraction_progress)
        if fill_w > 0:
            painter.fillRect(bar_x, bar_y, fill_w, bar_h, QColor("#FFF203"))

        # Track border
        painter.setPen(QColor("#454430"))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(bar_x, bar_y, bar_w - 1, bar_h - 1)

        # Progress text below bar
        font.setPointSize(11)
        painter.setFont(font)
        text_rect = QRectF(bar_x, bar_y + bar_h + 8, bar_w, 24)
        # Semi-transparent background for readability
        painter.fillRect(text_rect.adjusted(-8, -2, 8, 2), QColor(0, 0, 0, 140))
        painter.setPen(QColor("#FF8C00"))
        painter.drawText(
            text_rect,
            Qt.AlignCenter,
            f"{pct}%  ({current}/{self._extraction_total} frames)",
        )

    def _paint_annotations(self, painter: QPainter) -> None:
        """Draw annotation strokes and brush cursor on the viewport."""
        img = self._annotation_target_image()
        if img is None or self._annotation_model is None:
            return

        dest = self._image_rect(img, viewport=self._annotation_viewport())
        paintable_rect = self._annotation_paint_rect(img)
        if paintable_rect is None:
            return
        iw, ih = img.width(), img.height()

        # Draw existing strokes + in-progress stroke
        strokes = self._annotation_model.get_strokes(self._annotation_stem_idx)
        current = self._annotation_model.current_stroke
        painter.save()
        painter.setClipRect(paintable_rect)
        paint_annotations(painter, strokes, current, dest, iw, ih)
        painter.restore()
        paint_annotation_hud(
            painter,
            image_rect=paintable_rect,
            brush_type=self._annotation_mode,
            radius_image=self._brush_radius,
        )

        # Brush cursor at mouse position
        if self._annotation_mode and self.underMouse() and paintable_rect.contains(self._mouse_pos):
            radius_display = self._brush_radius * dest.width() / iw
            if self._resizing_brush:
                paint_resize_indicator(
                    painter,
                    self._mouse_pos,
                    radius_display,
                    self._brush_radius,
                    self._annotation_mode,
                )
            else:
                paint_brush_cursor(
                    painter,
                    self._mouse_pos,
                    radius_display,
                    self._annotation_mode,
                )

    def _display_to_image(self, display_pos: QPointF) -> tuple[float, float] | None:
        """Convert display coordinates to image-pixel coordinates.

        Returns None if the position is outside the image bounds.
        """
        img = self._annotation_target_image()
        if img is None:
            return None
        dest = self._image_rect(img, viewport=self._annotation_viewport())
        paintable_rect = self._annotation_paint_rect(img)
        if paintable_rect is None or not paintable_rect.contains(display_pos):
            return None
        iw, ih = img.width(), img.height()
        img_x = (display_pos.x() - dest.x()) * iw / dest.width()
        img_y = (display_pos.y() - dest.y()) * ih / dest.height()
        img_x = min(max(float(img_x), 0.0), float(iw - 1))
        img_y = min(max(float(img_y), 0.0), float(ih - 1))
        return (img_x, img_y)

    def _annotation_target_image(self) -> QImage | None:
        return self._single_image or self._left_image

    def _annotation_viewport(self) -> QRectF | None:
        """Return the left-panel viewport rect when in split mode, else None."""
        if self._split_enabled:
            divider_x = float(self.width()) * self._divider_pos
            return QRectF(0, 0, divider_x, float(self.height()))
        return None

    def _annotation_paint_rect(self, img: QImage) -> QRectF | None:
        rect = self._image_rect(img, viewport=self._annotation_viewport())
        if self._split_enabled:
            divider_x = float(self.width()) * self._divider_pos
            rect = rect.intersected(QRectF(0.0, 0.0, divider_x, float(self.height())))
        if rect.isEmpty():
            return None
        return rect

    def _image_rect(self, img: QImage, viewport: QRectF | None = None) -> QRectF:
        """Calculate the destination rect for an image with zoom/pan.

        Parameters
        ----------
        img : QImage
            The image to compute a destination rect for.
        viewport : QRectF, optional
            Sub-region of the widget to fit the image into (used by split view
            so each panel gets its own fit calculation). Defaults to the full
            widget area.
        """
        iw, ih = img.width(), img.height()
        if viewport is not None:
            vx, vy = viewport.x(), viewport.y()
            vw, vh = viewport.width(), viewport.height()
        else:
            vx, vy = 0.0, 0.0
            vw, vh = self.width(), self.height()

        # Fit to viewport at zoom=1.0
        scale_fit = min(vw / iw, vh / ih)
        display_w = iw * scale_fit * self._zoom
        display_h = ih * scale_fit * self._zoom

        # Center within viewport + pan offset
        x = vx + (vw - display_w) / 2 + self._pan.x()
        y = vy + (vh - display_h) / 2 + self._pan.y()

        return QRectF(x, y, display_w, display_h)

    # ── Mouse Events ──

    def _wipe_distance_to_line(self, pos: QPointF) -> float:
        """Signed perpendicular distance from pos to the wipe line (pixels)."""
        return wipe_distance_to_line(
            pos, self.width(), self.height(), self._wipe_angle, self._wipe_offset
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # Wipe mode: check handle and line hit
        if event.button() == Qt.LeftButton and self._wipe_mode:
            drag_type, drag_start, start_offset, start_angle = handle_wipe_press(
                event.position(),
                self.width(),
                self.height(),
                self._wipe_angle,
                self._wipe_offset,
                self._DIVIDER_HIT_ZONE,
            )
            if drag_type is not None:
                self._wipe_dragging = drag_type
                self._wipe_drag_start = drag_start
                self._wipe_drag_start_offset = start_offset
                self._wipe_drag_start_angle = start_angle
                return

        if event.button() == Qt.LeftButton and self._split_enabled:
            # Check divider hit
            divider_x = self.width() * self._divider_pos
            if abs(event.position().x() - divider_x) < self._DIVIDER_HIT_ZONE:
                self._dragging_divider = True
                return

        # Annotation: Shift+left-click = brush resize
        if (
            event.button() == Qt.LeftButton
            and self._annotation_mode
            and event.modifiers() & Qt.ShiftModifier
        ):
            self._resizing_brush = True
            self._resize_start_y = event.position().y()
            self._resize_start_radius = self._brush_radius
            self.update()
            return

        # Annotation: Alt+left-click = straight line
        if (
            event.button() == Qt.LeftButton
            and self._annotation_mode
            and self._annotation_model is not None
            and event.modifiers() & Qt.AltModifier
        ):
            pos = self._display_to_image(event.position())
            if pos is not None:
                self._drawing = True
                self._straight_line = True
                self._line_anchor = pos
                self._annotation_model.start_stroke(
                    self._annotation_stem_idx,
                    pos[0],
                    pos[1],
                    self._annotation_mode,
                    self._brush_radius,
                )
                self.update()
                return

        # Annotation: left-click = start freehand drawing
        if (
            event.button() == Qt.LeftButton
            and self._annotation_mode
            and self._annotation_model is not None
        ):
            pos = self._display_to_image(event.position())
            if pos is not None:
                self._drawing = True
                self._straight_line = False
                self._annotation_model.start_stroke(
                    self._annotation_stem_idx,
                    pos[0],
                    pos[1],
                    self._annotation_mode,
                    self._brush_radius,
                )
                self.update()
                return

        # Middle-click on wipe line: reset to default angle
        if event.button() == Qt.MiddleButton and self._wipe_mode:
            dist = abs(self._wipe_distance_to_line(event.position()))
            if dist < self._DIVIDER_HIT_ZONE:
                self._wipe_angle = -45.0
                self._wipe_offset = 0.0
                self.update()
                return

        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self._pan_start_offset = QPointF(self._pan)
            self.setCursor(Qt.ClosedHandCursor)
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._mouse_pos = event.position()

        # Wipe mode dragging
        if self._wipe_dragging:
            new_offset, new_angle = handle_wipe_drag(
                self._wipe_dragging,
                event.position(),
                self._wipe_drag_start,
                self._wipe_drag_start_offset,
                self._wipe_drag_start_angle,
                self.width(),
                self.height(),
                self._wipe_angle,
            )
            if self._wipe_dragging == "handle":
                self._wipe_offset = new_offset
            elif self._wipe_dragging == "line":
                self._wipe_angle = new_angle
            self.update()
            return

        # Wipe mode cursor feedback
        if self._wipe_mode and not self._panning:
            cursor = wipe_cursor_for_pos(
                event.position(),
                self.width(),
                self.height(),
                self._wipe_angle,
                self._wipe_offset,
                self._DIVIDER_HIT_ZONE,
            )
            if cursor is not None:
                self.setCursor(cursor)
            elif not self._annotation_mode:
                self.unsetCursor()

        if self._dragging_divider:
            self._divider_pos = max(0.05, min(0.95, event.position().x() / self.width()))
            self.update()
            return

        # Annotation: brush resize (Shift+drag)
        if self._resizing_brush:
            delta_y = self._resize_start_y - event.position().y()
            # 2 image-pixels per display-pixel of drag
            new_radius = self._resize_start_radius + delta_y * 2.0
            self._brush_radius = max(2.0, min(200.0, new_radius))
            self.update()
            return

        # Annotation: straight-line preview (Alt+drag)
        if self._drawing and self._straight_line and self._annotation_model is not None:
            pos = self._display_to_image(event.position())
            if pos is not None and self._line_anchor is not None:
                # Replace stroke points with anchor + current endpoint for live preview
                stroke = self._annotation_model.current_stroke
                if stroke is not None:
                    stroke.points = [self._line_anchor, pos]
                self.update()
            return

        # Annotation: freehand drawing stroke
        if self._drawing and self._annotation_model is not None:
            pos = self._display_to_image(event.position())
            if pos is not None:
                self._annotation_model.add_point(pos[0], pos[1])
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
            elif not self._annotation_mode:
                self.unsetCursor()

        # Update brush cursor position when annotation mode active
        if self._annotation_mode:
            self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._wipe_dragging:
            self._wipe_dragging = None
            return

        if self._dragging_divider:
            self._dragging_divider = False
            return

        if self._resizing_brush:
            self._resizing_brush = False
            self.update()
            return

        if self._drawing and self._annotation_model is not None:
            # Straight-line: finalize with anchor + release point
            if self._straight_line and self._line_anchor is not None:
                pos = self._display_to_image(event.position())
                if pos is not None:
                    stroke = self._annotation_model.current_stroke
                    if stroke is not None:
                        stroke.points = [self._line_anchor, pos]
                self._straight_line = False
                self._line_anchor = None
            self._annotation_model.finish_stroke()
            self._drawing = False
            self.stroke_finished.emit()
            self.update()
            return

        if self._panning:
            self._panning = False
            self.unsetCursor()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Double-click to reset zoom/pan (disabled during annotation)."""
        if event.button() == Qt.LeftButton and not self._annotation_mode:
            self.reset_zoom()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Ctrl+Wheel to zoom, plain wheel in wipe mode to slide divider."""
        mods = event.modifiers()
        # Scroll in wipe mode: slide the wipe line (up = A, down = B)
        # Shift+scroll for fine-grain control
        if self._wipe_mode and mods in (Qt.KeyboardModifier(0), Qt.ShiftModifier):
            delta = event.angleDelta().y()
            self._wipe_offset = handle_wipe_scroll(
                delta, bool(mods & Qt.ShiftModifier), self._wipe_offset
            )
            self.update()
            return
        if mods & Qt.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.1 if delta > 0 else 1.0 / 1.1
            new_zoom = self._zoom * factor
            new_zoom = max(self._zoom_min, min(self._zoom_max, new_zoom))

            # Adjust pan so the point under the cursor stays fixed
            cursor = event.position()
            vw, vh = self.width(), self.height()
            # Point relative to viewport center + current pan
            cx = cursor.x() - vw / 2 - self._pan.x()
            cy = cursor.y() - vh / 2 - self._pan.y()
            # Scale the offset by the zoom ratio
            ratio = new_zoom / self._zoom
            self._pan = QPointF(
                self._pan.x() - cx * (ratio - 1),
                self._pan.y() - cy * (ratio - 1),
            )

            self._zoom = new_zoom
            self.zoom_changed.emit(self._zoom)
            self.update()
        elif mods & Qt.ShiftModifier:
            # Shift+Wheel: horizontal pan (left/right)
            delta = event.angleDelta().y()
            self._pan = QPointF(self._pan.x() + delta * 0.5, self._pan.y())
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

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
    QPainter, QPen, QColor, QImage, QMouseEvent, QWheelEvent,
    QCursor,
)

from ui.widgets.annotation_overlay import (
    AnnotationModel, paint_annotations, paint_brush_cursor,
    paint_resize_indicator, paint_annotation_hud,
)


class SplitViewWidget(QWidget):
    """Image display with optional split view, zoom, and pan."""

    zoom_changed = Signal(float)  # current zoom level
    stroke_finished = Signal()    # emitted when an annotation stroke completes

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
        self._wipe_angle = -45.0      # degrees, default: bottom-left to top-right. Range -90 to 90
        self._wipe_offset = 0.0       # perpendicular offset from center (-0.5 to 0.5)
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
        self._straight_line: bool = False    # Alt+click straight-line mode
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

    # ── Wipe mode rendering ──

    def _wipe_line_endpoints(self):
        """Compute the wipe line endpoints from angle + offset.

        Returns (QPointF start, QPointF end, QPointF center) in widget coords.
        """
        import math
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0

        # Offset shifts the line perpendicular to its direction
        angle_rad = math.radians(self._wipe_angle)
        perp_x = -math.sin(angle_rad)
        perp_y = math.cos(angle_rad)
        diag = math.sqrt(w * w + h * h)
        offset_px = self._wipe_offset * diag

        # Center of line shifted by offset
        lx = cx + perp_x * offset_px
        ly = cy + perp_y * offset_px

        # Line direction (along the angle)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        # Extend line well beyond viewport
        ext = diag
        p1 = QPointF(lx - dx * ext, ly - dy * ext)
        p2 = QPointF(lx + dx * ext, ly + dy * ext)
        center = QPointF(lx, ly)
        return p1, p2, center

    def _wipe_handle_rect(self, center: QPointF, hit=False) -> QRectF:
        """Return the center square handle rect. hit=True returns 2x hitbox."""
        s = 12.0 if hit else 6.0
        return QRectF(center.x() - s, center.y() - s, s * 2, s * 2)

    def _paint_wipe(self, painter: QPainter) -> None:
        """Draw A/B wipe comparison with diagonal divider."""
        from PySide6.QtGui import QPolygonF, QPainterPath

        w, h = self.width(), self.height()
        p1, p2, center = self._wipe_line_endpoints()

        # Build clip path for side A (above/left of line)
        # The "above" side is determined by the perpendicular direction
        import math
        angle_rad = math.radians(self._wipe_angle)
        perp_x = -math.sin(angle_rad)
        perp_y = math.cos(angle_rad)

        # Push the line's endpoints outward perpendicular to create a polygon
        # covering side A (the "above" side = negative perpendicular direction)
        far = max(w, h) * 2
        a1 = QPointF(p1.x() - perp_x * far, p1.y() - perp_y * far)
        a2 = QPointF(p2.x() - perp_x * far, p2.y() - perp_y * far)

        side_a_poly = QPolygonF([p1, p2, a2, a1])
        side_a_path = QPainterPath()
        side_a_path.addPolygon(side_a_poly)

        # Side A: INPUT image (above/left of line)
        if self._left_image:
            dest = self._image_rect(self._left_image)
            painter.setClipPath(side_a_path)
            painter.drawImage(dest, self._left_image)

        # Side B: OUTPUT image (below/right of line) — everything NOT in side A
        if self._right_image:
            dest = self._image_rect(self._right_image)
            full = QPainterPath()
            full.addRect(QRectF(0, 0, w, h))
            side_b_path = full - side_a_path
            painter.setClipPath(side_b_path)
            painter.drawImage(dest, self._right_image)

        # Remove clip for divider drawing
        painter.setClipping(False)

        # Draw wipe line
        pen = QPen(QColor("#FFF203"), self._DIVIDER_WIDTH)
        painter.setPen(pen)
        painter.drawLine(p1, p2)

        # Draw center handle (filled square)
        handle = self._wipe_handle_rect(center)
        painter.setBrush(QColor("#FFF203"))
        painter.setPen(Qt.NoPen)
        painter.drawRect(handle)

        # Draw A/B labels near the line
        painter.setPen(QColor("#FFF203"))
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)

        label_offset = 16
        painter.drawText(
            QPointF(center.x() - perp_x * label_offset - 4,
                    center.y() - perp_y * label_offset + 4), "A")
        painter.drawText(
            QPointF(center.x() + perp_x * label_offset - 4,
                    center.y() + perp_y * label_offset + 4), "B")

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
            text_rect, Qt.AlignCenter,
            f"{pct}%  ({current}/{self._extraction_total} frames)",
        )

    def _paint_annotations(self, painter: QPainter) -> None:
        """Draw annotation strokes and brush cursor on the viewport."""
        img = self._annotation_target_image()
        if img is None or self._annotation_model is None:
            return

        dest = self._image_rect(img)
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
                    painter, self._mouse_pos, radius_display,
                    self._brush_radius, self._annotation_mode,
                )
            else:
                paint_brush_cursor(
                    painter, self._mouse_pos, radius_display,
                    self._annotation_mode,
                )

    def _display_to_image(self, display_pos: QPointF) -> tuple[float, float] | None:
        """Convert display coordinates to image-pixel coordinates.

        Returns None if the position is outside the image bounds.
        """
        img = self._annotation_target_image()
        if img is None:
            return None
        dest = self._image_rect(img)
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

    def _annotation_paint_rect(self, img: QImage) -> QRectF | None:
        rect = self._image_rect(img)
        if self._split_enabled:
            divider_x = float(self.width()) * self._divider_pos
            rect = rect.intersected(QRectF(0.0, 0.0, divider_x, float(self.height())))
        if rect.isEmpty():
            return None
        return rect

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

    def _wipe_distance_to_line(self, pos: QPointF) -> float:
        """Signed perpendicular distance from pos to the wipe line (pixels)."""
        import math
        angle_rad = math.radians(self._wipe_angle)
        _, _, center = self._wipe_line_endpoints()
        # Normal vector (perpendicular to line direction)
        nx = -math.sin(angle_rad)
        ny = math.cos(angle_rad)
        return (pos.x() - center.x()) * nx + (pos.y() - center.y()) * ny

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # Wipe mode: check handle and line hit
        if event.button() == Qt.LeftButton and self._wipe_mode:
            _, _, center = self._wipe_line_endpoints()
            if self._wipe_handle_rect(center, hit=True).contains(event.position()):
                self._wipe_dragging = "handle"
                self._wipe_drag_start = event.position()
                self._wipe_drag_start_offset = self._wipe_offset
                return
            dist = abs(self._wipe_distance_to_line(event.position()))
            if dist < self._DIVIDER_HIT_ZONE:
                self._wipe_dragging = "line"
                self._wipe_drag_start = event.position()
                self._wipe_drag_start_angle = self._wipe_angle
                return

        if event.button() == Qt.LeftButton and self._split_enabled:
            # Check divider hit
            divider_x = self.width() * self._divider_pos
            if abs(event.position().x() - divider_x) < self._DIVIDER_HIT_ZONE:
                self._dragging_divider = True
                return

        # Annotation: Shift+left-click = brush resize
        if (event.button() == Qt.LeftButton
                and self._annotation_mode
                and event.modifiers() & Qt.ShiftModifier):
            self._resizing_brush = True
            self._resize_start_y = event.position().y()
            self._resize_start_radius = self._brush_radius
            self.update()
            return

        # Annotation: Alt+left-click = straight line
        if (event.button() == Qt.LeftButton
                and self._annotation_mode
                and self._annotation_model is not None
                and event.modifiers() & Qt.AltModifier):
            pos = self._display_to_image(event.position())
            if pos is not None:
                self._drawing = True
                self._straight_line = True
                self._line_anchor = pos
                self._annotation_model.start_stroke(
                    self._annotation_stem_idx,
                    pos[0], pos[1],
                    self._annotation_mode,
                    self._brush_radius,
                )
                self.update()
                return

        # Annotation: left-click = start freehand drawing
        if (event.button() == Qt.LeftButton
                and self._annotation_mode
                and self._annotation_model is not None):
            pos = self._display_to_image(event.position())
            if pos is not None:
                self._drawing = True
                self._straight_line = False
                self._annotation_model.start_stroke(
                    self._annotation_stem_idx,
                    pos[0], pos[1],
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
        if self._wipe_dragging == "handle":
            # Translate line: move offset based on perpendicular mouse movement
            import math
            angle_rad = math.radians(self._wipe_angle)
            nx = -math.sin(angle_rad)
            ny = math.cos(angle_rad)
            dx = event.position().x() - self._wipe_drag_start.x()
            dy = event.position().y() - self._wipe_drag_start.y()
            diag = math.sqrt(self.width() ** 2 + self.height() ** 2)
            delta_offset = (dx * nx + dy * ny) / diag
            self._wipe_offset = max(-0.5, min(0.5,
                self._wipe_drag_start_offset + delta_offset))
            self.update()
            return
        if self._wipe_dragging == "line":
            # Rotate line: compute angle from mouse position relative to viewport center
            import math
            w, h = self.width(), self.height()
            cx, cy = w / 2.0, h / 2.0
            mx = event.position().x() - cx
            my = event.position().y() - cy
            # Angle of the line direction (tangent), not the normal
            angle = math.degrees(math.atan2(my, mx))
            # Wrap into -90..90 so A (left image) always stays on the left side.
            # atan2 returns -180..180; if past ±90, wrap to the nearest boundary.
            if angle > 90.0:
                angle = 90.0
            elif angle < -90.0:
                angle = -90.0
            self._wipe_angle = angle
            self.update()
            return

        # Wipe mode cursor feedback
        if self._wipe_mode and not self._panning:
            _, _, center = self._wipe_line_endpoints()
            if self._wipe_handle_rect(center, hit=True).contains(event.position()):
                self.setCursor(Qt.SizeAllCursor)
            elif abs(self._wipe_distance_to_line(event.position())) < self._DIVIDER_HIT_ZONE:
                self.setCursor(Qt.OpenHandCursor)
            elif not self._annotation_mode:
                self.unsetCursor()

        if self._dragging_divider:
            self._divider_pos = max(0.05, min(0.95,
                event.position().x() / self.width()))
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
            if mods & Qt.ShiftModifier:
                # Fine-grain: proportional to scroll delta for butter-smooth control
                self._wipe_offset = max(-0.5, min(0.5, self._wipe_offset - delta / 15000.0))
            else:
                # Normal: fixed steps per notch
                step = 0.03 if delta > 0 else -0.03
                self._wipe_offset = max(-0.5, min(0.5, self._wipe_offset + step))
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

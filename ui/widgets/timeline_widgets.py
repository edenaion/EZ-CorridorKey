"""Timeline helper widgets: slider, coverage bar, and marker overlay.

Extracted from frame_scrubber.py to reduce file size.
These are self-contained QWidget subclasses used by FrameScrubber.

CoverageBar: multi-lane bar showing alpha, inference, and annotation coverage.
MarkerOverlay: transparent overlay for in/out bracket markers with drag handles.
_FatSlider: QSlider with enlarged clickable groove area.
"""
from __future__ import annotations

from PySide6.QtWidgets import QSlider, QWidget, QStyle, QStyleOptionSlider
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QPainter, QColor, QPolygonF


class _FatSlider(QSlider):
    """QSlider with an enlarged clickable groove area.

    Overrides the style sub-control rect so the groove (and its click target)
    extends to fill the full widget height, making it much easier to click.
    """

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setFixedHeight(22)  # taller than default ~16px

    def _groove_rect(self):
        """Return the expanded groove rect used for hit-testing."""
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        base = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
        )
        # Expand groove to full widget height
        base.setTop(0)
        base.setBottom(self.height())
        return base

    def mousePressEvent(self, event):
        """Jump slider to click position anywhere in the expanded groove."""
        if event.button() == Qt.LeftButton:
            groove = self._groove_rect()
            if groove.width() > 0:
                ratio = (event.position().x() - groove.x()) / groove.width()
                ratio = max(0.0, min(1.0, ratio))
                value = self.minimum() + round(ratio * (self.maximum() - self.minimum()))
                self.setValue(value)
                event.accept()
                return
        super().mousePressEvent(event)


class CoverageBar(QWidget):
    """Thin multi-lane bar showing alpha, inference, and annotation frame coverage.

    Top lane: annotation markers (green dots for annotated frames).
    Middle lane: alpha hint coverage (white segments).
    Bottom lane: inference output coverage (brand yellow segments).
    """

    _ALPHA_COLOR = QColor(200, 200, 200)       # Soft white for alpha
    _INFERENCE_COLOR = QColor(255, 242, 3)      # Brand yellow #FFF203
    _ANNOTATION_COLOR = QColor(44, 195, 80)     # Green #2CC350 for annotations
    _TRACK_COLOR = QColor(26, 25, 0)            # Dark track
    _LANE_HEIGHT = 3
    _GAP = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._alpha: list[bool] = []
        self._inference: list[bool] = []
        self._annotated: list[bool] = []
        self._update_height()

    def _update_height(self) -> None:
        """Recalculate height based on whether annotation lane is visible."""
        lanes = 2  # alpha + inference always present
        if self._annotated:
            lanes = 3
        self.setFixedHeight(self._LANE_HEIGHT * lanes + self._GAP * (lanes - 1))

    def set_coverage(self, alpha: list[bool], inference: list[bool]) -> None:
        self._alpha = alpha
        self._inference = inference
        self.update()

    def set_annotation_markers(self, annotated: list[bool]) -> None:
        self._annotated = annotated
        self._update_height()
        self.update()

    def clear(self) -> None:
        self._alpha = []
        self._inference = []
        self._annotated = []
        self._update_height()
        self.update()

    def paintEvent(self, event) -> None:
        if not self._alpha and not self._inference and not self._annotated:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        w = self.width()
        y = 0

        # Draw annotation lane (top, only if annotations exist)
        if self._annotated:
            self._paint_lane(painter, self._annotated, y, w, self._ANNOTATION_COLOR)
            y += self._LANE_HEIGHT + self._GAP

        # Draw alpha lane
        self._paint_lane(painter, self._alpha, y, w, self._ALPHA_COLOR)
        y += self._LANE_HEIGHT + self._GAP

        # Draw inference lane
        self._paint_lane(painter, self._inference, y, w, self._INFERENCE_COLOR)

        painter.end()

    def _paint_lane(
        self,
        painter: QPainter,
        coverage: list[bool],
        y: int,
        total_width: int,
        fill_color: QColor,
    ) -> None:
        n = len(coverage)
        if n == 0:
            return

        # Draw track background
        painter.fillRect(0, y, total_width, self._LANE_HEIGHT, self._TRACK_COLOR)

        # Draw filled segments — batch contiguous runs for efficiency
        i = 0
        while i < n:
            if not coverage[i]:
                i += 1
                continue
            # Find contiguous run of True values
            run_start = i
            while i < n and coverage[i]:
                i += 1
            run_end = i
            # Map frame range to pixel range
            x0 = int(run_start * total_width / n)
            x1 = int(run_end * total_width / n)
            painter.fillRect(x0, y, max(1, x1 - x0), self._LANE_HEIGHT, fill_color)


class MarkerOverlay(QWidget):
    """Transparent overlay that paints in/out markers with draggable handles.

    Parented to the center column widget so it shares the exact same
    horizontal bounds as the coverage bar and slider.

    Mouse-transparent by default (WA_TransparentForMouseEvents=True) so the
    slider beneath works normally. Only becomes interactive when hovering
    near a marker handle, then reverts when the mouse moves away.
    """

    in_point_dragged = Signal(int)
    out_point_dragged = Signal(int)
    scrub_to_frame = Signal(int)  # request scrubber jump during drag

    GRAB_WIDTH = 8    # px hitbox for drag detection
    HANDLE_W = 6      # px visible handle width
    HANDLE_H = 8      # px handle nub height at bottom
    MARKER_WIDTH = 2  # px bracket line width
    MARKER_COLOR = QColor(255, 242, 3)      # Brand yellow
    DIM_COLOR = QColor(0, 0, 0, 120)        # Semi-transparent dim

    def __init__(self, parent=None):
        super().__init__(parent)
        self._in_point: int | None = None
        self._out_point: int | None = None
        self._total: int = 0
        self._dragging: str | None = None  # None | 'in' | 'out'
        # Start transparent — slider gets all events by default
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def set_in_out(self, in_point: int | None, out_point: int | None, total: int) -> None:
        self._in_point = in_point
        self._out_point = out_point
        self._total = total
        self.update()

    def clear(self) -> None:
        self._in_point = None
        self._out_point = None
        self.update()

    def _frame_to_x(self, frame: int) -> int:
        if self._total <= 0:
            return 0
        return int(frame * self.width() / self._total)

    def _x_to_frame(self, x: int) -> int:
        if self._total <= 0 or self.width() <= 0:
            return 0
        return max(0, min(self._total - 1, int(x * self._total / self.width())))

    def _marker_x(self, which: str) -> int | None:
        """Return pixel x for in or out marker, or None if not set."""
        if self._in_point is None or self._out_point is None or self._total <= 0:
            return None
        if which == 'in':
            return self._frame_to_x(self._in_point)
        else:
            return self._frame_to_x(self._out_point + 1)

    def _hit_marker(self, x: int) -> str | None:
        """Return 'in', 'out', or None based on proximity to markers."""
        x_in = self._marker_x('in')
        x_out = self._marker_x('out')
        if x_in is not None and abs(x - x_in) <= self.GRAB_WIDTH:
            return 'in'
        if x_out is not None and abs(x - x_out) <= self.GRAB_WIDTH:
            return 'out'
        return None

    def paintEvent(self, event) -> None:
        if self._in_point is None or self._out_point is None or self._total <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w = self.width()
        h = self.height()
        n = self._total

        x_in = int(self._in_point * w / n)
        x_out = int((self._out_point + 1) * w / n)

        # Dim region before in-point
        if self._in_point > 0:
            painter.fillRect(0, 0, x_in, h, self.DIM_COLOR)

        # Dim region after out-point
        if self._out_point < n - 1:
            painter.fillRect(x_out, 0, w - x_out, h, self.DIM_COLOR)

        # In-point bracket line
        painter.fillRect(x_in, 0, self.MARKER_WIDTH, h, self.MARKER_COLOR)

        # Out-point bracket line
        painter.fillRect(max(0, x_out - self.MARKER_WIDTH), 0,
                         self.MARKER_WIDTH, h, self.MARKER_COLOR)

        # Handle nubs at bottom (small triangles pointing inward)
        painter.setBrush(self.MARKER_COLOR)
        painter.setPen(Qt.NoPen)

        # In-point handle: triangle pointing right at bottom-left of bracket
        in_tri = QPolygonF([
            QPointF(x_in, h),
            QPointF(x_in + self.HANDLE_W, h),
            QPointF(x_in, h - self.HANDLE_H),
        ])
        painter.drawPolygon(in_tri)

        # Out-point handle: triangle pointing left at bottom-right of bracket
        ox = max(0, x_out - self.MARKER_WIDTH)
        out_tri = QPolygonF([
            QPointF(ox + self.MARKER_WIDTH, h),
            QPointF(ox + self.MARKER_WIDTH - self.HANDLE_W, h),
            QPointF(ox + self.MARKER_WIDTH, h - self.HANDLE_H),
        ])
        painter.drawPolygon(out_tri)

        painter.end()

    def _become_interactive(self) -> None:
        """Grab mouse events (near a marker handle)."""
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)
        self.setCursor(Qt.SizeHorCursor)

    def _become_transparent(self) -> None:
        """Release mouse events back to the slider."""
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.unsetCursor()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            hit = self._hit_marker(int(event.position().x()))
            if hit:
                self._dragging = hit
                event.accept()
                return
        elif event.button() == Qt.MiddleButton:
            # Middle-click on a marker resets it to boundary
            hit = self._hit_marker(int(event.position().x()))
            if hit == 'in':
                self._in_point = 0
                self.update()
                self.in_point_dragged.emit(0)
                event.accept()
                return
            elif hit == 'out' and self._total > 0:
                self._out_point = self._total - 1
                self.update()
                self.out_point_dragged.emit(self._total - 1)
                event.accept()
                return
        # Missed — go transparent so slider gets future events
        self._become_transparent()
        event.ignore()

    def mouseMoveEvent(self, event) -> None:
        x = int(event.position().x())
        if self._dragging:
            frame = self._x_to_frame(x)
            if self._dragging == 'in':
                if self._out_point is not None:
                    frame = min(frame, self._out_point)
                self._in_point = frame
            else:
                if self._in_point is not None:
                    frame = max(frame, self._in_point)
                self._out_point = frame
            self.update()
            # Scrub the playhead so user sees current position
            self.scrub_to_frame.emit(frame)
            event.accept()
        else:
            # If mouse drifted away from marker, go transparent
            hit = self._hit_marker(x)
            if not hit:
                self._become_transparent()
            event.ignore()

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging:
            if self._dragging == 'in' and self._in_point is not None:
                self.in_point_dragged.emit(self._in_point)
            elif self._dragging == 'out' and self._out_point is not None:
                self.out_point_dragged.emit(self._out_point)
            self._dragging = None
            # Check if still near a marker
            hit = self._hit_marker(int(event.position().x()))
            if not hit:
                self._become_transparent()
            event.accept()
        else:
            event.ignore()

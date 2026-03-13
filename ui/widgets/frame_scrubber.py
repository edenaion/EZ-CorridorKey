"""Frame scrubber timeline with transport controls, coverage overlay, and debounced frame loading.

Horizontal slider + frame counter + step buttons. Emits frame_changed only
after a debounce period (50ms) to coalesce rapid scrubbing events.

CoverageBar sits above the slider (same column) showing up to three lanes:
  - Annotation lane (green): which frames have brush annotations (only shown when present)
  - Alpha lane (white): which frames have AlphaHint
  - Inference lane (brand yellow): which frames have output

MarkerOverlay draws in/out bracket lines with draggable handles,
positioned exactly over the coverage bar + slider column.
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QPushButton, QToolTip, QStyleOptionSlider, QStyle
from PySide6.QtCore import Qt, Signal, QTimer, QPointF, QEvent, QSettings
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


class FrameScrubber(QWidget):
    """Frame navigation scrubber with transport controls, coverage bar, and debounced output."""

    frame_changed = Signal(int)      # emitted after debounce, stem index
    in_point_changed = Signal(int)   # in-point set at stem index
    out_point_changed = Signal(int)  # out-point set at stem index
    range_cleared = Signal()         # in/out range cleared

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(48)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 0, 8, 0)
        row.setSpacing(4)

        # Frame counter label (left)
        self._frame_label = QLabel("0 / 0")
        self._frame_label.setFixedWidth(90)
        self._frame_label.setAlignment(Qt.AlignCenter)
        self._frame_label.setStyleSheet("color: #808070; font-size: 11px;")
        row.addWidget(self._frame_label)

        # Transport controls
        btn_style = (
            "QPushButton { background: transparent; color: #808070; border: none; "
            "font-size: 14px; padding: 0 4px; font-family: sans-serif; }"
            "QPushButton:hover { color: #E0E0E0; }"
            "QPushButton:disabled { color: #3A3A30; }"
        )

        # Jump to start
        self._start_btn = QPushButton("\u25C0\u25C0")  # ◀◀
        self._start_btn.setFixedWidth(28)
        self._start_btn.setStyleSheet(btn_style)
        self._start_btn.setToolTip("Go to first frame")
        self._start_btn.clicked.connect(self._go_start)
        row.addWidget(self._start_btn)

        # Step back
        self._prev_btn = QPushButton("\u25C0")  # ◀
        self._prev_btn.setFixedWidth(24)
        self._prev_btn.setStyleSheet(btn_style)
        self._prev_btn.setToolTip("Previous frame")
        self._prev_btn.clicked.connect(self._step_back)
        row.addWidget(self._prev_btn)

        # Play/Pause toggle
        self._play_btn = QPushButton("\u25B6")  # ▶ (play icon)
        self._play_btn.setFixedWidth(24)
        self._play_btn.setStyleSheet(btn_style)
        self._play_btn.setToolTip("Play / Pause (Space)")
        self._play_btn.clicked.connect(self.toggle_playback)
        row.addWidget(self._play_btn)

        # ── Center column: coverage bar + slider ──
        self._center = QWidget()
        center_layout = QVBoxLayout(self._center)
        center_layout.setContentsMargins(0, 2, 0, 0)
        center_layout.setSpacing(0)

        # Coverage bar (top of center column — same width as slider)
        self._coverage_bar = CoverageBar()
        self._coverage_bar.setToolTip(
            "Coverage bar — shows which frames have been processed.\n"
            "Green lane: painted frames (brush strokes).\n"
            "White lane: alpha hint coverage.\n"
            "Yellow lane: inference output coverage."
        )
        center_layout.addWidget(self._coverage_bar)

        # Slider (bottom of center column) — tall hitbox for easy clicking
        self._slider = _FatSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setEnabled(False)
        self._slider.setToolTip("Scrub through frames. Scroll wheel or Left/Right to step.")
        self._slider.valueChanged.connect(self._on_slider_changed)
        center_layout.addWidget(self._slider)

        row.addWidget(self._center, 1)

        # Marker overlay — parented to center, sits on top
        self._marker_overlay = MarkerOverlay(self._center)
        self._marker_overlay.in_point_dragged.connect(self._on_in_dragged)
        self._marker_overlay.out_point_dragged.connect(self._on_out_dragged)
        self._marker_overlay.scrub_to_frame.connect(self._on_marker_scrub)
        self._marker_overlay.raise_()

        # Track mouse over slider/overlay to detect marker proximity + tooltips
        self._slider.setMouseTracking(True)
        self._slider.installEventFilter(self)
        self._marker_overlay.installEventFilter(self)
        self._center.installEventFilter(self)

        # Step forward
        self._next_btn = QPushButton("\u25B6")  # ▶
        self._next_btn.setFixedWidth(24)
        self._next_btn.setStyleSheet(btn_style)
        self._next_btn.setToolTip("Next frame")
        self._next_btn.clicked.connect(self._step_forward)
        row.addWidget(self._next_btn)

        # Jump to end
        self._end_btn = QPushButton("\u25B6\u25B6")  # ▶▶
        self._end_btn.setFixedWidth(28)
        self._end_btn.setStyleSheet(btn_style)
        self._end_btn.setToolTip("Go to last frame")
        self._end_btn.clicked.connect(self._go_end)
        row.addWidget(self._end_btn)

        # Debounce timer (50ms, Codex recommendation)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(50)
        self._debounce.timeout.connect(self._emit_frame)

        # Playback timer — slow stepping (PNG decode is too heavy for realtime)
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(333)  # ~3 frames per second
        self._playback_timer.timeout.connect(self._playback_tick)
        self._playing = False

        self._total = 0
        self._suppress_signal = False
        self._in_point: int | None = None
        self._out_point: int | None = None

    def resizeEvent(self, event) -> None:
        """Keep marker overlay sized to the center column."""
        super().resizeEvent(event)
        self._marker_overlay.setGeometry(0, 0, self._center.width(), self._center.height())
        self._marker_overlay.raise_()

    def eventFilter(self, obj, event) -> bool:
        """Watch slider for mouse moves (marker proximity) and tooltip forwarding."""
        if obj is self._slider:
            if event.type() == QEvent.MouseMove:
                x = int(event.position().x())
                hit = self._marker_overlay._hit_marker(x)
                if hit and not self._marker_overlay._dragging:
                    self._marker_overlay._become_interactive()
        # Forward tooltip events from overlay or center to correct child
        if event.type() == QEvent.ToolTip and obj in (
            self._marker_overlay, self._center,
        ):
            local_y = int(event.pos().y())
            coverage_h = self._coverage_bar.height()
            if local_y < coverage_h:
                tip = self._coverage_bar.toolTip()
            else:
                tip = self._slider.toolTip()
            if tip:
                QToolTip.showText(event.globalPos(), tip, obj)
                return True
        return super().eventFilter(obj, event)

    def set_range(self, total_frames: int) -> None:
        """Configure scrubber for a clip with total_frames stems."""
        if self._playing:
            self._stop_playback()
        self._total = total_frames
        enabled = total_frames > 0
        self._slider.setEnabled(enabled)
        self._slider.setMaximum(max(0, total_frames - 1))
        self._slider.setTickInterval(max(1, total_frames // 20))
        self._start_btn.setEnabled(enabled)
        self._prev_btn.setEnabled(enabled)
        self._play_btn.setEnabled(enabled)
        self._next_btn.setEnabled(enabled)
        self._end_btn.setEnabled(enabled)
        self._update_label()

    def set_frame(self, index: int) -> None:
        """Set current frame without emitting signal (external update)."""
        self._suppress_signal = True
        self._slider.setValue(index)
        self._suppress_signal = False
        self._update_label()

    def set_coverage(self, alpha: list[bool], inference: list[bool]) -> None:
        """Update the coverage overlay lanes."""
        self._coverage_bar.set_coverage(alpha, inference)

    def set_annotation_markers(self, annotated: list[bool]) -> None:
        """Update the annotation marker lane on the coverage bar."""
        self._coverage_bar.set_annotation_markers(annotated)

    def current_frame(self) -> int:
        return self._slider.value()

    # ── In/Out Markers ──

    def set_in_point(self, index: int | None) -> None:
        """Set in-point at stem index. None to clear just the in-point."""
        self._in_point = index
        self._sync_markers()
        if index is not None:
            self.in_point_changed.emit(index)

    def set_out_point(self, index: int | None) -> None:
        """Set out-point at stem index. None to clear just the out-point."""
        self._out_point = index
        self._sync_markers()
        if index is not None:
            self.out_point_changed.emit(index)

    def set_in_out(self, in_point: int | None, out_point: int | None) -> None:
        """Set both markers at once (for restore from session)."""
        self._in_point = in_point
        self._out_point = out_point
        self._sync_markers()

    def clear_in_out(self) -> None:
        """Clear both markers."""
        self._in_point = None
        self._out_point = None
        self._sync_markers()
        self.range_cleared.emit()

    def get_in_out(self) -> tuple[int | None, int | None]:
        """Return current (in_point, out_point)."""
        return self._in_point, self._out_point

    @property
    def has_range(self) -> bool:
        return self._in_point is not None and self._out_point is not None

    def _sync_markers(self) -> None:
        """Push current in/out state to marker overlay."""
        self._marker_overlay.set_in_out(self._in_point, self._out_point, self._total)

    def _on_in_dragged(self, frame: int) -> None:
        """Handle in-point drag from overlay."""
        self._in_point = frame
        self.in_point_changed.emit(frame)

    def _on_out_dragged(self, frame: int) -> None:
        """Handle out-point drag from overlay."""
        self._out_point = frame
        self.out_point_changed.emit(frame)

    def _on_marker_scrub(self, frame: int) -> None:
        """While dragging a marker, move the playhead and debounce the frame load."""
        self._suppress_signal = True
        self._slider.setValue(frame)
        self._suppress_signal = False
        self._update_label()
        # Use the same debounce as normal scrubbing (50ms) to avoid flooding
        self._debounce.start()

    def _on_slider_changed(self, value: int) -> None:
        self._update_label()
        if not self._suppress_signal:
            # Restart debounce timer (latest request wins)
            self._debounce.start()

    def _emit_frame(self) -> None:
        self.frame_changed.emit(self._slider.value())

    def _update_label(self) -> None:
        current = self._slider.value() + 1 if self._total > 0 else 0
        self._frame_label.setText(f"{current} / {self._total}")

    # Transport controls
    def _go_start(self) -> None:
        self._slider.setValue(0)

    def _step_back(self) -> None:
        self._slider.setValue(max(0, self._slider.value() - 1))

    def _step_forward(self) -> None:
        self._slider.setValue(min(self._slider.maximum(), self._slider.value() + 1))

    def _go_end(self) -> None:
        self._slider.setValue(self._slider.maximum())

    # ── Playback ──

    def toggle_playback(self) -> None:
        """Toggle play/pause state."""
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        if self._total <= 1:
            return
        self._playing = True
        self._play_btn.setText("\u275A\u275A")  # ❚❚ (pause icon)
        self._play_btn.setToolTip("Pause (Space)")
        self._playback_timer.start()

    def _stop_playback(self) -> None:
        self._playing = False
        self._playback_timer.stop()
        self._play_btn.setText("\u25B6")  # ▶ (play icon)
        self._play_btn.setToolTip("Play (Space)")

    def _playback_tick(self) -> None:
        """Advance one frame during playback. Loops within in/out range if set."""
        loop_enabled = QSettings().value("playback/loop", True, type=bool)

        current = self._slider.value()

        # Determine playback bounds
        if self._in_point is not None and self._out_point is not None:
            lo, hi = self._in_point, self._out_point
        else:
            lo, hi = 0, self._slider.maximum()

        next_frame = current + 1
        if next_frame > hi:
            if loop_enabled:
                next_frame = lo
            else:
                self._stop_playback()
                return

        # Suppress debounce during playback — emit frame_changed directly
        self._suppress_signal = True
        self._slider.setValue(next_frame)
        self._suppress_signal = False
        self.frame_changed.emit(next_frame)

    def wheelEvent(self, event) -> None:
        """Scroll wheel scrubs frames: scroll up = forward, scroll down = back.

        Steps by 2% of total frames per tick (minimum 1 frame).
        """
        if self._total <= 0:
            return
        step = max(1, self._total * 2 // 100)
        delta = event.angleDelta().y()
        if delta > 0:
            self._slider.setValue(min(self._slider.maximum(), self._slider.value() + step))
        elif delta < 0:
            self._slider.setValue(max(0, self._slider.value() - step))
        event.accept()

    def keyPressEvent(self, event) -> None:
        """Space for play/pause, Left/Right for stepping, I/O for markers."""
        if event.key() == Qt.Key_Space:
            self.toggle_playback()
        elif event.key() == Qt.Key_Left:
            self._step_back()
        elif event.key() == Qt.Key_Right:
            self._step_forward()
        elif event.key() == Qt.Key_Home:
            self._go_start()
        elif event.key() == Qt.Key_End:
            self._go_end()
        elif event.key() == Qt.Key_I:
            self.set_in_point(self._slider.value())
        elif event.key() == Qt.Key_O:
            self.set_out_point(self._slider.value())
        else:
            super().keyPressEvent(event)

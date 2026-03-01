"""Frame scrubber timeline with transport controls, coverage overlay, and debounced frame loading.

Horizontal slider + frame counter + step buttons. Emits frame_changed only
after a debounce period (50ms) to coalesce rapid scrubbing events.

CoverageBar sits above the slider showing two lanes:
  - Alpha lane (white): which frames have AlphaHint
  - Inference lane (brand yellow): which frames have output
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QPushButton
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPainter, QColor


class CoverageBar(QWidget):
    """Thin dual-lane bar showing alpha and inference frame coverage.

    Top lane: alpha hint coverage (white segments).
    Bottom lane: inference output coverage (brand yellow segments).
    """

    _ALPHA_COLOR = QColor(200, 200, 200)       # Soft white for alpha
    _INFERENCE_COLOR = QColor(255, 242, 3)      # Brand yellow #FFF203
    _TRACK_COLOR = QColor(26, 25, 0)            # Dark track
    _MARKER_COLOR = QColor(255, 242, 3)         # Brand yellow for in/out brackets
    _DIM_COLOR = QColor(0, 0, 0, 120)           # Semi-transparent dim overlay
    _MARKER_WIDTH = 2
    _LANE_HEIGHT = 3
    _GAP = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self._LANE_HEIGHT * 2 + self._GAP)
        self._alpha: list[bool] = []
        self._inference: list[bool] = []
        self._in_point: int | None = None
        self._out_point: int | None = None
        self._num_frames: int = 0

    def set_coverage(self, alpha: list[bool], inference: list[bool]) -> None:
        self._alpha = alpha
        self._inference = inference
        self.update()

    def set_in_out(self, in_point: int | None, out_point: int | None, num_frames: int) -> None:
        self._in_point = in_point
        self._out_point = out_point
        self._num_frames = num_frames
        self.update()

    def clear_in_out(self) -> None:
        self._in_point = None
        self._out_point = None
        self.update()

    def clear(self) -> None:
        self._alpha = []
        self._inference = []
        self._in_point = None
        self._out_point = None
        self._num_frames = 0
        self.update()

    def paintEvent(self, event) -> None:
        if not self._alpha and not self._inference:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        w = self.width()

        # Draw alpha lane (top)
        self._paint_lane(painter, self._alpha, 0, w, self._ALPHA_COLOR)

        # Draw inference lane (bottom)
        y_offset = self._LANE_HEIGHT + self._GAP
        self._paint_lane(painter, self._inference, y_offset, w, self._INFERENCE_COLOR)

        # Draw in/out range overlay (dim excluded regions, draw bracket lines)
        if (self._in_point is not None and self._out_point is not None
                and self._num_frames > 0):
            n = self._num_frames
            total_h = self.height()

            # Dim region before in-point
            if self._in_point > 0:
                x_in = int(self._in_point * w / n)
                painter.fillRect(0, 0, x_in, total_h, self._DIM_COLOR)

            # Dim region after out-point
            if self._out_point < n - 1:
                x_out = int((self._out_point + 1) * w / n)
                painter.fillRect(x_out, 0, w - x_out, total_h, self._DIM_COLOR)

            # In-point bracket line
            x_in = int(self._in_point * w / n)
            painter.fillRect(x_in, 0, self._MARKER_WIDTH, total_h, self._MARKER_COLOR)

            # Out-point bracket line
            x_out = int((self._out_point + 1) * w / n)
            painter.fillRect(
                max(0, x_out - self._MARKER_WIDTH), 0,
                self._MARKER_WIDTH, total_h, self._MARKER_COLOR,
            )

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


class FrameScrubber(QWidget):
    """Frame navigation scrubber with transport controls, coverage bar, and debounced output."""

    frame_changed = Signal(int)      # emitted after debounce, stem index
    in_point_changed = Signal(int)   # in-point set at stem index
    out_point_changed = Signal(int)  # out-point set at stem index
    range_cleared = Signal()         # in/out range cleared

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Coverage bar (top)
        self._coverage_bar = CoverageBar()
        main_layout.addWidget(self._coverage_bar)

        # Transport + slider row (bottom)
        transport = QHBoxLayout()
        transport.setContentsMargins(8, 2, 8, 2)
        transport.setSpacing(4)

        # Frame counter label (left)
        self._frame_label = QLabel("0 / 0")
        self._frame_label.setFixedWidth(90)
        self._frame_label.setAlignment(Qt.AlignCenter)
        self._frame_label.setStyleSheet("color: #808070; font-size: 11px;")
        transport.addWidget(self._frame_label)

        # Transport controls
        btn_style = (
            "QPushButton { background: transparent; color: #808070; border: none; "
            "font-size: 14px; padding: 0 4px; font-family: sans-serif; }"
            "QPushButton:hover { color: #E0E0E0; }"
            "QPushButton:disabled { color: #3A3A30; }"
        )

        # Jump to start — text glyphs (not emoji) so CSS color is respected
        self._start_btn = QPushButton("\u25C0\u25C0")  # ◀◀
        self._start_btn.setFixedWidth(28)
        self._start_btn.setStyleSheet(btn_style)
        self._start_btn.setToolTip("Go to first frame")
        self._start_btn.clicked.connect(self._go_start)
        transport.addWidget(self._start_btn)

        # Step back
        self._prev_btn = QPushButton("\u25C0")  # ◀
        self._prev_btn.setFixedWidth(24)
        self._prev_btn.setStyleSheet(btn_style)
        self._prev_btn.setToolTip("Previous frame")
        self._prev_btn.clicked.connect(self._step_back)
        transport.addWidget(self._prev_btn)

        # Slider
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider_changed)
        transport.addWidget(self._slider, 1)

        # Step forward
        self._next_btn = QPushButton("\u25B6")  # ▶
        self._next_btn.setFixedWidth(24)
        self._next_btn.setStyleSheet(btn_style)
        self._next_btn.setToolTip("Next frame")
        self._next_btn.clicked.connect(self._step_forward)
        transport.addWidget(self._next_btn)

        # Jump to end
        self._end_btn = QPushButton("\u25B6\u25B6")  # ▶▶
        self._end_btn.setFixedWidth(28)
        self._end_btn.setStyleSheet(btn_style)
        self._end_btn.setToolTip("Go to last frame")
        self._end_btn.clicked.connect(self._go_end)
        transport.addWidget(self._end_btn)

        # Total frames label (right)
        self._total_label = QLabel("")
        self._total_label.setFixedWidth(60)
        self._total_label.setStyleSheet("color: #808070; font-size: 10px;")
        transport.addWidget(self._total_label)

        main_layout.addLayout(transport)

        # Debounce timer (50ms, Codex recommendation)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(50)
        self._debounce.timeout.connect(self._emit_frame)

        self._total = 0
        self._suppress_signal = False
        self._in_point: int | None = None
        self._out_point: int | None = None

    def set_range(self, total_frames: int) -> None:
        """Configure scrubber for a clip with total_frames stems."""
        self._total = total_frames
        enabled = total_frames > 0
        self._slider.setEnabled(enabled)
        self._slider.setMaximum(max(0, total_frames - 1))
        self._slider.setTickInterval(max(1, total_frames // 20))
        self._start_btn.setEnabled(enabled)
        self._prev_btn.setEnabled(enabled)
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
        """Push current in/out state to coverage bar."""
        self._coverage_bar.set_in_out(self._in_point, self._out_point, self._total)

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

    def keyPressEvent(self, event) -> None:
        """Left/Right arrows for stepping, I/O for in/out markers."""
        if event.key() == Qt.Key_Left:
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

"""Frame scrubber timeline with transport controls and debounced frame loading.

Horizontal slider + frame counter + step buttons. Emits frame_changed only
after a debounce period (50ms) to coalesce rapid scrubbing events.
Codex finding: undebounced scrubber will stutter on long clips.
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSlider, QPushButton
from PySide6.QtCore import Qt, Signal, QTimer


class FrameScrubber(QWidget):
    """Frame navigation scrubber with transport controls and debounced output."""

    frame_changed = Signal(int)  # emitted after debounce, stem index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(4)

        # Frame counter label (left)
        self._frame_label = QLabel("0 / 0")
        self._frame_label.setFixedWidth(90)
        self._frame_label.setAlignment(Qt.AlignCenter)
        self._frame_label.setStyleSheet("color: #808070; font-size: 11px;")
        layout.addWidget(self._frame_label)

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
        layout.addWidget(self._start_btn)

        # Step back
        self._prev_btn = QPushButton("\u25C0")  # ◀
        self._prev_btn.setFixedWidth(24)
        self._prev_btn.setStyleSheet(btn_style)
        self._prev_btn.setToolTip("Previous frame")
        self._prev_btn.clicked.connect(self._step_back)
        layout.addWidget(self._prev_btn)

        # Slider
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, 1)

        # Step forward
        self._next_btn = QPushButton("\u25B6")  # ▶
        self._next_btn.setFixedWidth(24)
        self._next_btn.setStyleSheet(btn_style)
        self._next_btn.setToolTip("Next frame")
        self._next_btn.clicked.connect(self._step_forward)
        layout.addWidget(self._next_btn)

        # Jump to end
        self._end_btn = QPushButton("\u25B6\u25B6")  # ▶▶
        self._end_btn.setFixedWidth(28)
        self._end_btn.setStyleSheet(btn_style)
        self._end_btn.setToolTip("Go to last frame")
        self._end_btn.clicked.connect(self._go_end)
        layout.addWidget(self._end_btn)

        # Total frames label (right)
        self._total_label = QLabel("")
        self._total_label.setFixedWidth(60)
        self._total_label.setStyleSheet("color: #808070; font-size: 10px;")
        layout.addWidget(self._total_label)

        # Debounce timer (50ms, Codex recommendation)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(50)
        self._debounce.timeout.connect(self._emit_frame)

        self._total = 0
        self._suppress_signal = False

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

    def current_frame(self) -> int:
        return self._slider.value()

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
        """Left/Right arrows for single-frame stepping."""
        if event.key() == Qt.Key_Left:
            self._step_back()
        elif event.key() == Qt.Key_Right:
            self._step_forward()
        elif event.key() == Qt.Key_Home:
            self._go_start()
        elif event.key() == Qt.Key_End:
            self._go_end()
        else:
            super().keyPressEvent(event)

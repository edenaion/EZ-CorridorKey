"""Bottom status bar — progress, timer, and action buttons.

Layout (left to right):
- Progress bar + frame counter + elapsed/ETA timer (left, fills)
- Warning count
- [RUN INFERENCE] / [STOP] button (right, primary CTA)

GPU/VRAM info is displayed in the top brand bar (see main_window.py).
"""
from __future__ import annotations

import time

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton,
    QProgressBar,
)
from PySide6.QtCore import Qt, Signal, QTimer


def _fmt_duration(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    s = int(seconds)
    if s < 3600:
        return f"{s // 60}:{s % 60:02d}"
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


class StatusBar(QWidget):
    """Bottom bar with progress, elapsed timer, and run/stop CTA."""

    run_clicked = Signal()
    stop_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setStyleSheet("background-color: #0E0D00; border-top: 1px solid #2A2910;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(12)

        # Progress bar (left, fills)
        self._progress = QProgressBar()
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress, 1)

        # Frame counter + timer
        self._frame_label = QLabel("")
        self._frame_label.setStyleSheet("color: #999980; font-size: 11px;")
        self._frame_label.setMinimumWidth(220)
        layout.addWidget(self._frame_label)

        # Warning count
        self._warn_label = QLabel("")
        self._warn_label.setStyleSheet("color: #FFA500; font-size: 10px;")
        layout.addWidget(self._warn_label)

        # Run / Stop button (right, primary CTA like Topaz Export)
        self._run_btn = QPushButton("RUN INFERENCE")
        self._run_btn.setObjectName("runButton")
        self._run_btn.setFixedWidth(160)
        self._run_btn.clicked.connect(self.run_clicked.emit)
        layout.addWidget(self._run_btn)

        self._stop_btn = QPushButton("STOP")
        self._stop_btn.setObjectName("stopButton")
        self._stop_btn.setFixedWidth(80)
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        self._stop_btn.hide()
        layout.addWidget(self._stop_btn)

        self._warning_count = 0

        # Timer state
        self._job_start: float = 0.0
        self._is_indeterminate = False
        self._last_current = 0
        self._last_total = 0
        self._job_label: str = ""

        # 1-second tick timer for elapsed display
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._on_tick)

    def set_running(self, running: bool) -> None:
        """Toggle between run and stop state."""
        self._run_btn.setVisible(not running)
        self._stop_btn.setVisible(running)

    def start_job_timer(self, label: str = "", indeterminate: bool = False) -> None:
        """Start the elapsed timer for a new job.

        Args:
            label: Job description (e.g. "GVM Auto", "Inference").
            indeterminate: If True, show pulsing progress (no percentage).
        """
        self._job_start = time.monotonic()
        self._is_indeterminate = indeterminate
        self._last_current = 0
        self._last_total = 0
        self._job_label = label

        if indeterminate:
            self._progress.setRange(0, 0)  # Pulsing/indeterminate mode
            self._frame_label.setText(f"{label}  0:00")
        else:
            self._progress.setRange(0, 100)
            self._progress.setValue(0)

        self._tick_timer.start()

    def stop_job_timer(self) -> None:
        """Stop the elapsed timer."""
        self._tick_timer.stop()
        if self._is_indeterminate:
            self._progress.setRange(0, 100)  # Back to determinate
        self._is_indeterminate = False

    def update_progress(self, current: int, total: int) -> None:
        """Update progress bar, frame counter, and ETA."""
        self._last_current = current
        self._last_total = total

        if self._is_indeterminate:
            # GVM just reports 0/1 start and 1/1 end — stay indeterminate
            return

        elapsed = time.monotonic() - self._job_start if self._job_start > 0 else 0

        if total > 0:
            pct = int(current / total * 100)
            self._progress.setValue(pct)

            # ETA calculation
            eta_str = ""
            if current > 0 and current < total:
                rate = elapsed / current
                remaining = rate * (total - current)
                eta_str = f"  ETA {_fmt_duration(remaining)}"

            elapsed_str = _fmt_duration(elapsed)
            label = f"{self._job_label}  " if self._job_label else ""
            self._frame_label.setText(
                f"{label}{current}/{total}  {pct}%  {elapsed_str}{eta_str}"
            )
        else:
            self._progress.setValue(0)
            self._frame_label.setText("")

    def reset_progress(self) -> None:
        """Clear progress display and stop timer."""
        self.stop_job_timer()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._frame_label.setText("")
        self._warning_count = 0
        self._warn_label.setText("")
        self._job_start = 0.0

    def add_warning(self) -> None:
        """Increment warning counter."""
        self._warning_count += 1
        self._warn_label.setText(f"{self._warning_count} warnings")

    def set_run_enabled(self, enabled: bool) -> None:
        """Enable or disable the run button."""
        self._run_btn.setEnabled(enabled)

    def set_message(self, text: str) -> None:
        """Show a status message in the frame label area."""
        self._frame_label.setText(text)

    def _on_tick(self) -> None:
        """Called every second to update the elapsed display."""
        elapsed = time.monotonic() - self._job_start if self._job_start > 0 else 0
        elapsed_str = _fmt_duration(elapsed)

        if self._is_indeterminate:
            label = f"{self._job_label}  " if self._job_label else ""
            self._frame_label.setText(f"{label}{elapsed_str}")
        elif self._last_total > 0:
            # Re-render with updated elapsed (ETA recalculates too)
            self.update_progress(self._last_current, self._last_total)

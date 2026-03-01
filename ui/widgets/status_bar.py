"""Bottom status bar — progress, VRAM meter, GPU info, and action buttons.

Contains:
- [RUN INFERENCE] / [STOP] button
- Progress bar with frame counter
- VRAM usage meter
- GPU name badge
- Warning/error count indicator
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton,
    QProgressBar,
)
from PySide6.QtCore import Qt, Signal


class StatusBar(QWidget):
    """Bottom bar with run/stop, progress, and GPU monitoring."""

    run_clicked = Signal()
    stop_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setStyleSheet("background-color: #0E0D00; border-top: 1px solid #2A2910;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(12)

        # Run / Stop button
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

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress, 1)

        # Frame counter
        self._frame_label = QLabel("")
        self._frame_label.setStyleSheet("color: #999980; font-size: 11px;")
        self._frame_label.setMinimumWidth(100)
        layout.addWidget(self._frame_label)

        # Separator
        sep = QLabel("|")
        sep.setStyleSheet("color: #2A2910;")
        layout.addWidget(sep)

        # VRAM meter
        vram_layout = QHBoxLayout()
        vram_layout.setSpacing(6)

        self._vram_label = QLabel("VRAM")
        self._vram_label.setStyleSheet("color: #808070; font-size: 10px;")
        vram_layout.addWidget(self._vram_label)

        self._vram_bar = QProgressBar()
        self._vram_bar.setObjectName("vramMeter")
        self._vram_bar.setFixedWidth(80)
        self._vram_bar.setFixedHeight(8)
        self._vram_bar.setTextVisible(False)
        self._vram_bar.setRange(0, 100)
        self._vram_bar.setValue(0)
        vram_layout.addWidget(self._vram_bar)

        self._vram_text = QLabel("")
        self._vram_text.setStyleSheet("color: #999980; font-size: 10px;")
        self._vram_text.setMinimumWidth(70)
        vram_layout.addWidget(self._vram_text)

        layout.addLayout(vram_layout)

        # GPU name
        self._gpu_label = QLabel("")
        self._gpu_label.setStyleSheet("color: #808070; font-size: 10px;")
        layout.addWidget(self._gpu_label)

        # Warning count
        self._warn_label = QLabel("")
        self._warn_label.setStyleSheet("color: #FFA500; font-size: 10px;")
        layout.addWidget(self._warn_label)

        self._warning_count = 0

    def set_running(self, running: bool) -> None:
        """Toggle between run and stop state."""
        self._run_btn.setVisible(not running)
        self._stop_btn.setVisible(running)

    def update_progress(self, current: int, total: int) -> None:
        """Update progress bar and frame counter."""
        if total > 0:
            pct = int(current / total * 100)
            self._progress.setValue(pct)
            self._frame_label.setText(f"{current}/{total}  {pct}%")
        else:
            self._progress.setValue(0)
            self._frame_label.setText("")

    def reset_progress(self) -> None:
        """Clear progress display."""
        self._progress.setValue(0)
        self._frame_label.setText("")
        self._warning_count = 0
        self._warn_label.setText("")

    def update_vram(self, info: dict) -> None:
        """Update VRAM meter from GPUMonitor signal."""
        if not info.get("available"):
            self._vram_text.setText("No GPU")
            self._vram_bar.setValue(0)
            return

        pct = info.get("usage_pct", 0)
        used = info.get("used_gb", 0)
        total = info.get("total_gb", 0)
        self._vram_bar.setValue(int(pct))
        self._vram_text.setText(f"{used:.1f}/{total:.1f}GB")

    def set_gpu_name(self, name: str) -> None:
        """Display GPU name badge."""
        # Shorten common names
        short = name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        self._gpu_label.setText(short)

    def add_warning(self) -> None:
        """Increment warning counter."""
        self._warning_count += 1
        self._warn_label.setText(f"{self._warning_count} warnings")

    def set_run_enabled(self, enabled: bool) -> None:
        """Enable or disable the run button."""
        self._run_btn.setEnabled(enabled)

"""Bottom status bar — progress, timer, and action buttons.

Layout (left to right):
- Progress bar (compact) + frame counter + elapsed/ETA timer
- Stretch (pushes buttons right)
- [RUN INFERENCE] or [RUN SELECTED] / [RESUME] / [STOP] buttons

RESUME button appears only when partial outputs exist.
RUN text changes to "RUN SELECTED" when in/out range is set.

GPU/VRAM info is displayed in the top brand bar (see main_window.py).
"""

from __future__ import annotations

import textwrap
import time

from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QDialog,
    QVBoxLayout,
    QTextEdit,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QTextCursor


def _fmt_duration(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    s = int(seconds)
    if s < 3600:
        return f"{s // 60}:{s % 60:02d}"
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _fmt_bytes(num_bytes: int) -> str:
    """Format bytes in a compact human-readable form."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def _wrap_tooltip_text(text: str, width: int = 90) -> str:
    """Wrap tooltip text to a readable width without truncating content."""
    wrapped_lines: list[str] = []
    for line in text.splitlines() or [""]:
        if not line:
            wrapped_lines.append("")
            continue
        wrapped_lines.append(
            textwrap.fill(
                line,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(wrapped_lines)


class StatusBar(QWidget):
    """Bottom bar with progress, elapsed timer, and run/resume/stop CTAs."""

    run_clicked = Signal()
    extract_clicked = Signal()
    resume_clicked = Signal()
    stop_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setObjectName("statusBar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(8)

        # Progress bar (compact, left)
        self._progress = QProgressBar()
        self._progress.setFixedHeight(6)
        self._progress.setMaximumWidth(250)
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setToolTip("Inference progress for the current job")
        layout.addWidget(self._progress)

        # Frame counter + timer
        self._frame_label = QLabel("")
        self._frame_label.setStyleSheet("color: #999980; font-size: 11px;")
        self._frame_label.setMinimumWidth(220)
        layout.addWidget(self._frame_label)

        # Warning count (clickable)
        self._warn_btn = QPushButton("")
        self._warn_btn.setObjectName("warningButton")
        self._warn_btn.setStyleSheet(
            "QPushButton#warningButton { color: #FFA500; font-size: 10px; "
            "background: transparent; border: none; padding: 2px 6px; }"
            "QPushButton#warningButton:hover { background: #2A2910; border-radius: 3px; }"
        )
        self._warn_btn.setCursor(Qt.PointingHandCursor)
        self._warn_btn.clicked.connect(self._show_warnings_dialog)
        self._warn_btn.hide()
        layout.addWidget(self._warn_btn)

        # Stretch pushes buttons to the right
        layout.addStretch(1)

        # Run button (primary CTA)
        self._run_btn = QPushButton("RUN INFERENCE")
        self._run_btn.setObjectName("runButton")
        self._run_btn.setFixedWidth(160)
        self._run_btn.setFixedHeight(32)
        self._run_btn.setEnabled(False)
        self._run_btn.setToolTip(
            "Run AI keying on the selected clip (Ctrl+R).\n"
            "Requires a READY or COMPLETE clip with alpha hints.\n"
            "Respects in/out range if set (I/O hotkeys)."
        )
        self._run_mode = "inference"  # "inference" or "extraction"
        self._run_btn.clicked.connect(self._on_run_clicked)
        layout.addWidget(self._run_btn)

        # Divider between run and resume
        self._btn_divider = QWidget()
        self._btn_divider.setFixedWidth(1)
        self._btn_divider.setFixedHeight(24)
        self._btn_divider.setStyleSheet("background-color: #2A2910;")
        self._btn_divider.hide()
        layout.addWidget(self._btn_divider)

        # Resume button (secondary — only shown when partial outputs exist)
        self._resume_btn = QPushButton("RESUME")
        self._resume_btn.setObjectName("resumeButton")
        self._resume_btn.setFixedWidth(100)
        self._resume_btn.setFixedHeight(32)
        self._resume_btn.setToolTip(
            "Resume inference — skip already-processed frames,\n"
            "fill in remaining gaps across the full clip."
        )
        self._resume_btn.clicked.connect(self.resume_clicked.emit)
        self._resume_btn.hide()
        layout.addWidget(self._resume_btn)

        # Stop button (replaces run+resume during jobs)
        self._stop_btn = QPushButton("STOP")
        self._stop_btn.setObjectName("stopButton")
        self._stop_btn.setFixedWidth(80)
        self._stop_btn.setFixedHeight(32)
        self._stop_btn.setToolTip(
            "Stop the current job (Escape).\nAlready-processed frames are kept on disk."
        )
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        self._stop_btn.hide()
        layout.addWidget(self._stop_btn)

        self._warning_count = 0
        self._warnings: list[str] = []

        # Timer state
        self._job_start: float = 0.0
        self._last_current = 0
        self._last_total = 0
        self._last_fps: float = 0.0
        self._job_label: str = ""

        # 1-second tick timer for elapsed display
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._on_tick)

    def set_running(self, running: bool) -> None:
        """Toggle between run/resume and stop state."""
        self._run_btn.setVisible(not running)
        # Divider and resume visibility managed by update_button_state;
        # just hide them during a running job
        if running:
            self._btn_divider.hide()
            self._resume_btn.hide()
        self._stop_btn.setVisible(running)
        if running:
            self.set_stop_button_mode(force=False)

    def set_stop_button_mode(self, force: bool) -> None:
        """Switch STOP between cooperative cancel and hard-stop wording."""
        if force:
            self._stop_btn.setText("FORCE STOP")
            self._stop_btn.setFixedWidth(120)
            self._stop_btn.setToolTip(
                "The current GPU step is blocked.\n"
                "Force Stop will relaunch the app to break the stuck job."
            )
        else:
            self._stop_btn.setText("STOP")
            self._stop_btn.setFixedWidth(80)
            self._stop_btn.setToolTip(
                "Stop the current job (Escape).\nAlready-processed frames are kept on disk."
            )

    def _on_run_clicked(self) -> None:
        """Route run button click to the appropriate signal."""
        if self._run_mode == "extraction":
            self.extract_clicked.emit()
        else:
            self.run_clicked.emit()

    def update_button_state(
        self,
        can_run: bool,
        has_partial: bool,
        has_in_out: bool,
        batch_count: int = 0,
        needs_extraction: bool = False,
        needs_pipeline: bool = False,
    ) -> None:
        """Update run/resume button visibility and text based on clip state.

        Args:
            can_run: Whether the clip is in a runnable state (READY/COMPLETE).
            has_partial: Whether partial inference outputs exist on disk.
            has_in_out: Whether in/out markers are set on the scrubber.
            batch_count: Number of clips selected (>1 = batch mode).
            needs_extraction: Whether the clip needs frame extraction first.
            needs_pipeline: At least one selected clip needs alpha generation.
        """
        if needs_extraction:
            self._run_btn.setText("RUN EXTRACTION")
            self._run_btn.setEnabled(True)
            self._run_mode = "extraction"
        elif batch_count > 1 and needs_pipeline:
            self._run_btn.setText("RUN PIPELINE")
            self._run_btn.setEnabled(True)
            self._run_mode = "inference"
        elif batch_count > 1:
            self._run_btn.setText(f"RUN {batch_count} CLIPS")
            self._run_btn.setEnabled(True)
            self._run_mode = "inference"
        elif has_in_out:
            self._run_btn.setText("RUN SELECTED")
            self._run_btn.setEnabled(can_run)
            self._run_mode = "inference"
        else:
            self._run_btn.setText("RUN INFERENCE")
            self._run_btn.setEnabled(can_run)
            self._run_mode = "inference"

        # Show resume only when partial outputs exist, single clip, and can run
        show_resume = can_run and has_partial and batch_count <= 1 and not needs_extraction
        self._resume_btn.setVisible(show_resume)
        self._btn_divider.setVisible(show_resume)

    def start_job_timer(self, label: str = "") -> None:
        """Start the elapsed timer for a new job.

        Args:
            label: Job description (e.g. "GVM Auto", "Inference", "Extracting").
        """
        self._job_start = time.monotonic()
        self._last_current = 0
        self._last_total = 0
        self._last_fps = 0.0
        self._job_label = label
        self._phase = ""  # phase text shown during loading (e.g. "Loading frames...")

        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._frame_label.setText(f"{label}  0:00" if label else "0:00")

        self._tick_timer.start()

    def stop_job_timer(self) -> None:
        """Stop the elapsed timer."""
        self._tick_timer.stop()

    def update_progress(self, current: int, total: int, fps: float = 0.0) -> None:
        """Update progress bar, frame counter, FPS, and ETA."""
        self._last_current = current
        self._last_total = total
        if fps > 0:
            self._last_fps = fps
        # Clear phase text once real frame progress flows
        if total > 0 and current > 0:
            self._phase = ""

        elapsed = time.monotonic() - self._job_start if self._job_start > 0 else 0

        if total > 0:
            pct = int(current / total * 100)
            self._progress.setValue(pct)

            fps_str = ""
            eta_str = ""
            if current > 0 and current < total:
                if fps > 0:
                    fps_str = f"  {fps:.2f} fps"
                    remaining = (total - current) / fps
                else:
                    rate = elapsed / current
                    remaining = rate * (total - current)
                eta_str = f"  ETA {_fmt_duration(remaining)}"

            elapsed_str = _fmt_duration(elapsed)
            label = f"{self._job_label}  " if self._job_label else ""
            if total >= 1_000_000:
                current_text = _fmt_bytes(current)
                total_text = _fmt_bytes(total)
            else:
                current_text = str(current)
                total_text = str(total)
            self._frame_label.setText(
                f"{label}{current_text}/{total_text}  {pct}%{fps_str}  {elapsed_str}{eta_str}"
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
        self._phase = ""
        self._warning_count = 0
        self._warnings.clear()
        self._warn_btn.setText("")
        self._warn_btn.setToolTip("")
        self._warn_btn.hide()
        self._job_start = 0.0

    def add_warning(self, message: str = "") -> None:
        """Add a warning message and update the counter."""
        self._warning_count += 1
        if message:
            self._warnings.append(message)
        label = f"{self._warning_count} warning{'s' if self._warning_count != 1 else ''}"
        self._warn_btn.setText(label)
        self._warn_btn.show()
        if self._warnings:
            latest = _wrap_tooltip_text(self._warnings[-1])
            self._warn_btn.setToolTip(f"Latest:\n{latest}\n\nClick for all warnings")

    def set_message(self, text: str) -> None:
        """Show a status message in the frame label area."""
        self._frame_label.setText(text)

    def set_phase(self, phase: str) -> None:
        """Set loading-phase text shown during timer-only mode.

        The phase text (e.g. "Loading frames...") appears between the job
        label and the elapsed timer, and is automatically cleared once real
        frame progress data arrives via update_progress().
        """
        self._phase = phase

    def _on_tick(self) -> None:
        """Called every second to update the elapsed display."""
        elapsed = time.monotonic() - self._job_start if self._job_start > 0 else 0
        elapsed_str = _fmt_duration(elapsed)
        label = f"{self._job_label}  " if self._job_label else ""
        phase = getattr(self, "_phase", "")

        if self._last_total > 0 and self._last_current > 0:
            # Real frame progress — show frame counter + ETA + FPS
            self.update_progress(self._last_current, self._last_total, self._last_fps)
        elif phase:
            # Loading phase — show phase text + elapsed timer
            self._frame_label.setText(f"{label}{phase}  {elapsed_str}")
        elif self._job_start > 0:
            # Timer-only mode (no progress, no phase)
            self._frame_label.setText(f"{label}{elapsed_str}")

    def _show_warnings_dialog(self) -> None:
        """Show a modal dialog with all collected warnings."""
        if not self._warnings:
            return

        dlg = self._build_warnings_dialog()
        dlg.raise_()
        dlg.activateWindow()
        dlg.exec()

    def _dialog_parent(self) -> QWidget:
        """Parent dialogs to the main window so they show in a visible place."""
        window = self.window()
        if isinstance(window, QWidget) and window is not self:
            return window
        return self

    def _build_warnings_dialog(self) -> QDialog:
        """Build the warnings dialog."""
        dlg = QDialog(self._dialog_parent())
        dlg.setWindowTitle(f"Warnings ({self._warning_count})")
        dlg.setModal(True)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumSize(500, 300)
        dlg.setStyleSheet(
            "QDialog { background: #1A1900; }"
            "QTextEdit { background: #0E0D00; color: #CCCCAA; border: 1px solid #2A2910; "
            "font-family: 'Consolas', monospace; font-size: 11px; }"
        )

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)

        text = QTextEdit()
        text.setReadOnly(True)
        for i, msg in enumerate(self._warnings, 1):
            text.append(f"[{i}] {msg}\n")
        text.moveCursor(QTextCursor.Start)
        layout.addWidget(text, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        return dlg

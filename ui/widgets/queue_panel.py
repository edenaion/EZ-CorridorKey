"""Collapsible queue panel showing per-job progress.

The header bar ("QUEUE") is always pinned at its position. When expanded,
the job list appears ABOVE the header — the header itself never moves.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, Signal, QEvent

from backend.job_queue import GPUJobQueue, GPUJob, JobStatus, JobType


# Status display config
_STATUS_COLORS = {
    JobStatus.QUEUED: "#808070",
    JobStatus.RUNNING: "#FFF203",
    JobStatus.COMPLETED: "#22C55E",
    JobStatus.CANCELLED: "#808070",
    JobStatus.FAILED: "#D10000",
}

_STATUS_TEXT = {
    JobStatus.QUEUED: "STARTING...",
    JobStatus.RUNNING: "PROCESSING",
    JobStatus.COMPLETED: "DONE",
    JobStatus.CANCELLED: "CANCELLED",
    JobStatus.FAILED: "FAILED",
}

_JOB_TYPE_LABELS = {
    JobType.INFERENCE: "Inference",
    JobType.GVM_ALPHA: "GVM Auto",
    JobType.VIDEOMAMA_ALPHA: "VideoMaMa",
    JobType.PREVIEW_REPROCESS: "Preview",
}

_HEADER_H = 26
_BODY_MAX_H = 160


class QueuePanel(QWidget):
    """Fixed header bar with a popup job list that expands upward."""

    cancel_job_requested = Signal(str)  # job_id

    def __init__(self, queue: GPUJobQueue, parent=None):
        super().__init__(parent)
        self._queue = queue
        self.setFixedHeight(_HEADER_H)
        self.setStyleSheet(
            "QueuePanel { background-color: #0E0D00; border-top: 1px solid #2A2910; }"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)

        # "QUEUE ▶" button — entire text + caret is one clickable button
        self._header_btn = QPushButton("QUEUE \u25B6")  # ▶ collapsed
        self._header_btn.setCursor(Qt.PointingHandCursor)
        self._header_btn.setStyleSheet(
            "QPushButton { color: #CCCCAA; background: transparent; border: none; "
            "font-size: 11px; font-weight: 700; letter-spacing: 1px; padding: 0; }"
            "QPushButton:hover { color: #FFF203; }"
        )
        self._header_btn.setToolTip("Toggle queue panel (Q)")
        self._header_btn.clicked.connect(self.toggle_collapsed)
        layout.addWidget(self._header_btn)

        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #808070; font-size: 10px;")
        layout.addWidget(self._count_label)

        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.setFixedWidth(64)
        self._clear_btn.setFixedHeight(20)
        self._clear_btn.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 2px 8px; "
            "background: #1A1900; border: 1px solid #2A2910; color: #999980; }"
            "QPushButton:hover { color: #CCCCAA; border-color: #454430; }"
        )
        self._clear_btn.setToolTip("Clear completed and cancelled jobs from the list")
        self._clear_btn.clicked.connect(self._on_clear)
        self._clear_btn.hide()
        layout.addWidget(self._clear_btn)

        layout.addStretch()

        # Body — popup that appears ABOVE the header
        # Parented to same parent as us (set in reposition())
        self._body = QWidget(parent)
        self._body.setStyleSheet(
            "background-color: #0E0D00; border: 1px solid #2A2910; "
            "border-bottom: none;"
        )
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(8, 4, 8, 4)
        body_layout.setSpacing(2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        scroll.setMaximumHeight(_BODY_MAX_H)

        self._job_container = QWidget()
        self._job_layout = QVBoxLayout(self._job_container)
        self._job_layout.setContentsMargins(0, 0, 0, 0)
        self._job_layout.setSpacing(2)
        self._job_layout.addStretch()

        scroll.setWidget(self._job_container)
        body_layout.addWidget(scroll, 1)

        self._body.hide()
        self._collapsed = True

        # Hover/click sound on header button
        self._header_btn.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self._header_btn:
            from ui.sounds.audio_manager import UIAudio
            if event.type() == QEvent.Enter:
                UIAudio.hover(key="btn:queueHeader")
            elif event.type() == QEvent.MouseButtonPress:
                UIAudio.click()
        return super().eventFilter(obj, event)

    def toggle_collapsed(self) -> None:
        """Toggle the job list popup above the header."""
        self._collapsed = not self._collapsed
        self._body.setVisible(not self._collapsed)
        self._clear_btn.setVisible(not self._collapsed)
        if self._collapsed:
            self._header_btn.setText("QUEUE \u25B6")  # ▶
            self._header_btn.setToolTip("Expand queue panel (Q)")
        else:
            self._header_btn.setText("QUEUE \u25BC")  # ▼
            self._header_btn.setToolTip("Collapse queue panel (Q)")
        self.reposition()

    def reposition(self) -> None:
        """Position the body popup directly above the header bar."""
        if not self._body.isVisible():
            return
        # Body should be same width as header, positioned directly above it
        my_geo = self.geometry()
        body_h = min(self._body.sizeHint().height(), _BODY_MAX_H + 12)
        self._body.setFixedWidth(my_geo.width())
        self._body.move(my_geo.x(), my_geo.y() - body_h)
        self._body.raise_()

    def refresh(self) -> None:
        """Rebuild the job list from queue snapshot."""
        # Clear existing widgets (except the stretch at the end)
        while self._job_layout.count() > 1:
            item = self._job_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        jobs = self._queue.all_jobs_snapshot
        count = len(jobs)
        self._count_label.setText(f"{count} job{'s' if count != 1 else ''}" if count else "")

        for job in jobs:
            row = self._create_job_row(job)
            # Insert before the stretch
            self._job_layout.insertWidget(self._job_layout.count() - 1, row)

        if not self._collapsed:
            self.reposition()

    def _create_job_row(self, job: GPUJob) -> QFrame:
        """Create a single job row widget."""
        row = QFrame()
        row.setStyleSheet("background-color: #1A1900; padding: 2px 6px;")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

        # Job type label
        type_text = _JOB_TYPE_LABELS.get(job.job_type, "???")
        type_label = QLabel(type_text)
        type_label.setFixedWidth(70)
        type_label.setStyleSheet("color: #999980; font-size: 10px; font-weight: 700;")
        layout.addWidget(type_label)

        # Clip name
        name_label = QLabel(job.clip_name)
        name_label.setStyleSheet("font-size: 11px;")
        name_label.setMinimumWidth(100)
        layout.addWidget(name_label)

        # Status / Progress
        color = _STATUS_COLORS.get(job.status, "#808070")
        status_text = _STATUS_TEXT.get(job.status, "?")

        if job.status == JobStatus.RUNNING:
            if job.total_frames > 0:
                pct = int(job.current_frame / job.total_frames * 100)
                progress = QProgressBar()
                progress.setFixedHeight(6)
                progress.setFixedWidth(100)
                progress.setTextVisible(False)
                progress.setRange(0, 100)
                progress.setValue(pct)
                layout.addWidget(progress)

                frame_label = QLabel(f"{job.current_frame}/{job.total_frames}")
                frame_label.setStyleSheet("color: #999980; font-size: 10px;")
                layout.addWidget(frame_label)
            else:
                # Indeterminate (GVM monolithic call)
                progress = QProgressBar()
                progress.setFixedHeight(6)
                progress.setFixedWidth(100)
                progress.setTextVisible(False)
                progress.setRange(0, 0)  # indeterminate
                layout.addWidget(progress)

                stage_label = QLabel("Processing...")
                stage_label.setStyleSheet(f"color: {color}; font-size: 10px;")
                layout.addWidget(stage_label)
        elif job.status == JobStatus.QUEUED:
            progress = QProgressBar()
            progress.setFixedHeight(6)
            progress.setFixedWidth(60)
            progress.setTextVisible(False)
            progress.setRange(0, 0)  # indeterminate
            layout.addWidget(progress)

            status_label = QLabel(status_text)
            status_label.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 600;")
            layout.addWidget(status_label)
        elif job.status == JobStatus.CANCELLED:
            status_label = QLabel(status_text)
            status_label.setStyleSheet(
                f"color: {color}; font-size: 10px; text-decoration: line-through;"
            )
            layout.addWidget(status_label)
        else:
            status_label = QLabel(status_text)
            status_label.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 600;")
            layout.addWidget(status_label)

        layout.addStretch()

        # Dismiss button — only on finished jobs (completed/cancelled/failed)
        if job.status in (JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.FAILED):
            dismiss_btn = QPushButton("\u25C0")  # ◀
            dismiss_btn.setFixedSize(18, 18)
            dismiss_btn.setCursor(Qt.PointingHandCursor)
            dismiss_btn.setStyleSheet(
                "QPushButton { background: transparent; color: #555540; "
                "font-size: 9px; border: none; padding: 0; }"
                "QPushButton:hover { color: #999980; }"
            )
            dismiss_btn.setToolTip("Dismiss from list")
            job_id = job.id
            dismiss_btn.clicked.connect(
                lambda checked, jid=job_id: self._dismiss_job(jid)
            )
            layout.addWidget(dismiss_btn)

        return row

    def _dismiss_job(self, job_id: str) -> None:
        """Remove a single finished job from history and refresh."""
        self._queue.remove_job(job_id)
        self.refresh()

    def _on_clear(self) -> None:
        """Clear job history."""
        self._queue.clear_history()
        self.refresh()

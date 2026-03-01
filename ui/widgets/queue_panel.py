"""Collapsible queue panel showing per-job progress.

Model-driven (Codex recommendation): uses the queue's snapshot
methods to render current + queued + history jobs with their
statuses. Does NOT derive state solely from queue deque — uses
the full job lifecycle (queued/running/cancelled/completed/failed).

For GVM where per-frame progress is unavailable, shows indeterminate
progress plus stage text.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, Signal

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


class QueuePanel(QWidget):
    """Collapsible panel showing all jobs in the queue."""

    cancel_job_requested = Signal(str)  # job_id

    def __init__(self, queue: GPUJobQueue, parent=None):
        super().__init__(parent)
        self._queue = queue
        self.setMaximumHeight(180)
        self.setStyleSheet("background-color: #0E0D00; border-top: 1px solid #2A2910;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        # Header
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("QUEUE")
        title.setObjectName("sectionHeader")
        header.addWidget(title)
        header.addStretch()

        self._count_label = QLabel("0 jobs")
        self._count_label.setStyleSheet("color: #808070; font-size: 10px;")
        header.addWidget(self._count_label)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(50)
        clear_btn.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        clear_btn.setToolTip("Clear completed and cancelled jobs from the queue history")
        clear_btn.clicked.connect(self._on_clear)
        header.addWidget(clear_btn)

        layout.addLayout(header)

        # Scrollable job list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self._job_container = QWidget()
        self._job_layout = QVBoxLayout(self._job_container)
        self._job_layout.setContentsMargins(0, 0, 0, 0)
        self._job_layout.setSpacing(2)
        self._job_layout.addStretch()

        scroll.setWidget(self._job_container)
        layout.addWidget(scroll, 1)

    def refresh(self) -> None:
        """Rebuild the job list from queue snapshot."""
        # Clear existing widgets (except the stretch at the end)
        while self._job_layout.count() > 1:
            item = self._job_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        jobs = self._queue.all_jobs_snapshot
        self._count_label.setText(f"{len(jobs)} job{'s' if len(jobs) != 1 else ''}")

        for job in jobs:
            row = self._create_job_row(job)
            # Insert before the stretch
            self._job_layout.insertWidget(self._job_layout.count() - 1, row)

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
            # Show progress bar for running jobs
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
            # Show indeterminate bar + "Starting..." so user knows work is queued
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

        # Cancel button (only for queued or running)
        if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            cancel_btn = QPushButton("X")
            cancel_btn.setFixedSize(20, 20)
            cancel_btn.setStyleSheet(
                "background: #2A2910; color: #D10000; font-size: 10px; "
                "font-weight: 700; border: none; padding: 0;"
            )
            cancel_btn.setToolTip("Cancel this job")
            job_id = job.id
            cancel_btn.clicked.connect(lambda checked, jid=job_id: self.cancel_job_requested.emit(jid))
            layout.addWidget(cancel_btn)

        return row

    def _on_clear(self) -> None:
        """Clear job history."""
        self._queue.clear_history()
        self.refresh()

"""Collapsible queue panel showing per-job progress.

The header bar ("QUEUE") is always pinned at its position. When expanded,
the job list appears ABOVE the header — the header itself never moves.
Jobs grow upward: the first job sits directly on top of the header,
subsequent jobs stack above it.

Progress bars update in-place (no widget rebuild per frame) so the bar
moves smoothly instead of stuttering on every progress tick.
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
_ROW_H = 30      # approximate height per job row
_BODY_MAX_H = 160


class _JobRowCache:
    """Cached widgets for a single job row, enabling in-place updates."""
    __slots__ = ("frame", "progress_bar", "frame_label", "status_label",
                 "last_status", "last_current", "last_total")

    def __init__(self, frame: QFrame, progress_bar: QProgressBar | None,
                 frame_label: QLabel | None, status_label: QLabel | None):
        self.frame = frame
        self.progress_bar = progress_bar
        self.frame_label = frame_label
        self.status_label = status_label
        self.last_status: JobStatus | None = None
        self.last_current: int = -1
        self.last_total: int = -1


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

        # "QUEUE ▲" button — up arrow = collapsed (click to open upward)
        self._header_btn = QPushButton("QUEUE \u25B2")  # ▲ collapsed
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

        # Body — popup that appears ABOVE the header, flush against it
        self._body = QWidget(parent)
        self._body.setStyleSheet(
            "background-color: #0E0D00; border: 1px solid #2A2910; "
            "border-bottom: none;"
        )
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(4, 2, 4, 0)
        body_layout.setSpacing(2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setMaximumHeight(_BODY_MAX_H)
        self._scroll = scroll

        self._job_container = QWidget()
        self._job_layout = QVBoxLayout(self._job_container)
        self._job_layout.setContentsMargins(0, 0, 0, 0)
        self._job_layout.setSpacing(2)

        scroll.setWidget(self._job_container)
        body_layout.addWidget(scroll, 1)

        self._body.hide()
        self._collapsed = True

        # Cached row widgets keyed by job_id — avoids full rebuild on progress ticks
        self._row_cache: dict[str, _JobRowCache] = {}
        # Ordered list of job_ids currently displayed (for structure comparison)
        self._displayed_ids: list[str] = []

        # Hover/click sounds on interactive buttons
        self._sound_btns = {self._header_btn, self._clear_btn}
        for btn in self._sound_btns:
            btn.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj in self._sound_btns:
            from ui.sounds.audio_manager import UIAudio
            if event.type() == QEvent.Enter and obj.isVisible():
                UIAudio.hover(key=f"btn:{obj.text().replace(' ', '')}")
            elif event.type() == QEvent.MouseButtonPress:
                UIAudio.click()
        return super().eventFilter(obj, event)

    def toggle_collapsed(self) -> None:
        """Toggle the job list popup above the header."""
        self._collapsed = not self._collapsed
        self._body.setVisible(not self._collapsed)
        self._clear_btn.setVisible(not self._collapsed)
        if self._collapsed:
            self._header_btn.setText("QUEUE \u25B2")  # ▲ up to open
            self._header_btn.setToolTip("Expand queue panel (Q)")
        else:
            self._header_btn.setText("QUEUE \u25BC")  # ▼ down to close
            self._header_btn.setToolTip("Collapse queue panel (Q)")
        self.reposition()

    def reposition(self) -> None:
        """Position the body popup directly above the header bar — flush, no gap."""
        if not self._body.isVisible():
            return
        my_geo = self.geometry()
        job_count = self._job_layout.count()
        if job_count == 0:
            self._body.hide()
            return
        content_h = job_count * (_ROW_H + 2) + 6
        body_h = min(content_h, _BODY_MAX_H)
        self._body.setFixedWidth(my_geo.width())
        self._body.setFixedHeight(body_h)
        self._body.move(my_geo.x(), my_geo.y() - body_h)
        self._body.raise_()

    def refresh(self) -> None:
        """Update the job list — rebuild only when structure changes,
        otherwise update progress bars in-place for smooth animation."""
        jobs = self._queue.all_jobs_snapshot
        count = len(jobs)
        self._count_label.setText(f"{count} job{'s' if count != 1 else ''}" if count else "")

        # Build the ordered id list (reversed — newest closest to header)
        new_ids = [job.id for job in reversed(jobs)]

        # If the set of jobs or their order changed, do a full rebuild
        if new_ids != self._displayed_ids:
            self._full_rebuild(jobs)
            return

        # Otherwise just update progress in-place (fast path — no widget churn)
        for job in jobs:
            cache = self._row_cache.get(job.id)
            if cache is None:
                continue
            # Only update if something actually changed
            if (job.status == cache.last_status
                    and job.current_frame == cache.last_current
                    and job.total_frames == cache.last_total):
                continue
            self._update_row_in_place(cache, job)

    def _full_rebuild(self, jobs: list[GPUJob]) -> None:
        """Destroy all rows and recreate from scratch."""
        # Clear layout
        while self._job_layout.count() > 0:
            item = self._job_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._row_cache.clear()

        # Reversed: newest job closest to QUEUE bar (bottom of layout)
        for job in reversed(jobs):
            row, cache = self._create_job_row(job)
            self._job_layout.addWidget(row)
            self._row_cache[job.id] = cache

        self._displayed_ids = [job.id for job in reversed(jobs)]

        if not self._collapsed:
            self.reposition()

    def _create_job_row(self, job: GPUJob) -> tuple[QFrame, _JobRowCache]:
        """Create a single job row widget and return (frame, cache)."""
        row = QFrame()
        row.setFixedHeight(_ROW_H)
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

        # Status / Progress — create all possible widgets, show/hide as needed
        color = _STATUS_COLORS.get(job.status, "#808070")

        # Progress bar (used for RUNNING and QUEUED states)
        progress_bar = QProgressBar()
        progress_bar.setFixedHeight(6)
        progress_bar.setFixedWidth(100)
        progress_bar.setTextVisible(False)
        layout.addWidget(progress_bar)

        # Frame counter label (e.g. "65/499")
        frame_label = QLabel("")
        frame_label.setStyleSheet("color: #999980; font-size: 10px;")
        layout.addWidget(frame_label)

        # Status text label (e.g. "DONE", "CANCELLED", "Processing...")
        status_label = QLabel("")
        status_label.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 600;")
        layout.addWidget(status_label)

        layout.addStretch()

        # Dismiss button — only on finished jobs
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

        cache = _JobRowCache(row, progress_bar, frame_label, status_label)
        self._update_row_in_place(cache, job)
        return row, cache

    def _update_row_in_place(self, cache: _JobRowCache, job: GPUJob) -> None:
        """Update a cached row's widgets to reflect current job state.
        This is the fast path — no widget creation/destruction."""
        color = _STATUS_COLORS.get(job.status, "#808070")
        status_text = _STATUS_TEXT.get(job.status, "?")

        pb = cache.progress_bar
        fl = cache.frame_label
        sl = cache.status_label

        if job.status == JobStatus.RUNNING:
            if job.total_frames > 0:
                pct = int(job.current_frame / job.total_frames * 100)
                pb.setRange(0, 100)
                pb.setValue(pct)
                pb.setFixedWidth(100)
                pb.show()
                fl.setText(f"{job.current_frame}/{job.total_frames}")
                fl.show()
                sl.hide()
            else:
                # Indeterminate — only reset range if not already indeterminate
                if cache.last_total != 0 or cache.last_status != JobStatus.RUNNING:
                    pb.setRange(0, 0)
                pb.setFixedWidth(100)
                pb.show()
                fl.hide()
                sl.setText("Processing...")
                sl.setStyleSheet(f"color: {color}; font-size: 10px;")
                sl.show()
        elif job.status == JobStatus.QUEUED:
            if cache.last_status != JobStatus.QUEUED:
                pb.setRange(0, 0)
            pb.setFixedWidth(60)
            pb.show()
            fl.hide()
            sl.setText(status_text)
            sl.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 600;")
            sl.show()
        elif job.status == JobStatus.CANCELLED:
            pb.hide()
            fl.hide()
            sl.setText(status_text)
            sl.setStyleSheet(
                f"color: {color}; font-size: 10px; text-decoration: line-through;"
            )
            sl.show()
        else:
            # COMPLETED, FAILED, etc.
            pb.hide()
            fl.hide()
            sl.setText(status_text)
            sl.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 600;")
            sl.show()

        cache.last_status = job.status
        cache.last_current = job.current_frame
        cache.last_total = job.total_frames

    def _dismiss_job(self, job_id: str) -> None:
        """Remove a single finished job from history and refresh."""
        self._queue.remove_job(job_id)
        self.refresh()

    def _on_clear(self) -> None:
        """Clear job history."""
        self._queue.clear_history()
        self.refresh()

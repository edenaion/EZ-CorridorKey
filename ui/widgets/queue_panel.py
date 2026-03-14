"""Collapsible queue sidebar — left panel in the main horizontal splitter.

Collapsed: narrow 24px tab with vertical "QUEUE" label (always visible).
Expanded: 24px tab + ~216px content panel with header, scrollable job list.

The tab is always present — click it to expand or collapse.

Progress bars update in-place (no widget rebuild per frame) so the bar
moves smoothly instead of stuttering on every progress tick.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QScrollArea, QFrame, QSizePolicy,
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
    JobStatus.QUEUED: "QUEUED",
    JobStatus.RUNNING: "PROCESSING",
    JobStatus.COMPLETED: "DONE",
    JobStatus.CANCELLED: "CANCELLED",
    JobStatus.FAILED: "FAILED",
}

_JOB_TYPE_LABELS = {
    JobType.INFERENCE: "Inference",
    JobType.GVM_ALPHA: "GVM Auto",
    JobType.SAM2_PREVIEW: "Track Preview",
    JobType.SAM2_TRACK: "Track Mask",
    JobType.VIDEOMAMA_ALPHA: "VideoMaMa",
    JobType.MATANYONE2_ALPHA: "MatAnyone2",
    JobType.BIREFNET_ALPHA: "BiRefNet",
    JobType.PREVIEW_REPROCESS: "Preview",
}

_TAB_W = 24
_CONTENT_W = 216
_EXPANDED_W = _TAB_W + _CONTENT_W  # 240
_ROW_H = 60


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
    """Collapsible left sidebar showing GPU job queue.

    The 24px tab with vertical "QUEUE" is always visible.
    Clicking it toggles the content panel on/off.
    """

    cancel_job_requested = Signal(str)  # job_id

    def __init__(self, queue: GPUJobQueue, parent=None):
        super().__init__(parent)
        self._queue = queue
        self._collapsed = True

        self.setStyleSheet("background-color: #0E0D00;")
        self.setMinimumWidth(_TAB_W)
        self.setMaximumWidth(_EXPANDED_W)
        self.setFixedWidth(_TAB_W)

        # Horizontal layout: [tab | content panel]
        self._main_layout = QHBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        # ── Always-visible tab: vertical "QUEUE" ──
        self._tab = QPushButton("Q\nU\nE\nU\nE")
        self._tab.setCursor(Qt.PointingHandCursor)
        self._tab.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._tab.setFixedWidth(_TAB_W)
        self._tab.setToolTip("Toggle queue panel (Q)")
        self._tab.clicked.connect(self.toggle_collapsed)
        self._main_layout.addWidget(self._tab)
        self._apply_tab_style()

        # ── Content panel (shown when expanded) ──
        self._panel = QWidget()
        self._panel.setStyleSheet(
            "background-color: #0E0D00; border-right: 1px solid #2A2910;"
        )
        self._panel.setFixedWidth(_CONTENT_W)
        panel_layout = QVBoxLayout(self._panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        # Header bar
        header = QWidget()
        header.setFixedHeight(28)
        header.setStyleSheet("background: #1A1900; border-bottom: 1px solid #2A2910;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 2, 4, 2)
        header_layout.setSpacing(4)

        title = QLabel("QUEUE")
        title.setStyleSheet(
            "color: #CCCCAA; font-size: 11px; font-weight: 700; "
            "letter-spacing: 1px; background: transparent; border: none;"
        )
        header_layout.addWidget(title)

        self._count_label = QLabel("")
        self._count_label.setStyleSheet(
            "color: #808070; font-size: 10px; background: transparent; border: none;"
        )
        header_layout.addWidget(self._count_label)

        header_layout.addStretch()

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFixedHeight(18)
        self._clear_btn.setCursor(Qt.PointingHandCursor)
        self._clear_btn.setStyleSheet(
            "QPushButton { font-size: 9px; padding: 1px 6px; "
            "background: transparent; border: 1px solid #2A2910; "
            "color: #555540; }"
            "QPushButton:hover { color: #CCCCAA; border-color: #454430; }"
        )
        self._clear_btn.setToolTip("Clear completed and cancelled jobs")
        self._clear_btn.clicked.connect(self._on_clear)
        header_layout.addWidget(self._clear_btn)

        panel_layout.addWidget(header)

        # Scrollable job list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        self._scroll = scroll

        self._job_container = QWidget()
        self._job_container.setStyleSheet("background: transparent; border: none;")
        self._job_layout = QVBoxLayout(self._job_container)
        self._job_layout.setContentsMargins(0, 0, 0, 0)
        self._job_layout.setSpacing(1)
        self._job_layout.addStretch()

        scroll.setWidget(self._job_container)
        panel_layout.addWidget(scroll, 1)

        self._panel.hide()
        self._main_layout.addWidget(self._panel)

        # Cached row widgets keyed by job_id
        self._row_cache: dict[str, _JobRowCache] = {}
        self._displayed_ids: list[str] = []

    def _apply_tab_style(self) -> None:
        """Update tab color based on collapsed/expanded state."""
        color = "#808070" if self._collapsed else "#CCCCAA"
        self._tab.setStyleSheet(
            f"QPushButton {{ color: {color}; background: #0E0D00; "
            "border: none; border-right: 1px solid #2A2910; "
            "font-size: 11px; font-weight: 700; padding: 0; "
            "line-height: 14px; }"
            "QPushButton:hover { color: #FFF203; background: #1A1900; }"
        )

    def toggle_collapsed(self) -> None:
        """Toggle between collapsed (tab only) and expanded (tab + content)."""
        self._collapsed = not self._collapsed
        if self._collapsed:
            self._panel.hide()
            self.setFixedWidth(_TAB_W)
        else:
            self._panel.show()
            self.setFixedWidth(_EXPANDED_W)
            self.raise_()
        self._apply_tab_style()

    def refresh(self) -> None:
        """Update the job list — rebuild only when structure changes,
        otherwise update progress bars in-place for smooth animation."""
        jobs = self._queue.all_jobs_snapshot
        count = len(jobs)
        self._count_label.setText(f"({count})" if count else "")

        new_ids = [job.id for job in jobs]

        if new_ids != self._displayed_ids:
            self._full_rebuild(jobs)
            return

        # Fast path: update progress in-place
        for job in jobs:
            cache = self._row_cache.get(job.id)
            if cache is None:
                continue
            if (job.status == cache.last_status
                    and job.current_frame == cache.last_current
                    and job.total_frames == cache.last_total):
                continue
            self._update_row_in_place(cache, job)

    def _full_rebuild(self, jobs: list[GPUJob]) -> None:
        """Destroy all rows and recreate from scratch."""
        while self._job_layout.count() > 0:
            item = self._job_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._row_cache.clear()

        for job in jobs:
            row, cache = self._create_job_row(job)
            self._job_layout.addWidget(row)
            self._row_cache[job.id] = cache

        self._job_layout.addStretch()
        self._displayed_ids = [job.id for job in jobs]

    def _create_job_row(self, job: GPUJob) -> tuple[QFrame, _JobRowCache]:
        """Create a single job row — vertical stack: type+clip, progress/status."""
        row = QFrame()
        row.setFixedHeight(_ROW_H)
        row.setStyleSheet(
            "QFrame { background-color: #1A1900; border-bottom: 1px solid #151400; }"
        )
        layout = QVBoxLayout(row)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        # Top line: job type + dismiss button
        top = QHBoxLayout()
        top.setSpacing(6)

        type_text = _JOB_TYPE_LABELS.get(job.job_type, "???")
        type_label = QLabel(type_text)
        type_label.setStyleSheet(
            "color: #999980; font-size: 9px; font-weight: 700; "
            "letter-spacing: 0.5px; border: none; background: transparent;"
        )
        top.addWidget(type_label)

        top.addStretch()

        # Dismiss button for finished jobs
        if job.status in (JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.FAILED):
            dismiss_btn = QPushButton("\u2715")  # ✕
            dismiss_btn.setFixedSize(14, 14)
            dismiss_btn.setCursor(Qt.PointingHandCursor)
            dismiss_btn.setStyleSheet(
                "QPushButton { background: transparent; color: #555540; "
                "font-size: 9px; border: none; padding: 0; }"
                "QPushButton:hover { color: #999980; }"
            )
            dismiss_btn.setToolTip("Dismiss")
            job_id = job.id
            dismiss_btn.clicked.connect(
                lambda checked, jid=job_id: self._dismiss_job(jid)
            )
            top.addWidget(dismiss_btn)

        layout.addLayout(top)

        # Clip name (truncated)
        name_label = QLabel(job.clip_name)
        name_label.setStyleSheet(
            "color: #CCCCAA; font-size: 10px; border: none; background: transparent;"
        )
        name_label.setMaximumWidth(_CONTENT_W - 20)
        layout.addWidget(name_label)

        # Bottom line: progress bar + frame count / status text
        bottom = QHBoxLayout()
        bottom.setSpacing(4)

        progress_bar = QProgressBar()
        progress_bar.setFixedHeight(4)
        progress_bar.setTextVisible(False)
        progress_bar.setStyleSheet(
            "QProgressBar { background: #151400; border: none; border-radius: 2px; }"
            "QProgressBar::chunk { background: #FFF203; border-radius: 2px; }"
        )
        bottom.addWidget(progress_bar, 1)

        frame_label = QLabel("")
        frame_label.setStyleSheet(
            "color: #999980; font-size: 9px; border: none; background: transparent;"
        )
        bottom.addWidget(frame_label)

        status_label = QLabel("")
        status_label.setStyleSheet(
            "color: #808070; font-size: 9px; font-weight: 600; "
            "border: none; background: transparent;"
        )
        bottom.addWidget(status_label)

        layout.addLayout(bottom)

        cache = _JobRowCache(row, progress_bar, frame_label, status_label)
        self._update_row_in_place(cache, job)
        return row, cache

    def _update_row_in_place(self, cache: _JobRowCache, job: GPUJob) -> None:
        """Update a cached row's widgets to reflect current job state."""
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
                pb.show()
                fl.setText(f"{job.current_frame}/{job.total_frames}")
                fl.show()
                sl.hide()
            else:
                if cache.last_total != 0 or cache.last_status != JobStatus.RUNNING:
                    pb.setRange(0, 0)
                pb.show()
                fl.hide()
                sl.setText("Processing...")
                sl.setStyleSheet(
                    f"color: {color}; font-size: 9px; border: none; background: transparent;"
                )
                sl.show()
        elif job.status == JobStatus.QUEUED:
            if cache.last_status != JobStatus.QUEUED:
                pb.setRange(0, 0)
            pb.show()
            fl.hide()
            sl.setText(status_text)
            sl.setStyleSheet(
                f"color: {color}; font-size: 9px; font-weight: 600; "
                "border: none; background: transparent;"
            )
            sl.show()
        elif job.status == JobStatus.CANCELLED:
            pb.hide()
            fl.hide()
            sl.setText(status_text)
            sl.setStyleSheet(
                f"color: {color}; font-size: 9px; text-decoration: line-through; "
                "border: none; background: transparent;"
            )
            sl.show()
        else:
            pb.hide()
            fl.hide()
            sl.setText(status_text)
            sl.setStyleSheet(
                f"color: {color}; font-size: 9px; font-weight: 600; "
                "border: none; background: transparent;"
            )
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

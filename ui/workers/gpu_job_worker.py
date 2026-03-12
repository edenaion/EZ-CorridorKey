"""Single QThread GPU worker — consumes jobs from GPUJobQueue.

Design decisions (from Codex review):
- ONE dedicated QThread, single consumer for all GPU jobs
- Emits signals AFTER releasing any locks (no deadlock risk)
- Preview throttled to every N frames (configurable, default 5)
- Preview saved to temp file — QPixmap created on main thread only
- All signals carry job.id (stable, assigned at creation time)
- Job params are FROZEN SNAPSHOTS: paths + params dict, not live references
- Cancel path uses queue.mark_cancelled() to properly clear _current_job
- Model residency: service._ensure_model() unloads previous model type
"""
from __future__ import annotations

import copy
import os
import logging
import tempfile

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition

from backend import (
    CorridorKeyService,
    ClipEntry,
    ClipState,
    GPUJob,
    GPUJobQueue,
    InferenceParams,
    JobType,
)
from backend.job_queue import JobStatus
from backend.errors import JobCancelledError, CorridorKeyError

logger = logging.getLogger(__name__)


class GPUJobWorker(QThread):
    """Single-consumer GPU worker thread.

    Signals carry job.id (stable, assigned at submit time) so the UI
    can ignore stale signals from previous selections or cancelled jobs.
    """

    # Signals — all carry job_id as first arg for stale detection
    progress = Signal(str, str, int, int)      # job_id, clip_name, current_frame, total_frames
    preview_ready = Signal(str, str, int, str) # job_id, clip_name, frame_index, temp_file_path
    clip_finished = Signal(str, str, str)       # job_id, clip_name, job_type_value
    warning = Signal(str, str)                 # job_id, message
    status_update = Signal(str, str)           # job_id, status_text (phase label for status bar)
    error = Signal(str, str, str)              # job_id, clip_name, error_message
    queue_empty = Signal()                     # all jobs done
    reprocess_result = Signal(str, object)     # job_id, result_dict (for preview display)

    def __init__(self, service: CorridorKeyService, parent=None):
        super().__init__(parent)
        self._service = service
        self._queue = service.job_queue
        self._running = False
        self._mutex = QMutex()
        self._condition = QWaitCondition()
        self._preview_interval = 5  # emit preview every N frames
        self._preview_dir = tempfile.mkdtemp(prefix="corridorkey_preview_")

    @property
    def preview_dir(self) -> str:
        return self._preview_dir

    def wake(self) -> None:
        """Wake the worker to check for new jobs."""
        self._condition.wakeOne()

    def stop(self) -> None:
        """Signal the worker to stop after current job."""
        self._running = False
        self._queue.cancel_all()
        self._condition.wakeOne()

    def run(self) -> None:
        """Main consumer loop — process jobs one at a time."""
        self._running = True
        logger.info("GPU worker started")

        while self._running:
            job = self._queue.next_job()

            if job is None:
                # Wait for new jobs
                self._mutex.lock()
                if self._running and not self._queue.has_pending:
                    self._condition.wait(self._mutex, 500)  # 500ms timeout to recheck
                self._mutex.unlock()
                continue

            self._process_job(job)

        logger.info("GPU worker stopped")

    def _process_job(self, job: GPUJob) -> None:
        """Execute a single GPU job."""
        job_id = job.id
        self._queue.start_job(job)

        try:
            if job.job_type == JobType.INFERENCE:
                self._run_inference(job)
            elif job.job_type == JobType.GVM_ALPHA:
                self._run_gvm(job)
            elif job.job_type == JobType.SAM2_TRACK:
                self._run_sam2_track(job)
            elif job.job_type == JobType.VIDEOMAMA_ALPHA:
                self._run_videomama(job)
            elif job.job_type == JobType.MATANYONE2_ALPHA:
                self._run_matanyone2(job)
            elif job.job_type == JobType.PREVIEW_REPROCESS:
                self._run_preview_reprocess(job)
            else:
                self._queue.fail_job(job, f"Unknown job type: {job.job_type}")
                self.error.emit(job_id, job.clip_name, f"Unknown job type: {job.job_type}")
                return

            # If cancel was requested during a non-interruptible phase
            # (e.g. model load / CUDA teardown), honor it before marking the
            # job completed.
            if job.is_cancelled:
                self._queue.mark_cancelled(job)
                logger.info(f"Job cancelled after phase return [{job_id}]: {job.clip_name}")
                self.warning.emit(job_id, f"Cancelled: {job.clip_name}")
                return

            self._queue.complete_job(job)
            if job.job_type != JobType.PREVIEW_REPROCESS:
                self.clip_finished.emit(job_id, job.clip_name, job.job_type.value)

        except JobCancelledError:
            # Use mark_cancelled() to properly clear _current_job (Codex critical fix)
            self._queue.mark_cancelled(job)
            logger.info(f"Job cancelled [{job_id}]: {job.clip_name}")
            self.warning.emit(job_id, f"Cancelled: {job.clip_name}")

        except CorridorKeyError as e:
            logger.error(f"Job failed [{job_id}]: {job.clip_name} — {e}")
            self._queue.fail_job(job, str(e))
            self.error.emit(job_id, job.clip_name, str(e))

        except Exception as e:
            msg = f"Unexpected error: {e}"
            self._queue.fail_job(job, msg)
            self.error.emit(job_id, job.clip_name, msg)
            logger.exception(msg)

        # Check if queue is empty after this job
        if not self._queue.has_pending:
            self.queue_empty.emit()

    def _run_inference(self, job: GPUJob) -> None:
        """Run CorridorKey inference for a single clip."""
        clip = job.params.get("_clip_snapshot")
        params = job.params.get("_inference_params")
        skip_stems = job.params.get("_skip_stems", set())

        if clip is None or params is None:
            raise CorridorKeyError(f"Job [{job.id}] for '{job.clip_name}' missing clip or params snapshot")

        def on_progress(clip_name: str, current: int, total: int, **kwargs) -> None:
            self.progress.emit(job.id, clip_name, current, total)

            # Throttled preview — every N frames, save a comp preview to temp
            if current > 0 and current % self._preview_interval == 0:
                self._save_preview(job.id, clip, current)

        def on_warning(message: str) -> None:
            self.warning.emit(job.id, message)

        def on_status(message: str) -> None:
            self.status_update.emit(job.id, message)

        output_config = job.params.get("_output_config")
        frame_range = job.params.get("_frame_range")
        self._service.run_inference(
            clip=clip,
            params=params,
            job=job,
            on_progress=on_progress,
            on_warning=on_warning,
            on_status=on_status,
            skip_stems=skip_stems,
            output_config=output_config,
            frame_range=frame_range,
        )

    def _run_gvm(self, job: GPUJob) -> None:
        """Run GVM auto alpha generation."""
        clip = job.params.get("_clip_snapshot")
        if clip is None:
            raise CorridorKeyError(f"Job [{job.id}] for '{job.clip_name}' missing clip snapshot")

        def on_progress(clip_name: str, current: int, total: int, **kwargs) -> None:
            self.progress.emit(job.id, clip_name, current, total)

        def on_warning(message: str) -> None:
            self.warning.emit(job.id, message)

        self._service.run_gvm(
            clip=clip,
            job=job,
            on_progress=on_progress,
            on_warning=on_warning,
        )

    def _run_sam2_track(self, job: GPUJob) -> None:
        """Run SAM2 prompt-to-mask tracking."""
        clip = job.params.get("_clip_snapshot")
        if clip is None:
            raise CorridorKeyError(f"Job [{job.id}] for '{job.clip_name}' missing clip snapshot")

        def on_progress(clip_name: str, current: int, total: int, **kwargs) -> None:
            self.progress.emit(job.id, clip_name, current, total)

        def on_warning(message: str) -> None:
            self.warning.emit(job.id, message)

        def on_status(message: str) -> None:
            self.status_update.emit(job.id, message)

        self._service.run_sam2_track(
            clip=clip,
            job=job,
            on_progress=on_progress,
            on_warning=on_warning,
            on_status=on_status,
        )

    def _run_videomama(self, job: GPUJob) -> None:
        """Run VideoMaMa guided alpha generation."""
        clip = job.params.get("_clip_snapshot")
        chunk_size = job.params.get("_chunk_size", 16)

        if clip is None:
            raise CorridorKeyError(f"Job [{job.id}] for '{job.clip_name}' missing clip snapshot")

        def on_progress(clip_name: str, current: int, total: int, **kwargs) -> None:
            self.progress.emit(job.id, clip_name, current, total)

        def on_warning(message: str) -> None:
            self.warning.emit(job.id, message)

        def on_status(message: str) -> None:
            self.status_update.emit(job.id, message)

        self._service.run_videomama(
            clip=clip,
            job=job,
            on_progress=on_progress,
            on_warning=on_warning,
            on_status=on_status,
            chunk_size=chunk_size,
        )

    def _run_matanyone2(self, job: GPUJob) -> None:
        """Run MatAnyone2 video matting alpha generation."""
        clip = job.params.get("_clip_snapshot")

        if clip is None:
            raise CorridorKeyError(f"Job [{job.id}] for '{job.clip_name}' missing clip snapshot")

        def on_progress(clip_name: str, current: int, total: int, **kwargs) -> None:
            self.progress.emit(job.id, clip_name, current, total)

        def on_warning(message: str) -> None:
            self.warning.emit(job.id, message)

        def on_status(message: str) -> None:
            self.status_update.emit(job.id, message)

        self._service.run_matanyone2(
            clip=clip,
            job=job,
            on_progress=on_progress,
            on_warning=on_warning,
            on_status=on_status,
        )

    def _run_preview_reprocess(self, job: GPUJob) -> None:
        """Run single-frame reprocess through GPU queue (Codex: no GPU bypass)."""
        clip = job.params.get("_clip_snapshot")
        params = job.params.get("_inference_params")
        frame_index = job.params.get("_frame_index", 0)

        if clip is None or params is None:
            return

        result = self._service.reprocess_single_frame(
            clip=clip, params=params, frame_index=frame_index, job=job,
        )
        if result is not None:
            self.reprocess_result.emit(job.id, result)

    def _save_preview(self, job_id: str, clip: ClipEntry, frame_index: int) -> None:
        """Save a downscaled preview of the latest comp frame to temp dir.

        The main thread will load this as QPixmap — we never create
        QPixmap off the GUI thread (Codex finding).
        """
        try:
            comp_dir = os.path.join(clip.root_path, "Output", "Comp")
            if not os.path.isdir(comp_dir):
                return

            # Find the most recently written comp frame (natural sort)
            from backend.natural_sort import natsorted
            comp_files = natsorted(os.listdir(comp_dir))
            if not comp_files:
                return

            latest = os.path.join(comp_dir, comp_files[-1])
            img = cv2.imread(latest)
            if img is None:
                return

            # Downscale for preview (max 960px wide)
            h, w = img.shape[:2]
            if w > 960:
                scale = 960 / w
                img = cv2.resize(img, (960, int(h * scale)), interpolation=cv2.INTER_AREA)

            preview_path = os.path.join(self._preview_dir, f"preview_{job_id}.png")
            cv2.imwrite(preview_path, img)

            self.preview_ready.emit(job_id, clip.name, frame_index, preview_path)
        except Exception as e:
            logger.debug(f"Preview save skipped: {e}")


def create_job_snapshot(
    clip: ClipEntry,
    params: InferenceParams | None = None,
    job_type: JobType = JobType.INFERENCE,
    resume: bool = False,
    chunk_size: int = 16,
) -> GPUJob:
    """Create a frozen job snapshot for the queue.

    The clip is deep-copied so watcher rescans or UI mutations
    cannot desync the running job (Codex critical finding).

    Args:
        clip: The clip to process (will be deep-copied).
        params: Inference parameters (for INFERENCE jobs).
        job_type: Type of GPU job.
        resume: If True, populate skip_stems from existing outputs.
        chunk_size: VideoMaMa chunk size.
    """
    # Deep copy clip so the job holds frozen state, not a live reference
    clip_snapshot = copy.deepcopy(clip)

    job_params: dict = {"_clip_snapshot": clip_snapshot}

    if job_type == JobType.INFERENCE:
        if params is None:
            params = InferenceParams()
        job_params["_inference_params"] = params
        if resume:
            job_params["_skip_stems"] = clip.completed_stems()
    elif job_type == JobType.VIDEOMAMA_ALPHA:
        job_params["_chunk_size"] = chunk_size
    elif job_type == JobType.PREVIEW_REPROCESS:
        if params is None:
            params = InferenceParams()
        job_params["_inference_params"] = params

    return GPUJob(
        job_type=job_type,
        clip_name=clip.name,
        params=job_params,
    )

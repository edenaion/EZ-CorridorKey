"""GPU worker — consumes jobs from GPUJobQueue with optional parallelism.

Design decisions (from Codex review):
- ONE dedicated QThread orchestrates job dispatch
- INFERENCE jobs can run in parallel via ThreadPoolExecutor (configurable)
- Non-inference jobs (GVM, VideoMaMa, SAM2) run inline after draining pool
- Emits signals AFTER releasing any locks (no deadlock risk)
- Preview throttled to every N frames (configurable, default 5)
- Preview saved to temp file — QPixmap created on main thread only
- All signals carry job.id (stable, assigned at creation time)
- Job params are FROZEN SNAPSHOTS: paths + params dict, not live references
- Cancel path uses queue.mark_cancelled() to properly clear running state
- Model residency: service._ensure_model() unloads previous model type
- Lock order: service._gpu_lock > job_queue._lock > never hold while joining
"""

from __future__ import annotations

import os
import logging
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition

from backend import (
    CorridorKeyService,
    ClipEntry,
    GPUJob,
    InferenceParams,
    JobType,
)
from backend.errors import JobCancelledError, CorridorKeyError

# Re-export create_job_snapshot for backward compatibility
from ui.workers.job_helpers import create_job_snapshot

logger = logging.getLogger(__name__)

# Job types that can be processed in parallel (stateless, shared engine)
_PARALLEL_JOB_TYPES = frozenset({JobType.INFERENCE})


class GPUJobWorker(QThread):
    """GPU worker thread with optional parallel inference dispatch.

    Signals carry job.id (stable, assigned at submit time) so the UI
    can ignore stale signals from previous selections or cancelled jobs.

    Parallel mode (max_workers > 1):
        INFERENCE jobs are submitted to a ThreadPoolExecutor so multiple
        clips process concurrently.  The shared GPU lock in backend.service
        serializes model access while CPU pre/post-processing overlaps.
        Non-inference jobs (GVM, VideoMaMa, SAM2) drain the pool first,
        then run inline to respect model residency.
    """

    # Signals — all carry job_id as first arg for stale detection
    progress = Signal(
        str, str, int, int, float
    )  # job_id, clip_name, current_frame, total_frames, fps
    preview_ready = Signal(str, str, int, str)  # job_id, clip_name, frame_index, temp_file_path
    clip_finished = Signal(str, str, str)  # job_id, clip_name, job_type_value
    warning = Signal(str, str)  # job_id, message
    status_update = Signal(str, str)  # job_id, status_text (phase label for status bar)
    error = Signal(str, str, str)  # job_id, clip_name, error_message
    queue_empty = Signal()  # all jobs done
    reprocess_result = Signal(str, object)  # job_id, result_dict (for preview display)

    def __init__(self, service: CorridorKeyService, max_workers: int = 1, parent=None):
        super().__init__(parent)
        self._service = service
        self._queue = service.job_queue
        self._running = False
        self._mutex = QMutex()
        self._condition = QWaitCondition()
        self._preview_interval = 5  # emit preview every N frames
        self._preview_dir = tempfile.mkdtemp(prefix="corridorkey_preview_")

        # Parallel inference dispatch
        self._max_workers = max(1, max_workers)
        self._concurrency_sem = threading.Semaphore(self._max_workers)
        # Wire engine pool size for frame-level parallelism
        self._service.set_pool_size(self._max_workers)
        self._pool: ThreadPoolExecutor | None = None
        self._inflight: dict[str, Future] = {}  # job_id -> Future
        self._inflight_lock = threading.Lock()
        # When True, stop admitting new inference jobs to pool (drain for exclusive job)
        self._draining = False

    @property
    def preview_dir(self) -> str:
        return self._preview_dir

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def set_max_workers(self, n: int) -> None:
        """Hot-update the concurrency limit (takes effect for next job batch).

        Also updates the service engine pool size so parallel frame
        processing uses the right number of engines.
        """
        n = max(1, min(n, 8))
        if n == self._max_workers:
            return
        old = self._max_workers
        self._max_workers = n
        # Replace semaphore — safe because only the main loop acquires it
        self._concurrency_sem = threading.Semaphore(n)
        # Wire engine pool size for frame-level parallelism
        self._service.set_pool_size(n)
        logger.info(f"Parallel frames updated: {old} -> {n}")

    def wake(self) -> None:
        """Wake the worker to check for new jobs."""
        self._condition.wakeOne()

    def stop(self) -> None:
        """Signal the worker to stop after current jobs."""
        self._running = False
        self._queue.cancel_all()
        self._drain_pool()
        self._condition.wakeOne()

    def _ensure_pool(self) -> ThreadPoolExecutor:
        """Lazily create the thread pool."""
        if self._pool is None:
            # Pool size is the max we'd ever need; semaphore controls actual concurrency
            self._pool = ThreadPoolExecutor(
                max_workers=8,
                thread_name_prefix="corridorkey-infer",
            )
        return self._pool

    def _drain_pool(self) -> None:
        """Wait for all in-flight inference jobs to complete.

        IMPORTANT: Must NOT hold any queue lock while calling this (deadlock risk).
        """
        with self._inflight_lock:
            futures = list(self._inflight.values())
        for fut in futures:
            try:
                fut.result(timeout=600)  # 10min max per job
            except Exception:
                pass  # errors handled in _pool_job_done callback
        with self._inflight_lock:
            self._inflight.clear()

    def run(self) -> None:
        """Main consumer loop — dispatches jobs to pool or runs inline."""
        self._running = True
        self._queue_empty_emitted = True  # Start idle — no spurious emit on launch
        logger.info("GPU worker started (max_workers=%d)", self._max_workers)

        while self._running:
            job = self._queue.next_job()

            if job is None:
                # Also check if all inflight are done for queue_empty signal
                with self._inflight_lock:
                    has_inflight = len(self._inflight) > 0
                if not has_inflight and not self._queue.has_pending:
                    if not self._queue_empty_emitted:
                        self._queue_empty_emitted = True
                        self.queue_empty.emit()

                # Wait for new jobs
                self._mutex.lock()
                if self._running and not self._queue.has_pending:
                    self._condition.wait(self._mutex, 500)
                self._mutex.unlock()
                continue

            self._queue_empty_emitted = False  # Reset — will fire again when all done
            if job.job_type in _PARALLEL_JOB_TYPES and self._max_workers > 1 and not self._draining:
                # Submit inference job to thread pool
                self._submit_to_pool(job)
            else:
                # Exclusive job (GVM, VideoMaMa, etc.) — drain pool first
                if job.job_type not in _PARALLEL_JOB_TYPES:
                    self._draining = True
                    self._drain_pool()
                    self._draining = False
                self._process_job(job)

        # Shutdown
        self._drain_pool()
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        logger.info("GPU worker stopped")

    def _submit_to_pool(self, job: GPUJob) -> None:
        """Submit an inference job to the thread pool."""
        self._queue.start_job(job)
        pool = self._ensure_pool()

        def _run_in_pool() -> None:
            self._concurrency_sem.acquire()
            try:
                self._process_job_inner(job)
            finally:
                self._concurrency_sem.release()
                with self._inflight_lock:
                    self._inflight.pop(job.id, None)
                # Wake main loop to check for more work / queue_empty
                self._condition.wakeOne()

        future = pool.submit(_run_in_pool)
        with self._inflight_lock:
            self._inflight[job.id] = future

    def _process_job(self, job: GPUJob) -> None:
        """Execute a single GPU job (inline path — marks start/complete)."""
        self._queue.start_job(job)
        self._process_job_inner(job)

    def _process_job_inner(self, job: GPUJob) -> None:
        """Execute a GPU job. Called from main loop (inline) or pool thread.

        Assumes start_job() has already been called.
        """
        job_id = job.id
        logger.info(
            f">>> PROCESS_JOB START [{job_id}]: type={job.job_type.value}, clip={job.clip_name}"
        )

        try:
            if job.job_type == JobType.INFERENCE:
                self._run_inference(job)
            elif job.job_type == JobType.GVM_ALPHA:
                self._run_gvm(job)
            elif job.job_type == JobType.SAM2_PREVIEW:
                self._run_sam2_preview(job)
            elif job.job_type == JobType.SAM2_TRACK:
                self._run_sam2_track(job)
            elif job.job_type == JobType.VIDEOMAMA_ALPHA:
                self._run_videomama(job)
            elif job.job_type == JobType.MATANYONE2_ALPHA:
                self._run_matanyone2(job)
            elif job.job_type == JobType.BIREFNET_ALPHA:
                self._run_birefnet(job)
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
            if job.job_type not in (JobType.PREVIEW_REPROCESS, JobType.SAM2_PREVIEW):
                self.clip_finished.emit(job_id, job.clip_name, job.job_type.value)

        except JobCancelledError:
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

        # Check if queue is empty and no inflight jobs
        with self._inflight_lock:
            has_inflight = len(self._inflight) > 0
        if not self._queue.has_pending and not has_inflight:
            if not self._queue_empty_emitted:
                self._queue_empty_emitted = True
                self.queue_empty.emit()

    def _run_inference(self, job: GPUJob) -> None:
        """Run CorridorKey inference for a single clip."""
        clip = job.params.get("_clip_snapshot")
        params = job.params.get("_inference_params")
        skip_stems = job.params.get("_skip_stems", set())

        if clip is None or params is None:
            raise CorridorKeyError(
                f"Job [{job.id}] for '{job.clip_name}' missing clip or params snapshot"
            )

        def on_progress(clip_name: str, current: int, total: int, **kwargs) -> None:
            self.progress.emit(job.id, clip_name, current, total, kwargs.get("fps", 0.0))

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
            self.progress.emit(job.id, clip_name, current, total, kwargs.get("fps", 0.0))

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
        params = job.params.get("_inference_params")
        if clip is None:
            raise CorridorKeyError(f"Job [{job.id}] for '{job.clip_name}' missing clip snapshot")

        def on_progress(clip_name: str, current: int, total: int, **kwargs) -> None:
            self.progress.emit(job.id, clip_name, current, total, kwargs.get("fps", 0.0))

        def on_warning(message: str) -> None:
            self.warning.emit(job.id, message)

        def on_status(message: str) -> None:
            self.status_update.emit(job.id, message)

        self._service.run_sam2_track(
            clip=clip,
            input_is_linear=(
                params.input_is_linear if isinstance(params, InferenceParams) else None
            ),
            job=job,
            on_progress=on_progress,
            on_warning=on_warning,
            on_status=on_status,
        )

    def _run_sam2_preview(self, job: GPUJob) -> None:
        """Run a one-frame SAM2 preview without writing tracked masks."""
        clip = job.params.get("_clip_snapshot")
        frame_index = job.params.get("_frame_index")
        params = job.params.get("_inference_params")
        if clip is None:
            raise CorridorKeyError(f"Job [{job.id}] for '{job.clip_name}' missing clip snapshot")

        def on_progress(clip_name: str, current: int, total: int, **kwargs) -> None:
            self.progress.emit(job.id, clip_name, current, total, kwargs.get("fps", 0.0))

        def on_warning(message: str) -> None:
            self.warning.emit(job.id, message)

        def on_status(message: str) -> None:
            self.status_update.emit(job.id, message)

        result = self._service.preview_sam2_prompt(
            clip=clip,
            preferred_frame_index=frame_index,
            input_is_linear=(
                params.input_is_linear if isinstance(params, InferenceParams) else None
            ),
            job=job,
            on_progress=on_progress,
            on_warning=on_warning,
            on_status=on_status,
        )
        if result is not None:
            self.reprocess_result.emit(job.id, result)

    def _run_videomama(self, job: GPUJob) -> None:
        """Run VideoMaMa guided alpha generation."""
        clip = job.params.get("_clip_snapshot")
        chunk_size = job.params.get("_chunk_size", 16)

        if clip is None:
            raise CorridorKeyError(f"Job [{job.id}] for '{job.clip_name}' missing clip snapshot")

        def on_progress(clip_name: str, current: int, total: int, **kwargs) -> None:
            self.progress.emit(job.id, clip_name, current, total, kwargs.get("fps", 0.0))

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
            self.progress.emit(job.id, clip_name, current, total, kwargs.get("fps", 0.0))

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

    def _run_birefnet(self, job: GPUJob) -> None:
        """Run BiRefNet automatic alpha generation."""
        clip = job.params.get("_clip_snapshot")

        if clip is None:
            raise CorridorKeyError(f"Job [{job.id}] for '{job.clip_name}' missing clip snapshot")

        usage = job.params.get("_birefnet_usage", "Matting")

        def on_progress(clip_name: str, current: int, total: int, **kwargs) -> None:
            self.progress.emit(job.id, clip_name, current, total, kwargs.get("fps", 0.0))

        def on_warning(message: str) -> None:
            self.warning.emit(job.id, message)

        def on_status(message: str) -> None:
            self.status_update.emit(job.id, message)

        self._service.run_birefnet(
            clip=clip,
            usage=usage,
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

        def on_status(message: str) -> None:
            self.status_update.emit(job.id, message)

        result = self._service.reprocess_single_frame(
            clip=clip,
            params=params,
            frame_index=frame_index,
            job=job,
            on_status=on_status,
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


__all__ = ["GPUJobWorker", "create_job_snapshot"]

"""Core CorridorKey inference pipeline — dispatcher, sequential, and finalization."""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Callable, Optional

import cv2
import numpy as np

from ..clip_state import ClipEntry, ClipState
from ..errors import (
    CorridorKeyError,
    FrameReadError,
    JobCancelledError,
    WriteFailureError,
)
from ..frame_io import read_video_frame_at, read_video_mask_at
from ..job_queue import GPUJob
from ..validators import ensure_output_dirs, validate_frame_counts
from .inference_parallel import ParallelInferenceMixin

logger = logging.getLogger(__name__)


# Sentinel pushed by the prefetch thread to signal end-of-stream
_PREFETCH_DONE = object()


class InferenceMixin(ParallelInferenceMixin):
    """Mixin providing core inference pipeline for CorridorKeyService."""

    def _prefetch_frames(
        self,
        clip: ClipEntry,
        frame_indices,
        input_files: list[str],
        alpha_files: list[str],
        input_cap,
        alpha_cap,
        input_is_linear: bool,
        alpha_stem_lookup,
        prefetch_queue: Queue,
        job: Optional[GPUJob],
    ) -> None:
        """Push (frame_index, img, mask, stem, is_linear, error_or_none) into
        the queue ahead of the GPU loop, plus a final _PREFETCH_DONE sentinel.
        """
        for i in frame_indices:
            if job and job.is_cancelled:
                break
            try:
                img, stem, is_linear = self._read_input_frame(
                    clip, i, input_files, input_cap, input_is_linear,
                )
                mask = self._read_alpha_frame(
                    clip, i, alpha_files, alpha_cap,
                    input_stem=stem,
                    alpha_stem_lookup=alpha_stem_lookup,
                )
            except (FrameReadError, OSError, RuntimeError) as e:
                prefetch_queue.put((i, None, None, f"{i:05d}", input_is_linear, str(e)))
                continue
            prefetch_queue.put((i, img, mask, stem, is_linear, None))

        prefetch_queue.put(_PREFETCH_DONE)

    def run_inference(
        self,
        clip: ClipEntry,
        params,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[..., None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        skip_stems: Optional[set[str]] = None,
        output_config=None,
        frame_range: Optional[tuple[int, int]] = None,
    ) -> list:
        """Run CorridorKey inference on a single clip.

        Uses gated model switch protocol. If pool_size > 1, dispatches
        to _run_inference_parallel with N engines processing N frames
        concurrently. Otherwise uses _run_inference_sequential.
        """
        if clip.input_asset is None or clip.alpha_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input or alpha asset")

        self._begin_inference()
        try:
            if on_status:
                on_status("Loading model...")
            logger.info("run_inference: waiting for _gpu_lock")
            with self._gpu_lock:
                logger.info("run_inference: acquired _gpu_lock")
                engines = self._get_engine_pool(on_status=on_status)

            if len(engines) == 1:
                return self._run_inference_sequential(
                    clip, params, engines[0], job=job,
                    on_progress=on_progress, on_warning=on_warning,
                    on_status=on_status, skip_stems=skip_stems,
                    output_config=output_config, frame_range=frame_range,
                )
            return self._run_inference_parallel(
                clip, params, engines, job=job,
                on_progress=on_progress, on_warning=on_warning,
                on_status=on_status, skip_stems=skip_stems,
                output_config=output_config, frame_range=frame_range,
            )
        finally:
            self._end_inference()

    def _run_inference_sequential(
        self,
        clip: ClipEntry,
        params,
        engine,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[..., None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        skip_stems: Optional[set[str]] = None,
        output_config=None,
        frame_range: Optional[tuple[int, int]] = None,
    ) -> list:
        """Sequential frame loop — exact extraction of original run_inference."""
        from .core import FrameResult, OutputConfig
        t_start = time.monotonic()
        dirs = ensure_output_dirs(clip.output_dir)
        cfg = output_config or OutputConfig()

        self._write_manifest(dirs['root'], cfg, params)

        if clip.input_asset.asset_type == 'sequence' and clip.alpha_asset.asset_type == 'sequence':
            num_frames = clip.input_asset.frame_count
            if clip.input_asset.frame_count != clip.alpha_asset.frame_count:
                logger.warning(
                    "Clip '%s': sequence alpha count mismatch — input has %d, alpha has %d. "
                    "Using stem-matched alpha reads across the selected range.",
                    clip.name,
                    clip.input_asset.frame_count,
                    clip.alpha_asset.frame_count,
                )
        else:
            num_frames = validate_frame_counts(
                clip.name,
                clip.input_asset.frame_count,
                clip.alpha_asset.frame_count,
            )

        input_cap = None
        alpha_cap = None
        input_files: list[str] = []
        alpha_files: list[str] = []

        if clip.input_asset.asset_type == 'video':
            input_cap = cv2.VideoCapture(clip.input_asset.path)
        else:
            input_files = clip.input_asset.get_frame_files()

        if clip.alpha_asset.asset_type == 'video':
            alpha_cap = cv2.VideoCapture(clip.alpha_asset.path)
        else:
            alpha_files = clip.alpha_asset.get_frame_files()
        alpha_stem_lookup = (
            {os.path.splitext(fname)[0]: fname for fname in alpha_files}
            if alpha_files else None
        )

        results: list[FrameResult] = []
        skipped: list[int] = []
        skip_stems = skip_stems or set()
        frame_times: deque[float] = deque(maxlen=10)
        processed_count = 0

        if frame_range is not None:
            range_start = max(0, frame_range[0])
            range_end = min(num_frames - 1, frame_range[1])
            frame_indices = range(range_start, range_end + 1)
            range_count = range_end - range_start + 1
        else:
            frame_indices = range(num_frames)
            range_count = num_frames

        _warmup_done = False

        prefetch_q: Queue = Queue(maxsize=3)
        write_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ck-write")
        write_futures: list[tuple[int, str, "object"]] = []

        prefetch_thread = threading.Thread(
            target=self._prefetch_frames,
            args=(
                clip, frame_indices, input_files, alpha_files,
                input_cap, alpha_cap, params.input_is_linear,
                alpha_stem_lookup, prefetch_q, job,
            ),
            daemon=True,
        )
        prefetch_thread.start()

        try:
            progress_i = 0
            while True:
                if job and job.is_cancelled:
                    raise JobCancelledError(clip.name, progress_i)

                item = prefetch_q.get()
                if item is _PREFETCH_DONE:
                    break

                i, img, mask, input_stem, is_linear, read_error = item

                if not _warmup_done and on_status:
                    on_status("Compiling (first frame may take a minute)...")

                if on_progress:
                    timing_kwargs: dict[str, float] = {}
                    elapsed = time.monotonic() - t_start
                    timing_kwargs["elapsed"] = elapsed
                    if frame_times:
                        avg_time = sum(frame_times) / len(frame_times)
                        fps = 1.0 / avg_time if avg_time > 0 else 0.0
                        remaining = range_count - progress_i
                        timing_kwargs["fps"] = fps
                        timing_kwargs["eta_seconds"] = remaining * avg_time
                    on_progress(clip.name, progress_i, range_count, **timing_kwargs)

                if read_error:
                    skipped.append(i)
                    results.append(FrameResult(i, input_stem, False, read_error))
                    if on_warning:
                        on_warning(read_error)
                    progress_i += 1
                    continue

                if img is None:
                    skipped.append(i)
                    results.append(FrameResult(i, f"{i:05d}", False, "video read failed"))
                    progress_i += 1
                    continue

                if input_stem in skip_stems:
                    results.append(FrameResult(i, input_stem, True, "resumed (skipped)"))
                    progress_i += 1
                    continue

                if mask is None:
                    skipped.append(i)
                    results.append(FrameResult(i, input_stem, False, "alpha read failed"))
                    progress_i += 1
                    continue

                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

                try:
                    t_frame = time.monotonic()
                    with self._gpu_lock:
                        res = engine.process_frame(
                            img, mask,
                            input_is_linear=is_linear,
                            fg_is_straight=True,
                            despill_strength=params.despill_strength,
                            auto_despeckle=params.auto_despeckle,
                            despeckle_size=params.despeckle_size,
                            despeckle_dilation=params.despeckle_dilation,
                            despeckle_blur=params.despeckle_blur,
                            refiner_scale=params.refiner_scale,
                            source_passthrough=params.source_passthrough,
                            edge_erode_px=params.edge_erode_px,
                            edge_blur_px=params.edge_blur_px,
                        )
                except FrameReadError as e:
                    logger.warning(str(e))
                    skipped.append(i)
                    results.append(FrameResult(i, input_stem, False, str(e)))
                    if on_warning:
                        on_warning(str(e))
                    progress_i += 1
                    continue

                dt = time.monotonic() - t_frame
                frame_times.append(dt)
                processed_count += 1

                if not _warmup_done:
                    _warmup_done = True
                    if on_status:
                        on_status("")
                total_t = sum(frame_times)
                avg_fps = len(frame_times) / total_t if total_t > 0 else 0.0
                logger.debug(f"Frame {i}: {dt * 1000:.0f}ms ({avg_fps:.1f} fps avg)")

                future = write_pool.submit(
                    self._write_outputs, res, dirs, input_stem, clip.name, i, cfg,
                )
                write_futures.append((i, input_stem, future))
                results.append(FrameResult(i, input_stem, True))
                progress_i += 1

            if on_progress:
                final_elapsed = time.monotonic() - t_start
                final_kwargs: dict[str, float] = {"elapsed": final_elapsed, "eta_seconds": 0.0}
                total_t = sum(frame_times)
                if total_t > 0:
                    final_kwargs["fps"] = len(frame_times) / total_t
                on_progress(clip.name, range_count, range_count, **final_kwargs)

            # Drain async writes; flip the matching FrameResult to failure on error.
            for i, _stem, fut in write_futures:
                try:
                    fut.result()
                except WriteFailureError as e:
                    logger.error(str(e))
                    for r in results:
                        if r.frame_index == i and r.success:
                            r.success = False
                            r.warning = str(e)
                            break
                    if on_warning:
                        on_warning(str(e))

        finally:
            prefetch_thread.join(timeout=5)
            write_pool.shutdown(wait=True)
            if input_cap:
                input_cap.release()
            if alpha_cap:
                alpha_cap.release()

        return self._finalize_inference(
            clip, results, skipped, processed_count,
            t_start, frame_range, num_frames,
            on_warning=on_warning,
        )

    def _finalize_inference(
        self,
        clip: ClipEntry,
        results: list,
        skipped: list[int],
        processed_count: int,
        t_start: float,
        frame_range: Optional[tuple[int, int]],
        num_frames: int,
        on_warning: Optional[Callable[[str], None]] = None,
    ) -> list:
        """Shared post-processing for sequential and parallel inference."""
        range_count = len(results) if results else 0
        processed = sum(1 for r in results if r.success)
        if skipped:
            msg = (
                f"Clip '{clip.name}': {len(skipped)} frame(s) skipped: "
                f"{skipped[:20]}{'...' if len(skipped) > 20 else ''}"
            )
            logger.warning(msg)
            if on_warning:
                on_warning(msg)

        t_total = time.monotonic() - t_start
        avg_fps = processed_count / t_total if t_total > 0 and processed_count > 0 else 0.0
        range_label = f" (range {frame_range[0]}-{frame_range[1]})" if frame_range else ""
        logger.info(
            f"Clip '{clip.name}': inference complete{range_label}. {processed}/{range_count} frames "
            f"in {t_total:.1f}s ({t_total / max(processed, 1):.2f}s/frame, {avg_fps:.1f} fps avg)"
        )

        is_full_clip = (frame_range is None or
                        (frame_range[0] == 0 and frame_range[1] >= num_frames - 1))
        if processed == range_count and is_full_clip:
            try:
                clip.transition_to(ClipState.COMPLETE)
            except Exception as e:
                logger.warning(f"Clip '{clip.name}': state transition to COMPLETE failed: {e}")

        return results

    def reprocess_single_frame(
        self,
        clip: ClipEntry,
        params,
        frame_index: int,
        job: Optional[GPUJob] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Optional[dict]:
        """Reprocess a single frame with current params.

        Returns the result dict (fg, alpha, comp, processed) or None.
        Does NOT write to disk — returns in-memory results for preview.
        """
        from . import read_image_frame

        t_start = time.monotonic()
        if clip.input_asset is None or clip.alpha_asset is None:
            return None

        if job and job.is_cancelled:
            return None

        with self._gpu_lock:
            engines = self._get_engine_pool(on_status=on_status)
            engine = engines[0]

        # Read the specific input frame
        is_linear = params.input_is_linear
        input_stem = f"{frame_index:05d}"
        if clip.input_asset.asset_type == 'video':
            img = read_video_frame_at(clip.input_asset.path, frame_index)
        else:
            input_files = clip.input_asset.get_frame_files()
            if frame_index >= len(input_files):
                return None
            fpath = os.path.join(clip.input_asset.path, input_files[frame_index])
            input_stem = os.path.splitext(input_files[frame_index])[0]
            img = read_image_frame(fpath)
        if img is None:
            return None

        # Read the specific alpha frame
        if clip.alpha_asset.asset_type == 'video':
            mask = read_video_mask_at(clip.alpha_asset.path, frame_index)
        else:
            alpha_files = clip.alpha_asset.get_frame_files()
            alpha_stem_lookup = {os.path.splitext(fname)[0]: fname for fname in alpha_files}
            mask = self._read_alpha_frame(
                clip,
                frame_index,
                alpha_files,
                None,
                input_stem=input_stem,
                alpha_stem_lookup=alpha_stem_lookup,
            )
        if mask is None:
            return None

        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        if job and job.is_cancelled:
            return None

        with self._gpu_lock:
            res = engine.process_frame(
                img, mask,
                input_is_linear=is_linear,
                fg_is_straight=True,
                despill_strength=params.despill_strength,
                auto_despeckle=params.auto_despeckle,
                despeckle_size=params.despeckle_size,
                despeckle_dilation=params.despeckle_dilation,
                despeckle_blur=params.despeckle_blur,
                refiner_scale=params.refiner_scale,
                source_passthrough=params.source_passthrough,
                edge_erode_px=params.edge_erode_px,
                edge_blur_px=params.edge_blur_px,
            )
        logger.debug(f"Clip '{clip.name}' frame {frame_index}: reprocess {time.monotonic() - t_start:.3f}s")
        return res

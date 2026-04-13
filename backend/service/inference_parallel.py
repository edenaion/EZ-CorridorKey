"""Parallel inference pipeline — reader/worker/writer thread architecture."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Callable, Optional

import cv2

from ..clip_state import ClipEntry
from ..errors import (
    FrameReadError,
    JobCancelledError,
    WriteFailureError,
)
from ..job_queue import GPUJob
from ..validators import ensure_output_dirs, validate_frame_counts

logger = logging.getLogger(__name__)


class ParallelInferenceMixin:
    """Mixin providing the parallel inference pipeline for CorridorKeyService."""

    def _run_inference_parallel(
        self,
        clip: ClipEntry,
        params,
        engines: list,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[..., None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        skip_stems: Optional[set[str]] = None,
        output_config=None,
        frame_range: Optional[tuple[int, int]] = None,
    ) -> list:
        """Parallel frame pipeline: reader thread -> N workers -> writer thread."""
        from .core import FrameResult, OutputConfig

        N = len(engines)
        logger.info("Starting parallel inference with %d engines", N)

        t_start = time.monotonic()
        dirs = ensure_output_dirs(clip.output_dir)
        cfg = output_config or OutputConfig()
        self._write_manifest(dirs["root"], cfg, params)

        if clip.input_asset.asset_type == "sequence" and clip.alpha_asset.asset_type == "sequence":
            num_frames = clip.input_asset.frame_count
            if clip.input_asset.frame_count != clip.alpha_asset.frame_count:
                logger.warning(
                    "Clip '%s': sequence alpha count mismatch — input has %d, alpha has %d.",
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

        if frame_range is not None:
            range_start = max(0, frame_range[0])
            range_end = min(num_frames - 1, frame_range[1])
            frame_indices = list(range(range_start, range_end + 1))
            range_count = range_end - range_start + 1
        else:
            frame_indices = list(range(num_frames))
            range_count = num_frames

        skip_stems = skip_stems or set()

        # Bounded queues for pipeline stages
        in_q: queue.Queue = queue.Queue(maxsize=2 * N)
        out_q: queue.Queue = queue.Queue(maxsize=2 * N)
        stop = threading.Event()
        error_box: list = [None]

        if on_status:
            on_status("Compiling (first frame may take a minute)...")
        warmup_done = threading.Event()
        # t_steady marks when the first frame finishes (after Triton JIT).
        # fps/ETA are computed from this point, not t_start, so the one-time
        # compilation cost doesn't pollute the displayed rate.
        t_steady: list[float | None] = [None]

        # --- Worker threads (each bound to one engine) ---
        def worker(engine_idx: int) -> None:
            eng = engines[engine_idx]
            while not stop.is_set():
                try:
                    item = in_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:
                    break  # poison pill
                frame_idx, img, mask, stem, is_linear = item
                try:
                    res = eng.process_frame(
                        img,
                        mask,
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
                    out_q.put((frame_idx, stem, res, None))
                    if not warmup_done.is_set():
                        warmup_done.set()
                        t_steady[0] = time.monotonic()
                        if on_status:
                            on_status("")
                except Exception as e:
                    out_q.put((frame_idx, stem, None, e))
                    stop.set()
                    break

        # --- Reader thread ---
        def reader() -> None:
            input_cap = None
            alpha_cap = None
            input_files: list[str] = []
            alpha_files: list[str] = []

            try:
                if clip.input_asset.asset_type == "video":
                    input_cap = cv2.VideoCapture(clip.input_asset.path)
                else:
                    input_files = clip.input_asset.get_frame_files()

                if clip.alpha_asset.asset_type == "video":
                    alpha_cap = cv2.VideoCapture(clip.alpha_asset.path)
                else:
                    alpha_files = clip.alpha_asset.get_frame_files()
                alpha_stem_lookup = (
                    {os.path.splitext(fname)[0]: fname for fname in alpha_files}
                    if alpha_files
                    else None
                )

                for i in frame_indices:
                    if stop.is_set() or (job and job.is_cancelled):
                        break

                    img, input_stem, is_linear = self._read_input_frame(
                        clip,
                        i,
                        input_files,
                        input_cap,
                        params.input_is_linear,
                    )
                    if img is None:
                        out_q.put(
                            (i, f"{i:05d}", None, FrameReadError(clip.name, i, "video read failed"))
                        )
                        continue

                    if input_stem in skip_stems:
                        out_q.put((i, input_stem, "SKIP", None))
                        continue

                    mask = self._read_alpha_frame(
                        clip,
                        i,
                        alpha_files,
                        alpha_cap,
                        input_stem=input_stem,
                        alpha_stem_lookup=alpha_stem_lookup,
                    )
                    if mask is None:
                        out_q.put(
                            (i, input_stem, None, FrameReadError(clip.name, i, "alpha read failed"))
                        )
                        continue

                    if mask.shape[:2] != img.shape[:2]:
                        mask = cv2.resize(
                            mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
                        )

                    in_q.put((i, img, mask, input_stem, is_linear))
            finally:
                for _ in range(N):
                    try:
                        in_q.put(None, timeout=5)
                    except queue.Full:
                        pass
                if input_cap:
                    input_cap.release()
                if alpha_cap:
                    alpha_cap.release()

        # --- Writer thread ---
        results: list[FrameResult] = []
        skipped: list[int] = []
        processed_count_box = [0]

        def writer() -> None:
            reorder: dict = {}
            next_idx = frame_indices[0] if frame_indices else 0
            written = 0
            received = 0
            t_last_progress = time.monotonic()

            while received < range_count:
                if stop.is_set() and not reorder:
                    break
                try:
                    frame_idx, stem, result, err = out_q.get(timeout=1.0)
                except queue.Empty:
                    if stop.is_set():
                        break
                    continue
                received += 1

                if err is not None:
                    if isinstance(err, (FrameReadError, WriteFailureError)):
                        skipped.append(frame_idx)
                        results.append(FrameResult(frame_idx, stem, False, str(err)))
                        if on_warning:
                            on_warning(str(err))
                        next_idx = max(next_idx, frame_idx + 1)
                        continue
                    error_box[0] = err
                    stop.set()
                    break

                if result == "SKIP":
                    results.append(FrameResult(frame_idx, stem, True, "resumed (skipped)"))
                    next_idx = max(next_idx, frame_idx + 1)
                    continue

                reorder[frame_idx] = (result, stem)

                while next_idx in reorder:
                    res, s = reorder.pop(next_idx)
                    self._write_outputs(res, dirs, s, clip.name, next_idx, cfg)
                    processed_count_box[0] += 1
                    written += 1
                    results.append(FrameResult(next_idx, s, True))

                    now = time.monotonic()
                    if on_progress and (now - t_last_progress > 0.1 or written == range_count):
                        elapsed = now - t_start
                        timing_kw: dict[str, float] = {"elapsed": elapsed}
                        # Use t_steady (after first frame) for fps/ETA so the
                        # one-time Triton JIT cost doesn't drag down the rate.
                        t_fps_base = t_steady[0] or t_start
                        steady_elapsed = now - t_fps_base
                        # Frames completed after warmup (first frame is warmup)
                        steady_count = max(0, processed_count_box[0] - 1)
                        if steady_count > 0 and steady_elapsed > 0:
                            fps = steady_count / steady_elapsed
                            remaining = range_count - written
                            timing_kw["fps"] = fps
                            if fps > 0:
                                timing_kw["eta_seconds"] = remaining / fps
                        on_progress(clip.name, written, range_count, **timing_kw)
                        t_last_progress = now

                    next_idx += 1

            if on_progress:
                final_elapsed = time.monotonic() - t_start
                final_kw: dict[str, float] = {"elapsed": final_elapsed, "eta_seconds": 0.0}
                t_fps_base = t_steady[0] or t_start
                steady_elapsed = time.monotonic() - t_fps_base
                steady_count = max(0, processed_count_box[0] - 1)
                if steady_count > 0 and steady_elapsed > 0:
                    final_kw["fps"] = steady_count / steady_elapsed
                on_progress(clip.name, range_count, range_count, **final_kw)

        # Launch all threads
        threads = (
            [threading.Thread(target=reader, name="ck-reader")]
            + [threading.Thread(target=worker, args=(i,), name=f"ck-worker-{i}") for i in range(N)]
            + [threading.Thread(target=writer, name="ck-writer")]
        )
        for t in threads:
            t.start()

        for t in threads:
            while t.is_alive():
                t.join(timeout=0.5)
                if job and job.is_cancelled and not stop.is_set():
                    stop.set()

        if error_box[0]:
            raise error_box[0]
        if job and job.is_cancelled:
            raise JobCancelledError(clip.name, 0)

        return self._finalize_inference(
            clip,
            results,
            skipped,
            processed_count_box[0],
            t_start,
            frame_range,
            num_frames,
            on_warning=on_warning,
        )

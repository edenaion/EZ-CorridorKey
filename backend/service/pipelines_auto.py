"""Auto-alpha pipelines — GVM and BiRefNet (no annotations required)."""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional

from ..clip_state import ClipAsset, ClipEntry, ClipState
from ..errors import (
    CorridorKeyError,
    GPURequiredError,
    JobCancelledError,
)
from .model_manager import _ActiveModel

logger = logging.getLogger(__name__)


class AutoPipelinesMixin:
    """Mixin providing auto-alpha pipelines (GVM, BiRefNet) for CorridorKeyService."""

    def run_gvm(
        self,
        clip: ClipEntry,
        job=None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Run GVM auto alpha generation for a clip.

        Transitions clip: RAW -> READY (creates AlphaHint directory).
        """
        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for GVM")

        if self._device == 'cpu':
            raise GPURequiredError("GVM Auto Alpha")

        t_start = time.monotonic()

        logger.info("run_gvm: waiting for _gpu_lock")
        with self._gpu_lock:
            logger.info("run_gvm: acquired _gpu_lock")
            gvm = self._get_gvm()

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)

        if on_progress:
            on_progress(clip.name, 0, 1)

        # Check cancel before starting
        if job and job.is_cancelled:
            raise JobCancelledError(clip.name, 0)

        # Per-batch progress callback — GVM iterates over frames internally
        def _gvm_progress(batch_idx: int, total_batches: int) -> None:
            if on_progress:
                on_progress(clip.name, batch_idx, total_batches)
            # Check cancel between batches
            if job and job.is_cancelled:
                raise JobCancelledError(clip.name, batch_idx)

        try:
            gvm.process_sequence(
                input_path=clip.input_asset.path,
                output_dir=clip.root_path,
                num_frames_per_batch=1,
                decode_chunk_size=1,
                denoise_steps=1,
                mode='matte',
                write_video=False,
                direct_output_dir=alpha_dir,
                progress_callback=_gvm_progress,
            )
        except JobCancelledError:
            raise
        except Exception as e:
            if job and job.is_cancelled:
                raise JobCancelledError(clip.name, 0)
            raise CorridorKeyError(f"GVM failed for '{clip.name}': {e}") from e

        # Refresh alpha asset
        clip.alpha_asset = ClipAsset(alpha_dir, 'sequence')

        if on_progress:
            on_progress(clip.name, 1, 1)

        # Transition RAW -> READY
        try:
            clip.transition_to(ClipState.READY)
        except Exception as e:
            if on_warning:
                on_warning(f"State transition after GVM: {e}")

        logger.info(f"GVM complete for '{clip.name}': {clip.alpha_asset.frame_count} alpha frames in {time.monotonic() - t_start:.1f}s")

    def run_birefnet(
        self,
        clip: ClipEntry,
        usage: str = "Matting",
        job=None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Run BiRefNet automatic alpha generation for a clip.

        Fully automatic — no painting/annotation required.
        Transitions clip: RAW -> READY (creates AlphaHint directory).
        """
        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for BiRefNet")

        if self._device == 'cpu':
            raise GPURequiredError("BiRefNet")

        def _status(msg: str) -> None:
            logger.info(f"BiRefNet [{clip.name}]: {msg}")
            if on_status:
                on_status(msg)

        def _check_cancel() -> bool:
            return bool(job and job.is_cancelled)

        t_start = time.monotonic()

        # Phase 1: Load model
        _status(f"Loading BiRefNet ({usage})...")
        with self._gpu_lock:
            processor = self._get_birefnet(usage=usage, on_status=on_status)
        if _check_cancel():
            raise JobCancelledError(clip.name, 0)

        # Phase 2: Stream input frames
        #
        # Issue #95: a 109k-frame 4K UHD EXR clip OOMs if we materialize
        # every frame into a Python list up front (~2.6 TiB of RAM).
        # Instead, precompute everything that's cheap (filenames,
        # stems, count) and stream actual pixel data through a
        # generator so only one frame lives in RAM at a time.
        _status("Loading frames...")
        selected_input_names = self._selected_sequence_files(clip)
        if not selected_input_names:
            raise CorridorKeyError(f"Clip '{clip.name}' has no input frames for BiRefNet")

        frame_stems = [os.path.splitext(fname)[0] for fname in selected_input_names]
        num_frames = len(selected_input_names)

        # Deliberately pass on_status=None to the generator: in the
        # streaming path the wrapper owns the phase text ("Running
        # BiRefNet inference...") and the progress bar tracks frame
        # counts via progress_callback. If the generator also fired
        # "Loading frames (N/total)..." every 20 frames, it would
        # overwrite the wrapper's phase text and make the status bar
        # flip-flop once per chunk on long clips.
        frame_iter = (
            frame
            for _, frame in self._iter_named_sequence_frames(
                clip.input_asset,
                selected_input_names,
                clip.name,
                job=job,
                on_status=None,
            )
        )

        if _check_cancel():
            raise JobCancelledError(clip.name, 0)

        # Phase 3: Inference
        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        try:
            frames_written = processor.process_frames(
                input_frames=frame_iter,
                output_dir=alpha_dir,
                frame_names=frame_stems,
                progress_callback=on_progress,
                on_status=on_status,
                cancel_check=_check_cancel,
                clip_name=clip.name,
                num_frames=num_frames,
            )
        except Exception as e:
            if "CUDA out of memory" in str(e) or "OutOfMemoryError" in type(e).__name__:
                logger.error(f"BiRefNet OOM for '{clip.name}': {e}")
                self._birefnet_processor = None
                self._active_model = _ActiveModel.NONE
                try:
                    import torch as _torch
                    _torch.cuda.empty_cache()
                except Exception:
                    pass
                raise CorridorKeyError(
                    f"BiRefNet ran out of GPU memory processing '{clip.name}'. "
                    f"Try a lighter model variant (e.g. 'General Lite') or close other GPU applications."
                ) from e
            raise

        # Phase 4: Finalize
        clip.alpha_asset = ClipAsset(alpha_dir, 'sequence')

        try:
            clip.transition_to(ClipState.READY)
        except Exception as e:
            if on_warning:
                on_warning(f"State transition after BiRefNet: {e}")

        logger.info(
            f"BiRefNet complete for '{clip.name}': "
            f"{frames_written} alpha frames in {time.monotonic() - t_start:.1f}s"
        )

    def run_chroma_key(
        self,
        clip: ClipEntry,
        chroma_params: dict,
        job=None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Generate alpha hints via chroma key color-difference method.

        Pipelined: prefetch reads in a ThreadPool, process on GPU, write
        output PNGs in a ThreadPool. Disk I/O overlaps with GPU compute
        so throughput is limited by the slowest single stage, not the sum.
        Transitions clip: RAW -> READY (creates AlphaHint directory).

        chroma_params keys:
            screen_color: (r, g, b) tuple 0-255 or None
            screen_type: "green" or "blue"
            strength: float
            clip_black: float
            clip_white: float
            shrink_grow: int (pixels)
            edge_blur: int (pixels)
        """
        import cv2 as _cv2
        import numpy as _np
        from concurrent.futures import ThreadPoolExecutor, Future
        from CorridorKeyModule.core.chroma_key import chroma_key_matte

        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for chroma key")

        t_start = time.monotonic()

        selected_input_names = self._selected_sequence_files(clip)
        if not selected_input_names:
            raise CorridorKeyError(f"Clip '{clip.name}' has no input frames for chroma key")

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)

        num_frames = len(selected_input_names)
        if on_progress:
            on_progress(clip.name, 0, num_frames)

        input_dir = clip.input_asset.path
        ck_params = {
            "screen_color": chroma_params.get("screen_color"),
            "screen_type": chroma_params.get("screen_type", "green"),
            "strength": chroma_params.get("strength", 1.0),
            "clip_black": chroma_params.get("clip_black", 0.0),
            "clip_white": chroma_params.get("clip_white", 1.0),
            "shrink_grow": chroma_params.get("shrink_grow", 0),
            "edge_blur": chroma_params.get("edge_blur", 0),
        }

        # Load holdout mask (static per-clip, rasterized once we know dimensions)
        from ui.widgets.annotation_overlay import AnnotationModel
        _holdout_model = AnnotationModel(filename="holdout_strokes.json")
        _holdout_model.load(clip.root_path)
        _holdout_needs_raster = _holdout_model.has_annotations(0)
        _holdout_mask = None

        # ── Read helper (runs in read ThreadPool, releases GIL via cv2) ──
        def _read_frame(fname: str) -> tuple[str, _np.ndarray | None]:
            fpath = os.path.join(input_dir, fname)
            is_exr = fname.lower().endswith(".exr")
            if is_exr:
                bgr = _cv2.imread(fpath, _cv2.IMREAD_ANYCOLOR | _cv2.IMREAD_ANYDEPTH)
            else:
                bgr = _cv2.imread(fpath, _cv2.IMREAD_COLOR)
            if bgr is None:
                logger.warning(f"Chroma key: could not read {fpath}, skipping")
                return fname, None
            rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
            if rgb.dtype != _np.uint8:
                rgb = (_np.clip(rgb, 0.0, 1.0) * 255.0).astype(_np.uint8)
            return fname, rgb

        # ── Write helper (runs in write ThreadPool) ──
        def _write_matte(out_path: str, matte: _np.ndarray) -> None:
            _cv2.imwrite(out_path, matte)

        _PREFETCH = 4
        _WRITE_WORKERS = 4

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ck-read") as read_pool, \
             ThreadPoolExecutor(max_workers=_WRITE_WORKERS, thread_name_prefix="ck-write") as write_pool:

            # Submit first batch of reads
            read_futures: list[Future] = []
            for fname in selected_input_names[:_PREFETCH]:
                read_futures.append(read_pool.submit(_read_frame, fname))

            write_futures: list[Future] = []
            next_read_idx = _PREFETCH

            for i in range(num_frames):
                if job and job.is_cancelled:
                    raise JobCancelledError(clip.name, i)

                # Wait for next pre-read frame
                fname, frame_rgb = read_futures[i].result()

                # Submit next read (keep prefetch queue full)
                if next_read_idx < num_frames:
                    read_futures.append(
                        read_pool.submit(_read_frame, selected_input_names[next_read_idx])
                    )
                    next_read_idx += 1

                if frame_rgb is None:
                    continue

                # Rasterize holdout mask on first frame (need dimensions)
                if _holdout_needs_raster and _holdout_mask is None:
                    h, w = frame_rgb.shape[:2]
                    _holdout_mask = AnnotationModel.rasterize_holdout_mask(
                        _holdout_model.get_strokes(0), w, h
                    )
                    _holdout_needs_raster = False

                # GPU chroma key
                matte = chroma_key_matte(frame_rgb, holdout_mask=_holdout_mask, **ck_params)

                # Submit async write
                stem = os.path.splitext(fname)[0]
                out_path = os.path.join(alpha_dir, f"{stem}.png")
                write_futures.append(write_pool.submit(_write_matte, out_path, matte))

                # Drain completed writes to prevent unbounded memory growth
                if len(write_futures) > _WRITE_WORKERS * 2:
                    write_futures = [f for f in write_futures if not f.done()]

                if on_progress:
                    elapsed = time.monotonic() - t_start
                    fps = (i + 1) / elapsed if elapsed > 0 else 0
                    on_progress(clip.name, i + 1, num_frames, fps=fps)

            # Wait for all writes to finish
            for f in write_futures:
                f.result()

        clip.alpha_asset = ClipAsset(alpha_dir, 'sequence')

        try:
            clip.transition_to(ClipState.READY)
        except Exception as e:
            if on_warning:
                on_warning(f"State transition after chroma key: {e}")

        logger.info(
            f"Chroma key complete for '{clip.name}': "
            f"{num_frames} alpha frames in {time.monotonic() - t_start:.1f}s"
        )

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

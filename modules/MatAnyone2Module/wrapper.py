"""MatAnyone2 wrapper — adapts the vendored inference pipeline to CorridorKey's
alpha hint generator interface.

Usage:
    processor = MatAnyone2Processor(device='cuda')
    processor.process_frames(
        input_frames=[...],   # list of uint8 RGB numpy arrays (H,W,3)
        mask_frame=mask,      # uint8 grayscale (H,W) — first-frame segmentation mask
        output_dir="path/to/AlphaHint",
        frame_names=["frame_000000", ...],
        progress_callback=fn,
        on_status=fn,
        cancel_check=fn,
    )
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Callable, List, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Checkpoint URL for auto-download
_CKPT_URL = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
_CKPT_NAME = "matanyone2.pth"

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_CHECKPOINT_DIR = os.path.join(_MODULE_DIR, "checkpoints")


def _candidate_checkpoint_dirs() -> List[str]:
    """Return MatAnyone2 checkpoint directories to search, in priority order.

    Order:
      1. ``<data_dir>/modules/MatAnyone2Module/checkpoints`` — writable
         on every platform. ``scripts/setup_models.py`` writes here in
         frozen builds and the lazy runtime download targets here too.
         This is the only path that works on installed macOS builds
         because ``/Applications/EZ-CorridorKey.app/Contents/...`` is
         read-only for non-admin users.
      2. ``_BUNDLED_CHECKPOINT_DIR`` — the bundled directory beside
         this file. In dev mode this is the repo's checkpoints folder;
         in frozen builds this lives inside the bundle and is
         read-only on macOS, but we keep it as a fallback so existing
         manual/dev installs still work.
      3. ``<project_root>/pretrained_models`` — legacy fallback from
         the original MatAnyone2 repo layout.
    """
    dirs: List[str] = []
    try:
        from backend.project import get_data_dir  # Lazy import: avoid cycles
        data_dir = get_data_dir()
        if data_dir:
            dirs.append(
                os.path.join(
                    data_dir, "modules", "MatAnyone2Module", "checkpoints"
                )
            )
    except Exception:
        pass
    if _BUNDLED_CHECKPOINT_DIR not in dirs:
        dirs.append(_BUNDLED_CHECKPOINT_DIR)
    legacy = os.path.join(
        os.path.dirname(os.path.dirname(_MODULE_DIR)), "pretrained_models"
    )
    if legacy not in dirs:
        dirs.append(legacy)
    return dirs


def _ensure_checkpoint(ckpt_path: str | None = None) -> str:
    """Return path to matanyone2.pth, downloading if needed.

    Resolution order:
        1. Explicit ``ckpt_path`` if provided and exists.
        2. The first candidate dir (see ``_candidate_checkpoint_dirs``)
           that already contains ``matanyone2.pth``.
        3. Download to the first candidate dir — always the writable
           data-dir location — so a cold download never targets a
           read-only bundle path.
    """
    if ckpt_path and os.path.isfile(ckpt_path):
        return ckpt_path

    candidates = _candidate_checkpoint_dirs()
    for base in candidates:
        local_path = os.path.join(base, _CKPT_NAME)
        if os.path.isfile(local_path):
            return local_path

    # Nothing found — download to the writable data-dir target.
    download_dir = candidates[0]
    local_path = os.path.join(download_dir, _CKPT_NAME)
    logger.info(
        "MatAnyone2 checkpoint not found, downloading to %s", local_path
    )
    os.makedirs(download_dir, exist_ok=True)
    from torch.hub import download_url_to_file
    download_url_to_file(_CKPT_URL, local_path)
    logger.info(f"Downloaded checkpoint to {local_path}")
    return local_path


class MatAnyone2Processor:
    """Wraps MatAnyone2's InferenceCore for CorridorKey integration.

    Follows the same pattern as GVMProcessor and VideoMaMa pipeline:
    - Lazy model load
    - Process frames → write alpha PNGs
    - Support progress callbacks and cancel checks
    """

    def __init__(
        self,
        device: str = "cuda",
        ckpt_path: str | None = None,
        n_warmup: int = 10,
        r_erode: int = 10,
        r_dilate: int = 10,
        max_internal_size: int = 1080,
    ):
        self._device = device
        self._ckpt_path = ckpt_path
        self._n_warmup = n_warmup
        self._r_erode = r_erode
        self._r_dilate = r_dilate
        self._max_internal_size = max_internal_size
        self._processor = None  # InferenceCore, loaded lazily
        self._model = None      # MatAnyone2 nn.Module

    def _ensure_loaded(self, on_status: Optional[Callable[[str], None]] = None):
        """Load model + InferenceCore if not already loaded."""
        if self._processor is not None:
            return

        if on_status:
            on_status("Loading MatAnyone2 checkpoint...")

        # Ensure matanyone2 package is importable
        module_dir = os.path.dirname(os.path.abspath(__file__))
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        ckpt_path = _ensure_checkpoint(self._ckpt_path)

        from matanyone2.utils.get_default_model import get_matanyone2_model
        from matanyone2.inference.inference_core import InferenceCore

        # Enable TF32 for Ampere+ GPUs (RTX 30xx/40xx/50xx) — ~15-30% speedup
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        t0 = time.monotonic()
        self._model = get_matanyone2_model(ckpt_path, device=self._device)
        self._processor = InferenceCore(self._model, cfg=self._model.cfg)

        # Override max_internal_size so we don't process at full 4K resolution.
        # Alpha hints are guides — they get resized during CorridorKey inference anyway.
        # Default 1080 = ~4x fewer pixels than 4K = ~4x faster.
        self._processor.max_internal_size = self._max_internal_size
        logger.info(
            f"MatAnyone2 max_internal_size set to {self._max_internal_size} "
            f"(InferenceCore will downscale inputs above this shortest-side)"
        )

        # Clear Hydra global state so it doesn't poison other modules (e.g. SAM2)
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()

        logger.info(f"MatAnyone2 loaded in {time.monotonic() - t0:.1f}s")

        if on_status:
            on_status("MatAnyone2 model ready")

    def to(self, device: str):
        """Move model to device (for VRAM management compatibility)."""
        if self._model is not None:
            self._model.to(device)
        self._device = device

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def process_frames(
        self,
        input_frames: list[np.ndarray],
        mask_frame: np.ndarray,
        output_dir: str,
        frame_names: list[str],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        clip_name: str = "",
    ) -> int:
        """Process a sequence of frames and write alpha PNGs.

        Args:
            input_frames: List of uint8 RGB numpy arrays (H,W,3).
            mask_frame: First-frame segmentation mask, uint8 grayscale (H,W).
                        White (255) = foreground, black (0) = background.
            output_dir: Directory to write alpha hint PNGs.
            frame_names: Output filenames (without extension), one per input frame.
            progress_callback: Called as (clip_name, frames_done, total_frames).
            on_status: Phase status callback.
            cancel_check: Returns True if job should be cancelled.
            clip_name: For progress callback identification.

        Returns:
            Number of alpha frames written.

        Raises:
            RuntimeError: If mask_frame doesn't match first frame dimensions.
        """
        import traceback
        logger.info(
            "MatAnyone2 process_frames CALLED: clip=%s, num_frames=%d\n%s",
            clip_name, len(input_frames), "".join(traceback.format_stack()[-5:])
        )
        self._ensure_loaded(on_status=on_status)

        # Clear memory from any previous sequence to prevent state bleed
        self.clear()

        num_frames = len(input_frames)
        if num_frames == 0:
            return 0

        # Validate mask dimensions match first frame
        h, w = input_frames[0].shape[:2]
        mh, mw = mask_frame.shape[:2]
        if (mh, mw) != (h, w):
            raise RuntimeError(
                f"Mask dimensions ({mw}x{mh}) don't match frame dimensions ({w}x{h}). "
                f"MatAnyone2 requires a mask that matches the input frame size."
            )

        n_warmup = self._n_warmup
        r_erode = self._r_erode
        r_dilate = self._r_dilate

        # Prepare mask: apply morphological operations
        mask_np = mask_frame.astype(np.float32)
        if r_dilate > 0:
            from matanyone2.utils.inference_utils import gen_dilate
            mask_np = gen_dilate(mask_np, r_dilate, r_dilate)
        if r_erode > 0:
            from matanyone2.utils.inference_utils import gen_erosion
            mask_np = gen_erosion(mask_np, r_erode, r_erode)

        mask_tensor = torch.from_numpy(mask_np).float().to(self._device)

        # Stream frames one at a time to avoid loading all into RAM.
        # At 4K (2026x3840), 470 frames as float32 tensors = ~44 GB RAM.
        # Precompute first frame tensor for warmup reuse.
        first_frame_tensor = torch.from_numpy(input_frames[0]).permute(2, 0, 1).float()  # (3,H,W)
        total_length = num_frames + n_warmup

        # Use temp dir for atomic output
        tmp_dir = output_dir + "._matanyone2_tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        if on_status:
            on_status("Running MatAnyone2 inference...")

        objects = [1]
        frames_written = 0
        _t_frame = time.monotonic()

        try:
            for ti in range(total_length):
                # Cancel check
                if cancel_check and cancel_check():
                    logger.info(f"MatAnyone2 cancelled at frame {ti}")
                    raise _CancelledError()

                # Stream: warmup frames reuse first frame, real frames read on demand
                if ti < n_warmup:
                    image = first_frame_tensor
                else:
                    image = torch.from_numpy(input_frames[ti - n_warmup]).permute(2, 0, 1).float()
                image_input = (image / 255.0).float().to(self._device)

                if ti == 0:
                    # Encode the given mask
                    self._processor.step(image_input, mask_tensor, objects=objects)
                    # First frame prediction
                    output_prob = self._processor.step(image_input, first_frame_pred=True)
                elif ti <= n_warmup:
                    output_prob = self._processor.step(image_input, first_frame_pred=True)
                else:
                    output_prob = self._processor.step(image_input)

                # Convert probability to alpha matte
                mask_tensor = self._processor.output_prob_to_mask(output_prob)

                # Skip warmup frames — only write real frames
                if ti >= n_warmup:
                    pha = mask_tensor.unsqueeze(2).cpu().numpy()
                    pha_u8 = np.round(np.clip(pha * 255.0, 0, 255)).astype(np.uint8)

                    # Write to temp dir
                    frame_idx = ti - n_warmup
                    if frame_idx < len(frame_names):
                        out_name = f"{frame_names[frame_idx]}.png"
                    else:
                        out_name = f"frame_{frame_idx:06d}.png"

                    out_path = os.path.join(tmp_dir, out_name)
                    if not cv2.imwrite(out_path, pha_u8):
                        raise RuntimeError(f"Failed to write alpha frame: {out_path}")

                    frames_written += 1

                    # Per-frame timing diagnostic (every 50 frames)
                    if frames_written % 50 == 0 or frames_written == 1:
                        _elapsed = time.monotonic() - _t_frame
                        logger.info(
                            "MatAnyone2 DIAG: frame %d/%d, ti=%d, "
                            "last_50_elapsed=%.1fs (%.2fs/frame), "
                            "curr_ti=%d, last_mem_ti=%d",
                            frames_written, num_frames, ti,
                            _elapsed, _elapsed / min(frames_written, 50),
                            self._processor.curr_ti,
                            self._processor.last_mem_ti,
                        )
                        _t_frame = time.monotonic()

                    if progress_callback:
                        progress_callback(clip_name, frames_written, num_frames)

            # Atomic commit: move temp files to final output dir
            if on_status:
                on_status("Finalizing alpha hints...")
            os.makedirs(output_dir, exist_ok=True)
            for fname in os.listdir(tmp_dir):
                src = os.path.join(tmp_dir, fname)
                dst = os.path.join(output_dir, fname)
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)

        except _CancelledError:
            # Clean up temp dir on cancel
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        except Exception:
            # Clean up temp dir on error
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        finally:
            # Always try to clean up temp dir
            if os.path.isdir(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info(f"MatAnyone2 process_frames COMPLETE: wrote {frames_written} alpha frames to {output_dir}")
        return frames_written

    def clear(self):
        """Reset internal state (memory) for a new sequence."""
        if self._processor is not None:
            self._processor.clear_memory()


class _CancelledError(Exception):
    """Internal: raised when cancel_check returns True."""
    pass

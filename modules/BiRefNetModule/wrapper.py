"""BiRefNet wrapper — bilateral reference network for salient object segmentation.

Generates alpha hint mattes without requiring user annotation (automatic, like GVM).

Usage:
    handler = BiRefNetHandler(device='cuda')
    handler.process_frames(
        input_frames=[...],   # list of uint8 RGB numpy arrays (H,W,3)
        output_dir="path/to/AlphaHint",
        frame_names=["frame_000000", ...],
        progress_callback=fn,
        on_status=fn,
        cancel_check=fn,
    )
"""
from __future__ import annotations

import gc
import logging
import os
import shutil
import time
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)

# HuggingFace model IDs keyed by variant name.
_MODEL_IDS = {
    "General": "ZhengPeng7/BiRefNet",
    "General-Lite": "ZhengPeng7/BiRefNet_lite",
    "Matting": "ZhengPeng7/BiRefNet-matting",
    "Portrait": "ZhengPeng7/BiRefNet-portrait",
    "DIS": "ZhengPeng7/BiRefNet-DIS5K",
    "HRSOD": "ZhengPeng7/BiRefNet-HRSOD",
    "COD": "ZhengPeng7/BiRefNet-COD",
}


class BiRefNetHandler:
    """Wraps BiRefNet for CorridorKey alpha hint generation.

    Follows the same pattern as MatAnyone2Processor:
    - Lazy model load
    - Process frames -> write alpha PNGs
    - Support progress callbacks and cancel checks
    """

    def __init__(
        self,
        device: str = "cuda",
        model_variant: str = "General",
    ):
        self._device = device
        self._model_variant = model_variant
        self._model = None
        self._transform = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _ensure_loaded(self, on_status: Optional[Callable[[str], None]] = None):
        """Load BiRefNet model from HuggingFace if not already loaded."""
        if self._model is not None:
            return

        model_id = _MODEL_IDS.get(self._model_variant)
        if model_id is None:
            raise ValueError(
                f"Unknown BiRefNet variant '{self._model_variant}'. "
                f"Available: {', '.join(_MODEL_IDS)}"
            )

        if on_status:
            on_status(f"Loading BiRefNet ({self._model_variant})...")

        t0 = time.monotonic()
        from transformers import AutoModelForImageSegmentation

        self._model = AutoModelForImageSegmentation.from_pretrained(
            model_id, trust_remote_code=True,
        )
        self._model.to(self._device)
        self._model.eval()

        logger.info(
            f"BiRefNet ({self._model_variant}) loaded from {model_id} "
            f"in {time.monotonic() - t0:.1f}s"
        )

        if on_status:
            on_status("BiRefNet model ready")

    def to(self, device: str):
        """Move model to device (for VRAM management compatibility)."""
        if self._model is not None:
            self._model.to(device)
        self._device = device

    @torch.inference_mode()
    def process_frames(
        self,
        input_frames: list[np.ndarray],
        output_dir: str,
        frame_names: list[str],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        clip_name: str = "",
    ) -> int:
        """Process frames and write alpha PNGs.

        Args:
            input_frames: List of uint8 RGB numpy arrays (H,W,3).
            output_dir: Directory to write alpha hint PNGs.
            frame_names: Output filenames (without extension).
            progress_callback: Called as (clip_name, frames_done, total_frames).
            on_status: Phase status callback.
            cancel_check: Returns True if job should be cancelled.
            clip_name: For progress callback identification.

        Returns:
            Number of alpha frames written.
        """
        self._ensure_loaded(on_status=on_status)

        num_frames = len(input_frames)
        if num_frames == 0:
            return 0

        # Use temp dir for atomic output
        tmp_dir = output_dir + "._birefnet_tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        if on_status:
            on_status("Running BiRefNet segmentation...")

        frames_written = 0

        try:
            for i, frame in enumerate(input_frames):
                if cancel_check and cancel_check():
                    logger.info(f"BiRefNet cancelled at frame {i}")
                    raise _CancelledError()

                h, w = frame.shape[:2]

                # Preprocess: uint8 RGB -> float32 tensor, normalize
                img_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                img_t = self._transform(img_t).unsqueeze(0).to(self._device)

                # BiRefNet expects 1024x1024 input
                img_t = F.interpolate(img_t, size=(1024, 1024), mode="bilinear", align_corners=False)

                # Forward pass — autocast handles fp16/fp32 dtype matching
                with torch.autocast(device_type=self._device if isinstance(self._device, str) else self._device.type, dtype=torch.float16):
                    preds = self._model(img_t)[-1].sigmoid()

                # Resize back to original resolution
                pred = F.interpolate(preds, size=(h, w), mode="bilinear", align_corners=False)
                alpha = pred[0, 0].cpu().numpy()
                alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)

                # Write to temp dir
                out_name = f"{frame_names[i]}.png" if i < len(frame_names) else f"frame_{i:06d}.png"
                out_path = os.path.join(tmp_dir, out_name)
                if not cv2.imwrite(out_path, alpha_u8):
                    raise RuntimeError(f"Failed to write alpha frame: {out_path}")

                frames_written += 1

                if progress_callback:
                    progress_callback(clip_name, frames_written, num_frames)

            # Atomic commit
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
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        finally:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            # Free GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(
            f"BiRefNet process_frames COMPLETE: wrote {frames_written} "
            f"alpha frames to {output_dir}"
        )
        return frames_written

    def clear(self):
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class _CancelledError(Exception):
    """Internal: raised when cancel_check returns True."""
    pass

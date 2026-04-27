"""BiRefNet wrapper — bilateral reference network for automatic alpha hint generation.

Fully automatic (no painting/annotation required). Downloads the selected model
variant from HuggingFace on first use and caches it locally.

Usage:
    processor = BiRefNetProcessor(device='cuda', usage='Matting')
    processor.process_frames(
        input_frames=[...],       # list of uint8 RGB numpy arrays (H,W,3)
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
import time
from typing import Callable, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

_BUNDLED_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "checkpoints"
)


def _candidate_checkpoint_dirs() -> List[str]:
    """Return BiRefNet checkpoint directories to search, in priority order.

    Order:
      1. ``<data_dir>/modules/BiRefNetModule/checkpoints`` — writable on
         every platform. The setup wizard downloads here in frozen builds
         and the lazy runtime download in ``_ensure_loaded`` writes here
         too. This is the only path that works on installed macOS builds
         because ``/Applications/EZ-CorridorKey.app/Contents/...`` is
         read-only for non-admin users.
      2. ``_BUNDLED_CHECKPOINT_DIR`` — the directory next to this file.
         In dev mode this is ``modules/BiRefNetModule/checkpoints``. In
         frozen builds this lives inside the bundle and is read-only on
         macOS, but stays as a fallback so existing manual/dev installs
         keep working.
    """
    dirs: List[str] = []
    try:
        from backend.project import get_data_dir  # Lazy import: avoid cycles
        data_dir = get_data_dir()
        if data_dir:
            dirs.append(
                os.path.join(
                    data_dir, "modules", "BiRefNetModule", "checkpoints"
                )
            )
    except Exception:
        pass
    if _BUNDLED_CHECKPOINT_DIR not in dirs:
        dirs.append(_BUNDLED_CHECKPOINT_DIR)
    return dirs


def _resolve_model_dir(repo_name: str) -> Tuple[Optional[str], str]:
    """Locate an already-downloaded BiRefNet variant.

    Returns ``(existing_dir_or_None, preferred_download_dir)``. The
    preferred download dir is always the first candidate — the writable
    data-dir location — so a cold lazy download never targets the
    read-only bundled path.
    """
    candidates = _candidate_checkpoint_dirs()
    for base in candidates:
        candidate = os.path.join(base, repo_name)
        if os.path.isdir(candidate) and any(
            f.endswith((".safetensors", ".bin"))
            for f in os.listdir(candidate)
            if os.path.isfile(os.path.join(candidate, f))
        ):
            return candidate, os.path.join(candidates[0], repo_name)
    return None, os.path.join(candidates[0], repo_name)

# ── Model registry ──────────────────────────────────────────────────────────
# Display name → HuggingFace repo suffix under ZhengPeng7/
BIREFNET_MODELS: dict[str, str] = {
    "Matting": "BiRefNet-matting",
    "Matting HR": "BiRefNet_HR-Matting",
    "Matting Lite": "BiRefNet_lite-matting",
    "Matting Dynamic": "BiRefNet_dynamic-matting",
    "General": "BiRefNet",
    "General HR": "BiRefNet_HR",
    "General Lite": "BiRefNet_lite",
    "General Lite 2K": "BiRefNet_lite-2K",
    "General Dynamic": "BiRefNet_dynamic",
    "General 512": "BiRefNet_512x512",
    "Portrait": "BiRefNet-portrait",
    "DIS": "BiRefNet-DIS5K",
    "HRSOD": "BiRefNet-HRSOD",
    "COD": "BiRefNet-COD",
    "DIS TR_TEs": "BiRefNet-DIS5K-TR_TEs",
    "General Legacy": "BiRefNet-legacy",
}

DEFAULT_MODEL = "Matting"

# Resolution per model variant
_MODEL_RESOLUTIONS: dict[str, Tuple[int, int] | None] = {
    "BiRefNet_HR": (2048, 2048),
    "BiRefNet_HR-Matting": (2048, 2048),
    "BiRefNet_lite-2K": (2560, 1440),
    "BiRefNet_512x512": (512, 512),
    # Dynamic models: None = use input resolution (rounded to 32)
    "BiRefNet_dynamic": None,
    "BiRefNet_dynamic-matting": None,
}
_DEFAULT_RESOLUTION = (1024, 1024)


class _ImagePreprocessor:
    """Resize and normalize for BiRefNet inference."""

    def __init__(self, resolution: Tuple[int, int] = (1024, 1024)) -> None:
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


class BiRefNetProcessor:
    """Wraps BiRefNet for CorridorKey alpha hint generation.

    Follows the same pattern as MatAnyone2Processor:
    - Lazy model load with on_status callbacks
    - Process frames → write alpha PNGs
    - Support progress callbacks and cancel checks
    - Proper VRAM cleanup
    """

    def __init__(
        self,
        device: str = "cuda",
        usage: str = DEFAULT_MODEL,
    ):
        self._device = device
        self._usage = usage
        self._model = None
        self._resolution: Tuple[int, int] | None = None
        self._half = True  # FP16 by default

    def _ensure_loaded(self, on_status: Optional[Callable[[str], None]] = None):
        """Load model if not already loaded."""
        if self._model is not None:
            return

        repo_name = BIREFNET_MODELS.get(self._usage)
        if repo_name is None:
            raise ValueError(
                f"Unknown BiRefNet model '{self._usage}'. "
                f"Available: {list(BIREFNET_MODELS.keys())}"
            )

        repo_id = f"ZhengPeng7/{repo_name}"

        # Look for an existing checkpoint in the data-dir location first,
        # then fall back to the bundled directory (dev/manual installs).
        # If nothing exists, ``download_target`` is the writable data-dir
        # path — never the read-only bundled path, which is what caused
        # the "Permission denied" crash on installed macOS builds where
        # the module lives inside /Applications/*.app/Contents/...
        existing, download_target = _resolve_model_dir(repo_name)
        if existing is not None:
            model_local_dir = existing
            logger.info(f"BiRefNet model cached: {model_local_dir}")
        else:
            model_local_dir = download_target
            if on_status:
                on_status(f"Downloading {repo_name}...")
            logger.info(
                f"Downloading BiRefNet model: {repo_id} -> {model_local_dir}"
            )
            os.makedirs(model_local_dir, exist_ok=True)
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id,
                local_dir=model_local_dir,
                local_dir_use_symlinks=False,
            )

        if on_status:
            on_status(f"Loading BiRefNet ({self._usage})...")

        t0 = time.monotonic()

        # Enable TF32 for Ampere+ GPUs
        torch.set_float32_matmul_precision('high')

        from transformers import AutoModelForImageSegmentation
        self._model = AutoModelForImageSegmentation.from_pretrained(
            model_local_dir, trust_remote_code=True
        )
        self._model.to(self._device)
        self._model.eval()
        if self._half:
            self._model.half()

        # Set resolution based on model variant
        self._resolution = _MODEL_RESOLUTIONS.get(repo_name, _DEFAULT_RESOLUTION)

        logger.info(f"BiRefNet ({self._usage}) loaded in {time.monotonic() - t0:.1f}s")
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
        input_frames: Union[Iterable[np.ndarray], list[np.ndarray]],
        output_dir: str,
        frame_names: list[str],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        clip_name: str = "",
        dilate_radius: int = 0,
        num_frames: Optional[int] = None,
    ) -> int:
        """Process frames and write alpha hint PNGs.

        Args:
            input_frames: Iterable of uint8 RGB numpy arrays (H,W,3).
                Accepts both eager lists and lazy generators. Large
                clips should pass a generator — this loop only ever
                holds one frame in memory at a time, so RAM stays
                bounded regardless of clip length. This is the fix
                for issue #95 (109k-frame 4K UHD OOM).
            output_dir: Directory to write alpha hint PNGs.
            frame_names: Output filenames (without extension), one per input frame.
            progress_callback: Called as (clip_name, frames_done, total_frames).
            on_status: Phase status callback.
            cancel_check: Returns True if job should be cancelled.
            clip_name: For progress callback identification.
            dilate_radius: Mask expansion (>0) or contraction (<0). 0 = no change.
            num_frames: Explicit frame count for progress reporting when
                ``input_frames`` is a generator. If omitted, falls back
                to ``len(input_frames)`` which requires a sized iterable.

        Returns:
            Number of alpha frames written.
        """
        self._ensure_loaded(on_status=on_status)

        if num_frames is None:
            try:
                num_frames = len(input_frames)  # type: ignore[arg-type]
            except TypeError:
                # Generator / bare iterator — caller must pass num_frames
                # explicitly for a meaningful progress bar. Fall back to
                # len(frame_names) which is cheap and always known.
                num_frames = len(frame_names)
        if num_frames == 0:
            return 0

        if on_status:
            on_status("Running BiRefNet inference...")

        # Use temp dir for atomic output
        tmp_dir = output_dir + "._birefnet_tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        frames_written = 0
        _t_start = time.monotonic()

        try:
            for i, frame_rgb in enumerate(input_frames):
                # Cancel check
                if cancel_check and cancel_check():
                    logger.info(f"BiRefNet cancelled at frame {i}")
                    raise _CancelledError()

                # Convert numpy RGB to PIL
                pil_image = Image.fromarray(frame_rgb)

                # Handle dynamic resolution models
                resolution = self._resolution
                if resolution is None:
                    # Round to nearest multiple of 32
                    w, h = pil_image.size
                    resolution = (w // 32 * 32, h // 32 * 32)

                # Preprocess
                preprocessor = _ImagePreprocessor(resolution=resolution)
                image_tensor = preprocessor(pil_image).unsqueeze(0).to(self._device)
                if self._half:
                    image_tensor = image_tensor.half()

                # Inference
                preds = self._model(image_tensor)[-1].sigmoid().cpu()
                pred = preds[0].squeeze()

                # Convert to full-resolution alpha mask
                pred_pil = transforms.ToPILImage()(pred.float())
                target_size = (frame_rgb.shape[1], frame_rgb.shape[0])  # (W, H)
                mask = pred_pil.resize(target_size, Image.BILINEAR)
                mask_np = np.array(mask)  # uint8, 0-255, soft alpha

                # Optional dilation/erosion
                if dilate_radius != 0:
                    abs_radius = abs(dilate_radius)
                    k_size = abs_radius * 2 + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                    if dilate_radius > 0:
                        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                    else:
                        mask_np = cv2.erode(mask_np, kernel, iterations=1)

                # Write soft alpha (no binary threshold — let CorridorKey use the full gradient)
                if i < len(frame_names):
                    out_name = f"{frame_names[i]}.png"
                else:
                    out_name = f"frame_{i:06d}.png"

                out_path = os.path.join(tmp_dir, out_name)
                if not cv2.imwrite(out_path, mask_np):
                    raise RuntimeError(f"Failed to write alpha frame: {out_path}")

                frames_written += 1

                # Per-frame timing (every 50 frames)
                if frames_written % 50 == 0 or frames_written == 1:
                    elapsed = time.monotonic() - _t_start
                    logger.info(
                        "BiRefNet: frame %d/%d, %.2fs/frame",
                        frames_written, num_frames,
                        elapsed / frames_written,
                    )

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
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        except Exception:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        finally:
            if os.path.isdir(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

        elapsed = time.monotonic() - _t_start
        logger.info(
            f"BiRefNet complete: {frames_written} alpha frames to {output_dir} "
            f"in {elapsed:.1f}s ({elapsed / max(frames_written, 1):.2f}s/frame)"
        )
        return frames_written


class _CancelledError(Exception):
    """Internal: raised when cancel_check returns True."""
    pass

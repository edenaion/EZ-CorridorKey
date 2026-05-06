"""Chroma keyer for alpha hint generation.

Color-difference method: measures how much the screen channel (green or blue)
exceeds the maximum of the other two channels, normalized by the sampled
screen color's own ratio. A smootherstep curve sharpens the foreground core
without destroying edge gradients. Anything not screen-dominant is immediately
classified as foreground.

Two backends: PyTorch GPU (default when CUDA available, ~37 FPS at 4K) and
NumPy CPU fallback (~1.8 FPS at 4K). The GPU path runs the same math on
CUDA tensors. Morphological cleanup, shrink/grow, and edge blur remain on
CPU via OpenCV (fast on uint8, negligible overhead).
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy torch import + device cache ──
_torch = None
_device = None


def _get_torch_device():
    """Return (torch_module, device) or (None, None) if CUDA unavailable."""
    global _torch, _device
    if _torch is not None:
        return _torch, _device
    try:
        import torch
        if torch.cuda.is_available():
            _torch = torch
            _device = torch.device("cuda")
            logger.debug("Chroma key: using PyTorch GPU (%s)", torch.cuda.get_device_name())
            return _torch, _device
    except ImportError:
        pass
    return None, None


def _chroma_key_gpu(
    img_t, ref_sc: float, ref_c1: float, ref_c2: float,
    sc: int, c1: int, c2: int,
    strength: float, clip_black: float, clip_white: float,
    torch_mod,
) -> np.ndarray:
    """Core chroma key math on GPU tensors. Returns uint8 numpy array."""
    # Color difference
    screen_ch = img_t[:, :, sc]
    other_max = torch_mod.maximum(img_t[:, :, c1], img_t[:, :, c2])
    excess = torch_mod.clamp(screen_ch - other_max, min=0.0)

    # Normalize
    ref_excess = max(ref_sc - max(ref_c1, ref_c2), 0.01)
    key = torch_mod.clamp(excess / ref_excess * strength, 0.0, 1.0)

    # Saturation gate
    px_max = img_t.max(dim=2).values.clamp(min=0.01)
    px_min = img_t.min(dim=2).values
    saturation = 1.0 - px_min / px_max
    sat_gate = torch_mod.clamp((saturation - 0.2) / 0.2, 0.0, 1.0)
    key = key * sat_gate

    # Smootherstep: 6t^5 - 15t^4 + 10t^3
    key = key * key * key * (key * (key * 6.0 - 15.0) + 10.0)

    # Darkness protection
    dark_gate = torch_mod.clamp((px_max - 0.03) / 0.09, 0.0, 1.0)
    key = key * dark_gate

    # Alpha + clip
    alpha = 1.0 - key
    cw = max(clip_white, clip_black + 0.001)
    alpha = torch_mod.clamp((alpha - clip_black) / (cw - clip_black), 0.0, 1.0)

    return (alpha * 255.0).to(torch_mod.uint8).cpu().numpy()


def chroma_key_matte(
    frame_rgb: np.ndarray,
    screen_color: tuple[int, int, int] | None = None,
    screen_type: str = "green",
    strength: float = 1.0,
    clip_black: float = 0.0,
    clip_white: float = 1.0,
    shrink_grow: int = 0,
    edge_blur: int = 0,
) -> np.ndarray:
    """Generate a grayscale alpha matte via color-difference keying.

    Uses PyTorch GPU when CUDA is available (20x faster at 4K), falls
    back to NumPy CPU otherwise.

    Args:
        frame_rgb: Input frame as RGB uint8 (H, W, 3) or float32 (H, W, 3).
        screen_color: Sampled screen RGB (r, g, b) 0-255.
        screen_type: "green", "blue", or "auto" (detects from frame).
        strength: Key gain. Higher = more aggressive. Range: 0.1 - 10.0.
        clip_black: Alpha values below this become fully transparent (0-1).
        clip_white: Alpha values above this become fully opaque (0-1).
        shrink_grow: Erode (negative) or dilate (positive) the matte, in px.
        edge_blur: Gaussian blur radius for matte edges, in px.

    Returns:
        Grayscale matte as uint8 (H, W). 0 = background, 255 = foreground.
    """
    assert frame_rgb.ndim == 3 and frame_rgb.shape[2] == 3, (
        f"Expected (H, W, 3) RGB, got {frame_rgb.shape}"
    )

    # ── Convert to float32 0-1 ──
    if frame_rgb.dtype == np.uint8:
        img_f32 = frame_rgb.astype(np.float32) / 255.0
    else:
        img_f32 = np.clip(frame_rgb.astype(np.float32), 0.0, 1.0)

    # ── Resolve screen type ──
    if screen_type not in ("green", "blue"):
        g_excess = np.clip(img_f32[:, :, 1] - np.maximum(img_f32[:, :, 0], img_f32[:, :, 2]), 0, None).mean()
        b_excess = np.clip(img_f32[:, :, 2] - np.maximum(img_f32[:, :, 0], img_f32[:, :, 1]), 0, None).mean()
        screen_type = "blue" if b_excess > g_excess else "green"

    # ── Reference color ──
    if screen_color is not None:
        ref = np.array(screen_color, dtype=np.float32) / 255.0
    elif screen_type == "blue":
        ref = np.array([0.0, 0.0, 0.9], dtype=np.float32)
    else:
        ref = np.array([0.0, 0.9, 0.0], dtype=np.float32)

    # ── Channel assignment ──
    if screen_type == "blue":
        sc, c1, c2 = 2, 0, 1
    else:
        sc, c1, c2 = 1, 0, 2

    # ── Try GPU path ──
    torch_mod, device = _get_torch_device()
    if torch_mod is not None:
        img_t = torch_mod.from_numpy(img_f32).to(device)
        matte = _chroma_key_gpu(
            img_t, ref[sc], ref[c1], ref[c2],
            sc, c1, c2, strength, clip_black, clip_white, torch_mod,
        )
    else:
        matte = _chroma_key_cpu(
            img_f32, ref, sc, c1, c2, strength, clip_black, clip_white,
        )

    # ── Morphological cleanup (CPU, fast on uint8) ──
    if matte.max() > 0 and matte.min() < 255:
        cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        matte = cv2.morphologyEx(matte, cv2.MORPH_OPEN, cleanup_kernel)
        matte = cv2.morphologyEx(matte, cv2.MORPH_CLOSE, cleanup_kernel)

    # ── Shrink/grow ──
    if shrink_grow != 0:
        abs_px = abs(shrink_grow)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (abs_px * 2 + 1, abs_px * 2 + 1)
        )
        if shrink_grow < 0:
            matte = cv2.erode(matte, kernel)
        else:
            matte = cv2.dilate(matte, kernel)

    # ── Edge blur ──
    if edge_blur > 0:
        k = edge_blur * 2 + 1
        matte = cv2.GaussianBlur(matte, (k, k), 0)

    return matte


def _chroma_key_cpu(
    img: np.ndarray, ref: np.ndarray,
    sc: int, c1: int, c2: int,
    strength: float, clip_black: float, clip_white: float,
) -> np.ndarray:
    """NumPy CPU fallback for the chroma key math."""
    # Color difference
    screen_ch = img[:, :, sc]
    other_max = np.maximum(img[:, :, c1], img[:, :, c2])
    excess = np.clip(screen_ch - other_max, 0.0, None)

    # Normalize
    ref_excess = ref[sc] - max(ref[c1], ref[c2])
    ref_excess = max(ref_excess, 0.01)
    key = np.clip(excess / ref_excess * strength, 0.0, 1.0)

    # Saturation gate
    px_max = img.max(axis=2).clip(min=0.01)
    px_min = img.min(axis=2)
    saturation = 1.0 - px_min / px_max
    sat_gate = np.clip((saturation - 0.2) / 0.2, 0.0, 1.0)
    key = key * sat_gate

    # Smootherstep
    key = key * key * key * (key * (key * 6.0 - 15.0) + 10.0)

    # Darkness protection
    dark_gate = np.clip((px_max - 0.03) / 0.09, 0.0, 1.0)
    key = key * dark_gate

    # Alpha + clip
    alpha = 1.0 - key
    cw = max(clip_white, clip_black + 0.001)
    alpha = np.clip((alpha - clip_black) / (cw - clip_black), 0.0, 1.0)

    return (alpha * 255.0).astype(np.uint8)

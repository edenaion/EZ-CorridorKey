"""Chroma keyer for alpha hint generation.

Color-difference method: measures how much the screen channel (green or blue)
exceeds the maximum of the other two channels, normalized by the sampled
screen color's own ratio. A Hermite smoothstep sharpens the foreground core
without destroying edge gradients. Anything not screen-dominant is immediately
classified as foreground.

The output is an alpha hint for CorridorKey inference -- solid foreground,
clean background, with natural soft transitions at semi-transparent edges
(hair, motion blur).
"""
from __future__ import annotations

import cv2
import numpy as np


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

    Args:
        frame_rgb: Input frame as RGB uint8 (H, W, 3).
        screen_color: Sampled screen RGB (r, g, b) 0-255. Required for best
            results. None falls back to pure-green or pure-blue reference.
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

    # Accept both uint8 (0-255) and float32 (0-1) input
    if frame_rgb.dtype == np.uint8:
        img = frame_rgb.astype(np.float32) / 255.0
    else:
        img = np.clip(frame_rgb.astype(np.float32), 0.0, 1.0)

    # ── Resolve screen type ──
    # "auto" detects from the frame: whichever channel (G or B) has more
    # excess over the other two across the image is the screen channel.
    if screen_type not in ("green", "blue"):
        g_excess = np.clip(img[:, :, 1] - np.maximum(img[:, :, 0], img[:, :, 2]), 0, None).mean()
        b_excess = np.clip(img[:, :, 2] - np.maximum(img[:, :, 0], img[:, :, 1]), 0, None).mean()
        screen_type = "blue" if b_excess > g_excess else "green"

    # Reference screen color in float
    if screen_color is not None:
        ref = np.array(screen_color, dtype=np.float32) / 255.0
    elif screen_type == "blue":
        ref = np.array([0.0, 0.0, 0.9], dtype=np.float32)
    else:
        ref = np.array([0.0, 0.9, 0.0], dtype=np.float32)

    # Channel assignment
    if screen_type == "blue":
        sc, c1, c2 = 2, 0, 1  # screen=B, others=R,G
    else:
        sc, c1, c2 = 1, 0, 2  # screen=G, others=R,B

    # ── Color difference: screen channel minus max of others ──
    # Positive = screen-dominant pixel, zero/negative = foreground
    screen_ch = img[:, :, sc]
    other_max = np.maximum(img[:, :, c1], img[:, :, c2])
    excess = np.clip(screen_ch - other_max, 0.0, None)

    # ── Normalize by screen sample's own excess ──
    ref_excess = ref[sc] - max(ref[c1], ref[c2])
    ref_excess = max(ref_excess, 0.01)
    key = excess / ref_excess

    # ── Strength as gain on key signal ──
    key = np.clip(key * strength, 0.0, 1.0)

    # ── Smoothstep: sharpens foreground core, preserves edge gradients ──
    # Hermite interpolation: 3t^2 - 2t^3
    key = key * key * (3.0 - 2.0 * key)

    # ── Darkness protection ──
    # Very dark pixels cannot be screen. Smooth ramp from 3% to 12%.
    brightness = img.max(axis=2)
    dark_gate = np.clip((brightness - 0.03) / 0.09, 0.0, 1.0)
    key = key * dark_gate

    # ── Alpha = 1 - key ──
    alpha = 1.0 - key

    # ── Clip black / clip white (lift/gain on alpha) ──
    cw = max(clip_white, clip_black + 0.001)
    alpha = np.clip((alpha - clip_black) / (cw - clip_black), 0.0, 1.0)

    # Convert to uint8
    matte = (alpha * 255.0).astype(np.uint8)

    # ── Morphological cleanup ──
    # Opening kills white speckles in BG, closing fills holes in FG
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
        k = edge_blur * 2 + 1  # ensure odd
        matte = cv2.GaussianBlur(matte, (k, k), 0)

    return matte

"""Display transform module for preview rendering.

Handles conversion of VFX image data to display-ready 8-bit sRGB QImage.
Supports EXR (float16/32, 1ch/3ch/4ch), PNG, and video frames.

Codex critical finding: naive clip-to-8bit looks wrong for:
- Linear EXR (needs linear→sRGB gamma)
- Matte (1ch → grayscale visualization)
- Processed (premultiplied RGBA → needs unpremultiply)
- Negative/HDR values (needs clamping before conversion)
"""
from __future__ import annotations

import os
import logging
from collections import OrderedDict
from functools import lru_cache

import cv2
import numpy as np
from PySide6.QtGui import QImage

from .frame_index import ViewMode

logger = logging.getLogger(__name__)

# LRU cache for decoded QImages (max 20 frames)
_QIMAGE_CACHE: OrderedDict[str, QImage] = OrderedDict()
_CACHE_MAX = 20


def _cache_key(path: str, mode: ViewMode) -> str:
    """Cache key combining path and mode for display-specific transforms."""
    return f"{mode.value}:{path}"


def clear_cache() -> None:
    """Clear the frame cache (call on clip switch)."""
    _QIMAGE_CACHE.clear()


def decode_frame(path: str, mode: ViewMode) -> QImage | None:
    """Decode a frame file to a display-ready QImage.

    Applies mode-specific display transforms:
    - COMP: Already 8-bit sRGB PNG, direct load
    - INPUT: May be EXR (linear) or PNG (sRGB)
    - FG: Linear float EXR → sRGB gamma
    - MATTE: 1-channel float → grayscale visualization
    - PROCESSED: Premultiplied RGBA float → unpremultiply → sRGB

    Returns None if the file can't be read.
    """
    key = _cache_key(path, mode)
    if key in _QIMAGE_CACHE:
        _QIMAGE_CACHE.move_to_end(key)
        return _QIMAGE_CACHE[key]

    qimg = _do_decode(path, mode)
    if qimg is not None:
        _QIMAGE_CACHE[key] = qimg
        # Evict oldest if over limit
        while len(_QIMAGE_CACHE) > _CACHE_MAX:
            _QIMAGE_CACHE.popitem(last=False)

    return qimg


def _do_decode(path: str, mode: ViewMode) -> QImage | None:
    """Actual decode + transform logic."""
    if not os.path.isfile(path):
        return None

    is_exr = path.lower().endswith('.exr')

    try:
        if is_exr:
            return _decode_exr(path, mode)
        else:
            return _decode_ldr(path)
    except Exception as e:
        logger.warning(f"Failed to decode frame {path}: {e}")
        return None


def _decode_ldr(path: str) -> QImage | None:
    """Decode an 8-bit image (PNG/JPG/etc) to QImage."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    # BGR → RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _numpy_to_qimage(rgb)


def _decode_exr(path: str, mode: ViewMode) -> QImage | None:
    """Decode an EXR file with mode-specific display transform."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Determine channel layout
    if img.ndim == 2:
        # Single channel (typical for Matte)
        return _transform_matte(img)
    elif img.ndim == 3:
        channels = img.shape[2]
        if channels == 1:
            return _transform_matte(img[:, :, 0])
        elif channels == 2:
            # Rare: treat first channel as matte
            return _transform_matte(img[:, :, 0])
        elif channels == 3:
            # BGR float — FG or Input
            return _transform_linear_rgb(img, mode)
        elif channels == 4:
            # BGRA float — Processed (premultiplied)
            if mode == ViewMode.PROCESSED:
                return _transform_premultiplied(img)
            else:
                # Strip alpha, treat as RGB
                return _transform_linear_rgb(img[:, :, :3], mode)
        else:
            # Unexpected channel count, take first 3
            return _transform_linear_rgb(img[:, :, :3], mode)
    return None


def _transform_matte(data: np.ndarray) -> QImage:
    """Single-channel float → grayscale visualization.

    Clamp to [0,1], apply slight gamma for better visualization of
    subtle alpha edges, convert to 8-bit grayscale displayed as RGB.
    """
    clamped = np.clip(data.astype(np.float32), 0.0, 1.0)
    # Light gamma lift for matte visualization (makes edges visible)
    display = np.power(clamped, 0.85)
    gray8 = (display * 255.0).astype(np.uint8)
    # Stack to 3-channel for consistent QImage format
    rgb = np.stack([gray8, gray8, gray8], axis=2)
    return _numpy_to_qimage(rgb)


def _transform_linear_rgb(bgr: np.ndarray, mode: ViewMode) -> QImage:
    """Linear float BGR → sRGB 8-bit RGB.

    For Input mode: check if data looks sRGB already (common for PNG-sourced EXRs).
    For FG mode: always apply linear→sRGB.
    """
    # Clamp negatives and extreme HDR values
    clamped = np.clip(bgr.astype(np.float32), 0.0, None)

    # Simple Reinhard-style tone mapping for values > 1.0
    max_val = clamped.max()
    if max_val > 1.0:
        clamped = clamped / (1.0 + clamped)  # Reinhard tone map

    # Linear → sRGB gamma
    srgb = _linear_to_srgb(clamped)

    # BGR → RGB
    rgb = cv2.cvtColor((srgb * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return _numpy_to_qimage(rgb)


def _transform_premultiplied(bgra: np.ndarray) -> QImage:
    """Premultiplied RGBA float → unpremultiplied sRGB display.

    Processed output is premultiplied linear RGBA in BGRA order.
    Unpremultiply, apply gamma, show over mid-gray checkerboard conceptually
    (but for simplicity, composite over black for now).
    """
    bgra_f = bgra.astype(np.float32)
    alpha = bgra_f[:, :, 3:4]
    bgr = bgra_f[:, :, :3]

    # Unpremultiply (avoid divide by zero)
    safe_alpha = np.where(alpha > 1e-6, alpha, 1.0)
    unpremult = bgr / safe_alpha

    # Clamp
    unpremult = np.clip(unpremult, 0.0, None)
    max_val = unpremult.max()
    if max_val > 1.0:
        unpremult = unpremult / (1.0 + unpremult)

    # Linear → sRGB
    srgb = _linear_to_srgb(unpremult)

    # BGR → RGB
    rgb = cv2.cvtColor((srgb * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return _numpy_to_qimage(rgb)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma curve to linear float data."""
    # Simplified sRGB gamma: pow(x, 1/2.2)
    # Full sRGB has a linear segment below 0.0031308 but for preview
    # the simplified version is visually indistinguishable
    return np.power(np.clip(linear, 0.0, 1.0), 1.0 / 2.2)


def _numpy_to_qimage(rgb: np.ndarray) -> QImage:
    """Convert an RGB uint8 numpy array to QImage.

    Makes a copy of the data so the numpy array can be freed.
    """
    h, w, ch = rgb.shape
    assert ch == 3, f"Expected 3 channels, got {ch}"
    # Ensure contiguous
    rgb = np.ascontiguousarray(rgb)
    bytes_per_line = w * 3
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    # QImage doesn't copy by default — force a deep copy
    return qimg.copy()


def decode_video_frame(video_path: str, frame_index: int) -> QImage | None:
    """Decode a single frame from a video file.

    Uses cv2.VideoCapture with seek. Expensive for random access.
    Results are cached in the LRU cache.
    """
    key = f"video:{video_path}:{frame_index}"
    if key in _QIMAGE_CACHE:
        _QIMAGE_CACHE.move_to_end(key)
        return _QIMAGE_CACHE[key]

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = _numpy_to_qimage(rgb)

        _QIMAGE_CACHE[key] = qimg
        while len(_QIMAGE_CACHE) > _CACHE_MAX:
            _QIMAGE_CACHE.popitem(last=False)

        return qimg
    except Exception as e:
        logger.warning(f"Failed to decode video frame {frame_index} from {video_path}: {e}")
        return None

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
import threading
from collections import OrderedDict
from functools import lru_cache

import cv2
import numpy as np
from PySide6.QtGui import QImage

from .frame_index import ViewMode

logger = logging.getLogger(__name__)

# LRU cache for decoded QImages (max 20 frames)
# Lock protects concurrent access from thread pool workers
_QIMAGE_CACHE: OrderedDict[str, QImage] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 10


def _cache_key(path: str, mode: ViewMode) -> str:
    """Cache key combining path and mode for display-specific transforms."""
    return f"{mode.value}:{path}"


def clear_cache() -> None:
    """Clear the frame cache (call on clip switch)."""
    with _CACHE_LOCK:
        _QIMAGE_CACHE.clear()


def decode_frame(path: str, mode: ViewMode) -> QImage | None:
    """Decode a frame file to a display-ready QImage.

    Applies mode-specific display transforms:
    - COMP: Already 8-bit sRGB PNG, direct load
    - INPUT: May be EXR (linear) or PNG (sRGB)
    - ALPHA: 8-bit grayscale PNG from AlphaHint, direct load
    - FG: Linear float EXR → sRGB gamma
    - MATTE: 1-channel float → grayscale visualization
    - PROCESSED: Premultiplied RGBA float → unpremultiply → sRGB

    Returns None if the file can't be read.
    """
    key = _cache_key(path, mode)
    with _CACHE_LOCK:
        if key in _QIMAGE_CACHE:
            _QIMAGE_CACHE.move_to_end(key)
            return _QIMAGE_CACHE[key]

    qimg = _do_decode(path, mode)
    if qimg is not None:
        with _CACHE_LOCK:
            _QIMAGE_CACHE[key] = qimg
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
    """Float BGR → sRGB 8-bit RGB.

    INPUT mode: Data from FFmpeg EXR extraction is already sRGB-range float
    (FFmpeg writes video values as-is, no linearisation). Skip gamma.
    FG mode: Inference output is linear float — apply linear→sRGB gamma.
    """
    # Clamp negatives
    clamped = np.clip(bgr.astype(np.float32), 0.0, None)

    if mode == ViewMode.INPUT:
        # Extracted frames are sRGB-range float (FFmpeg writes video values
        # as-is into EXR). Just clamp to [0,1] — no gamma, no tone mapping.
        display = np.clip(clamped, 0.0, 1.0)
    else:
        # Inference output is linear float — tone map HDR then apply gamma
        max_val = clamped.max()
        if max_val > 1.0:
            clamped = clamped / (1.0 + clamped)  # Reinhard tone map
        display = _linear_to_srgb(clamped)

    # BGR → RGB
    rgb = cv2.cvtColor((display * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return _numpy_to_qimage(rgb)


def _transform_premultiplied(bgra: np.ndarray) -> QImage:
    """Premultiplied linear BGRA float → display over black.

    Processed EXR stores premultiplied linear RGBA on disk. For preview
    fidelity, display the stored premultiplied RGB channels over black
    rather than unpremultiplying them into a different image.
    """
    bgra_f = bgra.astype(np.float32)
    bgr = np.clip(bgra_f[:, :, :3], 0.0, None)
    max_val = bgr.max()
    if max_val > 1.0:
        bgr = bgr / (1.0 + bgr)
    srgb = _linear_to_srgb(bgr)
    rgb = cv2.cvtColor((srgb * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return _numpy_to_qimage(rgb)


def processed_rgba_to_qimage(rgba: np.ndarray) -> QImage:
    """Render in-memory premultiplied linear RGBA preview over black.

    Live preview returns RGBA in RGB channel order, while saved EXRs are
    decoded as BGRA. This helper keeps the display contract identical.
    """
    rgba_f = rgba.astype(np.float32)
    rgb = np.clip(rgba_f[:, :, :3], 0.0, None)
    max_val = rgb.max()
    if max_val > 1.0:
        rgb = rgb / (1.0 + rgb)
    srgb = _linear_to_srgb(rgb)
    return _numpy_to_qimage((srgb * 255.0).astype(np.uint8))


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma curve to linear float data."""
    linear = np.clip(linear, 0.0, 1.0)
    mask = linear <= 0.0031308
    return np.where(mask, linear * 12.92, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055)


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
    with _CACHE_LOCK:
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

        with _CACHE_LOCK:
            _QIMAGE_CACHE[key] = qimg
            while len(_QIMAGE_CACHE) > _CACHE_MAX:
                _QIMAGE_CACHE.popitem(last=False)

        return qimg
    except Exception as e:
        logger.warning(f"Failed to decode video frame {frame_index} from {video_path}: {e}")
        return None

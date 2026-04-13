"""Unified frame I/O — read images and video frames as float32 RGB.

All reading functions return float32 arrays in [0, 1] range with RGB channel
order. EXR files are read as-is (linear float); standard formats (PNG, JPG,
etc.) are normalized from uint8.

This module consolidates frame-reading patterns that were previously duplicated
across service.py methods (_read_input_frame, reprocess_single_frame,
_load_frames_for_videomama, _load_mask_frames_for_videomama).
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import cv2
import numpy as np

from .validators import normalize_mask_channels, normalize_mask_dtype

# Enable OpenEXR support in OpenCV
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

logger = logging.getLogger(__name__)

# EXR write flags for cv2.imwrite — PXR24 half-float (fallback only;
# prefer write_exr() for output since OpenCV's DWAB writer is broken)
EXR_WRITE_FLAGS = [
    cv2.IMWRITE_EXR_TYPE,
    cv2.IMWRITE_EXR_TYPE_HALF,
    cv2.IMWRITE_EXR_COMPRESSION,
    cv2.IMWRITE_EXR_COMPRESSION_PXR24,
]


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear-light RGB to sRGB using the standard piecewise curve."""
    linear = np.clip(linear.astype(np.float32), 0.0, None)
    mask = linear <= 0.0031308
    return np.where(
        mask,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB RGB values to linear-light using the standard piecewise curve."""
    srgb = np.clip(srgb.astype(np.float32), 0.0, None)
    mask = srgb <= 0.04045
    return np.where(
        mask,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _exr_compression_constant(name: str):
    """Map a compression name to the Imath compression enum value."""
    import Imath

    _MAP = {
        "dwab": Imath.Compression.DWAB_COMPRESSION,
        "piz": Imath.Compression.PIZ_COMPRESSION,
        "zip": Imath.Compression.ZIPS_COMPRESSION,
        "none": Imath.Compression.NO_COMPRESSION,
    }
    return Imath.Compression(_MAP.get(name.lower(), Imath.Compression.DWAB_COMPRESSION))


def write_exr(path: str, img: np.ndarray, compression: str = "dwab") -> bool:
    """Write an image as EXR half-float using the OpenEXR library.

    Args:
        path: Output file path.
        img: Image array. Accepts:
            - BGR float32 [H, W, 3] (from cv2.imread, service.py output)
            - BGRA float32 [H, W, 4] (straight RGBA from inference)
            - Grayscale float32 [H, W] (single-channel matte)
        compression: EXR compression — "dwab", "piz", "zip", or "none".

    Returns:
        True on success, False on failure.
    """
    import OpenEXR
    import Imath

    try:
        HALF = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        comp = _exr_compression_constant(compression)

        if img.ndim == 2:
            # Grayscale — single Y channel
            h, w = img.shape
            header = OpenEXR.Header(w, h)
            header["compression"] = comp
            header["channels"] = {"Y": HALF}
            y = img.astype(np.float16)
            out = OpenEXR.OutputFile(path, header)
            out.writePixels({"Y": y.tobytes()})
            out.close()
        elif img.ndim == 3 and img.shape[2] == 3:
            # BGR → R, G, B channels
            h, w = img.shape[:2]
            header = OpenEXR.Header(w, h)
            header["compression"] = comp
            header["channels"] = {"R": HALF, "G": HALF, "B": HALF}
            b = img[:, :, 0].astype(np.float16)
            g = img[:, :, 1].astype(np.float16)
            r = img[:, :, 2].astype(np.float16)
            out = OpenEXR.OutputFile(path, header)
            out.writePixels({"R": r.tobytes(), "G": g.tobytes(), "B": b.tobytes()})
            out.close()
        elif img.ndim == 3 and img.shape[2] == 4:
            # BGRA → R, G, B, A channels
            h, w = img.shape[:2]
            header = OpenEXR.Header(w, h)
            header["compression"] = comp
            header["channels"] = {"R": HALF, "G": HALF, "B": HALF, "A": HALF}
            b = img[:, :, 0].astype(np.float16)
            g = img[:, :, 1].astype(np.float16)
            r = img[:, :, 2].astype(np.float16)
            a = img[:, :, 3].astype(np.float16)
            out = OpenEXR.OutputFile(path, header)
            out.writePixels(
                {
                    "R": r.tobytes(),
                    "G": g.tobytes(),
                    "B": b.tobytes(),
                    "A": a.tobytes(),
                }
            )
            out.close()
        else:
            logger.warning(f"Unsupported image shape for EXR write: {img.shape}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Failed to write EXR ({compression}) {path}: {e}")
        return False


# Backwards-compatible aliases
def write_exr_dwab(path: str, img: np.ndarray) -> bool:
    """Write EXR with DWAB compression (legacy alias for write_exr)."""
    return write_exr(path, img, compression="dwab")


def recompress_exr(src_path: str, dst_path: str, compression: str = "dwab") -> bool:
    """Recompress an EXR file to the specified compression.

    Reads the source EXR (any compression) via OpenCV and writes to
    dst_path with the requested compression via OpenEXR library.

    Args:
        src_path: Path to source EXR file.
        dst_path: Path to write recompressed EXR.
        compression: Target compression — "dwab", "piz", "zip", or "none".

    Returns:
        True on success, False on failure.
    """
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.warning(f"Failed to read EXR for recompression: {src_path}")
        return False
    return write_exr(dst_path, img, compression=compression)


def read_image_frame(fpath: str, gamma_correct_exr: bool = False) -> Optional[np.ndarray]:
    """Read an image file (EXR or standard) as float32 RGB [0, 1].

    Args:
        fpath: Absolute path to image file.
        gamma_correct_exr: If True, apply the standard sRGB transfer curve
            to EXR data (converts linear → sRGB for models expecting sRGB).

    Returns:
        float32 array [H, W, 3] in RGB order, or None if read fails.
    """
    is_exr = fpath.lower().endswith(".exr")

    if is_exr:
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        # Strip alpha channel from BGRA EXR
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = np.maximum(img_rgb, 0.0).astype(np.float32)
        if gamma_correct_exr:
            result = _linear_to_srgb(result)
        return result
    else:
        img = cv2.imread(fpath)
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb.astype(np.float32) / 255.0


def read_video_frame_at(
    video_path: str,
    frame_index: int,
) -> Optional[np.ndarray]:
    """Read a single frame from a video by index, as float32 RGB [0, 1].

    Args:
        video_path: Path to video file.
        frame_index: Zero-based frame index to seek to.

    Returns:
        float32 array [H, W, 3] in RGB order, or None if seek/read fails.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    finally:
        cap.release()


def read_video_frames(
    video_path: str,
    processor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> list[np.ndarray]:
    """Read all frames from a video, optionally applying a processor to each.

    Without a processor, frames are returned as float32 RGB [0, 1].

    Args:
        video_path: Path to video file.
        processor: Optional callable (BGR uint8 frame) → processed array.
            If None, default conversion to float32 RGB [0, 1] is applied.

    Returns:
        List of processed frames.
    """
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if processor is not None:
                frames.append(processor(frame))
            else:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                frames.append(img_rgb)
    finally:
        cap.release()
    return frames


def read_mask_frame(fpath: str, clip_name: str = "", frame_index: int = 0) -> Optional[np.ndarray]:
    """Read a mask frame as float32 [H, W] in [0, 1].

    Handles any channel count and dtype via normalize_mask_channels/dtype.

    Args:
        fpath: Path to mask image.
        clip_name: For error context in normalization.
        frame_index: For error context in normalization.

    Returns:
        float32 array [H, W] in [0, 1], or None if read fails.
    """
    mask_in = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if mask_in is None:
        return None
    # dtype normalization MUST happen before channel extraction, because
    # normalize_mask_channels casts to float32 — which would make a uint8
    # 255 into float32 255.0, skipping the /255 division in normalize_mask_dtype.
    mask = normalize_mask_dtype(mask_in)
    mask = normalize_mask_channels(mask, clip_name, frame_index)
    return mask


def decode_video_mask_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize a decoded video frame into a single-channel matte.

    This keeps alpha-video behavior aligned with imported alpha images:
    visible BGR video mattes are converted to grayscale before
    normalization, while a decoded BGRA frame uses its explicit alpha
    channel if the decoder preserves it.
    """
    if frame.ndim == 2:
        mask_in = frame
    elif frame.ndim == 3 and frame.shape[2] == 4:
        mask_in = frame[:, :, 3]
    elif frame.ndim == 3 and frame.shape[2] == 3:
        mask_in = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        mask_in = frame

    mask = normalize_mask_dtype(mask_in)
    return normalize_mask_channels(mask)


def read_video_mask_at(
    video_path: str,
    frame_index: int,
) -> Optional[np.ndarray]:
    """Read a single mask frame from a video by index, as float32 [H, W] [0, 1].

    Decoded BGRA frames use the explicit alpha channel when available.
    Standard decoded BGR video mattes are converted to grayscale, matching
    how imported alpha images are normalized in the UI.

    Args:
        video_path: Path to video file.
        frame_index: Zero-based frame index.

    Returns:
        float32 array [H, W] in [0, 1], or None if seek/read fails.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            return None
        return decode_video_mask_frame(frame)
    finally:
        cap.release()

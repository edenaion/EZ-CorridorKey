"""Unified frame I/O — read images and video frames as float32 RGB.

All reading functions return float32 arrays in [0, 1] range with RGB channel
order. EXR files are read as-is (linear float); standard formats (PNG, JPG,
etc.) are normalized from uint8.

This module consolidates frame-reading patterns that were previously duplicated
across service.py methods (_read_input_frame, reprocess_single_frame,
_load_frames_for_videomama, _load_mask_frames_for_videomama).
"""
from __future__ import annotations

import atexit
import logging
import os
import shutil
import tempfile
from typing import Callable, Optional

# Enable OpenEXR support in OpenCV before cv2 is imported.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from .validators import normalize_mask_channels, normalize_mask_dtype

logger = logging.getLogger(__name__)

# EXR write flags for cv2.imwrite — PXR24 half-float (fallback only;
# prefer write_exr() for output since OpenCV's DWAB writer is broken)
EXR_WRITE_FLAGS = [
    cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
    cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PXR24,
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
            header['compression'] = comp
            header['channels'] = {'Y': HALF}
            y = img.astype(np.float16)
            out = OpenEXR.OutputFile(path, header)
            out.writePixels({'Y': y.tobytes()})
            out.close()
        elif img.ndim == 3 and img.shape[2] == 3:
            # BGR → R, G, B channels
            h, w = img.shape[:2]
            header = OpenEXR.Header(w, h)
            header['compression'] = comp
            header['channels'] = {'R': HALF, 'G': HALF, 'B': HALF}
            b = img[:, :, 0].astype(np.float16)
            g = img[:, :, 1].astype(np.float16)
            r = img[:, :, 2].astype(np.float16)
            out = OpenEXR.OutputFile(path, header)
            out.writePixels({'R': r.tobytes(), 'G': g.tobytes(), 'B': b.tobytes()})
            out.close()
        elif img.ndim == 3 and img.shape[2] == 4:
            # BGRA → R, G, B, A channels
            h, w = img.shape[:2]
            header = OpenEXR.Header(w, h)
            header['compression'] = comp
            header['channels'] = {'R': HALF, 'G': HALF, 'B': HALF, 'A': HALF}
            b = img[:, :, 0].astype(np.float16)
            g = img[:, :, 1].astype(np.float16)
            r = img[:, :, 2].astype(np.float16)
            a = img[:, :, 3].astype(np.float16)
            out = OpenEXR.OutputFile(path, header)
            out.writePixels({
                'R': r.tobytes(), 'G': g.tobytes(),
                'B': b.tobytes(), 'A': a.tobytes(),
            })
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


def _is_ascii(path: str) -> bool:
    """True if every character in the path is ASCII (safe for cv2's narrow API)."""
    try:
        path.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image from disk, safe for non-ASCII paths on Windows.

    cv2.imread uses a narrow (ANSI codepage) file API on Windows and returns
    None for any path containing characters outside that codepage (accented
    Latin, Cyrillic, CJK, Arabic, etc.). ASCII paths take the original
    cv2.imread route unchanged (byte-identical pixels, no extra read); non-ASCII
    paths decode from an in-memory buffer read through numpy's unicode-aware
    file API. cv2.imread and cv2.imdecode share the same codecs, so the decoded
    array is identical for every format and flag combination.

    Args:
        path: Image file path. May contain any Unicode characters.
        flags: cv2 IMREAD_* flags, forwarded verbatim to the decoder.

    Returns:
        Decoded ndarray, or None if the file is missing/unreadable/undecodable
        (matching cv2.imread's None-on-failure contract).
    """
    if _is_ascii(path):
        return cv2.imread(path, flags)
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


# ── Unicode-safe video opening ──────────────────────────────────────────────
# Some OpenCV builds open non-ASCII video paths fine (modern FFmpeg backend);
# older builds fail. When the plain open fails we stage the file at an ASCII
# path. Staged hardlinks/copies are deduped per source and cleaned at exit.
_stage_dir: Optional[str] = None
_staged: dict[str, str] = {}


def _short_path_windows(path: str) -> Optional[str]:
    """Return the Windows 8.3 short-path alias for an existing file, or None.

    The 8.3 alias is pure ASCII, so cv2's narrow file API can open it. Only
    available on Windows volumes with 8.3 generation enabled.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        get_short = ctypes.windll.kernel32.GetShortPathNameW
        get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(32768)
        n = get_short(path, buf, len(buf))
        if n and buf.value and _is_ascii(buf.value):
            return buf.value
    except Exception:
        pass
    return None


def _cleanup_staged() -> None:
    if _stage_dir and os.path.isdir(_stage_dir):
        shutil.rmtree(_stage_dir, ignore_errors=True)


def _stage_ascii(path: str) -> str:
    """Stage a file at an ASCII temp path via hardlink (else copy); deduped.

    Returns the staged ASCII path, or the original path if staging fails.
    """
    global _stage_dir
    existing = _staged.get(path)
    if existing and os.path.isfile(existing):
        return existing
    if _stage_dir is None:
        _stage_dir = tempfile.mkdtemp(prefix="ck_uvid_")
        atexit.register(_cleanup_staged)
    dst = os.path.join(_stage_dir, f"v{len(_staged):04d}{os.path.splitext(path)[1]}")
    try:
        os.link(path, dst)  # cheap: no data copy on the same volume
    except OSError:
        try:
            shutil.copy2(path, dst)
        except OSError:
            return path
    _staged[path] = dst
    return dst


def open_video(path: str) -> cv2.VideoCapture:
    """Open a video, safe for non-ASCII paths across all OpenCV builds.

    Plain VideoCapture works for ASCII paths and for modern OpenCV builds whose
    FFmpeg backend is wide-path aware. When it fails on a non-ASCII path (older
    builds), fall back to a Windows 8.3 short-path alias, then to a
    hardlink/copy at an ASCII temp path. Returns a normal VideoCapture, so
    callers keep their existing isOpened()/release() handling unchanged.
    """
    cap = cv2.VideoCapture(path)
    if cap.isOpened() or _is_ascii(path):
        return cap
    cap.release()

    short = _short_path_windows(path)
    if short and short != path:
        cap = cv2.VideoCapture(short)
        if cap.isOpened():
            return cap
        cap.release()

    staged = _stage_ascii(path)
    if staged != path:
        return cv2.VideoCapture(staged)
    return cv2.VideoCapture(path)


class _StagedVideoWriter:
    """cv2.VideoWriter proxy that writes to an ASCII temp path and moves the
    finished file to the (non-ASCII) final path on release()."""

    def __init__(self, writer: cv2.VideoWriter, tmp_path: str, final_path: str):
        self._writer = writer
        self._tmp_path = tmp_path
        self._final_path = final_path

    def isOpened(self) -> bool:  # noqa: N802 — mirror cv2.VideoWriter API
        return self._writer.isOpened()

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()
        try:
            shutil.move(self._tmp_path, self._final_path)
        except OSError as e:
            logger.error(f"Failed to move staged video to {self._final_path}: {e}")


def open_video_writer(path: str, fourcc: int, fps: float, frame_size: tuple):
    """Open a video writer, safe for non-ASCII paths across all OpenCV builds.

    Plain VideoWriter works for ASCII paths and for builds whose FFmpeg
    backend is wide-path aware. When it fails to open on a non-ASCII path,
    fall back to the parent directory's Windows 8.3 short-path alias, then to
    writing at an ASCII temp path that is moved into place on release().
    Returns an object with the cv2.VideoWriter interface
    (write/isOpened/release).
    """
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
    if writer.isOpened() or _is_ascii(path):
        return writer
    writer.release()

    # The target file does not exist yet, so alias the parent dir instead.
    parent = _short_path_windows(os.path.dirname(os.path.abspath(path)))
    name = os.path.basename(path)
    if parent and _is_ascii(name):
        writer = cv2.VideoWriter(os.path.join(parent, name), fourcc, fps, frame_size)
        if writer.isOpened():
            return writer
        writer.release()

    # Stage at an ASCII temp path (short-path the temp dir too: %TEMP% lives
    # under the user profile, so it inherits a non-ASCII username).
    tmp_dir = _short_path_windows(tempfile.gettempdir()) or tempfile.gettempdir()
    fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(path)[1] or ".mp4", dir=tmp_dir)
    os.close(fd)
    staged = cv2.VideoWriter(tmp_path, fourcc, fps, frame_size)
    return _StagedVideoWriter(staged, tmp_path, path)


def imwrite_unicode(path: str, img: np.ndarray, params: Optional[list] = None) -> bool:
    """Write an image to disk via cv2.imencode, safe for unicode paths.

    cv2.imwrite cannot handle non-ASCII paths on Windows. Encoding to an
    in-memory buffer and writing it through numpy's tofile goes through
    Python's unicode-aware file API instead, so the path round-trips
    correctly on every platform.

    Args:
        path: Output file path. The extension selects the encoder.
        img: Image array to write (any dtype/layout cv2.imencode accepts).
        params: Optional cv2.IMWRITE_* flag pairs forwarded to the encoder.

    Returns:
        True on success, False on failure.
    """
    try:
        ext = os.path.splitext(path)[1]
        ok, buf = cv2.imencode(ext, img, params or [])
        if not ok:
            logger.warning(f"Failed to encode image for {path}")
            return False
        buf.tofile(path)
        return True
    except Exception as e:
        logger.warning(f"Failed to write image {path}: {e}")
        return False


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
    img = imread_unicode(src_path, cv2.IMREAD_UNCHANGED)
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
    is_exr = fpath.lower().endswith('.exr')

    if is_exr:
        img = imread_unicode(fpath, cv2.IMREAD_UNCHANGED)
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
        img = imread_unicode(fpath)
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb.astype(np.float32) / 255.0


def read_video_frame_at(
    video_path: str, frame_index: int,
) -> Optional[np.ndarray]:
    """Read a single frame from a video by index, as float32 RGB [0, 1].

    Args:
        video_path: Path to video file.
        frame_index: Zero-based frame index to seek to.

    Returns:
        float32 array [H, W, 3] in RGB order, or None if seek/read fails.
    """
    cap = open_video(video_path)
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
    cap = open_video(video_path)
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
    mask_in = imread_unicode(fpath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
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
    video_path: str, frame_index: int,
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
    cap = open_video(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            return None
        return decode_video_mask_frame(frame)
    finally:
        cap.release()

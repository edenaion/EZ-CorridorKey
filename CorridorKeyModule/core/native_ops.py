"""Thin wrapper around the optional Rust extension corridorkey-native.

Falls back to pure NumPy implementations when the native extension is not installed.
"""
import numpy as np

try:
    from corridorkey_native import (
        gbr_planar_to_rgb as _gbr_planar_to_rgb_native,
        bgr_u8_to_rgb_f32 as _bgr_u8_to_rgb_f32_native,
        rgb_f32_to_bgr_u8 as _rgb_f32_to_bgr_u8_native,
    )
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False


def gbr_planar_to_rgb(raw: bytes, height: int, width: int) -> np.ndarray:
    """Convert FFmpeg gbrpf32le planar buffer to interleaved RGB [H, W, 3] float32."""
    if HAS_NATIVE:
        flat = np.frombuffer(raw, dtype=np.float32)
        return _gbr_planar_to_rgb_native(flat, height, width)
    planes = np.frombuffer(raw, dtype=np.float32).reshape(3, height, width)
    return np.stack([planes[2], planes[0], planes[1]], axis=-1)


def bgr_u8_to_rgb_f32(image: np.ndarray) -> np.ndarray:
    """Fused BGR uint8 → RGB float32 [0, 1]."""
    if HAS_NATIVE:
        return _bgr_u8_to_rgb_f32_native(np.ascontiguousarray(image))
    return image[:, :, ::-1].astype(np.float32) / 255.0


def rgb_f32_to_bgr_u8(image: np.ndarray) -> np.ndarray:
    """Fused RGB float32 → BGR uint8."""
    if HAS_NATIVE:
        return _rgb_f32_to_bgr_u8_native(np.ascontiguousarray(image))
    return (np.clip(image[:, :, ::-1], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

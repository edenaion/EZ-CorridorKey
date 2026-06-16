"""
Apple Vision framework foreground subject mask generation via PyObjC.

Uses VNGenerateForegroundInstanceMaskRequest (macOS 14+) to detect
foreground subjects on Apple Neural Engine. Returns uint8 alpha mattes.

Requires: macOS 14+, pyobjc-framework-Vision
    pip install pyobjc-framework-Vision pyobjc-framework-CoreImage pyobjc-framework-Quartz
"""

import logging
import sys

import cv2
import numpy as np

logger = logging.getLogger(__name__)

if sys.platform != "darwin":
    raise ImportError("apple_vision module requires macOS")

import objc  # noqa: E402, F401
import Quartz  # noqa: E402
import Vision  # noqa: E402
from Quartz import (  # noqa: E402
    CIImage,
    CGColorSpaceCreateWithName,
    kCGColorSpaceSRGB,
)

# Cache the Vision request object across frames (preserves compiled inference graph)
_cached_request = None


def _get_request():
    """Get or create the VNGenerateForegroundInstanceMaskRequest (latest revision)."""
    global _cached_request
    if _cached_request is None:
        req = Vision.VNGenerateForegroundInstanceMaskRequest.alloc().init()
        # Use the latest supported revision for best quality
        try:
            revisions = Vision.VNGenerateForegroundInstanceMaskRequest.supportedRevisions()
            if revisions and len(revisions) > 0:
                latest = max(revisions)
                req.setRevision_(latest)
                logger.info("Apple Vision: using revision %d (latest of %s)", latest, list(revisions))
            else:
                logger.info("Apple Vision: using default revision")
        except Exception:
            logger.info("Apple Vision: using default revision (supportedRevisions unavailable)")
        _cached_request = req
    return _cached_request


def generate_foreground_mask(rgb_frame: np.ndarray) -> np.ndarray:
    """Generate a foreground alpha mask using Apple Vision.

    Args:
        rgb_frame: Input image as uint8 RGB numpy array [H, W, 3].

    Returns:
        Alpha matte as uint8 numpy array [H, W] (0=background, 255=foreground).
    """
    h, w = rgb_frame.shape[:2]

    # Convert RGB numpy array to CIImage (RGBA format)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb_frame
    rgba[:, :, 3] = 255
    rgba_data = rgba.tobytes()

    ci_image = CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
        rgba_data,
        w * 4,
        (w, h),
        24,  # kCIFormatRGBA8 = 24
        CGColorSpaceCreateWithName(kCGColorSpaceSRGB),
    )

    if ci_image is None:
        logger.error("Apple Vision: failed to create CIImage from frame")
        return np.zeros((h, w), dtype=np.uint8)

    handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
        ci_image, None
    )

    request = _get_request()
    success, error = handler.performRequests_error_([request], None)

    if not success or error is not None:
        logger.warning(f"Apple Vision request failed: {error}")
        return np.zeros((h, w), dtype=np.uint8)

    results = request.results()
    if not results or len(results) == 0:
        logger.debug("Apple Vision: no foreground instances detected")
        return np.zeros((h, w), dtype=np.uint8)

    # Generate mask combining all foreground instances
    observation = results[0]
    all_indices = observation.allInstances()

    try:
        mask_buffer, gen_error = observation.generateScaledMaskForImageForInstances_fromRequestHandler_error_(
            all_indices, handler, None
        )
    except Exception as e:
        logger.warning(f"Apple Vision: generateScaledMask failed: {e}")
        return np.zeros((h, w), dtype=np.uint8)

    if mask_buffer is None or gen_error is not None:
        logger.warning(f"Apple Vision: mask generation failed: {gen_error}")
        return np.zeros((h, w), dtype=np.uint8)

    return _cvpixelbuffer_to_uint8(mask_buffer, h, w)


def _cvpixelbuffer_to_uint8(pixel_buffer, target_h: int, target_w: int) -> np.ndarray:
    """Convert a Vision mask CVPixelBuffer to a uint8 alpha image.

    Apple Vision can return OneComponent8, OneComponent16Half, or
    OneComponent32Float masks. Match the Swift app's path: read the buffer
    according to its real pixel format, normalize float formats, then resize
    only if Vision returned a non-target size.
    """
    lock_readonly = getattr(Quartz, "kCVPixelBufferLock_ReadOnly", 0)
    Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, lock_readonly)
    try:
        base_address = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
        buf_w = Quartz.CVPixelBufferGetWidth(pixel_buffer)
        buf_h = Quartz.CVPixelBufferGetHeight(pixel_buffer)
        bytes_per_row = Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer)
        pixel_format = Quartz.CVPixelBufferGetPixelFormatType(pixel_buffer)

        if base_address is None:
            logger.warning("Apple Vision: empty CVPixelBuffer base address")
            return np.zeros((target_h, target_w), dtype=np.uint8)

        buf_size = buf_h * bytes_per_row
        mv = base_address.as_buffer(buf_size)

        fmt_u8 = getattr(Quartz, "kCVPixelFormatType_OneComponent8", None)
        fmt_f16 = getattr(Quartz, "kCVPixelFormatType_OneComponent16Half", None)
        fmt_f32 = getattr(Quartz, "kCVPixelFormatType_OneComponent32Float", None)

        if fmt_u8 is not None and pixel_format == fmt_u8:
            arr = np.frombuffer(mv, dtype=np.uint8).reshape(buf_h, bytes_per_row)[:, :buf_w].copy()
        elif fmt_f32 is not None and pixel_format == fmt_f32:
            floats_per_row = bytes_per_row // 4
            arr_f = np.frombuffer(mv, dtype=np.float32).reshape(buf_h, floats_per_row)[:, :buf_w]
            arr = np.rint(np.clip(arr_f, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif fmt_f16 is not None and pixel_format == fmt_f16:
            halfs_per_row = bytes_per_row // 2
            arr_f = np.frombuffer(mv, dtype=np.float16).reshape(buf_h, halfs_per_row)[:, :buf_w]
            arr = np.rint(np.clip(arr_f.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            logger.warning("Apple Vision: unsupported mask pixel format %s", pixel_format)
            return np.zeros((target_h, target_w), dtype=np.uint8)

    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, lock_readonly)

    logger.debug("Apple Vision mask buffer: %dx%d -> target %dx%d",
                 buf_w, buf_h, target_w, target_h)

    if arr.shape[0] != target_h or arr.shape[1] != target_w:
        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return np.clip(arr, 0, 255).astype(np.uint8)


def is_available() -> bool:
    """Check if Apple Vision foreground masking is available on this system.

    Requires macOS 14+ (Sonoma) for VNGenerateForegroundInstanceMaskRequest.
    """
    if sys.platform != "darwin":
        return False

    try:
        import platform
        version = tuple(int(x) for x in platform.mac_ver()[0].split(".")[:2])
        if version < (14, 0):
            logger.debug(f"Apple Vision: macOS {version} < 14.0, not available")
            return False
    except (ValueError, IndexError):
        return False

    try:
        # Verify the API exists
        _ = Vision.VNGenerateForegroundInstanceMaskRequest
        return True
    except AttributeError:
        logger.debug("Apple Vision: VNGenerateForegroundInstanceMaskRequest not found")
        return False

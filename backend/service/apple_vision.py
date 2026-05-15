"""
Apple Vision framework foreground subject mask generation via PyObjC.

Uses VNGenerateForegroundInstanceMaskRequest (macOS 14+) to detect
foreground subjects on Apple Neural Engine. Returns uint8 alpha mattes.

Requires: macOS 14+, pyobjc-framework-Vision
    pip install pyobjc-framework-Vision pyobjc-framework-CoreImage
"""

import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)

if sys.platform != "darwin":
    raise ImportError("apple_vision module requires macOS")

import objc  # noqa: E402
import Vision  # noqa: E402
from Quartz import (  # noqa: E402
    CIImage,
    CGColorSpaceCreateWithName,
    kCGColorSpaceSRGB,
)

# Cache the Vision request object across frames (preserves compiled inference graph)
_cached_request = None


def _get_request():
    """Get or create the VNGenerateForegroundInstanceMaskRequest."""
    global _cached_request
    if _cached_request is None:
        _cached_request = Vision.VNGenerateForegroundInstanceMaskRequest.alloc().init()
        logger.info("Apple Vision: created VNGenerateForegroundInstanceMaskRequest")
    return _cached_request


def generate_foreground_mask(rgb_frame: np.ndarray) -> np.ndarray:
    """Generate a foreground alpha mask using Apple Vision.

    Args:
        rgb_frame: Input image as uint8 RGB numpy array [H, W, 3].

    Returns:
        Alpha matte as uint8 numpy array [H, W] (0=background, 255=foreground).
    """
    h, w = rgb_frame.shape[:2]

    # Convert RGB numpy array to CIImage
    # CIImage expects RGBA or BGRA data. We'll create RGBA.
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb_frame
    rgba[:, :, 3] = 255
    rgba_data = rgba.tobytes()

    # Create CIImage from bitmap data
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

    # Create request handler with the image
    # VNImageRequestHandler with CIImage uses .downMirrored because
    # CIImage has bottom-left origin, but we pre-flip to avoid confusion.
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
    all_indices = results  # All observations are foreground instances

    try:
        mask_buffer, gen_error = observation.generateScaledMaskForImage_forInstances_error_(
            ci_image, all_indices, None
        )
    except Exception as e:
        logger.warning(f"Apple Vision: generateScaledMask failed: {e}")
        return np.zeros((h, w), dtype=np.uint8)

    if mask_buffer is None or gen_error is not None:
        logger.warning(f"Apple Vision: mask generation failed: {gen_error}")
        return np.zeros((h, w), dtype=np.uint8)

    # Convert CVPixelBuffer to numpy array
    mask_np = _cvpixelbuffer_to_numpy(mask_buffer, h, w)
    return mask_np


def _cvpixelbuffer_to_numpy(pixel_buffer, target_h: int, target_w: int) -> np.ndarray:
    """Convert a CVPixelBuffer (from Vision) to a uint8 numpy array.

    Vision outputs masks as OneComponent32Float CVPixelBuffers.
    We convert to uint8 [0, 255] and resize to target dimensions.
    """
    import CoreVideo  # noqa: E402
    import cv2

    CoreVideo.CVPixelBufferLockBaseAddress(pixel_buffer, 0)
    try:
        base_address = CoreVideo.CVPixelBufferGetBaseAddress(pixel_buffer)
        buf_w = CoreVideo.CVPixelBufferGetWidth(pixel_buffer)
        buf_h = CoreVideo.CVPixelBufferGetHeight(pixel_buffer)
        bytes_per_row = CoreVideo.CVPixelBufferGetBytesPerRow(pixel_buffer)

        # OneComponent32Float: each pixel is float32
        import ctypes
        float_count = buf_h * (bytes_per_row // 4)
        arr = np.ctypeslib.as_array(
            ctypes.cast(base_address, ctypes.POINTER(ctypes.c_float)),
            shape=(float_count,),
        ).copy()  # Copy while buffer is locked

        # Reshape accounting for row stride
        floats_per_row = bytes_per_row // 4
        arr = arr.reshape(buf_h, floats_per_row)[:, :buf_w]

    finally:
        CoreVideo.CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)

    # Convert float [0.0, 1.0] to uint8 [0, 255]
    mask_u8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Resize to target dimensions if needed
    if mask_u8.shape[0] != target_h or mask_u8.shape[1] != target_w:
        mask_u8 = cv2.resize(mask_u8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return mask_u8


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

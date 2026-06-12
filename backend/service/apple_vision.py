"""
Apple Vision framework foreground subject mask generation via PyObjC.

Uses VNGenerateForegroundInstanceMaskRequest (macOS 14+) to detect
foreground subjects on Apple Neural Engine. Returns float32 alpha mattes
with guided-filter edge refinement for smooth, anti-aliased edges.

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

    Returns a float32 matte with guided-filter edge refinement for smooth,
    anti-aliased edges suitable for VFX compositing.

    Args:
        rgb_frame: Input image as uint8 RGB numpy array [H, W, 3].

    Returns:
        Alpha matte as uint8 numpy array [H, W] (0=background, 255=foreground).
        Internally processed as float32 with guided-filter refinement before
        final quantization.
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

    # Extract float32 mask from CVPixelBuffer (preserve full precision)
    mask_f32 = _cvpixelbuffer_to_float32(mask_buffer, h, w)

    # Refine edges using guided filter (source image guides edge placement)
    mask_f32 = _guided_filter_refine(mask_f32, rgb_frame)

    # Final quantization to uint8 for PNG output
    return (np.clip(mask_f32, 0.0, 1.0) * 255.0).astype(np.uint8)


def _cvpixelbuffer_to_float32(pixel_buffer, target_h: int, target_w: int) -> np.ndarray:
    """Convert a CVPixelBuffer to a float32 numpy array [0.0, 1.0].

    Vision outputs masks as OneComponent32Float CVPixelBuffers at some
    internal resolution. We preserve float32 precision and use Lanczos
    resampling to scale to target dimensions.
    """
    Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, 0)
    try:
        base_address = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
        buf_w = Quartz.CVPixelBufferGetWidth(pixel_buffer)
        buf_h = Quartz.CVPixelBufferGetHeight(pixel_buffer)
        bytes_per_row = Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer)

        buf_size = buf_h * bytes_per_row
        mv = base_address.as_buffer(buf_size)
        arr = np.frombuffer(mv, dtype=np.float32).copy()

        # Reshape accounting for row stride padding
        floats_per_row = bytes_per_row // 4
        arr = arr.reshape(buf_h, floats_per_row)[:, :buf_w]

    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)

    logger.debug("Apple Vision mask buffer: %dx%d -> target %dx%d",
                 buf_w, buf_h, target_w, target_h)

    arr = np.clip(arr, 0.0, 1.0)

    # Lanczos resize to target dimensions (much sharper than bilinear)
    if arr.shape[0] != target_h or arr.shape[1] != target_w:
        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return arr


def _guided_filter_refine(
    mask: np.ndarray,
    guide_rgb: np.ndarray,
    radius: int = 16,
    eps: float = 1e-4,
) -> np.ndarray:
    """Refine mask edges using the source image as a guide.

    The guided filter smooths flat regions of the mask while preserving
    edges that align with the source image structure. This converts
    blocky segmentation boundaries into smooth, image-aligned edges.

    Uses cv2.ximgproc.guidedFilter if available (opencv-contrib), otherwise
    falls back to a numpy box-filter implementation.

    Args:
        mask: Float32 alpha matte [H, W] in [0.0, 1.0].
        guide_rgb: Source image as uint8 RGB [H, W, 3].
        radius: Filter kernel radius in pixels.
        eps: Regularization (smaller = sharper edges, larger = smoother).

    Returns:
        Refined float32 alpha matte [H, W].
    """
    guide_f = guide_rgb.astype(np.float32) / 255.0

    # Try opencv-contrib guided filter first (fastest, GPU-optimized)
    try:
        refined = cv2.ximgproc.guidedFilter(
            guide=guide_f, src=mask, radius=radius, eps=eps, dDepth=-1
        )
        return np.clip(refined, 0.0, 1.0)
    except AttributeError:
        pass

    # Fallback: numpy guided filter with grayscale guide
    guide_gray = cv2.cvtColor(guide_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return _guided_filter_numpy(guide_gray, mask, radius, eps)


def _guided_filter_numpy(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Guided filter using numpy box filters. Grayscale guide only."""
    ksize = 2 * radius + 1

    def box(x):
        return cv2.blur(x, (ksize, ksize))

    mean_I = box(guide)
    mean_p = box(src)
    corr_Ip = box(guide * src)
    var_I = box(guide * guide) - mean_I * mean_I

    a = (corr_Ip - mean_I * mean_p) / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box(a)
    mean_b = box(b)

    return np.clip(mean_a * guide + mean_b, 0.0, 1.0)


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

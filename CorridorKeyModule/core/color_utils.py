import math
from functools import lru_cache

import torch
import torch.nn.functional as F
import numpy as np

def _is_tensor(x):
    return isinstance(x, torch.Tensor)


# --- Pure-tensor paths (torch.compile safe, no isinstance dispatch) ---

def linear_to_srgb_tensor(x: torch.Tensor) -> torch.Tensor:
    """Pure-tensor Linear→sRGB. Use this for torch.compile contexts."""
    x = x.clamp(min=0.0)
    mask = x <= 0.0031308
    return torch.where(mask, x * 12.92, 1.055 * torch.pow(x, 1.0 / 2.4) - 0.055)


def srgb_to_linear_tensor(x: torch.Tensor) -> torch.Tensor:
    """Pure-tensor sRGB→Linear. Use this for torch.compile contexts."""
    x = x.clamp(min=0.0)
    mask = x <= 0.04045
    return torch.where(mask, x / 12.92, torch.pow((x + 0.055) / 1.055, 2.4))

def linear_to_srgb(x):
    """
    Converts Linear to sRGB using the official piecewise sRGB transfer function.
    Supports both Numpy arrays and PyTorch tensors.
    """
    if _is_tensor(x):
        x = x.clamp(min=0.0)
        mask = x <= 0.0031308
        return torch.where(mask, x * 12.92, 1.055 * torch.pow(x, 1.0/2.4) - 0.055)
    else:
        x = np.clip(x, 0.0, None)
        mask = x <= 0.0031308
        return np.where(mask, x * 12.92, 1.055 * np.power(x, 1.0/2.4) - 0.055)

def srgb_to_linear(x):
    """
    Converts sRGB to Linear using the official piecewise sRGB transfer function.
    Supports both Numpy arrays and PyTorch tensors.
    """
    if _is_tensor(x):
        x = x.clamp(min=0.0)
        mask = x <= 0.04045
        return torch.where(mask, x / 12.92, torch.pow((x + 0.055) / 1.055, 2.4))
    else:
        x = np.clip(x, 0.0, None)
        mask = x <= 0.04045
        return np.where(mask, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))

def premultiply(fg, alpha):
    """
    Premultiplies foreground by alpha.
    fg: Color [..., C] or [C, ...]
    alpha: Alpha [..., 1] or [1, ...]
    """
    return fg * alpha

def unpremultiply(fg, alpha, eps=1e-6):
    """
    Un-premultiplies foreground by alpha.
    Ref: fg_straight = fg_premul / (alpha + eps)
    """
    if _is_tensor(fg):
        return fg / (alpha + eps)
    else:
        return fg / (alpha + eps)

def composite_straight(fg, bg, alpha):
    """
    Composites Straight FG over BG.
    Formula: FG * Alpha + BG * (1 - Alpha)
    """
    return fg * alpha + bg * (1.0 - alpha)

def composite_premul(fg, bg, alpha):
    """
    Composites Premultiplied FG over BG.
    Formula: FG + BG * (1 - Alpha)
    """
    return fg + bg * (1.0 - alpha)


def match_luminance(source_rgb, image_rgb, min_scale=0.9, max_scale=1.15, strength=1.0, eps=1e-6):
    """
    Re-match image luminance to a source reference while keeping the image chroma.

    Both inputs are expected to be linear RGB float images with matching shapes.
    A per-pixel Rec. 709 luminance ratio is computed and clamped to avoid
    aggressive swings from noisy pixels or edge cases.
    """
    if strength <= 0.0:
        return image_rgb

    if _is_tensor(image_rgb):
        weights = image_rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(*([1] * (image_rgb.dim() - 1)), 3)
        src_y = (source_rgb * weights).sum(dim=-1, keepdim=True)
        img_y = (image_rgb * weights).sum(dim=-1, keepdim=True)
        scale = (src_y / torch.clamp(img_y, min=eps)).clamp(min=min_scale, max=max_scale)
        corrected = image_rgb * scale
        if strength < 1.0:
            corrected = image_rgb * (1.0 - strength) + corrected * strength
        return corrected.clamp(min=0.0)

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    src_y = np.sum(source_rgb * weights, axis=-1, keepdims=True)
    img_y = np.sum(image_rgb * weights, axis=-1, keepdims=True)
    scale = np.clip(src_y / np.maximum(img_y, eps), min_scale, max_scale)
    corrected = image_rgb * scale
    if strength < 1.0:
        corrected = image_rgb * (1.0 - strength) + corrected * strength
    return np.clip(corrected, 0.0, None)

def rgb_to_yuv(image):
    """
    Converts RGB to YUV (Rec. 601).
    Input: [..., 3, H, W] or [..., 3] depending on layout. 
    Supports standard PyTorch BCHW.
    """
    if not _is_tensor(image):
        raise TypeError("rgb_to_yuv only supports dict/tensor inputs currently")

    # Weights for RGB -> Y
    # Rec. 601: 0.299, 0.587, 0.114
    
    # Assume BCHW layout if 4 dims
    if image.dim() == 4:
        r = image[:, 0:1, :, :]
        g = image[:, 1:2, :, :]
        b = image[:, 2:3, :, :]
    elif image.dim() == 3 and image.shape[0] == 3: # CHW
        r = image[0:1, :, :]
        g = image[1:2, :, :]
        b = image[2:3, :, :]
    else:
        # Last dim conversion
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.492 * (b - y)
    v = 0.877 * (r - y)
    
    if image.dim() >= 3 and image.shape[-3] == 3: # Concatenate along Channel dim
         return torch.cat([y, u, v], dim=-3)
    else:
         return torch.stack([y, u, v], dim=-1)

def dilate_mask(mask, radius):
    """
    Dilates a mask by a given radius.
    Supports Numpy (using cv2) and PyTorch (using MaxPool).
    radius: Int (pixels). 0 = No change.
    """
    if radius <= 0:
        return mask

    if _is_tensor(mask):
        # PyTorch Dilation (using Max Pooling)
        # Expects [B, C, H, W]
        orig_dim = mask.dim()
        if orig_dim == 2: mask = mask.unsqueeze(0).unsqueeze(0)
        elif orig_dim == 3: mask = mask.unsqueeze(0)
        
        kernel_size = int(radius * 2 + 1)
        padding = radius
        dilated = torch.nn.functional.max_pool2d(mask, kernel_size, stride=1, padding=padding)
        
        if orig_dim == 2: return dilated.squeeze()
        elif orig_dim == 3: return dilated.squeeze(0)
        return dilated
    else:
        # Numpy Dilation (using OpenCV)
        import cv2
        kernel_size = int(radius * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(mask, kernel)

def apply_garbage_matte(predicted_matte, garbage_matte_input, dilation=10):
    """
    Multiplies predicted matte by a dilated garbage matte to clean up background.
    """
    if garbage_matte_input is None:
        return predicted_matte
        
    garbage_mask = dilate_mask(garbage_matte_input, dilation)
    
    # Ensure dimensions match for multiplication
    if _is_tensor(predicted_matte):
        # Handle broadcasting if needed
        pass 
    else:
        # Numpy
        if garbage_mask.ndim == 2 and predicted_matte.ndim == 3:
            garbage_mask = garbage_mask[:, :, np.newaxis]
            
    return predicted_matte * garbage_mask

def despill(image, green_limit_mode='average', strength=1.0):
    """
    Removes green spill from an RGB image using a luminance-preserving method.
    image: RGB float (0-1).
    green_limit_mode: 'average' ((R+B)/2) or 'max' (max(R, B)).
    strength: 0.0 to 1.0 multiplier for the despill effect.
    """
    if strength <= 0.0:
        return image
        
    if _is_tensor(image):
        # PyTorch Impl
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]
        
        if green_limit_mode == 'max':
            limit = torch.max(r, b)
        else:
            limit = (r + b) / 2.0
            
        spill_amount = torch.clamp(g - limit, min=0.0)
        
        g_new = g - spill_amount
        r_new = r + (spill_amount * 0.5)
        b_new = b + (spill_amount * 0.5)
        
        despilled = torch.stack([r_new, g_new, b_new], dim=-1)
        
        if strength < 1.0:
            return image * (1.0 - strength) + despilled * strength
        return despilled
    else:
        # Numpy Impl
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]
        
        if green_limit_mode == 'max':
            limit = np.maximum(r, b)
        else:
            limit = (r + b) / 2.0
            
        spill_amount = np.maximum(g - limit, 0.0)
        
        g_new = g - spill_amount
        r_new = r + (spill_amount * 0.5)
        b_new = b + (spill_amount * 0.5)
        
        despilled = np.stack([r_new, g_new, b_new], axis=-1)
        
        if strength < 1.0:
            return image * (1.0 - strength) + despilled * strength
        return despilled

def clean_matte(alpha_np, area_threshold=300, dilation=15, blur_size=5):
    """
    Cleans up small disconnected components (like tracking markers) from a predicted alpha matte.
    alpha_np: Numpy array [H, W] or [H, W, 1] float (0.0 - 1.0)
    """
    import cv2
    import numpy as np
    
    # Needs to be 2D
    is_3d = False
    if alpha_np.ndim == 3:
        is_3d = True
        alpha_np = alpha_np[:, :, 0]
        
    # Threshold to binary at a LOW value so that wispy hair strands (alpha < 0.5)
    # stay connected to the main subject. A 0.5 threshold — the previous default —
    # severed hair strands from the body and the component-area filter then deleted
    # them, producing the hair-hole regression. 0.02 keeps anything the model
    # thinks is even slightly foreground as part of the candidate mask.
    mask_8u = (alpha_np > 0.02).astype(np.uint8) * 255

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_8u, connectivity=8)

    # Create an empty mask for the cleaned components
    cleaned_mask = np.zeros_like(mask_8u)

    # Always keep the largest non-background component — that is the subject,
    # and with the low threshold above, hair strands are part of it.
    # Additionally keep any other component whose area exceeds area_threshold
    # (covers legitimate detached pieces like stray curls). Everything smaller
    # is treated as noise / tracking markers and dropped.
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = int(np.argmax(areas)) + 1
        cleaned_mask[labels == largest_label] = 255
        for i in range(1, num_labels):
            if i == largest_label:
                continue
            if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
                cleaned_mask[labels == i] = 255
            
    # Dilate
    if dilation > 0:
        kernel_size = int(dilation * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned_mask = cv2.dilate(cleaned_mask, kernel)
        
    # Blur
    if blur_size > 0:
        b_size = int(blur_size * 2 + 1)
        cleaned_mask = cv2.GaussianBlur(cleaned_mask, (b_size, b_size), 0)
        
    # Convert back to 0-1 float
    safe_zone = cleaned_mask.astype(np.float32) / 255.0
    
    # Multiply original alpha by the safe zone
    result_alpha = alpha_np * safe_zone
    
    if is_3d:
        result_alpha = result_alpha[:, :, np.newaxis]

    return result_alpha


_gaussian_kernel_cache: dict[tuple, torch.Tensor] = {}


def _gaussian_kernel_2d(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a normalized 2D Gaussian kernel [1, 1, size, size]."""
    key = (size, device, dtype)
    cached = _gaussian_kernel_cache.get(key)
    if cached is not None:
        return cached
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-0.5 * (coords / max((size - 1) / 6.0, 1e-6)) ** 2)
    kernel = g[:, None] * g[None, :]
    kernel /= kernel.sum()
    result = kernel.reshape(1, 1, size, size)
    _gaussian_kernel_cache[key] = result
    return result


def clear_gpu_caches():
    """Release cached GPU tensors (call on engine teardown)."""
    _gaussian_kernel_cache.clear()


def clean_matte_gpu(alpha_t: torch.Tensor, area_threshold: int = 300,
                    dilation: int = 15, blur_size: int = 5) -> torch.Tensor:
    """GPU-resident matte cleanup — no CPU roundtrip.

    Uses morphological opening (erode then dilate) to remove small islands,
    replacing OpenCV's connectedComponentsWithStats. The kernel size is derived
    from area_threshold: kernel ≈ 2 * sqrt(area / π). Elongated thin structures
    smaller than the kernel may also be removed (acceptable for matte cleanup).

    Note: this version does not preserve hair-strand connectivity the way the
    CPU clean_matte() does. Prefer the CPU version for production output;
    use the GPU one for previews/realtime where speed matters more than detail.

    alpha_t: Tensor [H, W] or [H, W, 1] float 0-1 on any device.
    """
    is_3d = alpha_t.ndim == 3
    if is_3d:
        alpha_t_2d = alpha_t[:, :, 0]
    else:
        alpha_t_2d = alpha_t

    # Threshold to binary
    mask = (alpha_t_2d > 0.5).float()

    # Morphological opening to remove small islands
    open_k = max(3, int(2 * math.sqrt(area_threshold / math.pi)) | 1)  # ensure odd
    mask_4d = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    pad = open_k // 2

    # Erode: invert → max_pool → invert
    eroded = 1.0 - F.max_pool2d(1.0 - mask_4d, open_k, stride=1, padding=pad)
    # Dilate: max_pool
    opened = F.max_pool2d(eroded, open_k, stride=1, padding=pad)

    # Additional dilation
    if dilation > 0:
        d_k = int(dilation * 2 + 1)
        d_pad = d_k // 2
        opened = F.max_pool2d(opened, d_k, stride=1, padding=d_pad)

    # Gaussian blur
    if blur_size > 0:
        b_k = int(blur_size * 2 + 1)
        b_pad = b_k // 2
        g_kernel = _gaussian_kernel_2d(b_k, alpha_t.device, alpha_t.dtype)
        opened = F.conv2d(opened, g_kernel, padding=b_pad)

    safe_zone = opened.squeeze(0).squeeze(0)  # [H, W]
    result = alpha_t_2d * safe_zone

    if is_3d:
        result = result.unsqueeze(-1)

    return result


def source_passthrough(original_srgb, model_fg_srgb, alpha, erode_px=None, blur_px=None):
    """
    Blend original source pixels into the model's foreground prediction.

    Where the alpha matte is confidently opaque (interior of subject), we use
    the original source pixels directly — they haven't been through the model
    so they retain full quality.  Near edges (where alpha transitions), we use
    the model's predicted foreground which handles green-screen separation.

    Erosion and blur values scale with image resolution to prevent visible
    seams at attention window boundaries, especially on 4K+ footage.

    Args:
        original_srgb: [H, W, 3] float32 sRGB, original frame.
        model_fg_srgb: [H, W, 3] float32 sRGB, model's predicted foreground.
        alpha:         [H, W, 1] or [H, W] float32 (0-1), predicted alpha matte.
        erode_px:      Pixels to erode the interior mask inward from the edge.
                       None = auto-scale based on resolution.
        blur_px:       Gaussian blur radius for the transition band.
                       None = auto-scale based on resolution.

    Returns:
        [H, W, 3] float32 sRGB, blended foreground.
    """
    import cv2

    h, w = original_srgb.shape[:2]
    long_edge = max(h, w)

    # Scale erosion and blur proportional to resolution.
    # Reference: 1080p (1920px) → erode 5, blur 11
    #            4K   (3840px) → erode 10, blur 23
    #            8K   (7680px) → erode 20, blur 45
    scale = long_edge / 1920.0
    if erode_px is None:
        erode_px = max(3, int(round(5 * scale)))
    if blur_px is None:
        blur_px = max(7, int(round(11 * scale))) | 1  # ensure odd

    # Work with 2D alpha
    a = alpha[:, :, 0] if alpha.ndim == 3 else alpha

    # Interior mask: where the subject is fully opaque
    # Use a high threshold — we only want to pass through pixels that are
    # definitively interior (no edge ambiguity at all)
    interior = (a > 0.95).astype(np.float32)

    # Erode inward to create a safety buffer around edges.
    # This ensures we never use original pixels where green spill might exist.
    if erode_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
        )
        interior = cv2.erode(interior, k)

    # Smooth the transition so there's no visible seam between
    # original pixels and model-predicted pixels.
    if blur_px > 0:
        ks = blur_px | 1  # ensure odd
        interior = cv2.GaussianBlur(interior, (ks, ks), 0)

    # Expand to 3-channel for broadcasting
    blend = interior[:, :, np.newaxis]  # 1.0 = use original, 0.0 = use model

    return blend * original_srgb + (1.0 - blend) * model_fg_srgb


@lru_cache(maxsize=8)
def create_checkerboard(width, height, checker_size=64, color1=0.2, color2=0.4):
    """
    Creates a linear grayscale checkerboard pattern.
    Cached because the same (w, h, checker_size) is reused on every frame.
    Returns: Numpy array [H, W, 3] float (0.0-1.0)
    """
    import numpy as np
    
    # Create coordinate grids
    x = np.arange(width)
    y = np.arange(height)
    
    # Determine tile parity
    x_tiles = x // checker_size
    y_tiles = y // checker_size
    
    # Broadcast to 2D
    x_grid, y_grid = np.meshgrid(x_tiles, y_tiles)
    
    # XOR for checker pattern (1 if odd, 0 if even)
    checker = (x_grid + y_grid) % 2
    
    # Map 0 to color1 and 1 to color2
    bg_img = np.where(checker == 0, color1, color2).astype(np.float32)

    # Make it 3-channel and freeze (cached array — callers must not mutate)
    result = np.stack([bg_img, bg_img, bg_img], axis=-1)
    result.flags.writeable = False
    return result


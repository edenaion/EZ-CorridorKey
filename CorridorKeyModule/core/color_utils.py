import cv2
import torch
import numpy as np

def _is_tensor(x):
    return isinstance(x, torch.Tensor)


def detect_screen_color(frame: np.ndarray) -> str:
    """Detect whether a frame has a green or blue chroma key background.

    Downsamples the entire frame, converts to HSV, and counts how many
    saturated pixels are green vs blue. Whichever has more wins.

    Args:
        frame: RGB uint8 image (H, W, 3).

    Returns:
        "green" or "blue".
    """
    # Downsample to ~256px wide for speed
    h, w = frame.shape[:2]
    scale = 256.0 / max(w, 1)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(small.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

    # Only count saturated pixels (S > 40) — skin, clothes, dark areas are excluded
    sat_mask = hsv[:, 1] > 40
    if sat_mask.sum() < 10:
        return "green"

    hues = hsv[sat_mask, 0]

    # OpenCV hue: 0-179. Green ~35-80, Blue ~85-130.
    green_count = int(((hues >= 35) & (hues < 80)).sum())
    blue_count = int(((hues >= 85) & (hues <= 130)).sum())

    if blue_count > green_count:
        return "blue"
    return "green"

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

def refiner_additive_guard(model_alpha, hint_mask, shell_px=4):
    """
    Per-pixel max of model alpha and hint, except in a thin shell just
    outside the model's solid silhouette.

    The plain max() guard protects guide detail the refiner erodes (hair
    wisps), but it also burns the hint's edge over-coverage into the
    output: alpha hints routinely extend 1-3px past the true subject edge
    (generator slack, hint resizing), and max() resurrects that band even
    though the model correctly rejected it. Once the garbage matte clears
    the surrounding gunk, the band reads as a ~1px white outline around
    the subject.

    Inside the shell (within shell_px of the solid body) the model's edge
    is trusted as-is; beyond it the additive guard applies in full, so
    detached hair strands and wisps stay protected.

    model_alpha: float alpha [H, W] or [H, W, 1], model prediction.
    hint_mask: float alpha hint, same spatial size.
    shell_px: width of the trust-the-model shell around the solid body.
    """
    if hint_mask is None:
        return model_alpha

    import cv2

    m2 = model_alpha[:, :, 0] if model_alpha.ndim == 3 else model_alpha
    solid = (m2 > 0.5).astype(np.uint8)
    k = 2 * int(shell_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    shell = (cv2.dilate(solid, kernel) > 0) & (solid == 0)
    if model_alpha.ndim == 3:
        shell = shell[:, :, np.newaxis]

    guarded = np.maximum(model_alpha, hint_mask)
    return np.where(shell, model_alpha, guarded)

def apply_garbage_matte(predicted_matte, hint_mask, dilation_px, feather_px=3,
                        hint_threshold=0.1):
    """
    Zeros out predicted alpha outside the hint region expanded by dilation_px.

    The hint is binarized before dilation so its soft edge values never
    scale the subject's own edge alpha (a fractional multiply at small
    dilation_px reads as a 1px outline hugging the subject). The cut
    boundary is feathered outward, with the dilation widened by feather_px
    so the falloff never intrudes into the requested dilation_px margin.

    predicted_matte: float alpha [H, W] or [H, W, 1].
    hint_mask: float alpha hint [H, W] or [H, W, 1].
    dilation_px: full-protection margin (px) around the binarized hint.
    feather_px: width (px) of the soft falloff beyond that margin.
    hint_threshold: hint values above this count as inside the matte.
    """
    if hint_mask is None or dilation_px <= 0:
        return predicted_matte

    import cv2

    hint_2d = hint_mask[:, :, 0] if hint_mask.ndim == 3 else hint_mask
    hint_bin = (hint_2d > hint_threshold).astype(np.float32)

    radius = int(dilation_px) + int(feather_px)
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    expanded = cv2.dilate(hint_bin, kernel)

    if feather_px > 0:
        fk = 2 * int(feather_px) + 1
        expanded = cv2.GaussianBlur(expanded, (fk, fk), 0)

    if expanded.ndim == 2 and predicted_matte.ndim == 3:
        expanded = expanded[:, :, np.newaxis]

    return predicted_matte * expanded

def despill(image, green_limit_mode='average', strength=1.0, screen_color='green'):
    """
    Removes screen spill from an RGB image using a luminance-preserving method.
    image: RGB float (0-1).
    green_limit_mode: 'average' or 'max' limit for the spill channel.
    strength: 0.0 to 1.0 multiplier for the despill effect.
    screen_color: 'green' (spill ch=1) or 'blue' (spill ch=2).
    """
    if strength <= 0.0:
        return image

    # Channel indices: spill channel and the two "other" channels
    if screen_color == 'blue':
        sp, o1, o2 = 2, 0, 1  # blue is spill, red and green are others
    else:
        sp, o1, o2 = 1, 0, 2  # green is spill, red and blue are others

    if _is_tensor(image):
        spill_ch = image[..., sp]
        other1 = image[..., o1]
        other2 = image[..., o2]

        if green_limit_mode == 'max':
            limit = torch.max(other1, other2)
        else:
            limit = (other1 + other2) / 2.0

        spill_amount = torch.clamp(spill_ch - limit, min=0.0)

        channels = [image[..., 0], image[..., 1], image[..., 2]]
        channels[sp] = spill_ch - spill_amount
        channels[o1] = other1 + (spill_amount * 0.5)
        channels[o2] = other2 + (spill_amount * 0.5)

        despilled = torch.stack(channels, dim=-1)

        if strength < 1.0:
            return image * (1.0 - strength) + despilled * strength
        return despilled
    else:
        spill_ch = image[..., sp]
        other1 = image[..., o1]
        other2 = image[..., o2]

        if green_limit_mode == 'max':
            limit = np.maximum(other1, other2)
        else:
            limit = (other1 + other2) / 2.0

        spill_amount = np.maximum(spill_ch - limit, 0.0)

        channels = [image[..., 0].copy(), image[..., 1].copy(), image[..., 2].copy()]
        channels[sp] = spill_ch - spill_amount
        channels[o1] = other1 + (spill_amount * 0.5)
        channels[o2] = other2 + (spill_amount * 0.5)

        despilled = np.stack(channels, axis=-1)

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


def create_checkerboard(width, height, checker_size=64, color1=0.2, color2=0.4):
    """
    Creates a linear grayscale checkerboard pattern.
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
    
    # Make it 3-channel
    return np.stack([bg_img, bg_img, bg_img], axis=-1)


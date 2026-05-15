"""
MLX inference engine for CorridorKey on Apple Silicon.

Drop-in replacement for CorridorKeyEngine. Same process_frame() interface,
same post-processing (despeckle, despill, source passthrough, compositing).

The only difference is the neural network forward pass: MLX instead of
PyTorch, running natively on Apple Silicon unified memory.

Usage:
    engine = MLXCorridorKeyEngine(
        checkpoint_path="CorridorKeyModule/checkpoints/CorridorKey.mlx.safetensors",
        img_size=2048,
    )
    result = engine.process_frame(image, mask)
"""

import logging
import os
import sys
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Shared inference parameter defaults (single source of truth)
from .inference_engine import INFERENCE_DEFAULTS


class MLXCorridorKeyEngine:
    """MLX-based inference engine for Apple Silicon Macs."""

    def __init__(self, checkpoint_path, img_size=2048, use_refiner=True, on_status=None):
        logger.info("MLXCorridorKeyEngine.__init__: begin")
        self.img_size = img_size
        self.checkpoint_path = checkpoint_path
        self.use_refiner = use_refiner
        self._on_status = on_status

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.model = self._load_model()

    def _status(self, msg: str) -> None:
        logger.info(msg)
        if self._on_status:
            self._on_status(msg)

    def _load_model(self):
        import mlx.core as mx
        from safetensors.numpy import load_file
        from .core.mlx_model import GreenFormerMLX

        self._status("Initializing MLX model backbone...")
        t0 = time.monotonic()
        model = GreenFormerMLX(img_size=self.img_size, use_refiner=self.use_refiner)
        logger.info(f"MLX model init: {time.monotonic() - t0:.1f}s")

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"MLX checkpoint not found: {self.checkpoint_path}")

        self._status("Loading MLX weights...")
        t0 = time.monotonic()
        weight_dict = load_file(self.checkpoint_path)

        # Resize pos_embed if checkpoint was saved at different img_size
        grid = self.img_size // 4
        expected_n = grid * grid
        pe_key = "encoder.pos_embed"
        if pe_key in weight_dict:
            pe = weight_dict[pe_key]
            if pe.shape[1] != expected_n:
                import math
                src_grid = int(math.sqrt(pe.shape[1]))
                dst_grid = int(math.sqrt(expected_n))
                logger.info(f"Resizing pos_embed: {src_grid}x{src_grid} -> {dst_grid}x{dst_grid}")
                # Bicubic resize via reshape + nn.Upsample
                C = pe.shape[2]
                pe_mx = mx.array(pe).reshape(1, src_grid, src_grid, C)
                from mlx.nn import Upsample
                scale = dst_grid / src_grid
                up = Upsample(scale_factor=scale, mode="linear", align_corners=False)
                pe_resized = up(pe_mx).reshape(1, expected_n, C)
                mx.eval(pe_resized)
                import numpy as _np
                weight_dict[pe_key] = _np.array(pe_resized)

        mx_weights = [(k, mx.array(v)) for k, v in weight_dict.items()]
        model.load_weights(mx_weights)
        logger.info(f"MLX weights loaded: {time.monotonic() - t0:.1f}s")

        # Set eval mode (BatchNorm uses running stats) and evaluate parameters
        model.eval()
        mx.eval(model.parameters())

        self._status("MLX model ready")
        return model

    _D = INFERENCE_DEFAULTS

    def process_frame(
        self,
        image,
        mask_linear,
        refiner_scale=_D["refiner_scale"],
        input_is_linear=False,
        fg_is_straight=True,
        despill_strength=_D["despill_strength"],
        auto_despeckle=_D["auto_despeckle"],
        despeckle_size=_D["despeckle_size"],
        despeckle_dilation=_D["despeckle_dilation"],
        despeckle_blur=_D["despeckle_blur"],
        source_passthrough=_D["source_passthrough"],
        edge_erode_px=_D["edge_erode_px"],
        edge_blur_px=_D["edge_blur_px"],
        screen_color=_D["screen_color"],
        garbage_matte_px=_D["garbage_matte_px"],
    ):
        """Process a single frame. Same interface as CorridorKeyEngine.process_frame()."""
        import mlx.core as mx
        from .core import color_utils as cu

        t0 = time.monotonic()

        # 1. Input normalization
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0

        h, w = image.shape[:2]

        if mask_linear.ndim == 2:
            mask_linear = mask_linear[:, :, np.newaxis]

        # 2. Resize to model size
        if input_is_linear:
            img_resized_lin = cv2.resize(image, (self.img_size, self.img_size),
                                         interpolation=cv2.INTER_LINEAR)
            img_resized = cu.linear_to_srgb(img_resized_lin)
        else:
            img_resized = cv2.resize(image, (self.img_size, self.img_size),
                                     interpolation=cv2.INTER_LINEAR)

        mask_resized = cv2.resize(mask_linear, (self.img_size, self.img_size),
                                  interpolation=cv2.INTER_LINEAR)
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

        # 3. Normalize (ImageNet)
        img_norm = (img_resized - self.mean) / self.std

        # 4. Prepare input: NHWC for MLX
        inp_np = np.concatenate([img_norm, mask_resized], axis=-1)  # [H, W, 4]
        inp_mx = mx.array(inp_np[np.newaxis])  # [1, H, W, 4]

        # 5. Inference
        out = self.model(inp_mx, refiner_scale=refiner_scale)
        mx.eval(out["alpha"], out["fg"])

        # Convert back to numpy (already NHWC)
        pred_alpha = np.array(out["alpha"][0])  # [H, W, 1]
        pred_fg = np.array(out["fg"][0])        # [H, W, 3]

        # 6. Resize back to original resolution (Lanczos)
        res_alpha = cv2.resize(pred_alpha, (w, h), interpolation=cv2.INTER_LANCZOS4)
        res_fg = cv2.resize(pred_fg, (w, h), interpolation=cv2.INTER_LANCZOS4)

        if res_alpha.ndim == 2:
            res_alpha = res_alpha[:, :, np.newaxis]

        # Refiner additive guard
        res_alpha = np.maximum(res_alpha, mask_linear)

        # Source passthrough
        if source_passthrough:
            if input_is_linear:
                original_srgb = cu.linear_to_srgb(image)
            else:
                original_srgb = image
            res_fg = cu.source_passthrough(
                original_srgb, res_fg, res_alpha,
                erode_px=edge_erode_px, blur_px=edge_blur_px
            )

        # Clean matte
        if auto_despeckle:
            processed_alpha = cu.clean_matte(res_alpha, area_threshold=despeckle_size,
                                             dilation=despeckle_dilation, blur_size=despeckle_blur)
        else:
            processed_alpha = res_alpha

        # Garbage matte
        if garbage_matte_px > 0:
            k = 2 * garbage_matte_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            hint_2d = mask_linear[:, :, 0] if mask_linear.ndim == 3 else mask_linear
            expanded_hint = cv2.dilate(hint_2d, kernel)
            if expanded_hint.ndim == 2:
                expanded_hint = expanded_hint[:, :, np.newaxis]
            processed_alpha = processed_alpha * expanded_hint

        # Despill
        fg_despilled = cu.despill(res_fg, green_limit_mode='average',
                                  strength=despill_strength, screen_color=screen_color)

        # Linear conversion and premultiply
        fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
        fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)
        processed_rgba = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)

        # Composite on checkerboard
        bg_srgb = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
        bg_lin = cu.srgb_to_linear(bg_srgb)

        if fg_is_straight:
            comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
        else:
            comp_lin = cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha)

        comp_srgb = cu.linear_to_srgb(comp_lin)

        logger.debug(f"mlx process_frame: {h}x{w} in {time.monotonic() - t0:.3f}s")

        return {
            'alpha': res_alpha,
            'fg': res_fg,
            'comp': comp_srgb,
            'processed': processed_rgba,
        }

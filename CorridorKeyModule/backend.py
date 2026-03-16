"""Backend factory — selects Torch or MLX engine and normalizes output contracts."""

from __future__ import annotations

import glob
import logging
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
TORCH_EXT = ".pth"
MLX_EXT = ".safetensors"
DEFAULT_IMG_SIZE = 2048

BACKEND_ENV_VAR = "CORRIDORKEY_BACKEND"
VALID_BACKENDS = ("auto", "torch", "mlx")


def resolve_backend(requested: str | None = None) -> str:
    """Resolve backend: CLI flag > env var > auto-detect.

    Auto mode: Apple Silicon + corridorkey_mlx importable + .safetensors found -> mlx.
    Otherwise -> torch.

    Raises RuntimeError if explicit backend is unavailable.
    """
    if requested is None or requested.lower() == "auto":
        backend = os.environ.get(BACKEND_ENV_VAR, "auto").lower()
    else:
        backend = requested.lower()

    if backend == "auto":
        return _auto_detect_backend()

    if backend not in VALID_BACKENDS:
        raise RuntimeError(f"Unknown backend '{backend}'. Valid: {', '.join(VALID_BACKENDS)}")

    if backend == "mlx":
        _validate_mlx_available()

    return backend


def _auto_detect_backend() -> str:
    """Try MLX on Apple Silicon, fall back to Torch."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        logger.info("Not Apple Silicon — using torch backend")
        return "torch"

    try:
        import corridorkey_mlx  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        logger.info("corridorkey_mlx not installed — using torch backend")
        return "torch"

    safetensor_files = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{MLX_EXT}"))
    if not safetensor_files:
        logger.info("No %s checkpoint found — using torch backend", MLX_EXT)
        return "torch"

    logger.info("Apple Silicon + MLX available — using mlx backend")
    return "mlx"


def _validate_mlx_available() -> None:
    """Raise RuntimeError with actionable message if MLX can't be used."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX backend requires Apple Silicon (M1+ Mac)")

    try:
        import corridorkey_mlx  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as err:
        raise RuntimeError(
            "MLX backend requested but corridorkey_mlx is not installed. "
            "Install with: uv pip install -e '.[mlx]'"
        ) from err


def _discover_checkpoint(ext: str) -> Path:
    """Find exactly one checkpoint with the given extension.

    Raises FileNotFoundError (0 found) or ValueError (>1 found).
    Includes cross-reference hints when wrong extension files exist.
    """
    matches = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{ext}"))

    if len(matches) == 0:
        other_ext = MLX_EXT if ext == TORCH_EXT else TORCH_EXT
        other_files = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{other_ext}"))
        hint = ""
        if other_files:
            other_backend = "mlx" if other_ext == MLX_EXT else "torch"
            hint = f" (Found {other_ext} files — did you mean CORRIDORKEY_BACKEND={other_backend}?)"
        raise FileNotFoundError(f"No {ext} checkpoint found in {CHECKPOINT_DIR}.{hint}")

    if len(matches) > 1:
        names = [os.path.basename(f) for f in matches]
        raise ValueError(f"Multiple {ext} checkpoints in {CHECKPOINT_DIR}: {names}. Keep exactly one.")

    return Path(matches[0])


def _wrap_mlx_output(
    raw: dict,
    source_image: np.ndarray,
    input_is_linear: bool,
    despill_strength: float,
    auto_despeckle: bool,
    despeckle_size: int,
    despeckle_dilation: int = 25,
    despeckle_blur: int = 5,
) -> dict:
    """Normalize MLX uint8 output to match Torch float32 contract.

    Torch contract:
      alpha:     [H,W,1] float32 0-1
      fg:        [H,W,3] float32 0-1 sRGB
      comp:      [H,W,3] float32 0-1 sRGB
      processed: [H,W,4] float32 linear premultiplied RGBA
    """
    from CorridorKeyModule.core import color_utils as cu

    # alpha: uint8 [H,W] -> float32 [H,W,1]
    alpha_raw = raw["alpha"]
    alpha = alpha_raw.astype(np.float32) / 255.0
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]

    # fg: uint8 [H,W,3] -> float32 [H,W,3] (sRGB)
    fg = raw["fg"].astype(np.float32) / 255.0

    # Apply despeckle (MLX stubs this)
    if auto_despeckle:
        processed_alpha = cu.clean_matte(
            alpha, area_threshold=despeckle_size,
            dilation=despeckle_dilation, blur_size=despeckle_blur,
        )
    else:
        processed_alpha = alpha

    # Apply despill (MLX stubs this)
    fg_despilled = cu.despill(fg, green_limit_mode="average", strength=despill_strength)

    if source_image.dtype != np.float32:
        source_rgb = source_image.astype(np.float32)
        if source_rgb.max() > 1.0:
            source_rgb /= 255.0
    else:
        source_rgb = source_image

    source_srgb = cu.linear_to_srgb(source_rgb) if input_is_linear else source_rgb
    source_lin = source_rgb if input_is_linear else cu.srgb_to_linear(source_rgb)

    # Composite over checkerboard for comp output
    h, w = fg.shape[:2]
    bg_srgb = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
    bg_lin = cu.srgb_to_linear(bg_srgb)
    fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
    fg_despilled_lin = _restore_opaque_source_detail(
        source_lin,
        source_srgb,
        fg_despilled_lin,
        processed_alpha,
    )
    comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
    comp_srgb = cu.linear_to_srgb(comp_lin)

    # Build processed: [H,W,4] premultiplied linear RGBA
    fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)
    processed_rgba = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)

    # Restore source detail on FG for display accuracy (MLX model colors
    # are lower fidelity than CUDA — blend original source back in on
    # opaque regions so FG view matches what the user shot).
    fg_lin = cu.srgb_to_linear(fg)
    fg_restored_lin = _restore_opaque_source_detail(
        source_lin, source_srgb, fg_lin, processed_alpha,
    )
    fg_restored = cu.linear_to_srgb(fg_restored_lin)

    return {
        "alpha": alpha,
        "fg": fg_restored,
        "comp": comp_srgb,
        "processed": processed_rgba,
    }


def _assemble_mlx_output(
    *,
    alpha: np.ndarray,
    fg: np.ndarray,
    source_image: np.ndarray,
    input_is_linear: bool,
    despill_strength: float,
    auto_despeckle: bool,
    despeckle_size: int,
    despeckle_dilation: int = 25,
    despeckle_blur: int = 5,
) -> dict:
    """Assemble the shared Torch-style contract from MLX float outputs."""
    from CorridorKeyModule.core import color_utils as cu

    alpha = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]

    fg = np.clip(fg.astype(np.float32, copy=False), 0.0, 1.0)

    if auto_despeckle:
        processed_alpha = cu.clean_matte(
            alpha, area_threshold=despeckle_size,
            dilation=despeckle_dilation, blur_size=despeckle_blur,
        )
    else:
        processed_alpha = alpha

    fg_despilled = cu.despill(fg, green_limit_mode="average", strength=despill_strength)

    if source_image.dtype != np.float32:
        source_rgb = source_image.astype(np.float32)
        if source_rgb.max() > 1.0:
            source_rgb /= 255.0
    else:
        source_rgb = source_image

    source_srgb = cu.linear_to_srgb(source_rgb) if input_is_linear else source_rgb
    source_lin = source_rgb if input_is_linear else cu.srgb_to_linear(source_rgb)

    h, w = fg.shape[:2]
    bg_srgb = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
    bg_lin = cu.srgb_to_linear(bg_srgb)
    fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
    fg_despilled_lin = _restore_opaque_source_detail(
        source_lin,
        source_srgb,
        fg_despilled_lin,
        processed_alpha,
    )
    comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
    comp_srgb = cu.linear_to_srgb(comp_lin)

    fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)
    processed_rgba = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)

    # Restore source detail on FG for display accuracy (MLX model colors
    # are lower fidelity than CUDA — blend original source back in on
    # opaque regions so FG view matches what the user shot).
    fg_lin = cu.srgb_to_linear(fg)
    fg_restored_lin = _restore_opaque_source_detail(
        source_lin, source_srgb, fg_lin, processed_alpha,
    )
    fg_restored = cu.linear_to_srgb(fg_restored_lin)

    return {
        "alpha": alpha,
        "fg": fg_restored,
        "comp": comp_srgb,
        "processed": processed_rgba,
    }


def _prepare_mlx_image_u8(image: np.ndarray, input_is_linear: bool) -> np.ndarray:
    """Prepare image input for corridorkey-mlx.

    The upstream MLX engine currently assumes sRGB/ImageNet-normalized inputs;
    its `input_is_linear` parameter is a compatibility no-op. So for true
    linear inputs we must convert to sRGB before quantizing to uint8.
    """
    from CorridorKeyModule.core import color_utils as cu

    if image.dtype == np.uint8:
        return image

    image_f32 = image.astype(np.float32, copy=False)
    if image_f32.max() > 1.0:
        image_f32 = image_f32 / 255.0

    if input_is_linear:
        image_f32 = cu.linear_to_srgb(image_f32)

    return (np.clip(image_f32, 0.0, 1.0) * 255.0).astype(np.uint8)


def _resize_float_image_bicubic(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize float image data with PIL bicubic to match upstream MLX behavior."""
    if image.ndim == 2:
        return np.asarray(
            Image.fromarray(image.astype(np.float32), mode="F").resize(size, Image.Resampling.BICUBIC),
            dtype=np.float32,
        )

    if image.ndim == 3 and image.shape[2] == 1:
        resized = np.asarray(
            Image.fromarray(image[:, :, 0].astype(np.float32), mode="F").resize(
                size, Image.Resampling.BICUBIC
            ),
            dtype=np.float32,
        )
        return resized[:, :, np.newaxis]

    if image.ndim == 3 and image.shape[2] >= 2:
        channels = [
            np.asarray(
                Image.fromarray(image[:, :, idx].astype(np.float32), mode="F").resize(
                    size, Image.Resampling.BICUBIC
                ),
                dtype=np.float32,
            )
            for idx in range(image.shape[2])
        ]
        return np.stack(channels, axis=-1)

    raise ValueError(f"Unsupported image shape for float resize: {image.shape}")


def _restore_opaque_source_detail(
    source_lin: np.ndarray,
    source_srgb: np.ndarray,
    image_lin: np.ndarray,
    alpha: np.ndarray,
    *,
    opaque_threshold: float = 0.92,
    opaque_softness: float = 0.08,
    edge_band_radius: int = 8,
    edge_band_softness: int = 6,
    edge_source_alpha_threshold: float = 0.78,
    edge_source_alpha_softness: float = 0.18,
    green_spill_threshold: float = 0.03,
    green_spill_softness: float = 0.08,
) -> np.ndarray:
    """Prefer source detail away from the alpha edge band.

    Near the edge, source color is only preserved when the pixel is still
    reasonably opaque *and* the source does not look green-contaminated.
    This keeps opaque clothing edges from drifting orange/red while still
    letting MLX own real green-spill zones like wispy hair.
    """
    alpha_plane = alpha if alpha.ndim == 3 else alpha[:, :, np.newaxis]
    alpha_plane = np.clip(alpha_plane.astype(np.float32, copy=False), 0.0, 1.0)

    opaque_weight = np.clip(
        (alpha_plane - opaque_threshold) / max(opaque_softness, 1e-6),
        0.0,
        1.0,
    )
    alpha_2d = alpha_plane[:, :, 0]
    opaque_mask = (alpha_2d >= opaque_threshold).astype(np.uint8)

    if opaque_mask.size == 0 or not opaque_mask.any():
        interior_weight = np.zeros_like(alpha_plane, dtype=np.float32)
    elif opaque_mask.all():
        interior_weight = np.ones_like(alpha_plane, dtype=np.float32)
    else:
        dist_to_edge = cv2.distanceTransform(opaque_mask, cv2.DIST_L2, 5)
        interior_weight = np.clip(
            (dist_to_edge - float(edge_band_radius)) / max(float(edge_band_softness), 1e-6),
            0.0,
            1.0,
        )[:, :, np.newaxis]

    edge_source_weight = np.clip(
        (alpha_plane - edge_source_alpha_threshold) / max(edge_source_alpha_softness, 1e-6),
        0.0,
        1.0,
    )

    source_srgb = np.clip(source_srgb.astype(np.float32, copy=False), 0.0, 1.0)
    green_excess = (
        source_srgb[..., 1:2]
        - np.maximum(source_srgb[..., 0:1], source_srgb[..., 2:3])
    )
    green_spill_weight = np.clip(
        (green_excess - green_spill_threshold) / max(green_spill_softness, 1e-6),
        0.0,
        1.0,
    )

    safe_edge_source_weight = edge_source_weight * (1.0 - green_spill_weight)
    detail_weight = np.maximum(opaque_weight * interior_weight, safe_edge_source_weight)
    return image_lin * (1.0 - detail_weight) + source_lin * detail_weight


def _try_mlx_float_outputs(raw_engine, image_u8: np.ndarray, mask_u8: np.ndarray, refiner_scale: float) -> dict | None:
    """Use corridorkey-mlx internals to recover float outputs before uint8 quantization."""
    if not hasattr(raw_engine, "_model") or not hasattr(raw_engine, "_img_size"):
        return None

    try:
        import mlx.core as mx
        from corridorkey_mlx.io.image import preprocess
    except Exception as exc:
        logger.debug("MLX float-output path unavailable: %s", exc)
        return None

    original_h, original_w = image_u8.shape[:2]
    target_size = int(getattr(raw_engine, "_img_size"))

    rgb_f32 = image_u8.astype(np.float32) / 255.0
    mask_plane = mask_u8 if mask_u8.ndim == 2 else mask_u8[:, :, 0]
    mask_f32 = mask_plane.astype(np.float32)[:, :, np.newaxis] / 255.0

    if rgb_f32.shape[0] != target_size or rgb_f32.shape[1] != target_size:
        resized_rgb_u8 = np.asarray(
            Image.fromarray(image_u8, mode="RGB").resize(
                (target_size, target_size), Image.Resampling.BICUBIC
            ),
            dtype=np.uint8,
        )
        resized_mask_u8 = np.asarray(
            Image.fromarray(mask_plane, mode="L").resize(
                (target_size, target_size), Image.Resampling.BICUBIC
            ),
            dtype=np.uint8,
        )
        rgb_f32 = resized_rgb_u8.astype(np.float32) / 255.0
        mask_f32 = resized_mask_u8.astype(np.float32)[:, :, np.newaxis] / 255.0

    x = preprocess(rgb_f32, mask_f32)
    outputs = raw_engine._model(x)
    mx.eval(outputs)

    alpha_coarse = outputs["alpha_coarse"]
    fg_coarse = outputs["fg_coarse"]
    alpha_refined = outputs["alpha_final"]
    fg_refined = outputs["fg_final"]

    use_refiner = bool(getattr(raw_engine, "_use_refiner", True))
    if not use_refiner or refiner_scale == 0.0:
        alpha_out = alpha_coarse
        fg_out = fg_coarse
    elif refiner_scale == 1.0:
        alpha_out = alpha_refined
        fg_out = fg_refined
    else:
        s = refiner_scale
        alpha_out = alpha_coarse * (1.0 - s) + alpha_refined * s
        fg_out = fg_coarse * (1.0 - s) + fg_refined * s

    alpha = np.array(alpha_out[0], dtype=np.float32)
    fg = np.array(fg_out[0], dtype=np.float32)

    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]

    if alpha.shape[:2] != (original_h, original_w):
        alpha = _resize_float_image_bicubic(alpha, (original_w, original_h))
        fg = _resize_float_image_bicubic(fg, (original_w, original_h))

    return {
        "alpha": np.clip(alpha, 0.0, 1.0).astype(np.float32),
        "fg": np.clip(fg, 0.0, 1.0).astype(np.float32),
    }


class _MLXEngineAdapter:
    """Wraps CorridorKeyMLXEngine to match Torch process_frame() contract."""

    def __init__(self, raw_engine):
        self._engine = raw_engine
        logger.info("MLX adapter: despill/despeckle handled by adapter, not native MLX")

    def process_frame(
        self,
        image,
        mask_linear,
        refiner_scale=1.0,
        input_is_linear=False,
        fg_is_straight=True,
        despill_strength=1.0,
        auto_despeckle=True,
        despeckle_size=400,
        despeckle_dilation=25,
        despeckle_blur=5,
    ):
        """Delegate to MLX engine, then normalize output to Torch contract."""
        # corridorkey-mlx expects sRGB uint8 input even when input_is_linear=True.
        image_u8 = _prepare_mlx_image_u8(image, input_is_linear)

        if mask_linear.dtype != np.uint8:
            mask_u8 = (np.clip(mask_linear, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            mask_u8 = mask_linear

        # Squeeze mask to 2D for MLX
        if mask_u8.ndim == 3:
            mask_u8 = mask_u8[:, :, 0]

        float_outputs = _try_mlx_float_outputs(self._engine, image_u8, mask_u8, refiner_scale)
        if float_outputs is not None:
            return _assemble_mlx_output(
                alpha=float_outputs["alpha"],
                fg=float_outputs["fg"],
                source_image=image,
                input_is_linear=input_is_linear,
                despill_strength=despill_strength,
                auto_despeckle=auto_despeckle,
                despeckle_size=despeckle_size,
                despeckle_dilation=despeckle_dilation,
                despeckle_blur=despeckle_blur,
            )

        raw = self._engine.process_frame(
            image_u8,
            mask_u8,
            refiner_scale=refiner_scale,
            input_is_linear=input_is_linear,
            fg_is_straight=fg_is_straight,
            despill_strength=0.0,      # disable MLX stubs — adapter applies these
            auto_despeckle=False,
            despeckle_size=despeckle_size,
        )

        return _wrap_mlx_output(
            raw, image, input_is_linear, despill_strength, auto_despeckle,
            despeckle_size, despeckle_dilation, despeckle_blur,
        )


DEFAULT_MLX_TILE_SIZE = 512
DEFAULT_MLX_TILE_OVERLAP = 64


def create_engine(
    backend: str | None = None,
    device: str | None = None,
    img_size: int = DEFAULT_IMG_SIZE,
    optimization_mode: str = "auto",
    tile_size: int | None = DEFAULT_MLX_TILE_SIZE,
    overlap: int = DEFAULT_MLX_TILE_OVERLAP,
    on_status=None,
):
    """Factory: returns an engine with process_frame() matching the Torch contract.

    Args:
        backend: 'auto', 'torch', or 'mlx'. None means auto.
        device: Torch device string ('cuda', 'cpu', 'mps'). Ignored for MLX.
        img_size: Model inference resolution.
        optimization_mode: Torch only — 'auto', 'speed', or 'lowvram'.
        tile_size: MLX only — tile size for tiled inference (default 512).
            Set to None to disable tiling and use full-frame inference.
        overlap: MLX only — overlap pixels between tiles (default 64).
        on_status: Optional status callback for UI.
    """
    backend = resolve_backend(backend)

    if backend == "mlx":
        ckpt = _discover_checkpoint(MLX_EXT)
        from corridorkey_mlx import CorridorKeyMLXEngine  # type: ignore[import-not-found]

        try:
            raw_engine = CorridorKeyMLXEngine(
                str(ckpt), img_size=img_size, tile_size=tile_size, overlap=overlap,
            )
            mode = f"tiled (tile={tile_size}, overlap={overlap})" if tile_size else "full-frame"
        except TypeError:
            # Older corridorkey_mlx without tiled inference support
            raw_engine = CorridorKeyMLXEngine(str(ckpt), img_size=img_size)
            mode = "full-frame (tiling not supported by installed corridorkey_mlx)"
        logger.info("MLX engine loaded: %s [%s]", ckpt.name, mode)
        return _MLXEngineAdapter(raw_engine)
    else:
        ckpt = _discover_checkpoint(TORCH_EXT)
        from CorridorKeyModule.inference_engine import CorridorKeyEngine

        logger.info("Torch engine loaded: %s (device=%s)", ckpt.name, device)
        return CorridorKeyEngine(
            checkpoint_path=str(ckpt),
            device=device or "cpu",
            img_size=img_size,
            optimization_mode=optimization_mode,
            on_status=on_status,
        )

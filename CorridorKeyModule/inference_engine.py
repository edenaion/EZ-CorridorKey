import logging
import math
import os
import time
import types

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
from .core.model_transformer import GreenFormer
from .core import color_utils as cu

# ── Single source of truth for inference parameter defaults ──
# Every engine (CUDA, MLX) and the service layer import from here.
# Change a default ONCE, it propagates everywhere.
INFERENCE_DEFAULTS = {
    "refiner_scale": 1.0,
    "despill_strength": 0.5,
    "auto_despeckle": True,
    "despeckle_size": 400,
    "despeckle_dilation": 25,
    "despeckle_blur": 5,
    "source_passthrough": True,
    "edge_erode_px": None,
    "edge_blur_px": None,
}

# Optional accelerators — try-import once at module level so the wheels
# stay optional and users without them get the SDPA / FP16 fallbacks.
try:
    from sageattention import sageattn as _sageattn  # type: ignore
    _SAGE_AVAILABLE = True
except Exception:
    _sageattn = None
    _SAGE_AVAILABLE = False

try:
    from torchao.quantization import quantize_ as _ao_quantize  # type: ignore
    from torchao.quantization import NVFP4Config as _NVFP4Config  # type: ignore
    _TORCHAO_AVAILABLE = True
except Exception:
    _ao_quantize = None
    _NVFP4Config = None
    _TORCHAO_AVAILABLE = False


def _env_override(name: str) -> bool | None:
    """Read a tri-state env var: True / False / None (unset)."""
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def _device_capability(device: torch.device) -> tuple[int, int]:
    """(major, minor) compute capability, or (0, 0) if not CUDA."""
    if device.type != "cuda":
        return (0, 0)
    try:
        return torch.cuda.get_device_capability(device)
    except Exception:
        return (0, 0)


def _auto_enable_sage(device: torch.device) -> bool:
    """Auto-enable SageAttention on Ampere+ when installed.

    Override with CORRIDORKEY_USE_SAGE=0 to force off, =1 to force on
    even when the capability check would skip it.
    """
    override = _env_override("CORRIDORKEY_USE_SAGE")
    if override is not None:
        return override and _SAGE_AVAILABLE
    if not _SAGE_AVAILABLE:
        return False
    # Sage 2++ requires Ampere (SM 8.0+); Sage 3 auto-engages on Blackwell (SM 12+).
    major, _ = _device_capability(device)
    return major >= 8


def _auto_enable_nvfp4(device: torch.device) -> bool:
    """Auto-enable NVFP4 weight quantization on Blackwell when torchao is installed.

    Override with CORRIDORKEY_NVFP4=0 to force off, =1 to force on.
    """
    override = _env_override("CORRIDORKEY_NVFP4")
    if override is not None:
        return override and _TORCHAO_AVAILABLE
    if not _TORCHAO_AVAILABLE:
        return False
    # NVFP4 hardware lives on Blackwell 5th gen Tensor Cores (SM 12+)
    major, _ = _device_capability(device)
    return major >= 12


def _attention_kernel(device: torch.device):
    """Active attention kernel: sageattn when auto-enabled or forced, else SDPA."""
    return _sageattn if _auto_enable_sage(device) else F.scaled_dot_product_attention


def _patch_hiera_global_attention(hiera_model: nn.Module, device: torch.device) -> int:
    """Monkey-patch MaskUnitAttention.forward on global-attention blocks.

    Hiera's MaskUnitAttention creates Q/K/V with shape
    [B, heads, num_windows, N, head_dim]. When num_windows == 1
    (global attention), this 5-D non-contiguous tensor causes PyTorch's
    SDPA to silently fall back to the VRAM-hungry math backend.

    This patch forces Q/K/V to standard 4-D contiguous tensors, enabling
    FlashAttention and dropping VRAM usage per block dramatically.

    Credit: Jhe Kimchi (Discord contribution)
    """
    attn_fn = _attention_kernel(device)
    patched = 0

    for blk in hiera_model.blocks:
        attn = blk.attn

        # Only patch global attention blocks — windowed attention is fine
        if attn.use_mask_unit_attn:
            continue

        def _make_patched_forward(original_attn, kernel):
            def _patched_forward(self, x: torch.Tensor) -> torch.Tensor:
                B, N, _ = x.shape
                qkv = self.qkv(x)
                qkv = qkv.reshape(B, N, 3, self.heads, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
                q, k, v = qkv.unbind(0)             # each [B, heads, N, head_dim]

                if self.q_stride > 1:
                    q = q.view(
                        B, self.heads, self.q_stride, -1, self.head_dim
                    ).amax(dim=2)

                # Force contiguous layout so SDPA can use FlashAttention
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()

                x = kernel(q, k, v)
                x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
                x = self.proj(x)
                return x

            return types.MethodType(_patched_forward, original_attn)

        attn.forward = _make_patched_forward(attn, attn_fn)
        patched += 1

    return patched


def _try_nvfp4_quantize(model: nn.Module, device: torch.device) -> bool:
    """NVFP4 weight quantization — auto on Blackwell when torchao is installed."""
    if not _auto_enable_nvfp4(device):
        return False
    try:
        _ao_quantize(model, _NVFP4Config())
        logger.info("Model quantized to NVFP4")
        return True
    except Exception as e:
        logger.warning(f"NVFP4 quantization failed: {type(e).__name__}: {e}")
        return False


def _validate_sage_attention(device: torch.device, tolerance: float = 5e-2) -> bool:
    """Cross-check sageattn vs SDPA on a dummy tensor; return False if they diverge."""
    if not _SAGE_AVAILABLE:
        return False
    # Hiera Base Plus: 14 heads × 64 head_dim, 256 tokens at the smallest global block
    B, H, N, D = 1, 14, 256, 64
    try:
        torch.manual_seed(0)
        q = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
        with torch.inference_mode():
            ref = F.scaled_dot_product_attention(q, k, v)
            cand = _sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        diff = (ref.float() - cand.float()).abs().max().item()
        ok = diff <= tolerance
        logger.info(f"SageAttention validation: max|diff| vs SDPA = {diff:.3e} ({'OK' if ok else 'FAIL, falling back'})")
        return ok
    except Exception as e:
        logger.warning(f"SageAttention validation failed: {type(e).__name__}: {e}")
        return False

class CorridorKeyEngine:
    # VRAM threshold for optimization profile selection.
    # Below this: tiled refiner + selective compile. Above: full-frame compile.
    _VRAM_TILE_THRESHOLD_GB = 12

    # Optimization modes:
    #   'auto'  — detect VRAM and pick best strategy (default)
    #   'speed' — torch.compile, no tiling (12GB+ VRAM)
    #   'lowvram' — tiled refiner + compiled tile kernel (8GB GPUs)
    VALID_OPT_MODES = ('auto', 'speed', 'lowvram')

    def __init__(self, checkpoint_path, device='cuda', img_size=2048, use_refiner=True,
                 optimization_mode='auto', tile_overlap=128, on_status=None):
        logger.info("CorridorKeyEngine.__init__: begin")
        self.device = torch.device(device)
        self.img_size = img_size
        self.checkpoint_path = checkpoint_path
        self.use_refiner = use_refiner
        self.tile_overlap = tile_overlap
        self._on_status = on_status
        self._eager_model = None
        self._compiled_model = None
        self._compile_error = None

        # Allow env var override: CORRIDORKEY_OPT_MODE=speed|lowvram|auto
        env_mode = os.environ.get('CORRIDORKEY_OPT_MODE', '').lower()
        if env_mode in self.VALID_OPT_MODES:
            optimization_mode = env_mode
            logger.info(f"Optimization mode override from env: {env_mode}")

        # Resolve optimization profile.
        # Low-VRAM mode keeps tiling, but graph-breaks the tile scheduler and
        # compiles the fixed-shape tile CNN separately.
        #
        # NOTE: _get_vram_gb() uses pynvml first (driver-level, no CUDA
        # context) so it's safe during model handoff (GVM -> inference).
        # The old torch.cuda.get_device_properties() call stalled after
        # GVM teardown — pynvml doesn't have that problem.
        # MPS (Apple Silicon): always use lowvram mode — unified memory is
        # shared with the OS, and torch.compile/Triton doesn't support Metal.
        is_mps = self.device.type == 'mps'
        if is_mps:
            optimization_mode = 'lowvram'
            logger.info("MPS device detected — forcing low-VRAM mode (no torch.compile on Metal)")

        if optimization_mode == 'speed':
            self.tile_size = 0
            self._use_compile = True
            logger.info("Optimization: speed mode (torch.compile, no tiling)")
        elif optimization_mode == 'lowvram':
            self.tile_size = 512
            self._use_compile = not is_mps  # Triton/inductor doesn't support MPS
            logger.info(f"Optimization: low-VRAM mode (tiled refiner 512x512"
                        f"{'' if self._use_compile else ', no torch.compile'})")
        else:  # auto
            logger.info("Optimization: auto mode - probing VRAM...")
            vram_gb = self._get_vram_gb()
            if vram_gb > 0 and vram_gb < self._VRAM_TILE_THRESHOLD_GB:
                self.tile_size = 512
                self._use_compile = True
                logger.info(f"Optimization: auto -> low-VRAM mode "
                            f"({vram_gb:.1f} GB < {self._VRAM_TILE_THRESHOLD_GB} GB threshold)")
            else:
                self.tile_size = 0
                self._use_compile = True
                logger.info(f"Optimization: auto -> speed mode "
                            f"({vram_gb:.1f} GB >= {self._VRAM_TILE_THRESHOLD_GB} GB threshold)")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        logger.info("CorridorKeyEngine.__init__: entering _load_model")
        self.model = self._load_model()

    @staticmethod
    def _get_vram_gb() -> float:
        """Return total GPU VRAM in GB.

        Prefer NVML (no CUDA context calls) and fall back to torch.cuda.
        """
        logger.debug("_get_vram_gb: entering")
        try:
            import pynvml
            logger.debug("_get_vram_gb: pynvml imported, calling nvmlInit...")
            pynvml.nvmlInit()
            logger.debug("_get_vram_gb: nvmlInit done, getting handle...")
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            logger.debug("_get_vram_gb: got handle, querying memory...")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram = mem.total / (1024 ** 3)
            logger.debug(f"_get_vram_gb: pynvml reports {vram:.1f} GB")
            return vram
        except Exception as e:
            logger.debug(f"_get_vram_gb: pynvml failed: {e}")

        try:
            logger.debug("_get_vram_gb: falling back to torch.cuda...")
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                logger.debug(f"_get_vram_gb: torch.cuda reports {vram:.1f} GB")
                return vram
        except Exception as e:
            logger.debug(f"_get_vram_gb: torch.cuda failed: {e}")
        logger.debug("_get_vram_gb: all probes failed, returning 0")
        return 0.0
        
    def _status(self, msg: str) -> None:
        """Emit status to UI callback and log."""
        logger.info(msg)
        if self._on_status:
            self._on_status(msg)

    @staticmethod
    def _iter_exception_chain(exc: Exception):
        """Yield an exception plus its cause/context chain without looping."""
        seen = set()
        current = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            yield current
            current = current.__cause__ or current.__context__

    def _has_compiled_artifacts(self) -> bool:
        """Return True if this engine may still touch compiled Torch/Triton code."""
        refiner = getattr(self._eager_model, 'refiner', None)
        tile_kernel = getattr(refiner, '_compiled_process_tile', None) if refiner is not None else None
        return self._compiled_model is not None or tile_kernel is not None

    def _is_compile_failure(self, exc: Exception) -> bool:
        """Best-effort filter for torch.compile/Triton/Inductor runtime failures."""
        markers = (
            'triton',
            'torchinductor',
            'torch._inductor',
            'torch._dynamo',
            'backendcompilerfailed',
            'loweringexception',
            'asynccompile',
            'compileworker',
            'inductor',
            'compiler: cl is not found',
            'compiler: cl.exe is not found',
        )
        for err in self._iter_exception_chain(exc):
            text = f"{type(err).__module__} {type(err).__name__} {err}".lower()
            if any(marker in text for marker in markers):
                return True
        return False

    def _disable_compile(self, exc: Exception) -> None:
        """Permanently fall back to eager mode for this engine instance."""
        self._compile_error = f"{type(exc).__name__}: {exc}"
        self._use_compile = False
        self._compiled_model = None

        if self._eager_model is not None:
            refiner = getattr(self._eager_model, 'refiner', None)
            if refiner is not None and getattr(refiner, '_compiled_process_tile', None) is not None:
                refiner._compiled_process_tile = None
            self.model = self._eager_model

        try:
            import torch._dynamo as _dynamo
            _dynamo.reset()
        except Exception as reset_error:
            logger.debug(f"dynamo reset skipped after compile failure: {reset_error}")

        try:
            if self.device.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as cache_error:
            logger.debug(f"cuda cache clear skipped after compile failure: {cache_error}")

        logger.warning(
            "Compile runtime failed (%s); falling back to eager mode for this session",
            self._compile_error,
        )

    def _forward_model(self, inp_t: torch.Tensor, refiner_scale_t: torch.Tensor):
        """Run the model once, retrying eagerly only for compile-specific failures."""
        try:
            return self.model(inp_t, refiner_scale=refiner_scale_t)
        except Exception as e:
            if not self._has_compiled_artifacts() or not self._is_compile_failure(e):
                raise
            self._disable_compile(e)
            return self.model(inp_t, refiner_scale=refiner_scale_t)

    def _load_model(self):
        import time as _time
        import logging as _logging

        def _diag(msg):
            """Force-flush diagnostic — visible in log file immediately."""
            logger.info(msg)
            for h in _logging.getLogger().handlers:
                h.flush()

        _diag("_load_model ENTERED")
        logger.info(f"Loading CorridorKey from {self.checkpoint_path}...")

        # Step 1: Initialize backbone
        _diag("Step 1: GreenFormer init...")
        self._status("Initializing model backbone...")
        t0 = _time.monotonic()
        model = GreenFormer(encoder_name='hiera_base_plus_224.mae_in1k_ft_in1k',
                          img_size=self.img_size,
                          use_refiner=self.use_refiner)
        _diag(f"Step 1 done: {_time.monotonic() - t0:.1f}s")
        logger.info(f"GreenFormer init: {_time.monotonic() - t0:.1f}s")

        # Step 2: Move to GPU
        _diag("Step 2: model.to(device)...")
        self._status("Moving model to GPU...")
        t0 = _time.monotonic()
        model = model.to(self.device)
        model.eval()
        # Convert weights to fp16, saves ~450 MB VRAM.
        # Inference already runs in fp16 via autocast, so this is lossless.
        if self.device.type != 'cpu':
            model.half()
            _diag("Model weights converted to fp16")
        _diag(f"Step 2 done: {_time.monotonic() - t0:.1f}s")
        logger.info(f"Model to device: {_time.monotonic() - t0:.1f}s")

        # Step 3: Load checkpoint
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        _diag("Step 3: torch.load checkpoint...")
        self._status("Loading checkpoint weights...")
        t0 = _time.monotonic()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint)
        _diag(f"Step 3 done: {_time.monotonic() - t0:.1f}s")
        logger.info(f"Checkpoint loaded: {_time.monotonic() - t0:.1f}s")

        # Step 4: Fix Compiled Model Prefix & Handle PosEmbed Mismatch
        new_state_dict = {}
        model_state = model.state_dict()

        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                k = k[10:]

            # Check for PosEmbed Mismatch
            if 'pos_embed' in k and k in model_state:
                if v.shape != model_state[k].shape:
                    logger.warning(f"PosEmbed shape mismatch: resizing {k} from {v.shape} to {model_state[k].shape}")
                    N_src = v.shape[1]
                    N_dst = model_state[k].shape[1]
                    C = v.shape[2]

                    grid_src = int(math.sqrt(N_src))
                    grid_dst = int(math.sqrt(N_dst))

                    v_img = v.permute(0, 2, 1).view(1, C, grid_src, grid_src)
                    v_resized = F.interpolate(v_img, size=(grid_dst, grid_dst), mode='bicubic', align_corners=False)
                    v = v_resized.flatten(2).transpose(1, 2)

            new_state_dict[k] = v

        _diag("Step 4: load_state_dict...")
        self._status("Loading state dict...")
        t0 = _time.monotonic()
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0:
            logger.warning(f"Missing keys in checkpoint: {missing}")
        if len(unexpected) > 0:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected}")
        _diag(f"Step 4 done: {_time.monotonic() - t0:.1f}s")
        logger.info(f"State dict loaded: {_time.monotonic() - t0:.1f}s")

        # Step 4b: NVFP4 quantization on Blackwell. Must run AFTER load_state_dict.
        _try_nvfp4_quantize(model, self.device)

        # TF32 and cuDNN benchmark settings are now applied at app startup
        # in ui/app.py::_configure_runtime_backends() so they take effect
        # before the first inference, not just after lazy model load.

        # Step 5: Hiera attention patch.
        # _attention_kernel(device) auto-picks sageattn on Ampere+ when
        # installed, SDPA otherwise. If sageattn diverges from SDPA on these
        # tensor shapes we set CORRIDORKEY_USE_SAGE=0 so subsequent passes
        # take the SDPA path even though Sage is "available".
        if _auto_enable_sage(self.device) and not _validate_sage_attention(self.device):
            os.environ["CORRIDORKEY_USE_SAGE"] = "0"
            logger.warning("SageAttention disabled for this session — falling back to SDPA")

        self._status("Patching attention blocks...")
        t0 = _time.monotonic()
        try:
            hiera = model.encoder.model
            n_patched = _patch_hiera_global_attention(hiera, self.device)
            kernel_name = "sageattn" if _auto_enable_sage(self.device) else "SDPA"
            logger.info(f"Hiera attention patch: {n_patched} blocks via {kernel_name} ({_time.monotonic() - t0:.1f}s)")
        except Exception as e:
            logger.warning(f"Hiera attention patch failed: {type(e).__name__}: {e}")

        # Configure tiled refiner for VRAM-constrained processing.
        if self.tile_size > 0 and hasattr(model, 'refiner') and model.refiner is not None:
            model.refiner._tile_size = self.tile_size
            model.refiner._tile_overlap = self.tile_overlap
            logger.info(f"Tiled refiner: {self.tile_size}x{self.tile_size} tiles, {self.tile_overlap}px overlap")

        eager_model = model
        self._eager_model = eager_model
        self._compiled_model = None

        # Step 6: torch.compile (Triton JIT — slow on first run)
        if self._use_compile:
            self._status("Compiling model (first run may take a minute)...")
            import subprocess
            import sys

            if sys.platform == 'win32' and not getattr(subprocess.Popen, '_corridorkey_no_window', False):
                _orig_popen_init = subprocess.Popen.__init__

                def _silent_popen_init(self, *args, **kwargs):
                    kwargs.setdefault('creationflags', subprocess.CREATE_NO_WINDOW)
                    _orig_popen_init(self, *args, **kwargs)

                subprocess.Popen.__init__ = _silent_popen_init
                subprocess.Popen._corridorkey_no_window = True

            try:
                import triton  # noqa: F401
                # Frozen builds: if triton_windows .dist-info was missing,
                # the runtime hook flagged it.  Patch the backends dict now.
                if os.environ.get('_TRITON_PATCH_BACKENDS') == '1':
                    from triton.backends import backends, Backend, _find_concrete_subclasses
                    from triton.backends.driver import DriverBase
                    from triton.backends.compiler import BaseBackend
                    if 'nvidia' not in backends:
                        import triton.backends.nvidia.compiler as _nv_compiler
                        import triton.backends.nvidia.driver as _nv_driver
                        backends['nvidia'] = Backend(
                            _find_concrete_subclasses(_nv_compiler, BaseBackend),
                            _find_concrete_subclasses(_nv_driver, DriverBase),
                        )
                        logger.info("Patched triton backends: nvidia registered")
                    del os.environ['_TRITON_PATCH_BACKENDS']
            except Exception as e:
                self._use_compile = False
                logger.warning(f"Triton unavailable, skipping torch.compile: {type(e).__name__}: {e}")

            if self._use_compile and self.tile_size > 0 and hasattr(eager_model, 'refiner') and eager_model.refiner is not None:
                try:
                    t0 = _time.monotonic()
                    eager_model.refiner.compile_tile_kernel()
                    logger.info(f"Refiner tile compile: {_time.monotonic() - t0:.1f}s")
                except Exception as e:
                    logger.warning(
                        f"Refiner tile kernel compile failed (falling back to eager tiles): "
                        f"{type(e).__name__}: {e}"
                    )

            if self._use_compile:
                try:
                    # Enable FX graph caching — skips Triton tracing on subsequent runs.
                    # First launch pays the full compile cost; every launch after is near-instant.
                    try:
                        torch._inductor.config.fx_graph_cache = True
                        logger.info("FX graph cache enabled (subsequent launches will skip recompilation)")
                    except AttributeError:
                        logger.debug("FX graph cache config not available in this PyTorch version")

                    t0 = _time.monotonic()
                    self._compiled_model = torch.compile(eager_model, fullgraph=False)
                    model = self._compiled_model
                    logger.info(f"torch.compile complete: {_time.monotonic() - t0:.1f}s")
                except Exception as e:
                    self._compiled_model = None
                    logger.warning(f"torch.compile failed (falling back to eager): {type(e).__name__}: {e}")

        self._status("Model ready")
        return model

    _D = INFERENCE_DEFAULTS

    @torch.no_grad()
    def process_frame(self, image, mask_linear, refiner_scale=_D["refiner_scale"], input_is_linear=False, fg_is_straight=True, despill_strength=_D["despill_strength"], auto_despeckle=_D["auto_despeckle"], despeckle_size=_D["despeckle_size"], despeckle_dilation=_D["despeckle_dilation"], despeckle_blur=_D["despeckle_blur"], source_passthrough=_D["source_passthrough"], edge_erode_px=_D["edge_erode_px"], edge_blur_px=_D["edge_blur_px"]):
        """
        Process a single frame.
        Args:
            image: Numpy array [H, W, 3] (0.0-1.0 or 0-255).
                   - If input_is_linear=False (Default): Assumed sRGB.
                   - If input_is_linear=True: Assumed Linear.
            mask_linear: Numpy array [H, W] or [H, W, 1] (0.0-1.0). Assumed Linear.
            refiner_scale: Multiplier for Refiner Deltas (default 1.0).
            input_is_linear: bool. If True, resizes in Linear then transforms to sRGB.
                             If False, resizes in sRGB (standard).
            fg_is_straight: bool. If True, assumes FG output is Straight (unpremultiplied).
                            If False, assumes FG output is Premultiplied.
            despill_strength: float. 0.0 to 1.0 multiplier for the despill effect.
            auto_despeckle: bool. If True, cleans up small disconnected components from the predicted alpha matte.
            despeckle_size: int. Minimum number of consecutive pixels required to keep an island.
            source_passthrough: bool. If True, passes original source pixels through in opaque interior
                                regions instead of using the model's fg prediction. Only the edge
                                transition band uses model fg. Preserves full source quality.
            edge_erode_px: int. Pixels to erode the interior mask inward. Safety buffer to avoid
                           using original pixels where green spill might contaminate.
            edge_blur_px: int. Gaussian blur radius for the transition blend between source and
                          model fg. Controls smoothness of the seam.
        Returns:
             dict: {'alpha': np, 'fg': np (sRGB), 'comp': np (sRGB on Gray)}
        """
        t0 = time.monotonic()

        # 1. Inputs Check & Normalization
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            
        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0
            
        h, w = image.shape[:2]
        
        # Ensure Mask Shape
        if mask_linear.ndim == 2:
            mask_linear = mask_linear[:, :, np.newaxis]
            
        # 2. Resize to Model Size
        # If input is linear, we resize in linear to preserve energy/highlights,
        # THEN convert to sRGB for the model.
        if input_is_linear:
             # Resize in Linear
             img_resized_lin = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
             # Convert to sRGB for Model
             img_resized = cu.linear_to_srgb(img_resized_lin)
        else:
             # Standard sRGB Resize
             img_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        mask_resized = cv2.resize(mask_linear, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

        # 3. Normalize (ImageNet)
        # Model expects sRGB input normalized
        img_norm = (img_resized - self.mean) / self.std

        # 4. Prepare Tensor
        inp_np = np.concatenate([img_norm, mask_resized], axis=-1) # [H, W, 4]
        inp_t = torch.from_numpy(inp_np.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        
        # 5. Inference
        refiner_scale_t = inp_t.new_tensor(refiner_scale)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            out = self._forward_model(inp_t, refiner_scale_t)
            
        pred_alpha = out['alpha']
        pred_fg = out['fg'] # Output is sRGB (Sigmoid)
        
        # 6. Post-Process (Resize Back to Original Resolution)
        # We use Lanczos4 for high-quality resampling to minimize blur when going back to 4K/Original.
        res_alpha = pred_alpha[0].permute(1, 2, 0).float().cpu().numpy()
        res_fg = pred_fg[0].permute(1, 2, 0).float().cpu().numpy()
        res_alpha = cv2.resize(res_alpha, (w, h), interpolation=cv2.INTER_LANCZOS4)
        res_fg = cv2.resize(res_fg, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        if res_alpha.ndim == 2: res_alpha = res_alpha[:, :, np.newaxis]

        # --- REFINER ADDITIVE GUARD ---
        # CorridorKey's refiner was trained to tighten rough green-screen
        # alphas, so when it is fed an already-good input mask (e.g. from
        # GVM Auto or VideoMaMa), it can erode wispy hair strands the guide
        # had captured correctly. Take the per-pixel maximum between the
        # refiner output and the original input guide: the refiner may
        # *add* confidence (extend into areas the guide missed) but may
        # never *subtract* confidence the guide already had. mask_linear is
        # already at full (h, w) resolution and in [0, 1] float space, so
        # no resize or normalization is needed here.
        res_alpha = np.maximum(res_alpha, mask_linear)

        # --- SOURCE PASSTHROUGH ---
        # Interior pixels (face, body, clothes — everywhere alpha ≈ 1.0) are passed
        # through from the original source frame untouched.  The model's fg prediction
        # is only used in the edge transition band where it handles green-screen
        # separation, hair strands, and semi-transparency.
        if source_passthrough:
            if input_is_linear:
                original_srgb = cu.linear_to_srgb(image)
            else:
                original_srgb = image
            res_fg = cu.source_passthrough(
                original_srgb, res_fg, res_alpha,
                erode_px=edge_erode_px, blur_px=edge_blur_px
            )

        # --- ADVANCED COMPOSITING ---

        # A. Clean Matte (Auto-Despeckle)
        if auto_despeckle:
            processed_alpha = cu.clean_matte(res_alpha, area_threshold=despeckle_size, dilation=despeckle_dilation, blur_size=despeckle_blur)
        else:
            processed_alpha = res_alpha

        # B. Despill FG
        # res_fg is sRGB (with source passthrough blended in where applicable).
        fg_despilled = cu.despill(res_fg, green_limit_mode='average', strength=despill_strength)

        # C. Convert to linear for output assembly.
        fg_despilled_lin = cu.srgb_to_linear(fg_despilled)

        # D. Premultiply and pack linear RGBA for NLE/compositor interchange.
        fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)
        processed_rgba = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)

        # ----------------------------
        
        # 7. Composite (on Checkerboard) for checking
        # Generate Dark/Light Gray Checkerboard (in sRGB, convert to Linear)
        bg_srgb = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
        bg_lin = cu.srgb_to_linear(bg_srgb)
        
        if fg_is_straight:
             comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha) 
        else:
             # If premultiplied model, we shouldn't multiply again (though our pipeline forces straight)
             comp_lin = cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha)
             
        comp_srgb = cu.linear_to_srgb(comp_lin)
        
        logger.debug(f"process_frame: {h}x{w} in {time.monotonic() - t0:.3f}s")

        return {
            'alpha': res_alpha,        # Linear, Raw Prediction
            'fg': res_fg,              # sRGB, Raw Prediction (Straight)
            'comp': comp_srgb,         # sRGB, Composite
            'processed': processed_rgba,  # Linear/Straight RGBA, garbage matted + despilled
        }

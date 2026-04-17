"""CorridorKeyService — clean backend API for the UI and CLI.

This module wraps all processing logic into a service layer. The UI never
calls inference engines directly — it calls methods here which handle
validation, state transitions, and error reporting.

Model Residency Policy:
    Only ONE heavy model is loaded at a time. Before loading a new
    model type, the previous is unloaded and VRAM freed via
    torch.cuda.empty_cache(). This prevents OOM on 24GB cards.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional

from ..clip_state import ClipEntry, ClipState, scan_clips_dir
from ..job_queue import GPUJobQueue
from .model_manager import ModelManagerMixin, _ActiveModel
from .frame_ops import FrameOpsMixin
from .inference import InferenceMixin
from .pipelines import PipelinesMixin

logger = logging.getLogger(__name__)

# Pull defaults from CorridorKeyModule
from CorridorKeyModule.inference_engine import INFERENCE_DEFAULTS as _D


@dataclass
class InferenceParams:
    """Frozen parameters for a single inference job.

    Defaults are pulled from INFERENCE_DEFAULTS (single source of truth
    in CorridorKeyModule.inference_engine).  Change a default there and
    it propagates to every engine path and the service layer.
    """
    input_is_linear: bool = False
    despill_strength: float = _D["despill_strength"]
    auto_despeckle: bool = _D["auto_despeckle"]
    despeckle_size: int = _D["despeckle_size"]
    despeckle_dilation: int = _D["despeckle_dilation"]
    despeckle_blur: int = _D["despeckle_blur"]
    refiner_scale: float = _D["refiner_scale"]
    source_passthrough: bool = _D["source_passthrough"]
    edge_erode_px: int = _D["edge_erode_px"]
    edge_blur_px: int = _D["edge_blur_px"]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'InferenceParams':
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class OutputConfig:
    """Which output types to produce and their format."""
    fg_enabled: bool = True
    fg_format: str = "exr"   # "exr" or "png"
    matte_enabled: bool = True
    matte_format: str = "exr"
    comp_enabled: bool = True
    comp_format: str = "png"
    processed_enabled: bool = True
    processed_format: str = "exr"
    exr_compression: str = "dwab"  # "dwab", "piz", "zip", or "none"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'OutputConfig':
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def enabled_outputs(self) -> list[str]:
        """Return list of enabled output names for manifest."""
        out = []
        if self.fg_enabled:
            out.append("fg")
        if self.matte_enabled:
            out.append("matte")
        if self.comp_enabled:
            out.append("comp")
        if self.processed_enabled:
            out.append("processed")
        return out


@dataclass
class FrameResult:
    """Result summary for a single processed frame (no numpy in this struct)."""
    frame_index: int
    input_stem: str
    success: bool
    warning: Optional[str] = None


class CorridorKeyService(
    ModelManagerMixin,
    FrameOpsMixin,
    InferenceMixin,
    PipelinesMixin,
):
    """Main backend service — scan, validate, process, write.

    Usage:
        service = CorridorKeyService()
        clips = service.scan_clips("/path/to/ClipsForInference")
        ready = service.get_clips_by_state(clips, ClipState.READY)

        for clip in ready:
            params = InferenceParams(despill_strength=0.8)
            service.run_inference(clip, params, on_progress=my_callback)
    """

    # Model resolution options — user-facing setting.
    # 2048 = full quality (matches OG CK), 1024 = faster / less memory.
    VALID_MODEL_RESOLUTIONS = (1024, 2048)

    def __init__(self):
        self._engine_pool: list = []
        self._pool_size: int = 1
        self._model_resolution: int = 2048  # default: full quality
        self._inference_backend: str = "auto"  # auto | mlx | torch (macOS only)
        self._gvm_processor = None
        self._sam2_tracker = None
        self._videomama_pipeline = None
        self._matanyone2_processor = None
        self._birefnet_processor = None
        self._active_model = _ActiveModel.NONE
        self._device: str = 'cpu'
        self._job_queue: Optional[GPUJobQueue] = None
        self._sam2_model_id: str = "facebook/sam2.1-hiera-base-plus"
        # GPU mutex — serializes ALL model operations
        self._gpu_lock = threading.Lock()
        # Gated model switch — prevents starvation during model switch
        self._inference_active: int = 0
        self._switch_pending: bool = False
        self._gate_lock = threading.Lock()
        self._inference_idle = threading.Event()
        self._inference_idle.set()

    @property
    def job_queue(self) -> GPUJobQueue:
        """Lazy-init GPU job queue (only needed when UI is running)."""
        if self._job_queue is None:
            self._job_queue = GPUJobQueue()
        return self._job_queue

    @property
    def sam2_model_id(self) -> str:
        """Current SAM2 checkpoint preference."""
        return self._sam2_model_id

    def set_sam2_model(self, model_id: str) -> None:
        """Update the SAM2 checkpoint used for future tracking jobs."""
        if not model_id or model_id == self._sam2_model_id:
            return
        logger.info("SAM2 model preference changed: %s -> %s", self._sam2_model_id, model_id)
        self._sam2_model_id = model_id
        if self._sam2_tracker is not None:
            self._safe_offload(self._sam2_tracker)
            self._sam2_tracker = None
            if self._active_model == _ActiveModel.SAM2:
                self._active_model = _ActiveModel.NONE
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logger.debug("CUDA cache clear skipped after SAM2 model switch", exc_info=True)

    @property
    def model_resolution(self) -> int:
        """Current model inference resolution (1024 or 2048)."""
        return self._model_resolution

    def set_model_resolution(self, res: int) -> None:
        """Set model inference resolution. Requires engine reload to take effect."""
        if res not in self.VALID_MODEL_RESOLUTIONS:
            logger.warning("Invalid model resolution %d, ignoring", res)
            return
        if res != self._model_resolution:
            logger.info("Model resolution: %d -> %d (engine reload required)", self._model_resolution, res)
            self._model_resolution = res
            # Clear engine pool — next inference will rebuild at new resolution
            for eng in self._engine_pool:
                self._safe_offload(eng)
            self._engine_pool.clear()
            if self._active_model == _ActiveModel.INFERENCE:
                self._active_model = _ActiveModel.NONE
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def set_inference_backend(self, backend: str) -> None:
        """Change the inference backend (macOS: auto/mlx/torch). Drops the engine pool
        so the next inference rebuilds on the new backend without a restart.
        """
        backend = (backend or "auto").lower()
        if backend not in ("auto", "mlx", "torch"):
            logger.warning("Invalid inference backend %r, ignoring", backend)
            return
        if backend == self._inference_backend:
            return
        logger.info("Inference backend: %s -> %s (engine reload required)", self._inference_backend, backend)
        self._inference_backend = backend
        for eng in self._engine_pool:
            self._safe_offload(eng)
        self._engine_pool.clear()
        if self._active_model == _ActiveModel.INFERENCE:
            self._active_model = _ActiveModel.NONE
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def set_pool_size(self, n: int) -> None:
        """Set the number of parallel inference engines."""
        n = max(1, n)
        if n != self._pool_size:
            logger.info("Engine pool size: %d -> %d", self._pool_size, n)
            old_size = self._pool_size
            self._pool_size = n
            # Trim excess engines to free VRAM immediately
            if n < old_size and len(self._engine_pool) > n:
                excess = self._engine_pool[n:]
                self._engine_pool = self._engine_pool[:n]
                for eng in excess:
                    del eng
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                logger.info("Trimmed %d excess engine(s), freed VRAM", len(excess))

    def _begin_inference(self) -> None:
        """Mark an inference session as active (gated model switch)."""
        with self._gate_lock:
            if self._switch_pending:
                # Model switch is waiting — block until it completes
                pass
            self._inference_active += 1
            self._inference_idle.clear()
        # If switch_pending, wait outside gate_lock to avoid deadlock
        while self._switch_pending:
            time.sleep(0.05)

    def _end_inference(self) -> None:
        """Mark an inference session as finished."""
        with self._gate_lock:
            self._inference_active -= 1
            if self._inference_active <= 0:
                self._inference_active = 0
                self._inference_idle.set()

    # --- Device Utilities ---

    def detect_device(self) -> str:
        """Detect best available compute device (CUDA > MPS > CPU)."""
        try:
            import torch
            if torch.cuda.is_available():
                self._device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = 'mps'
                logger.info("Apple MPS acceleration available")
            else:
                self._device = 'cpu'
                logger.warning("No GPU acceleration available — using CPU (will be very slow)")
        except ImportError:
            self._device = 'cpu'
            logger.warning("PyTorch not installed — using CPU")
        logger.info(f"Compute device: {self._device}")
        return self._device

    def get_vram_info(self) -> dict[str, float]:
        """Get GPU VRAM info in GB. Returns empty dict if not CUDA."""
        try:
            import torch
            if not torch.cuda.is_available():
                return {}
            props = torch.cuda.get_device_properties(0)
            total_bytes = props.total_mem
            reserved = torch.cuda.memory_reserved(0)
            return {
                'total': total_bytes / (1024**3),
                'reserved': reserved / (1024**3),
                'allocated': torch.cuda.memory_allocated(0) / (1024**3),
                'free': (total_bytes - reserved) / (1024**3),
                'name': torch.cuda.get_device_name(0),
            }
        except Exception as e:
            logger.debug(f"VRAM query failed: {e}")
            return {}

    # --- Clip Scanning ---

    def scan_clips(
        self, clips_dir: str, allow_standalone_videos: bool = True,
    ) -> list[ClipEntry]:
        """Scan a directory for clip folders."""
        return scan_clips_dir(clips_dir, allow_standalone_videos=allow_standalone_videos)

    def get_clips_by_state(
        self,
        clips: list[ClipEntry],
        state: ClipState,
    ) -> list[ClipEntry]:
        """Filter clips by state."""
        return [c for c in clips if c.state == state]

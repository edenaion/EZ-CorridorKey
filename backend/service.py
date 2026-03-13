"""CorridorKeyService — clean backend API for the UI and CLI.

This module wraps all processing logic from clip_manager.py into a
service layer. The UI never calls inference engines directly — it
calls methods here which handle validation, state transitions, and
error reporting.

Model Residency Policy:
    Only ONE heavy model is loaded at a time. Before loading a new
    model type, the previous is unloaded and VRAM freed via
    torch.cuda.empty_cache(). This prevents OOM on 24GB cards.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import importlib
import gc
import logging
import queue
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

# Enable OpenEXR support (must be before cv2 import)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

from .clip_state import (
    ClipAsset,
    ClipEntry,
    ClipState,
    MASK_TRACK_MANIFEST,
    mask_sequence_is_videomama_ready,
    scan_clips_dir,
)
from .annotation_prompts import load_annotation_prompt_frames
from .errors import (
    CorridorKeyError,
    FrameReadError,
    GPURequiredError,
    WriteFailureError,
    JobCancelledError,
)
from .validators import (
    validate_frame_counts,
    validate_frame_read,
    validate_write,
    ensure_output_dirs,
)
from .frame_io import (
    write_exr,
    read_image_frame,
    read_mask_frame,
    read_video_frame_at,
    read_video_frames,
    read_video_mask_at,
)
from .job_queue import GPUJob, GPUJobQueue

logger = logging.getLogger(__name__)


def _configure_runtime_warnings() -> None:
    """Hide non-actionable NVML deprecation chatter during startup/runtime checks."""
    warnings.filterwarnings(
        "ignore",
        message=r"The pynvml package is deprecated\..*",
        category=FutureWarning,
    )


_configure_runtime_warnings()

# Project paths — frozen-build aware
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_matanyone2_processor_class():
    """Import MatAnyone2Processor from either the new or legacy module layout.

    Supported layouts:
    - modules/MatAnyone2Module/wrapper.py  -> modules.MatAnyone2Module.wrapper
    - MatAnyone2Module/wrapper.py          -> MatAnyone2Module.wrapper
    """
    candidates = (
        ("modules.MatAnyone2Module.wrapper", BASE_DIR),
        ("MatAnyone2Module.wrapper", os.path.join(BASE_DIR, "modules")),
        ("MatAnyone2Module.wrapper", BASE_DIR),
    )
    missing_roots = (
        os.path.join(BASE_DIR, "modules", "MatAnyone2Module"),
        os.path.join(BASE_DIR, "MatAnyone2Module"),
    )
    last_error: ModuleNotFoundError | None = None

    for module_name, path_entry in candidates:
        if path_entry and path_entry not in sys.path:
            sys.path.insert(0, path_entry)
        try:
            module = importlib.import_module(module_name)
            return module.MatAnyone2Processor
        except ModuleNotFoundError as exc:
            # Only fall through on the layout module itself missing. If an inner
            # dependency is missing, bubble that error up unchanged.
            if exc.name not in {
                "modules",
                "modules.MatAnyone2Module",
                "modules.MatAnyone2Module.wrapper",
                "MatAnyone2Module",
                "MatAnyone2Module.wrapper",
            }:
                raise
            last_error = exc

    expected = " or ".join(missing_roots)
    raise ModuleNotFoundError(
        f"MatAnyone2 module not found. Expected {expected}"
    ) from last_error

class _ActiveModel(Enum):
    """Tracks which heavy model is currently loaded in VRAM."""
    NONE = "none"
    INFERENCE = "inference"
    GVM = "gvm"
    SAM2 = "sam2"
    VIDEOMAMA = "videomama"
    MATANYONE2 = "matanyone2"


@dataclass
class InferenceParams:
    """Frozen parameters for a single inference job."""
    input_is_linear: bool = False
    despill_strength: float = 1.0  # 0.0 to 1.0
    auto_despeckle: bool = True
    despeckle_size: int = 400
    despeckle_dilation: int = 25   # clean_matte dilation radius
    despeckle_blur: int = 5        # clean_matte blur kernel half-size
    refiner_scale: float = 1.0

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


def export_masks_headless(clip: "ClipEntry") -> str | None:
    """Export annotation masks for a clip without requiring the UI viewer.

    Loads annotations from disk, determines frame dimensions from the first
    input frame, and calls AnnotationModel.export_masks().

    Returns:
        Path to VideoMamaMaskHint directory, or None if no annotations found.
    """
    from ui.widgets.annotation_overlay import AnnotationModel

    model = AnnotationModel()
    model.load(clip.root_path)

    if not model.has_annotations():
        return None

    if clip.input_asset is None or clip.input_asset.asset_type != "sequence":
        return None

    frame_files = clip.input_asset.get_frame_files()
    if not frame_files:
        return None

    stems = [os.path.splitext(f)[0] for f in frame_files]

    # Get dimensions from first input frame
    first_path = os.path.join(clip.input_asset.path, frame_files[0])
    sample = read_image_frame(first_path)
    if sample is None:
        return None
    h, w = sample.shape[:2]

    # Respect in/out range
    start_idx = 0
    if clip.in_out_range:
        lo = clip.in_out_range.in_point
        hi = clip.in_out_range.out_point
        stems = stems[lo:hi + 1]
        start_idx = lo

    return model.export_masks(clip.root_path, stems, w, h, start_index=start_idx)


class CorridorKeyService:
    """Main backend service — scan, validate, process, write.

    Usage:
        service = CorridorKeyService()
        clips = service.scan_clips("/path/to/ClipsForInference")
        ready = service.get_clips_by_state(clips, ClipState.READY)

        for clip in ready:
            params = InferenceParams(despill_strength=0.8)
            service.run_inference(clip, params, on_progress=my_callback)
    """

    def __init__(self):
        self._engine_pool: list = []
        self._pool_size: int = 1
        self._gvm_processor = None
        self._sam2_tracker = None
        self._videomama_pipeline = None
        self._matanyone2_processor = None
        self._active_model = _ActiveModel.NONE
        self._device: str = 'cpu'
        self._job_queue: Optional[GPUJobQueue] = None
        self._sam2_model_id: str = "facebook/sam2.1-hiera-base-plus"
        # GPU mutex — serializes ALL model operations (Codex: thread safety)
        self._gpu_lock = threading.Lock()
        # Gated model switch (Codex: prevents starvation during model switch)
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

    def set_pool_size(self, n: int) -> None:
        """Set the number of parallel inference engines.

        Takes effect on the next _get_engine_pool() call. If shrinking,
        excess engines are deleted immediately to free memory.
        OOM during creation gracefully stops at however many engines fit.
        """
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

    # --- Device & Engine Management ---

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

    @staticmethod
    def _vram_allocated_mb() -> float:
        """Return current VRAM allocated in MB, or 0 if unavailable."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(0) / (1024 ** 2)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _safe_offload(obj: object) -> None:
        """Move a model's GPU tensors to CPU before dropping the reference.

        Handles diffusers pipelines (.to('cpu')), plain nn.Modules (.cpu()),
        and objects with an explicit unload() method.
        """
        if obj is None:
            return
        name = type(obj).__name__
        logger.debug(f"Offloading model: {name}")
        try:
            if hasattr(obj, 'unload'):
                obj.unload()
            elif hasattr(obj, 'to'):
                obj.to('cpu')
            elif hasattr(obj, 'cpu'):
                obj.cpu()
            else:
                logger.warning(f"Model {name} has no .to()/.cpu()/.unload() — VRAM may leak")
        except Exception as e:
            logger.warning(f"Model offload failed for {name}: {e}")

    def _ensure_model(self, needed: _ActiveModel, on_status=None) -> None:
        """Model residency manager — unload current model if switching types.

        Only ONE heavy model stays in VRAM at a time. Before loading a
        different model, the previous is moved to CPU and dereferenced.

        Uses gated protocol: sets _switch_pending to block new inference
        sessions, then waits for active sessions to drain before switching.
        """
        if self._active_model == needed:
            return

        # Gate: block new inference sessions and wait for active ones to drain
        if self._active_model == _ActiveModel.INFERENCE and needed != _ActiveModel.INFERENCE:
            with self._gate_lock:
                self._switch_pending = True
            # Wait for all active inference sessions to finish
            if not self._inference_idle.wait(timeout=300):  # 5 min max
                logger.warning("Timed out waiting for inference sessions to drain")
            with self._gate_lock:
                self._switch_pending = False

        # Unload whatever is currently loaded
        if self._active_model != _ActiveModel.NONE:
            # Snapshot VRAM before unload for leak diagnosis
            vram_before_mb = self._vram_allocated_mb()
            if on_status:
                on_status(f"Switching {self._active_model.value} -> {needed.value}...")
            logger.info(f"Unloading {self._active_model.value} model for {needed.value}"
                        f" (VRAM before: {vram_before_mb:.0f}MB)")

            t0 = time.monotonic()
            if on_status:
                on_status(f"Offloading {self._active_model.value}...")
            if self._active_model == _ActiveModel.INFERENCE:
                for eng in self._engine_pool:
                    self._safe_offload(eng)
                self._engine_pool.clear()
            elif self._active_model == _ActiveModel.GVM:
                # GVM has circular refs (pipe ↔ vae ↔ unet) — break them
                # explicitly so gc can reclaim everything in one pass.
                gvm = self._gvm_processor
                self._gvm_processor = None
                if gvm is not None:
                    self._safe_offload(gvm)
                    # Break circular references inside the diffusion pipeline
                    for attr in ('pipe', 'vae', 'unet', 'scheduler'):
                        try:
                            setattr(gvm, attr, None)
                        except Exception:
                            pass
                    del gvm
            elif self._active_model == _ActiveModel.SAM2:
                self._safe_offload(self._sam2_tracker)
                self._sam2_tracker = None
            elif self._active_model == _ActiveModel.VIDEOMAMA:
                self._safe_offload(self._videomama_pipeline)
                self._videomama_pipeline = None
            elif self._active_model == _ActiveModel.MATANYONE2:
                self._safe_offload(self._matanyone2_processor)
                self._matanyone2_processor = None
            logger.info(f"_safe_offload took {time.monotonic() - t0:.1f}s")

            import gc
            t0 = time.monotonic()
            # Two GC passes: first breaks cycles, second reclaims freed refs
            if on_status:
                on_status("Releasing Python references...")
            gc.collect()
            gc.collect()
            logger.info(f"gc.collect took {time.monotonic() - t0:.1f}s")

            try:
                import torch
                if torch.cuda.is_available():
                    t0 = time.monotonic()
                    if on_status:
                        on_status("Waiting for CUDA to finish...")
                    torch.cuda.synchronize()
                    logger.info(f"cuda.synchronize took {time.monotonic() - t0:.1f}s")
                    t0 = time.monotonic()
                    if on_status:
                        on_status("Clearing CUDA cache...")
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    logger.info(f"cuda.empty_cache took {time.monotonic() - t0:.1f}s")
            except ImportError:
                logger.debug("torch not available for cache clear during model switch")

            # Reset Triton/dynamo compilation cache so torch.compile
            # doesn't choke on stale CUDA state from the previous model
            # (e.g. GVM diffusion pipeline leaves Triton state that causes
            # torch.compile to hang when building a fresh CorridorKeyEngine).
            try:
                import torch._dynamo
                torch._dynamo.reset()
                logger.info("torch._dynamo.reset() — cleared compilation cache")
            except Exception as e:
                logger.debug(f"dynamo reset skipped: {e}")

            vram_after_mb = self._vram_allocated_mb()
            freed = vram_before_mb - vram_after_mb
            logger.info(f"VRAM after unload: {vram_after_mb:.0f}MB (freed {freed:.0f}MB)")

        self._active_model = needed

    def _get_engine_pool(self, on_status=None) -> list:
        """Lazy-load the CorridorKey inference engine pool.

        Creates up to _pool_size engines. If the pool already has enough
        engines, returns immediately. OOM during creation shrinks the pool
        to however many engines fit in VRAM.

        Uses the backend factory to auto-detect MLX on Apple Silicon.
        MLX forces pool_size=1 (unified memory, no multi-engine benefit).
        """
        self._ensure_model(_ActiveModel.INFERENCE, on_status=on_status)

        # Pool already has enough engines
        if len(self._engine_pool) >= self._pool_size:
            return self._engine_pool[:self._pool_size]

        from CorridorKeyModule.backend import create_engine, resolve_backend

        backend = resolve_backend()
        opt_mode = os.environ.get('CORRIDORKEY_OPT_MODE', 'auto')
        _img_size = 1024 if self._device == 'mps' else 2048

        # MLX: unified memory, single engine only
        pool_size = 1 if backend == "mlx" else self._pool_size

        # Create engines serially (warmup each before creating next)
        import torch
        for i in range(len(self._engine_pool), pool_size):
            if on_status:
                on_status(f"Loading engine {i + 1}/{pool_size}...")
            logger.info("Creating engine %d/%d (backend=%s)", i + 1, pool_size, backend)
            t0 = time.monotonic()
            try:
                engine = create_engine(
                    backend=backend,
                    device=self._device,
                    img_size=_img_size,
                    optimization_mode=opt_mode,
                    on_status=on_status if i == 0 else None,
                )
                self._engine_pool.append(engine)
                logger.info(f"Engine {i + 1} loaded in {time.monotonic() - t0:.1f}s")
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                logger.warning(
                    "OOM creating engine %d/%d, using %d engine(s)",
                    i + 1, pool_size, len(self._engine_pool),
                )
                gc.collect()
                if backend == "torch":
                    torch.cuda.empty_cache()
                break

        if not self._engine_pool:
            raise RuntimeError("Failed to create any inference engine")

        return self._engine_pool

    def _get_gvm(self):
        """Lazy-load the GVM processor."""
        self._ensure_model(_ActiveModel.GVM)

        if self._gvm_processor is not None:
            return self._gvm_processor

        from gvm_core import GVMProcessor
        logger.info("Loading GVM processor...")
        t0 = time.monotonic()
        self._gvm_processor = GVMProcessor(device=self._device)
        logger.info(f"GVM loaded in {time.monotonic() - t0:.1f}s")
        return self._gvm_processor

    def _get_sam2_tracker(
        self,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ):
        """Lazy-load the optional SAM2 tracker."""
        self._ensure_model(_ActiveModel.SAM2)

        if self._sam2_tracker is not None:
            return self._sam2_tracker

        from sam2_tracker import SAM2Tracker

        logger.info("Loading SAM2 tracker...")
        t0 = time.monotonic()
        self._sam2_tracker = SAM2Tracker(
            model_id=self._sam2_model_id,
            device=self._device,
            # Meta's VOS path aggressively torch.compiles multiple components.
            # That is a poor default for this GUI app on Windows: it can spawn
            # compiler subprocesses, freeze the desktop, and hide progress.
            vos_optimized=False,
            offload_video_to_cpu=self._device.startswith("cuda"),
            offload_state_to_cpu=False,
        )
        if self._sam2_tracker is not None:
            self._sam2_tracker.prepare(on_progress=on_progress, on_status=on_status)
        logger.info(f"SAM2 tracker ready in {time.monotonic() - t0:.1f}s")
        return self._sam2_tracker

    def _get_videomama_pipeline(self):
        """Lazy-load the VideoMaMa inference pipeline."""
        self._ensure_model(_ActiveModel.VIDEOMAMA)

        if self._videomama_pipeline is not None:
            return self._videomama_pipeline

        sys.path.insert(0, os.path.join(BASE_DIR, "VideoMaMaInferenceModule"))
        from VideoMaMaInferenceModule.inference import load_videomama_model
        logger.info("Loading VideoMaMa pipeline...")
        t0 = time.monotonic()
        self._videomama_pipeline = load_videomama_model(device=self._device)
        logger.info(f"VideoMaMa loaded in {time.monotonic() - t0:.1f}s")
        return self._videomama_pipeline

    def _get_matanyone2(self):
        """Lazy-load the MatAnyone2 processor."""
        self._ensure_model(_ActiveModel.MATANYONE2)

        if self._matanyone2_processor is not None:
            return self._matanyone2_processor

        MatAnyone2Processor = _import_matanyone2_processor_class()
        logger.info("Loading MatAnyone2 processor...")
        t0 = time.monotonic()
        self._matanyone2_processor = MatAnyone2Processor(device=self._device)
        logger.info(f"MatAnyone2 loaded in {time.monotonic() - t0:.1f}s")
        return self._matanyone2_processor

    def unload_engines(self) -> None:
        """Free GPU memory by unloading all engines."""
        for eng in self._engine_pool:
            self._safe_offload(eng)
        self._engine_pool.clear()
        self._safe_offload(self._gvm_processor)
        self._safe_offload(self._sam2_tracker)
        self._safe_offload(self._videomama_pipeline)
        self._safe_offload(self._matanyone2_processor)
        self._gvm_processor = None
        self._sam2_tracker = None
        self._videomama_pipeline = None
        self._matanyone2_processor = None
        self._active_model = _ActiveModel.NONE
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            logger.debug("torch not available for cache clear during unload")
        logger.info("All engines unloaded, VRAM freed")

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

    # --- Frame I/O Helpers ---

    def _read_input_frame(
        self,
        clip: ClipEntry,
        frame_index: int,
        input_files: list[str],
        input_cap: Optional[Any],
        input_is_linear: bool,
    ) -> tuple[Optional[np.ndarray], str, bool]:
        """Read a single input frame.

        Returns:
            (image_float32, stem_name, is_linear)
        """
        logger.debug(f"Reading input frame {frame_index} for '{clip.name}'")
        input_stem = f"{frame_index:05d}"

        if input_cap:
            ret, frame = input_cap.read()
            if not ret:
                return None, input_stem, False
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return img_rgb.astype(np.float32) / 255.0, input_stem, input_is_linear
        else:
            fpath = os.path.join(clip.input_asset.path, input_files[frame_index])
            input_stem = os.path.splitext(input_files[frame_index])[0]
            img = read_image_frame(fpath)
            validate_frame_read(img, clip.name, frame_index, fpath)
            return img, input_stem, input_is_linear

    def _read_alpha_frame(
        self,
        clip: ClipEntry,
        frame_index: int,
        alpha_files: list[str],
        alpha_cap: Optional[Any],
        *,
        input_stem: str | None = None,
        alpha_stem_lookup: Optional[dict[str, str]] = None,
    ) -> Optional[np.ndarray]:
        """Read a single alpha/mask frame and normalize to [H, W] float32."""
        if alpha_cap:
            ret, frame = alpha_cap.read()
            if not ret:
                return None
            return frame[:, :, 2].astype(np.float32) / 255.0
        else:
            fname: str | None = None
            if input_stem is not None and alpha_stem_lookup is not None:
                fname = alpha_stem_lookup.get(input_stem)
            if fname is None:
                if frame_index >= len(alpha_files):
                    return None
                fname = alpha_files[frame_index]
            fpath = os.path.join(clip.alpha_asset.path, fname)
            mask = read_mask_frame(fpath, clip.name, frame_index)
            validate_frame_read(mask, clip.name, frame_index, fpath)
            return mask

    def _write_image(
        self, img: np.ndarray, path: str, fmt: str, clip_name: str, frame_index: int,
        exr_compression: str = "dwab",
    ) -> None:
        """Write a single image in the requested format."""
        if fmt == "exr":
            # EXR requires float32 — convert if uint8 (e.g. pre-converted comp)
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.dtype != np.float32:
                img = img.astype(np.float32)
            validate_write(
                write_exr(path, img, compression=exr_compression),
                clip_name, frame_index, path,
            )
        else:
            # PNG 8-bit
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            validate_write(cv2.imwrite(path, img), clip_name, frame_index, path)

    def _write_manifest(
        self,
        output_root: str,
        output_config: OutputConfig,
        params: InferenceParams,
    ) -> None:
        """Write run manifest recording expected outputs/extensions per run.

        Codex: base resume on manifest, not hardcoded FG/Matte intersection.
        Uses atomic write (tmp + rename) to prevent corruption.
        """
        manifest = {
            "version": 1,
            "enabled_outputs": output_config.enabled_outputs,
            "formats": {
                "fg": output_config.fg_format,
                "matte": output_config.matte_format,
                "comp": output_config.comp_format,
                "processed": output_config.processed_format,
            },
            "params": params.to_dict(),
        }
        manifest_path = os.path.join(output_root, ".corridorkey_manifest.json")
        tmp_path = manifest_path + ".tmp"
        try:
            with open(tmp_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            # Atomic replace (os.replace is atomic on both POSIX and Windows)
            os.replace(tmp_path, manifest_path)
        except Exception as e:
            logger.warning(f"Failed to write manifest: {e}")

    def _write_outputs(
        self,
        res: dict,
        dirs: dict[str, str],
        input_stem: str,
        clip_name: str,
        frame_index: int,
        output_config: OutputConfig | None = None,
    ) -> None:
        """Write output types for a single frame respecting OutputConfig."""
        cfg = output_config or OutputConfig()
        logger.debug(f"Writing outputs for '{clip_name}' frame {frame_index} stem='{input_stem}'")

        pred_fg = res['fg']
        pred_alpha = res['alpha']

        # FG
        if cfg.fg_enabled:
            fg_bgr = cv2.cvtColor(pred_fg, cv2.COLOR_RGB2BGR)
            fg_path = os.path.join(dirs['fg'], f"{input_stem}.{cfg.fg_format}")
            self._write_image(fg_bgr, fg_path, cfg.fg_format, clip_name, frame_index,
                              exr_compression=cfg.exr_compression)

        # Matte
        if cfg.matte_enabled:
            alpha = pred_alpha
            if alpha.ndim == 3:
                alpha = alpha[:, :, 0]
            matte_path = os.path.join(dirs['matte'], f"{input_stem}.{cfg.matte_format}")
            self._write_image(alpha, matte_path, cfg.matte_format, clip_name, frame_index,
                              exr_compression=cfg.exr_compression)

        # Comp
        if cfg.comp_enabled:
            comp_srgb = res['comp']
            comp_bgr = cv2.cvtColor(
                (np.clip(comp_srgb, 0.0, 1.0) * 255.0).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            comp_path = os.path.join(dirs['comp'], f"{input_stem}.{cfg.comp_format}")
            self._write_image(comp_bgr, comp_path, cfg.comp_format, clip_name, frame_index,
                              exr_compression=cfg.exr_compression)

        # Processed (RGBA premultiplied)
        if cfg.processed_enabled and 'processed' in res:
            proc_rgba = res['processed']
            proc_bgra = cv2.cvtColor(proc_rgba, cv2.COLOR_RGBA2BGRA)
            proc_path = os.path.join(dirs['processed'], f"{input_stem}.{cfg.processed_format}")
            self._write_image(proc_bgra, proc_path, cfg.processed_format, clip_name, frame_index,
                              exr_compression=cfg.exr_compression)

    # --- Processing ---

    def run_inference(
        self,
        clip: ClipEntry,
        params: InferenceParams,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[..., None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        skip_stems: Optional[set[str]] = None,
        output_config: Optional[OutputConfig] = None,
        frame_range: Optional[tuple[int, int]] = None,
    ) -> list[FrameResult]:
        """Run CorridorKey inference on a single clip.

        Uses gated model switch protocol. If pool_size > 1, dispatches
        to _run_inference_parallel with N engines processing N frames
        concurrently. Otherwise uses _run_inference_sequential.

        Args:
            clip: Must be in READY or COMPLETE state with both input_asset and alpha_asset.
            params: Frozen inference parameters.
            job: Optional GPUJob for cancel checking.
            on_progress: Called with (clip_name, current_frame, total_frames, **kwargs).
                Optional kwargs: fps, elapsed, eta_seconds.
            on_warning: Called with warning messages for non-fatal issues.
            on_status: Called with phase status text (e.g. "Loading model...").
            skip_stems: Set of frame stems to skip (for resume support).
            output_config: Which outputs to write and their formats.

        Returns:
            List of FrameResult for each frame.

        Raises:
            JobCancelledError: If job.is_cancelled becomes True.
            Various CorridorKeyError subclasses for fatal issues.
        """
        if clip.input_asset is None or clip.alpha_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input or alpha asset")

        self._begin_inference()
        try:
            if on_status:
                on_status("Loading model...")
            logger.info("run_inference: waiting for _gpu_lock")
            with self._gpu_lock:
                logger.info("run_inference: acquired _gpu_lock")
                engines = self._get_engine_pool(on_status=on_status)

            if len(engines) == 1:
                return self._run_inference_sequential(
                    clip, params, engines[0], job=job,
                    on_progress=on_progress, on_warning=on_warning,
                    on_status=on_status, skip_stems=skip_stems,
                    output_config=output_config, frame_range=frame_range,
                )
            return self._run_inference_parallel(
                clip, params, engines, job=job,
                on_progress=on_progress, on_warning=on_warning,
                on_status=on_status, skip_stems=skip_stems,
                output_config=output_config, frame_range=frame_range,
            )
        finally:
            self._end_inference()

    def _run_inference_sequential(
        self,
        clip: ClipEntry,
        params: InferenceParams,
        engine,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[..., None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        skip_stems: Optional[set[str]] = None,
        output_config: Optional[OutputConfig] = None,
        frame_range: Optional[tuple[int, int]] = None,
    ) -> list[FrameResult]:
        """Sequential frame loop — exact extraction of original run_inference."""
        t_start = time.monotonic()
        dirs = ensure_output_dirs(clip.root_path)
        cfg = output_config or OutputConfig()

        self._write_manifest(dirs['root'], cfg, params)

        if clip.input_asset.asset_type == 'sequence' and clip.alpha_asset.asset_type == 'sequence':
            num_frames = clip.input_asset.frame_count
            if clip.input_asset.frame_count != clip.alpha_asset.frame_count:
                logger.warning(
                    "Clip '%s': sequence alpha count mismatch — input has %d, alpha has %d. "
                    "Using stem-matched alpha reads across the selected range.",
                    clip.name,
                    clip.input_asset.frame_count,
                    clip.alpha_asset.frame_count,
                )
        else:
            num_frames = validate_frame_counts(
                clip.name,
                clip.input_asset.frame_count,
                clip.alpha_asset.frame_count,
            )

        input_cap = None
        alpha_cap = None
        input_files: list[str] = []
        alpha_files: list[str] = []

        if clip.input_asset.asset_type == 'video':
            input_cap = cv2.VideoCapture(clip.input_asset.path)
        else:
            input_files = clip.input_asset.get_frame_files()

        if clip.alpha_asset.asset_type == 'video':
            alpha_cap = cv2.VideoCapture(clip.alpha_asset.path)
        else:
            alpha_files = clip.alpha_asset.get_frame_files()
        alpha_stem_lookup = (
            {os.path.splitext(fname)[0]: fname for fname in alpha_files}
            if alpha_files else None
        )

        results: list[FrameResult] = []
        skipped: list[int] = []
        skip_stems = skip_stems or set()
        frame_times: deque[float] = deque(maxlen=10)
        processed_count = 0

        if frame_range is not None:
            range_start = max(0, frame_range[0])
            range_end = min(num_frames - 1, frame_range[1])
            frame_indices = range(range_start, range_end + 1)
            range_count = range_end - range_start + 1
        else:
            frame_indices = range(num_frames)
            range_count = num_frames

        _warmup_done = False

        try:
            for progress_i, i in enumerate(frame_indices):
                if job and job.is_cancelled:
                    raise JobCancelledError(clip.name, i)

                if not _warmup_done and on_status:
                    on_status("Compiling (first frame may take a minute)...")

                if on_progress:
                    timing_kwargs: dict[str, float] = {}
                    elapsed = time.monotonic() - t_start
                    timing_kwargs["elapsed"] = elapsed
                    if frame_times:
                        avg_time = sum(frame_times) / len(frame_times)
                        fps = 1.0 / avg_time if avg_time > 0 else 0.0
                        remaining = range_count - progress_i
                        timing_kwargs["fps"] = fps
                        timing_kwargs["eta_seconds"] = remaining * avg_time
                    on_progress(clip.name, progress_i, range_count, **timing_kwargs)

                try:
                    img, input_stem, is_linear = self._read_input_frame(
                        clip, i, input_files, input_cap, params.input_is_linear,
                    )
                    if img is None:
                        skipped.append(i)
                        results.append(FrameResult(i, f"{i:05d}", False, "video read failed"))
                        continue

                    if input_stem in skip_stems:
                        results.append(FrameResult(i, input_stem, True, "resumed (skipped)"))
                        continue

                    mask = self._read_alpha_frame(
                        clip, i, alpha_files, alpha_cap,
                        input_stem=input_stem,
                        alpha_stem_lookup=alpha_stem_lookup,
                    )
                    if mask is None:
                        skipped.append(i)
                        results.append(FrameResult(i, input_stem, False, "alpha read failed"))
                        continue

                    if mask.shape[:2] != img.shape[:2]:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

                    t_frame = time.monotonic()
                    with self._gpu_lock:
                        res = engine.process_frame(
                            img, mask,
                            input_is_linear=is_linear,
                            fg_is_straight=True,
                            despill_strength=params.despill_strength,
                            auto_despeckle=params.auto_despeckle,
                            despeckle_size=params.despeckle_size,
                            despeckle_dilation=params.despeckle_dilation,
                            despeckle_blur=params.despeckle_blur,
                            refiner_scale=params.refiner_scale,
                        )
                    dt = time.monotonic() - t_frame
                    frame_times.append(dt)
                    processed_count += 1

                    if not _warmup_done:
                        _warmup_done = True
                        if on_status:
                            on_status("")
                    total_t = sum(frame_times)
                    avg_fps = len(frame_times) / total_t if total_t > 0 else 0.0
                    logger.debug(
                        f"Frame {i}: {dt * 1000:.0f}ms ({avg_fps:.1f} fps avg)"
                    )

                    self._write_outputs(res, dirs, input_stem, clip.name, i, cfg)
                    results.append(FrameResult(i, input_stem, True))

                except FrameReadError as e:
                    logger.warning(str(e))
                    skipped.append(i)
                    results.append(FrameResult(i, f"{i:05d}", False, str(e)))
                    if on_warning:
                        on_warning(str(e))

                except WriteFailureError as e:
                    logger.error(str(e))
                    results.append(FrameResult(i, f"{i:05d}", False, str(e)))
                    if on_warning:
                        on_warning(str(e))

            if on_progress:
                final_elapsed = time.monotonic() - t_start
                final_kwargs: dict[str, float] = {"elapsed": final_elapsed, "eta_seconds": 0.0}
                total_t = sum(frame_times)
                if total_t > 0:
                    final_kwargs["fps"] = len(frame_times) / total_t
                on_progress(clip.name, range_count, range_count, **final_kwargs)

        finally:
            if input_cap:
                input_cap.release()
            if alpha_cap:
                alpha_cap.release()

        return self._finalize_inference(
            clip, results, skipped, processed_count,
            t_start, frame_range, num_frames,
            on_warning=on_warning,
        )

    def _run_inference_parallel(
        self,
        clip: ClipEntry,
        params: InferenceParams,
        engines: list,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[..., None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        skip_stems: Optional[set[str]] = None,
        output_config: Optional[OutputConfig] = None,
        frame_range: Optional[tuple[int, int]] = None,
    ) -> list[FrameResult]:
        """Parallel frame pipeline: reader thread -> N workers -> writer thread.

        Each worker is permanently bound to one engine instance. PyTorch
        releases the GIL during CUDA ops, so threads naturally overlap GPU work.
        No explicit CUDA streams needed (they fail on Windows anyway).
        """
        N = len(engines)
        logger.info("Starting parallel inference with %d engines", N)

        t_start = time.monotonic()
        dirs = ensure_output_dirs(clip.root_path)
        cfg = output_config or OutputConfig()
        self._write_manifest(dirs['root'], cfg, params)

        if clip.input_asset.asset_type == 'sequence' and clip.alpha_asset.asset_type == 'sequence':
            num_frames = clip.input_asset.frame_count
            if clip.input_asset.frame_count != clip.alpha_asset.frame_count:
                logger.warning(
                    "Clip '%s': sequence alpha count mismatch — input has %d, alpha has %d.",
                    clip.name, clip.input_asset.frame_count, clip.alpha_asset.frame_count,
                )
        else:
            num_frames = validate_frame_counts(
                clip.name, clip.input_asset.frame_count, clip.alpha_asset.frame_count,
            )

        if frame_range is not None:
            range_start = max(0, frame_range[0])
            range_end = min(num_frames - 1, frame_range[1])
            frame_indices = list(range(range_start, range_end + 1))
            range_count = range_end - range_start + 1
        else:
            frame_indices = list(range(num_frames))
            range_count = num_frames

        skip_stems = skip_stems or set()

        # Bounded queues for pipeline stages
        in_q: queue.Queue = queue.Queue(maxsize=2 * N)
        out_q: queue.Queue = queue.Queue(maxsize=2 * N)
        stop = threading.Event()
        error_box: list = [None]

        if on_status:
            on_status("Compiling (first frame may take a minute)...")
        warmup_done = threading.Event()

        # --- Worker threads (each bound to one engine) ---
        def worker(engine_idx: int) -> None:
            eng = engines[engine_idx]
            while not stop.is_set():
                try:
                    item = in_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:
                    break  # poison pill
                frame_idx, img, mask, stem, is_linear = item
                try:
                    res = eng.process_frame(
                        img, mask,
                        input_is_linear=is_linear,
                        fg_is_straight=True,
                        despill_strength=params.despill_strength,
                        auto_despeckle=params.auto_despeckle,
                        despeckle_size=params.despeckle_size,
                        despeckle_dilation=params.despeckle_dilation,
                        despeckle_blur=params.despeckle_blur,
                        refiner_scale=params.refiner_scale,
                    )
                    out_q.put((frame_idx, stem, res, None))
                    if not warmup_done.is_set():
                        warmup_done.set()
                        if on_status:
                            on_status("")
                except Exception as e:
                    out_q.put((frame_idx, stem, None, e))
                    stop.set()
                    break

        # --- Reader thread ---
        def reader() -> None:
            input_cap = None
            alpha_cap = None
            input_files: list[str] = []
            alpha_files: list[str] = []

            try:
                if clip.input_asset.asset_type == 'video':
                    input_cap = cv2.VideoCapture(clip.input_asset.path)
                else:
                    input_files = clip.input_asset.get_frame_files()

                if clip.alpha_asset.asset_type == 'video':
                    alpha_cap = cv2.VideoCapture(clip.alpha_asset.path)
                else:
                    alpha_files = clip.alpha_asset.get_frame_files()
                alpha_stem_lookup = (
                    {os.path.splitext(fname)[0]: fname for fname in alpha_files}
                    if alpha_files else None
                )

                for i in frame_indices:
                    if stop.is_set() or (job and job.is_cancelled):
                        break

                    img, input_stem, is_linear = self._read_input_frame(
                        clip, i, input_files, input_cap, params.input_is_linear,
                    )
                    if img is None:
                        out_q.put((i, f"{i:05d}", None, FrameReadError(clip.name, i, "video read failed")))
                        continue

                    if input_stem in skip_stems:
                        out_q.put((i, input_stem, "SKIP", None))
                        continue

                    mask = self._read_alpha_frame(
                        clip, i, alpha_files, alpha_cap,
                        input_stem=input_stem, alpha_stem_lookup=alpha_stem_lookup,
                    )
                    if mask is None:
                        out_q.put((i, input_stem, None, FrameReadError(clip.name, i, "alpha read failed")))
                        continue

                    if mask.shape[:2] != img.shape[:2]:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

                    in_q.put((i, img, mask, input_stem, is_linear))
            finally:
                for _ in range(N):
                    try:
                        in_q.put(None, timeout=5)
                    except queue.Full:
                        pass
                if input_cap:
                    input_cap.release()
                if alpha_cap:
                    alpha_cap.release()

        # --- Writer thread ---
        results: list[FrameResult] = []
        skipped: list[int] = []
        processed_count_box = [0]

        def writer() -> None:
            reorder: dict = {}
            next_idx = frame_indices[0] if frame_indices else 0
            written = 0
            received = 0
            t_last_progress = time.monotonic()

            while received < range_count:
                if stop.is_set() and not reorder:
                    break
                try:
                    frame_idx, stem, result, err = out_q.get(timeout=1.0)
                except queue.Empty:
                    if stop.is_set():
                        break
                    continue
                received += 1

                if err is not None:
                    if isinstance(err, (FrameReadError, WriteFailureError)):
                        skipped.append(frame_idx)
                        results.append(FrameResult(frame_idx, stem, False, str(err)))
                        if on_warning:
                            on_warning(str(err))
                        next_idx = max(next_idx, frame_idx + 1)
                        continue
                    error_box[0] = err
                    stop.set()
                    break

                if result == "SKIP":
                    results.append(FrameResult(frame_idx, stem, True, "resumed (skipped)"))
                    next_idx = max(next_idx, frame_idx + 1)
                    continue

                reorder[frame_idx] = (result, stem)

                while next_idx in reorder:
                    res, s = reorder.pop(next_idx)
                    self._write_outputs(res, dirs, s, clip.name, next_idx, cfg)
                    processed_count_box[0] += 1
                    written += 1
                    results.append(FrameResult(next_idx, s, True))

                    now = time.monotonic()
                    if on_progress and (now - t_last_progress > 0.1 or written == range_count):
                        elapsed = now - t_start
                        timing_kw: dict[str, float] = {"elapsed": elapsed}
                        if processed_count_box[0] > 0:
                            fps = processed_count_box[0] / elapsed
                            remaining = range_count - written
                            timing_kw["fps"] = fps
                            if fps > 0:
                                timing_kw["eta_seconds"] = remaining / fps
                        on_progress(clip.name, written, range_count, **timing_kw)
                        t_last_progress = now

                    next_idx += 1

            if on_progress:
                final_elapsed = time.monotonic() - t_start
                final_kw: dict[str, float] = {"elapsed": final_elapsed, "eta_seconds": 0.0}
                if processed_count_box[0] > 0:
                    final_kw["fps"] = processed_count_box[0] / final_elapsed
                on_progress(clip.name, range_count, range_count, **final_kw)

        # Launch all threads
        threads = (
            [threading.Thread(target=reader, name="ck-reader")] +
            [threading.Thread(target=worker, args=(i,), name=f"ck-worker-{i}") for i in range(N)] +
            [threading.Thread(target=writer, name="ck-writer")]
        )
        for t in threads:
            t.start()

        for t in threads:
            while t.is_alive():
                t.join(timeout=0.5)
                if job and job.is_cancelled and not stop.is_set():
                    stop.set()

        if error_box[0]:
            raise error_box[0]
        if job and job.is_cancelled:
            raise JobCancelledError(clip.name, 0)

        return self._finalize_inference(
            clip, results, skipped, processed_count_box[0],
            t_start, frame_range, num_frames,
            on_warning=on_warning,
        )

    def _finalize_inference(
        self,
        clip: ClipEntry,
        results: list[FrameResult],
        skipped: list[int],
        processed_count: int,
        t_start: float,
        frame_range: Optional[tuple[int, int]],
        num_frames: int,
        on_warning: Optional[Callable[[str], None]] = None,
    ) -> list[FrameResult]:
        """Shared post-processing for sequential and parallel inference."""
        range_count = len(results) if results else 0
        processed = sum(1 for r in results if r.success)
        if skipped:
            msg = (
                f"Clip '{clip.name}': {len(skipped)} frame(s) skipped: "
                f"{skipped[:20]}{'...' if len(skipped) > 20 else ''}"
            )
            logger.warning(msg)
            if on_warning:
                on_warning(msg)

        t_total = time.monotonic() - t_start
        avg_fps = processed_count / t_total if t_total > 0 and processed_count > 0 else 0.0
        range_label = f" (range {frame_range[0]}-{frame_range[1]})" if frame_range else ""
        logger.info(
            f"Clip '{clip.name}': inference complete{range_label}. {processed}/{range_count} frames "
            f"in {t_total:.1f}s ({t_total / max(processed, 1):.2f}s/frame, {avg_fps:.1f} fps avg)"
        )

        is_full_clip = (frame_range is None or
                        (frame_range[0] == 0 and frame_range[1] >= num_frames - 1))
        if processed == range_count and is_full_clip:
            try:
                clip.transition_to(ClipState.COMPLETE)
            except Exception as e:
                logger.warning(f"Clip '{clip.name}': state transition to COMPLETE failed: {e}")

        return results

    # --- Single-Frame Reprocess (Preview) ---

    def is_engine_loaded(self) -> bool:
        """True if the inference engine is already loaded in VRAM."""
        return self._active_model == _ActiveModel.INFERENCE and len(self._engine_pool) > 0

    def reprocess_single_frame(
        self,
        clip: ClipEntry,
        params: InferenceParams,
        frame_index: int,
        job: Optional[GPUJob] = None,
    ) -> Optional[dict]:
        """Reprocess a single frame with current params.

        Returns the result dict (fg, alpha, comp, processed) or None.
        This runs through the GPU lock for thread safety.
        Does NOT write to disk — returns in-memory results for preview.
        """
        t_start = time.monotonic()
        if clip.input_asset is None or clip.alpha_asset is None:
            return None

        if job and job.is_cancelled:
            return None

        with self._gpu_lock:
            engines = self._get_engine_pool()
            engine = engines[0]

        # Read the specific input frame
        is_linear = params.input_is_linear
        input_stem = f"{frame_index:05d}"
        if clip.input_asset.asset_type == 'video':
            img = read_video_frame_at(clip.input_asset.path, frame_index)
        else:
            input_files = clip.input_asset.get_frame_files()
            if frame_index >= len(input_files):
                return None
            fpath = os.path.join(clip.input_asset.path, input_files[frame_index])
            input_stem = os.path.splitext(input_files[frame_index])[0]
            img = read_image_frame(fpath)
        if img is None:
            return None

        # Read the specific alpha frame
        if clip.alpha_asset.asset_type == 'video':
            mask = read_video_mask_at(clip.alpha_asset.path, frame_index)
        else:
            alpha_files = clip.alpha_asset.get_frame_files()
            alpha_stem_lookup = {os.path.splitext(fname)[0]: fname for fname in alpha_files}
            mask = self._read_alpha_frame(
                clip,
                frame_index,
                alpha_files,
                None,
                input_stem=input_stem,
                alpha_stem_lookup=alpha_stem_lookup,
            )
        if mask is None:
            return None

        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        if job and job.is_cancelled:
            return None

        with self._gpu_lock:
            res = engine.process_frame(
                img, mask,
                input_is_linear=is_linear,
                fg_is_straight=True,
                despill_strength=params.despill_strength,
                auto_despeckle=params.auto_despeckle,
                despeckle_size=params.despeckle_size,
                despeckle_dilation=params.despeckle_dilation,
                despeckle_blur=params.despeckle_blur,
                refiner_scale=params.refiner_scale,
            )
        logger.debug(f"Clip '{clip.name}' frame {frame_index}: reprocess {time.monotonic() - t_start:.3f}s")
        return res

    # --- GVM Alpha Generation ---

    def run_gvm(
        self,
        clip: ClipEntry,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Run GVM auto alpha generation for a clip.

        Transitions clip: RAW → READY (creates AlphaHint directory).

        Args:
            clip: Must be in RAW state with input_asset.
            job: Optional GPUJob for cancel checking.
            on_progress: Progress callback (GVM is monolithic, reports start/end).
            on_warning: Warning callback.
        """
        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for GVM")

        if self._device == 'cpu':
            raise GPURequiredError("GVM Auto Alpha")

        t_start = time.monotonic()

        logger.info("run_gvm: waiting for _gpu_lock")
        with self._gpu_lock:
            logger.info("run_gvm: acquired _gpu_lock")
            gvm = self._get_gvm()

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)

        if on_progress:
            on_progress(clip.name, 0, 1)

        # Check cancel before starting
        if job and job.is_cancelled:
            raise JobCancelledError(clip.name, 0)

        # Per-batch progress callback — GVM iterates over frames internally
        def _gvm_progress(batch_idx: int, total_batches: int) -> None:
            if on_progress:
                on_progress(clip.name, batch_idx, total_batches)
            # Check cancel between batches
            if job and job.is_cancelled:
                raise JobCancelledError(clip.name, batch_idx)

        try:
            gvm.process_sequence(
                input_path=clip.input_asset.path,
                output_dir=clip.root_path,
                num_frames_per_batch=1,
                decode_chunk_size=1,
                denoise_steps=1,
                mode='matte',
                write_video=False,
                direct_output_dir=alpha_dir,
                progress_callback=_gvm_progress,
            )
        except JobCancelledError:
            raise
        except Exception as e:
            if job and job.is_cancelled:
                raise JobCancelledError(clip.name, 0)
            raise CorridorKeyError(f"GVM failed for '{clip.name}': {e}") from e

        # Refresh alpha asset
        clip.alpha_asset = ClipAsset(alpha_dir, 'sequence')

        if on_progress:
            on_progress(clip.name, 1, 1)

        # Transition RAW → READY
        try:
            clip.transition_to(ClipState.READY)
        except Exception as e:
            if on_warning:
                on_warning(f"State transition after GVM: {e}")

        logger.info(f"GVM complete for '{clip.name}': {clip.alpha_asset.frame_count} alpha frames in {time.monotonic() - t_start:.1f}s")

    def _selected_sequence_files(self, clip: ClipEntry) -> list[str]:
        """Return the ordered frame filenames for the clip's active in/out range."""
        if clip.input_asset is None or clip.input_asset.asset_type != "sequence":
            return []
        files = clip.input_asset.get_frame_files()
        if clip.in_out_range is not None:
            lo = clip.in_out_range.in_point
            hi = clip.in_out_range.out_point
            files = files[lo:hi + 1]
        return files

    def _load_named_sequence_frames(
        self,
        asset: ClipAsset,
        file_names: list[str],
        clip_name: str,
        *,
        gamma_correct_exr: bool = False,
        job: Optional[GPUJob] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> list[tuple[str, np.ndarray]]:
        """Load named image-sequence frames as uint8 RGB for tracker/VideoMaMa."""
        named_frames: list[tuple[str, np.ndarray]] = []
        total = len(file_names)
        for index, fname in enumerate(file_names):
            if job and job.is_cancelled:
                raise JobCancelledError(clip_name, index)
            fpath = os.path.join(asset.path, fname)
            img = read_image_frame(fpath, gamma_correct_exr=gamma_correct_exr)
            if img is None:
                raise FrameReadError(clip_name, index, fpath)
            named_frames.append(
                (fname, (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))
            )
            if on_status and index % 20 == 0 and index > 0:
                on_status(f"Loading frames ({index}/{total})...")
        return named_frames

    @staticmethod
    def _resolve_sequence_input_is_linear(
        clip: ClipEntry,
        input_is_linear: bool | None,
    ) -> bool:
        """Honor explicit UI override, otherwise default from clip source type."""
        if input_is_linear is not None:
            return input_is_linear
        return clip.should_default_input_linear()

    @staticmethod
    def _write_mask_track_manifest(
        clip: ClipEntry,
        *,
        source: str,
        frame_stems: list[str],
        model_id: str | None = None,
    ) -> None:
        """Persist provenance for dense VideoMaMa-ready mask tracks."""
        manifest_path = os.path.join(clip.root_path, MASK_TRACK_MANIFEST)
        payload = {
            "source": source,
            "frame_stems": frame_stems,
            "model_id": model_id,
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @staticmethod
    def _remove_alpha_hint_dir(clip: ClipEntry) -> None:
        """Remove AlphaHint so a new mask/alpha run is authoritative."""
        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir):
            shutil.rmtree(alpha_dir, ignore_errors=True)

    def run_sam2_track(
        self,
        clip: ClipEntry,
        input_is_linear: bool | None = None,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Turn sparse annotations into dense VideoMaMa mask hints with SAM2."""
        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for SAM2 tracking")
        if clip.input_asset.asset_type != "sequence":
            raise CorridorKeyError("SAM2 tracking currently requires an extracted image sequence")

        selected_files = self._selected_sequence_files(clip)
        if not selected_files:
            raise CorridorKeyError(f"Clip '{clip.name}' has no input frames for SAM2 tracking")

        def _status(message: str) -> None:
            logger.info(f"SAM2 [{clip.name}]: {message}")
            if on_status:
                on_status(message)

        def _check_cancel() -> None:
            if job and job.is_cancelled:
                raise JobCancelledError(clip.name, 0)

        _status("Loading model...")
        with self._gpu_lock:
            tracker = self._get_sam2_tracker(
                on_progress=(
                    None
                    if on_progress is None
                    else lambda current, total: on_progress(clip.name, current, total)
                ),
                on_status=on_status,
            )
        _check_cancel()

        _status("Loading frames...")
        sequence_is_linear = self._resolve_sequence_input_is_linear(clip, input_is_linear)
        named_frames = self._load_named_sequence_frames(
            clip.input_asset,
            selected_files,
            clip.name,
            gamma_correct_exr=sequence_is_linear,
            job=job,
            on_status=on_status,
        )
        _check_cancel()
        if not named_frames:
            raise CorridorKeyError(f"Clip '{clip.name}' has no readable input frames for SAM2 tracking")

        start_index = clip.in_out_range.in_point if clip.in_out_range is not None else 0
        allowed_indices = list(range(start_index, start_index + len(selected_files)))
        prompt_frames = load_annotation_prompt_frames(
            clip.root_path,
            allowed_indices=allowed_indices,
        )
        if not prompt_frames:
            raise CorridorKeyError(
                f"Clip '{clip.name}' has no usable annotations for SAM2 tracking"
            )

        from sam2_tracker import PromptFrame, SAM2NotInstalledError

        local_prompts = [
            PromptFrame(
                frame_index=prompt.frame_index - start_index,
                positive_points=prompt.positive_points,
                negative_points=prompt.negative_points,
                box=prompt.box,
            )
            for prompt in prompt_frames
        ]
        if not any(
            prompt.positive_points or prompt.box is not None
            for prompt in local_prompts
        ):
            message = "SAM2 tracking requires at least one non-empty foreground prompt"
            if on_warning:
                on_warning(message)
            raise CorridorKeyError(message)
        total_pos = sum(len(prompt.positive_points) for prompt in local_prompts)
        total_neg = sum(len(prompt.negative_points) for prompt in local_prompts)
        logger.info(
            "SAM2 [%s]: prompt frames=%d, fg points=%d, bg points=%d",
            clip.name,
            len(local_prompts),
            total_pos,
            total_neg,
        )

        _status("Running SAM2 tracker...")
        try:
            masks = tracker.track_video(
                [frame for _, frame in named_frames],
                local_prompts,
                on_progress=(
                    None
                    if on_progress is None
                    else lambda current, total: on_progress(clip.name, current, total)
                ),
                on_status=on_status,
                check_cancel=_check_cancel,
            )
        except SAM2NotInstalledError as exc:
            raise CorridorKeyError(str(exc)) from exc
        except ValueError as exc:
            message = str(exc)
            if on_warning:
                on_warning(message)
            raise CorridorKeyError(message) from exc
        _check_cancel()

        mask_dir = os.path.join(clip.root_path, "VideoMamaMaskHint")
        os.makedirs(mask_dir, exist_ok=True)
        for fname in os.listdir(mask_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                os.remove(os.path.join(mask_dir, fname))

        stems: list[str] = []
        for (fname, _), mask in zip(named_frames, masks):
            stem = os.path.splitext(fname)[0]
            stems.append(stem)
            out_path = os.path.join(mask_dir, f"{stem}.png")
            if not cv2.imwrite(out_path, mask):
                raise WriteFailureError(clip.name, len(stems) - 1, out_path)

        self._write_mask_track_manifest(
            clip,
            source="sam2",
            frame_stems=stems,
            model_id=getattr(tracker, "model_id", None),
        )
        self._remove_alpha_hint_dir(clip)
        clip.alpha_asset = None
        clip.mask_asset = None
        clip.find_assets()
        clip.state = ClipState.MASKED

        logger.info(
            "SAM2 tracking complete for '%s': %d dense masks",
            clip.name,
            len(stems),
        )

    def preview_sam2_prompt(
        self,
        clip: ClipEntry,
        *,
        preferred_frame_index: int | None = None,
        input_is_linear: bool | None = None,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Optional[dict[str, Any]]:
        """Run a fast SAM2 preview on one annotated frame without writing to disk."""
        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for SAM2 tracking")
        if clip.input_asset.asset_type != "sequence":
            raise CorridorKeyError("SAM2 tracking currently requires an extracted image sequence")

        selected_files = self._selected_sequence_files(clip)
        if not selected_files:
            raise CorridorKeyError(f"Clip '{clip.name}' has no input frames for SAM2 tracking")

        def _status(message: str) -> None:
            logger.info(f"SAM2 Preview [{clip.name}]: {message}")
            if on_status:
                on_status(message)

        def _check_cancel() -> None:
            if job and job.is_cancelled:
                raise JobCancelledError(clip.name, 0)

        _status("Loading model...")
        with self._gpu_lock:
            tracker = self._get_sam2_tracker(
                on_progress=(
                    None
                    if on_progress is None
                    else lambda current, total: on_progress(clip.name, current, total)
                ),
                on_status=on_status,
            )
        _check_cancel()

        start_index = clip.in_out_range.in_point if clip.in_out_range is not None else 0
        allowed_indices = list(range(start_index, start_index + len(selected_files)))
        prompt_frames = load_annotation_prompt_frames(
            clip.root_path,
            allowed_indices=allowed_indices,
        )
        if not prompt_frames:
            raise CorridorKeyError(
                f"Clip '{clip.name}' has no usable annotations for SAM2 tracking"
            )

        prompt = next(
            (item for item in prompt_frames if item.frame_index == preferred_frame_index),
            prompt_frames[0],
        )
        if preferred_frame_index is not None and prompt.frame_index != preferred_frame_index and on_warning:
            on_warning(
                f"No prompts on frame {preferred_frame_index + 1}; previewing annotated frame {prompt.frame_index + 1} instead."
            )

        local_index = prompt.frame_index - start_index
        if local_index < 0 or local_index >= len(selected_files):
            raise CorridorKeyError("Annotated frame is outside the selected in/out range")

        _status("Loading preview frame...")
        sequence_is_linear = self._resolve_sequence_input_is_linear(clip, input_is_linear)
        named_frames = self._load_named_sequence_frames(
            clip.input_asset,
            [selected_files[local_index]],
            clip.name,
            gamma_correct_exr=sequence_is_linear,
            job=job,
            on_status=on_status,
        )
        _check_cancel()
        if not named_frames:
            raise CorridorKeyError(f"Clip '{clip.name}' has no readable input frames for SAM2 tracking")

        from sam2_tracker import PromptFrame, SAM2NotInstalledError

        local_prompt = PromptFrame(
            frame_index=0,
            positive_points=prompt.positive_points,
            negative_points=prompt.negative_points,
            box=prompt.box,
        )

        _status("Previewing SAM2 on annotated frame...")
        try:
            masks = tracker.track_video(
                [named_frames[0][1]],
                [local_prompt],
                on_progress=(
                    None
                    if on_progress is None
                    else lambda current, total: on_progress(clip.name, current, total)
                ),
                on_status=on_status,
                check_cancel=_check_cancel,
            )
        except SAM2NotInstalledError as exc:
            raise CorridorKeyError(str(exc)) from exc
        except ValueError as exc:
            message = str(exc)
            if on_warning:
                on_warning(message)
            raise CorridorKeyError(message) from exc
        _check_cancel()

        mask = masks[0]
        return {
            "kind": "sam2_preview",
            "clip_name": clip.name,
            "frame_index": prompt.frame_index,
            "frame_name": named_frames[0][0],
            "frame_rgb": named_frames[0][1],
            "mask": mask,
            "fill": float((mask > 0).mean()),
        }

    # --- VideoMaMa Alpha Generation ---

    def run_videomama(
        self,
        clip: ClipEntry,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        chunk_size: int = 16,
    ) -> None:
        """Run VideoMaMa guided alpha generation for a clip.

        Transitions clip: MASKED → READY (creates AlphaHint directory).

        Args:
            clip: Must be in MASKED state with input_asset and mask_asset.
            job: Optional GPUJob for cancel checking.
            on_progress: Progress callback with per-chunk updates.
            on_warning: Warning callback.
            on_status: Phase status callback (e.g. "Loading model...").
            chunk_size: Frames per chunk (lower = less VRAM, default 16).
        """
        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for VideoMaMa")
        if clip.mask_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing mask asset for VideoMaMa")

        if self._device == 'cpu':
            raise GPURequiredError("VideoMaMa")
        if clip.input_asset.asset_type != "sequence":
            raise CorridorKeyError("VideoMaMa currently requires an extracted image sequence")
        ann_path = os.path.join(clip.root_path, "annotations.json")
        has_annotations = os.path.isfile(ann_path) and os.path.getsize(ann_path) > 2
        if has_annotations and not mask_sequence_is_videomama_ready(clip.root_path):
            raise CorridorKeyError(
                "VideoMaMa requires dense tracked masks. Run Track Mask first."
            )

        def _status(msg: str) -> None:
            logger.info(f"VideoMaMa [{clip.name}]: {msg}")
            if on_status:
                on_status(msg)

        def _check_cancel(phase: str = "") -> None:
            if job and job.is_cancelled:
                raise JobCancelledError(clip.name, 0)

        t_start = time.monotonic()

        # ── Phase 1: Load model ──
        _status("Loading model...")
        with self._gpu_lock:
            pipeline = self._get_videomama_pipeline()
        _check_cancel("model load")

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)

        # Don't report progress yet — phase status is showing in the status bar.
        # Sending on_progress(0, N) would switch the status bar to frame-counter
        # mode and overwrite the phase text on every tick.

        # ── Phase 2: Load input frames ──
        _status("Loading frames...")
        selected_input_names = self._selected_sequence_files(clip)
        if not selected_input_names:
            raise CorridorKeyError(f"Clip '{clip.name}' has no input frames for VideoMaMa")
        named_input_frames = self._load_named_sequence_frames(
            clip.input_asset,
            selected_input_names,
            clip.name,
            job=job,
            on_status=on_status,
        )
        input_frames = [frame for _, frame in named_input_frames]
        _check_cancel("frame load")

        # ── Phase 3: Load + stem-match masks ──
        _status("Loading masks...")
        mask_stems: dict[str, np.ndarray] = {}
        if clip.mask_asset.asset_type == 'sequence':
            mask_files = clip.mask_asset.get_frame_files()
            for i, fname in enumerate(mask_files):
                _check_cancel("mask load")
                fpath = os.path.join(clip.mask_asset.path, fname)
                m = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    _, binary = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
                    stem = os.path.splitext(fname)[0]
                    mask_stems[stem] = binary
        else:
            raw_masks = self._load_mask_frames_for_videomama(clip.mask_asset, clip.name)
            for i, m in enumerate(raw_masks):
                mask_stems[f"frame_{i:06d}"] = m

        # Build output filenames from the selected input stems
        input_names = [fname for fname, _ in named_input_frames]

        # Align masks to input frames by stem, defaulting to all-black
        num_frames = len(input_frames)
        mask_frames = []
        for fname in input_names:
            stem = os.path.splitext(fname)[0]
            if stem in mask_stems:
                mask_frames.append(mask_stems[stem])
            else:
                h_m, w_m = input_frames[0].shape[:2] if input_frames else (4, 4)
                mask_frames.append(np.zeros((h_m, w_m), dtype=np.uint8))

        # ── Resume logic ──
        existing_alpha = []
        if os.path.isdir(alpha_dir):
            existing_alpha = [f for f in os.listdir(alpha_dir)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        n_existing = len(existing_alpha)
        completed_chunks = n_existing // chunk_size
        start_chunk = max(0, completed_chunks - 1)
        start_frame = start_chunk * chunk_size
        if start_frame > 0:
            keep = set()
            for i in range(start_frame):
                if i < len(input_names):
                    stem = os.path.splitext(input_names[i])[0]
                    keep.add(f"{stem}.png")
            for fname in existing_alpha:
                if fname not in keep:
                    os.remove(os.path.join(alpha_dir, fname))
            logger.info(f"VideoMaMa resuming for '{clip.name}': {n_existing} alpha frames existed, "
                        f"rolling back to chunk {start_chunk} (frame {start_frame})")

        # ── Phase 4: Inference (per-chunk) ──
        sys.path.insert(0, os.path.join(BASE_DIR, "VideoMaMaInferenceModule"))
        from VideoMaMaInferenceModule.inference import run_inference

        total_chunks = (num_frames + chunk_size - 1) // chunk_size
        _status(f"Running inference (chunk 1/{total_chunks})...")
        frames_written = start_frame
        for chunk_idx, chunk_output in enumerate(
            run_inference(pipeline, input_frames, mask_frames,
                          chunk_size=chunk_size, on_status=on_status)
        ):
            _check_cancel("inference")

            # Skip already-completed chunks (resume)
            if chunk_idx < start_chunk:
                frames_written += len(chunk_output)
                if on_progress:
                    on_progress(clip.name, frames_written, num_frames)
                continue

            _status(f"Processing chunk {chunk_idx + 1}/{total_chunks}...")

            # Write chunk frames
            t_chunk = time.monotonic()
            for frame in chunk_output:
                if frame.dtype == np.uint8:
                    frame_u8 = frame
                else:
                    frame_u8 = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
                out_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
                if frames_written < len(input_names):
                    stem = os.path.splitext(input_names[frames_written])[0]
                    out_name = f"{stem}.png"
                else:
                    out_name = f"frame_{frames_written:06d}.png"
                out_path = os.path.join(alpha_dir, out_name)
                if not cv2.imwrite(out_path, out_bgr):
                    raise WriteFailureError(clip.name, frames_written, out_path)
                frames_written += 1
            logger.debug(f"Clip '{clip.name}' chunk {chunk_idx}: {len(chunk_output)} frames in {time.monotonic() - t_chunk:.3f}s")

            if on_progress:
                on_progress(clip.name, frames_written, num_frames)

        # Refresh alpha asset
        clip.alpha_asset = ClipAsset(alpha_dir, 'sequence')

        # Transition MASKED → READY
        try:
            clip.transition_to(ClipState.READY)
        except Exception as e:
            if on_warning:
                on_warning(f"State transition after VideoMaMa: {e}")

        logger.info(f"VideoMaMa complete for '{clip.name}': {frames_written} alpha frames in {time.monotonic() - t_start:.1f}s")

    def run_matanyone2(
        self,
        clip: ClipEntry,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Run MatAnyone2 video matting alpha generation for a clip.

        Transitions clip: MASKED → READY (creates AlphaHint directory).
        Requires a first-frame (frame 0) segmentation mask.

        Args:
            clip: Must be in MASKED state with input_asset and mask_asset.
            job: Optional GPUJob for cancel checking.
            on_progress: Progress callback (clip_name, current, total).
            on_warning: Warning callback.
            on_status: Phase status callback.
        """
        # ── Preflight validation ──
        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for MatAnyone2")
        if clip.mask_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing mask asset for MatAnyone2")
        if self._device == 'cpu':
            raise GPURequiredError("MatAnyone2")
        if clip.input_asset.asset_type != "sequence":
            raise CorridorKeyError("MatAnyone2 requires an extracted image sequence")

        def _status(msg: str) -> None:
            logger.info(f"MatAnyone2 [{clip.name}]: {msg}")
            if on_status:
                on_status(msg)

        def _check_cancel() -> bool:
            return bool(job and job.is_cancelled)

        t_start = time.monotonic()

        # ── Phase 1: Load model ──
        _status("Loading MatAnyone2 model...")
        with self._gpu_lock:
            processor = self._get_matanyone2()
        if _check_cancel():
            raise JobCancelledError(clip.name, 0)

        # ── Phase 2: Load input frames ──
        _status("Loading frames...")
        selected_input_names = self._selected_sequence_files(clip)
        if not selected_input_names:
            raise CorridorKeyError(f"Clip '{clip.name}' has no input frames for MatAnyone2")

        named_input_frames = self._load_named_sequence_frames(
            clip.input_asset,
            selected_input_names,
            clip.name,
            job=job,
            on_status=on_status,
        )
        input_frames = [frame for _, frame in named_input_frames]
        input_names = [fname for fname, _ in named_input_frames]
        frame_stems = [os.path.splitext(fname)[0] for fname in input_names]
        if _check_cancel():
            raise JobCancelledError(clip.name, 0)

        # ── Phase 3: Load first-frame mask ──
        _status("Loading first-frame mask...")
        mask_frame = self._load_first_frame_mask(clip, input_frames[0].shape[:2])
        if mask_frame is None:
            raise CorridorKeyError(
                f"Clip '{clip.name}': MatAnyone2 requires a mask for the first frame (frame 0). "
                f"Please ensure your annotation or mask covers the very first frame."
            )

        # ── Phase 4: Inference ──
        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        try:
            frames_written = processor.process_frames(
                input_frames=input_frames,
                mask_frame=mask_frame,
                output_dir=alpha_dir,
                frame_names=frame_stems,
                progress_callback=on_progress,
                on_status=on_status,
                cancel_check=_check_cancel,
                clip_name=clip.name,
            )
        except Exception as e:
            # On OOM, clean up and re-raise without poisoning service
            if "CUDA out of memory" in str(e) or "OutOfMemoryError" in type(e).__name__:
                logger.error(f"MatAnyone2 OOM for '{clip.name}': {e}")
                self._matanyone2_processor = None
                self._active_model = _ActiveModel.NONE
                try:
                    import torch as _torch
                    _torch.cuda.empty_cache()
                except Exception:
                    pass
                raise CorridorKeyError(
                    f"MatAnyone2 ran out of GPU memory processing '{clip.name}'. "
                    f"Try closing other GPU applications or using a smaller clip."
                ) from e
            raise

        # ── Phase 5: Finalize ──
        clip.alpha_asset = ClipAsset(alpha_dir, 'sequence')

        try:
            clip.transition_to(ClipState.READY)
        except Exception as e:
            if on_warning:
                on_warning(f"State transition after MatAnyone2: {e}")

        logger.info(
            f"MatAnyone2 complete for '{clip.name}': "
            f"{frames_written} alpha frames in {time.monotonic() - t_start:.1f}s"
        )

    def _load_first_frame_mask(
        self, clip: ClipEntry, frame_shape: tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Load the first-frame mask for MatAnyone2.

        Tries mask_asset's first file. Returns grayscale uint8 (H,W) or None.
        """
        if clip.mask_asset is None:
            return None

        if clip.mask_asset.asset_type == 'sequence':
            mask_files = clip.mask_asset.get_frame_files()
            if not mask_files:
                return None
            first_mask_path = os.path.join(clip.mask_asset.path, mask_files[0])
            mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return None
            # Binarize
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            # Resize to match input frame if needed
            target_h, target_w = frame_shape
            if mask.shape[:2] != (target_h, target_w):
                mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            return mask

        return None

    def _load_frames_for_videomama(
        self, asset: ClipAsset, clip_name: str,
        job: Optional[GPUJob] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> list[np.ndarray]:
        """Load input frames for VideoMaMa as uint8 RGB [0, 255].

        The VideoMaMa inference code expects uint8 arrays for PIL conversion.
        Reports loading progress via on_status and checks cancel via job.
        """
        if asset.asset_type == 'video':
            raw = read_video_frames(asset.path)
            return [(np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8) for f in raw]
        frames = []
        files = asset.get_frame_files()
        total = len(files)
        for i, fname in enumerate(files):
            if job and job.is_cancelled:
                from .errors import JobCancelledError
                raise JobCancelledError(clip_name, i)
            fpath = os.path.join(asset.path, fname)
            img = read_image_frame(fpath, gamma_correct_exr=True)
            if img is not None:
                frames.append((np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))
            if on_status and i % 20 == 0 and i > 0:
                on_status(f"Loading frames ({i}/{total})...")
        return frames

    def _load_mask_frames_for_videomama(
        self, asset: ClipAsset, clip_name: str
    ) -> list[np.ndarray]:
        """Load mask frames for VideoMaMa as uint8 grayscale [0, 255].

        The VideoMaMa inference code expects uint8 arrays for PIL conversion.
        Binary threshold at 10: anything above → 255 (foreground), else → 0.
        """
        def _threshold_mask(bgr_frame: np.ndarray) -> np.ndarray:
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            return binary  # uint8

        if asset.asset_type == 'video':
            return read_video_frames(asset.path, processor=_threshold_mask)
        masks = []
        for fname in asset.get_frame_files():
            fpath = os.path.join(asset.path, fname)
            mask = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            masks.append(binary)  # uint8
        return masks

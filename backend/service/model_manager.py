"""Model loading, residency management, and VRAM lifecycle.

Implements the single-model-in-VRAM policy: only one heavy model
stays loaded at a time. Switching triggers offload + gc + cache clear.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from enum import Enum
from typing import Callable

from .helpers import BASE_DIR, _import_matanyone2_processor_class

logger = logging.getLogger(__name__)


class _ActiveModel(Enum):
    """Tracks which heavy model is currently loaded in VRAM."""

    NONE = "none"
    INFERENCE = "inference"
    GVM = "gvm"
    SAM2 = "sam2"
    VIDEOMAMA = "videomama"
    MATANYONE2 = "matanyone2"
    BIREFNET = "birefnet"


class ModelManagerMixin:
    """Mixin providing model loading, residency, and VRAM management."""

    @staticmethod
    def _vram_allocated_mb() -> float:
        """Return current VRAM allocated in MB, or 0 if unavailable."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(0) / (1024**2)
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
            if hasattr(obj, "unload"):
                obj.unload()
            elif hasattr(obj, "to"):
                obj.to("cpu")
            elif hasattr(obj, "cpu"):
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
            logger.info(
                f"Unloading {self._active_model.value} model for {needed.value}"
                f" (VRAM before: {vram_before_mb:.0f}MB)"
            )

            t0 = time.monotonic()
            if on_status:
                on_status(f"Offloading {self._active_model.value}...")
            if self._active_model == _ActiveModel.INFERENCE:
                for eng in self._engine_pool:
                    self._safe_offload(eng)
                self._engine_pool.clear()
            elif self._active_model == _ActiveModel.GVM:
                # GVM has circular refs (pipe <-> vae <-> unet) — break them
                # explicitly so gc can reclaim everything in one pass.
                gvm = self._gvm_processor
                self._gvm_processor = None
                if gvm is not None:
                    self._safe_offload(gvm)
                    # Break circular references inside the diffusion pipeline
                    for attr in ("pipe", "vae", "unet", "scheduler"):
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
            elif self._active_model == _ActiveModel.BIREFNET:
                self._safe_offload(self._birefnet_processor)
                self._birefnet_processor = None
            logger.info(f"_safe_offload took {time.monotonic() - t0:.1f}s")

            import gc as _gc

            t0 = time.monotonic()
            # Two GC passes: first breaks cycles, second reclaims freed refs
            if on_status:
                on_status("Releasing Python references...")
            _gc.collect()
            _gc.collect()
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

            # Reset Triton/dynamo/inductor compilation cache so torch.compile
            # doesn't choke on stale CUDA state from the previous model
            try:
                import torch._dynamo

                torch._dynamo.reset()
                logger.info("torch._dynamo.reset() — cleared compilation cache")
            except Exception as e:
                logger.debug(f"dynamo reset skipped: {e}")

            try:
                import torch._inductor

                torch._inductor.codecache.PyCodeCache.clear()
                logger.info("torch._inductor code cache cleared")
            except Exception as e:
                logger.debug(f"inductor cache clear skipped: {e}")

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
            return self._engine_pool[: self._pool_size]

        from CorridorKeyModule.backend import create_engine, resolve_backend

        # Read user's backend preference (macOS: auto/mlx/torch)
        requested_backend = None
        try:
            from PySide6.QtCore import QSettings

            requested_backend = QSettings().value("inference/backend", "auto", type=str)
            if requested_backend == "auto":
                requested_backend = None  # let resolve_backend auto-detect
        except Exception:
            pass
        backend = resolve_backend(requested_backend)
        opt_mode = os.environ.get("CORRIDORKEY_OPT_MODE", "auto")
        _img_size = self._model_resolution

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
                    i + 1,
                    pool_size,
                    len(self._engine_pool),
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

    def _get_birefnet(self, usage: str = "Matting", on_status=None):
        """Lazy-load the BiRefNet processor for a given model variant."""
        self._ensure_model(_ActiveModel.BIREFNET, on_status=on_status)

        # If already loaded with the same usage, reuse
        if self._birefnet_processor is not None and self._birefnet_processor._usage == usage:
            return self._birefnet_processor

        # Different variant requested — release the old one
        if self._birefnet_processor is not None:
            self._safe_offload(self._birefnet_processor)
            self._birefnet_processor = None

        from modules.BiRefNetModule.wrapper import BiRefNetProcessor

        logger.info(f"Loading BiRefNet ({usage})...")
        t0 = time.monotonic()
        self._birefnet_processor = BiRefNetProcessor(
            device=self._device,
            usage=usage,
        )
        # Trigger actual model download/load now so VRAM is claimed
        self._birefnet_processor._ensure_loaded(on_status=on_status)
        logger.info(f"BiRefNet ({usage}) loaded in {time.monotonic() - t0:.1f}s")
        return self._birefnet_processor

    def unload_engines(self) -> None:
        """Free GPU memory by unloading all engines."""
        for eng in self._engine_pool:
            self._safe_offload(eng)
        self._engine_pool.clear()
        self._safe_offload(self._gvm_processor)
        self._safe_offload(self._sam2_tracker)
        self._safe_offload(self._videomama_pipeline)
        self._safe_offload(self._matanyone2_processor)
        self._safe_offload(self._birefnet_processor)
        self._gvm_processor = None
        self._sam2_tracker = None
        self._videomama_pipeline = None
        self._matanyone2_processor = None
        self._birefnet_processor = None
        self._active_model = _ActiveModel.NONE
        import gc as _gc

        _gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            logger.debug("torch not available for cache clear during unload")
        logger.info("All engines unloaded, VRAM freed")

    def is_engine_loaded(self) -> bool:
        """True if the inference engine is already loaded in VRAM."""
        return self._active_model == _ActiveModel.INFERENCE and len(self._engine_pool) > 0

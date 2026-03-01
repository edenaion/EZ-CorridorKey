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
import sys
import glob as glob_module
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
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
    scan_clips_dir,
)
from .errors import (
    CorridorKeyError,
    FrameReadError,
    WriteFailureError,
    JobCancelledError,
)
from .validators import (
    validate_frame_counts,
    normalize_mask_channels,
    normalize_mask_dtype,
    validate_frame_read,
    validate_write,
    ensure_output_dirs,
)
from .job_queue import GPUJob, GPUJobQueue, JobType

logger = logging.getLogger(__name__)

# Project paths — frozen-build aware
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# EXR write flags — PXR24 half-float (smallest working compression)
_EXR_FLAGS = [
    cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
    cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PXR24,
]


class _ActiveModel(Enum):
    """Tracks which heavy model is currently loaded in VRAM."""
    NONE = "none"
    INFERENCE = "inference"
    GVM = "gvm"
    VIDEOMAMA = "videomama"


@dataclass
class InferenceParams:
    """Frozen parameters for a single inference job."""
    input_is_linear: bool = False
    despill_strength: float = 1.0  # 0.0 to 1.0
    auto_despeckle: bool = True
    despeckle_size: int = 400
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
        self._engine = None
        self._gvm_processor = None
        self._videomama_pipeline = None
        self._active_model = _ActiveModel.NONE
        self._device: str = 'cpu'
        self._job_queue: Optional[GPUJobQueue] = None
        # GPU mutex — serializes ALL model operations (Codex: thread safety)
        self._gpu_lock = threading.Lock()

    @property
    def job_queue(self) -> GPUJobQueue:
        """Lazy-init GPU job queue (only needed when UI is running)."""
        if self._job_queue is None:
            self._job_queue = GPUJobQueue()
        return self._job_queue

    # --- Device & Engine Management ---

    def detect_device(self) -> str:
        """Detect best available compute device."""
        try:
            import torch
            if torch.cuda.is_available():
                self._device = 'cuda'
            else:
                self._device = 'cpu'
                logger.warning("CUDA not available — using CPU (will be very slow)")
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

    def _ensure_model(self, needed: _ActiveModel) -> None:
        """Model residency manager — unload current model if switching types.

        Only ONE heavy model stays in VRAM at a time. Before loading a
        different model, the previous is deleted and cache cleared.
        """
        if self._active_model == needed:
            return

        # Unload whatever is currently loaded
        if self._active_model != _ActiveModel.NONE:
            logger.info(f"Unloading {self._active_model.value} model for {needed.value}")
            if self._active_model == _ActiveModel.INFERENCE:
                self._engine = None
            elif self._active_model == _ActiveModel.GVM:
                self._gvm_processor = None
            elif self._active_model == _ActiveModel.VIDEOMAMA:
                self._videomama_pipeline = None

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                logger.debug("torch not available for cache clear during model switch")

        self._active_model = needed

    def _get_engine(self):
        """Lazy-load the CorridorKey inference engine."""
        self._ensure_model(_ActiveModel.INFERENCE)

        if self._engine is not None:
            return self._engine

        from CorridorKeyModule.inference_engine import CorridorKeyEngine

        ckpt_dir = os.path.join(BASE_DIR, "CorridorKeyModule", "checkpoints")
        ckpt_files = glob_module.glob(os.path.join(ckpt_dir, "*.pth"))

        if len(ckpt_files) == 0:
            raise FileNotFoundError(f"No .pth checkpoint found in {ckpt_dir}")
        elif len(ckpt_files) > 1:
            raise ValueError(
                f"Multiple checkpoints found in {ckpt_dir}. "
                f"Please ensure only one exists: {[os.path.basename(f) for f in ckpt_files]}"
            )

        ckpt_path = ckpt_files[0]
        logger.info(f"Loading checkpoint: {os.path.basename(ckpt_path)}")
        t0 = time.monotonic()
        self._engine = CorridorKeyEngine(
            checkpoint_path=ckpt_path,
            device=self._device,
            img_size=2048,
        )
        logger.info(f"Engine loaded in {time.monotonic() - t0:.1f}s")
        return self._engine

    def _get_gvm(self):
        """Lazy-load the GVM processor."""
        self._ensure_model(_ActiveModel.GVM)

        if self._gvm_processor is not None:
            return self._gvm_processor

        from gvm_core import GVMProcessor
        self._gvm_processor = GVMProcessor(device=self._device)
        return self._gvm_processor

    def _get_videomama_pipeline(self):
        """Lazy-load the VideoMaMa inference pipeline."""
        self._ensure_model(_ActiveModel.VIDEOMAMA)

        if self._videomama_pipeline is not None:
            return self._videomama_pipeline

        sys.path.insert(0, os.path.join(BASE_DIR, "VideoMaMaInferenceModule"))
        from VideoMaMaInferenceModule.inference import load_videomama_model
        self._videomama_pipeline = load_videomama_model(device=self._device)
        return self._videomama_pipeline

    def unload_engines(self) -> None:
        """Free GPU memory by unloading all engines."""
        self._engine = None
        self._gvm_processor = None
        self._videomama_pipeline = None
        self._active_model = _ActiveModel.NONE
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            logger.debug("torch not available for cache clear during unload")
        logger.info("All engines unloaded, VRAM freed")

    # --- Clip Scanning ---

    def scan_clips(self, clips_dir: str) -> list[ClipEntry]:
        """Scan a directory for clip folders."""
        return scan_clips_dir(clips_dir)

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
            (image_float32, stem_name, is_linear_override)
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
            is_exr = fpath.lower().endswith('.exr')

            if is_exr:
                img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                validate_frame_read(img, clip.name, frame_index, fpath)
                # Handle BGRA EXR — strip alpha channel
                if img.ndim == 3 and img.shape[2] == 4:
                    img = img[:, :, :3]
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return np.maximum(img_rgb, 0.0).astype(np.float32), input_stem, input_is_linear
            else:
                img_bgr = cv2.imread(fpath)
                validate_frame_read(img_bgr, clip.name, frame_index, fpath)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                return img_rgb.astype(np.float32) / 255.0, input_stem, input_is_linear

    def _read_alpha_frame(
        self,
        clip: ClipEntry,
        frame_index: int,
        alpha_files: list[str],
        alpha_cap: Optional[Any],
    ) -> Optional[np.ndarray]:
        """Read a single alpha/mask frame and normalize to [H, W] float32."""
        if alpha_cap:
            ret, frame = alpha_cap.read()
            if not ret:
                return None
            return frame[:, :, 2].astype(np.float32) / 255.0
        else:
            fpath = os.path.join(clip.alpha_asset.path, alpha_files[frame_index])
            mask_in = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
            validate_frame_read(mask_in, clip.name, frame_index, fpath)

            # Normalize channels — always extract to single channel
            mask = normalize_mask_channels(mask_in, clip.name, frame_index)
            # Normalize dtype to float32 [0, 1]
            mask = normalize_mask_dtype(mask)
            return mask

    def _write_image(
        self, img: np.ndarray, path: str, fmt: str, clip_name: str, frame_index: int,
    ) -> None:
        """Write a single image in the requested format."""
        if fmt == "exr":
            validate_write(cv2.imwrite(path, img, _EXR_FLAGS), clip_name, frame_index, path)
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
            self._write_image(fg_bgr, fg_path, cfg.fg_format, clip_name, frame_index)

        # Matte
        if cfg.matte_enabled:
            alpha = pred_alpha
            if alpha.ndim == 3:
                alpha = alpha[:, :, 0]
            matte_path = os.path.join(dirs['matte'], f"{input_stem}.{cfg.matte_format}")
            self._write_image(alpha, matte_path, cfg.matte_format, clip_name, frame_index)

        # Comp
        if cfg.comp_enabled:
            comp_srgb = res['comp']
            comp_bgr = cv2.cvtColor(
                (np.clip(comp_srgb, 0.0, 1.0) * 255.0).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            comp_path = os.path.join(dirs['comp'], f"{input_stem}.{cfg.comp_format}")
            self._write_image(comp_bgr, comp_path, cfg.comp_format, clip_name, frame_index)

        # Processed (RGBA premultiplied)
        if cfg.processed_enabled and 'processed' in res:
            proc_rgba = res['processed']
            proc_bgra = cv2.cvtColor(proc_rgba, cv2.COLOR_RGBA2BGRA)
            proc_path = os.path.join(dirs['processed'], f"{input_stem}.{cfg.processed_format}")
            self._write_image(proc_bgra, proc_path, cfg.processed_format, clip_name, frame_index)

    # --- Processing ---

    def run_inference(
        self,
        clip: ClipEntry,
        params: InferenceParams,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        skip_stems: Optional[set[str]] = None,
        output_config: Optional[OutputConfig] = None,
    ) -> list[FrameResult]:
        """Run CorridorKey inference on a single clip.

        Args:
            clip: Must be in READY or COMPLETE state with both input_asset and alpha_asset.
            params: Frozen inference parameters.
            job: Optional GPUJob for cancel checking.
            on_progress: Called with (clip_name, current_frame, total_frames).
            on_warning: Called with warning messages for non-fatal issues.
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

        t_start = time.monotonic()

        with self._gpu_lock:
            engine = self._get_engine()
        dirs = ensure_output_dirs(clip.root_path)
        cfg = output_config or OutputConfig()

        # Write run manifest (Codex: resume must know which outputs were enabled)
        self._write_manifest(dirs['root'], cfg, params)

        num_frames = validate_frame_counts(
            clip.name,
            clip.input_asset.frame_count,
            clip.alpha_asset.frame_count,
        )

        # Open video captures or get file lists
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

        results: list[FrameResult] = []
        skipped: list[int] = []
        skip_stems = skip_stems or set()

        try:
            for i in range(num_frames):
                # Check cancellation between frames
                if job and job.is_cancelled:
                    raise JobCancelledError(clip.name, i)

                # Report progress
                if on_progress and i % 5 == 0:
                    on_progress(clip.name, i, num_frames)

                try:
                    # Read input
                    img, input_stem, is_linear = self._read_input_frame(
                        clip, i, input_files, input_cap, params.input_is_linear,
                    )
                    if img is None:
                        skipped.append(i)
                        results.append(FrameResult(i, f"{i:05d}", False, "video read failed"))
                        continue

                    # Resume: skip frames that already have outputs
                    if input_stem in skip_stems:
                        results.append(FrameResult(i, input_stem, True, "resumed (skipped)"))
                        continue

                    # Read alpha
                    mask = self._read_alpha_frame(clip, i, alpha_files, alpha_cap)
                    if mask is None:
                        skipped.append(i)
                        results.append(FrameResult(i, input_stem, False, "alpha read failed"))
                        continue

                    # Resize mask if dimensions don't match input
                    if mask.shape[:2] != img.shape[:2]:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

                    # Process (GPU-locked — process_frame mutates model hooks)
                    t_frame = time.monotonic()
                    with self._gpu_lock:
                        res = engine.process_frame(
                            img,
                            mask,
                            input_is_linear=is_linear,
                            fg_is_straight=True,
                            despill_strength=params.despill_strength,
                            auto_despeckle=params.auto_despeckle,
                            despeckle_size=params.despeckle_size,
                            refiner_scale=params.refiner_scale,
                        )
                    logger.debug(f"Clip '{clip.name}' frame {i}: process_frame {time.monotonic() - t_frame:.3f}s")

                    # Write outputs
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

            # Final progress
            if on_progress:
                on_progress(clip.name, num_frames, num_frames)

        finally:
            if input_cap:
                input_cap.release()
            if alpha_cap:
                alpha_cap.release()

        # Summary
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
        logger.info(
            f"Clip '{clip.name}': inference complete. {processed}/{num_frames} frames "
            f"in {t_total:.1f}s ({t_total / max(processed, 1):.2f}s/frame avg)"
        )

        # State transition — only set COMPLETE if ALL frames succeeded
        if processed == num_frames:
            try:
                clip.transition_to(ClipState.COMPLETE)
            except Exception as e:
                logger.warning(f"Clip '{clip.name}': state transition to COMPLETE failed: {e}")

        return results

    # --- Single-Frame Reprocess (Preview) ---

    def is_engine_loaded(self) -> bool:
        """True if the inference engine is already loaded in VRAM."""
        return self._active_model == _ActiveModel.INFERENCE and self._engine is not None

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
            engine = self._get_engine()

        # Read the specific frame
        input_files = clip.input_asset.get_frame_files() if clip.input_asset.asset_type == 'sequence' else []
        if clip.input_asset.asset_type == 'video':
            cap = cv2.VideoCapture(clip.input_asset.path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            if frame_index >= len(input_files):
                return None
            fpath = os.path.join(clip.input_asset.path, input_files[frame_index])
            is_exr = fpath.lower().endswith('.exr')
            if is_exr:
                img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                if img is None:
                    return None
                if img.ndim == 3 and img.shape[2] == 4:
                    img = img[:, :, :3]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.maximum(img, 0.0).astype(np.float32)
            else:
                img = cv2.imread(fpath)
                if img is None:
                    return None
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Read alpha
        alpha_files = clip.alpha_asset.get_frame_files() if clip.alpha_asset.asset_type == 'sequence' else []
        if clip.alpha_asset.asset_type == 'video':
            cap = cv2.VideoCapture(clip.alpha_asset.path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            mask = frame[:, :, 2].astype(np.float32) / 255.0
        else:
            if frame_index >= len(alpha_files):
                return None
            fpath = os.path.join(clip.alpha_asset.path, alpha_files[frame_index])
            mask_in = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
            if mask_in is None:
                return None
            mask = normalize_mask_channels(mask_in, clip.name, frame_index)
            mask = normalize_mask_dtype(mask)

        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        if job and job.is_cancelled:
            return None

        with self._gpu_lock:
            res = engine.process_frame(
                img, mask,
                input_is_linear=params.input_is_linear,
                fg_is_straight=True,
                despill_strength=params.despill_strength,
                auto_despeckle=params.auto_despeckle,
                despeckle_size=params.despeckle_size,
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

        t_start = time.monotonic()

        with self._gpu_lock:
            gvm = self._get_gvm()

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)

        if on_progress:
            on_progress(clip.name, 0, 1)

        # Check cancel before starting (GVM call is monolithic)
        if job and job.is_cancelled:
            raise JobCancelledError(clip.name, 0)

        try:
            gvm.process_sequence(
                input_path=clip.input_asset.path,
                mode='matte',
                direct_output_dir=alpha_dir,
            )
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

    # --- VideoMaMa Alpha Generation ---

    def run_videomama(
        self,
        clip: ClipEntry,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        chunk_size: int = 50,
    ) -> None:
        """Run VideoMaMa guided alpha generation for a clip.

        Transitions clip: MASKED → READY (creates AlphaHint directory).

        Args:
            clip: Must be in MASKED state with input_asset and mask_asset.
            job: Optional GPUJob for cancel checking.
            on_progress: Progress callback with per-chunk updates.
            on_warning: Warning callback.
            chunk_size: Frames per chunk (lower = less RAM, default 50).
        """
        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for VideoMaMa")
        if clip.mask_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing mask asset for VideoMaMa")

        t_start = time.monotonic()

        with self._gpu_lock:
            pipeline = self._get_videomama_pipeline()

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)

        # Read input frames
        input_frames = self._load_frames_for_videomama(clip.input_asset, clip.name)
        mask_frames = self._load_mask_frames_for_videomama(clip.mask_asset, clip.name)

        num_frames = len(input_frames)
        total_chunks = (num_frames + chunk_size - 1) // chunk_size

        sys.path.insert(0, os.path.join(BASE_DIR, "VideoMaMaInferenceModule"))
        from VideoMaMaInferenceModule.inference import run_inference

        frames_written = 0
        for chunk_idx, chunk_output in enumerate(
            run_inference(pipeline, input_frames, mask_frames, chunk_size=chunk_size)
        ):
            # Check cancel between chunks
            if job and job.is_cancelled:
                raise JobCancelledError(clip.name, frames_written)

            # Write chunk frames
            t_chunk = time.monotonic()
            for frame in chunk_output:
                out_bgr = cv2.cvtColor(
                    (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                )
                out_path = os.path.join(alpha_dir, f"_alphaHint_{frames_written:05d}.png")
                cv2.imwrite(out_path, out_bgr)
                frames_written += 1
            logger.debug(f"Clip '{clip.name}' chunk {chunk_idx}: {len(chunk_output)} frames in {time.monotonic() - t_chunk:.3f}s")

            if on_progress:
                on_progress(clip.name, chunk_idx + 1, total_chunks)

        # Refresh alpha asset
        clip.alpha_asset = ClipAsset(alpha_dir, 'sequence')

        # Transition MASKED → READY
        try:
            clip.transition_to(ClipState.READY)
        except Exception as e:
            if on_warning:
                on_warning(f"State transition after VideoMaMa: {e}")

        logger.info(f"VideoMaMa complete for '{clip.name}': {frames_written} alpha frames in {time.monotonic() - t_start:.1f}s")

    def _load_frames_for_videomama(
        self, asset: ClipAsset, clip_name: str
    ) -> list[np.ndarray]:
        """Load input frames for VideoMaMa, applying gamma correction for EXR."""
        frames = []
        if asset.asset_type == 'video':
            cap = cv2.VideoCapture(asset.path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                frames.append(img_rgb)
            cap.release()
        else:
            for fname in asset.get_frame_files():
                fpath = os.path.join(asset.path, fname)
                is_exr = fpath.lower().endswith('.exr')
                if is_exr:
                    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    if img.ndim == 3 and img.shape[2] == 4:
                        img = img[:, :, :3]
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Gamma correction: linear EXR → sRGB for VideoMaMa
                    img_rgb = np.power(np.maximum(img_rgb, 0.0), 1.0 / 2.2).astype(np.float32)
                    frames.append(img_rgb)
                else:
                    img = cv2.imread(fpath)
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    frames.append(img_rgb)
        return frames

    def _load_mask_frames_for_videomama(
        self, asset: ClipAsset, clip_name: str
    ) -> list[np.ndarray]:
        """Load mask frames for VideoMaMa with binary thresholding."""
        masks = []
        if asset.asset_type == 'video':
            cap = cv2.VideoCapture(asset.path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Binary threshold at 10 (matching CLI behavior)
                _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                masks.append(binary.astype(np.float32) / 255.0)
            cap.release()
        else:
            for fname in asset.get_frame_files():
                fpath = os.path.join(asset.path, fname)
                mask = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
                masks.append(binary.astype(np.float32) / 255.0)
        return masks

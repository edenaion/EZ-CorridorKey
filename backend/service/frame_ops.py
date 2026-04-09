"""Frame I/O operations for the service layer.

Read/write individual frames, masks, manifests, and output images.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Optional

import cv2
import numpy as np

from ..clip_state import ClipAsset, ClipEntry
from ..errors import FrameReadError, JobCancelledError
from ..frame_io import (
    _linear_to_srgb,
    _srgb_to_linear,
    decode_video_mask_frame,
    read_video_frame_at,
    read_video_frames,
)
from ..validators import validate_frame_read, validate_write

logger = logging.getLogger(__name__)


class FrameOpsMixin:
    """Mixin providing frame I/O operations for CorridorKeyService."""

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
        from . import read_image_frame
        
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
        from . import read_mask_frame
        
        if alpha_cap:
            ret, frame = alpha_cap.read()
            if not ret:
                return None
            return decode_video_mask_frame(frame)
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
        import cv2
        from . import write_exr
        
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
        output_config,
        params,
    ) -> None:
        """Write run manifest recording expected outputs/extensions per run.

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
        output_config=None,
    ) -> None:
        """Write output types for a single frame respecting OutputConfig."""
        from .core import OutputConfig
        cfg = output_config or OutputConfig()
        logger.debug(f"Writing outputs for '{clip_name}' frame {frame_index} stem='{input_stem}'")

        pred_fg = res['fg']
        pred_alpha = res['alpha']

        # FG
        if cfg.fg_enabled:
            fg_rgb = pred_fg
            if cfg.fg_format == "exr":
                fg_rgb = _srgb_to_linear(fg_rgb)
            fg_bgr = cv2.cvtColor(fg_rgb.astype(np.float32), cv2.COLOR_RGB2BGR)
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
            if cfg.comp_format == "exr":
                comp_rgb = _srgb_to_linear(comp_srgb)
                comp_bgr = cv2.cvtColor(comp_rgb.astype(np.float32), cv2.COLOR_RGB2BGR)
            else:
                comp_bgr = cv2.cvtColor(
                    (np.clip(comp_srgb, 0.0, 1.0) * 255.0).astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                )
            comp_path = os.path.join(dirs['comp'], f"{input_stem}.{cfg.comp_format}")
            self._write_image(comp_bgr, comp_path, cfg.comp_format, clip_name, frame_index,
                              exr_compression=cfg.exr_compression)

        # Processed (RGBA — linear premul for EXR, sRGB straight for PNG)
        if cfg.processed_enabled and 'processed' in res:
            proc_rgba = res['processed']  # [H,W,4] premultiplied linear float
            if cfg.processed_format == "exr":
                proc_bgra = cv2.cvtColor(proc_rgba, cv2.COLOR_RGBA2BGRA)
            else:
                # PNG needs sRGB straight alpha, not premultiplied linear.
                # Un-premultiply, convert linear→sRGB, recombine with alpha.
                rgb_premul = proc_rgba[:, :, :3]
                alpha = proc_rgba[:, :, 3:4]
                safe_alpha = np.where(alpha > 1e-6, alpha, 1.0)
                rgb_straight = rgb_premul / safe_alpha
                rgb_srgb = _linear_to_srgb(np.clip(rgb_straight, 0.0, 1.0))
                proc_srgb = np.concatenate([rgb_srgb, alpha], axis=-1)
                proc_bgra = cv2.cvtColor(proc_srgb, cv2.COLOR_RGBA2BGRA)
            proc_path = os.path.join(dirs['processed'], f"{input_stem}.{cfg.processed_format}")
            self._write_image(proc_bgra, proc_path, cfg.processed_format, clip_name, frame_index,
                              exr_compression=cfg.exr_compression)

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
        job=None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> list[np.ndarray]:
        """Load input frames for VideoMaMa as uint8 RGB [0, 255]."""
        from . import read_image_frame
        
        if asset.asset_type == 'video':
            raw = read_video_frames(asset.path)
            return [(np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8) for f in raw]
        frames = []
        files = asset.get_frame_files()
        total = len(files)
        for i, fname in enumerate(files):
            if job and job.is_cancelled:
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
        """Load mask frames for VideoMaMa as uint8 grayscale [0, 255]."""
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
        job=None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> list[tuple[str, np.ndarray]]:
        """Load named image-sequence frames as uint8 RGB for tracker/VideoMaMa."""
        from . import read_image_frame
        
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
        from ..clip_state import MASK_TRACK_MANIFEST
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
        import shutil
        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir):
            shutil.rmtree(alpha_dir, ignore_errors=True)

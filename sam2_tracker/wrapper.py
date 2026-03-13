"""Thin SAM2 video-tracking wrapper used by CorridorKey."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_PROMPT_REFINEMENT_POS_BATCH = 6
_PROMPT_REFINEMENT_NEG_BATCH = 2


def _disable_external_progress_bars() -> None:
    """Disable third-party console progress bars in the GUI integration."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    try:
        from huggingface_hub.utils import disable_progress_bars
    except Exception:
        disable_progress_bars = None
    if disable_progress_bars is not None:
        try:
            disable_progress_bars()
        except Exception:
            logger.debug("Failed to disable HF progress bars", exc_info=True)

    try:
        from tqdm import tqdm as base_tqdm
        import sam2.sam2_video_predictor as sam2_video_predictor
        import sam2.utils.misc as sam2_misc
    except Exception:
        return

    def _silent_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return base_tqdm(*args, **kwargs)

    sam2_misc.tqdm = _silent_tqdm
    sam2_video_predictor.tqdm = _silent_tqdm


@dataclass(frozen=True)
class PromptFrame:
    """Prompt bundle for one frame in clip-local coordinates."""

    frame_index: int
    positive_points: list[tuple[float, float]] = field(default_factory=list)
    negative_points: list[tuple[float, float]] = field(default_factory=list)
    box: tuple[float, float, float, float] | None = None
    mask: np.ndarray | None = None


class SAM2NotInstalledError(RuntimeError):
    """Raised when the optional SAM2 dependency is unavailable."""


class SAM2Tracker:
    """Lazy-loading wrapper around Meta's SAM2 video predictor."""

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-base-plus",
        *,
        device: str = "cuda",
        vos_optimized: bool = False,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.vos_optimized = vos_optimized
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu
        self._predictor = None

    def unload(self) -> None:
        """Move the predictor back to CPU if possible."""
        if self._predictor is not None and hasattr(self._predictor, "to"):
            try:
                self._predictor.to("cpu")
            except Exception:
                logger.debug("SAM2 predictor CPU offload skipped", exc_info=True)

    def prepare(
        self,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        """Ensure the predictor is loaded and the checkpoint is present locally."""
        self._get_predictor(on_progress=on_progress, on_status=on_status)

    def _make_download_progress_class(
        self,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ):
        class _DownloadProgress:
            def __init__(self, *args, total=None, initial=0, desc="", disable=False, **kwargs):
                self.total = int(total or 0)
                self.n = int(initial or 0)
                self.desc = desc or "SAM2 model"
                if on_status:
                    on_status(f"Downloading {self.desc}")
                if on_progress and self.total > 0:
                    on_progress(self.n, self.total)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.close()
                return False

            def update(self, n=1):
                self.n += int(n or 0)
                if on_progress and self.total > 0:
                    on_progress(min(self.n, self.total), self.total)

            def close(self):
                if on_progress and self.total > 0:
                    on_progress(self.total, self.total)
                if on_status:
                    on_status(f"Downloaded {self.desc}")

        return _DownloadProgress

    def _get_predictor(
        self,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ):
        if self._predictor is not None:
            return self._predictor

        try:
            from huggingface_hub import hf_hub_download
            from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2_video_predictor
        except ImportError as exc:
            raise SAM2NotInstalledError(
                "SAM2 is not installed. Install the optional tracker dependency "
                "to generate dense masks from annotations."
            ) from exc

        # GUI launches already have their own progress UI; external console bars
        # only create stderr failures and noisy logs.
        if sys.stderr is None:
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        _disable_external_progress_bars()

        logger.info(
            "Loading SAM2 tracker (%s, vos_optimized=%s)",
            self.model_id,
            self.vos_optimized,
        )
        config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[self.model_id]
        if on_status:
            on_status("Checking model cache")
        ckpt_path = hf_hub_download(
            repo_id=self.model_id,
            filename=checkpoint_name,
            tqdm_class=self._make_download_progress_class(
                on_progress=on_progress,
                on_status=on_status,
            ),
        )
        self._predictor = build_sam2_video_predictor(
            config_file=config_name,
            ckpt_path=ckpt_path,
            device=self.device,
            vos_optimized=self.vos_optimized,
        )
        return self._predictor

    def track_video(
        self,
        frames: Sequence[np.ndarray],
        prompt_frames: Sequence[PromptFrame],
        *,
        on_progress: Callable[[int, int], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        check_cancel: Callable[[], None] | None = None,
    ) -> list[np.ndarray]:
        """Track a single object through a clip from sparse prompt frames."""
        if not frames:
            return []
        if not prompt_frames:
            raise ValueError("SAM2 tracking requires at least one prompt frame")

        frame_shape = frames[0].shape[:2]
        sanitized_prompts = [
            self._sanitize_prompt_frame(prompt, frame_shape)
            for prompt in prompt_frames
        ]
        sanitized_prompts = [
            prompt for prompt in sanitized_prompts
            if (
                prompt.mask is not None
                or prompt.positive_points
                or prompt.negative_points
                or prompt.box is not None
            )
        ]
        if logger.isEnabledFor(logging.INFO):
            total_pos = sum(len(prompt.positive_points) for prompt in sanitized_prompts)
            total_neg = sum(len(prompt.negative_points) for prompt in sanitized_prompts)
            total_mask_frames = sum(1 for prompt in sanitized_prompts if prompt.mask is not None)
            total_mask_pixels = sum(
                int(np.count_nonzero(prompt.mask))
                for prompt in sanitized_prompts
                if prompt.mask is not None
            )
            logger.info(
                "SAM2 prompts after sanitize: frames=%d, fg=%d, bg=%d, mask_frames=%d, fg_pixels=%d",
                len(sanitized_prompts),
                total_pos,
                total_neg,
                total_mask_frames,
                total_mask_pixels,
            )
        if not sanitized_prompts:
            raise ValueError("SAM2 tracking requires at least one usable prompt frame")
        if not any(self._has_foreground_signal(prompt) for prompt in sanitized_prompts):
            raise ValueError("SAM2 tracking requires at least one non-empty foreground prompt")

        predictor = self._get_predictor(on_progress=on_progress, on_status=on_status)

        try:
            import torch
        except ImportError as exc:
            raise SAM2NotInstalledError("PyTorch is required for SAM2 tracking") from exc

        temp_root = Path(tempfile.mkdtemp(prefix="corridorkey_sam2_"))
        frames_dir = temp_root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            if on_status:
                on_status("Preparing JPEG frames for SAM2")
            for idx, frame in enumerate(frames):
                if check_cancel:
                    check_cancel()
                frame_path = frames_dir / f"{idx:05d}.jpg"
                Image.fromarray(frame).save(frame_path, quality=95)

            autocast_ctx = nullcontext()
            if self.device.startswith("cuda") and torch.cuda.is_available():
                autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)

            masks_by_frame: dict[int, np.ndarray] = {}
            total = len(frames)
            sorted_prompts = sorted(sanitized_prompts, key=lambda item: item.frame_index)
            earliest_prompt = sorted_prompts[0].frame_index
            latest_prompt = sorted_prompts[-1].frame_index

            with torch.inference_mode(), autocast_ctx:
                if on_status:
                    on_status("Initializing SAM2")
                inference_state = predictor.init_state(
                    video_path=str(frames_dir),
                    offload_video_to_cpu=self.offload_video_to_cpu,
                    offload_state_to_cpu=self.offload_state_to_cpu,
                )

                if on_status:
                    on_status("Applying annotation prompts")
                for prompt in sorted_prompts:
                    if check_cancel:
                        check_cancel()
                    if prompt.mask is not None:
                        frame_idx, obj_ids, mask_logits = predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=prompt.frame_index,
                            obj_id=1,
                            mask=(prompt.mask > 0),
                        )
                    else:
                        frame_idx = prompt.frame_index
                        obj_ids = []
                        mask_logits = None
                        for batch_points, batch_labels, batch_box, clear_old_points in self._iter_prompt_refinement_batches(prompt):
                            if check_cancel:
                                check_cancel()
                            frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=prompt.frame_index,
                                obj_id=1,
                                points=batch_points if batch_points.size else None,
                                labels=batch_labels if batch_labels.size else None,
                                clear_old_points=clear_old_points,
                                box=batch_box,
                            )
                    masks_by_frame[frame_idx] = self._extract_object_mask(
                        obj_ids=obj_ids,
                        mask_logits=mask_logits,
                        fallback_shape=frames[0].shape[:2],
                    )
                    if on_progress:
                        on_progress(len(masks_by_frame), total)

                if on_status:
                    on_status("SAM2 propagation")
                for pass_start, reverse in (
                    (earliest_prompt, False),
                    (latest_prompt, True),
                ):
                    max_frames = total - pass_start if not reverse else pass_start + 1
                    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=pass_start,
                        max_frame_num_to_track=max_frames,
                        reverse=reverse,
                    ):
                        if check_cancel:
                            check_cancel()
                        masks_by_frame[frame_idx] = self._extract_object_mask(
                            obj_ids=obj_ids,
                            mask_logits=mask_logits,
                            fallback_shape=frames[0].shape[:2],
                        )
                        if on_progress:
                            on_progress(len(masks_by_frame), total)

            empty = np.zeros(frames[0].shape[:2], dtype=np.uint8)
            return [masks_by_frame.get(i, empty.copy()) for i in range(total)]
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    @staticmethod
    def _points_and_labels(prompt: PromptFrame) -> tuple[np.ndarray, np.ndarray]:
        points: list[tuple[float, float]] = []
        labels: list[int] = []
        for x, y in prompt.positive_points:
            points.append((x, y))
            labels.append(1)
        for x, y in prompt.negative_points:
            points.append((x, y))
            labels.append(0)
        if not points:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)
        return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32)

    @staticmethod
    def _iter_prompt_refinement_batches(
        prompt: PromptFrame,
        *,
        positive_batch_size: int = _PROMPT_REFINEMENT_POS_BATCH,
        negative_batch_size: int = _PROMPT_REFINEMENT_NEG_BATCH,
    ):
        """Yield same-frame prompt refinements in SAM-friendly sparse batches."""
        pos_batch = max(1, int(positive_batch_size))
        neg_batch = max(1, int(negative_batch_size))
        pos_cursor = 0
        neg_cursor = 0
        step = 0

        if not prompt.positive_points and not prompt.negative_points:
            yield (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                prompt.box,
                True,
            )
            return

        while pos_cursor < len(prompt.positive_points) or neg_cursor < len(prompt.negative_points):
            pos_chunk = prompt.positive_points[pos_cursor:pos_cursor + pos_batch]
            neg_chunk = prompt.negative_points[neg_cursor:neg_cursor + neg_batch]
            points: list[tuple[float, float]] = [*pos_chunk, *neg_chunk]
            labels: list[int] = ([1] * len(pos_chunk)) + ([0] * len(neg_chunk))
            yield (
                np.asarray(points, dtype=np.float32),
                np.asarray(labels, dtype=np.int32),
                prompt.box if step == 0 else None,
                step == 0,
            )
            pos_cursor += len(pos_chunk)
            neg_cursor += len(neg_chunk)
            step += 1

    @staticmethod
    def _sanitize_prompt_frame(
        prompt: PromptFrame,
        frame_shape: tuple[int, int],
    ) -> PromptFrame:
        """Clamp prompt geometry to frame bounds and drop non-finite values."""
        height, width = frame_shape
        max_x = float(max(0, width - 1))
        max_y = float(max(0, height - 1))

        def _clamp_points(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
            cleaned: list[tuple[float, float]] = []
            seen: set[tuple[int, int]] = set()
            for x, y in points:
                xf, yf = float(x), float(y)
                if not np.isfinite(xf) or not np.isfinite(yf):
                    continue
                clamped_x = min(max(xf, 0.0), max_x)
                clamped_y = min(max(yf, 0.0), max_y)
                key = (int(round(clamped_x)), int(round(clamped_y)))
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append((float(key[0]), float(key[1])))
            return cleaned

        def _clamp_box(
            box: tuple[float, float, float, float] | None,
        ) -> tuple[float, float, float, float] | None:
            if box is None:
                return None
            x0, y0, x1, y1 = (float(v) for v in box)
            if not all(np.isfinite(v) for v in (x0, y0, x1, y1)):
                return None
            lo_x = min(max(min(x0, x1), 0.0), max_x)
            hi_x = min(max(max(x0, x1), 0.0), max_x)
            lo_y = min(max(min(y0, y1), 0.0), max_y)
            hi_y = min(max(max(y0, y1), 0.0), max_y)
            if hi_x <= lo_x or hi_y <= lo_y:
                return None
            return (lo_x, lo_y, hi_x, hi_y)

        mask = None
        if prompt.mask is not None:
            mask = np.asarray(prompt.mask)
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = np.squeeze(mask, axis=-1)
            if mask.ndim != 2:
                raise ValueError("SAM2 mask prompts must be 2D arrays")
            if mask.shape != frame_shape:
                raise ValueError(
                    "SAM2 mask prompt dimensions must match the input frame size"
                )
            mask = np.where(mask > 0, 255, 0).astype(np.uint8, copy=False)

        positive_points = _clamp_points(prompt.positive_points)
        negative_points = _clamp_points(prompt.negative_points)
        box = _clamp_box(prompt.box)
        if mask is not None:
            positive_points = []
            negative_points = []
            box = None

        return PromptFrame(
            frame_index=int(prompt.frame_index),
            positive_points=positive_points,
            negative_points=negative_points,
            box=box,
            mask=mask,
        )

    @staticmethod
    def _has_foreground_signal(prompt: PromptFrame) -> bool:
        if prompt.mask is not None:
            return bool(np.any(prompt.mask > 0))
        return bool(prompt.positive_points or prompt.box is not None)

    @staticmethod
    def _extract_object_mask(
        *,
        obj_ids,
        mask_logits,
        fallback_shape: tuple[int, int],
        object_id: int = 1,
    ) -> np.ndarray:
        ids = obj_ids.tolist() if hasattr(obj_ids, "tolist") else list(obj_ids)
        if object_id not in ids:
            return np.zeros(fallback_shape, dtype=np.uint8)

        idx = ids.index(object_id)
        mask = (mask_logits[idx] > 0.0).detach().cpu().numpy()
        return (np.squeeze(mask).astype(np.uint8) * 255)

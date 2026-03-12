"""Thin SAM2 video-tracking wrapper used by CorridorKey."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


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
    positive_points: list[tuple[float, float]]
    negative_points: list[tuple[float, float]]
    box: tuple[float, float, float, float] | None = None


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
        if not any(p.positive_points for p in prompt_frames):
            raise ValueError("SAM2 tracking requires at least one foreground prompt")

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
            sorted_prompts = sorted(prompt_frames, key=lambda item: item.frame_index)
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
                    points, labels = self._points_and_labels(prompt)
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=prompt.frame_index,
                        obj_id=1,
                        points=points if points.size else None,
                        labels=labels if labels.size else None,
                        box=prompt.box,
                    )

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

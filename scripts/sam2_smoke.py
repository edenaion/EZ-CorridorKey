"""Quick SAM2 smoke test on the first N frames of a real clip.

This is intentionally small and fast:
- loads only the first `--frames` input frames
- loads only prompt frames in that same range
- runs the real SAM2 tracker
- writes an overlay contact sheet for visual inspection
- fails on obvious black-mask or blowout regressions
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from backend.annotation_prompts import load_annotation_mask_frames
from backend.annotation_prompts import load_annotation_prompt_frames
from backend.clip_state import ClipAsset
from backend.service import CorridorKeyService
from sam2_tracker import PromptFrame, SAM2Tracker

SAM2_MODEL_IDS = {
    "small": "facebook/sam2.1-hiera-small",
    "base-plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}


def _print(message: str) -> None:
    print(f"[sam2-smoke] {message}", flush=True)


def _resolve_frames_dir(clip_root: Path) -> Path:
    frames_dir = clip_root / "Frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Missing Frames/ under {clip_root}")
    return frames_dir


def _input_files(frames_dir: Path) -> list[str]:
    files = sorted(
        name for name in os.listdir(frames_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff"))
    )
    if not files:
        raise FileNotFoundError(f"No image sequence files found in {frames_dir}")
    return files


def _write_contact_sheet(
    out_path: Path,
    named_frames: list[tuple[str, np.ndarray]],
    masks: list[np.ndarray],
) -> None:
    thumbs: list[Image.Image] = []
    for idx, ((_, frame), mask) in enumerate(zip(named_frames, masks)):
        fill = float((mask > 0).mean())
        comp = _outline_overlay(frame, mask)
        preview = comp.copy()
        preview.thumbnail((900, 475))
        bordered = ImageOps.expand(preview, border=6, fill=(20, 20, 20))
        draw = ImageDraw.Draw(bordered)
        draw.rectangle((0, 0, bordered.width, 34), fill=(0, 0, 0))
        draw.text((10, 8), f"frame {idx} fill {fill:.3f}", fill=(255, 255, 255))
        thumbs.append(bordered)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_w = max(im.width for im in thumbs)
    sheet_h = sum(im.height for im in thumbs)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (10, 10, 10))
    y = 0
    for im in thumbs:
        sheet.paste(im.convert("RGB"), (0, y))
        y += im.height
    sheet.save(out_path)


def _outline_overlay(frame: np.ndarray, mask: np.ndarray) -> Image.Image:
    overlay = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (0, 230, 255), 4)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


def _write_mask_frames(
    out_dir: Path,
    named_frames: list[tuple[str, np.ndarray]],
    masks: list[np.ndarray],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for (name, _), mask in zip(named_frames, masks):
        stem = Path(name).stem
        Image.fromarray(mask).save(out_dir / f"{stem}.png")


def run_smoke(
    clip_root: Path,
    *,
    frames: int,
    sam2_model: str,
    prompt_mode: str,
    min_fill: float,
    max_fill: float,
    output: Path,
) -> int:
    frames_dir = _resolve_frames_dir(clip_root)
    file_names = _input_files(frames_dir)[:frames]

    service = CorridorKeyService()
    asset = ClipAsset(str(frames_dir), "sequence")
    named_frames = service._load_named_sequence_frames(asset, file_names, clip_root.name)
    tracker = SAM2Tracker(model_id=SAM2_MODEL_IDS[sam2_model], device=service._device)
    frame_height, frame_width = named_frames[0][1].shape[:2]
    if prompt_mode == "mask":
        prompt_frames = load_annotation_mask_frames(
            str(clip_root),
            width=frame_width,
            height=frame_height,
            allowed_indices=list(range(len(file_names))),
        )
        tracker_prompts = [
            PromptFrame(
                frame_index=prompt.frame_index,
                mask=prompt.mask,
            )
            for prompt in prompt_frames
        ]
        prompt_counts = [
            (
                prompt.frame_index,
                int(np.count_nonzero(prompt.mask)),
                round(float((prompt.mask > 0).mean()), 4),
            )
            for prompt in prompt_frames
        ]
    else:
        prompt_frames = load_annotation_prompt_frames(
            str(clip_root),
            allowed_indices=list(range(len(file_names))),
        )
        tracker_prompts = [
            PromptFrame(
                frame_index=prompt.frame_index,
                positive_points=prompt.positive_points,
                negative_points=prompt.negative_points,
                box=prompt.box,
            )
            for prompt in prompt_frames
        ]
        prompt_counts = [
            (
                prompt.frame_index,
                len(prompt.positive_points),
                len(prompt.negative_points),
                prompt.box is not None,
            )
            for prompt in prompt_frames
        ]

    if not prompt_frames:
        _print("no usable annotation prompts found in the requested frame range")
        return 2

    masks = tracker.track_video(
        [frame for _, frame in named_frames],
        tracker_prompts,
    )

    fills = [float((mask > 0).mean()) for mask in masks]
    _write_contact_sheet(output, named_frames, masks)
    _write_mask_frames(output.parent / "masks", named_frames, masks)

    _print(f"clip: {clip_root}")
    _print(f"frames tested: {len(named_frames)}")
    _print(f"prompt mode: {prompt_mode}")
    if prompt_mode == "mask":
        _print(f"prompt frames (index, fg_pixels, fill): {prompt_counts}")
    else:
        _print(f"prompt frames (index, fg_points, bg_points, has_box): {prompt_counts}")
    _print(f"fills: {[round(v, 4) for v in fills]}")
    _print(f"contact sheet: {output}")
    _print(f"mask frames: {output.parent / 'masks'}")

    black_frames = [idx for idx, fill in enumerate(fills) if fill <= 0.0]
    low_frames = [idx for idx, fill in enumerate(fills) if 0.0 < fill < min_fill]
    high_frames = [idx for idx, fill in enumerate(fills) if fill > max_fill]

    if black_frames:
        _print(f"FAIL: black masks on frames {black_frames}")
        return 1
    if low_frames:
        _print(f"FAIL: suspiciously sparse masks on frames {low_frames}")
        return 1
    if high_frames:
        _print(f"FAIL: suspiciously large masks on frames {high_frames}")
        return 1

    _print("PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick SAM2 smoke test on the first N frames of a clip")
    parser.add_argument("clip_root", help="Path to the clip root containing Frames/ and annotations.json")
    parser.add_argument("--frames", type=int, default=5, help="Number of initial frames to test (default: 5)")
    parser.add_argument(
        "--sam2-model",
        choices=sorted(SAM2_MODEL_IDS.keys()),
        default="base-plus",
        help="SAM2 checkpoint to use (default: base-plus)",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("point", "mask"),
        default="point",
        help="How to interpret annotations for SAM2 (default: point)",
    )
    parser.add_argument(
        "--min-fill",
        type=float,
        default=0.01,
        help="Fail if any non-black mask covers less than this fraction of the frame",
    )
    parser.add_argument(
        "--max-fill",
        type=float,
        default=0.60,
        help="Fail if any mask covers more than this fraction of the frame",
    )
    parser.add_argument(
        "--output",
        default=".tmp/sam2_smoke/contact.png",
        help="Where to write the overlay contact sheet",
    )
    args = parser.parse_args()

    if args.frames <= 0:
        print("--frames must be > 0", file=sys.stderr)
        return 2

    return run_smoke(
        Path(args.clip_root),
        frames=args.frames,
        sam2_model=args.sam2_model,
        prompt_mode=args.prompt_mode,
        min_fill=args.min_fill,
        max_fill=args.max_fill,
        output=Path(args.output),
    )


if __name__ == "__main__":
    raise SystemExit(main())

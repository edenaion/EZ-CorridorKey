"""Repeatable local GPU QA harness for SAM2 and VideoMaMa.

This script generates a tiny synthetic green-screen clip, writes real
annotations.json prompts, and then runs the same backend service paths the
desktop app uses:

    annotations.json -> run_sam2_track() -> VideoMamaMaskHint/
    VideoMamaMaskHint/ -> run_videomama() -> AlphaHint/

It is meant for manual QA on a real CUDA machine. The pass/fail criteria are
kept deliberately narrow:
- SAM2 must write a dense mask sequence and achieve a minimum mean IoU against
  the synthetic ground-truth masks.
- VideoMaMa must write AlphaHint frames for every input frame and meet a
  minimum mean IoU against the same synthetic ground-truth masks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from backend import ClipAsset, ClipEntry, ClipState, CorridorKeyService

SAM2_MODEL_IDS = {
    "small": "facebook/sam2.1-hiera-small",
    "base-plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}


def _human_part(
    image: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    center: tuple[int, int] | None = None,
    radius: int | None = None,
    ellipse_axes: tuple[int, int] | None = None,
    line: tuple[tuple[int, int], tuple[int, int], int] | None = None,
) -> None:
    """Draw one body part onto both the RGB frame and the binary mask."""
    if center is not None and radius is not None:
        cv2.circle(image, center, radius, color_bgr, -1, lineType=cv2.LINE_AA)
        cv2.circle(mask, center, radius, 255, -1, lineType=cv2.LINE_AA)
        return
    if center is not None and ellipse_axes is not None:
        cv2.ellipse(image, center, ellipse_axes, 0, 0, 360, color_bgr, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(mask, center, ellipse_axes, 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)
        return
    if line is not None:
        start, end, thickness = line
        cv2.line(image, start, end, color_bgr, thickness, lineType=cv2.LINE_AA)
        cv2.line(mask, start, end, 255, thickness, lineType=cv2.LINE_AA)


def _make_fixture_frame(
    index: int,
    total: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int]]]:
    """Generate one synthetic green-screen frame plus its binary ground truth."""
    # Background: green-screen with a subtle gradient so the clip is less trivial.
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    bg[..., 0] = np.clip(30 + 18 * xv, 0, 255).astype(np.uint8)  # B
    bg[..., 1] = np.clip(135 + 55 * (1.0 - yv), 0, 255).astype(np.uint8)  # G
    bg[..., 2] = np.clip(25 + 10 * xv, 0, 255).astype(np.uint8)  # R

    # Humanoid motion path.
    t = index / max(total - 1, 1)
    cx = 96 + int((width - 192) * t)
    cy = height // 2 + int(8 * math.sin(index * 0.65))
    arm_swing = int(16 * math.sin(index * 0.8))
    leg_swing = int(12 * math.cos(index * 0.7))

    frame = bg.copy()
    mask = np.zeros((height, width), dtype=np.uint8)

    head = (cx, cy - 70)
    torso = (cx, cy - 15)
    left_hand = (cx - 42, cy - 20 + arm_swing)
    right_hand = (cx + 42, cy - 20 - arm_swing)
    left_foot = (cx - 18 - leg_swing, cy + 78)
    right_foot = (cx + 18 + leg_swing, cy + 78)

    # Hair / head / torso / arms / legs
    _human_part(frame, mask, (25, 25, 25), center=(head[0], head[1] - 6), radius=16)
    _human_part(frame, mask, (172, 191, 219), center=head, radius=16)
    _human_part(frame, mask, (72, 86, 145), center=torso, ellipse_axes=(26, 44))
    _human_part(frame, mask, (68, 78, 126), line=((cx - 14, cy - 42), left_hand, 12))
    _human_part(frame, mask, (68, 78, 126), line=((cx + 14, cy - 42), right_hand, 12))
    _human_part(frame, mask, (58, 62, 78), line=((cx - 10, cy + 18), left_foot, 14))
    _human_part(frame, mask, (58, 62, 78), line=((cx + 10, cy + 18), right_foot, 14))

    # Add a simple white shirt highlight so the figure has internal structure.
    cv2.ellipse(
        frame, (cx, cy - 18), (12, 24), 0, 0, 360, (180, 190, 220), -1, lineType=cv2.LINE_AA
    )

    return (
        frame,
        mask,
        {
            "head": head,
            "torso": torso,
            "hips": (cx, cy + 18),
            "left_shoulder": (cx - 26, cy - 34),
            "right_shoulder": (cx + 26, cy - 34),
            "left_hand": left_hand,
            "right_hand": right_hand,
            "left_foot": left_foot,
            "right_foot": right_foot,
            "left_bg": (max(18, cx - 86), cy - 12),
            "right_bg": (min(width - 18, cx + 86), cy - 12),
            "low_bg": (cx, min(height - 18, cy + 108)),
        },
    )


def _write_fixture_clip(root: Path, *, frames: int, width: int, height: int) -> tuple[Path, Path]:
    """Create Frames/ and GroundTruthMask/ for the synthetic clip."""
    frames_dir = root / "Frames"
    truth_dir = root / "GroundTruthMask"
    frames_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    annotation_indices = sorted({0, frames // 2, frames - 1})
    annotations: dict[str, list[dict[str, object]]] = {}

    for index in range(frames):
        frame, mask, pts = _make_fixture_frame(index, frames, width, height)
        frame_name = f"frame_{index:05d}.png"
        mask_name = f"frame_{index:05d}.png"
        cv2.imwrite(str(frames_dir / frame_name), frame)
        cv2.imwrite(str(truth_dir / mask_name), mask)

        if index in annotation_indices:
            annotations[str(index)] = [
                {
                    "points": [
                        [pts["head"][0], pts["head"][1] - 6],
                        [pts["head"][0], pts["head"][1]],
                        [pts["torso"][0], pts["torso"][1] - 18],
                        [pts["torso"][0], pts["torso"][1]],
                        [pts["hips"][0], pts["hips"][1]],
                    ],
                    "brush_type": "fg",
                    "radius": 18.0,
                },
                {
                    "points": [
                        [pts["left_shoulder"][0], pts["left_shoulder"][1]],
                        [pts["left_hand"][0], pts["left_hand"][1]],
                        [pts["left_foot"][0], pts["left_foot"][1]],
                    ],
                    "brush_type": "fg",
                    "radius": 16.0,
                },
                {
                    "points": [
                        [pts["right_shoulder"][0], pts["right_shoulder"][1]],
                        [pts["right_hand"][0], pts["right_hand"][1]],
                        [pts["right_foot"][0], pts["right_foot"][1]],
                    ],
                    "brush_type": "fg",
                    "radius": 16.0,
                },
                {
                    "points": [
                        [pts["left_bg"][0], pts["left_bg"][1] - 24],
                        [pts["left_bg"][0], pts["left_bg"][1]],
                        [pts["left_bg"][0], pts["left_bg"][1] + 24],
                    ],
                    "brush_type": "bg",
                    "radius": 20.0,
                },
                {
                    "points": [
                        [pts["right_bg"][0], pts["right_bg"][1] - 24],
                        [pts["right_bg"][0], pts["right_bg"][1]],
                        [pts["right_bg"][0], pts["right_bg"][1] + 24],
                        [pts["low_bg"][0], pts["low_bg"][1]],
                    ],
                    "brush_type": "bg",
                    "radius": 20.0,
                },
            ]

    with open(root / "annotations.json", "w", encoding="utf-8") as handle:
        json.dump(annotations, handle, indent=2)

    return frames_dir, truth_dir


def _sequence_iou(pred_dir: Path, truth_dir: Path, *, threshold: int = 127) -> list[float]:
    """Compute per-frame IoU between predicted and ground-truth grayscale masks."""
    scores: list[float] = []
    for truth_name in sorted(os.listdir(truth_dir)):
        truth = cv2.imread(str(truth_dir / truth_name), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(pred_dir / truth_name), cv2.IMREAD_GRAYSCALE)
        if truth is None or pred is None:
            scores.append(0.0)
            continue
        truth_bin = truth > threshold
        pred_bin = pred > threshold
        union = np.logical_or(truth_bin, pred_bin).sum()
        if union == 0:
            scores.append(1.0)
            continue
        inter = np.logical_and(truth_bin, pred_bin).sum()
        scores.append(float(inter / union))
    return scores


def _nonzero_fraction(sequence_dir: Path, *, threshold: int = 16) -> list[float]:
    """Return non-zero coverage fraction per frame for a grayscale sequence."""
    fractions: list[float] = []
    for name in sorted(os.listdir(sequence_dir)):
        img = cv2.imread(str(sequence_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            fractions.append(0.0)
            continue
        fractions.append(float((img > threshold).mean()))
    return fractions


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def run_local_gpu_qa(
    *,
    frames: int = 8,
    width: int = 320,
    height: int = 256,
    chunk_size: int = 8,
    sam2_model: str | None = None,
    min_sam2_iou: float = 0.75,
    min_videomama_iou: float | None = 0.90,
    keep_dir: str | None = None,
) -> dict[str, object]:
    """Run the local QA harness and return a JSON-serializable summary."""
    if keep_dir:
        root = Path(keep_dir).resolve()
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(tempfile.mkdtemp(prefix="corridorkey_gpu_qa_"))

    frames_dir, truth_dir = _write_fixture_clip(
        root,
        frames=frames,
        width=width,
        height=height,
    )

    service = CorridorKeyService()
    device = service.detect_device()
    if device != "cuda":
        raise RuntimeError(f"GPU QA requires CUDA. Detected device: {device}")
    if sam2_model is not None:
        service.set_sam2_model(SAM2_MODEL_IDS[sam2_model])

    clip = ClipEntry(
        name="gpu_qa_fixture",
        root_path=str(root),
        state=ClipState.RAW,
        input_asset=ClipAsset(str(frames_dir), "sequence"),
    )

    summary: dict[str, object] = {
        "root_dir": str(root),
        "device": device,
        "frames": frames,
        "width": width,
        "height": height,
        "chunk_size": chunk_size,
        "sam2_model_key": sam2_model or "base-plus",
        "sam2_model": service.sam2_model_id,
        "status_messages": [],
    }

    def _status(message: str) -> None:
        print(f"[qa] {message}", flush=True)
        cast = summary["status_messages"]
        assert isinstance(cast, list)
        cast.append(message)

    print(f"[qa] fixture: {root}", flush=True)
    print(f"[qa] device: {device}", flush=True)

    t0 = time.monotonic()
    summary["vram_before_sam2"] = service.get_vram_info()
    service.run_sam2_track(clip, on_status=_status)
    summary["sam2_seconds"] = round(time.monotonic() - t0, 3)
    summary["vram_after_sam2"] = service.get_vram_info()

    mask_dir = root / "VideoMamaMaskHint"
    if not mask_dir.is_dir():
        raise RuntimeError("SAM2 did not create VideoMamaMaskHint/")

    sam2_iou = _sequence_iou(mask_dir, truth_dir)
    sam2_fill = _nonzero_fraction(mask_dir)
    summary["sam2_frame_count"] = len(list(mask_dir.glob("*.png")))
    summary["sam2_iou"] = _summarize(sam2_iou)
    summary["sam2_fill_fraction"] = _summarize(sam2_fill)

    print(
        "[qa] SAM2: "
        f"{summary['sam2_frame_count']} masks, "
        f"mean IoU={summary['sam2_iou']['mean']:.3f}, "
        f"time={summary['sam2_seconds']:.1f}s",
        flush=True,
    )

    if summary["sam2_iou"]["mean"] < min_sam2_iou:
        raise RuntimeError(
            f"SAM2 mean IoU {summary['sam2_iou']['mean']:.3f} "
            f"fell below threshold {min_sam2_iou:.3f}"
        )

    t0 = time.monotonic()
    summary["vram_before_videomama"] = service.get_vram_info()
    service.run_videomama(clip, on_status=_status, chunk_size=chunk_size)
    summary["videomama_seconds"] = round(time.monotonic() - t0, 3)
    summary["vram_after_videomama"] = service.get_vram_info()

    alpha_dir = root / "AlphaHint"
    if not alpha_dir.is_dir():
        raise RuntimeError("VideoMaMa did not create AlphaHint/")

    videomama_iou = _sequence_iou(alpha_dir, truth_dir, threshold=96)
    videomama_fill = _nonzero_fraction(alpha_dir, threshold=16)
    summary["videomama_frame_count"] = len(list(alpha_dir.glob("*.png")))
    summary["videomama_iou"] = _summarize(videomama_iou)
    summary["videomama_fill_fraction"] = _summarize(videomama_fill)

    print(
        "[qa] VideoMaMa: "
        f"{summary['videomama_frame_count']} alpha frames, "
        f"mean IoU={summary['videomama_iou']['mean']:.3f}, "
        f"time={summary['videomama_seconds']:.1f}s",
        flush=True,
    )

    if summary["videomama_frame_count"] != frames:
        raise RuntimeError(
            f"VideoMaMa wrote {summary['videomama_frame_count']} frames, expected {frames}"
        )

    if min_videomama_iou is not None and summary["videomama_iou"]["mean"] < min_videomama_iou:
        raise RuntimeError(
            f"VideoMaMa mean IoU {summary['videomama_iou']['mean']:.3f} "
            f"fell below threshold {min_videomama_iou:.3f}"
        )

    summary["result"] = "pass"
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Local GPU QA harness for SAM2 + VideoMaMa")
    parser.add_argument(
        "--frames", type=int, default=8, help="Number of synthetic frames to generate"
    )
    parser.add_argument("--width", type=int, default=320, help="Fixture frame width")
    parser.add_argument("--height", type=int, default=256, help="Fixture frame height")
    parser.add_argument("--chunk-size", type=int, default=8, help="VideoMaMa chunk size")
    parser.add_argument(
        "--sam2-model",
        choices=sorted(SAM2_MODEL_IDS.keys()),
        default="base-plus",
        help="SAM2 checkpoint to test. Default: base-plus.",
    )
    parser.add_argument(
        "--keep-dir",
        type=str,
        default="",
        help="Directory to keep outputs instead of using a temp dir",
    )
    parser.add_argument(
        "--json-out", type=str, default="", help="Optional path to write the JSON summary"
    )
    parser.add_argument(
        "--min-sam2-iou", type=float, default=0.75, help="Fail if SAM2 mean IoU drops below this"
    )
    parser.add_argument(
        "--min-videomama-iou",
        type=float,
        default=None,
        help="Fail if VideoMaMa mean IoU drops below this. Default: 0.90.",
    )
    args = parser.parse_args()

    summary = run_local_gpu_qa(
        frames=args.frames,
        width=args.width,
        height=args.height,
        chunk_size=args.chunk_size,
        sam2_model=args.sam2_model,
        min_sam2_iou=args.min_sam2_iou,
        min_videomama_iou=args.min_videomama_iou,
        keep_dir=args.keep_dir or None,
    )

    print(json.dumps(summary, indent=2))
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

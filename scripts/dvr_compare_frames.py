"""Build a visual and numeric source-vs-processed QA compare.

This is intentionally a diagnostic script, not an alternate render path.
It reads already-rendered EZCK outputs and produces:

1. a side-by-side PNG
2. a JSON report with luminance ratios at several alpha thresholds

That gives us a repeatable way to answer "is the processed export globally
darker than the source?" without relying on screenshots from another machine.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np

from backend.frame_io import _linear_to_srgb


def _read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read source frame '{path}'")

    if image.ndim != 3:
        raise RuntimeError(f"Expected 3D image at '{path}', got shape {image.shape}")

    if image.dtype == np.uint8:
        image_f = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        image_f = image.astype(np.float32) / 65535.0
    else:
        image_f = image.astype(np.float32)

    channels = image.shape[2]
    if channels == 4:
        rgba = cv2.cvtColor(image_f, cv2.COLOR_BGRA2RGBA)
        return np.clip(rgba[:, :, :3], 0.0, 1.0)
    if channels == 3:
        return np.clip(cv2.cvtColor(image_f, cv2.COLOR_BGR2RGB), 0.0, 1.0)
    raise RuntimeError(f"Unsupported source channel count in '{path}': {channels}")


def _read_processed_rgba(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read processed frame '{path}'")
    if image.ndim != 3 or image.shape[2] != 4:
        raise RuntimeError(f"Expected 4-channel processed frame at '{path}', got shape {image.shape}")
    return np.clip(cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGRA2RGBA), 0.0, 1.0)


def _checkerboard(height: int, width: int, checker_size: int) -> np.ndarray:
    ys, xs = np.indices((height, width))
    checks = ((ys // checker_size + xs // checker_size) % 2).astype(np.float32)[..., np.newaxis]
    return 0.15 * (1.0 - checks) + 0.55 * checks


def _display_rgb(rgb: np.ndarray, *, source_is_linear: bool) -> np.ndarray:
    if source_is_linear:
        return np.clip(_linear_to_srgb(rgb), 0.0, 1.0)
    return np.clip(rgb, 0.0, 1.0)


def _luma(rgb: np.ndarray) -> np.ndarray:
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.sum(rgb * weights, axis=-1)


def _stats_for_thresholds(
    source_display: np.ndarray,
    processed_display: np.ndarray,
    alpha: np.ndarray,
) -> dict[str, object]:
    report: dict[str, object] = {}
    for threshold in (0.05, 0.5, 0.95):
        mask = alpha[:, :, 0] > threshold
        if not np.any(mask):
            continue
        src_luma = _luma(source_display[mask])
        proc_luma = _luma(processed_display[mask])
        report[f"alpha_gt_{threshold}"] = {
            "count": int(mask.sum()),
            "source_display_luma_mean": float(src_luma.mean()),
            "processed_display_luma_mean": float(proc_luma.mean()),
            "display_ratio_processed_over_source": float(
                proc_luma.mean() / max(src_luma.mean(), 1e-8)
            ),
        }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a source frame to an EZCK Processed EXR.")
    parser.add_argument("--source-frame", required=True, help="Path to the source frame image.")
    parser.add_argument("--processed-frame", required=True, help="Path to the Processed RGBA EXR.")
    parser.add_argument(
        "--source-is-linear",
        action="store_true",
        help="Apply the same linear-to-display transform to the source before comparison.",
    )
    parser.add_argument(
        "--checker-size",
        type=int,
        default=128,
        help="Checker size for processed compositing preview (default: 128).",
    )
    parser.add_argument("--image-out", required=True, help="Output side-by-side PNG path.")
    parser.add_argument("--json-out", required=True, help="Output JSON report path.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    source_rgb = _read_rgb(args.source_frame)
    processed_rgba = _read_processed_rgba(args.processed_frame)

    if source_rgb.shape[:2] != processed_rgba.shape[:2]:
        raise RuntimeError(
            f"Resolution mismatch: source {source_rgb.shape[:2]} vs processed {processed_rgba.shape[:2]}"
        )

    processed_rgb = processed_rgba[:, :, :3]
    alpha = processed_rgba[:, :, 3:4]

    source_display = _display_rgb(source_rgb, source_is_linear=bool(args.source_is_linear))
    processed_display = np.clip(_linear_to_srgb(processed_rgb), 0.0, 1.0)
    checker = _checkerboard(source_rgb.shape[0], source_rgb.shape[1], args.checker_size)
    processed_comp = np.clip(processed_display * alpha + checker * (1.0 - alpha), 0.0, 1.0)

    panel = np.concatenate(
        [
            cv2.cvtColor((source_display * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.cvtColor((processed_comp * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR),
        ],
        axis=1,
    )

    image_out = Path(args.image_out)
    json_out = Path(args.json_out)
    image_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(image_out), panel):
        raise RuntimeError(f"Failed to write compare image '{image_out}'")

    report = {
        "source_frame": os.path.abspath(args.source_frame),
        "processed_frame": os.path.abspath(args.processed_frame),
        "source_is_linear": bool(args.source_is_linear),
        "source_shape": list(source_rgb.shape),
        "processed_shape": list(processed_rgba.shape),
        "stats": _stats_for_thresholds(source_display, processed_display, alpha),
    }
    with open(json_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

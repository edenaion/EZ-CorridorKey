"""Repeatable EZCK -> Resolve roundtrip harness using the real app code path.

This script is deliberately narrow. It does not implement a custom keying path.
Instead it mirrors the current desktop app flow for a single clip:

1. Create or reuse a project clip.
2. Extract a source video to Frames/ using backend.ffmpeg_tools.extract_frames().
3. Import alpha hints using the same rename/copy rules as MainWindow._on_import_alpha().
4. Run CorridorKeyService.run_inference() with real InferenceParams/OutputConfig.
5. Emit a JSON report that records the exact inputs, params, and output paths.

That makes the resulting Processed/FG/Matte/Comp outputs appropriate for
Resolve-side QA without relying on a one-off image-processing script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2

from backend import ClipEntry, CorridorKeyService, InferenceParams, OutputConfig
from backend.ffmpeg_tools import (
    build_exr_vf,
    extract_frames,
    probe_video,
    require_ffmpeg_install,
    write_video_metadata,
)
from backend.project import create_project_from_media, get_clip_dirs, projects_root

logger = logging.getLogger("dvr_roundtrip_qa")


def _git_short_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    sha = (result.stdout or "").strip()
    return sha or None


def _natural_key(path: str) -> list[object]:
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _resolve_clip(project_dir: str) -> ClipEntry:
    clip_dirs = get_clip_dirs(project_dir)
    if len(clip_dirs) != 1:
        raise RuntimeError(
            f"DVR harness expects exactly one clip in project '{project_dir}', found {len(clip_dirs)}"
        )
    clip_root = clip_dirs[0]
    clip = ClipEntry(name=os.path.basename(clip_root), root_path=clip_root)
    clip.find_assets()
    return clip


def _extract_video_like_app(clip: ClipEntry, *, force_reextract: bool = False) -> dict[str, object]:
    if clip.input_asset is None:
        raise RuntimeError(f"Clip '{clip.name}' has no input asset to extract")
    if clip.input_asset.asset_type != "video":
        return {
            "skipped": True,
            "reason": "clip already points at a sequence",
            "frames_dir": clip.input_asset.path,
        }

    video_path = clip.input_asset.path
    clip_root = clip.root_path
    source_dir = os.path.join(clip_root, "Source")
    target_dir = os.path.join(clip_root, "Frames" if os.path.isdir(source_dir) else "Input")
    metadata_path = os.path.join(clip_root, ".video_metadata.json")

    if force_reextract and os.path.isdir(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)
    if force_reextract and os.path.isfile(metadata_path):
        os.remove(metadata_path)

    require_ffmpeg_install(require_probe=True)
    info = probe_video(video_path)
    total_frames = int(info.get("frame_count", 0) or 0)

    extracted = extract_frames(
        video_path=video_path,
        out_dir=target_dir,
        total_frames=total_frames,
    )

    metadata = {
        "source_path": video_path,
        "fps": info.get("fps", 24.0),
        "width": info.get("width", 0),
        "height": info.get("height", 0),
        "frame_count": extracted,
        "codec": info.get("codec", "unknown"),
        "duration": info.get("duration", 0),
        "exr_vf": build_exr_vf(info),
        "source_probe": {
            "frame_count": info.get("frame_count", 0),
            "pix_fmt": info.get("pix_fmt", ""),
            "color_space": info.get("color_space", ""),
            "color_primaries": info.get("color_primaries", ""),
            "color_transfer": info.get("color_transfer", ""),
            "color_range": info.get("color_range", ""),
            "chroma_location": info.get("chroma_location", ""),
            "bits_per_raw_sample": info.get("bits_per_raw_sample", 0),
        },
    }
    write_video_metadata(clip_root, metadata)

    clip.find_assets()
    return {
        "skipped": False,
        "video_path": video_path,
        "frames_dir": target_dir,
        "frame_count": extracted,
        "metadata_path": metadata_path,
    }


def _collect_alpha_source_files(alpha_source_dir: str) -> list[str]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.exr")
    files: list[str] = []
    for pattern in patterns:
        files.extend(str(path) for path in Path(alpha_source_dir).glob(pattern))
    files.sort(key=_natural_key)
    return files


def _import_alpha_like_app(
    clip: ClipEntry,
    alpha_source_dir: str,
    *,
    replace_existing: bool = True,
) -> dict[str, object]:
    if clip.input_asset is None:
        raise RuntimeError(f"Clip '{clip.name}' has no input asset for alpha import")
    if clip.input_asset.asset_type != "sequence":
        raise RuntimeError(
            "Alpha import mirror requires a sequence input. Extract the source video to Frames/ first."
        )

    src_files = _collect_alpha_source_files(alpha_source_dir)
    if not src_files:
        raise RuntimeError(f"No alpha images found in '{alpha_source_dir}'")

    input_files = clip.input_asset.get_frame_files()
    if not input_files:
        raise RuntimeError(f"Clip '{clip.name}' has no input frame files in '{clip.input_asset.path}'")

    alpha_dir = os.path.join(clip.root_path, "AlphaHint")
    if replace_existing and os.path.isdir(alpha_dir):
        shutil.rmtree(alpha_dir, ignore_errors=True)
    os.makedirs(alpha_dir, exist_ok=True)

    paired = min(len(src_files), len(input_files))
    converted = 0
    for index in range(paired):
        src_path = src_files[index]
        input_stem = os.path.splitext(input_files[index])[0]
        dst_path = os.path.join(alpha_dir, f"{input_stem}.png")
        src_ext = os.path.splitext(src_path)[1].lower()
        if src_ext == ".png":
            shutil.copy2(src_path, dst_path)
            continue

        image = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read alpha image '{src_path}'")
        if not cv2.imwrite(dst_path, image):
            raise RuntimeError(f"Failed to write converted alpha image '{dst_path}'")
        converted += 1

    clip.find_assets()
    return {
        "source_alpha_dir": alpha_source_dir,
        "alpha_dir": alpha_dir,
        "input_frame_count": len(input_files),
        "source_alpha_count": len(src_files),
        "paired_frame_count": paired,
        "converted_to_png": converted,
    }


def _make_params(args: argparse.Namespace) -> InferenceParams:
    return InferenceParams(
        input_is_linear=args.input_linear,
        despill_strength=args.despill,
        auto_despeckle=not args.no_despeckle,
        despeckle_size=args.despeckle_size,
        despeckle_dilation=25,
        despeckle_blur=5,
        refiner_scale=args.refiner,
    )


def _make_output_config(args: argparse.Namespace) -> OutputConfig:
    if args.processed_only:
        return OutputConfig(
            fg_enabled=False,
            matte_enabled=False,
            comp_enabled=False,
            processed_enabled=True,
            processed_format=args.processed_format,
            exr_compression=args.exr_compression,
        )
    return OutputConfig(
        fg_enabled=True,
        fg_format=args.fg_format,
        matte_enabled=True,
        matte_format=args.matte_format,
        comp_enabled=True,
        comp_format=args.comp_format,
        processed_enabled=True,
        processed_format=args.processed_format,
        exr_compression=args.exr_compression,
    )


def _write_report(report_path: str, data: dict[str, object]) -> None:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _run_inference_like_app(
    clip: ClipEntry,
    params: InferenceParams,
    output_config: OutputConfig,
    *,
    frame_range: tuple[int, int] | None = None,
) -> dict[str, object]:
    service = CorridorKeyService()
    timings: dict[str, object] = {}
    last_progress = 0
    total_frames = 0
    status_messages: list[str] = []
    warnings: list[str] = []

    t0 = time.monotonic()

    def on_progress(_clip_name: str, current: int, total: int, **kwargs: object) -> None:
        nonlocal last_progress, total_frames
        last_progress = current
        total_frames = total
        fps = kwargs.get("fps")
        if current == total or current == 1 or current % 10 == 0:
            if isinstance(fps, (int, float)):
                print(f"[progress] {current}/{total} ({float(fps):.2f} fps)")
            else:
                print(f"[progress] {current}/{total}")

    def on_warning(message: str) -> None:
        warnings.append(message)
        print(f"[warning] {message}")

    def on_status(message: str) -> None:
        if message:
            status_messages.append(message)
            print(f"[status] {message}")

    results = service.run_inference(
        clip=clip,
        params=params,
        on_progress=on_progress,
        on_warning=on_warning,
        on_status=on_status,
        output_config=output_config,
        frame_range=frame_range,
    )
    timings["elapsed_seconds"] = time.monotonic() - t0
    timings["progress_frames"] = last_progress
    timings["total_frames"] = total_frames
    timings["success_count"] = sum(1 for result in results if result.success)
    timings["warning_count"] = sum(1 for result in results if result.warning)
    timings["service_warning_messages"] = warnings
    timings["status_messages"] = status_messages
    return timings


def _default_report_path(clip: ClipEntry) -> str:
    return os.path.join(clip.root_path, "dvr_roundtrip_report.json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real EZCK app-path roundtrip harness for Resolve QA."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-video",
        help="Source video to ingest into a fresh project and extract like the app.",
    )
    source.add_argument(
        "--project-dir",
        help="Existing EZCK project root with exactly one clip to reuse.",
    )

    parser.add_argument(
        "--alpha-dir",
        help="Folder of user-supplied alpha hint images to import like the app.",
    )
    parser.add_argument(
        "--display-name",
        default="EZCK_DVR_TEST",
        help="Display name for a fresh project created from --source-video.",
    )
    parser.add_argument(
        "--copy-video",
        action="store_true",
        help="Copy the source video into the project Source/ folder for fresh projects.",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Delete Frames/ and .video_metadata.json and re-extract from Source/ video.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stop after extract/import-alpha and just write the report.",
    )

    parser.add_argument("--input-linear", action="store_true", help="Mirror Color Space = Linear.")
    parser.add_argument("--despill", type=float, default=1.0, help="Despill strength (default: 1.0).")
    parser.add_argument("--refiner", type=float, default=1.0, help="Refiner scale (default: 1.0).")
    parser.add_argument(
        "--no-despeckle",
        action="store_true",
        help="Disable despeckle, matching the UI toggle off state.",
    )
    parser.add_argument(
        "--despeckle-size",
        type=int,
        default=400,
        help="Despeckle size in pixels (default: 400).",
    )

    parser.add_argument(
        "--processed-only",
        action="store_true",
        help="Only write Processed output instead of the app's full default output set.",
    )
    parser.add_argument("--fg-format", default="exr", choices=("exr", "png"))
    parser.add_argument("--matte-format", default="exr", choices=("exr", "png"))
    parser.add_argument("--comp-format", default="png", choices=("png", "exr"))
    parser.add_argument("--processed-format", default="exr", choices=("exr", "png"))
    parser.add_argument(
        "--exr-compression",
        default="dwab",
        choices=("dwab", "piz", "zip", "none"),
        help="EXR compression to match app preferences (default: dwab).",
    )
    parser.add_argument(
        "--json-out",
        help="Optional explicit JSON report path. Defaults to <clip>/dvr_roundtrip_report.json.",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        help="Optional inclusive start frame index, mirroring app in/out markers.",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        help="Optional inclusive end frame index, mirroring app in/out markers.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    project_dir: str
    fresh_project = args.source_video is not None

    if args.source_video:
        project_dir = create_project_from_media(
            video_paths=[os.path.abspath(args.source_video)],
            copy_video=bool(args.copy_video),
            display_name=args.display_name,
        )
    else:
        project_dir = os.path.abspath(args.project_dir)

    clip = _resolve_clip(project_dir)
    extraction: dict[str, object] | None = None
    alpha_import: dict[str, object] | None = None

    if args.force_reextract or (clip.input_asset is not None and clip.input_asset.asset_type == "video"):
        extraction = _extract_video_like_app(clip, force_reextract=bool(args.force_reextract))

    if args.alpha_dir:
        alpha_import = _import_alpha_like_app(
            clip,
            os.path.abspath(args.alpha_dir),
            replace_existing=True,
        )

    params = _make_params(args)
    output_config = _make_output_config(args)
    frame_range: tuple[int, int] | None = None
    if args.frame_start is not None or args.frame_end is not None:
        if args.frame_start is None or args.frame_end is None:
            raise RuntimeError("--frame-start and --frame-end must be provided together")
        if args.frame_start < 0 or args.frame_end < args.frame_start:
            raise RuntimeError("Invalid frame range")
        frame_range = (args.frame_start, args.frame_end)

    report: dict[str, object] = {
        "harness": "dvr_roundtrip_qa",
        "git_sha": _git_short_sha(),
        "python": sys.version,
        "projects_root": projects_root(),
        "fresh_project": fresh_project,
        "project_dir": project_dir,
        "clip_root": clip.root_path,
        "clip_name": clip.name,
        "input_asset_type": clip.input_asset.asset_type if clip.input_asset else None,
        "alpha_asset_type": clip.alpha_asset.asset_type if clip.alpha_asset else None,
        "params": params.to_dict(),
        "output_config": output_config.to_dict(),
        "frame_range": frame_range,
        "equivalent_app_path": [
            "backend.project.create_project_from_media",
            "backend.ffmpeg_tools.extract_frames",
            "ui.main_window._on_import_alpha rename/copy semantics",
            "backend.service.CorridorKeyService.run_inference",
            "backend.service.CorridorKeyService._write_outputs",
        ],
    }
    if extraction is not None:
        report["extraction"] = extraction
    if alpha_import is not None:
        report["alpha_import"] = alpha_import

    if not args.prepare_only:
        inference = _run_inference_like_app(
            clip,
            params,
            output_config,
            frame_range=frame_range,
        )
        report["inference"] = inference
        report["output_dirs"] = {
            "fg": os.path.join(clip.root_path, "Output", "FG"),
            "matte": os.path.join(clip.root_path, "Output", "Matte"),
            "comp": os.path.join(clip.root_path, "Output", "Comp"),
            "processed": os.path.join(clip.root_path, "Output", "Processed"),
        }

    report_path = os.path.abspath(args.json_out) if args.json_out else _default_report_path(clip)
    _write_report(report_path, report)
    print(f"[done] report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

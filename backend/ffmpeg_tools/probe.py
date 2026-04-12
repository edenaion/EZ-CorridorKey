"""Video probing via ffprobe."""
from __future__ import annotations

import json
import logging
import os
import subprocess

from .discovery import require_ffmpeg_install

logger = logging.getLogger(__name__)


def probe_video(path: str) -> dict:
    """Probe a video file for metadata.

    Returns dict with keys: fps (float), width (int), height (int),
    frame_count (int), codec (str), duration (float), pix_fmt (str),
    color_range (str), color_space (str), color_primaries (str),
    color_transfer (str), chroma_location (str), bits_per_raw_sample (int).
    Raises RuntimeError if ffprobe fails.
    """
    validation = require_ffmpeg_install(require_probe=True)
    ffprobe = validation.ffprobe_path
    if not ffprobe:
        raise RuntimeError("FFprobe not found")

    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        path,
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30,
        encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:500]}")

    data = json.loads(result.stdout)

    # Find first video stream
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        raise RuntimeError(f"No video stream found in {path}")

    # Parse fps from r_frame_rate (e.g. "24000/1001")
    fps_str = video_stream.get("r_frame_rate", "24/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 24.0
    else:
        fps = float(fps_str)

    # Frame count: prefer nb_frames, fall back to duration * fps
    frame_count = 0
    if "nb_frames" in video_stream:
        try:
            frame_count = int(video_stream["nb_frames"])
        except (ValueError, TypeError):
            pass

    if frame_count <= 0:
        duration = float(video_stream.get("duration", 0) or
                         data.get("format", {}).get("duration", 0))
        if duration > 0:
            frame_count = int(duration * fps)

    return {
        "fps": round(fps, 4),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "frame_count": frame_count,
        "codec": video_stream.get("codec_name", "unknown"),
        "duration": float(video_stream.get("duration", 0) or
                          data.get("format", {}).get("duration", 0)),
        # Color metadata for building explicit conversion filters
        "pix_fmt": video_stream.get("pix_fmt", ""),
        "color_space": video_stream.get("color_space", ""),
        "color_primaries": video_stream.get("color_primaries", ""),
        "color_transfer": video_stream.get("color_transfer", ""),
        "color_range": video_stream.get("color_range", ""),
        "chroma_location": video_stream.get("chroma_location", ""),
        "bits_per_raw_sample": int(video_stream.get("bits_per_raw_sample", 0) or 0),
    }

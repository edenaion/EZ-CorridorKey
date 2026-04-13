"""FFmpeg subprocess wrapper for video extraction and stitching.

Pure Python, no Qt deps. Provides:
- find_ffmpeg() / find_ffprobe() — locate binaries
- detect_hwaccel() — auto-detect best hardware decoder per platform
- probe_video() — get fps, resolution, frame count, codec
- extract_frames() — video -> EXR DWAB half-float image sequence
- stitch_video() — image sequence -> video (H.264)
- write/read_video_metadata() — sidecar JSON for roundtrip fidelity
"""

from __future__ import annotations

# Re-export stdlib modules that tests monkeypatch on the module object
import shutil
import subprocess
import sys

# Re-export all public symbols from submodules
from .color import build_exr_vf
from .discovery import (
    FFmpegValidationResult,
    FFmpegVersionInfo,
    _local_ffmpeg_binary,
    find_ffmpeg,
    find_ffprobe,
    get_ffmpeg_install_help,
    repair_ffmpeg_install,
    require_ffmpeg_install,
    validate_ffmpeg_install,
)
from .extraction import (
    _recompress_to_dwab,
    detect_hwaccel,
    extract_frames,
)
from .metadata import (
    read_video_metadata,
    write_video_metadata,
)
from .probe import probe_video
from .stitching import stitch_video

__all__ = [
    # stdlib re-exports (tests monkeypatch these on the package)
    "shutil",
    "subprocess",
    "sys",
    # discovery
    "FFmpegValidationResult",
    "FFmpegVersionInfo",
    "_local_ffmpeg_binary",
    "find_ffmpeg",
    "find_ffprobe",
    "get_ffmpeg_install_help",
    "repair_ffmpeg_install",
    "require_ffmpeg_install",
    "validate_ffmpeg_install",
    # probe
    "probe_video",
    # color
    "build_exr_vf",
    # extraction
    "detect_hwaccel",
    "_recompress_to_dwab",
    "extract_frames",
    # stitching
    "stitch_video",
    # metadata
    "write_video_metadata",
    "read_video_metadata",
]

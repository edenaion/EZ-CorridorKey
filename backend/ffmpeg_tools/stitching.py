"""Video stitching from image sequences via FFmpeg."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from typing import Callable, Optional

from .discovery import require_ffmpeg_install

logger = logging.getLogger(__name__)


def stitch_video(
    in_dir: str,
    out_path: str,
    fps: float = 24.0,
    pattern: str = "frame_%06d.png",
    codec: str = "libx264",
    crf: int = 18,
    start_number: int = 0,
    on_progress: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Stitch image sequence back into a video file.

    Args:
        in_dir: Directory containing frame images.
        out_path: Output video file path (.mp4 or .webm).
        fps: Frame rate.
        pattern: Frame filename pattern.
        codec: Video codec (libx264, libx265, libvpx-vp9, etc.).
        crf: Quality (0-51 for x264, 0-63 for VP9; lower = better).
        start_number: First frame number in the sequence.
        on_progress: Callback(current_frame, total_frames).
        cancel_event: Set to cancel stitching.

    Raises:
        RuntimeError if ffmpeg is not found or stitching fails.
    """
    validation = require_ffmpeg_install(require_probe=True)
    ffmpeg = validation.ffmpeg_path
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found")

    # Count total frames
    total_frames = len([f for f in os.listdir(in_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.exr'))])

    # Auto-detect codec/pixel format from output extension
    ext = os.path.splitext(out_path)[1].lower()
    is_webm = ext == '.webm'
    is_mov = ext == '.mov'
    is_exr = pattern.lower().endswith('.exr')

    if is_mov:
        codec = "prores_ks"
        pix_fmt = "yuva444p10le"
    elif is_webm:
        codec = "libvpx-vp9"
        pix_fmt = "yuva420p"
    else:
        pix_fmt = "yuv420p"

    # EXR frames are linear float — tell FFmpeg to apply sRGB gamma
    cmd = [ffmpeg, "-framerate", str(fps), "-start_number", str(start_number)]
    if is_exr:
        cmd.extend(["-apply_trc", "iec61966_2_1"])
    cmd.extend(["-i", in_dir + "/" + pattern, "-c:v", codec])
    if is_mov:
        cmd.extend(["-profile:v", "4444", "-qscale:v", "9", "-vendor", "apl0"])
    else:
        cmd.extend(["-crf", str(crf)])
    cmd.extend(["-pix_fmt", pix_fmt])
    if is_webm:
        cmd.extend(["-b:v", "0", "-auto-alt-ref", "0", "-deadline", "good", "-row-mt", "1"])
    cmd.extend([out_path, "-y"])

    logger.info(f"Stitching video: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    frame_re = re.compile(r"frame=\s*(\d+)")
    stderr_lines: list[str] = []

    try:
        for line in proc.stderr:
            stderr_lines.append(line.rstrip())

            if cancel_event and cancel_event.is_set():
                try:
                    proc.stdin.write("q\n")
                    proc.stdin.flush()
                except Exception:
                    pass
                proc.wait(timeout=5)
                logger.info("Stitching cancelled")
                return

            match = frame_re.search(line)
            if match:
                current = int(match.group(1))
                if on_progress and total_frames > 0:
                    on_progress(current, total_frames)

        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("FFmpeg stitching timed out")

    if proc.returncode != 0 and not (cancel_event and cancel_event.is_set()):
        full = "\n".join(stderr_lines) if stderr_lines else "(no output)"
        logger.error(f"FFmpeg stitching failed (exit {proc.returncode}):\n{full}")
        raise RuntimeError(f"FFmpeg stitching failed:\n{full}")

    logger.info(f"Video stitched: {out_path}")

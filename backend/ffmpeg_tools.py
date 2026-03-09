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

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_METADATA_FILENAME = ".video_metadata.json"

# Common install locations per platform
_FFMPEG_SEARCH_PATHS_WINDOWS = [
    r"C:\Program Files\ffmpeg\bin",
    r"C:\Program Files (x86)\ffmpeg\bin",
    r"C:\ffmpeg\bin",
]

_FFMPEG_SEARCH_PATHS_UNIX = [
    "/opt/homebrew/bin",        # macOS Homebrew (Apple Silicon)
    "/usr/local/bin",           # macOS Homebrew (Intel) / Linux manual install
    "/usr/bin",                 # Linux system package
    "/snap/bin",                # Linux snap
    os.path.expanduser("~/bin"),
]

_FFMPEG_SEARCH_PATHS = (
    _FFMPEG_SEARCH_PATHS_WINDOWS if sys.platform == "win32"
    else _FFMPEG_SEARCH_PATHS_UNIX
)


def find_ffmpeg() -> str | None:
    """Locate ffmpeg binary. Checks PATH then common install dirs."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    ext = ".exe" if sys.platform == "win32" else ""
    for d in _FFMPEG_SEARCH_PATHS:
        candidate = os.path.join(d, f"ffmpeg{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None


def find_ffprobe() -> str | None:
    """Locate ffprobe binary. Checks PATH then common install dirs."""
    found = shutil.which("ffprobe")
    if found:
        return found
    ext = ".exe" if sys.platform == "win32" else ""
    for d in _FFMPEG_SEARCH_PATHS:
        candidate = os.path.join(d, f"ffprobe{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Hardware-accelerated decode — cross-platform auto-detection
# ---------------------------------------------------------------------------

# Priority order per platform. First available wins.
# Each entry: (hwaccel_name, pre-input flags for FFmpeg)
_HWACCEL_PRIORITY: dict[str, list[tuple[str, list[str]]]] = {
    "win32": [
        ("cuda",    ["-hwaccel", "cuda"]),
        ("d3d11va", ["-hwaccel", "d3d11va"]),
        ("dxva2",   ["-hwaccel", "dxva2"]),
    ],
    "linux": [
        ("cuda",  ["-hwaccel", "cuda"]),
        ("vaapi", ["-hwaccel", "vaapi"]),
    ],
    "darwin": [
        ("videotoolbox", ["-hwaccel", "videotoolbox"]),
    ],
}

_cached_hwaccel: list[str] | None = None  # cached result of detect_hwaccel()


def detect_hwaccel(ffmpeg: str | None = None) -> list[str]:
    """Detect the best FFmpeg hardware accelerator for this platform.

    Probes ``ffmpeg -hwaccels`` once, caches the result, and returns
    the pre-input flags to inject before ``-i``.  Returns an empty list
    (software fallback) if no hardware decoder is available.
    """
    global _cached_hwaccel
    if _cached_hwaccel is not None:
        return list(_cached_hwaccel)

    if ffmpeg is None:
        ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        _cached_hwaccel = []
        return []

    # Query available methods
    try:
        result = subprocess.run(
            [ffmpeg, "-hwaccels"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        available = set(result.stdout.lower().split())
    except Exception:
        _cached_hwaccel = []
        return []

    # Match platform to best available
    platform_key = sys.platform  # win32, linux, darwin
    candidates = _HWACCEL_PRIORITY.get(platform_key, [])

    for name, flags in candidates:
        if name in available:
            logger.info(f"FFmpeg hardware decode: using {name}")
            _cached_hwaccel = flags
            return list(flags)

    logger.info("FFmpeg hardware decode: none available, using software decode")
    _cached_hwaccel = []
    return []


def probe_video(path: str) -> dict:
    """Probe a video file for metadata.

    Returns dict with keys: fps (float), width (int), height (int),
    frame_count (int), codec (str), duration (float).
    Raises RuntimeError if ffprobe fails.
    """
    ffprobe = find_ffprobe()
    if not ffprobe:
        raise RuntimeError("ffprobe not found")

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
    }


def _recompress_to_dwab(
    out_dir: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Recompress FFmpeg ZIP16 EXR files to DWAB in-place.

    Launches a standalone subprocess to do the heavy lifting so the
    parent process (and its GIL / Qt event loop) stay completely free.
    The subprocess uses multiprocessing internally and prints progress
    lines to stdout which we parse for the callback.
    """
    marker = os.path.join(out_dir, ".dwab_done")
    if os.path.isfile(marker):
        return

    exr_files = sorted([f for f in os.listdir(out_dir)
                        if f.lower().endswith('.exr')])
    total = len(exr_files)
    if total == 0:
        return

    # Locate the Python interpreter from the same venv
    python = sys.executable

    # Write a temp script file.  ProcessPoolExecutor on Windows uses the
    # "spawn" start method which re-imports __main__ — this only works
    # from a real .py file, not from ``python -c``.
    import tempfile
    script_content = r'''
import os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def recompress_one(args):
    src, tmp = args
    try:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2, numpy as np, OpenEXR, Imath
        img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False
        HALF = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        h, w = img.shape[:2]
        hdr = OpenEXR.Header(w, h)
        hdr['compression'] = Imath.Compression(Imath.Compression.DWAB_COMPRESSION)
        if img.ndim == 2:
            hdr['channels'] = {'Y': HALF}
            out = OpenEXR.OutputFile(tmp, hdr)
            out.writePixels({'Y': img.astype(np.float16).tobytes()})
            out.close()
        elif img.ndim == 3 and img.shape[2] == 3:
            hdr['channels'] = {'R': HALF, 'G': HALF, 'B': HALF}
            out = OpenEXR.OutputFile(tmp, hdr)
            out.writePixels({
                'R': img[:,:,2].astype(np.float16).tobytes(),
                'G': img[:,:,1].astype(np.float16).tobytes(),
                'B': img[:,:,0].astype(np.float16).tobytes(),
            })
            out.close()
        elif img.ndim == 3 and img.shape[2] == 4:
            hdr['channels'] = {'R': HALF, 'G': HALF, 'B': HALF, 'A': HALF}
            out = OpenEXR.OutputFile(tmp, hdr)
            out.writePixels({
                'R': img[:,:,2].astype(np.float16).tobytes(),
                'G': img[:,:,1].astype(np.float16).tobytes(),
                'B': img[:,:,0].astype(np.float16).tobytes(),
                'A': img[:,:,3].astype(np.float16).tobytes(),
            })
            out.close()
        else:
            return False
        os.replace(tmp, src)
        return True
    except Exception:
        if os.path.isfile(tmp):
            os.remove(tmp)
        return False

if __name__ == "__main__":
    out_dir = sys.argv[1]
    files = sorted(f for f in os.listdir(out_dir) if f.lower().endswith('.exr'))
    total = len(files)
    if total == 0:
        sys.exit(0)
    workers = max(1, min((os.cpu_count() or 4) // 2, 16))
    work = [(os.path.join(out_dir, f), os.path.join(out_dir, f + ".tmp"))
            for f in files]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(recompress_one, item): item for item in work}
        for fut in as_completed(futs):
            fut.result()
            done += 1
            print(f"PROGRESS {done} {total}", flush=True)
    print("DONE", flush=True)
'''

    # Write to a temp .py file next to the output dir (same drive avoids
    # cross-device issues).  Cleaned up after completion.
    script_path = os.path.join(out_dir, "_dwab_recompress.py")
    with open(script_path, 'w') as f:
        f.write(script_content)

    logger.info(f"Recompressing {total} EXR frames to DWAB (subprocess)...")

    proc = subprocess.Popen(
        [python, script_path, out_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    # Read stdout in a background thread so cancel checks aren't blocked
    import queue as _queue
    line_q: _queue.Queue[str | None] = _queue.Queue()

    def _reader():
        for ln in proc.stdout:
            line_q.put(ln.strip())
        line_q.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    try:
        while True:
            if cancel_event and cancel_event.is_set():
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                logger.info("DWAB recompression cancelled")
                return

            try:
                line = line_q.get(timeout=0.2)
            except _queue.Empty:
                if proc.poll() is not None:
                    break
                continue

            if line is None:
                break

            if line.startswith("PROGRESS "):
                parts = line.split()
                if len(parts) == 3:
                    done_n, total_n = int(parts[1]), int(parts[2])
                    if on_progress:
                        on_progress(done_n, total_n)
            elif line == "DONE":
                break

        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        logger.error("DWAB recompression subprocess timed out")
        return
    finally:
        # Always clean up temp script
        try:
            os.remove(script_path)
        except OSError:
            pass

    if proc.returncode != 0:
        stderr_out = proc.stderr.read() if proc.stderr else ""
        logger.error(f"DWAB recompression failed (code {proc.returncode}): "
                     f"{stderr_out[:500]}")
        return

    # Mark completion so resume doesn't redo
    with open(marker, 'w') as f:
        f.write("done")
    logger.info(f"DWAB recompression complete: {total} frames")


# ---------------------------------------------------------------------------
#  Probe-driven colour-space filter chain
# ---------------------------------------------------------------------------

# Values ffprobe reports that mean "unknown / missing"
_UNKNOWN_COLOR = {"", "unknown", "unspecified", "reserved", None}


def _is_rgb_pix_fmt(pix_fmt: str) -> bool:
    """Return True if the pixel format is already RGB-family."""
    if not pix_fmt:
        return False
    pf = pix_fmt.lower()
    return (pf.startswith("rgb") or pf.startswith("bgr") or
            pf.startswith("gbr") or pf.startswith("argb") or
            pf.startswith("abgr") or pf == "pal8")


def _build_exr_vf(video_info: dict) -> str:
    """Build the -vf string for converting to gbrpf32le (EXR output).

    For RGB inputs, just does format conversion.
    For YUV inputs, uses setparams to fill any missing color metadata
    on the AVFrame so swscaler has everything it needs for the
    YUV→RGB conversion. Works with all FFmpeg builds.
    """
    pix_fmt = video_info.get("pix_fmt", "")

    if _is_rgb_pix_fmt(pix_fmt):
        return "format=gbrpf32le"

    # YUV input — swscaler needs trc, primaries, matrix, and range
    # to be non-null on the frame. setparams fills any gaps.
    cs = video_info.get("color_space", "")
    cp = video_info.get("color_primaries", "")
    ct = video_info.get("color_transfer", "")
    cr = video_info.get("color_range", "")
    w = video_info.get("width", 0)
    h = video_info.get("height", 0)

    # Fill missing with heuristics based on resolution
    hd = w >= 1280 or h > 576
    if cs in _UNKNOWN_COLOR:
        cs = "bt709" if hd else "smpte170m"
    if cp in _UNKNOWN_COLOR:
        cp = "bt709" if hd else "smpte170m"
    if ct in _UNKNOWN_COLOR:
        ct = "bt709"  # safe default for SDR content
    if cr in _UNKNOWN_COLOR:
        cr = "tv"  # limited range is standard for video

    # Map color_range to setparams range value
    sp_range = "limited" if cr == "tv" else ("full" if cr == "pc" else cr)

    logger.info(f"Color metadata: pix_fmt={pix_fmt} space={cs} "
                f"primaries={cp} transfer={ct} range={cr}")

    # setparams stamps complete metadata onto every frame, then
    # format=gbrpf32le triggers swscaler which now sees all fields.
    return (f"setparams=colorspace={cs}:color_primaries={cp}"
            f":color_trc={ct}:color_range={sp_range},"
            f"format=gbrpf32le")


def extract_frames(
    video_path: str,
    out_dir: str,
    pattern: str = "frame_%06d.exr",
    on_progress: Optional[Callable[[int, int], None]] = None,
    on_recompress_progress: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    total_frames: int = 0,
) -> int:
    """Extract video frames to EXR DWAB half-float image sequence.

    Two-pass extraction:
    1. FFmpeg extracts to EXR ZIP16 half-float (genuine float precision)
    2. OpenCV recompresses each frame to DWAB (VFX-standard compression)

    Args:
        video_path: Path to input video file.
        out_dir: Directory to write frames into (created if needed).
        pattern: Frame filename pattern (FFmpeg style).
        on_progress: Callback(current_frame, total_frames) for extraction.
        on_recompress_progress: Callback(current, total) for DWAB pass.
        cancel_event: Set to cancel extraction.
        total_frames: Expected total (for progress). Probed if 0.

    Returns:
        Number of frames extracted.

    Raises:
        RuntimeError if ffmpeg is not found or extraction fails.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    os.makedirs(out_dir, exist_ok=True)

    # Always probe — we need color metadata for the filter chain,
    # and total_frames for progress
    video_info = None
    try:
        video_info = probe_video(video_path)
        if total_frames <= 0:
            total_frames = video_info.get("frame_count", 0)
    except Exception:
        if total_frames <= 0:
            total_frames = 0

    # Resume: detect existing frames and skip ahead with conservative rollback.
    # Delete the last few frames (may be corrupt from mid-write or FFmpeg
    # output buffering) and re-extract from that point.
    _RESUME_ROLLBACK = 3  # frames to re-extract for safety
    start_frame = 0

    # Check for completed DWAB recompression marker — if present, extraction
    # is fully done, just count frames.
    dwab_marker = os.path.join(out_dir, ".dwab_done")
    if os.path.isfile(dwab_marker):
        extracted = len([f for f in os.listdir(out_dir)
                         if f.lower().endswith('.exr')])
        logger.info(f"Extraction already complete: {extracted} DWAB frames")
        return extracted

    existing = sorted([f for f in os.listdir(out_dir)
                       if f.lower().endswith('.exr')])
    if existing:
        # Remove the last N frames — they may be corrupt or incomplete
        remove_count = min(_RESUME_ROLLBACK, len(existing))
        for fname in existing[-remove_count:]:
            os.remove(os.path.join(out_dir, fname))
        start_frame = max(0, len(existing) - remove_count)
        if start_frame > 0:
            logger.info(f"Resuming extraction from frame {start_frame} "
                        f"({len(existing)} existed, rolled back {remove_count})")

    # EXR-specific FFmpeg args: ZIP16 compression, half-float.
    # Build an explicit colour conversion filter from probed metadata
    # so FFmpeg never has to guess missing trc/primaries/matrix.
    vf_chain = _build_exr_vf(video_info or {})
    exr_args = ["-compression", "3", "-format", "1", "-vf", vf_chain]

    # Hardware-accelerated decode (NVDEC / VideoToolbox / VAAPI / D3D11VA)
    # Falls back to software decode if none available
    hwaccel_flags = detect_hwaccel(ffmpeg)

    def _build_cmd(hw_flags: list[str]) -> list[str]:
        if start_frame > 0 and total_frames > 0:
            if video_info is None:
                _vi = probe_video(video_path)
            else:
                _vi = video_info
            fps = _vi.get("fps", 24.0)
            seek_sec = start_frame / fps
            return [
                ffmpeg,
                *hw_flags,
                "-ss", f"{seek_sec:.4f}",
                "-i", video_path,
                "-start_number", str(start_frame),
                "-vsync", "passthrough",
                *exr_args,
                out_dir + "/" + pattern,
                "-y",
            ]
        return [
            ffmpeg,
            *hw_flags,
            "-i", video_path,
            "-start_number", "0",
            "-vsync", "passthrough",
            *exr_args,
            out_dir + "/" + pattern,
            "-y",
        ]

    def _run_ffmpeg(hw_flags: list[str]) -> tuple[int, str]:
        """Run FFmpeg extraction. Returns (return_code, last_stderr_lines)."""
        nonlocal last_frame

        cmd = _build_cmd(hw_flags)
        hwaccel_label = hw_flags[1] if hw_flags else "software"
        logger.info(f"Extracting frames (EXR half-float, decode={hwaccel_label}): "
                    f"{video_path} -> {out_dir} (start_frame={start_frame})")

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        frame_re = re.compile(r"frame=\s*(\d+)")
        stderr_tail: list[str] = []  # keep last N lines for error reporting

        import queue as _queue
        line_q: _queue.Queue[str | None] = _queue.Queue()

        def _reader():
            for ln in proc.stderr:
                line_q.put(ln)
            line_q.put(None)

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    logger.info("Extraction cancelled — FFmpeg killed")
                    return 0, ""

                try:
                    line = line_q.get(timeout=0.2)
                except _queue.Empty:
                    if proc.poll() is not None:
                        break
                    continue

                if line is None:
                    break

                stderr_tail.append(line.rstrip())
                if len(stderr_tail) > 30:
                    stderr_tail.pop(0)

                match = frame_re.search(line)
                if match:
                    last_frame = start_frame + int(match.group(1))
                    if on_progress and total_frames > 0:
                        on_progress(last_frame, total_frames)

            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError("FFmpeg extraction timed out")

        if proc.returncode != 0:
            tail = "\n".join(stderr_tail[-15:])
            logger.error(f"FFmpeg failed (code {proc.returncode}):\n{tail}")

        return proc.returncode, "\n".join(stderr_tail[-15:])

    last_frame = start_frame
    returncode, stderr_out = _run_ffmpeg(hwaccel_flags)

    # If hardware decode failed, retry with software decode
    if returncode != 0 and hwaccel_flags and not (cancel_event and cancel_event.is_set()):
        logger.warning(f"Hardware decode failed (code {returncode}), "
                       f"retrying with software decode...")
        # Clean up any partial frames from the failed attempt
        for f in os.listdir(out_dir):
            if f.lower().endswith('.exr'):
                os.remove(os.path.join(out_dir, f))
        last_frame = start_frame
        returncode, stderr_out = _run_ffmpeg([])  # empty = software decode

    if returncode != 0 and not (cancel_event and cancel_event.is_set()):
        # Extract a meaningful error message from FFmpeg stderr
        err_detail = ""
        if stderr_out:
            for line in stderr_out.splitlines():
                low = line.lower()
                if any(kw in low for kw in ("error", "invalid", "no such",
                                             "not found", "unknown",
                                             "unrecognized", "failed")):
                    err_detail = line.strip()
                    break
        msg = f"FFmpeg extraction failed (code {returncode})"
        if err_detail:
            msg += f": {err_detail}"
        raise RuntimeError(msg)

    # Count extracted frames
    extracted = len([f for f in os.listdir(out_dir)
                     if f.lower().endswith('.exr')])
    logger.info(f"Extracted {extracted} EXR frames (ZIP16)")

    # Pass 2: Recompress ZIP16 → DWAB
    if extracted > 0 and not (cancel_event and cancel_event.is_set()):
        _recompress_to_dwab(out_dir, on_recompress_progress, cancel_event)

    return extracted


def stitch_video(
    in_dir: str,
    out_path: str,
    fps: float = 24.0,
    pattern: str = "frame_%06d.png",
    codec: str = "libx264",
    crf: int = 18,
    on_progress: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Stitch image sequence back into a video file.

    Args:
        in_dir: Directory containing frame images.
        out_path: Output video file path.
        fps: Frame rate.
        pattern: Frame filename pattern.
        codec: Video codec (libx264, libx265, etc.).
        crf: Quality (0-51, lower = better).
        on_progress: Callback(current_frame, total_frames).
        cancel_event: Set to cancel stitching.

    Raises:
        RuntimeError if ffmpeg is not found or stitching fails.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    # Count total frames
    total_frames = len([f for f in os.listdir(in_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.exr'))])

    cmd = [
        ffmpeg,
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", in_dir + "/" + pattern,
        "-c:v", codec,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        out_path,
        "-y",
    ]

    logger.info(f"Stitching video: {in_dir} -> {out_path}")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    frame_re = re.compile(r"frame=\s*(\d+)")

    try:
        for line in proc.stderr:
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
        raise RuntimeError(f"FFmpeg stitching failed with code {proc.returncode}")

    logger.info(f"Video stitched: {out_path}")


def write_video_metadata(clip_root: str, metadata: dict) -> None:
    """Write video metadata sidecar JSON to clip root.

    Metadata typically includes: source_path, fps, width, height,
    frame_count, codec, duration.
    """
    path = os.path.join(clip_root, _METADATA_FILENAME)
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.debug(f"Video metadata written: {path}")


def read_video_metadata(clip_root: str) -> dict | None:
    """Read video metadata sidecar from clip root. Returns None if not found."""
    path = os.path.join(clip_root, _METADATA_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"Failed to read video metadata: {e}")
        return None

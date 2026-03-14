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
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

_METADATA_FILENAME = ".video_metadata.json"
_MIN_FFMPEG_MAJOR = 7
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_FFMPEG_BIN = os.path.join(_REPO_ROOT, "tools", "ffmpeg", "bin")
_WINDOWS_FFMPEG_BUNDLE_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/"
    "ffmpeg-master-latest-win64-gpl.zip"
)

# Common install locations per platform
_FFMPEG_SEARCH_PATHS_WINDOWS = [
    _LOCAL_FFMPEG_BIN,
    r"C:\Program Files\ffmpeg\bin",
    r"C:\Program Files (x86)\ffmpeg\bin",
    r"C:\ffmpeg\bin",
]

_FFMPEG_SEARCH_PATHS_UNIX = [
    _LOCAL_FFMPEG_BIN,
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
_FFMPEG_RELEASE_RE = re.compile(
    r"\b(?:ffmpeg|ffprobe)(?:\.exe)?\s+version\s+(?:n)?(?P<major>\d+)(?:\.\d+)*",
    re.IGNORECASE,
)
_FFMPEG_DEV_BUILD_RE = re.compile(
    r"\b(?:ffmpeg|ffprobe)\s+version\s+(?:n-|git-|master-)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FFmpegVersionInfo:
    """Parsed `ffmpeg -version` or `ffprobe -version` first-line summary."""

    first_line: str
    major: int | None
    is_dev_build: bool = False


@dataclass(frozen=True)
class FFmpegValidationResult:
    """Validation result for the current FFmpeg installation."""

    ok: bool
    message: str
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None
    ffmpeg_version: FFmpegVersionInfo | None = None
    ffprobe_version: FFmpegVersionInfo | None = None


def _local_ffmpeg_binary(name: str) -> str | None:
    """Return the bundled repo-local FFmpeg binary if present."""
    ext = ".exe" if sys.platform == "win32" else ""
    candidate = os.path.join(_LOCAL_FFMPEG_BIN, f"{name}{ext}")
    return candidate if os.path.isfile(candidate) else None


def find_ffmpeg() -> str | None:
    """Locate ffmpeg binary. Prefer the bundled local build when present."""
    local = _local_ffmpeg_binary("ffmpeg")
    if local:
        return local
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
    """Locate ffprobe binary. Prefer the bundled local build when present."""
    local = _local_ffmpeg_binary("ffprobe")
    if local:
        return local
    found = shutil.which("ffprobe")
    if found:
        return found
    ext = ".exe" if sys.platform == "win32" else ""
    for d in _FFMPEG_SEARCH_PATHS:
        candidate = os.path.join(d, f"ffprobe{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None


def _read_program_version(binary_path: str, program_name: str) -> FFmpegVersionInfo:
    """Run `<program> -version` and parse the first output line."""
    result = subprocess.run(
        [binary_path, "-version"],
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"{program_name} failed to report its version: {stderr[:300]}"
        )

    output = result.stdout or result.stderr or ""
    first_line = next((line.strip() for line in output.splitlines() if line.strip()), "")
    if not first_line:
        raise RuntimeError(f"{program_name} did not report a version string")

    match = _FFMPEG_RELEASE_RE.search(first_line)
    if match:
        return FFmpegVersionInfo(first_line=first_line, major=int(match.group("major")))
    if _FFMPEG_DEV_BUILD_RE.search(first_line):
        return FFmpegVersionInfo(first_line=first_line, major=None, is_dev_build=True)

    raise RuntimeError(
        f"Could not determine {program_name} version from: {first_line}"
    )


def validate_ffmpeg_install(require_probe: bool = True) -> FFmpegValidationResult:
    """Validate FFmpeg/FFprobe availability, age, and Windows build type."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return FFmpegValidationResult(
            ok=False,
            message=(
                "FFmpeg not found. CorridorKey requires FFmpeg 7.0+ and FFprobe. "
                "Install a current FFmpeg build or re-run the installer."
            ),
        )

    ffprobe = find_ffprobe()
    if require_probe and not ffprobe:
        return FFmpegValidationResult(
            ok=False,
            message=(
                "FFprobe not found. CorridorKey requires both FFmpeg and FFprobe. "
                "Install a full FFmpeg build or re-run the installer."
            ),
            ffmpeg_path=ffmpeg,
        )

    try:
        ffmpeg_version = _read_program_version(ffmpeg, "ffmpeg")
    except RuntimeError as exc:
        return FFmpegValidationResult(ok=False, message=str(exc), ffmpeg_path=ffmpeg)

    if ffmpeg_version.major is not None and ffmpeg_version.major < _MIN_FFMPEG_MAJOR:
        return FFmpegValidationResult(
            ok=False,
            message=(
                f"FFmpeg 7.0 or newer is required. Detected {ffmpeg_version.first_line}."
            ),
            ffmpeg_path=ffmpeg,
            ffprobe_path=ffprobe,
            ffmpeg_version=ffmpeg_version,
        )

    if sys.platform == "win32" and "essentials_build" in ffmpeg_version.first_line.lower():
        return FFmpegValidationResult(
            ok=False,
            message=(
                "CorridorKey requires a full FFmpeg build on Windows. "
                "Detected a Gyan essentials build."
            ),
            ffmpeg_path=ffmpeg,
            ffprobe_path=ffprobe,
            ffmpeg_version=ffmpeg_version,
        )

    ffprobe_version: FFmpegVersionInfo | None = None
    if require_probe and ffprobe:
        try:
            ffprobe_version = _read_program_version(ffprobe, "ffprobe")
        except RuntimeError as exc:
            return FFmpegValidationResult(
                ok=False,
                message=str(exc),
                ffmpeg_path=ffmpeg,
                ffprobe_path=ffprobe,
                ffmpeg_version=ffmpeg_version,
            )

        if ffprobe_version.major is not None and ffprobe_version.major < _MIN_FFMPEG_MAJOR:
            return FFmpegValidationResult(
                ok=False,
                message=(
                    f"FFprobe 7.0 or newer is required. Detected {ffprobe_version.first_line}."
                ),
                ffmpeg_path=ffmpeg,
                ffprobe_path=ffprobe,
                ffmpeg_version=ffmpeg_version,
                ffprobe_version=ffprobe_version,
            )

        if (
            ffmpeg_version.major is not None
            and ffprobe_version.major is not None
            and ffmpeg_version.major != ffprobe_version.major
        ):
            return FFmpegValidationResult(
                ok=False,
                message=(
                    "FFmpeg and FFprobe major versions do not match. "
                    f"Detected ffmpeg {ffmpeg_version.major} and ffprobe {ffprobe_version.major}."
                ),
                ffmpeg_path=ffmpeg,
                ffprobe_path=ffprobe,
                ffmpeg_version=ffmpeg_version,
                ffprobe_version=ffprobe_version,
            )

        if (
            sys.platform == "win32"
            and "essentials_build" in ffprobe_version.first_line.lower()
        ):
            return FFmpegValidationResult(
                ok=False,
                message=(
                    "CorridorKey requires a full FFmpeg build on Windows. "
                    "Detected a Gyan essentials build."
                ),
                ffmpeg_path=ffmpeg,
                ffprobe_path=ffprobe,
                ffmpeg_version=ffmpeg_version,
                ffprobe_version=ffprobe_version,
            )

    if ffprobe_version is not None:
        summary = (
            f"FFmpeg OK: {ffmpeg_version.first_line} | {ffprobe_version.first_line}"
        )
    else:
        summary = f"FFmpeg OK: {ffmpeg_version.first_line}"

    return FFmpegValidationResult(
        ok=True,
        message=summary,
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
        ffmpeg_version=ffmpeg_version,
        ffprobe_version=ffprobe_version,
    )


def require_ffmpeg_install(require_probe: bool = True) -> FFmpegValidationResult:
    """Return the validated FFmpeg install or raise RuntimeError with detail."""
    result = validate_ffmpeg_install(require_probe=require_probe)
    if not result.ok:
        raise RuntimeError(result.message)
    return result


def get_ffmpeg_install_help() -> str:
    """Return concise install guidance for the current platform."""
    if sys.platform == "win32":
        return (
            "Use the CorridorKey Repair FFmpeg action or re-run 1-install.bat.\n"
            "CorridorKey will install a full bundled FFmpeg build into tools\\ffmpeg."
        )
    if sys.platform == "darwin":
        return (
            "Install a current FFmpeg build with Homebrew:\n"
            "    brew install ffmpeg\n\n"
            "Then verify:\n"
            "    ffmpeg -version\n"
            "    ffprobe -version"
        )
    if os.path.isfile("/etc/debian_version"):
        install_cmd = "sudo apt install ffmpeg"
    elif os.path.isfile("/etc/fedora-release"):
        install_cmd = "sudo dnf install ffmpeg"
    elif os.path.isfile("/etc/arch-release"):
        install_cmd = "sudo pacman -S ffmpeg"
    else:
        install_cmd = "Install ffmpeg with your package manager"
    return (
        f"{install_cmd}\n\n"
        "Then verify:\n"
        "    ffmpeg -version\n"
        "    ffprobe -version"
    )


def repair_ffmpeg_install(
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> FFmpegValidationResult:
    """Repair FFmpeg for the current platform.

    On Windows, downloads and installs a bundled full build into tools/ffmpeg.
    On macOS, installs via Homebrew (no sudo needed).
    On Linux, raises with install instructions (sudo requires a terminal).
    """
    if sys.platform == "darwin":
        def _emit(phase: str, current: int = 0, total: int = 0) -> None:
            if progress_callback:
                progress_callback(phase, current, total)

        if not shutil.which("brew"):
            raise RuntimeError(
                "Homebrew is not installed.\n\n"
                "Install it first:\n"
                '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"\n\n'
                "Then retry Repair FFmpeg."
            )

        _emit("Installing FFmpeg via Homebrew", 0, 0)
        try:
            subprocess.run(
                ["brew", "install", "ffmpeg"],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"FFmpeg install failed:\n{exc.stderr or exc.stdout or str(exc)}"
            ) from exc

        _emit("Validating FFmpeg", 0, 0)
        result = validate_ffmpeg_install(require_probe=True)
        if not result.ok:
            raise RuntimeError(result.message)
        return result

    if sys.platform != "win32":
        # Linux: needs sudo, can't run from GUI — show instructions instead
        raise RuntimeError(get_ffmpeg_install_help())

    def _emit(phase: str, current: int = 0, total: int = 0) -> None:
        if progress_callback:
            progress_callback(phase, current, total)

    tools_dir = os.path.join(_REPO_ROOT, "tools")
    dest_dir = os.path.join(tools_dir, "ffmpeg")
    temp_dir = os.path.join(_REPO_ROOT, ".tmp", "ffmpeg-repair")
    zip_path = os.path.join(temp_dir, "ffmpeg-master-latest-win64-gpl.zip")
    extract_dir = os.path.join(temp_dir, "extract")

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(tools_dir, exist_ok=True)

    _emit("Downloading FFmpeg", 0, 0)
    with urllib.request.urlopen(_WINDOWS_FFMPEG_BUNDLE_URL, timeout=60) as response:
        total_header = response.headers.get("Content-Length", "")
        total_bytes = int(total_header) if total_header.isdigit() else 0
        downloaded = 0
        with open(zip_path, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                _emit("Downloading FFmpeg", downloaded, total_bytes)

    _emit("Extracting FFmpeg", 0, 0)
    if os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir, ignore_errors=True)
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    inner_dir = None
    for name in os.listdir(extract_dir):
        candidate = os.path.join(extract_dir, name)
        if os.path.isdir(candidate) and name.lower().startswith("ffmpeg-"):
            inner_dir = candidate
            break
    if inner_dir is None:
        raise RuntimeError("Downloaded FFmpeg archive had an unexpected folder layout.")

    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir, ignore_errors=True)
    shutil.move(inner_dir, dest_dir)

    _emit("Validating FFmpeg", 0, 0)
    result = validate_ffmpeg_install(require_probe=True)
    if not result.ok:
        raise RuntimeError(result.message)
    return result


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


def _recompress_to_dwab(
    out_dir: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Recompress FFmpeg ZIP16 EXR files to DWAB in-place.

    NOTE: This always uses DWAB intentionally — it's an internal storage
    optimization for extracted video frames, not user output.  The user's
    EXR compression preference (PIZ/ZIP/etc.) applies only to inference
    output written by service.py._write_image().

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
_UNKNOWN_COLOR = {"", "unknown", "unspecified", "reserved", "unknown/unknown", None}


def _is_rgb_pix_fmt(pix_fmt: str) -> bool:
    """Return True if the pixel format is already RGB-family."""
    if not pix_fmt:
        return False
    pf = pix_fmt.lower()
    return (pf.startswith("rgb") or pf.startswith("bgr") or
            pf.startswith("gbr") or pf.startswith("argb") or
            pf.startswith("abgr") or pf == "pal8")


def _is_yuv_pix_fmt(pix_fmt: str) -> bool:
    """Return True for common YUV-family formats."""
    if not pix_fmt:
        return False
    pf = pix_fmt.lower()
    return (pf.startswith("yuv") or pf.startswith("yuva") or
            pf.startswith("nv12") or pf.startswith("nv16") or
            pf.startswith("nv21") or pf.startswith("p010") or
            pf.startswith("p016") or pf.startswith("p210") or
            pf.startswith("p216") or pf.startswith("p410") or
            pf.startswith("p416") or pf.startswith("y210") or
            pf.startswith("y212") or pf.startswith("y216"))


def _clean_color_value(value: str | None) -> str:
    """Normalize ffprobe color values for filter construction."""
    if value is None:
        return ""
    cleaned = str(value).strip().lower()
    return "" if cleaned in _UNKNOWN_COLOR else cleaned


def _default_matrix(width: int, height: int, primaries: str) -> str:
    """Best-effort matrix fallback when metadata is missing."""
    if primaries == "bt2020":
        return "bt2020nc"
    if height == 576:
        return "bt470bg"
    if height in (480, 486):
        return "smpte170m"
    hd = width >= 1280 or height > 576
    return "bt709" if hd else "smpte170m"


def _default_primaries(width: int, height: int, matrix: str) -> str:
    """Best-effort primaries fallback when metadata is missing."""
    if matrix in {"bt2020nc", "bt2020c", "bt2020cl"}:
        return "bt2020"
    if matrix == "bt470bg":
        return "bt470bg"
    if matrix == "smpte170m":
        return "smpte170m"
    hd = width >= 1280 or height > 576
    return "bt709" if hd else "smpte170m"


def _default_transfer(primaries: str, bits_per_raw_sample: int) -> str:
    """Best-effort transfer fallback when metadata is missing."""
    if primaries == "bt2020":
        return "bt2020-12" if bits_per_raw_sample > 10 else "bt2020-10"
    if primaries == "bt470bg":
        return "bt470bg"
    if primaries == "smpte170m":
        return "smpte170m"
    return "bt709"


def _default_range(pix_fmt: str) -> str:
    """Default range fallback for missing ffprobe metadata."""
    pf = (pix_fmt or "").lower()
    return "pc" if pf.startswith("yuvj") else "tv"


# ---------------------------------------------------------------------------
#  ffprobe → FFmpeg scale filter value mapping
# ---------------------------------------------------------------------------
# ffprobe reports ITU-T H.264/H.265 colour identifiers.  FFmpeg's `scale`
# filter only accepts a subset of named constants.  Values not in the
# "known safe" set get mapped to compatible equivalents.
#
# Built from libavutil/pixfmt.h + libswscale colour table.  If ffprobe
# reports a value we haven't seen, _safe_scale_value() logs a WARNING
# so we can add it before it crashes an extraction.
# ---------------------------------------------------------------------------

# in_color_matrix (swscale colorspace table)
_SCALE_MATRIX_MAP = {
    "bt470bg": "bt601",          # BT.470 System B/G = same matrix as BT.601
    "bt2020c": "bt2020ncl",     # constant-luminance → non-constant (swscale compat)
}
_KNOWN_MATRICES = {
    "bt709", "fcc", "bt601", "smpte170m", "smpte240m",
    "bt2020nc", "bt2020ncl",
}

# in_primaries
_SCALE_PRIMARIES_MAP = {
    # Most ffprobe primaries pass through. Guard edge cases.
    "film": "bt470m",           # "film" (SMPTE-C) → bt470m
}
_KNOWN_PRIMARIES = {
    "bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m",
    "film", "bt2020", "smpte428", "smpte431", "smpte432",
}

# in_transfer
_SCALE_TRANSFER_MAP = {
    "bt470bg": "gamma28",       # BT.470 System B/G = gamma 2.8
    "bt470m": "gamma22",        # BT.470 System M = gamma 2.2
    "bt2020-12": "bt2020-12",   # pass through (explicit for clarity)
    "bt2020-10": "bt2020-10",
}
_KNOWN_TRANSFERS = {
    "bt709", "gamma22", "gamma28", "smpte170m", "smpte240m",
    "linear", "log", "log_sqrt", "iec61966-2-4", "bt1361e",
    "iec61966-2-1", "bt2020-10", "bt2020-12", "smpte2084",
    "smpte428", "arib-std-b67",
}


def _safe_scale_value(value: str, mapping: dict, known: set, param_name: str) -> str:
    """Map an ffprobe colour identifier to an FFmpeg scale-filter-safe name.

    Logs a WARNING for unrecognised values so we can add them to the
    mapping before they crash an extraction.
    """
    mapped = mapping.get(value, value)
    if mapped and mapped not in known:
        logger.warning(
            "Unknown %s value '%s' (mapped from '%s') — FFmpeg may reject this. "
            "Add it to _SCALE_%s_MAP in ffmpeg_tools.py",
            param_name, mapped, value, param_name.upper(),
        )
    return mapped


def build_exr_vf(video_info: dict) -> str:
    """Build the -vf string for converting to gbrpf32le (EXR output).

    For RGB inputs, just do format conversion.
    For YUV inputs, use an explicit scale+format chain and provide
    input colour metadata directly to the scaler. This preserves the
    current swscale conversion path for well-tagged files and only
    falls back to heuristics when the source metadata is missing.
    """
    pix_fmt = (video_info.get("pix_fmt", "") or "").lower()

    if _is_rgb_pix_fmt(pix_fmt):
        return "format=gbrpf32le"

    # Unknown / oddball formats keep the legacy implicit path.
    if not _is_yuv_pix_fmt(pix_fmt):
        return "format=gbrpf32le"

    cs = _clean_color_value(video_info.get("color_space"))
    cp = _clean_color_value(video_info.get("color_primaries"))
    ct = _clean_color_value(video_info.get("color_transfer"))
    cr = _clean_color_value(video_info.get("color_range"))
    w = video_info.get("width", 0)
    h = video_info.get("height", 0)
    bits = int(video_info.get("bits_per_raw_sample", 0) or 0)

    if not cs:
        cs = _default_matrix(w, h, cp)
    if not cp:
        cp = _default_primaries(w, h, cs)
    if not ct:
        ct = _default_transfer(cp, bits)
    if not cr:
        cr = _default_range(pix_fmt)

    # Map ffprobe identifiers → scale-filter-safe names.
    cs = _safe_scale_value(cs, _SCALE_MATRIX_MAP, _KNOWN_MATRICES, "matrix")

    logger.info(
        "EXR colour conversion: pix_fmt=%s matrix=%s range=%s",
        pix_fmt, cs, cr,
    )

    # Only in_color_matrix and in_range are standard swscale options.
    # in_primaries / in_transfer are NOT supported by FFmpeg's scale filter
    # and cause "Option not found" on standard builds.
    #
    # FFmpeg 8.x swscaler rejects YUV→RGB when TRC is null (bug #11585).
    # Prepend setparams to tag the missing TRC — metadata only, no pixel math.
    raw_ct = video_info.get("color_transfer")
    prefix = f"setparams=color_trc={ct}," if raw_ct is None else ""

    return (
        f"{prefix}scale=in_color_matrix={cs}:in_range={cr},format=gbrpf32le"
    )


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
    validation = require_ffmpeg_install(require_probe=True)
    ffmpeg = validation.ffmpeg_path
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found")

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
    vf_chain = build_exr_vf(video_info or {})
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
    validation = require_ffmpeg_install(require_probe=True)
    ffmpeg = validation.ffmpeg_path
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found")

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
    frame_count, codec, duration, plus optional diagnostic fields such
    as source_probe and exr_vf for extraction bug reports.
    """
    path = os.path.join(clip_root, _METADATA_FILENAME)
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.debug(f"Video metadata written: {path}")


class FFmpegFrameReader:
    """Stream-read video frames via FFmpeg subprocess pipe.

    Decodes frames to raw RGB float32 in memory, no intermediate files.
    Uses hardware decode when available (NVDEC, VideoToolbox, VAAPI).

    Usage:
        reader = FFmpegFrameReader("input.mp4")
        for frame in reader:
            process(frame)  # np.ndarray [H, W, 3] float32 0-1
        reader.close()
    """

    def __init__(self, video_path: str, hwaccel: bool = True):
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("FFmpeg not found")

        info = probe_video(video_path)
        self.width = info["width"]
        self.height = info["height"]
        self.frame_count = info.get("frame_count", 0)
        self._frame_bytes = self.width * self.height * 3 * 4  # float32 RGB

        hw_flags = detect_hwaccel(ffmpeg) if hwaccel else []
        cmd = [
            ffmpeg,
            *hw_flags,
            "-i", video_path,
            "-f", "rawvideo",
            "-pix_fmt", "gbrpf32le",
            "-vsync", "passthrough",
            "-v", "error",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    def __iter__(self):
        return self

    def __next__(self) -> "np.ndarray":
        from CorridorKeyModule.core.native_ops import gbr_planar_to_rgb
        raw = self._proc.stdout.read(self._frame_bytes)
        if len(raw) < self._frame_bytes:
            self.close()
            raise StopIteration
        return gbr_planar_to_rgb(raw, self.height, self.width)

    def close(self):
        if self._proc.poll() is None:
            self._proc.stdout.close()
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FFmpegFrameWriter:
    """Stream-write RGB frames to video via FFmpeg subprocess pipe.

    Uses NVENC when available for near-zero-cost encoding (~1ms/frame).
    Falls back to libx264 otherwise.

    Usage:
        writer = FFmpegFrameWriter("output.mp4", width=1920, height=1080, fps=24)
        writer.write(frame)  # np.ndarray [H, W, 3] uint8
        writer.close()
    """

    def __init__(self, output_path: str, width: int, height: int, fps: float = 24.0, crf: int = 18):
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("FFmpeg not found")

        self.width = width
        self.height = height
        self._frame_bytes = width * height * 3  # uint8 RGB

        # Detect hardware encoder: NVENC (NVIDIA), AMF (AMD), QSV (Intel)
        codec = "libx264"
        hwaccels = detect_hwaccel(ffmpeg)
        hw_set = set(hwaccels)
        if "cuda" in hw_set:
            codec = "h264_nvenc"
        elif "vaapi" in hw_set or "d3d11va" in hw_set:
            codec = "h264_amf" if os.name == "nt" else "h264_vaapi"
        elif "qsv" in hw_set:
            codec = "h264_qsv"

        cmd = [
            ffmpeg,
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", codec,
            "-pix_fmt", "yuv420p",
            "-v", "error",
        ]
        if codec == "h264_nvenc":
            cmd.extend(["-preset", "p4", "-rc", "constqp", "-qp", str(crf)])
        elif codec == "h264_amf":
            cmd.extend(["-quality", "speed", "-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf)])
        elif codec == "h264_vaapi":
            cmd.extend(["-qp", str(crf)])
        elif codec == "h264_qsv":
            cmd.extend(["-preset", "faster", "-global_quality", str(crf)])
        else:
            cmd.extend(["-crf", str(crf), "-preset", "fast"])
        cmd.append(output_path)

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    def write(self, frame: "np.ndarray") -> None:
        """Write a single RGB uint8 frame."""
        self._proc.stdin.write(frame.tobytes())

    def close(self):
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        if self._proc.poll() is None:
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


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

"""Frame extraction from video files via FFmpeg."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
from typing import Callable, Optional

from .color import build_exr_vf
from .discovery import find_ffmpeg, require_ffmpeg_install
from .probe import probe_video

logger = logging.getLogger(__name__)

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

    In frozen (PyInstaller) builds, uses multiprocessing.ProcessPoolExecutor
    with spawn start method (requires freeze_support() in main.py).
    In dev mode, spawns a standalone subprocess for full GIL bypass.
    Both paths keep the parent process (and its Qt event loop) completely free.
    """
    marker = os.path.join(out_dir, ".dwab_done")
    if os.path.isfile(marker):
        return

    exr_files = sorted([f for f in os.listdir(out_dir)
                        if f.lower().endswith('.exr')])
    total = len(exr_files)
    if total == 0:
        return

    if getattr(sys, "frozen", False):
        _recompress_multiprocess(out_dir, exr_files, total, marker,
                                 on_progress, cancel_event)
    else:
        _recompress_subprocess(out_dir, exr_files, total, marker,
                               on_progress, cancel_event)


def _recompress_one_exr(args: tuple) -> bool:
    """Recompress a single EXR file to DWAB. Runs in a child process."""
    src, tmp = args
    try:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2
        import numpy as np
        import OpenEXR
        import Imath
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
                'R': img[:, :, 2].astype(np.float16).tobytes(),
                'G': img[:, :, 1].astype(np.float16).tobytes(),
                'B': img[:, :, 0].astype(np.float16).tobytes(),
            })
            out.close()
        elif img.ndim == 3 and img.shape[2] == 4:
            hdr['channels'] = {'R': HALF, 'G': HALF, 'B': HALF, 'A': HALF}
            out = OpenEXR.OutputFile(tmp, hdr)
            out.writePixels({
                'R': img[:, :, 2].astype(np.float16).tobytes(),
                'G': img[:, :, 1].astype(np.float16).tobytes(),
                'B': img[:, :, 0].astype(np.float16).tobytes(),
                'A': img[:, :, 3].astype(np.float16).tobytes(),
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


def _recompress_multiprocess(
    out_dir: str,
    exr_files: list[str],
    total: int,
    marker: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """DWAB recompress using ProcessPoolExecutor (frozen builds).

    Uses multiprocessing with spawn start method so each worker is a real
    child process — no GIL contention with the parent Qt event loop.
    Requires multiprocessing.freeze_support() in main.py (already present).
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    logger.info(f"Recompressing {total} EXR frames to DWAB (multiprocess)...")
    workers = max(1, min((os.cpu_count() or 4) // 2, 16))
    done = 0

    ctx = multiprocessing.get_context("spawn")
    work = [(os.path.join(out_dir, f), os.path.join(out_dir, f + ".tmp"))
            for f in exr_files]

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = {pool.submit(_recompress_one_exr, item): item for item in work}
        for fut in as_completed(futs):
            if cancel_event and cancel_event.is_set():
                pool.shutdown(wait=False, cancel_futures=True)
                logger.info("DWAB recompression cancelled")
                return
            fut.result()
            done += 1
            if on_progress:
                on_progress(done, total)

    with open(marker, 'w') as f:
        f.write("done")
    logger.info(f"DWAB recompression complete: {total} frames")


def _recompress_subprocess(
    out_dir: str,
    exr_files: list[str],
    total: int,
    marker: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """DWAB recompress using a subprocess with ProcessPoolExecutor (dev mode)."""
    python = sys.executable

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

    script_path = os.path.join(out_dir, "_dwab_recompress.py")
    with open(script_path, 'w') as f:
        f.write(script_content)

    logger.info(f"Recompressing {total} EXR frames to DWAB (subprocess)...")

    proc = subprocess.Popen(
        [python, script_path, out_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

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
        try:
            os.remove(script_path)
        except OSError:
            pass

    if proc.returncode != 0:
        stderr_out = proc.stderr.read() if proc.stderr else ""
        logger.error(f"DWAB recompression failed (code {proc.returncode}): "
                     f"{stderr_out[:500]}")
        return

    with open(marker, 'w') as f:
        f.write("done")
    logger.info(f"DWAB recompression complete: {total} frames")


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
            encoding="utf-8", errors="replace",
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

    # Pass 2: Recompress ZIP16 -> DWAB
    if extracted > 0 and not (cancel_event and cancel_event.is_set()):
        _recompress_to_dwab(out_dir, on_recompress_progress, cancel_event)

    return extracted

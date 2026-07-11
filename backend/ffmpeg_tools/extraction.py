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

# ---------------------------------------------------------------------------
# Adaptive EXR encoder threading — commit-headroom aware
# ---------------------------------------------------------------------------
# FFmpeg's EXR encoder is frame-threaded: N threads hold N complete frames
# in flight. Measured on the pinned build with a 4K clip (issue #184 bench):
# ~435 MB of committed memory per thread (~52 bytes/pixel/thread) plus
# ~2 GB pipeline base. With threads=auto a 32-core machine commits ~15.6 GB
# for one 4K import, which fails with "Cannot allocate memory" (-12) on any
# machine whose commit headroom is smaller — a loaded 16 GB laptop can never
# import 4K at auto threads. Cap threads to what the machine can actually
# promise, and leave fast machines at full speed.
_ENC_BYTES_PER_PIXEL_PER_THREAD = 52
_ENC_COMMIT_SAFETY = 0.5    # use at most half the free commit headroom
_ENC_MIN_THREADS = 2
_ENC_MAX_THREADS = 64


def _free_commit_bytes() -> int | None:
    """Free Windows commit charge (limit - total), or None if unavailable.

    Commit is the constraint malloc actually hits — physical RAM can look
    free while the commit ledger is full (issue #184 QA finding).
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class PERFORMANCE_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("CommitTotal", ctypes.c_size_t),
                ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t),
                ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t),
                ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t),
                ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t),
                ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_uint32),
                ("ProcessCount", ctypes.c_uint32),
                ("ThreadCount", ctypes.c_uint32),
            ]

        info = PERFORMANCE_INFORMATION()
        info.cb = ctypes.sizeof(info)
        if not ctypes.windll.psapi.GetPerformanceInfo(
            ctypes.byref(info), info.cb
        ):
            return None
        return (info.CommitLimit - info.CommitTotal) * info.PageSize
    except Exception:
        return None


def adaptive_encoder_threads(width: int, height: int) -> list[str]:
    """Return ["-threads:v", N] when commit headroom demands a cap, else [].

    Empty list means FFmpeg picks its default (one thread per core), which
    is correct whenever the machine can afford it.
    """
    if width <= 0 or height <= 0:
        return []
    free = _free_commit_bytes()
    if free is None:
        return []
    cores = os.cpu_count() or 4
    per_thread = _ENC_BYTES_PER_PIXEL_PER_THREAD * width * height
    if per_thread <= 0:
        return []
    afford = int((free * _ENC_COMMIT_SAFETY) / per_thread)
    threads = max(_ENC_MIN_THREADS, min(afford, cores, _ENC_MAX_THREADS))
    if threads >= cores:
        return []  # machine can afford full auto threading
    logger.info(
        f"EXR encoder threads capped at {threads} "
        f"(free commit {free / 1e9:.1f} GB, "
        f"~{per_thread / 1e6:.0f} MB/thread at {width}x{height})"
    )
    return ["-threads:v", str(threads)]


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
) -> list[str]:
    """Recompress FFmpeg ZIP16 EXR files to DWAB in-place.

    NOTE: This always uses DWAB intentionally — it's an internal storage
    optimization for extracted video frames, not user output.  The user's
    EXR compression preference (PIZ/ZIP/etc.) applies only to inference
    output written by service.py._write_image().

    In frozen (PyInstaller) builds, uses multiprocessing.ProcessPoolExecutor
    with spawn start method (requires freeze_support() in main.py).
    In dev mode, spawns a standalone subprocess for full GIL bypass.
    Both paths keep the parent process (and its Qt event loop) completely free.

    Returns the list of frame filenames that could not be read back and
    recompressed (unreadable = corrupt FFmpeg output, see issue #184).
    The .dwab_done marker is only written when every frame succeeded.
    """
    marker = os.path.join(out_dir, ".dwab_done")
    if os.path.isfile(marker):
        return []

    exr_files = sorted([f for f in os.listdir(out_dir)
                        if f.lower().endswith('.exr')])
    total = len(exr_files)
    if total == 0:
        return []

    if getattr(sys, "frozen", False):
        return _recompress_multiprocess(out_dir, exr_files, total, marker,
                                        on_progress, cancel_event)
    return _recompress_subprocess(out_dir, exr_files, total, marker,
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
        from backend.frame_io import imread_unicode
        img = imread_unicode(src, cv2.IMREAD_UNCHANGED)
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


def _recompress_sequential(
    out_dir: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[str]:
    """DWAB recompress in-process, one frame at a time — no worker pool.

    Fallback for the safe-mode retry in extract_frames: slower than the
    pooled paths, but immune to pool-level failures (BrokenProcessPool from
    AV kills, OOM, worker crashes). Returns filenames that failed to read.
    """
    marker = os.path.join(out_dir, ".dwab_done")
    if os.path.isfile(marker):
        return []

    exr_files = sorted([f for f in os.listdir(out_dir)
                        if f.lower().endswith('.exr')])
    total = len(exr_files)
    if total == 0:
        return []

    logger.info(f"Recompressing {total} EXR frames to DWAB (sequential)...")
    failed: list[str] = []
    for done, fname in enumerate(exr_files, 1):
        if cancel_event and cancel_event.is_set():
            logger.info("DWAB recompression cancelled")
            return []
        src = os.path.join(out_dir, fname)
        if not _recompress_one_exr((src, src + ".tmp")):
            failed.append(fname)
        if on_progress:
            on_progress(done, total)

    if failed:
        failed.sort()
        logger.warning(f"DWAB recompression (sequential): {len(failed)}/{total} "
                       f"frame(s) unreadable (corrupt): {failed[:5]}")
        return failed

    with open(marker, 'w') as f:
        f.write("done")
    logger.info(f"DWAB recompression complete: {total} frames")
    return []


def _recompress_multiprocess(
    out_dir: str,
    exr_files: list[str],
    total: int,
    marker: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[str]:
    """DWAB recompress using ProcessPoolExecutor (frozen builds).

    Uses multiprocessing with spawn start method so each worker is a real
    child process — no GIL contention with the parent Qt event loop.
    Requires multiprocessing.freeze_support() in main.py (already present).

    Returns filenames of frames that failed to read back (corrupt).
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    logger.info(f"Recompressing {total} EXR frames to DWAB (multiprocess)...")
    workers = max(1, min((os.cpu_count() or 4) // 2, 16))
    done = 0
    failed: list[str] = []

    ctx = multiprocessing.get_context("spawn")
    work = [(os.path.join(out_dir, f), os.path.join(out_dir, f + ".tmp"))
            for f in exr_files]

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = {pool.submit(_recompress_one_exr, item): item for item in work}
        for fut in as_completed(futs):
            if cancel_event and cancel_event.is_set():
                pool.shutdown(wait=False, cancel_futures=True)
                logger.info("DWAB recompression cancelled")
                return []
            # A worker terminated abruptly (BrokenProcessPool: AV kill, OOM,
            # native crash) must not abort the whole pass — mark the item
            # failed so the safe-mode retry in extract_frames can heal it.
            try:
                ok = fut.result()
            except Exception as exc:
                logger.warning(f"DWAB worker failed for "
                               f"{os.path.basename(futs[fut][0])}: {exc}")
                ok = False
            if not ok:
                failed.append(os.path.basename(futs[fut][0]))
            done += 1
            if on_progress:
                on_progress(done, total)

    if failed:
        failed.sort()
        logger.warning(f"DWAB recompression: {len(failed)}/{total} frame(s) "
                       f"unreadable (corrupt): {failed[:5]}")
        return failed

    with open(marker, 'w') as f:
        f.write("done")
    logger.info(f"DWAB recompression complete: {total} frames")
    return []


def _recompress_subprocess(
    out_dir: str,
    exr_files: list[str],
    total: int,
    marker: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[str]:
    """DWAB recompress using a subprocess with ProcessPoolExecutor (dev mode).

    Returns filenames of frames that failed to read back (corrupt).
    """
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
        # Unicode-safe read: cv2.imread fails on non-ASCII paths on Windows.
        # This standalone script cannot import the backend.frame_io facade, so
        # the imdecode(np.fromfile(...)) pattern is inlined here.
        try:
            img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except (OSError, ValueError):
            img = None
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
            # Worker died abruptly (BrokenProcessPool etc.) -> mark item
            # failed instead of aborting the whole pass.
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if not ok:
                print(f"FAILED {os.path.basename(futs[fut][0])}", flush=True)
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

    failed: list[str] = []
    try:
        while True:
            if cancel_event and cancel_event.is_set():
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                logger.info("DWAB recompression cancelled")
                return []

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
            elif line.startswith("FAILED "):
                failed.append(line.split(maxsplit=1)[1])
            elif line == "DONE":
                break

        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        logger.error("DWAB recompression subprocess timed out")
        # The pass did not complete, so no frame is verified — report every
        # frame as suspect so extract_frames runs the safe-mode retry
        # (which re-validates with the pool-free sequential pass) instead of
        # treating a crashed recompress as clean.
        return list(exr_files)
    finally:
        try:
            os.remove(script_path)
        except OSError:
            pass

    if proc.returncode != 0:
        stderr_out = proc.stderr.read() if proc.stderr else ""
        # Log the TAIL of stderr — the actual exception sits at the bottom
        # of a Python traceback, a head-slice hides it.
        logger.error(f"DWAB recompression failed (code {proc.returncode}): "
                     f"{stderr_out[-2000:]}")
        # Same as the timeout path: incomplete pass = nothing verified.
        return list(exr_files)

    if failed:
        failed.sort()
        logger.warning(f"DWAB recompression: {len(failed)}/{total} frame(s) "
                       f"unreadable (corrupt): {failed[:5]}")
        return failed

    with open(marker, 'w') as f:
        f.write("done")
    logger.info(f"DWAB recompression complete: {total} frames")
    return []


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

    # Cap EXR encoder threads when commit headroom can't afford
    # one full frame buffer per core (issue #184 bench).
    enc_thread_args = adaptive_encoder_threads(
        (video_info or {}).get("width", 0),
        (video_info or {}).get("height", 0),
    )

    def _build_cmd(hw_flags: list[str],
                   extra_out_args: list[str] | None = None) -> list[str]:
        out_args = extra_out_args or []
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
                # -fps_mode:v passthrough replaces the legacy -vsync passthrough.
                # -vsync was deprecated in 2023 and is removed in current FFmpeg
                # git-master builds, which made imports fail with
                # "Unrecognized option 'vsync'". -fps_mode exists since FFmpeg 5.1
                # and our minimum is FFmpeg 7, so it is always available.
                "-fps_mode:v", "passthrough",
                *exr_args,
                *out_args,
                out_dir + "/" + pattern,
                "-y",
            ]
        return [
            ffmpeg,
            *hw_flags,
            "-i", video_path,
            "-start_number", "0",
            # See note above: -fps_mode:v passthrough replaces removed -vsync.
            "-fps_mode:v", "passthrough",
            *exr_args,
            *out_args,
            out_dir + "/" + pattern,
            "-y",
        ]

    def _run_ffmpeg(hw_flags: list[str],
                    extra_out_args: list[str] | None = None) -> tuple[int, str]:
        """Run FFmpeg extraction. Returns (return_code, last_stderr_lines)."""
        nonlocal last_frame

        cmd = _build_cmd(hw_flags, extra_out_args)
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
    returncode, stderr_out = _run_ffmpeg(hwaccel_flags, enc_thread_args)

    # If hardware decode failed, retry with software decode
    if returncode != 0 and hwaccel_flags and not (cancel_event and cancel_event.is_set()):
        logger.warning(f"Hardware decode failed (code {returncode}), "
                       f"retrying with software decode...")
        # Clean up any partial frames from the failed attempt. ALL frames are
        # deleted (including any pre-existing resume prefix), so the retry
        # must restart from frame 0 — leaving start_frame > 0 here would
        # seek past the beginning and permanently drop the prefix frames.
        for f in os.listdir(out_dir):
            if f.lower().endswith('.exr'):
                os.remove(os.path.join(out_dir, f))
        start_frame = 0
        last_frame = 0
        # empty hw flags = software decode
        returncode, stderr_out = _run_ffmpeg([], enc_thread_args)

    if returncode != 0 and not (cancel_event and cancel_event.is_set()):
        # Out-of-memory gets a human message: the raw FFmpeg error reads as
        # a broken app, but the cause is commit exhaustion on the machine
        # (issue #184 QA finding).
        if stderr_out and "cannot allocate memory" in stderr_out.lower():
            raise RuntimeError(
                "Not enough free memory to import this clip. "
                "Close other applications and run the extraction again."
            )
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

    # Pass 2: Recompress ZIP16 -> DWAB. The recompress workers read every
    # frame back, so this pass doubles as an integrity check on FFmpeg's
    # output — corrupt frames fail to read and are reported back here.
    if extracted > 0 and not (cancel_event and cancel_event.is_set()):
        corrupt = _recompress_to_dwab(out_dir, on_recompress_progress,
                                      cancel_event)

        # Corrupt frames = FFmpeg wrote unreadable EXR data (issue #184:
        # a race in the hwaccel-decode + threaded-EXR-encode pipeline emits
        # invalid zlib chunks inside ZIP16 EXRs on some machines). Wipe and
        # re-extract once with both race suspects disabled — software decode
        # AND a single encoder thread — then re-validate. Much slower, but it
        # only runs when the fast path provably produced corrupt output.
        # Fail loudly rather than let inference trip over unreadable frames
        # mid-job later.
        retried = False
        if corrupt and not (cancel_event and cancel_event.is_set()):
            logger.warning(
                f"{len(corrupt)} corrupt frame(s) in FFmpeg output "
                f"(e.g. {corrupt[0]}) — re-extracting with software decode, "
                f"single-threaded encode...")
            for f in os.listdir(out_dir):
                if f.lower().endswith('.exr'):
                    os.remove(os.path.join(out_dir, f))
            start_frame = 0
            last_frame = 0
            retried = True
            returncode, stderr_out = _run_ffmpeg([], ["-threads:v", "1"])
            if returncode != 0 and not (cancel_event and cancel_event.is_set()):
                raise RuntimeError(
                    f"FFmpeg safe-mode retry failed (code {returncode})")
            extracted = len([f for f in os.listdir(out_dir)
                             if f.lower().endswith('.exr')])
            # Sequential recompress: immune to worker-pool failures, so a
            # broken pool can't mask the state of the re-extracted frames.
            corrupt = _recompress_sequential(out_dir, on_recompress_progress,
                                             cancel_event)
        if corrupt and not (cancel_event and cancel_event.is_set()):
            raise RuntimeError(
                f"Extraction produced {len(corrupt)} unreadable frame(s) "
                f"even after safe-mode retry (e.g. {corrupt[0]}). "
                f"Run Repair FFmpeg in Preferences and re-import the clip.")
        if retried and not corrupt:
            logger.info("Safe-mode re-extraction succeeded: all frames "
                        "readable")

    return extracted

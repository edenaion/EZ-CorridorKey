"""Background video extraction worker.

Runs FFmpeg extraction in a separate QThread — does NOT block the GPU
queue (extraction is CPU/hardware decode, not GPU inference).

Supports multiple concurrent extraction requests via an internal job
queue. Each clip gets its own cancel event for independent cancellation.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from collections import deque
from dataclasses import dataclass, field

from PySide6.QtCore import QThread, Signal

from backend.ffmpeg_tools import (
    require_ffmpeg_install,
    probe_video,
    extract_frames,
    write_video_metadata,
    build_exr_vf,
)

logger = logging.getLogger(__name__)


@dataclass
class _ExtractJob:
    clip_name: str
    video_path: str
    clip_root: str
    cancel_event: threading.Event = field(default_factory=threading.Event)


class ExtractWorker(QThread):
    """Background worker for video → image sequence extraction.

    Signals:
        progress(clip_name, current_frame, total_frames)
        finished(clip_name, frame_count)
        error(clip_name, error_message)
    """

    progress = Signal(str, int, int)
    finished = Signal(str, int)
    error = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs: deque[_ExtractJob] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop_flag = False
        self._busy = False  # True while actively processing a job
        self._current_job: _ExtractJob | None = None  # actively running job

    @property
    def is_busy(self) -> bool:
        """True if actively extracting or has pending jobs."""
        with self._lock:
            return self._busy or bool(self._jobs)

    def submit(self, clip_name: str, video_path: str, clip_root: str) -> None:
        """Queue a video extraction job."""
        with self._lock:
            # Skip if already queued
            for job in self._jobs:
                if job.clip_name == clip_name:
                    logger.debug(f"Extraction already queued: {clip_name}")
                    return
        job = _ExtractJob(clip_name=clip_name, video_path=video_path, clip_root=clip_root)
        with self._lock:
            self._jobs.append(job)
        self._wake.set()
        logger.info(f"Extraction queued: {clip_name}")

    def cancel(self, clip_name: str) -> None:
        """Cancel extraction for a specific clip."""
        with self._lock:
            # Check currently-running job
            if self._current_job and self._current_job.clip_name == clip_name:
                self._current_job.cancel_event.set()
                return
            # Check pending queue
            for job in self._jobs:
                if job.clip_name == clip_name:
                    job.cancel_event.set()
                    return

    def stop(self) -> None:
        """Stop the worker thread — cancels current + all pending jobs."""
        self._stop_flag = True
        with self._lock:
            # Cancel the actively-running job
            if self._current_job is not None:
                self._current_job.cancel_event.set()
            # Cancel all pending
            for job in self._jobs:
                job.cancel_event.set()
        self._wake.set()

    def run(self) -> None:
        """Consumer loop — processes extraction jobs sequentially."""
        while not self._stop_flag:
            job = None
            with self._lock:
                if self._jobs:
                    job = self._jobs.popleft()

            if job is None:
                self._wake.clear()
                self._wake.wait(timeout=1.0)
                continue

            if job.cancel_event.is_set():
                continue

            self._busy = True
            with self._lock:
                self._current_job = job
            try:
                self._process_job(job)
            finally:
                with self._lock:
                    self._current_job = None
                self._busy = False

    def _process_job(self, job: _ExtractJob) -> None:
        """Extract a single video to image sequence."""
        try:
            # Check FFmpeg availability
            try:
                require_ffmpeg_install(require_probe=True)
            except RuntimeError as exc:
                self.error.emit(job.clip_name, str(exc))
                return

            # Probe video for metadata
            try:
                info = probe_video(job.video_path)
            except Exception as e:
                self.error.emit(job.clip_name, f"Failed to probe video: {e}")
                return

            total_frames = info.get("frame_count", 0)

            # Target: Frames/ for new-format projects, Input/ for legacy
            source_dir = os.path.join(job.clip_root, "Source")
            if os.path.isdir(source_dir):
                target_dir = os.path.join(job.clip_root, "Frames")
            else:
                target_dir = os.path.join(job.clip_root, "Input")
            os.makedirs(target_dir, exist_ok=True)

            # Progress callbacks → signal (throttled to avoid flooding the
            # main thread with expensive UI updates).  Emit at most every
            # 100 ms, plus always emit the final frame so the bar hits 100%.
            import time as _time

            _MIN_INTERVAL = 0.10  # seconds between progress emissions
            _last_emit = [0.0]  # mutable closure — [timestamp]

            def _throttled_emit(clip: str, cur: int, tot: int) -> None:
                now = _time.monotonic()
                if cur >= tot or (now - _last_emit[0]) >= _MIN_INTERVAL:
                    self.progress.emit(clip, cur, tot)
                    _last_emit[0] = now

            def on_progress(current: int, total: int) -> None:
                _throttled_emit(job.clip_name, current, total)

            def on_recompress(current: int, total: int) -> None:
                # Report as total_frames + progress to show "compressing"
                # without resetting the counter
                _throttled_emit(job.clip_name, total_frames + current, total_frames + total)

            # Run extraction (two-pass: FFmpeg EXR ZIP16 → DWAB recompress)
            extracted = extract_frames(
                video_path=job.video_path,
                out_dir=target_dir,
                on_progress=on_progress,
                on_recompress_progress=on_recompress,
                cancel_event=job.cancel_event,
                total_frames=total_frames,
            )

            if job.cancel_event.is_set():
                logger.info(f"Extraction cancelled: {job.clip_name}")
                return

            # Write metadata sidecar (for stitching back later)
            metadata = {
                "source_path": job.video_path,
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
            write_video_metadata(job.clip_root, metadata)

            logger.info(f"Extraction complete: {job.clip_name} ({extracted} frames)")

            # Auto-detect companion alpha hint video
            if not job.cancel_event.is_set():
                self._try_extract_alphahint(job, extracted)

            self.finished.emit(job.clip_name, extracted)

        except Exception as e:
            logger.error(f"Extraction failed for {job.clip_name}: {e}")
            self.error.emit(job.clip_name, str(e))

    # ------------------------------------------------------------------
    # Alpha-hint auto-detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_alphahint_video(video_path: str) -> str | None:
        """Look for a companion ``*_alphahint.*`` video next to *video_path*.

        Naming convention:  ``input_alphahint.mov`` alongside ``input.mov``.
        Also checks the Source/ folder if the video lives there.
        """

        stem = os.path.splitext(os.path.basename(video_path))[0]
        parent = os.path.dirname(video_path)
        video_exts = (".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v")

        # Pattern: {stem}_alphahint.{ext}  (case-insensitive match)
        for f in os.listdir(parent):
            name_lower = f.lower()
            f_stem, f_ext = os.path.splitext(name_lower)
            if f_ext in video_exts and f_stem == f"{stem.lower()}_alphahint":
                return os.path.join(parent, f)

        # Also check clip.json for original_path and look there
        clip_root = os.path.dirname(parent)  # Source/ -> clip_root
        clip_json = os.path.join(clip_root, "clip.json")
        if os.path.isfile(clip_json):
            import json

            try:
                with open(clip_json, "r") as fh:
                    data = json.load(fh)
                orig = data.get("source", {}).get("original_path", "")
                if orig and os.path.isfile(orig):
                    orig_stem = os.path.splitext(os.path.basename(orig))[0]
                    orig_dir = os.path.dirname(orig)
                    for f in os.listdir(orig_dir):
                        name_lower = f.lower()
                        f_stem, f_ext = os.path.splitext(name_lower)
                        if f_ext in video_exts and f_stem == f"{orig_stem.lower()}_alphahint":
                            return os.path.join(orig_dir, f)
            except Exception:
                pass

        return None

    def _try_extract_alphahint(self, job: _ExtractJob, frame_count: int) -> None:
        """If a companion alpha-hint video exists, extract it as grayscale PNGs."""
        alpha_video = self._find_alphahint_video(job.video_path)
        if alpha_video is None:
            return

        alpha_dir = os.path.join(job.clip_root, "AlphaHint")
        if os.path.isdir(alpha_dir) and os.listdir(alpha_dir):
            logger.info("AlphaHint/ already populated, skipping auto-extract")
            return

        logger.info("Auto-detected alpha hint video: %s", alpha_video)
        try:
            validation = require_ffmpeg_install()
            ffmpeg = validation.ffmpeg_path
            if not ffmpeg:
                logger.warning("FFmpeg not available for alpha hint extraction")
                return

            os.makedirs(alpha_dir, exist_ok=True)
            out_pattern = os.path.join(alpha_dir, "frame%06d.png")

            import subprocess

            cmd = [
                ffmpeg,
                "-y",
                "-i",
                alpha_video,
                "-vf",
                "format=gray",
                "-pix_fmt",
                "gray",
                out_pattern,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if result.returncode != 0:
                logger.error("Alpha hint extraction failed: %s", result.stderr[-500:])
                # Clean up partial output
                if os.path.isdir(alpha_dir):
                    shutil.rmtree(alpha_dir)
                return

            # Count extracted frames
            alpha_frames = len([f for f in os.listdir(alpha_dir) if f.lower().endswith(".png")])
            logger.info("Auto-extracted %d alpha hint frames into %s", alpha_frames, alpha_dir)

            if alpha_frames != frame_count:
                logger.warning(
                    "Alpha hint frame count (%d) != source frame count (%d)",
                    alpha_frames,
                    frame_count,
                )

        except Exception as exc:
            logger.error("Alpha hint auto-extraction failed: %s", exc)
            # Don't fail the main extraction over this
            if os.path.isdir(alpha_dir):
                shutil.rmtree(alpha_dir, ignore_errors=True)

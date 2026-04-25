"""Streaming FFmpeg readers/writers for in-memory video pipe I/O.

FFmpegFrameReader: decode frames via stdout pipe (no intermediate files)
FFmpegFrameWriter: encode frames via stdin pipe with hardware encode auto-detection
"""
from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

from .discovery import find_ffmpeg
from .extraction import detect_hwaccel
from .probe import probe_video

if TYPE_CHECKING:
    import numpy as np


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

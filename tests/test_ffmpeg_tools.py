"""Tests for backend.ffmpeg_tools probe and EXR filter selection."""
import io
import json
import subprocess
import sys
from pathlib import Path

import backend.ffmpeg_tools as ffmpeg_tools
from backend.ffmpeg_tools import discovery as _discovery
from backend.ffmpeg_tools import extraction as _extraction
from backend.ffmpeg_tools import probe as _probe


class TestProbeVideo:
    def test_probe_video_returns_color_metadata(self, monkeypatch):
        stream = {
            "codec_type": "video",
            "codec_name": "prores",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "25/1",
            "nb_frames": "100",
            "duration": "4.0",
            "pix_fmt": "yuv422p10le",
            "color_space": "bt709",
            "color_primaries": "bt709",
            "color_transfer": "unknown",
            "color_range": "tv",
            "chroma_location": "left",
            "bits_per_raw_sample": "10",
        }
        payload = json.dumps({"streams": [stream], "format": {"duration": "4.0"}})

        fake_validation = ffmpeg_tools.FFmpegValidationResult(
            ok=True,
            message="ok",
            ffmpeg_path="ffmpeg",
            ffprobe_path="ffprobe",
        )
        monkeypatch.setattr(
            _probe,
            "require_ffmpeg_install",
            lambda require_probe=True: fake_validation,
        )
        monkeypatch.setattr(
            _probe.subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args[0], 0, stdout=payload, stderr="",
            ),
        )

        info = ffmpeg_tools.probe_video("clip.mov")

        assert info["pix_fmt"] == "yuv422p10le"
        assert info["color_space"] == "bt709"
        assert info["color_primaries"] == "bt709"
        assert info["color_transfer"] == "unknown"
        assert info["color_range"] == "tv"
        assert info["chroma_location"] == "left"
        assert info["bits_per_raw_sample"] == 10


class TestBuildExrVf:
    def test_rgb_input_uses_format_only(self):
        vf = ffmpeg_tools.build_exr_vf({"pix_fmt": "gbrp10le"})
        assert vf == "format=gbrpf32le"

    def test_yuv_input_with_missing_transfer_uses_explicit_scale(self):
        vf = ffmpeg_tools.build_exr_vf({
            "pix_fmt": "yuv422p10le",
            "width": 1920,
            "height": 1080,
            "color_space": "bt709",
            "color_primaries": "bt709",
            "color_transfer": "",
            "color_range": "",
            "bits_per_raw_sample": 10,
        })

        assert (
            vf ==
            "scale=in_color_matrix=bt709:in_range=tv,format=gbrpf32le"
        )

    def test_complete_yuv_metadata_is_preserved(self):
        vf = ffmpeg_tools.build_exr_vf({
            "pix_fmt": "yuv420p10le",
            "width": 3840,
            "height": 2160,
            "color_space": "bt2020nc",
            "color_primaries": "bt2020",
            "color_transfer": "smpte2084",
            "color_range": "tv",
            "bits_per_raw_sample": 10,
        })

        assert (
            vf ==
            "scale=in_color_matrix=bt2020nc:in_range=tv,format=gbrpf32le"
        )

    def test_sd_missing_transfer_uses_sd_fallback(self):
        vf = ffmpeg_tools.build_exr_vf({
            "pix_fmt": "yuv420p",
            "width": 720,
            "height": 576,
            "color_space": "bt470bg",
            "color_primaries": "bt470bg",
            "color_transfer": "",
            "color_range": "tv",
            "bits_per_raw_sample": 8,
        })

        # bt470bg is remapped: matrix->bt601
        # (FFmpeg's scale filter doesn't accept 'bt470bg' as in_color_matrix)
        assert (
            vf ==
            "scale=in_color_matrix=bt601:in_range=tv,format=gbrpf32le"
        )


class TestExtractFrames:
    def test_extract_frames_uses_vf_chain_instead_of_pix_fmt(self, monkeypatch, tmp_path):
        commands = []

        class _FakeProc:
            def __init__(self, cmd):
                self.cmd = cmd
                self.stdin = None
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

            def kill(self):
                self.returncode = -9

        def fake_popen(cmd, **kwargs):
            commands.append(cmd)
            return _FakeProc(cmd)

        fake_validation = ffmpeg_tools.FFmpegValidationResult(
            ok=True,
            message="ok",
            ffmpeg_path="ffmpeg",
            ffprobe_path="ffprobe",
        )
        monkeypatch.setattr(
            _extraction,
            "require_ffmpeg_install",
            lambda require_probe=True: fake_validation,
        )
        monkeypatch.setattr(_extraction, "detect_hwaccel", lambda ffmpeg=None: [])
        monkeypatch.setattr(_extraction, "_recompress_to_dwab", lambda *args, **kwargs: None)
        monkeypatch.setattr(_extraction, "probe_video", lambda path: {
            "fps": 25.0,
            "width": 1920,
            "height": 1080,
            "frame_count": 10,
            "codec": "prores",
            "duration": 0.4,
            "pix_fmt": "yuv422p10le",
            "color_space": "bt709",
            "color_primaries": "bt709",
            "color_transfer": "",
            "color_range": "",
            "bits_per_raw_sample": 10,
        })
        monkeypatch.setattr(_extraction.subprocess, "Popen", fake_popen)

        out_dir = tmp_path / "frames"
        extracted = ffmpeg_tools.extract_frames("clip.mov", str(out_dir))

        assert extracted == 0
        assert len(commands) == 1
        assert "-vf" in commands[0]
        assert "-pix_fmt" not in commands[0]
        vf_index = commands[0].index("-vf")
        assert commands[0][vf_index + 1].startswith("scale=in_color_matrix=bt709")


class TestVideoMetadata:
    def test_write_and_read_preserves_probe_diagnostics(self, tmp_path):
        clip_root = tmp_path / "clip"
        clip_root.mkdir()
        metadata = {
            "source_path": "clip.mov",
            "fps": 25.0,
            "width": 1920,
            "height": 1080,
            "frame_count": 100,
            "codec": "prores",
            "duration": 4.0,
            "exr_vf": (
                "scale=in_color_matrix=bt709:in_range=tv,format=gbrpf32le"
            ),
            "source_probe": {
                "frame_count": 100,
                "pix_fmt": "yuv422p10le",
                "color_space": "bt709",
                "color_primaries": "bt709",
                "color_transfer": "",
                "color_range": "tv",
                "chroma_location": "left",
                "bits_per_raw_sample": 10,
            },
        }

        ffmpeg_tools.write_video_metadata(str(clip_root), metadata)
        loaded = ffmpeg_tools.read_video_metadata(str(clip_root))

        assert loaded == metadata


class TestValidateFFmpegInstall:
    def test_local_ffmpeg_is_preferred_over_path(self, monkeypatch):
        monkeypatch.setattr(_discovery, "_local_ffmpeg_binary", lambda name: f"/local/{name}")
        monkeypatch.setattr(_discovery.shutil, "which", lambda name: f"/path/{name}")

        assert ffmpeg_tools.find_ffmpeg() == "/local/ffmpeg"
        assert ffmpeg_tools.find_ffprobe() == "/local/ffprobe"

    def test_missing_ffprobe_is_rejected(self, monkeypatch):
        monkeypatch.setattr(_discovery, "find_ffmpeg", lambda: "ffmpeg")
        monkeypatch.setattr(_discovery, "find_ffprobe", lambda: None)

        result = ffmpeg_tools.validate_ffmpeg_install()

        assert not result.ok
        assert "FFprobe not found" in result.message

    def test_old_ffmpeg_is_rejected(self, monkeypatch):
        monkeypatch.setattr(_discovery, "find_ffmpeg", lambda: "ffmpeg")
        monkeypatch.setattr(_discovery, "find_ffprobe", lambda: "ffprobe")

        def fake_run(cmd, **kwargs):
            program = Path(cmd[0]).name
            first_line = f"{program} version 6.1.1"
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{first_line}\n", stderr="")

        monkeypatch.setattr(_discovery.subprocess, "run", fake_run)

        result = ffmpeg_tools.validate_ffmpeg_install()

        assert not result.ok
        assert "FFmpeg 7.0 or newer is required" in result.message

    def test_windows_essentials_build_is_rejected(self, monkeypatch):
        monkeypatch.setattr(_discovery, "find_ffmpeg", lambda: "ffmpeg.exe")
        monkeypatch.setattr(_discovery, "find_ffprobe", lambda: "ffprobe.exe")
        monkeypatch.setattr(_discovery.sys, "platform", "win32")

        def fake_run(cmd, **kwargs):
            program = Path(cmd[0]).name
            first_line = (
                f"{program} version 7.1.1-essentials_build-www.gyan.dev"
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{first_line}\n", stderr="")

        monkeypatch.setattr(_discovery.subprocess, "run", fake_run)

        result = ffmpeg_tools.validate_ffmpeg_install()

        assert not result.ok
        assert "full FFmpeg build" in result.message

    def test_dev_build_is_accepted(self, monkeypatch):
        monkeypatch.setattr(_discovery, "find_ffmpeg", lambda: "ffmpeg")
        monkeypatch.setattr(_discovery, "find_ffprobe", lambda: "ffprobe")

        def fake_run(cmd, **kwargs):
            program = Path(cmd[0]).name
            first_line = f"{program} version N-120000-gabcdef1234"
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{first_line}\n", stderr="")

        monkeypatch.setattr(_discovery.subprocess, "run", fake_run)

        result = ffmpeg_tools.validate_ffmpeg_install()

        assert result.ok
        assert "FFmpeg OK" in result.message

    def test_install_help_mentions_local_windows_repair(self, monkeypatch):
        monkeypatch.setattr(_discovery.sys, "platform", "win32")

        help_text = ffmpeg_tools.get_ffmpeg_install_help()

        assert "Repair FFmpeg" in help_text
        assert "tools\\ffmpeg" in help_text


class TestBuildExrVfGbrRegression:
    """Regression tests for the GBR-on-YUV matrix bug (issue #95 follow-up).

    ffprobe reports color_space=gbr for certain YUV ProRes clips.  "gbr" is a
    pixel-format family label, not a valid in_color_matrix constant.  FFmpeg
    rejects it with:
        Unable to parse "in_color_matrix" option value "gbr"
    The fix discards the bogus tag and falls back to the resolution heuristic.
    """

    _BASE = {
        "pix_fmt": "yuv422p10le",
        "width": 1920,
        "height": 1080,
        "color_primaries": "bt709",
        "color_transfer": "bt709",
        "color_range": "tv",
        "bits_per_raw_sample": 10,
    }

    def test_gbr_on_yuv_hd_falls_back_to_bt709(self):
        vf = ffmpeg_tools.build_exr_vf({**self._BASE, "color_space": "gbr"})
        assert "gbr" not in vf, "Bogus 'gbr' matrix must not reach the FFmpeg filter"
        assert "in_color_matrix=bt709" in vf

    def test_gbr_on_yuv_uhd_falls_back_to_bt709(self):
        info = {**self._BASE, "color_space": "gbr", "width": 3840, "height": 2160}
        vf = ffmpeg_tools.build_exr_vf(info)
        assert "gbr" not in vf
        assert "in_color_matrix=bt709" in vf

    def test_gbr_on_yuv_sd_falls_back_to_sd_matrix(self):
        """SD content with bogus GBR tag should use smpte170m, not bt709."""
        info = {
            "pix_fmt": "yuv422p",
            "width": 720,
            "height": 480,
            "color_space": "gbr",
            "color_primaries": "smpte170m",
            "color_transfer": "smpte170m",
            "color_range": "tv",
            "bits_per_raw_sample": 8,
        }
        vf = ffmpeg_tools.build_exr_vf(info)
        assert "gbr" not in vf
        assert "in_color_matrix=smpte170m" in vf

    def test_gbr_pix_fmt_on_rgb_stream_is_untouched(self):
        """gbrp* pixel formats are genuine RGB — no matrix needed."""
        vf = ffmpeg_tools.build_exr_vf({"pix_fmt": "gbrp10le", "color_space": "gbr"})
        assert vf == "format=gbrpf32le"

    def test_normal_yuv_bt709_unchanged(self):
        """Sanity-check: well-tagged YUV clips must not be affected by the fix."""
        vf = ffmpeg_tools.build_exr_vf({**self._BASE, "color_space": "bt709"})
        assert "in_color_matrix=bt709" in vf


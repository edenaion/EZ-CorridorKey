"""Tests for backend.ffmpeg_tools probe and EXR filter selection."""
import importlib.util
import io
import json
import subprocess
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "backend" / "ffmpeg_tools.py"
_SPEC = importlib.util.spec_from_file_location("test_ffmpeg_tools_module", _MODULE_PATH)
ffmpeg_tools = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(ffmpeg_tools)


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

        monkeypatch.setattr(ffmpeg_tools, "find_ffprobe", lambda: "ffprobe")
        monkeypatch.setattr(
            ffmpeg_tools.subprocess,
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
            "scale=in_color_matrix=bt709:in_primaries=bt709:"
            "in_transfer=bt709:in_range=tv,format=gbrpf32le"
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
            "scale=in_color_matrix=bt2020nc:in_primaries=bt2020:"
            "in_transfer=smpte2084:in_range=tv,format=gbrpf32le"
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

        # bt470bg is remapped: matrix→bt601, transfer→gamma28
        # (FFmpeg's scale filter doesn't accept 'bt470bg' as in_color_matrix)
        assert (
            vf ==
            "scale=in_color_matrix=bt601:in_primaries=bt470bg:"
            "in_transfer=gamma28:in_range=tv,format=gbrpf32le"
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

        monkeypatch.setattr(ffmpeg_tools, "find_ffmpeg", lambda: "ffmpeg")
        monkeypatch.setattr(ffmpeg_tools, "detect_hwaccel", lambda ffmpeg=None: [])
        monkeypatch.setattr(ffmpeg_tools, "_recompress_to_dwab", lambda *args, **kwargs: None)
        monkeypatch.setattr(ffmpeg_tools, "probe_video", lambda path: {
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
        monkeypatch.setattr(ffmpeg_tools.subprocess, "Popen", fake_popen)

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
                "scale=in_color_matrix=bt709:in_primaries=bt709:"
                "in_transfer=bt709:in_range=tv,format=gbrpf32le"
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

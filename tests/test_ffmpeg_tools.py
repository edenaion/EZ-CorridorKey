"""Tests for backend.ffmpeg_tools probe and EXR filter selection."""
import io
import json
import subprocess
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

        # bt2020nc must map to plain bt2020: the FFmpeg 8.x scale filter
        # rejects the ffprobe-style nc/ncl/c/cl variants (issue #91)
        assert (
            vf ==
            "scale=in_color_matrix=bt2020:in_range=tv,format=gbrpf32le"
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
        # Issue #175: -vsync was removed in FFmpeg git-master; we must use
        # -fps_mode:v passthrough instead, and never emit the legacy -vsync.
        assert "-fps_mode:v" in commands[0]
        fps_index = commands[0].index("-fps_mode:v")
        assert commands[0][fps_index + 1] == "passthrough"
        assert "-vsync" not in commands[0]

    def test_extract_frames_resume_branch_uses_fps_mode_not_vsync(self, monkeypatch, tmp_path):
        """The resume/seek branch (start_frame > 0) must also use -fps_mode:v."""
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
            ok=True, message="ok", ffmpeg_path="ffmpeg", ffprobe_path="ffprobe",
        )
        monkeypatch.setattr(
            _extraction, "require_ffmpeg_install",
            lambda require_probe=True: fake_validation,
        )
        monkeypatch.setattr(_extraction, "detect_hwaccel", lambda ffmpeg=None: [])
        monkeypatch.setattr(_extraction, "_recompress_to_dwab", lambda *a, **k: None)
        monkeypatch.setattr(_extraction, "probe_video", lambda path: {
            "fps": 24.0, "width": 1920, "height": 1080, "frame_count": 1000,
            "pix_fmt": "yuv420p", "color_space": "bt709",
            "color_primaries": "bt709", "color_transfer": "", "color_range": "",
        })
        monkeypatch.setattr(_extraction.subprocess, "Popen", fake_popen)

        out_dir = tmp_path / "frames"
        out_dir.mkdir()
        # Pre-seed many partial frames so resume rolls back yet start_frame > 0.
        for i in range(50):
            (out_dir / f"frame_{i:06d}.exr").write_bytes(b"x")

        ffmpeg_tools.extract_frames("clip.mov", str(out_dir), total_frames=1000)

        assert commands, "no FFmpeg command captured"
        cmd = commands[0]
        assert "-ss" in cmd, "resume branch should seek with -ss"
        assert "-fps_mode:v" in cmd
        assert cmd[cmd.index("-fps_mode:v") + 1] == "passthrough"
        assert "-vsync" not in cmd


class TestCorruptFrameRetry:
    """Issue #184: hardware-decode extraction can write corrupt EXR frames.
    The DWAB pass reports them; extract_frames must retry with software
    decode and fail loudly if frames are still unreadable."""

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

    def _setup(self, monkeypatch, out_dir, recompress_results):
        """Wire fakes: Popen writes dummy frames; recompress pops results."""
        commands = []

        def fake_popen(cmd, **kwargs):
            commands.append(cmd)
            for i in range(3):
                (out_dir / f"frame_{i:06d}.exr").write_bytes(b"x")
            return self._FakeProc(cmd)

        calls = list(recompress_results)

        def fake_recompress(dir_, on_progress=None, cancel_event=None):
            return calls.pop(0) if calls else []

        # First pass uses the pooled recompress; the safe-mode retry uses
        # the sequential fallback. Both pop from the same result queue.
        monkeypatch.setattr(_extraction, "_recompress_sequential",
                            fake_recompress)

        fake_validation = ffmpeg_tools.FFmpegValidationResult(
            ok=True, message="ok", ffmpeg_path="ffmpeg", ffprobe_path="ffprobe",
        )
        monkeypatch.setattr(
            _extraction, "require_ffmpeg_install",
            lambda require_probe=True: fake_validation,
        )
        monkeypatch.setattr(_extraction, "probe_video", lambda path: {
            "fps": 25.0, "width": 960, "height": 540, "frame_count": 3,
            "pix_fmt": "yuv420p", "color_space": "smpte170m",
            "color_primaries": "", "color_transfer": "", "color_range": "tv",
        })
        monkeypatch.setattr(_extraction, "_recompress_to_dwab", fake_recompress)
        monkeypatch.setattr(_extraction.subprocess, "Popen", fake_popen)
        # Hermetic: adaptive threading must not depend on this machine's
        # live commit headroom during unrelated tests.
        monkeypatch.setattr(_extraction, "_free_commit_bytes", lambda: None)
        return commands

    def test_corrupt_frames_trigger_software_retry(self, monkeypatch, tmp_path):
        out_dir = tmp_path / "frames"
        out_dir.mkdir()
        monkeypatch.setattr(
            _extraction, "detect_hwaccel",
            lambda ffmpeg=None: ["-hwaccel", "cuda"],
        )
        commands = self._setup(
            monkeypatch, out_dir,
            recompress_results=[["frame_000001.exr"], []],
        )

        extracted = ffmpeg_tools.extract_frames(
            "clip.mp4", str(out_dir), total_frames=3,
        )

        assert extracted == 3
        assert len(commands) == 2, "corrupt frames must trigger one retry"
        assert "-hwaccel" in commands[0]
        assert "-hwaccel" not in commands[1], "retry must use software decode"
        thr = commands[1].index("-threads:v")
        assert commands[1][thr + 1] == "1", "retry must single-thread the encode"

    def test_still_corrupt_after_retry_raises(self, monkeypatch, tmp_path):
        out_dir = tmp_path / "frames"
        out_dir.mkdir()
        monkeypatch.setattr(
            _extraction, "detect_hwaccel",
            lambda ffmpeg=None: ["-hwaccel", "cuda"],
        )
        self._setup(
            monkeypatch, out_dir,
            recompress_results=[["frame_000001.exr"], ["frame_000001.exr"]],
        )

        try:
            ffmpeg_tools.extract_frames("clip.mp4", str(out_dir), total_frames=3)
            assert False, "unreadable frames after retry must raise"
        except RuntimeError as exc:
            assert "unreadable" in str(exc)
            assert "Repair FFmpeg" in str(exc)

    def test_software_decode_corruption_also_retries(self, monkeypatch, tmp_path):
        # The encode-thread race is decode-independent, so a corrupt
        # software-decode pass still gets one single-threaded retry.
        out_dir = tmp_path / "frames"
        out_dir.mkdir()
        monkeypatch.setattr(_extraction, "detect_hwaccel", lambda ffmpeg=None: [])
        commands = self._setup(
            monkeypatch, out_dir,
            recompress_results=[["frame_000002.exr"], []],
        )

        extracted = ffmpeg_tools.extract_frames(
            "clip.mp4", str(out_dir), total_frames=3,
        )
        assert extracted == 3
        assert len(commands) == 2
        assert "-threads:v" in commands[1]

    def test_clean_extraction_no_retry(self, monkeypatch, tmp_path):
        out_dir = tmp_path / "frames"
        out_dir.mkdir()
        monkeypatch.setattr(
            _extraction, "detect_hwaccel",
            lambda ffmpeg=None: ["-hwaccel", "cuda"],
        )
        commands = self._setup(monkeypatch, out_dir, recompress_results=[[]])

        extracted = ffmpeg_tools.extract_frames(
            "clip.mp4", str(out_dir), total_frames=3,
        )
        assert extracted == 3
        assert len(commands) == 1

    def test_hw_decode_failure_on_resume_restarts_from_zero(
        self, monkeypatch, tmp_path,
    ):
        """Hardware-decode failure wipes ALL frames (including the resume
        prefix), so the software retry must restart from frame 0 — not seek
        to the old resume point and silently drop the prefix."""
        out_dir = tmp_path / "frames"
        out_dir.mkdir()
        # Pre-seed frames so the resume branch picks start_frame > 0
        for i in range(20):
            (out_dir / f"frame_{i:06d}.exr").write_bytes(b"x")

        monkeypatch.setattr(
            _extraction, "detect_hwaccel",
            lambda ffmpeg=None: ["-hwaccel", "cuda"],
        )
        commands = self._setup(monkeypatch, out_dir, recompress_results=[[]])

        # First (hardware) run fails, software retry succeeds
        outer = self

        def failing_first_popen(cmd, **kwargs):
            commands.append(cmd)
            proc = outer._FakeProc(cmd)
            if "-hwaccel" in cmd:
                proc.returncode = 1

                def poll():
                    return 1
                proc.poll = poll
            else:
                for i in range(3):
                    (out_dir / f"frame_{i:06d}.exr").write_bytes(b"x")
            return proc

        monkeypatch.setattr(_extraction.subprocess, "Popen", failing_first_popen)

        ffmpeg_tools.extract_frames(
            "clip.mp4", str(out_dir), total_frames=1000,
        )

        assert len(commands) == 2
        assert "-ss" in commands[0], "resume branch should seek initially"
        assert "-ss" not in commands[1], "software retry must not seek"
        sn = commands[1].index("-start_number")
        assert commands[1][sn + 1] == "0", "software retry must restart at 0"


class TestAdaptiveEncoderThreads:
    """Issue #184: EXR encode is frame-threaded (~52 B/px/thread measured);
    cap threads to commit headroom so 4K imports survive loaded machines."""

    def test_no_cap_when_headroom_large(self, monkeypatch):
        monkeypatch.setattr(_extraction, "_free_commit_bytes",
                            lambda: 64 * 1024**3)
        monkeypatch.setattr(_extraction.os, "cpu_count", lambda: 16)
        assert _extraction.adaptive_encoder_threads(3840, 2160) == []

    def test_caps_when_headroom_small(self, monkeypatch):
        # 4 GB free, 4K: afford = 4GB*0.5 / (52*3840*2160 B ~ 431 MB) = 4
        monkeypatch.setattr(_extraction, "_free_commit_bytes",
                            lambda: 4 * 1024**3)
        monkeypatch.setattr(_extraction.os, "cpu_count", lambda: 32)
        args = _extraction.adaptive_encoder_threads(3840, 2160)
        assert args[0] == "-threads:v"
        assert 2 <= int(args[1]) < 32

    def test_floor_of_two_threads(self, monkeypatch):
        monkeypatch.setattr(_extraction, "_free_commit_bytes",
                            lambda: 256 * 1024**2)  # 256 MB free
        monkeypatch.setattr(_extraction.os, "cpu_count", lambda: 8)
        args = _extraction.adaptive_encoder_threads(3840, 2160)
        assert args == ["-threads:v", "2"]

    def test_no_probe_no_cap(self, monkeypatch):
        monkeypatch.setattr(_extraction, "_free_commit_bytes", lambda: None)
        assert _extraction.adaptive_encoder_threads(3840, 2160) == []

    def test_small_resolution_uncapped_on_modest_machine(self, monkeypatch):
        # 1080p on 6 GB free, 8 cores: per-thread ~108 MB, afford ~27 > cores
        monkeypatch.setattr(_extraction, "_free_commit_bytes",
                            lambda: 6 * 1024**3)
        monkeypatch.setattr(_extraction.os, "cpu_count", lambda: 8)
        assert _extraction.adaptive_encoder_threads(1920, 1080) == []

    def test_cap_args_reach_ffmpeg_command(self, monkeypatch, tmp_path):
        out_dir = tmp_path / "frames"
        out_dir.mkdir()
        commands = []

        class _P(TestCorruptFrameRetry._FakeProc):
            pass

        def fake_popen(cmd, **kwargs):
            commands.append(cmd)
            for i in range(3):
                (out_dir / f"frame_{i:06d}.exr").write_bytes(b"x")
            return _P(cmd)

        fake_validation = ffmpeg_tools.FFmpegValidationResult(
            ok=True, message="ok", ffmpeg_path="ffmpeg", ffprobe_path="ffprobe",
        )
        monkeypatch.setattr(
            _extraction, "require_ffmpeg_install",
            lambda require_probe=True: fake_validation,
        )
        monkeypatch.setattr(_extraction, "probe_video", lambda path: {
            "fps": 25.0, "width": 3840, "height": 2160, "frame_count": 3,
            "pix_fmt": "yuv420p", "color_space": "bt709",
            "color_primaries": "", "color_transfer": "", "color_range": "tv",
        })
        monkeypatch.setattr(_extraction, "detect_hwaccel", lambda ffmpeg=None: [])
        monkeypatch.setattr(_extraction, "_recompress_to_dwab",
                            lambda *a, **k: [])
        monkeypatch.setattr(_extraction.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(_extraction, "adaptive_encoder_threads",
                            lambda w, h: ["-threads:v", "6"])

        ffmpeg_tools.extract_frames("clip.mp4", str(out_dir), total_frames=3)
        thr = commands[0].index("-threads:v")
        assert commands[0][thr + 1] == "6"


class TestOutOfMemoryMessage:
    """FFmpeg -12 'Cannot allocate memory' must surface as a human message,
    not a raw codec error (issue #184 QA)."""

    def test_oom_stderr_raises_human_message(self, monkeypatch, tmp_path):
        out_dir = tmp_path / "frames"
        out_dir.mkdir()

        class _OOMProc:
            def __init__(self, cmd, **kwargs):
                self.stdin = None
                self.stderr = io.StringIO(
                    "[vf#0:0] Error while filtering: Cannot allocate memory\n"
                )
                self.returncode = 1

            def poll(self):
                return 1

            def wait(self, timeout=None):
                return 1

            def kill(self):
                pass

        fake_validation = ffmpeg_tools.FFmpegValidationResult(
            ok=True, message="ok", ffmpeg_path="ffmpeg", ffprobe_path="ffprobe",
        )
        monkeypatch.setattr(
            _extraction, "require_ffmpeg_install",
            lambda require_probe=True: fake_validation,
        )
        monkeypatch.setattr(_extraction, "probe_video", lambda path: {
            "fps": 25.0, "width": 3840, "height": 2160, "frame_count": 3,
            "pix_fmt": "yuv420p", "color_space": "bt709",
            "color_primaries": "", "color_transfer": "", "color_range": "tv",
        })
        monkeypatch.setattr(_extraction, "detect_hwaccel", lambda ffmpeg=None: [])
        monkeypatch.setattr(_extraction, "_free_commit_bytes", lambda: None)
        monkeypatch.setattr(_extraction.subprocess, "Popen", _OOMProc)

        try:
            ffmpeg_tools.extract_frames("clip.mp4", str(out_dir), total_frames=3)
            assert False, "OOM must raise"
        except RuntimeError as exc:
            assert "Not enough free memory" in str(exc)
            assert "Close other applications" in str(exc)


class TestRecompressSubprocessFailure:
    """A crashed or timed-out recompress subprocess verifies nothing — it
    must report every frame as suspect, never return 'clean'."""

    class _DeadProc:
        def __init__(self, cmd, **kwargs):
            self.stdout = io.StringIO("")   # no PROGRESS/DONE ever printed
            self.stderr = io.StringIO("Traceback: worker pool exploded")
            self.returncode = 1

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

        def kill(self):
            pass

    def test_nonzero_exit_reports_all_frames_suspect(self, monkeypatch, tmp_path):
        out_dir = tmp_path / "frames"
        out_dir.mkdir()
        files = []
        for i in range(4):
            name = f"frame_{i:06d}.exr"
            (out_dir / name).write_bytes(b"x")
            files.append(name)

        monkeypatch.setattr(_extraction.subprocess, "Popen", self._DeadProc)

        failed = _extraction._recompress_subprocess(
            str(out_dir), files, len(files),
            str(out_dir / ".dwab_done"),
        )
        assert failed == files, "incomplete pass must mark all frames suspect"
        assert not (out_dir / ".dwab_done").exists()


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


def _fake_version_run(monkeypatch, banner_by_program):
    """Patch subprocess.run so each program reports the given banner line.

    banner_by_program: {"ffmpeg": "<first line>", "ffprobe": "<first line>"}
    """
    def fake_run(cmd, **kwargs):
        program = Path(cmd[0]).name.replace(".exe", "")
        first_line = banner_by_program[program]
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{first_line}\n", stderr="")

    monkeypatch.setattr(_discovery.subprocess, "run", fake_run)


def _validate_with_banner(monkeypatch, ffmpeg_line, ffprobe_line=None, platform="win32"):
    monkeypatch.setattr(_discovery, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(_discovery, "find_ffprobe", lambda: "ffprobe")
    monkeypatch.setattr(_discovery.sys, "platform", platform)
    _fake_version_run(monkeypatch, {
        "ffmpeg": ffmpeg_line,
        "ffprobe": ffprobe_line or ffmpeg_line.replace("ffmpeg", "ffprobe", 1),
    })
    return ffmpeg_tools.validate_ffmpeg_install()


# Real -version first lines captured from actual binaries (e2e matrix,
# 2026-07-11). The gyan git line is the exact build from the issue #184
# regression report.
PINNED_LINE = "ffmpeg version n8.1.2-20260627 Copyright (c) 2000-2026 the FFmpeg developers"
GYAN_GIT_LINE = ("ffmpeg version 2026-07-09-git-8de8405796-full_build-www.gyan.dev "
                 "Copyright (c) 2000-2026 the FFmpeg developers")
GYAN_FULL_LINE = ("ffmpeg version 8.1.2-full_build-www.gyan.dev "
                  "Copyright (c) 2000-2026 the FFmpeg developers")
GYAN_ESSENTIALS_LINE = ("ffmpeg version 8.1.2-essentials_build-www.gyan.dev "
                        "Copyright (c) 2000-2026 the FFmpeg developers")
BTBN_MASTER_LINE = ("ffmpeg version N-125519-g300cac3078-20260711 "
                    "Copyright (c) 2000-2026 the FFmpeg developers")
BTBN_POST_TAG_LINE = ("ffmpeg version n8.1.2-22-g94138f6973-20260710 "
                      "Copyright (c) 2000-2026 the FFmpeg developers")
OLD_LINE = ("ffmpeg version 6.1.1-full_build-www.gyan.dev "
            "Copyright (c) 2000-2023 the FFmpeg developers")


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

    def test_path_is_used_when_no_bundled_build(self, monkeypatch):
        # Option B: with no bundled build and no custom dir, discovery falls
        # through to the system PATH. Validation then decides accept/reject.
        monkeypatch.setattr(_discovery, "_local_ffmpeg_binary", lambda name: None)
        monkeypatch.setattr(_discovery, "_get_custom_ffmpeg_dir", lambda: None)
        monkeypatch.setattr(_discovery.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert ffmpeg_tools.find_ffmpeg() == "/usr/bin/ffmpeg"
        assert ffmpeg_tools.find_ffprobe() == "/usr/bin/ffprobe"

    def test_install_help_mentions_local_windows_repair(self, monkeypatch):
        monkeypatch.setattr(_discovery.sys, "platform", "win32")

        help_text = ffmpeg_tools.get_ffmpeg_install_help()

        assert "Repair FFmpeg" in help_text
        assert "tools\\ffmpeg" in help_text


class TestWindowsVerifiedBuildPolicy:
    """Windows accepts ONLY verified tag-exact 8.1.2 full builds (issue
    #184, twice). Both packagers of the same source tag pass, so a user
    holding either is not forced to switch. Every other build must fail
    validation with known_bad=True and be routed to Repair.
    """

    def test_btbn_tag_build_passes(self, monkeypatch):
        result = _validate_with_banner(monkeypatch, PINNED_LINE)
        assert result.ok
        assert not result.known_bad

    def test_btbn_tag_without_date_suffix_passes(self, monkeypatch):
        result = _validate_with_banner(
            monkeypatch,
            "ffmpeg version n8.1.2 Copyright (c) 2000-2026 the FFmpeg developers",
        )
        assert result.ok

    def test_gyan_full_812_passes(self, monkeypatch):
        # Same source tag as the BtbN build; also what Repair now installs.
        result = _validate_with_banner(monkeypatch, GYAN_FULL_LINE)
        assert result.ok
        assert not result.known_bad

    def test_gyan_full_shared_is_rejected(self, monkeypatch):
        # The shared build was never run through the operation matrix, so it
        # is not on the allowlist even though it is the same tag.
        result = _validate_with_banner(
            monkeypatch,
            "ffmpeg version 8.1.2-full_build-shared-www.gyan.dev Copyright (c) 2026",
        )
        assert not result.ok
        assert result.known_bad

    def test_unverified_future_release_is_rejected(self, monkeypatch):
        # 8.1.3 may be fine, but it has not been through the operation
        # matrix. Nothing joins the allowlist without being tested.
        result = _validate_with_banner(
            monkeypatch,
            "ffmpeg version 8.1.3-full_build-www.gyan.dev Copyright (c) 2026",
        )
        assert not result.ok
        assert result.known_bad

    def test_gyan_git_nightly_is_rejected(self, monkeypatch):
        # The exact build from the regression report: parsed as release
        # "2026" by the old code, which let Repair claim nothing was wrong
        # while export died on the removed -apply_trc option.
        result = _validate_with_banner(monkeypatch, GYAN_GIT_LINE)
        assert not result.ok
        assert result.known_bad
        assert "Repair FFmpeg" in result.message

    def test_gyan_essentials_is_rejected(self, monkeypatch):
        result = _validate_with_banner(monkeypatch, GYAN_ESSENTIALS_LINE)
        assert not result.ok
        assert result.known_bad

    def test_btbn_master_nightly_is_rejected(self, monkeypatch):
        result = _validate_with_banner(monkeypatch, BTBN_MASTER_LINE)
        assert not result.ok
        assert result.known_bad

    def test_btbn_post_tag_rolling_is_rejected(self, monkeypatch):
        # Post-tag n8.1.2 rolling builds carry the issue #184 CUDA race and
        # must not match the pin regex despite sharing the tag prefix.
        result = _validate_with_banner(monkeypatch, BTBN_POST_TAG_LINE)
        assert not result.ok
        assert result.known_bad

    def test_old_stable_is_rejected(self, monkeypatch):
        result = _validate_with_banner(monkeypatch, OLD_LINE)
        assert not result.ok
        assert result.known_bad

    def test_pinned_ffmpeg_with_foreign_ffprobe_is_rejected(self, monkeypatch):
        # Mixed installs (bundled ffmpeg, PATH ffprobe) are still broken.
        result = _validate_with_banner(
            monkeypatch, PINNED_LINE,
            ffprobe_line=GYAN_GIT_LINE.replace("ffmpeg", "ffprobe", 1),
        )
        assert not result.ok
        assert result.known_bad
        assert "FFprobe" in result.message


class TestUnixStableAllowlist:
    """macOS/Linux cannot pin (brew/apt); tagged stable >= 7 accepted,
    dev/unknown/old rejected."""

    def test_stable_release_passes(self, monkeypatch):
        result = _validate_with_banner(
            monkeypatch, "ffmpeg version 8.0 Copyright (c) 2000-2026",
            platform="darwin",
        )
        assert result.ok

    def test_distro_suffix_passes(self, monkeypatch):
        result = _validate_with_banner(
            monkeypatch, "ffmpeg version 7.1.1-1ubuntu1 Copyright (c) 2000-2025",
            platform="linux",
        )
        assert result.ok

    def test_old_release_is_rejected(self, monkeypatch):
        result = _validate_with_banner(
            monkeypatch, "ffmpeg version 6.1.1 Copyright (c) 2000-2023",
            platform="darwin",
        )
        assert not result.ok
        assert "7.0 or newer" in result.message

    def test_dev_build_is_rejected(self, monkeypatch):
        result = _validate_with_banner(
            monkeypatch, BTBN_MASTER_LINE, platform="darwin",
        )
        assert not result.ok
        assert result.known_bad

    def test_major_mismatch_is_rejected(self, monkeypatch):
        result = _validate_with_banner(
            monkeypatch, "ffmpeg version 8.0 Copyright",
            ffprobe_line="ffprobe version 7.1 Copyright",
            platform="darwin",
        )
        assert not result.ok
        assert "do not match" in result.message


class TestPinnedFFmpegAsset:
    """Issue #184: repair must install the pinned verified build, never a
    rolling URL whose contents change under us."""

    def test_resolver_returns_pinned_asset(self):
        name, url = _discovery._resolve_windows_ffmpeg_asset(None)
        assert name == _discovery._PINNED_ASSET
        assert url == _discovery._PINNED_URL

    def test_pin_is_a_versioned_immutable_release(self):
        # A rolling URL re-publishes different binaries under the same
        # address, so the pin must point at a versioned release asset, and it
        # must be a zip (Repair extracts with stdlib zipfile).
        assert "/latest/" not in _discovery._PINNED_URL
        assert "/download/8.1.2/" in _discovery._PINNED_URL
        assert "git" not in _discovery._PINNED_ASSET
        assert "essentials" not in _discovery._PINNED_ASSET
        assert _discovery._PINNED_ASSET.endswith(".zip")

    def test_pinned_asset_version_is_on_the_verified_list(self):
        # Pin and allowlist must stay in sync: the banner token the pinned
        # asset produces has to satisfy a verified-build regex.
        import re as _re
        m = _re.search(r"ffmpeg-(\d+\.\d+\.\d+)-full_build", _discovery._PINNED_ASSET)
        assert m, _discovery._PINNED_ASSET
        token = f"{m.group(1)}-full_build-www.gyan.dev"
        assert any(rx.match(token) for rx in _discovery._VERIFIED_BUILD_TOKEN_RES)

    def test_pin_url_ends_with_pin_asset(self):
        # 1-install.bat derives the temp zip name from the asset; the URL
        # must actually serve that asset.
        assert _discovery._PINNED_URL.endswith(_discovery._PINNED_ASSET)

    def test_installer_pin_printer_matches_discovery(self):
        # scripts/check_ffmpeg.py --print-pin-url is the installer's single
        # source of truth; it must echo discovery's constants verbatim.
        import importlib.util as _ilu
        from pathlib import Path as _P
        script = _P(_discovery.__file__).resolve().parents[2] / "scripts" / "check_ffmpeg.py"
        spec = _ilu.spec_from_file_location("corridorkey_check_ffmpeg", script)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tools = mod._load_ffmpeg_tools()
        assert tools._PINNED_URL == _discovery._PINNED_URL
        assert tools._PINNED_ASSET == _discovery._PINNED_ASSET

    def test_resolver_needs_no_network(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("pinned resolver must not touch the network")

        monkeypatch.setattr(_discovery.urllib.request, "urlopen", boom)
        name, url = _discovery._resolve_windows_ffmpeg_asset(None)
        assert name == _discovery._PINNED_ASSET


class TestVersionTokenClassification:
    """The parser must never mistake a snapshot build for a release.

    The issue #184 regression: "2026-07-09-git-..." parsed as release major
    2026 and sailed through validation."""

    def _parse(self, monkeypatch, line):
        _fake_version_run(monkeypatch, {"ffmpeg": line})
        return _discovery._read_program_version("ffmpeg", "ffmpeg")

    def test_release_tokens(self, monkeypatch):
        cases = [
            (PINNED_LINE, 8),
            (GYAN_FULL_LINE, 8),
            (GYAN_ESSENTIALS_LINE, 8),
            (OLD_LINE, 6),
            ("ffmpeg version 7.1.1", 7),
            ("ffmpeg version n7.1-latest", 7),
            ("ffmpeg version 7.1.1-1ubuntu1", 7),
            ("ffmpeg version 8.0", 8),
        ]
        for line, major in cases:
            info = self._parse(monkeypatch, line)
            assert info.major == major, line
            assert not info.is_dev_build and not info.is_unknown, line

    def test_dev_tokens(self, monkeypatch):
        for line in (
            GYAN_GIT_LINE,
            BTBN_MASTER_LINE,
            BTBN_POST_TAG_LINE,
            "ffmpeg version N-125505-gc57660fb18-20260708",
            "ffmpeg version git-2026-07-01-abcdef",
            "ffmpeg version master-20260630",
            "ffmpeg version n8.1.2-3-gdba917bab5-20260629",
        ):
            info = self._parse(monkeypatch, line)
            assert info.is_dev_build, line
            assert info.major is None, line

    def test_unknown_token(self, monkeypatch):
        info = self._parse(monkeypatch, "ffmpeg version customvendorbuild")
        assert info.is_unknown
        assert info.major is None

    def test_verified_detection(self, monkeypatch):
        assert self._parse(monkeypatch, PINNED_LINE).is_verified_build
        assert self._parse(
            monkeypatch, "ffmpeg version n8.1.2 Copyright",
        ).is_verified_build
        assert self._parse(monkeypatch, GYAN_FULL_LINE).is_verified_build
        # Post-tag rolling build shares the tag prefix but is NOT verified
        assert not self._parse(monkeypatch, BTBN_POST_TAG_LINE).is_verified_build
        assert not self._parse(monkeypatch, GYAN_ESSENTIALS_LINE).is_verified_build
        assert not self._parse(monkeypatch, GYAN_GIT_LINE).is_verified_build


class TestRepairSafetyHelpers:
    """Repair must validate staging and never strand a working FFmpeg."""

    def test_safe_extract_rejects_zip_slip(self, tmp_path):
        import zipfile as _zip
        bad = tmp_path / "evil.zip"
        with _zip.ZipFile(bad, "w") as z:
            z.writestr("../escape.txt", "pwned")
        dest = tmp_path / "out"
        dest.mkdir()
        try:
            _discovery._safe_extract_zip(str(bad), str(dest))
            assert False, "zip-slip member should have been rejected"
        except RuntimeError as exc:
            assert "unsafe path" in str(exc)
        assert not (tmp_path / "escape.txt").exists()

    def test_find_staged_dir_selects_by_contents(self, tmp_path):
        root = tmp_path / "extract" / "ffmpeg-n8.1-latest-win64-gpl-8.1" / "bin"
        root.mkdir(parents=True)
        (root / "ffmpeg.exe").write_bytes(b"x")
        (root / "ffprobe.exe").write_bytes(b"x")
        found = _discovery._find_staged_ffmpeg_dir(str(tmp_path / "extract"))
        assert found is not None
        assert found.endswith("ffmpeg-n8.1-latest-win64-gpl-8.1")

    def test_staged_validation_rejects_unverified_build(self, tmp_path, monkeypatch):
        # A staged download that is not a verified build must never be
        # swapped into tools/ffmpeg (swapped or tampered download).
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "ffmpeg.exe").write_bytes(b"x")
        (bin_dir / "ffprobe.exe").write_bytes(b"x")
        _fake_version_run(monkeypatch, {
            "ffmpeg": GYAN_GIT_LINE,
            "ffprobe": GYAN_GIT_LINE.replace("ffmpeg", "ffprobe", 1),
        })
        try:
            _discovery._validate_staged_ffmpeg(str(tmp_path))
            assert False, "unverified staged build should be rejected"
        except RuntimeError as exc:
            assert "pinned" in str(exc)

    def test_staged_validation_accepts_pin(self, tmp_path, monkeypatch):
        # GYAN_FULL_LINE is the banner of the pinned Repair asset.
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "ffmpeg.exe").write_bytes(b"x")
        (bin_dir / "ffprobe.exe").write_bytes(b"x")
        _fake_version_run(monkeypatch, {
            "ffmpeg": GYAN_FULL_LINE,
            "ffprobe": GYAN_FULL_LINE.replace("ffmpeg", "ffprobe", 1),
        })
        _discovery._validate_staged_ffmpeg(str(tmp_path))

    def test_find_staged_dir_none_when_binaries_missing(self, tmp_path):
        (tmp_path / "extract").mkdir()
        assert _discovery._find_staged_ffmpeg_dir(str(tmp_path / "extract")) is None

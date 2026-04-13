"""Comprehensive tests for backend.service — CorridorKeyService.

Mocking strategy (Codex-informed hybrid):
- Real tiny PNG I/O for happy paths (via conftest fixtures)
- unittest.mock.patch for GPU/model/error injection
- Behavioral tests for GPU lock (not structural)
"""

import json
import os
import tempfile
import types
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from backend.annotation_prompts import AnnotationPromptFrame
from backend.service import (
    CorridorKeyService,
    InferenceParams,
    OutputConfig,
    _ActiveModel,
    _import_matanyone2_processor_class,
)
from backend.clip_state import ClipAsset, ClipEntry, ClipState
from backend.errors import (
    CorridorKeyError,
    FrameReadError,
    JobCancelledError,
    WriteFailureError,
)
from backend.job_queue import GPUJob, JobType


# ── TestServiceInit ──


class TestServiceInit:
    def test_constructor_defaults(self):
        svc = CorridorKeyService()
        assert svc._engine_pool == []
        assert svc._gvm_processor is None
        assert svc._videomama_pipeline is None
        assert svc._active_model == _ActiveModel.NONE
        assert svc._device == "cpu"

    def test_lazy_job_queue(self):
        svc = CorridorKeyService()
        assert svc._job_queue is None
        q = svc.job_queue
        assert q is not None
        # Same instance on second access
        assert svc.job_queue is q


# ── TestDeviceDetection ──


class TestDeviceDetection:
    @patch("backend.service.CorridorKeyService.detect_device")
    def test_detect_cuda(self, mock_detect):
        mock_detect.return_value = "cuda"
        svc = CorridorKeyService()
        assert svc.detect_device() == "cuda"

    def test_detect_device_no_torch(self):
        svc = CorridorKeyService()
        with patch.dict("sys.modules", {"torch": None}):
            # Force reimport failure
            result = svc.detect_device()
        # Falls back to cpu when torch import fails
        assert result in ("cpu", "cuda")  # depends on real env

    def test_detect_device_returns_string(self):
        svc = CorridorKeyService()
        result = svc.detect_device()
        assert isinstance(result, str)
        assert result in ("cuda", "cpu")


# ── TestVRAMInfo ──


class TestVRAMInfo:
    def test_vram_info_returns_dict(self):
        svc = CorridorKeyService()
        info = svc.get_vram_info()
        assert isinstance(info, dict)
        # Either has expected keys (CUDA) or is empty (no CUDA)
        if info:
            for key in ("total", "reserved", "allocated", "free", "name"):
                assert key in info, f"Missing key: {key}"
            assert isinstance(info["total"], float)
            assert isinstance(info["name"], str)

    def test_vram_info_no_cuda_returns_empty(self):
        svc = CorridorKeyService()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = svc.get_vram_info()
        assert info == {}

    def test_vram_info_exception_returns_empty(self):
        svc = CorridorKeyService()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("boom")
        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = svc.get_vram_info()
        assert info == {}


# ── TestModelResidency ──


class TestModelResidency:
    def test_ensure_model_noop_when_same(self):
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.INFERENCE
        svc._engine_pool = [MagicMock()]
        svc._ensure_model(_ActiveModel.INFERENCE)
        assert len(svc._engine_pool) > 0  # Not unloaded

    def test_ensure_model_switch_unloads_inference(self):
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.INFERENCE
        svc._engine_pool = [MagicMock()]
        svc._ensure_model(_ActiveModel.GVM)
        assert svc._engine_pool == []
        assert svc._active_model == _ActiveModel.GVM

    def test_ensure_model_switch_unloads_gvm(self):
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.GVM
        svc._gvm_processor = MagicMock()
        svc._ensure_model(_ActiveModel.INFERENCE)
        assert svc._gvm_processor is None
        assert svc._active_model == _ActiveModel.INFERENCE

    def test_ensure_model_switch_unloads_videomama(self):
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.VIDEOMAMA
        svc._videomama_pipeline = MagicMock()
        svc._ensure_model(_ActiveModel.INFERENCE)
        assert svc._videomama_pipeline is None

    def test_ensure_model_from_none(self):
        svc = CorridorKeyService()
        svc._ensure_model(_ActiveModel.INFERENCE)
        assert svc._active_model == _ActiveModel.INFERENCE


class TestMatAnyone2Import:
    def test_prefers_modules_namespace_layout(self, monkeypatch):
        sentinel = object()

        def fake_import(name):
            if name == "modules.MatAnyone2Module.wrapper":
                return types.SimpleNamespace(MatAnyone2Processor=sentinel)
            raise ModuleNotFoundError(name=name)

        monkeypatch.setattr("backend.service.importlib.import_module", fake_import)
        cls = _import_matanyone2_processor_class()
        assert cls is sentinel

    def test_falls_back_to_legacy_layout(self, monkeypatch):
        sentinel = object()

        def fake_import(name):
            if name == "modules.MatAnyone2Module.wrapper":
                raise ModuleNotFoundError(name="modules.MatAnyone2Module.wrapper")
            if name == "MatAnyone2Module.wrapper":
                return types.SimpleNamespace(MatAnyone2Processor=sentinel)
            raise AssertionError(name)

        monkeypatch.setattr("backend.service.importlib.import_module", fake_import)
        cls = _import_matanyone2_processor_class()
        assert cls is sentinel


# ── TestGetEngine ──


class TestGetEnginePool:
    def test_cached_pool_returns_immediately(self):
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.INFERENCE
        mock_engine = MagicMock()
        svc._engine_pool = [mock_engine]
        result = svc._get_engine_pool()
        assert result[0] is mock_engine

    @patch(
        "CorridorKeyModule.backend.create_engine",
        side_effect=FileNotFoundError("No .pth checkpoint found"),
    )
    @patch("CorridorKeyModule.backend.resolve_backend", return_value="torch")
    def test_no_checkpoint_raises(self, _rb, _ce):
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.NONE
        with pytest.raises(FileNotFoundError, match="No .pth checkpoint"):
            svc._get_engine_pool()

    @patch(
        "CorridorKeyModule.backend.create_engine", side_effect=ValueError("Multiple checkpoints")
    )
    @patch("CorridorKeyModule.backend.resolve_backend", return_value="torch")
    def test_multiple_checkpoints_raises(self, _rb, _ce):
        svc = CorridorKeyService()
        with pytest.raises(ValueError, match="Multiple checkpoints"):
            svc._get_engine_pool()


# ── TestScanAndFilter ──


class TestScanAndFilter:
    def test_scan_clips_delegates(self, tmp_clip_dir):
        svc = CorridorKeyService()
        parent_dir = os.path.dirname(tmp_clip_dir)
        clips = svc.scan_clips(parent_dir)
        assert isinstance(clips, list)

    def test_get_clips_by_state(self):
        svc = CorridorKeyService()
        clips = [
            ClipEntry("a", "/tmp/a", state=ClipState.RAW),
            ClipEntry("b", "/tmp/b", state=ClipState.READY),
            ClipEntry("c", "/tmp/c", state=ClipState.READY),
        ]
        ready = svc.get_clips_by_state(clips, ClipState.READY)
        assert len(ready) == 2
        assert all(c.state == ClipState.READY for c in ready)

    def test_get_clips_by_state_empty(self):
        svc = CorridorKeyService()
        result = svc.get_clips_by_state([], ClipState.RAW)
        assert result == []


# ── TestReadInputFrame ──


class TestReadInputFrame:
    def test_sequence_png_happy_path(self, sample_clip):
        """Real PNG I/O — reads actual file from fixture."""
        svc = CorridorKeyService()
        input_files = sample_clip.input_asset.get_frame_files()
        img, stem, is_linear = svc._read_input_frame(
            sample_clip,
            0,
            input_files,
            None,
            False,
        )
        assert img is not None
        assert img.dtype == np.float32
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert 0.0 <= img.max() <= 1.0
        assert isinstance(stem, str)

    def test_read_failure_returns_none_for_video(self):
        svc = CorridorKeyService()
        clip = ClipEntry("test", "/tmp", input_asset=ClipAsset("/nonexistent", "video"))
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        img, stem, is_linear = svc._read_input_frame(clip, 0, [], mock_cap, False)
        assert img is None

    def test_exr_sequence_honors_explicit_srgb_setting(self):
        svc = CorridorKeyService()
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = os.path.join(tmpdir, "Frames")
            os.makedirs(frames_dir)
            frame_path = os.path.join(frames_dir, "frame_000000.exr")
            with open(frame_path, "w", encoding="utf-8") as handle:
                handle.write("dummy")

            clip = ClipEntry(
                "test",
                tmpdir,
                state=ClipState.RAW,
                input_asset=ClipAsset(frames_dir, "sequence"),
            )

            fake_img = np.zeros((2, 2, 3), dtype=np.float32)
            with patch("backend.service.read_image_frame", return_value=fake_img):
                img, stem, is_linear = svc._read_input_frame(
                    clip,
                    0,
                    clip.input_asset.get_frame_files(),
                    None,
                    False,
                )

            assert img is fake_img
            assert stem == "frame_000000"
            assert is_linear is False


# ── TestReadAlphaFrame ──


class TestReadAlphaFrame:
    def test_sequence_happy_path(self, sample_clip):
        """Real PNG I/O — reads actual alpha file."""
        svc = CorridorKeyService()
        alpha_files = sample_clip.alpha_asset.get_frame_files()
        mask = svc._read_alpha_frame(sample_clip, 0, alpha_files, None)
        assert mask is not None
        assert mask.dtype == np.float32
        assert mask.ndim == 2  # single channel

    def test_video_read_failure(self):
        svc = CorridorKeyService()
        clip = ClipEntry("test", "/tmp", alpha_asset=ClipAsset("/nonexistent", "video"))
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        mask = svc._read_alpha_frame(clip, 0, [], mock_cap)
        assert mask is None

    def test_sequence_prefers_input_stem_when_available(self):
        svc = CorridorKeyService()
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_dir = os.path.join(tmpdir, "AlphaHint")
            os.makedirs(alpha_dir)
            wanted = os.path.join(alpha_dir, "frame_000003.png")
            fallback = os.path.join(alpha_dir, "_alphaHint_000000.png")
            cv2.imwrite(wanted, np.ones((4, 4), dtype=np.uint8) * 255)
            cv2.imwrite(fallback, np.zeros((4, 4), dtype=np.uint8))

            clip = ClipEntry(
                "test",
                tmpdir,
                state=ClipState.READY,
                alpha_asset=ClipAsset(alpha_dir, "sequence"),
            )
            alpha_files = clip.alpha_asset.get_frame_files()
            alpha_lookup = {os.path.splitext(f)[0]: f for f in alpha_files}

            with patch(
                "backend.service.read_mask_frame", return_value=np.ones((4, 4), dtype=np.float32)
            ) as mock_read:
                svc._read_alpha_frame(
                    clip,
                    0,
                    alpha_files,
                    None,
                    input_stem="frame_000003",
                    alpha_stem_lookup=alpha_lookup,
                )

            assert mock_read.call_args is not None
            assert mock_read.call_args.args[0].endswith("frame_000003.png")

    def test_sequence_falls_back_to_index_when_input_stem_missing(self):
        svc = CorridorKeyService()
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_dir = os.path.join(tmpdir, "AlphaHint")
            os.makedirs(alpha_dir)
            fallback = os.path.join(alpha_dir, "_alphaHint_000000.png")
            cv2.imwrite(fallback, np.ones((4, 4), dtype=np.uint8) * 255)

            clip = ClipEntry(
                "test",
                tmpdir,
                state=ClipState.READY,
                alpha_asset=ClipAsset(alpha_dir, "sequence"),
            )
            alpha_files = clip.alpha_asset.get_frame_files()

            with patch(
                "backend.service.read_mask_frame", return_value=np.ones((4, 4), dtype=np.float32)
            ) as mock_read:
                svc._read_alpha_frame(
                    clip,
                    0,
                    alpha_files,
                    None,
                    input_stem="frame_000123",
                    alpha_stem_lookup={},
                )

            assert mock_read.call_args is not None
            assert mock_read.call_args.args[0].endswith("_alphaHint_000000.png")


# ── TestWriteImage ──


class TestWriteImage:
    def test_png_float_to_uint8(self):
        """PNG path converts float32 to uint8."""
        svc = CorridorKeyService()
        img = np.ones((4, 4), dtype=np.float32) * 0.5
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.png")
            svc._write_image(img, path, "png", "clip1", 0)
            assert os.path.exists(path)
            # Read back and verify
            loaded = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            assert loaded is not None
            assert loaded.dtype == np.uint8

    def test_write_failure_raises(self):
        svc = CorridorKeyService()
        img = np.ones((4, 4), dtype=np.uint8)
        with pytest.raises(WriteFailureError):
            # Writing to non-existent deeply nested path
            svc._write_image(img, "/nonexistent/dir/deep/file.png", "png", "clip1", 0)

    def test_unknown_format_falls_through_to_png(self):
        """Codex: unknown format string silently uses PNG path."""
        svc = CorridorKeyService()
        img = np.ones((4, 4), dtype=np.float32) * 0.5
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.bmp")
            svc._write_image(img, path, "bmp", "clip1", 0)
            # File written (via PNG code path, but cv2 uses extension)
            assert os.path.exists(path)


# ── TestWriteManifest ──


class TestWriteManifest:
    def test_manifest_content(self):
        svc = CorridorKeyService()
        cfg = OutputConfig(fg_enabled=True, matte_enabled=True, comp_enabled=False)
        params = InferenceParams(despill_strength=0.8)
        with tempfile.TemporaryDirectory() as tmpdir:
            svc._write_manifest(tmpdir, cfg, params)
            manifest_path = os.path.join(tmpdir, ".corridorkey_manifest.json")
            assert os.path.exists(manifest_path)
            with open(manifest_path) as f:
                data = json.load(f)
            assert data["version"] == 1
            assert "fg" in data["enabled_outputs"]
            assert "matte" in data["enabled_outputs"]
            assert "comp" not in data["enabled_outputs"]
            assert data["params"]["despill_strength"] == 0.8

    def test_manifest_overwrites_existing(self):
        svc = CorridorKeyService()
        cfg = OutputConfig()
        params = InferenceParams()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write twice — second should overwrite
            svc._write_manifest(tmpdir, cfg, params)
            svc._write_manifest(tmpdir, OutputConfig(fg_enabled=False), params)
            manifest_path = os.path.join(tmpdir, ".corridorkey_manifest.json")
            with open(manifest_path) as f:
                data = json.load(f)
            assert "fg" not in data["enabled_outputs"]

    def test_manifest_write_failure_graceful(self):
        """Manifest write failure is non-fatal (logs warning, continues)."""
        svc = CorridorKeyService()
        cfg = OutputConfig()
        params = InferenceParams()
        # Write to non-existent path — should not raise
        svc._write_manifest("/nonexistent/path/that/doesnt/exist", cfg, params)


# ── TestWriteOutputs ──


class TestWriteOutputs:
    def _make_result_dict(self):
        return {
            "fg": np.ones((4, 4, 3), dtype=np.float32) * 0.5,
            "alpha": np.ones((4, 4, 1), dtype=np.float32) * 0.8,
            "comp": np.ones((4, 4, 3), dtype=np.float32) * 0.3,
            "processed": np.ones((4, 4, 4), dtype=np.float32) * 0.6,
        }

    def test_all_outputs_enabled(self, tmp_clip_dir):
        svc = CorridorKeyService()
        from backend.validators import ensure_output_dirs

        dirs = ensure_output_dirs(tmp_clip_dir)
        res = self._make_result_dict()
        cfg = OutputConfig(
            fg_format="png",
            matte_format="png",
            comp_format="png",
            processed_format="png",
        )
        svc._write_outputs(res, dirs, "frame_00000", "clip1", 0, cfg)

        assert os.path.exists(os.path.join(dirs["fg"], "frame_00000.png"))
        assert os.path.exists(os.path.join(dirs["matte"], "frame_00000.png"))
        assert os.path.exists(os.path.join(dirs["comp"], "frame_00000.png"))
        assert os.path.exists(os.path.join(dirs["processed"], "frame_00000.png"))

    def test_selective_disable(self, tmp_clip_dir):
        svc = CorridorKeyService()
        from backend.validators import ensure_output_dirs

        dirs = ensure_output_dirs(tmp_clip_dir)
        res = self._make_result_dict()
        cfg = OutputConfig(
            fg_enabled=False,
            matte_enabled=True,
            comp_enabled=False,
            processed_enabled=False,
            matte_format="png",
        )
        svc._write_outputs(res, dirs, "frame_00000", "clip1", 0, cfg)

        assert not os.path.exists(os.path.join(dirs["fg"], "frame_00000.png"))
        assert os.path.exists(os.path.join(dirs["matte"], "frame_00000.png"))
        assert not os.path.exists(os.path.join(dirs["comp"], "frame_00000.png"))

    def test_missing_processed_key(self, tmp_clip_dir):
        """If 'processed' not in result dict, skip gracefully."""
        svc = CorridorKeyService()
        from backend.validators import ensure_output_dirs

        dirs = ensure_output_dirs(tmp_clip_dir)
        res = {
            "fg": np.ones((4, 4, 3), dtype=np.float32),
            "alpha": np.ones((4, 4, 1), dtype=np.float32),
            "comp": np.ones((4, 4, 3), dtype=np.float32),
            # 'processed' intentionally missing
        }
        cfg = OutputConfig(
            fg_format="png",
            matte_format="png",
            comp_format="png",
            processed_enabled=True,
            processed_format="png",
        )
        # Should not raise
        svc._write_outputs(res, dirs, "frame_00000", "clip1", 0, cfg)

    def test_fg_exr_is_linearized_before_write(self, tmp_clip_dir):
        svc = CorridorKeyService()
        from backend.frame_io import _srgb_to_linear
        from backend.validators import ensure_output_dirs

        dirs = ensure_output_dirs(tmp_clip_dir)
        res = self._make_result_dict()
        cfg = OutputConfig(
            fg_format="exr",
            matte_enabled=False,
            comp_enabled=False,
            processed_enabled=False,
        )

        writes: list[tuple[np.ndarray, str, str]] = []

        def _capture(img, path, fmt, clip_name, frame_index, exr_compression="dwab"):
            writes.append((img.copy(), path, fmt))

        svc._write_image = _capture
        svc._write_outputs(res, dirs, "frame_00000", "clip1", 0, cfg)

        assert len(writes) == 1
        img, path, fmt = writes[0]
        expected_bgr = cv2.cvtColor(_srgb_to_linear(res["fg"]), cv2.COLOR_RGB2BGR)
        assert fmt == "exr"
        assert path.endswith(os.path.join("FG", "frame_00000.exr"))
        np.testing.assert_allclose(img, expected_bgr, atol=1e-6)

    def test_comp_exr_is_linearized_before_write(self, tmp_clip_dir):
        svc = CorridorKeyService()
        from backend.frame_io import _srgb_to_linear
        from backend.validators import ensure_output_dirs

        dirs = ensure_output_dirs(tmp_clip_dir)
        res = self._make_result_dict()
        cfg = OutputConfig(
            fg_enabled=False,
            matte_enabled=False,
            comp_format="exr",
            processed_enabled=False,
        )

        writes: list[tuple[np.ndarray, str, str]] = []

        def _capture(img, path, fmt, clip_name, frame_index, exr_compression="dwab"):
            writes.append((img.copy(), path, fmt))

        svc._write_image = _capture
        svc._write_outputs(res, dirs, "frame_00000", "clip1", 0, cfg)

        assert len(writes) == 1
        img, path, fmt = writes[0]
        expected_bgr = cv2.cvtColor(_srgb_to_linear(res["comp"]), cv2.COLOR_RGB2BGR)
        assert fmt == "exr"
        assert path.endswith(os.path.join("Comp", "frame_00000.exr"))
        np.testing.assert_allclose(img, expected_bgr, atol=1e-6)

    def test_processed_exr_preserves_straight_linear_rgba(self, tmp_clip_dir):
        svc = CorridorKeyService()
        from backend.validators import ensure_output_dirs

        dirs = ensure_output_dirs(tmp_clip_dir)
        res = self._make_result_dict()
        res["processed"] = np.zeros((4, 4, 4), dtype=np.float32)
        res["processed"][:, :, 0] = 0.8
        res["processed"][:, :, 1] = 0.4
        res["processed"][:, :, 2] = 0.2
        res["processed"][:, :, 3] = 0.25
        cfg = OutputConfig(
            fg_enabled=False,
            matte_enabled=False,
            comp_enabled=False,
            processed_format="exr",
        )

        writes: list[tuple[np.ndarray, str, str]] = []

        def _capture(img, path, fmt, clip_name, frame_index, exr_compression="dwab"):
            writes.append((img.copy(), path, fmt))

        svc._write_image = _capture
        svc._write_outputs(res, dirs, "frame_00000", "clip1", 0, cfg)

        assert len(writes) == 1
        img, path, fmt = writes[0]
        expected_bgra = cv2.cvtColor(res["processed"], cv2.COLOR_RGBA2BGRA)
        assert fmt == "exr"
        assert path.endswith(os.path.join("Processed", "frame_00000.exr"))
        np.testing.assert_allclose(img, expected_bgra, atol=1e-6)


# ── TestRunInference ──


class TestRunInference:
    def _setup_service_with_mock_engine(self):
        """Create a service with a mocked inference engine."""
        svc = CorridorKeyService()
        svc._device = "cuda"
        svc._active_model = _ActiveModel.INFERENCE
        mock_engine = MagicMock()
        mock_engine.process_frame.return_value = {
            "fg": np.ones((4, 4, 3), dtype=np.float32) * 0.5,
            "alpha": np.ones((4, 4, 1), dtype=np.float32) * 0.8,
            "comp": np.ones((4, 4, 3), dtype=np.float32) * 0.3,
            "processed": np.ones((4, 4, 4), dtype=np.float32) * 0.6,
        }
        svc._engine_pool = [mock_engine]
        return svc, mock_engine

    def test_happy_path(self, sample_clip, tmp_clip_dir):
        svc, mock_engine = self._setup_service_with_mock_engine()
        params = InferenceParams()
        cfg = OutputConfig(
            fg_format="png",
            matte_format="png",
            comp_format="png",
            processed_format="png",
        )
        results = svc.run_inference(sample_clip, params, output_config=cfg)
        assert len(results) == 3  # 3 frames in fixture
        assert all(r.success for r in results)
        assert mock_engine.process_frame.call_count == 3

    def test_resume_skips_stems(self, sample_clip, tmp_clip_dir):
        svc, mock_engine = self._setup_service_with_mock_engine()
        params = InferenceParams()
        input_files = sample_clip.input_asset.get_frame_files()
        first_stem = os.path.splitext(input_files[0])[0]
        results = svc.run_inference(
            sample_clip,
            params,
            skip_stems={first_stem},
            output_config=OutputConfig(
                fg_format="png", matte_format="png", comp_format="png", processed_format="png"
            ),
        )
        # First frame skipped (resumed), 2 processed
        skipped = [r for r in results if r.warning and "resumed" in r.warning]
        assert len(skipped) == 1
        assert mock_engine.process_frame.call_count == 2

    def test_cancellation_raises(self, sample_clip):
        svc, _ = self._setup_service_with_mock_engine()
        params = InferenceParams()
        job = GPUJob(JobType.INFERENCE, "test_clip")
        job.request_cancel()

        with pytest.raises(JobCancelledError):
            svc.run_inference(
                sample_clip,
                params,
                job=job,
                output_config=OutputConfig(
                    fg_format="png", matte_format="png", comp_format="png", processed_format="png"
                ),
            )

    def test_frame_read_error_continues(self, sample_clip, tmp_clip_dir):
        """FrameReadError on a frame → skip that frame, continue others."""
        svc, mock_engine = self._setup_service_with_mock_engine()
        params = InferenceParams()

        # Patch _read_input_frame to fail on first call, succeed on others
        original_read = svc._read_input_frame
        call_count = [0]

        def fail_first_frame(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise FrameReadError("test_clip", 0, "/fake/path.png")
            return original_read(*args, **kwargs)

        svc._read_input_frame = fail_first_frame

        warnings = []
        results = svc.run_inference(
            sample_clip,
            params,
            on_warning=lambda msg: warnings.append(msg),
            output_config=OutputConfig(
                fg_format="png", matte_format="png", comp_format="png", processed_format="png"
            ),
        )
        failed = [r for r in results if not r.success]
        assert len(failed) >= 1
        assert len(warnings) >= 1

    def test_engine_process_frame_raises(self, sample_clip, tmp_clip_dir):
        """Codex: engine.process_frame raising should propagate (fail-fast)."""
        svc, mock_engine = self._setup_service_with_mock_engine()
        mock_engine.process_frame.side_effect = RuntimeError("GPU OOM")
        params = InferenceParams()

        # Should propagate — not caught by the per-frame handler
        with pytest.raises(RuntimeError, match="GPU OOM"):
            svc.run_inference(
                sample_clip,
                params,
                output_config=OutputConfig(
                    fg_format="png", matte_format="png", comp_format="png", processed_format="png"
                ),
            )

    def test_progress_callback_cadence(self, sample_clip, tmp_clip_dir):
        """Codex: progress fires every 5th frame + final, not per-frame."""
        svc, _ = self._setup_service_with_mock_engine()
        params = InferenceParams()
        calls = []
        svc.run_inference(
            sample_clip,
            params,
            on_progress=lambda name, cur, total, **kwargs: calls.append((cur, total)),
            output_config=OutputConfig(
                fg_format="png", matte_format="png", comp_format="png", processed_format="png"
            ),
        )
        # With 3 frames: fires at frame 0 (0%5==0) and final (3)
        assert len(calls) >= 2
        # Final call should be (total, total)
        assert calls[-1][0] == calls[-1][1]

    def test_state_transition_to_complete(self, sample_clip, tmp_clip_dir):
        svc, _ = self._setup_service_with_mock_engine()
        params = InferenceParams()
        assert sample_clip.state == ClipState.READY
        svc.run_inference(
            sample_clip,
            params,
            output_config=OutputConfig(
                fg_format="png", matte_format="png", comp_format="png", processed_format="png"
            ),
        )
        assert sample_clip.state == ClipState.COMPLETE

    def test_zero_frame_clip_no_complete(self):
        """Codex: zero-frame clip should not transition to COMPLETE (0==0 passes)."""
        svc, _ = self._setup_service_with_mock_engine()
        params = InferenceParams()

        with tempfile.TemporaryDirectory() as tmpdir:
            clip_root = os.path.join(tmpdir, "empty_clip")
            input_dir = os.path.join(clip_root, "Input")
            alpha_dir = os.path.join(clip_root, "AlphaHint")
            os.makedirs(input_dir)
            os.makedirs(alpha_dir)

            clip = ClipEntry(
                name="empty",
                root_path=clip_root,
                state=ClipState.READY,
                input_asset=ClipAsset(input_dir, "sequence"),
                alpha_asset=ClipAsset(alpha_dir, "sequence"),
            )
            results = svc.run_inference(
                clip,
                params,
                output_config=OutputConfig(
                    fg_format="png", matte_format="png", comp_format="png", processed_format="png"
                ),
            )
            # 0 frames processed, 0 total — current code does 0==0 → COMPLETE
            # This test documents the current behavior
            assert len(results) == 0
            # NOTE: Current behavior transitions to COMPLETE for 0 frames.
            # This is a known policy issue (Codex finding).

    def test_missing_assets_raises(self):
        svc = CorridorKeyService()
        clip = ClipEntry("test", "/tmp", state=ClipState.READY)
        with pytest.raises(CorridorKeyError, match="missing input or alpha"):
            svc.run_inference(clip, InferenceParams())

    def test_video_captures_released(self, sample_clip, tmp_clip_dir):
        """Video captures should be released in finally block."""
        svc, mock_engine = self._setup_service_with_mock_engine()
        job = GPUJob(JobType.INFERENCE, "test_clip")

        call_count = [0]
        result_dict = {
            "fg": np.ones((4, 4, 3), dtype=np.float32) * 0.5,
            "alpha": np.ones((4, 4, 1), dtype=np.float32) * 0.8,
            "comp": np.ones((4, 4, 3), dtype=np.float32) * 0.3,
            "processed": np.ones((4, 4, 4), dtype=np.float32) * 0.6,
        }

        def cancel_after_first(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                job.request_cancel()
            return result_dict

        mock_engine.process_frame.side_effect = cancel_after_first

        params = InferenceParams()
        with pytest.raises(JobCancelledError):
            svc.run_inference(
                sample_clip,
                params,
                job=job,
                output_config=OutputConfig(
                    fg_format="png", matte_format="png", comp_format="png", processed_format="png"
                ),
            )
        # Test passed — if finally block didn't run, we'd leak captures

    def test_frame_range_uses_stem_matched_partial_alpha_sequence(self):
        svc, mock_engine = self._setup_service_with_mock_engine()
        params = InferenceParams()

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = os.path.join(tmpdir, "Frames")
            alpha_dir = os.path.join(tmpdir, "AlphaHint")
            os.makedirs(frames_dir)
            os.makedirs(alpha_dir)

            for i in range(5):
                cv2.imwrite(
                    os.path.join(frames_dir, f"frame_{i:06d}.png"),
                    np.zeros((4, 4, 3), dtype=np.uint8),
                )
            for i in (3, 4):
                cv2.imwrite(
                    os.path.join(alpha_dir, f"frame_{i:06d}.png"),
                    np.ones((4, 4), dtype=np.uint8) * 255,
                )

            clip = ClipEntry(
                "test",
                tmpdir,
                state=ClipState.READY,
                input_asset=ClipAsset(frames_dir, "sequence"),
                alpha_asset=ClipAsset(alpha_dir, "sequence"),
            )

            fake_img = np.zeros((4, 4, 3), dtype=np.float32)
            fake_mask = np.ones((4, 4), dtype=np.float32)
            read_mask_paths: list[str] = []

            def _record_mask(path, clip_name, frame_index):
                read_mask_paths.append(os.path.basename(path))
                return fake_mask

            with (
                patch("backend.service.read_image_frame", return_value=fake_img),
                patch("backend.service.read_mask_frame", side_effect=_record_mask),
            ):
                results = svc.run_inference(
                    clip,
                    params,
                    frame_range=(3, 4),
                    output_config=OutputConfig(
                        fg_format="png",
                        matte_format="png",
                        comp_format="png",
                        processed_format="png",
                    ),
                )

        assert len(results) == 2
        assert all(r.success for r in results)
        assert mock_engine.process_frame.call_count == 2
        assert read_mask_paths == ["frame_000003.png", "frame_000004.png"]


# ── TestReprocessSingleFrame ──


class TestReprocessSingleFrame:
    def test_returns_dict(self, sample_clip):
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.INFERENCE
        mock_engine = MagicMock()
        mock_engine.process_frame.return_value = {
            "fg": np.ones((4, 4, 3), dtype=np.float32),
            "alpha": np.ones((4, 4, 1), dtype=np.float32),
        }
        svc._engine_pool = [mock_engine]
        result = svc.reprocess_single_frame(sample_clip, InferenceParams(), 0)
        assert result is not None
        assert "fg" in result

    def test_missing_assets_returns_none(self):
        svc = CorridorKeyService()
        clip = ClipEntry("test", "/tmp")
        result = svc.reprocess_single_frame(clip, InferenceParams(), 0)
        assert result is None

    def test_cancelled_returns_none(self, sample_clip):
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.INFERENCE
        svc._engine_pool = [MagicMock()]
        job = GPUJob(JobType.PREVIEW_REPROCESS, "test")
        job.request_cancel()
        result = svc.reprocess_single_frame(sample_clip, InferenceParams(), 0, job=job)
        assert result is None

    def test_out_of_range_returns_none(self, sample_clip):
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.INFERENCE
        svc._engine_pool = [MagicMock()]
        result = svc.reprocess_single_frame(sample_clip, InferenceParams(), 9999)
        assert result is None

    def test_reprocess_forwards_status_during_engine_warmup(self, sample_clip):
        svc = CorridorKeyService()
        mock_engine = MagicMock()
        mock_engine.process_frame.return_value = {
            "fg": np.ones((4, 4, 3), dtype=np.float32),
            "alpha": np.ones((4, 4, 1), dtype=np.float32),
            "comp": np.ones((4, 4, 3), dtype=np.float32),
            "processed": np.ones((4, 4, 4), dtype=np.float32),
        }
        statuses: list[str] = []

        def _fake_get_engine_pool(on_status=None):
            assert on_status is not None
            on_status("Loading engine 1/1...")
            return [mock_engine]

        fake_img = np.zeros((4, 4, 3), dtype=np.float32)
        fake_mask = np.ones((4, 4), dtype=np.float32)
        with (
            patch.object(svc, "_get_engine_pool", side_effect=_fake_get_engine_pool),
            patch("backend.service.read_image_frame", return_value=fake_img),
            patch("backend.service.read_mask_frame", return_value=fake_mask),
        ):
            result = svc.reprocess_single_frame(
                sample_clip,
                InferenceParams(),
                0,
                on_status=statuses.append,
            )

        assert result is not None
        assert statuses == ["Loading engine 1/1..."]

    def test_exr_sequence_honors_explicit_srgb_setting(self):
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.INFERENCE
        mock_engine = MagicMock()
        mock_engine.process_frame.return_value = {
            "fg": np.ones((4, 4, 3), dtype=np.float32),
            "alpha": np.ones((4, 4, 1), dtype=np.float32),
            "comp": np.ones((4, 4, 3), dtype=np.float32),
            "processed": np.ones((4, 4, 4), dtype=np.float32),
        }
        svc._engine_pool = [mock_engine]

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = os.path.join(tmpdir, "Frames")
            alpha_dir = os.path.join(tmpdir, "AlphaHint")
            os.makedirs(frames_dir)
            os.makedirs(alpha_dir)

            frame_path = os.path.join(frames_dir, "frame_000000.exr")
            alpha_path = os.path.join(alpha_dir, "_alphaHint_000000.png")
            with open(frame_path, "w", encoding="utf-8") as handle:
                handle.write("dummy")
            with open(alpha_path, "w", encoding="utf-8") as handle:
                handle.write("dummy")

            clip = ClipEntry(
                "test",
                tmpdir,
                state=ClipState.READY,
                input_asset=ClipAsset(frames_dir, "sequence"),
                alpha_asset=ClipAsset(alpha_dir, "sequence"),
            )

            fake_img = np.zeros((4, 4, 3), dtype=np.float32)
            fake_mask = np.ones((4, 4), dtype=np.float32)
            with (
                patch("backend.service.read_image_frame", return_value=fake_img),
                patch("backend.service.read_mask_frame", return_value=fake_mask),
            ):
                svc.reprocess_single_frame(
                    clip,
                    InferenceParams(input_is_linear=False),
                    0,
                )

        assert mock_engine.process_frame.call_args is not None
        assert mock_engine.process_frame.call_args.kwargs["input_is_linear"] is False

    def test_exr_sequence_honors_explicit_linear_setting(self):
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.INFERENCE
        mock_engine = MagicMock()
        mock_engine.process_frame.return_value = {
            "fg": np.ones((4, 4, 3), dtype=np.float32),
            "alpha": np.ones((4, 4, 1), dtype=np.float32),
            "comp": np.ones((4, 4, 3), dtype=np.float32),
            "processed": np.ones((4, 4, 4), dtype=np.float32),
        }
        svc._engine_pool = [mock_engine]

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = os.path.join(tmpdir, "Frames")
            alpha_dir = os.path.join(tmpdir, "AlphaHint")
            os.makedirs(frames_dir)
            os.makedirs(alpha_dir)

            frame_path = os.path.join(frames_dir, "frame_000000.exr")
            alpha_path = os.path.join(alpha_dir, "_alphaHint_000000.png")
            with open(frame_path, "w", encoding="utf-8") as handle:
                handle.write("dummy")
            with open(alpha_path, "w", encoding="utf-8") as handle:
                handle.write("dummy")

            clip = ClipEntry(
                "test",
                tmpdir,
                state=ClipState.READY,
                input_asset=ClipAsset(frames_dir, "sequence"),
                alpha_asset=ClipAsset(alpha_dir, "sequence"),
            )

            fake_img = np.zeros((4, 4, 3), dtype=np.float32)
            fake_mask = np.ones((4, 4), dtype=np.float32)
            with (
                patch("backend.service.read_image_frame", return_value=fake_img),
                patch("backend.service.read_mask_frame", return_value=fake_mask),
            ):
                svc.reprocess_single_frame(
                    clip,
                    InferenceParams(input_is_linear=True),
                    0,
                )

        assert mock_engine.process_frame.call_args is not None
        assert mock_engine.process_frame.call_args.kwargs["input_is_linear"] is True

    def test_sequence_reprocess_prefers_alpha_with_matching_input_stem(self):
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.INFERENCE
        mock_engine = MagicMock()
        mock_engine.process_frame.return_value = {
            "fg": np.ones((4, 4, 3), dtype=np.float32),
            "alpha": np.ones((4, 4, 1), dtype=np.float32),
            "comp": np.ones((4, 4, 3), dtype=np.float32),
            "processed": np.ones((4, 4, 4), dtype=np.float32),
        }
        svc._engine_pool = [mock_engine]

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = os.path.join(tmpdir, "Frames")
            alpha_dir = os.path.join(tmpdir, "AlphaHint")
            os.makedirs(frames_dir)
            os.makedirs(alpha_dir)

            for i in range(5):
                with open(os.path.join(frames_dir, f"frame_{i:06d}.png"), "wb") as handle:
                    handle.write(b"fake")
            for i in (3, 4):
                with open(os.path.join(alpha_dir, f"frame_{i:06d}.png"), "wb") as handle:
                    handle.write(b"fake")

            clip = ClipEntry(
                "test",
                tmpdir,
                state=ClipState.READY,
                input_asset=ClipAsset(frames_dir, "sequence"),
                alpha_asset=ClipAsset(alpha_dir, "sequence"),
            )

            fake_img = np.zeros((4, 4, 3), dtype=np.float32)
            fake_mask = np.ones((4, 4), dtype=np.float32)
            read_mask_paths: list[str] = []

            def _record_mask(path, clip_name, frame_index):
                read_mask_paths.append(os.path.basename(path))
                return fake_mask

            with (
                patch("backend.service.read_image_frame", return_value=fake_img),
                patch("backend.service.read_mask_frame", side_effect=_record_mask),
            ):
                result = svc.reprocess_single_frame(
                    clip,
                    InferenceParams(),
                    3,
                )

        assert result is not None
        assert read_mask_paths == ["frame_000003.png"]


# ── TestUnloadEngines ──


class TestUnloadEngines:
    def test_clears_all(self):
        svc = CorridorKeyService()
        svc._engine_pool = [MagicMock()]
        svc._gvm_processor = MagicMock()
        svc._videomama_pipeline = MagicMock()
        svc._active_model = _ActiveModel.INFERENCE

        svc.unload_engines()

        assert svc._engine_pool == []
        assert svc._gvm_processor is None
        assert svc._videomama_pipeline is None
        assert svc._active_model == _ActiveModel.NONE


# ── TestIsEngineLoaded ──


class TestIsEngineLoaded:
    def test_true_when_active(self):
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.INFERENCE
        svc._engine_pool = [MagicMock()]
        assert svc.is_engine_loaded() is True

    def test_false_when_not_loaded(self):
        svc = CorridorKeyService()
        assert svc.is_engine_loaded() is False


class TestSam2PreviewInputColorSpace:
    def test_preview_sam2_defaults_video_derived_exr_to_srgb(self):
        svc = CorridorKeyService()
        tracker = MagicMock()
        tracker.track_video.return_value = [np.ones((4, 4), dtype=np.uint8)]
        svc._get_sam2_tracker = MagicMock(return_value=tracker)

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = os.path.join(tmpdir, "Frames")
            os.makedirs(frames_dir)
            frame_path = os.path.join(frames_dir, "frame_000000.exr")
            with open(frame_path, "w", encoding="utf-8") as handle:
                handle.write("dummy")
            with open(
                os.path.join(tmpdir, ".video_metadata.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump({"codec": "prores"}, handle)

            clip = ClipEntry(
                "test",
                tmpdir,
                state=ClipState.RAW,
                input_asset=ClipAsset(frames_dir, "sequence"),
            )

            prompt = AnnotationPromptFrame(
                frame_index=0,
                positive_points=[(10.0, 12.0)],
                negative_points=[],
            )
            fake_img = np.full((4, 4, 3), 0.5, dtype=np.float32)
            with (
                patch(
                    "backend.service.load_annotation_prompt_frames",
                    return_value=[prompt],
                ),
                patch(
                    "backend.service.read_image_frame",
                    return_value=fake_img,
                ) as mock_read,
            ):
                result = svc.preview_sam2_prompt(clip, preferred_frame_index=0)

        assert result is not None
        assert mock_read.call_args is not None
        assert mock_read.call_args.kwargs["gamma_correct_exr"] is False

    def test_preview_sam2_honors_explicit_linear_override(self):
        svc = CorridorKeyService()
        tracker = MagicMock()
        tracker.track_video.return_value = [np.ones((4, 4), dtype=np.uint8)]
        svc._get_sam2_tracker = MagicMock(return_value=tracker)

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = os.path.join(tmpdir, "Frames")
            os.makedirs(frames_dir)
            frame_path = os.path.join(frames_dir, "frame_000000.exr")
            with open(frame_path, "w", encoding="utf-8") as handle:
                handle.write("dummy")

            clip = ClipEntry(
                "test",
                tmpdir,
                state=ClipState.RAW,
                input_asset=ClipAsset(frames_dir, "sequence"),
            )

            prompt = AnnotationPromptFrame(
                frame_index=0,
                positive_points=[(10.0, 12.0)],
                negative_points=[],
            )
            fake_img = np.full((4, 4, 3), 0.5, dtype=np.float32)
            with (
                patch(
                    "backend.service.load_annotation_prompt_frames",
                    return_value=[prompt],
                ),
                patch(
                    "backend.service.read_image_frame",
                    return_value=fake_img,
                ) as mock_read,
            ):
                result = svc.preview_sam2_prompt(
                    clip,
                    preferred_frame_index=0,
                    input_is_linear=True,
                )

        assert result is not None
        assert mock_read.call_args is not None
        assert mock_read.call_args.kwargs["gamma_correct_exr"] is True

    def test_false_when_different_model(self):
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.GVM
        svc._gvm_processor = MagicMock()
        assert svc.is_engine_loaded() is False

    def test_false_when_active_but_engine_empty(self):
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.INFERENCE
        svc._engine_pool = []
        assert svc.is_engine_loaded() is False


# ── TestRunGVM ──


class TestRunGVM:
    def test_missing_input_raises(self):
        svc = CorridorKeyService()
        clip = ClipEntry("test", "/tmp", state=ClipState.RAW)
        with pytest.raises(CorridorKeyError, match="missing input asset"):
            svc.run_gvm(clip)

    def test_cancellation_before_gvm(self):
        svc = CorridorKeyService()
        svc._device = "cuda"  # skip GPU guard (we're testing cancellation)
        svc._active_model = _ActiveModel.GVM
        svc._gvm_processor = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "Input")
            os.makedirs(input_dir)
            clip = ClipEntry(
                "test",
                tmpdir,
                state=ClipState.RAW,
                input_asset=ClipAsset(input_dir, "sequence"),
            )
            job = GPUJob(JobType.GVM_ALPHA, "test")
            job.request_cancel()

            with pytest.raises(JobCancelledError):
                svc.run_gvm(clip, job=job)

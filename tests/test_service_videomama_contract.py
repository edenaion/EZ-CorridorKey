"""Tests for VideoMaMa dtype/range contracts and failure behavior.

Contract: _load_frames_for_videomama() returns uint8 RGB [0,255] and
_load_mask_frames_for_videomama() returns uint8 grayscale [0,255] with
binary threshold at 10. The inference module expects uint8 for PIL conversion.
The output from run_inference is float32 [0,1], converted to uint8 during write.
"""
import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from backend.service import CorridorKeyService, _ActiveModel
from backend.clip_state import ClipAsset, ClipEntry, ClipState
from backend.errors import CorridorKeyError, JobCancelledError
from backend.job_queue import GPUJob, JobType


class TestVideoMaMaLoadFrames:
    """Contract: _load_frames_for_videomama returns uint8 RGB [0,255]."""

    def test_png_frames_dtype_and_range(self):
        """PNG frames should be uint8 RGB."""
        svc = CorridorKeyService()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write 2 tiny PNGs
            for i in range(2):
                img = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(tmpdir, f"frame_{i:05d}.png"), img)

            asset = ClipAsset(tmpdir, "sequence")
            frames = svc._load_frames_for_videomama(asset, "test")

            assert len(frames) == 2
            for f in frames:
                assert f.dtype == np.uint8, f"Expected uint8, got {f.dtype}"
                assert f.ndim == 3
                assert f.shape[2] == 3  # RGB

    def test_skips_unreadable_files(self):
        """Files that cv2.imread can't read should be silently skipped."""
        svc = CorridorKeyService()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write one valid PNG and one corrupt file
            cv2.imwrite(
                os.path.join(tmpdir, "frame_00000.png"),
                np.zeros((4, 4, 3), dtype=np.uint8),
            )
            with open(os.path.join(tmpdir, "frame_00001.png"), "wb") as f:
                f.write(b"not a png")

            asset = ClipAsset(tmpdir, "sequence")
            frames = svc._load_frames_for_videomama(asset, "test")
            # Only the valid one loaded
            assert len(frames) == 1


class TestVideoMaMaMaskFrames:
    """Contract: _load_mask_frames_for_videomama returns uint8 grayscale binary."""

    def test_mask_dtype_and_range(self):
        svc = CorridorKeyService()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a mask with varied values
            mask = np.array([[5, 50, 200, 255]], dtype=np.uint8)
            mask_2d = np.tile(mask, (4, 1))
            cv2.imwrite(os.path.join(tmpdir, "mask_00000.png"), mask_2d)

            asset = ClipAsset(tmpdir, "sequence")
            masks = svc._load_mask_frames_for_videomama(asset, "test")

            assert len(masks) == 1
            m = masks[0]
            assert m.dtype == np.uint8
            assert m.ndim == 2  # grayscale

    def test_binary_threshold_at_10(self):
        """Values <= 10 should be 0, values > 10 should be 255."""
        svc = CorridorKeyService()
        with tempfile.TemporaryDirectory() as tmpdir:
            mask = np.array([[0, 10, 11, 255]], dtype=np.uint8)
            mask_2d = np.tile(mask, (2, 1))
            cv2.imwrite(os.path.join(tmpdir, "mask_00000.png"), mask_2d)

            asset = ClipAsset(tmpdir, "sequence")
            masks = svc._load_mask_frames_for_videomama(asset, "test")

            m = masks[0]
            # Pixels with value 0 and 10 should be 0 (below threshold)
            assert m[0, 0] == 0
            assert m[0, 1] == 0
            # Pixels with value 11 and 255 should be 255 (above threshold)
            assert m[0, 2] == 255
            assert m[0, 3] == 255


class TestVideoMaMaOutputWrite:
    """Contract: output frames are clipped to [0,1] then scaled to uint8."""

    def test_float_output_produces_valid_png(self):
        """If VideoMaMa returns float32 [0,1], output write should produce valid PNGs."""
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.VIDEOMAMA

        # Simulate what run_videomama does at line 842-843
        frame = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        out_bgr = cv2.cvtColor(
            (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        assert out_bgr.dtype == np.uint8
        assert out_bgr.max() <= 255
        # Value should be ~128 (0.5 * 255)
        assert 120 <= out_bgr.mean() <= 135

    def test_float_output_range_clamp(self):
        """Output values outside [0,1] are clamped before scaling to uint8."""
        frame = np.array([[[1.5, -0.1, 0.5]]], dtype=np.float32)
        out = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        assert out[0, 0, 0] == 255  # 1.5 clamped to 1.0
        assert out[0, 0, 1] == 0    # -0.1 clamped to 0.0
        assert 125 <= out[0, 0, 2] <= 130  # 0.5 → ~128


class TestVideoMaMaMissingAssets:
    def test_no_input_asset_raises(self):
        svc = CorridorKeyService()
        clip = ClipEntry("test", "/tmp", state=ClipState.MASKED)
        with pytest.raises(CorridorKeyError, match="missing input asset"):
            svc.run_videomama(clip)

    def test_no_mask_asset_raises(self):
        svc = CorridorKeyService()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "Input")
            os.makedirs(input_dir)
            clip = ClipEntry(
                "test", tmpdir, state=ClipState.MASKED,
                input_asset=ClipAsset(input_dir, "sequence"),
            )
            with pytest.raises(CorridorKeyError, match="missing mask asset"):
                svc.run_videomama(clip)


class TestVideoMaMaCancellation:
    def test_cancel_between_chunks(self):
        """Cancel is checked between chunks — partial AlphaHint files may exist."""
        svc = CorridorKeyService()
        svc._active_model = _ActiveModel.VIDEOMAMA
        svc._videomama_pipeline = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "Input")
            mask_dir = os.path.join(tmpdir, "VideoMamaMaskHint")
            os.makedirs(input_dir)
            os.makedirs(mask_dir)

            # Write one input frame and one mask
            cv2.imwrite(
                os.path.join(input_dir, "frame_00000.png"),
                np.zeros((4, 4, 3), dtype=np.uint8),
            )
            cv2.imwrite(
                os.path.join(mask_dir, "mask_00000.png"),
                np.zeros((4, 4), dtype=np.uint8),
            )

            clip = ClipEntry(
                "test", tmpdir, state=ClipState.MASKED,
                input_asset=ClipAsset(input_dir, "sequence"),
                mask_asset=ClipAsset(mask_dir, "sequence"),
            )

            job = GPUJob(JobType.VIDEOMAMA_ALPHA, "test")

            # Mock run_inference — it's imported locally inside run_videomama
            # via: from VideoMaMaInferenceModule.inference import run_inference
            chunk_output = [np.ones((4, 4, 3), dtype=np.float32)]

            def mock_run_inference(pipeline, frames, masks, chunk_size=50, on_status=None):
                yield chunk_output
                # Job will be cancelled before next chunk is processed
                job.request_cancel()
                yield chunk_output  # this chunk should trigger cancel

            mock_module = MagicMock()
            mock_module.run_inference = mock_run_inference
            with patch.dict("sys.modules", {
                "VideoMaMaInferenceModule": MagicMock(),
                "VideoMaMaInferenceModule.inference": mock_module,
            }):
                with pytest.raises(JobCancelledError):
                    svc.run_videomama(clip, job=job)

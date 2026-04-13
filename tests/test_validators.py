"""Tests for backend.validators module."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from backend.validators import (
    validate_frame_counts,
    normalize_mask_channels,
    normalize_mask_dtype,
    validate_frame_read,
    validate_write,
    ensure_output_dirs,
)
from backend.frame_io import read_mask_frame
from backend.errors import (
    FrameMismatchError,
    FrameReadError,
    MaskChannelError,
    WriteFailureError,
)


# --- validate_frame_counts ---


class TestValidateFrameCounts:
    def test_matching_counts(self):
        assert validate_frame_counts("shot1", 100, 100) == 100

    def test_mismatch_non_strict_returns_min(self):
        assert validate_frame_counts("shot1", 100, 80) == 80

    def test_mismatch_non_strict_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            validate_frame_counts("shot1", 100, 80)
        assert "frame count mismatch" in caplog.text

    def test_mismatch_strict_raises(self):
        with pytest.raises(FrameMismatchError) as exc_info:
            validate_frame_counts("shot1", 100, 80, strict=True)
        assert exc_info.value.input_count == 100
        assert exc_info.value.alpha_count == 80


# --- normalize_mask_channels ---


class TestNormalizeMaskChannels:
    def test_2d_passthrough(self):
        mask = np.ones((100, 200), dtype=np.float32)
        result = normalize_mask_channels(mask)
        assert result.ndim == 2
        assert result.shape == (100, 200)

    def test_3ch_extracts_first(self):
        mask = np.zeros((100, 200, 3), dtype=np.float32)
        mask[:, :, 0] = 1.0  # First channel = white
        mask[:, :, 1] = 0.5  # Second channel = gray
        result = normalize_mask_channels(mask)
        assert result.ndim == 2
        assert result.shape == (100, 200)
        np.testing.assert_array_equal(result, 1.0)

    def test_4ch_extracts_first(self):
        mask = np.zeros((100, 200, 4), dtype=np.float32)
        mask[:, :, 0] = 0.7
        result = normalize_mask_channels(mask)
        assert result.ndim == 2
        np.testing.assert_allclose(result, 0.7)

    def test_2ch_extracts_first(self):
        mask = np.zeros((100, 200, 2), dtype=np.float32)
        mask[:, :, 0] = 0.3
        result = normalize_mask_channels(mask)
        assert result.ndim == 2
        np.testing.assert_allclose(result, 0.3)

    def test_1ch_extracts_first(self):
        mask = np.ones((100, 200, 1), dtype=np.float32) * 0.5
        result = normalize_mask_channels(mask)
        assert result.ndim == 2
        np.testing.assert_allclose(result, 0.5)

    def test_0ch_raises(self):
        mask = np.zeros((100, 200, 0), dtype=np.float32)
        with pytest.raises(MaskChannelError):
            normalize_mask_channels(mask, "shot1", 5)


# --- normalize_mask_dtype ---


class TestNormalizeMaskDtype:
    def test_uint8(self):
        mask = np.array([0, 128, 255], dtype=np.uint8)
        result = normalize_mask_dtype(mask)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [0.0, 128 / 255.0, 1.0], atol=1e-5)

    def test_uint16(self):
        mask = np.array([0, 32768, 65535], dtype=np.uint16)
        result = normalize_mask_dtype(mask)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [0.0, 32768 / 65535.0, 1.0], atol=1e-5)

    def test_float32_passthrough(self):
        mask = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = normalize_mask_dtype(mask)
        assert result is mask  # Same object, no copy

    def test_float64_converts(self):
        mask = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        result = normalize_mask_dtype(mask)
        assert result.dtype == np.float32


# --- validate_frame_read ---


class TestValidateFrameRead:
    def test_valid_frame(self):
        frame = np.ones((100, 100, 3))
        result = validate_frame_read(frame, "shot1", 0, "/path/frame.exr")
        assert result is frame

    def test_none_raises(self):
        with pytest.raises(FrameReadError) as exc_info:
            validate_frame_read(None, "shot1", 5, "/path/frame.exr")
        assert exc_info.value.frame_index == 5
        assert "shot1" in str(exc_info.value)


# --- validate_write ---


class TestValidateWrite:
    def test_success(self):
        validate_write(True, "shot1", 0, "/path/frame.exr")

    def test_failure_raises(self):
        with pytest.raises(WriteFailureError) as exc_info:
            validate_write(False, "shot1", 42, "/path/frame.exr")
        assert exc_info.value.frame_index == 42


# --- ensure_output_dirs ---


class TestEnsureOutputDirs:
    def test_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = ensure_output_dirs(tmpdir)
            assert os.path.isdir(dirs["root"])
            assert os.path.isdir(dirs["fg"])
            assert os.path.isdir(dirs["matte"])
            assert os.path.isdir(dirs["comp"])
            assert os.path.isdir(dirs["processed"])

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirs1 = ensure_output_dirs(tmpdir)
            dirs2 = ensure_output_dirs(tmpdir)
            assert dirs1 == dirs2


# --- read_mask_frame integration (regression for dtype→channel ordering bug) ---


class TestReadMaskFrameNormalization:
    """Regression: read_mask_frame must normalize dtype BEFORE extracting channels.

    Previously normalize_mask_channels ran first and cast uint8→float32,
    so normalize_mask_dtype saw float32 and skipped the /255 division.
    A uint8 mask of 255 came through as float32 255.0 instead of 1.0.
    """

    def test_uint8_png_normalized_to_01(self, tmp_path):
        """uint8 PNG mask with value 255 must read back as 1.0, not 255.0."""
        mask = np.full((64, 64), 255, dtype=np.uint8)
        fpath = str(tmp_path / "mask.png")
        cv2.imwrite(fpath, mask)

        result = read_mask_frame(fpath)
        assert result is not None
        assert result.dtype == np.float32
        assert result.max() <= 1.0, f"Mask max {result.max()} exceeds 1.0 — dtype normalization bug"
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_uint8_3ch_png_normalized(self, tmp_path):
        """3-channel uint8 PNG mask — first channel extracted AND normalized."""
        mask = np.zeros((64, 64, 3), dtype=np.uint8)
        mask[:, :, 0] = 200  # First channel
        mask[:, :, 1] = 100
        mask[:, :, 2] = 50
        fpath = str(tmp_path / "mask_3ch.png")
        cv2.imwrite(fpath, mask)

        result = read_mask_frame(fpath)
        assert result is not None
        assert result.ndim == 2
        assert result.max() <= 1.0
        # OpenCV writes/reads BGR, so channel 0 in file = B channel.
        # read_mask_frame extracts first channel of what cv2 reads.

    def test_uint16_mask_normalized(self, tmp_path):
        """uint16 mask must be normalized by /65535."""
        mask = np.full((32, 32), 65535, dtype=np.uint16)
        fpath = str(tmp_path / "mask16.png")
        cv2.imwrite(fpath, mask)

        result = read_mask_frame(fpath)
        assert result is not None
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        np.testing.assert_allclose(result, 1.0, atol=1e-3)

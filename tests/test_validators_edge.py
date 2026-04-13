"""Edge case tests for backend.validators.

Covers dtype casting behaviour for non-standard integer types, boundary frame
counts, unexpected array dimensions, empty/minimal frames, and directory
creation idempotency.
"""

import os
import tempfile

import numpy as np
import pytest

from backend.validators import (
    ensure_output_dirs,
    normalize_mask_channels,
    normalize_mask_dtype,
    validate_frame_counts,
    validate_frame_read,
)
from backend.errors import MaskChannelError


# ---------------------------------------------------------------------------
# normalize_mask_dtype — non-standard integer dtypes
# ---------------------------------------------------------------------------


class TestNormalizeMaskDtypeEdge:
    """normalize_mask_dtype falls through to the bare astype(float32) branch
    for any dtype not explicitly handled (uint8, uint16, float32, float64).
    int32, int64, and uint32 all land in that branch: values are cast
    as-is with NO normalization to [0, 1].  This is intentional — the
    function documents only the handled dtypes; callers are responsible for
    ensuring meaningful value ranges for unrecognised dtypes.
    """

    def test_int32_cast_not_normalized(self):
        # Value 1000 stays 1000.0 — it is NOT scaled to [0, 1].
        mask = np.array([1000], dtype=np.int32)
        result = normalize_mask_dtype(mask)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1000.0])

    def test_int64_cast_not_normalized(self):
        # Same pass-through behaviour for int64.
        mask = np.array([1000], dtype=np.int64)
        result = normalize_mask_dtype(mask)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1000.0])

    def test_bool_true_becomes_one_false_becomes_zero(self):
        # bool is also unrecognised → cast branch; True→1.0, False→0.0.
        mask = np.array([True, False, True], dtype=bool)
        result = normalize_mask_dtype(mask)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1.0, 0.0, 1.0])

    def test_uint32_cast_not_normalized(self):
        # uint32 is distinct from uint8/uint16 and hits the else branch.
        mask = np.array([0, 500, 65536], dtype=np.uint32)
        result = normalize_mask_dtype(mask)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [0.0, 500.0, 65536.0])


# ---------------------------------------------------------------------------
# validate_frame_counts — boundary counts
# ---------------------------------------------------------------------------


class TestValidateFrameCountsEdge:
    def test_both_zero_returns_zero(self):
        # min(0, 0) == 0; no mismatch, no warning.
        assert validate_frame_counts("empty_clip", 0, 0) == 0

    def test_single_frame_returns_one(self):
        assert validate_frame_counts("single_frame", 1, 1) == 1

    def test_zero_input_with_nonzero_alpha_returns_zero(self):
        # min(0, 5) == 0; input side is the bottleneck.
        result = validate_frame_counts("mixed", 0, 5)
        assert result == 0


# ---------------------------------------------------------------------------
# normalize_mask_channels — unsupported dimensions
# ---------------------------------------------------------------------------


class TestNormalizeMaskChannelsEdge:
    def test_1d_raises_mask_channel_error(self):
        # ndim == 1: neither 2 nor 3, so MaskChannelError is raised.
        mask = np.zeros((10,), dtype=np.float32)
        with pytest.raises(MaskChannelError) as exc_info:
            normalize_mask_channels(mask, "clip1", 0)
        # The error stores the ndim as the channels field (see validator source).
        assert exc_info.value.channels == 1

    def test_4d_raises_mask_channel_error(self):
        # ndim == 4: also unsupported.
        mask = np.zeros((2, 3, 3, 1), dtype=np.float32)
        with pytest.raises(MaskChannelError) as exc_info:
            normalize_mask_channels(mask, "clip2", 7)
        assert exc_info.value.channels == 4


# ---------------------------------------------------------------------------
# validate_frame_read — empty and minimal valid arrays
# ---------------------------------------------------------------------------


class TestValidateFrameReadEdge:
    def test_empty_array_shape_passes(self):
        # An array with a zero dimension is NOT None — it passes validation.
        frame = np.empty((0, 0, 3), dtype=np.uint8)
        result = validate_frame_read(frame, "clip1", 0, "/some/path.png")
        assert result is frame

    def test_minimal_1x1x3_array_passes(self):
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        result = validate_frame_read(frame, "clip1", 0, "/some/path.png")
        assert result is frame


# ---------------------------------------------------------------------------
# ensure_output_dirs — nested creation and idempotency
# ---------------------------------------------------------------------------


class TestEnsureOutputDirsEdge:
    def test_nested_creation_creates_all_subdirs(self):
        # clip_root itself does not have to exist beforehand; makedirs handles
        # the full chain because exist_ok=True is used internally.
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_root = os.path.join(tmpdir, "level1", "level2", "clip")
            dirs = ensure_output_dirs(nested_root)
            for key in ("root", "fg", "matte", "comp", "processed"):
                assert os.path.isdir(dirs[key]), f"Missing dir for key '{key}'"

    def test_calling_twice_does_not_raise(self):
        # exist_ok=True means repeated calls are always safe.
        with tempfile.TemporaryDirectory() as tmpdir:
            ensure_output_dirs(tmpdir)
            dirs = ensure_output_dirs(tmpdir)  # second call — must not raise
            assert os.path.isdir(dirs["root"])

"""Tests for native_ops — Rust extension and Python fallback equivalence."""
import numpy as np
import pytest

from CorridorKeyModule.core.native_ops import (
    HAS_NATIVE,
    gbr_planar_to_rgb,
    bgr_u8_to_rgb_f32,
    rgb_f32_to_bgr_u8,
)


class TestGbrPlanarToRgb:
    def test_correct_channel_order(self):
        h, w = 2, 3
        g = np.full((h, w), 0.1, dtype=np.float32)
        b = np.full((h, w), 0.2, dtype=np.float32)
        r = np.full((h, w), 0.3, dtype=np.float32)
        raw = np.concatenate([g.ravel(), b.ravel(), r.ravel()]).tobytes()
        result = gbr_planar_to_rgb(raw, h, w)
        assert result.shape == (h, w, 3)
        np.testing.assert_allclose(result[0, 0], [0.3, 0.1, 0.2])

    def test_matches_numpy_reference(self):
        h, w = 64, 48
        planes = np.random.rand(3, h, w).astype(np.float32)
        raw = planes.tobytes()
        result = gbr_planar_to_rgb(raw, h, w)
        expected = np.stack([planes[2], planes[0], planes[1]], axis=-1)
        np.testing.assert_array_equal(result, expected)


class TestBgrU8ToRgbF32:
    def test_channel_swap_and_normalize(self):
        bgr = np.array([[[255, 0, 128]]], dtype=np.uint8)
        result = bgr_u8_to_rgb_f32(bgr)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result[0, 0, 0], 128 / 255.0, atol=1e-3)
        np.testing.assert_allclose(result[0, 0, 1], 0.0, atol=1e-3)
        np.testing.assert_allclose(result[0, 0, 2], 1.0, atol=1e-3)

    def test_matches_opencv_reference(self):
        bgr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = bgr_u8_to_rgb_f32(bgr)
        expected = bgr[:, :, ::-1].astype(np.float32) / 255.0
        np.testing.assert_allclose(result, expected, atol=1e-3)


class TestRgbF32ToBgrU8:
    def test_channel_swap_and_quantize(self):
        rgb = np.array([[[0.5, 0.0, 1.0]]], dtype=np.float32)
        result = rgb_f32_to_bgr_u8(rgb)
        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 255   # B from R=1.0
        assert result[0, 0, 1] == 0     # G
        assert result[0, 0, 2] == 128   # R from B=0.5

    def test_clamps_out_of_range(self):
        rgb = np.array([[[-0.5, 1.5, 0.5]]], dtype=np.float32)
        result = rgb_f32_to_bgr_u8(rgb)
        assert result[0, 0, 0] == 128  # B from 0.5
        assert result[0, 0, 1] == 255  # G from 1.5 clamped to 1.0
        assert result[0, 0, 2] == 0    # R from -0.5 clamped to 0.0

    def test_roundtrip(self):
        bgr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        rgb = bgr_u8_to_rgb_f32(bgr)
        back = rgb_f32_to_bgr_u8(rgb)
        np.testing.assert_array_equal(back, bgr)


class TestNativeAvailability:
    def test_has_native_is_bool(self):
        assert isinstance(HAS_NATIVE, bool)

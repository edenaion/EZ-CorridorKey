"""Tests for CorridorKeyModule.core.color_utils — checkerboard caching,
clean_matte, despill, and pure-tensor sRGB conversion paths.
"""
import numpy as np
import torch

from CorridorKeyModule.core.color_utils import (
    create_checkerboard,
    clean_matte,
    clean_matte_gpu,
    despill,
    linear_to_srgb,
    linear_to_srgb_tensor,
    srgb_to_linear,
    srgb_to_linear_tensor,
)


# ── Checkerboard caching ──


class TestCheckerboardCache:
    def test_same_args_return_same_object(self):
        a = create_checkerboard(128, 64)
        b = create_checkerboard(128, 64)
        assert a is b

    def test_different_args_return_different_object(self):
        a = create_checkerboard(128, 64)
        b = create_checkerboard(64, 128)
        assert a is not b

    def test_cached_array_is_readonly(self):
        arr = create_checkerboard(16, 16)
        assert not arr.flags.writeable

    def test_shape_and_dtype(self):
        arr = create_checkerboard(100, 50, checker_size=10)
        assert arr.shape == (50, 100, 3)
        assert arr.dtype == np.float32


# ── Vectorized clean_matte ──


class TestCleanMatteVectorized:
    def test_removes_small_islands(self):
        alpha = np.zeros((100, 100), dtype=np.float32)
        # Large island
        alpha[10:50, 10:50] = 1.0
        # Small island (5x5 = 25 pixels)
        alpha[80:85, 80:85] = 1.0
        result = clean_matte(alpha, area_threshold=100, dilation=0, blur_size=0)
        assert result[40, 40] > 0.5, "Large island should survive"
        assert result[82, 82] == 0.0, "Small island should be removed"

    def test_preserves_3d_shape(self):
        alpha = np.random.rand(50, 50, 1).astype(np.float32)
        result = clean_matte(alpha)
        assert result.ndim == 3
        assert result.shape[2] == 1

    def test_preserves_2d_shape(self):
        alpha = np.random.rand(50, 50).astype(np.float32)
        result = clean_matte(alpha)
        assert result.ndim == 2


# ── Despill in-place (numpy) ──


class TestDespillInPlace:
    def test_numpy_despill_does_not_modify_input(self):
        image = np.array([[[0.2, 0.8, 0.2]]], dtype=np.float32)  # very green
        original = image.copy()
        despill(image, strength=1.0)
        np.testing.assert_array_equal(image, original)

    def test_numpy_despill_reduces_green(self):
        image = np.array([[[0.2, 0.9, 0.2]]], dtype=np.float32)
        result = despill(image, strength=1.0)
        assert result[0, 0, 1] < image[0, 0, 1], "Green should be reduced"

    def test_tensor_despill_matches_numpy(self):
        image_np = np.random.rand(10, 10, 3).astype(np.float32)
        image_np[..., 1] += 0.3  # add green spill
        image_np = np.clip(image_np, 0, 1)
        image_t = torch.from_numpy(image_np.copy())
        result_np = despill(image_np, strength=0.7)
        result_t = despill(image_t, strength=0.7).numpy()
        np.testing.assert_allclose(result_np, result_t, atol=1e-6)


# ── Pure-tensor sRGB functions ──


class TestPureTensorSRGB:
    def test_srgb_to_linear_tensor_matches_dispatcher(self):
        t = torch.rand(10, 10, 3)
        a = srgb_to_linear(t)
        b = srgb_to_linear_tensor(t)
        torch.testing.assert_close(a, b)

    def test_linear_to_srgb_tensor_matches_dispatcher(self):
        t = torch.rand(10, 10, 3)
        a = linear_to_srgb(t)
        b = linear_to_srgb_tensor(t)
        torch.testing.assert_close(a, b)

    def test_roundtrip(self):
        t = torch.rand(5, 5, 3)
        roundtrip = srgb_to_linear_tensor(linear_to_srgb_tensor(t))
        torch.testing.assert_close(roundtrip, t, atol=1e-5, rtol=1e-5)


# ── GPU clean_matte ──


class TestCleanMatteGPU:
    def test_removes_small_islands(self):
        alpha = torch.zeros(100, 100)
        # Large island
        alpha[10:50, 10:50] = 1.0
        # Small island (5x5 = 25 pixels)
        alpha[80:85, 80:85] = 1.0
        result = clean_matte_gpu(alpha, area_threshold=100, dilation=0, blur_size=0)
        assert result[40, 40] > 0.5, "Large island should survive"
        assert result[82, 82] == 0.0, "Small island should be removed"

    def test_preserves_large_region(self):
        alpha = torch.zeros(200, 200)
        alpha[20:180, 20:180] = 1.0
        result = clean_matte_gpu(alpha, area_threshold=300, dilation=0, blur_size=0)
        assert result[100, 100] > 0.5, "Large region must survive"

    def test_preserves_3d_shape(self):
        alpha = torch.rand(50, 50, 1)
        result = clean_matte_gpu(alpha)
        assert result.ndim == 3
        assert result.shape[2] == 1

    def test_preserves_2d_shape(self):
        alpha = torch.rand(50, 50)
        result = clean_matte_gpu(alpha)
        assert result.ndim == 2

    def test_output_stays_on_same_device(self):
        alpha = torch.rand(50, 50)
        result = clean_matte_gpu(alpha)
        assert result.device == alpha.device

    def test_dilation_preserves_edges(self):
        # Gradient alpha that the opening might erode at the edges
        alpha = torch.zeros(100, 100)
        alpha[20:80, 20:80] = 1.0
        alpha[18:20, 20:80] = 0.6  # soft edge that threshold would exclude
        result = clean_matte_gpu(alpha, area_threshold=10, dilation=5, blur_size=0)
        # Dilation should extend safe zone to cover the soft edge rows
        assert result[19, 50] > 0, "Dilation should recover soft-edge pixels"

    def test_blur_produces_gradients(self):
        alpha = torch.zeros(100, 100)
        alpha[30:70, 30:70] = 1.0
        result = clean_matte_gpu(alpha, area_threshold=10, dilation=0, blur_size=5)
        # Inside the region, blur softens the safe_zone boundary
        # At the inner edge (row 30), safe zone is blurred so result < alpha
        assert result[30, 50] < alpha[30, 50], "Blur should soften the safe zone boundary"

    def test_all_zero_unchanged(self):
        alpha = torch.zeros(50, 50)
        result = clean_matte_gpu(alpha, area_threshold=10, dilation=5, blur_size=3)
        assert result.sum() == 0.0

    def test_all_one_unchanged(self):
        alpha = torch.ones(50, 50)
        result = clean_matte_gpu(alpha, area_threshold=10, dilation=0, blur_size=0)
        torch.testing.assert_close(result, alpha)

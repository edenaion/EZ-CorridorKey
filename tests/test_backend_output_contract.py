"""Tests for backend output contracts shared across torch and MLX."""

import sys
import types

import numpy as np

from CorridorKeyModule.backend import (
    _assemble_mlx_output,
    _prepare_mlx_image_u8,
    _try_mlx_float_outputs,
    _wrap_mlx_output,
)
from CorridorKeyModule.core import color_utils as cu


def test_wrap_mlx_output_returns_straight_linear_processed_rgba():
    raw = {
        "fg": np.array([[[128, 64, 32]]], dtype=np.uint8),
        "alpha": np.array([[64]], dtype=np.uint8),
    }
    source = raw["fg"].astype(np.float32) / 255.0

    wrapped = _wrap_mlx_output(
        raw,
        source_image=source,
        input_is_linear=False,
        despill_strength=0.0,
        auto_despeckle=False,
        despeckle_size=400,
    )

    expected_rgb = cu.srgb_to_linear(raw["fg"].astype(np.float32) / 255.0)
    expected_alpha = raw["alpha"].astype(np.float32)[:, :, np.newaxis] / 255.0

    np.testing.assert_allclose(wrapped["processed"][:, :, :3], expected_rgb, atol=1e-6)
    np.testing.assert_allclose(wrapped["processed"][:, :, 3:], expected_alpha, atol=1e-6)


def test_wrap_mlx_output_matches_source_luminance_within_clamp():
    raw = {
        "fg": np.array([[[96, 48, 24]]], dtype=np.uint8),
        "alpha": np.array([[255]], dtype=np.uint8),
    }
    source = np.array([[[0.6, 0.3, 0.15]]], dtype=np.float32)

    wrapped = _wrap_mlx_output(
        raw,
        source_image=source,
        input_is_linear=True,
        despill_strength=0.0,
        auto_despeckle=False,
        despeckle_size=400,
    )

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    src_luma = float(np.sum(source * weights, axis=-1).mean())
    out_luma = float(np.sum(wrapped["processed"][:, :, :3] * weights, axis=-1).mean())

    assert out_luma > float(np.sum(cu.srgb_to_linear(raw["fg"].astype(np.float32) / 255.0) * weights, axis=-1).mean())
    assert out_luma <= src_luma * 1.15


def test_prepare_mlx_image_u8_converts_linear_input_to_srgb_before_quantizing():
    linear = np.array([[[0.18, 0.18, 0.18]]], dtype=np.float32)

    prepared = _prepare_mlx_image_u8(linear, input_is_linear=True)

    expected = (np.clip(cu.linear_to_srgb(linear), 0.0, 1.0) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(prepared, expected)


def test_prepare_mlx_image_u8_leaves_srgb_input_in_srgb_space():
    srgb = np.array([[[0.5, 0.25, 0.125]]], dtype=np.float32)

    prepared = _prepare_mlx_image_u8(srgb, input_is_linear=False)

    expected = (np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(prepared, expected)


def test_assemble_mlx_output_preserves_float_precision_for_processed_rgba():
    fg = np.array([[[0.12345, 0.45678, 0.78901]]], dtype=np.float32)
    alpha = np.array([[[0.54321]]], dtype=np.float32)

    wrapped = _assemble_mlx_output(
        alpha=alpha,
        fg=fg,
        source_image=fg.copy(),
        input_is_linear=False,
        despill_strength=0.0,
        auto_despeckle=False,
        despeckle_size=400,
    )

    expected_rgb = cu.srgb_to_linear(fg)
    np.testing.assert_allclose(wrapped["processed"][:, :, :3], expected_rgb, atol=1e-6)
    np.testing.assert_allclose(wrapped["processed"][:, :, 3:], alpha, atol=1e-6)


def test_try_mlx_float_outputs_uses_raw_model_predictions_before_uint8_quantization(monkeypatch):
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.eval = lambda outputs: None
    fake_mlx_pkg = types.ModuleType("mlx")
    fake_mlx_pkg.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    fake_image_mod = types.ModuleType("corridorkey_mlx.io.image")
    fake_image_mod.preprocess = lambda rgb, mask: {"rgb": rgb, "mask": mask}
    monkeypatch.setitem(sys.modules, "corridorkey_mlx.io.image", fake_image_mod)

    alpha_final = np.array([[[[0.1], [0.9]], [[0.3], [0.7]]]], dtype=np.float32)
    fg_final = np.array(
        [[[[0.2, 0.4, 0.6], [0.8, 0.6, 0.4]], [[0.1, 0.3, 0.5], [0.7, 0.5, 0.3]]]],
        dtype=np.float32,
    )

    class _FakeModel:
        def __call__(self, _x):
            return {
                "alpha_coarse": alpha_final * 0.5,
                "fg_coarse": fg_final * 0.5,
                "alpha_final": alpha_final,
                "fg_final": fg_final,
            }

    class _FakeEngine:
        _img_size = 2
        _use_refiner = True
        _model = _FakeModel()

    image_u8 = np.zeros((2, 2, 3), dtype=np.uint8)
    mask_u8 = np.zeros((2, 2), dtype=np.uint8)

    result = _try_mlx_float_outputs(_FakeEngine(), image_u8, mask_u8, refiner_scale=1.0)

    assert result is not None
    np.testing.assert_allclose(result["alpha"], alpha_final[0], atol=1e-6)
    np.testing.assert_allclose(result["fg"], fg_final[0], atol=1e-6)


def test_try_mlx_float_outputs_handles_resize_path(monkeypatch):
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.eval = lambda outputs: None
    fake_mlx_pkg = types.ModuleType("mlx")
    fake_mlx_pkg.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    fake_image_mod = types.ModuleType("corridorkey_mlx.io.image")
    fake_image_mod.preprocess = lambda rgb, mask: {"rgb": rgb, "mask": mask}
    monkeypatch.setitem(sys.modules, "corridorkey_mlx.io.image", fake_image_mod)

    alpha_final = np.array([[[[0.25]]]], dtype=np.float32)
    fg_final = np.array([[[[0.2, 0.4, 0.6]]]], dtype=np.float32)

    class _FakeModel:
        def __call__(self, _x):
            return {
                "alpha_coarse": alpha_final,
                "fg_coarse": fg_final,
                "alpha_final": alpha_final,
                "fg_final": fg_final,
            }

    class _FakeEngine:
        _img_size = 1
        _use_refiner = True
        _model = _FakeModel()

    image_u8 = np.zeros((2, 2, 3), dtype=np.uint8)
    mask_u8 = np.zeros((2, 2), dtype=np.uint8)

    result = _try_mlx_float_outputs(_FakeEngine(), image_u8, mask_u8, refiner_scale=1.0)

    assert result is not None
    assert result["alpha"].shape == (2, 2, 1)
    assert result["fg"].shape == (2, 2, 3)

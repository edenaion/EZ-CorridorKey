"""Tests for backend output contracts shared across torch and MLX."""

import numpy as np

from CorridorKeyModule.backend import _wrap_mlx_output
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

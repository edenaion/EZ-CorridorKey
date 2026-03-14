"""Tests for backend output contracts shared across torch and MLX."""

import numpy as np

from CorridorKeyModule.backend import _wrap_mlx_output
from CorridorKeyModule.core import color_utils as cu


def test_wrap_mlx_output_returns_straight_linear_processed_rgba():
    raw = {
        "fg": np.array([[[128, 64, 32]]], dtype=np.uint8),
        "alpha": np.array([[64]], dtype=np.uint8),
    }

    wrapped = _wrap_mlx_output(
        raw,
        despill_strength=0.0,
        auto_despeckle=False,
        despeckle_size=400,
    )

    expected_rgb = cu.srgb_to_linear(raw["fg"].astype(np.float32) / 255.0)
    expected_alpha = raw["alpha"].astype(np.float32)[:, :, np.newaxis] / 255.0

    np.testing.assert_allclose(wrapped["processed"][:, :, :3], expected_rgb, atol=1e-6)
    np.testing.assert_allclose(wrapped["processed"][:, :, 3:], expected_alpha, atol=1e-6)

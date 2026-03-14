"""Tests for backend.frame_io colour handling and EXR ingest."""
import os

import numpy as np
import pytest

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2

from backend.frame_io import _linear_to_srgb, _srgb_to_linear, read_image_frame


def _write_exr_or_skip(path: str, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, bgr):
        pytest.skip("OpenCV EXR write support is unavailable in this environment")


class TestLinearToSrgb:
    def test_matches_reference_points(self):
        linear = np.array(
            [0.0, 0.001, 0.0031308, 0.01, 0.18, 0.5, 1.0],
            dtype=np.float32,
        )
        expected = np.array(
            [0.0, 0.01292, 0.04044994, 0.09985282, 0.46135613, 0.7353569, 1.0],
            dtype=np.float32,
        )

        converted = _linear_to_srgb(linear)

        np.testing.assert_allclose(converted, expected, atol=1e-6)


class TestSrgbToLinear:
    def test_matches_reference_points(self):
        srgb = np.array(
            [0.0, 0.01292, 0.04045, 0.09985282, 0.46135613, 0.7353569, 1.0],
            dtype=np.float32,
        )
        expected = np.array(
            [0.0, 0.001, 0.0031308, 0.01, 0.18, 0.5, 1.0],
            dtype=np.float32,
        )

        converted = _srgb_to_linear(srgb)

        np.testing.assert_allclose(converted, expected, atol=1e-6)


class TestReadImageFrame:
    def test_exr_gamma_correction_uses_true_srgb_curve(self, tmp_path):
        rgb = np.array(
            [[[0.001, 0.18, 0.5], [0.0031308, 0.01, 1.0]]],
            dtype=np.float32,
        )
        path = os.path.join(str(tmp_path), "gamma.exr")
        _write_exr_or_skip(path, rgb)

        img = read_image_frame(path, gamma_correct_exr=True)

        assert img is not None
        np.testing.assert_allclose(img, _linear_to_srgb(rgb), atol=2e-4)

    def test_exr_read_preserves_linear_values_when_gamma_disabled(self, tmp_path):
        rgb = np.array(
            [[[0.001, 0.18, 0.5], [0.0031308, 0.01, 1.0]]],
            dtype=np.float32,
        )
        path = os.path.join(str(tmp_path), "linear.exr")
        _write_exr_or_skip(path, rgb)

        img = read_image_frame(path, gamma_correct_exr=False)

        assert img is not None
        np.testing.assert_allclose(img, rgb, atol=2e-4)

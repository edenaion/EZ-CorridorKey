"""Tests for backend.frame_io colour handling and EXR ingest."""
import os

import numpy as np
import pytest

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2

from backend.frame_io import (
    _linear_to_srgb,
    _srgb_to_linear,
    decode_video_mask_frame,
    read_image_frame,
    read_video_mask_at,
)


def _write_exr_or_skip(path: str, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2BGR)
    try:
        success = cv2.imwrite(path, bgr)
    except cv2.error:
        pytest.skip("OpenCV EXR write support is unavailable in this environment")
    if not success:
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


class TestVideoMaskRead:
    def test_decode_video_mask_frame_matches_grayscale_import_behavior(self):
        frame = np.array([[[0, 0, 255]]], dtype=np.uint8)  # pure red in BGR

        mask = decode_video_mask_frame(frame)

        expected = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        np.testing.assert_allclose(mask, expected, atol=1e-6)

    def test_decode_video_mask_frame_uses_alpha_channel_when_present(self):
        frame = np.array([[[10, 20, 30, 128]]], dtype=np.uint8)

        mask = decode_video_mask_frame(frame)

        expected = np.array([[128 / 255.0]], dtype=np.float32)
        np.testing.assert_allclose(mask, expected, atol=1e-6)

    def test_read_video_mask_at_uses_normalized_decode_path(self, monkeypatch):
        frame = np.array([[[0, 255, 0]]], dtype=np.uint8)

        class _FakeCapture:
            def __init__(self, path):
                self.path = path

            def set(self, *_args):
                return True

            def read(self):
                return True, frame

            def release(self):
                return None

        monkeypatch.setattr("backend.frame_io.cv2.VideoCapture", _FakeCapture)

        mask = read_video_mask_at("alpha.mov", 12)

        expected = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        np.testing.assert_allclose(mask, expected, atol=1e-6)

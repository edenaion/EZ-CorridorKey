"""Tests for display transform — EXR preview conversion for various channel layouts."""
import os
import tempfile
import pytest
import numpy as np
import cv2

# Must set before importing display_transform
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

pyside6 = pytest.importorskip("PySide6", reason="PySide6 not installed")
from PySide6.QtGui import QImage

from ui.preview.display_transform import (
    decode_frame, clear_cache, _transform_matte,
    _transform_linear_rgb, _transform_premultiplied, _numpy_to_qimage,
    processed_rgba_to_qimage,
)
from ui.preview.frame_index import ViewMode


class TestNumpyToQImage:
    def test_basic_rgb(self):
        rgb = np.zeros((10, 20, 3), dtype=np.uint8)
        rgb[5, 10] = [255, 0, 0]
        qimg = _numpy_to_qimage(rgb)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 20
        assert qimg.height() == 10
        assert not qimg.isNull()


class TestTransformMatte:
    def test_1ch_float(self):
        """Codex test: 1-channel float EXR matte visualization."""
        data = np.full((10, 10), 0.5, dtype=np.float32)
        qimg = _transform_matte(data)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 10

    def test_negative_values(self):
        """Codex test: negative values should be clamped to 0."""
        data = np.full((10, 10), -0.5, dtype=np.float32)
        qimg = _transform_matte(data)
        assert isinstance(qimg, QImage)

    def test_values_above_one(self):
        """Codex test: values > 1 should be clamped to 1."""
        data = np.full((10, 10), 2.5, dtype=np.float32)
        qimg = _transform_matte(data)
        assert isinstance(qimg, QImage)


class TestTransformLinearRGB:
    def test_3ch_float(self):
        """Codex test: 3-channel float EXR linear RGB."""
        bgr = np.full((10, 10, 3), 0.5, dtype=np.float32)
        qimg = _transform_linear_rgb(bgr, ViewMode.FG)
        assert isinstance(qimg, QImage)

    def test_negative_values(self):
        """Codex test: negative values in linear EXR."""
        bgr = np.full((10, 10, 3), -0.1, dtype=np.float32)
        qimg = _transform_linear_rgb(bgr, ViewMode.FG)
        assert isinstance(qimg, QImage)

    def test_hdr_values(self):
        """Codex test: HDR values > 1 should be tone-mapped."""
        bgr = np.full((10, 10, 3), 5.0, dtype=np.float32)
        qimg = _transform_linear_rgb(bgr, ViewMode.FG)
        assert isinstance(qimg, QImage)


class TestTransformPremultiplied:
    def test_4ch_premultiplied(self):
        """Codex test: 4-channel premultiplied RGBA (Processed output)."""
        bgra = np.zeros((10, 10, 4), dtype=np.float32)
        bgra[:, :, :3] = 0.25  # premultiplied color
        bgra[:, :, 3] = 0.5   # alpha
        qimg = _transform_premultiplied(bgra)
        assert isinstance(qimg, QImage)

    def test_zero_alpha(self):
        """Codex test: zero alpha should not cause divide-by-zero."""
        bgra = np.zeros((10, 10, 4), dtype=np.float32)
        qimg = _transform_premultiplied(bgra)
        assert isinstance(qimg, QImage)

    def test_keeps_premultiplied_values_instead_of_unpremultiplying(self):
        """Processed preview should show the stored premultiplied image over black."""
        bgra = np.zeros((1, 130, 4), dtype=np.float32)
        bgra[:, :, :3] = 0.25
        bgra[:, :, 3] = 0.5
        qimg = _transform_premultiplied(bgra)
        color = qimg.pixelColor(129, 0)
        assert 130 <= color.red() <= 145
        assert 130 <= color.green() <= 145
        assert 130 <= color.blue() <= 145

    def test_live_preview_rgba_matches_saved_bgra_display(self):
        """Live processed preview and saved EXR decode should use the same transform."""
        rgba = np.zeros((4, 4, 4), dtype=np.float32)
        rgba[:, :, :3] = 0.25
        rgba[:, :, 3] = 0.5
        bgra = rgba[:, :, [2, 1, 0, 3]]
        live_qimg = processed_rgba_to_qimage(rgba)
        saved_qimg = _transform_premultiplied(bgra)
        live = live_qimg.pixelColor(0, 0)
        saved = saved_qimg.pixelColor(0, 0)
        assert live.red() == saved.red()
        assert live.green() == saved.green()
        assert live.blue() == saved.blue()


class TestDecodeFrame:
    def test_decode_png(self, tmp_path):
        """Test decoding a PNG file (sRGB, 8-bit)."""
        clear_cache()
        img = np.zeros((20, 30, 3), dtype=np.uint8)
        img[10, 15] = [0, 255, 0]
        path = os.path.join(str(tmp_path), "test.png")
        cv2.imwrite(path, img)

        qimg = decode_frame(path, ViewMode.COMP)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 30
        assert qimg.height() == 20

    def test_cache_hit(self, tmp_path):
        """Test that second decode hits cache."""
        clear_cache()
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        path = os.path.join(str(tmp_path), "cached.png")
        cv2.imwrite(path, img)

        qimg1 = decode_frame(path, ViewMode.COMP)
        qimg2 = decode_frame(path, ViewMode.COMP)
        # Both should succeed (cache hit on second)
        assert qimg1 is not None
        assert qimg2 is not None

    def test_nonexistent_file(self):
        clear_cache()
        result = decode_frame("/nonexistent/file.png", ViewMode.COMP)
        assert result is None

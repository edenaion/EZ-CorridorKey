"""Tests for format validation behavior in _write_image and OutputConfig.

Key design facts documented by these tests:
- _write_image has exactly two branches: "exr" and everything-else.
- Any format string that is not "exr" (including "tiff", "bmp", "webp", "")
  silently falls through to the PNG/uint8 path.  The actual file extension is
  determined by the caller (_write_outputs), not by _write_image itself.
- OutputConfig accepts arbitrary format strings; no validation is enforced at
  construction time.  Callers are responsible for passing valid values.
"""

from unittest.mock import patch

import numpy as np

from backend.service import CorridorKeyService, OutputConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _float32_frame(h: int = 4, w: int = 4, c: int = 3) -> np.ndarray:
    """Return a small float32 array whose values span [0, 1]."""
    rng = np.random.default_rng(0)
    return rng.random((h, w, c), dtype=np.float32).astype(np.float32)


def _make_service() -> CorridorKeyService:
    return CorridorKeyService()


# ---------------------------------------------------------------------------
# TestWriteImageFormats
# ---------------------------------------------------------------------------


class TestWriteImageFormats:
    """Unit tests for _write_image format-routing logic."""

    def test_exr_format_calls_write_exr(self, tmp_path):
        """'exr' must call write_exr (OpenEXR library) instead of cv2.imwrite."""
        service = _make_service()
        img = _float32_frame()
        out_path = str(tmp_path / "frame.exr")

        with patch("backend.service.write_exr", return_value=True) as mock_exr:
            service._write_image(img, out_path, "exr", "clip", 0)

        assert mock_exr.call_count == 1
        _, args, kwargs = mock_exr.mock_calls[0]
        assert args[0] == out_path
        assert "compression" in kwargs, "compression must be passed as keyword arg"
        assert kwargs["compression"] == "dwab"

    def test_png_format_converts_float_to_uint8(self, tmp_path):
        """'png' must clip to [0,1] and cast to uint8 before writing."""
        service = _make_service()
        # Values outside [0, 1] to verify clipping is applied
        img = np.array([[[2.0, -0.5, 0.5]]], dtype=np.float32)
        out_path = str(tmp_path / "frame.png")

        captured = {}

        def fake_imwrite(path, arr, *args):
            captured["arr"] = arr
            return True

        with patch("backend.service.cv2.imwrite", side_effect=fake_imwrite):
            service._write_image(img, out_path, "png", "clip", 0)

        written = captured["arr"]
        assert written.dtype == np.uint8, "png path must produce uint8 output"
        # 2.0 clipped → 1.0 → 255; -0.5 clipped → 0.0 → 0; 0.5 → 127/128
        assert written[0, 0, 0] == 255
        assert written[0, 0, 1] == 0
        assert written[0, 0, 2] == 127

    def test_unknown_format_falls_through_to_png_path(self, tmp_path):
        """Non-'exr' formats like 'tiff' and 'bmp' silently use the PNG branch.

        _write_image has no format registry — any string that is not the
        literal "exr" falls into the else branch.  This means the caller
        controls the actual file extension (via the path argument); the
        format string only determines whether EXR flags are forwarded.
        """
        service = _make_service()
        img = _float32_frame()

        captured_calls = {}

        def fake_imwrite(path, arr, *extra_args):
            captured_calls[path] = {"arr": arr, "extra_args": extra_args}
            return True

        with patch("backend.service.cv2.imwrite", side_effect=fake_imwrite):
            tiff_path = str(tmp_path / "frame.tiff")
            service._write_image(img, tiff_path, "tiff", "clip", 0)

            bmp_path = str(tmp_path / "frame.bmp")
            service._write_image(img, bmp_path, "bmp", "clip", 0)

        for path, info in captured_calls.items():
            # PNG branch: no extra flags passed
            assert info["extra_args"] == (), f"Unknown format at {path} must not forward EXR flags"
            # PNG branch: array must be uint8
            assert info["arr"].dtype == np.uint8, (
                f"Unknown format at {path} must produce uint8 output"
            )

    def test_empty_string_format_falls_through_to_png_path(self, tmp_path):
        """Empty format string is not 'exr', so it takes the PNG else branch."""
        service = _make_service()
        img = _float32_frame()
        out_path = str(tmp_path / "frame.")

        captured = {}

        def fake_imwrite(path, arr, *extra_args):
            captured["dtype"] = arr.dtype
            captured["extra_args"] = extra_args
            return True

        with patch("backend.service.cv2.imwrite", side_effect=fake_imwrite):
            service._write_image(img, out_path, "", "clip", 0)

        assert captured["dtype"] == np.uint8
        assert captured["extra_args"] == ()

    def test_png_path_skips_conversion_when_already_uint8(self, tmp_path):
        """If the array is already uint8, the PNG branch must write it unchanged."""
        service = _make_service()
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        out_path = str(tmp_path / "frame.png")

        captured = {}

        def fake_imwrite(path, arr, *extra_args):
            captured["arr"] = arr
            return True

        with patch("backend.service.cv2.imwrite", side_effect=fake_imwrite):
            service._write_image(img, out_path, "png", "clip", 0)

        written = captured["arr"]
        assert written.dtype == np.uint8
        np.testing.assert_array_equal(written, img)


# ---------------------------------------------------------------------------
# TestOutputConfigNonStandardFormats
# ---------------------------------------------------------------------------


class TestOutputConfigNonStandardFormats:
    """Verify OutputConfig accepts arbitrary format strings without error."""

    def test_non_standard_format_does_not_break_enabled_outputs(self):
        """enabled_outputs reflects enabled flags regardless of format string."""
        cfg = OutputConfig(
            fg_enabled=True,
            fg_format="tiff",
            matte_enabled=True,
            matte_format="bmp",
            comp_enabled=False,
            comp_format="webp",
            processed_enabled=True,
            processed_format="jp2",
        )
        assert cfg.enabled_outputs == ["fg", "matte", "processed"]

    def test_to_dict_from_dict_roundtrip_with_garbage_format(self):
        """Arbitrary format strings survive a to_dict/from_dict round-trip."""
        original = OutputConfig(
            fg_enabled=True,
            fg_format="tiff",
            matte_enabled=False,
            matte_format="???",
            comp_enabled=True,
            comp_format="",
            processed_enabled=True,
            processed_format="jp2000",
        )
        d = original.to_dict()
        restored = OutputConfig.from_dict(d)

        assert restored.fg_format == "tiff"
        assert restored.matte_format == "???"
        assert restored.comp_format == ""
        assert restored.processed_format == "jp2000"
        assert restored.fg_enabled is True
        assert restored.matte_enabled is False
        assert restored.comp_enabled is True
        assert restored.processed_enabled is True

    def test_from_dict_roundtrip_preserves_enabled_outputs(self):
        """enabled_outputs computed from restored config matches original."""
        original = OutputConfig(
            fg_enabled=True,
            fg_format="tiff",
            matte_enabled=True,
            matte_format="bmp",
            comp_enabled=False,
            comp_format="webp",
            processed_enabled=False,
            processed_format="jp2",
        )
        restored = OutputConfig.from_dict(original.to_dict())
        assert restored.enabled_outputs == original.enabled_outputs
        assert restored.enabled_outputs == ["fg", "matte"]

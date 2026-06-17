"""Round-trip tests for the unicode-safe media I/O facade.

cv2.imread / cv2.imwrite use a narrow (ANSI codepage) file API on Windows and
fail on any path containing characters outside that codepage. The facade
(backend.frame_io.imread_unicode / imwrite_unicode / open_video) routes ASCII
paths to the original cv2 calls (byte-identical) and non-ASCII paths through an
in-memory buffer. These tests assert correctness across the scripts used by the
14 shipped UI languages plus a few extra edge cases (RTL, emoji, spaces).
"""
import os
import shutil
import tempfile

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import pytest

from backend.frame_io import imread_unicode, imwrite_unicode, open_video

# One sample folder/file name per script we must support. Keyed by a readable
# label; the value lands in both a directory name and a file stem so both the
# parent path and the leaf are exercised.
SCRIPT_SAMPLES = {
    "spanish": "niña_corazón",
    "french": "café_éàü",
    "german": "Grüße_Straße_öä",
    "portuguese": "ação_são",
    "russian": "Свадебное_цветное",
    "ukrainian": "Привіт_їжак",
    "chinese": "婚礼_视频",
    "japanese": "日本語のフレーム",
    "korean": "결혼식_영상",
    "arabic": "حفل_زفاف",
    "hindi": "नमस्ते_फ्रेम",
    "thai": "งานแต่งงาน",
    "spaces": "my project name",
    "emoji": "wedding_🎬_clip",
}


@pytest.fixture
def workdir():
    d = tempfile.mkdtemp(prefix="ck_unicode_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _nested(workdir: str, name: str, ext: str) -> str:
    """Build a non-ASCII directory + non-ASCII file stem path."""
    sub = os.path.join(workdir, name)
    os.makedirs(sub, exist_ok=True)
    return os.path.join(sub, f"{name}{ext}")


@pytest.mark.parametrize("label,name", list(SCRIPT_SAMPLES.items()))
def test_ldr_png_roundtrip(workdir, label, name):
    """8-bit BGR PNG round-trips byte-exact through a non-ASCII path."""
    img = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    path = _nested(workdir, name, ".png")
    assert imwrite_unicode(path, img), f"write failed for {label}"
    assert os.path.isfile(path)
    got = imread_unicode(path, cv2.IMREAD_COLOR)
    assert got is not None, f"read returned None for {label}"
    assert np.array_equal(got, img), f"pixels differ for {label}"


@pytest.mark.parametrize("label,name", list(SCRIPT_SAMPLES.items()))
def test_exr_float_roundtrip(workdir, label, name):
    """Half-float EXR (the app's primary format) preserves values > 1.0."""
    img = np.random.rand(8, 8, 3).astype(np.float32)
    img[0, 0] = [5.0, 0.5, 1234.0]  # values > 1 must survive (no clamp to uint8)
    path = _nested(workdir, name, ".exr")
    ok = imwrite_unicode(
        path, img, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]
    )
    assert ok, f"EXR write failed for {label}"
    got = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    assert got is not None, f"EXR read returned None for {label}"
    assert got.dtype == np.float32, f"EXR dtype not float for {label}"
    # Half precision: compare with a tolerance that fits float16 rounding.
    assert got[0, 0, 0] == pytest.approx(5.0, rel=1e-2)
    assert got[0, 0, 2] == pytest.approx(1234.0, rel=1e-2)


@pytest.mark.parametrize("label,name", list(SCRIPT_SAMPLES.items()))
def test_16bit_png_mask_roundtrip(workdir, label, name):
    """16-bit PNG mattes keep uint16 precision through a non-ASCII path."""
    mask = (np.random.rand(8, 8) * 65535).astype(np.uint16)
    path = _nested(workdir, name, ".png")
    assert imwrite_unicode(path, mask)
    got = imread_unicode(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    assert got is not None and got.dtype == np.uint16
    assert np.array_equal(got, mask), f"16-bit mask differs for {label}"


def test_ascii_fast_path_identical_to_cv2(workdir):
    """ASCII paths must produce byte-identical results to raw cv2.imread."""
    img = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
    path = os.path.join(workdir, "ascii_frame.png")
    imwrite_unicode(path, img)
    assert np.array_equal(
        imread_unicode(path, cv2.IMREAD_COLOR),
        cv2.imread(path, cv2.IMREAD_COLOR),
    )


def test_missing_file_returns_none(workdir):
    """Matches cv2.imread's None-on-failure contract for missing files."""
    assert imread_unicode(os.path.join(workdir, "Свадебное_missing.png")) is None
    assert imread_unicode(os.path.join(workdir, "ascii_missing.png")) is None


@pytest.mark.parametrize("label,name", list(SCRIPT_SAMPLES.items()))
def test_open_video_non_ascii(workdir, label, name):
    """open_video can read a frame from a video at a non-ASCII path.

    Generates a tiny clip via cv2.VideoWriter at an ASCII temp path (writers
    have the same narrow-path limitation), then copies it to the non-ASCII path
    under test and opens it through the facade.
    """
    ascii_src = os.path.join(workdir, "src.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(ascii_src, fourcc, 10.0, (32, 32))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter unavailable in this build")
    for _ in range(5):
        writer.write((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
    writer.release()
    if not os.path.isfile(ascii_src) or os.path.getsize(ascii_src) == 0:
        pytest.skip("video encoding produced no file")

    dst = _nested(workdir, name, ".mp4")
    shutil.copy2(ascii_src, dst)
    cap = open_video(dst)
    try:
        assert cap.isOpened(), f"open_video failed for {label}"
        ret, frame = cap.read()
        assert ret and frame is not None, f"frame read failed for {label}"
        assert frame.shape == (32, 32, 3)
    finally:
        cap.release()

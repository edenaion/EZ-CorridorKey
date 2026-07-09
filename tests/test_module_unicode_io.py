"""Regression tests for issue #184: AI model modules on non-ASCII paths.

The #167 unicode facade only covered backend/ and ui/. The vendored AI model
modules (gvm_core, modules/*, VideoMaMaInferenceModule) kept raw cv2 file
calls, so every AI alpha-hint method failed for users whose Windows username
contains non-ASCII characters (accented Latin, Cyrillic, CJK, ...) while
extraction and chroma key (facade paths) worked. These tests exercise the
exact module-level readers the app calls, rooted in non-ASCII directories.
"""
import os
import shutil
import tempfile

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import pytest

from backend.frame_io import imwrite_unicode

# Representative scripts: accented Latin (the #184 reporter's case), Cyrillic,
# and CJK. The full 14-script matrix is covered by test_unicode_paths.py; here
# the point is the module code path, not the codec.
SCRIPT_SAMPLES = {
    "italian": "Pontirólo_città",
    "russian": "Свадебное_видео",
    "chinese": "婚礼_视频",
}


@pytest.fixture
def workdir():
    d = tempfile.mkdtemp(prefix="ck_module_unicode_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _frames_dir(workdir: str, name: str) -> str:
    """Create a non-ASCII Frames-style directory with two tiny EXR frames."""
    frames = os.path.join(workdir, name, "Frames")
    os.makedirs(frames, exist_ok=True)
    for i in range(2):
        img = np.random.rand(8, 8, 3).astype(np.float32)
        ok = imwrite_unicode(
            os.path.join(frames, f"frame_{i:06d}.exr"),
            img,
            [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF],
        )
        assert ok, "test fixture EXR write failed"
    return frames


@pytest.mark.parametrize("label,name", list(SCRIPT_SAMPLES.items()))
def test_gvm_image_sequence_reader_non_ascii(workdir, label, name):
    """GVM's ImageSequenceReader reads EXR frames from a non-ASCII dir.

    This is the exact crash site from issue #184: 'Failed to read
    ...frame_000000.exr' raised from ImageSequenceReader.__getitem__.
    """
    torch = pytest.importorskip("torch")  # noqa: F841 — Dataset base class
    from gvm_core.gvm.utils.inference_utils import ImageSequenceReader

    frames = _frames_dir(workdir, name)
    reader = ImageSequenceReader(frames)

    assert len(reader) == 2, f"reader found no frames for {label}"
    assert reader.origin_shape == (8, 8), f"origin_shape failed for {label}"
    first = reader[0]  # raised ValueError('Failed to read ...') before the fix
    assert first["filename"] == "frame_000000.exr"
    assert first["image"].size == (8, 8), f"unexpected frame size for {label}"


@pytest.mark.parametrize("label,name", list(SCRIPT_SAMPLES.items()))
def test_matanyone2_read_frames_non_ascii(workdir, label, name):
    """MatAnyone2's read_frame_from_videos loads PNGs from a non-ASCII dir.

    Before the fix the raw cv2.imread returned None and the [..., [2,1,0]]
    swizzle crashed with 'NoneType is not subscriptable'.
    """
    pytest.importorskip("torchvision")
    # The app puts modules/MatAnyone2Module on sys.path before importing
    # matanyone2 (backend/service/helpers.py); mirror that here.
    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mod_dir = os.path.join(repo_root, "modules", "MatAnyone2Module")
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    from matanyone2.utils.inference_utils import read_frame_from_videos

    frame_root = os.path.join(workdir, name, "frames_png")
    os.makedirs(frame_root, exist_ok=True)
    for i in range(2):
        img = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
        assert imwrite_unicode(os.path.join(frame_root, f"{i:05d}.png"), img)

    frames, fps, length, video_name = read_frame_from_videos(frame_root)
    assert length == 2, f"frame count wrong for {label}"
    assert frames.shape == (2, 3, 8, 8), f"tensor shape wrong for {label}"

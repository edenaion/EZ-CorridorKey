"""Shared fixtures for ez-CorridorKey backend tests."""
import os
import tempfile

import cv2
import numpy as np
import pytest

from backend.clip_state import ClipAsset, ClipEntry, ClipState


@pytest.fixture
def sample_frame():
    """4x4 float32 RGB frame in [0, 1]."""
    return np.random.rand(4, 4, 3).astype(np.float32)


@pytest.fixture
def sample_mask():
    """4x4 float32 single-channel mask in [0, 1]."""
    return np.random.rand(4, 4).astype(np.float32)


@pytest.fixture
def tmp_clip_dir(sample_frame, sample_mask):
    """Temp directory with real Input/AlphaHint/Output structure and tiny PNGs.

    Layout:
        clip_root/
            Input/
                frame_00000.png  (4x4 RGB)
                frame_00001.png
                frame_00002.png
            AlphaHint/
                _alphaHint_00000.png  (4x4 grayscale)
                _alphaHint_00001.png
                _alphaHint_00002.png
            Output/
                FG/
                Matte/
                Comp/
                Processed/
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        clip_root = os.path.join(tmpdir, "test_clip")
        input_dir = os.path.join(clip_root, "Input")
        alpha_dir = os.path.join(clip_root, "AlphaHint")
        output_root = os.path.join(clip_root, "Output")

        for d in [input_dir, alpha_dir]:
            os.makedirs(d)
        for subdir in ("FG", "Matte", "Comp", "Processed"):
            os.makedirs(os.path.join(output_root, subdir))

        # Write 3 tiny PNG frames
        for i in range(3):
            # Input frame (BGR for cv2)
            img_bgr = (sample_frame * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(input_dir, f"frame_{i:05d}.png"), img_bgr)

            # Alpha frame (grayscale)
            mask_u8 = (sample_mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(alpha_dir, f"_alphaHint_{i:05d}.png"), mask_u8)

        yield clip_root


@pytest.fixture
def sample_clip(tmp_clip_dir):
    """ClipEntry with populated assets pointing to tmp_clip_dir."""
    input_dir = os.path.join(tmp_clip_dir, "Input")
    alpha_dir = os.path.join(tmp_clip_dir, "AlphaHint")

    clip = ClipEntry(
        name="test_clip",
        root_path=tmp_clip_dir,
        state=ClipState.READY,
        input_asset=ClipAsset(input_dir, "sequence"),
        alpha_asset=ClipAsset(alpha_dir, "sequence"),
    )
    return clip

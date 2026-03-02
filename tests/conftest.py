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


@pytest.fixture
def tmp_project_dir(sample_frame, sample_mask):
    """Temp directory with v1 project structure (Frames/, Source/).

    Layout:
        project_root/
            Source/
                test_video.mp4 (dummy)
            Frames/
                frame_000000.png (4x4 RGB)
                frame_000001.png
                frame_000002.png
            AlphaHint/
                _alphaHint_000000.png (4x4 grayscale)
                _alphaHint_000001.png
                _alphaHint_000002.png
            Output/
                FG/ Matte/ Comp/ Processed/
            project.json
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = os.path.join(tmpdir, "2026-03-01_093000_test")
        source_dir = os.path.join(project_root, "Source")
        frames_dir = os.path.join(project_root, "Frames")
        alpha_dir = os.path.join(project_root, "AlphaHint")
        output_root = os.path.join(project_root, "Output")

        for d in [source_dir, frames_dir, alpha_dir]:
            os.makedirs(d)
        for subdir in ("FG", "Matte", "Comp", "Processed"):
            os.makedirs(os.path.join(output_root, subdir))

        # Dummy source video
        with open(os.path.join(source_dir, "test_video.mp4"), "wb") as f:
            f.write(b"\x00" * 100)

        # Write frames and alpha
        for i in range(3):
            img_bgr = (sample_frame * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(frames_dir, f"frame_{i:06d}.png"), img_bgr)

            mask_u8 = (sample_mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(alpha_dir, f"_alphaHint_{i:06d}.png"), mask_u8)

        # project.json
        import json
        with open(os.path.join(project_root, "project.json"), "w") as f:
            json.dump({
                "version": 1,
                "display_name": "Test Project",
                "source": {"filename": "test_video.mp4"},
            }, f)

        yield project_root


@pytest.fixture
def tmp_v2_project_dir(sample_frame, sample_mask):
    """Temp directory with v2 project structure (clips/ with nested clip subdirs).

    Layout:
        project_root/
            project.json (v2)
            clips/
                test_clip/
                    Source/test_video.mp4
                    Frames/frame_000000.png ...
                    AlphaHint/_alphaHint_000000.png ...
                    Output/FG/ Matte/ Comp/ Processed/
                    clip.json
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = os.path.join(tmpdir, "2026-03-01_093000_test")
        clip_dir = os.path.join(project_root, "clips", "test_clip")
        source_dir = os.path.join(clip_dir, "Source")
        frames_dir = os.path.join(clip_dir, "Frames")
        alpha_dir = os.path.join(clip_dir, "AlphaHint")
        output_root = os.path.join(clip_dir, "Output")

        for d in [source_dir, frames_dir, alpha_dir]:
            os.makedirs(d)
        for subdir in ("FG", "Matte", "Comp", "Processed"):
            os.makedirs(os.path.join(output_root, subdir))

        with open(os.path.join(source_dir, "test_video.mp4"), "wb") as f:
            f.write(b"\x00" * 100)

        for i in range(3):
            img_bgr = (sample_frame * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(frames_dir, f"frame_{i:06d}.png"), img_bgr)

            mask_u8 = (sample_mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(alpha_dir, f"_alphaHint_{i:06d}.png"), mask_u8)

        import json
        with open(os.path.join(project_root, "project.json"), "w") as f:
            json.dump({
                "version": 2,
                "display_name": "Test Project",
                "clips": ["test_clip"],
            }, f)

        with open(os.path.join(clip_dir, "clip.json"), "w") as f:
            json.dump({
                "source": {"filename": "test_video.mp4", "copied": True},
            }, f)

        yield project_root

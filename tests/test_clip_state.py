"""Tests for backend.clip_state module — state machine transitions."""
import os
import tempfile

import pytest

from backend.clip_state import (
    ClipAsset,
    ClipEntry,
    ClipState,
    scan_clips_dir,
)
from backend.errors import InvalidStateTransitionError, ClipScanError


# --- ClipState transitions ---

class TestClipStateTransitions:
    def _make_clip(self, state: ClipState = ClipState.RAW) -> ClipEntry:
        clip = ClipEntry(name="test_shot", root_path="/tmp/test_shot")
        clip.state = state
        return clip

    def test_raw_to_masked(self):
        clip = self._make_clip(ClipState.RAW)
        clip.transition_to(ClipState.MASKED)
        assert clip.state == ClipState.MASKED

    def test_raw_to_ready(self):
        clip = self._make_clip(ClipState.RAW)
        clip.transition_to(ClipState.READY)
        assert clip.state == ClipState.READY

    def test_masked_to_ready(self):
        clip = self._make_clip(ClipState.MASKED)
        clip.transition_to(ClipState.READY)
        assert clip.state == ClipState.READY

    def test_ready_to_complete(self):
        clip = self._make_clip(ClipState.READY)
        clip.transition_to(ClipState.COMPLETE)
        assert clip.state == ClipState.COMPLETE

    def test_ready_to_error(self):
        clip = self._make_clip(ClipState.READY)
        clip.transition_to(ClipState.ERROR)
        assert clip.state == ClipState.ERROR

    def test_error_to_ready(self):
        clip = self._make_clip(ClipState.ERROR)
        clip.transition_to(ClipState.READY)
        assert clip.state == ClipState.READY

    def test_complete_to_ready_for_reprocess(self):
        """Phase 2: COMPLETE clips can be reprocessed with different params."""
        clip = self._make_clip(ClipState.COMPLETE)
        clip.transition_to(ClipState.READY)
        assert clip.state == ClipState.READY

    def test_complete_to_error_invalid(self):
        clip = self._make_clip(ClipState.COMPLETE)
        with pytest.raises(InvalidStateTransitionError):
            clip.transition_to(ClipState.ERROR)

    def test_raw_to_complete_invalid(self):
        clip = self._make_clip(ClipState.RAW)
        with pytest.raises(InvalidStateTransitionError):
            clip.transition_to(ClipState.COMPLETE)

    def test_raw_to_error_on_gvm_failure(self):
        """Phase 2: RAW clips can error when GVM fails."""
        clip = self._make_clip(ClipState.RAW)
        clip.transition_to(ClipState.ERROR)
        assert clip.state == ClipState.ERROR

    def test_masked_to_error_on_videomama_failure(self):
        """Phase 2: MASKED clips can error when VideoMaMa fails."""
        clip = self._make_clip(ClipState.MASKED)
        clip.transition_to(ClipState.ERROR)
        assert clip.state == ClipState.ERROR

    def test_error_to_raw_for_retry(self):
        """Phase 2: ERROR clips can go back to RAW for fresh retry."""
        clip = self._make_clip(ClipState.ERROR)
        clip.transition_to(ClipState.RAW)
        assert clip.state == ClipState.RAW

    def test_masked_to_complete_invalid(self):
        clip = self._make_clip(ClipState.MASKED)
        with pytest.raises(InvalidStateTransitionError):
            clip.transition_to(ClipState.COMPLETE)

    def test_set_error_stores_message(self):
        clip = self._make_clip(ClipState.READY)
        clip.set_error("VRAM exhausted")
        assert clip.state == ClipState.ERROR
        assert clip.error_message == "VRAM exhausted"

    def test_transition_clears_error(self):
        clip = self._make_clip(ClipState.READY)
        clip.set_error("some error")
        clip.transition_to(ClipState.READY)  # ERROR → READY
        assert clip.error_message is None


# --- ClipEntry asset scanning ---

class TestClipEntryFindAssets:
    def test_finds_input_sequence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shot_dir = os.path.join(tmpdir, "shot1")
            input_dir = os.path.join(shot_dir, "Input")
            os.makedirs(input_dir)
            # Create dummy frames
            for i in range(5):
                with open(os.path.join(input_dir, f"{i:05d}.png"), "w") as f:
                    f.write("dummy")

            clip = ClipEntry(name="shot1", root_path=shot_dir)
            clip.find_assets()

            assert clip.input_asset is not None
            assert clip.input_asset.asset_type == 'sequence'
            assert clip.input_asset.frame_count == 5
            assert clip.state == ClipState.RAW

    def test_finds_alpha_hint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shot_dir = os.path.join(tmpdir, "shot1")
            input_dir = os.path.join(shot_dir, "Input")
            alpha_dir = os.path.join(shot_dir, "AlphaHint")
            os.makedirs(input_dir)
            os.makedirs(alpha_dir)

            for i in range(3):
                with open(os.path.join(input_dir, f"{i:05d}.exr"), "w") as f:
                    f.write("dummy")
                with open(os.path.join(alpha_dir, f"{i:05d}.png"), "w") as f:
                    f.write("dummy")

            clip = ClipEntry(name="shot1", root_path=shot_dir)
            clip.find_assets()

            assert clip.alpha_asset is not None
            assert clip.state == ClipState.READY

    def test_empty_input_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shot_dir = os.path.join(tmpdir, "shot1")
            input_dir = os.path.join(shot_dir, "Input")
            os.makedirs(input_dir)
            # Empty Input dir

            clip = ClipEntry(name="shot1", root_path=shot_dir)
            with pytest.raises(ClipScanError, match="empty"):
                clip.find_assets()

    def test_no_input_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shot_dir = os.path.join(tmpdir, "shot1")
            os.makedirs(shot_dir)
            # No Input at all

            clip = ClipEntry(name="shot1", root_path=shot_dir)
            with pytest.raises(ClipScanError, match="no Input"):
                clip.find_assets()

    def test_finds_frames_dir(self):
        """New format: Frames/ is preferred over Input/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shot_dir = os.path.join(tmpdir, "shot1")
            frames_dir = os.path.join(shot_dir, "Frames")
            os.makedirs(frames_dir)
            for i in range(3):
                with open(os.path.join(frames_dir, f"frame_{i:06d}.png"), "w") as f:
                    f.write("dummy")

            clip = ClipEntry(name="shot1", root_path=shot_dir)
            clip.find_assets()

            assert clip.input_asset is not None
            assert clip.input_asset.asset_type == "sequence"
            assert clip.input_asset.frame_count == 3
            assert clip.state == ClipState.RAW

    def test_frames_preferred_over_input(self):
        """When both Frames/ and Input/ exist, Frames/ wins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shot_dir = os.path.join(tmpdir, "shot1")
            frames_dir = os.path.join(shot_dir, "Frames")
            input_dir = os.path.join(shot_dir, "Input")
            os.makedirs(frames_dir)
            os.makedirs(input_dir)
            for i in range(3):
                with open(os.path.join(frames_dir, f"frame_{i:06d}.png"), "w") as f:
                    f.write("dummy")
            with open(os.path.join(input_dir, "old_frame.png"), "w") as f:
                f.write("dummy")

            clip = ClipEntry(name="shot1", root_path=shot_dir)
            clip.find_assets()

            assert clip.input_asset.path == frames_dir
            assert clip.input_asset.frame_count == 3

    def test_finds_source_video(self):
        """New format: Source/ directory with video file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shot_dir = os.path.join(tmpdir, "shot1")
            source_dir = os.path.join(shot_dir, "Source")
            os.makedirs(source_dir)
            with open(os.path.join(source_dir, "video.mp4"), "wb") as f:
                f.write(b"\x00" * 100)

            clip = ClipEntry(name="shot1", root_path=shot_dir)
            clip.find_assets()

            assert clip.input_asset is not None
            assert clip.input_asset.asset_type == "video"
            assert clip.state == ClipState.EXTRACTING

    def test_source_dir_no_video_raises(self):
        """Source/ with no video files should raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shot_dir = os.path.join(tmpdir, "shot1")
            source_dir = os.path.join(shot_dir, "Source")
            os.makedirs(source_dir)
            # Put a non-video file
            with open(os.path.join(source_dir, "readme.txt"), "w") as f:
                f.write("not a video")

            clip = ClipEntry(name="shot1", root_path=shot_dir)
            with pytest.raises(ClipScanError, match="Source"):
                clip.find_assets()

    def test_display_name_from_project_json(self):
        """find_assets picks up display_name from project.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shot_dir = os.path.join(tmpdir, "2026-03-01_093000_test")
            input_dir = os.path.join(shot_dir, "Input")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "frame.png"), "w") as f:
                f.write("dummy")

            # Write project.json with display_name
            import json
            with open(os.path.join(shot_dir, "project.json"), "w") as f:
                json.dump({"display_name": "My Custom Name"}, f)

            clip = ClipEntry(name="2026-03-01_093000_test", root_path=shot_dir)
            clip.find_assets()

            assert clip.name == "My Custom Name"


# --- scan_clips_dir ---

class TestScanClipsDir:
    def test_scans_multiple_clips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["shot_a", "shot_b", "shot_c"]:
                input_dir = os.path.join(tmpdir, name, "Input")
                os.makedirs(input_dir)
                with open(os.path.join(input_dir, "00000.png"), "w") as f:
                    f.write("dummy")

            clips = scan_clips_dir(tmpdir)
            assert len(clips) == 3
            names = {c.name for c in clips}
            assert names == {"shot_a", "shot_b", "shot_c"}

    def test_skips_hidden_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Regular clip
            input_dir = os.path.join(tmpdir, "shot1", "Input")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "00000.png"), "w") as f:
                f.write("dummy")
            # Hidden dir
            os.makedirs(os.path.join(tmpdir, ".hidden"))
            # Underscore dir
            os.makedirs(os.path.join(tmpdir, "_internal"))

            clips = scan_clips_dir(tmpdir)
            assert len(clips) == 1
            assert clips[0].name == "shot1"

    def test_missing_dir_returns_empty(self):
        clips = scan_clips_dir("/nonexistent/path")
        assert clips == []

    def test_allow_standalone_videos_false(self):
        """Projects root: loose video files at top level are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # A proper project folder
            proj_dir = os.path.join(tmpdir, "2026-01-01_120000_test")
            input_dir = os.path.join(proj_dir, "Input")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "00000.png"), "w") as f:
                f.write("dummy")

            # A loose video file (should be ignored)
            with open(os.path.join(tmpdir, "stray.mp4"), "wb") as f:
                f.write(b"\x00" * 100)

            clips = scan_clips_dir(tmpdir, allow_standalone_videos=False)
            names = {c.name for c in clips}
            assert "stray" not in names

    def test_new_format_project_scans(self):
        """New-format project with Source/ and Frames/ scans correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = os.path.join(tmpdir, "2026-03-01_093000_Woman")
            source_dir = os.path.join(proj_dir, "Source")
            frames_dir = os.path.join(proj_dir, "Frames")
            os.makedirs(source_dir)
            os.makedirs(frames_dir)
            # Source video
            with open(os.path.join(source_dir, "Woman.mp4"), "wb") as f:
                f.write(b"\x00" * 100)
            # Extracted frames
            for i in range(5):
                with open(os.path.join(frames_dir, f"frame_{i:06d}.png"), "w") as f:
                    f.write("dummy")

            clips = scan_clips_dir(proj_dir + "/..")
            found = [c for c in clips if "Woman" in c.root_path]
            assert len(found) == 1
            clip = found[0]
            # Frames/ should be found (sequence), not Source/ video
            assert clip.input_asset.asset_type == "sequence"
            assert clip.input_asset.frame_count == 5
            assert clip.state == ClipState.RAW

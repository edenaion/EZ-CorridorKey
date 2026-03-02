"""Tests for backend.clip_state module — state machine transitions."""
import json
import os
import tempfile

import pytest

from backend.clip_state import (
    ClipAsset,
    ClipEntry,
    ClipState,
    scan_clips_dir,
    scan_project_clips,
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

    def test_v1_format_project_scans(self):
        """Legacy v1 project with Source/ and Frames/ scans correctly."""
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

    def test_v2_project_scans_nested_clips(self):
        """v2 project with clips/ subdir scans all clip subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = os.path.join(tmpdir, "2026-03-01_093000_Batch")
            clips_dir = os.path.join(proj_dir, "clips")

            # Create 2 clips inside clips/
            for name in ["Woman_Jumps", "Man_Walks"]:
                clip_dir = os.path.join(clips_dir, name)
                source_dir = os.path.join(clip_dir, "Source")
                os.makedirs(source_dir)
                with open(os.path.join(source_dir, f"{name}.mp4"), "wb") as f:
                    f.write(b"\x00" * 100)

            # project.json at project root
            with open(os.path.join(proj_dir, "project.json"), "w") as f:
                json.dump({"version": 2, "clips": ["Woman_Jumps", "Man_Walks"]}, f)

            # Scan from Projects root (parent of proj_dir)
            clips = scan_clips_dir(tmpdir, allow_standalone_videos=False)
            assert len(clips) == 2
            names = {c.name for c in clips}
            assert "Woman_Jumps" in names
            assert "Man_Walks" in names

            # root_path should point to clip subfolder, not project dir
            for clip in clips:
                assert "clips" in clip.root_path
                assert clip.state == ClipState.EXTRACTING

    def test_v2_project_clips_with_frames(self):
        """v2 clips that have extracted frames scan as RAW."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = os.path.join(tmpdir, "proj")
            clip_dir = os.path.join(proj_dir, "clips", "my_clip")
            frames_dir = os.path.join(clip_dir, "Frames")
            os.makedirs(frames_dir)
            for i in range(3):
                with open(os.path.join(frames_dir, f"frame_{i:06d}.png"), "w") as f:
                    f.write("dummy")

            clips = scan_clips_dir(tmpdir, allow_standalone_videos=False)
            assert len(clips) == 1
            assert clips[0].state == ClipState.RAW
            assert clips[0].input_asset.frame_count == 3


# --- scan_project_clips ---

class TestScanProjectClips:
    def test_scans_v2_project(self):
        """scan_project_clips finds clips inside clips/ subdir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clips_dir = os.path.join(tmpdir, "clips")
            for name in ["clip_a", "clip_b"]:
                input_dir = os.path.join(clips_dir, name, "Input")
                os.makedirs(input_dir)
                with open(os.path.join(input_dir, "00000.png"), "w") as f:
                    f.write("dummy")

            clips = scan_project_clips(tmpdir)
            assert len(clips) == 2
            names = {c.name for c in clips}
            assert names == {"clip_a", "clip_b"}

    def test_v1_fallback(self):
        """scan_project_clips treats project dir as single clip for v1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "Input")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "00000.png"), "w") as f:
                f.write("dummy")

            clips = scan_project_clips(tmpdir)
            assert len(clips) == 1
            assert clips[0].root_path == tmpdir

    def test_v1_fallback_no_assets(self):
        """v1 project with no valid assets returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clips = scan_project_clips(tmpdir)
            assert clips == []

    def test_skips_hidden_in_clips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clips_dir = os.path.join(tmpdir, "clips")
            input_dir = os.path.join(clips_dir, "real_clip", "Input")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "00000.png"), "w") as f:
                f.write("dummy")
            os.makedirs(os.path.join(clips_dir, ".hidden"))
            os.makedirs(os.path.join(clips_dir, "_internal"))

            clips = scan_project_clips(tmpdir)
            assert len(clips) == 1

    def test_clip_json_display_name(self):
        """Clips read display_name from clip.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clips_dir = os.path.join(tmpdir, "clips")
            clip_dir = os.path.join(clips_dir, "Woman_Jumps")
            input_dir = os.path.join(clip_dir, "Input")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "00000.png"), "w") as f:
                f.write("dummy")

            # Write clip.json with custom display name
            with open(os.path.join(clip_dir, "clip.json"), "w") as f:
                json.dump({"display_name": "My Custom Clip"}, f)

            clips = scan_project_clips(tmpdir)
            assert len(clips) == 1
            assert clips[0].name == "My Custom Clip"

    def test_in_out_range_from_clip_json(self):
        """in_out_range is loaded from clip.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clips_dir = os.path.join(tmpdir, "clips")
            clip_dir = os.path.join(clips_dir, "test_clip")
            input_dir = os.path.join(clip_dir, "Input")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "00000.png"), "w") as f:
                f.write("dummy")

            with open(os.path.join(clip_dir, "clip.json"), "w") as f:
                json.dump({
                    "in_out_range": {"in_point": 5, "out_point": 20},
                }, f)

            clips = scan_project_clips(tmpdir)
            assert len(clips) == 1
            assert clips[0].in_out_range is not None
            assert clips[0].in_out_range.in_point == 5
            assert clips[0].in_out_range.out_point == 20

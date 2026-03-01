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

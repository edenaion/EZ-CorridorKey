"""Tests for FrameIndex — stem-based cross-mode navigation."""
import os
import tempfile
import pytest
from ui.preview.frame_index import FrameIndex, ViewMode, build_frame_index


class TestFrameIndex:
    def _make_clip_dir(self, tmp_path, stems_by_mode):
        """Create a temporary clip directory with files for each mode."""
        clip_root = os.path.join(tmp_path, "TestClip")
        os.makedirs(clip_root, exist_ok=True)

        for mode, stems in stems_by_mode.items():
            if mode == ViewMode.INPUT:
                dir_path = os.path.join(clip_root, "Input")
            elif mode == ViewMode.FG:
                dir_path = os.path.join(clip_root, "Output", "FG")
            elif mode == ViewMode.MATTE:
                dir_path = os.path.join(clip_root, "Output", "Matte")
            elif mode == ViewMode.COMP:
                dir_path = os.path.join(clip_root, "Output", "Comp")
            elif mode == ViewMode.PROCESSED:
                dir_path = os.path.join(clip_root, "Output", "Processed")
            else:
                continue

            os.makedirs(dir_path, exist_ok=True)
            ext = ".png" if mode == ViewMode.COMP else ".exr"
            for stem in stems:
                open(os.path.join(dir_path, f"{stem}{ext}"), 'w').close()

        return clip_root

    def test_basic_index(self, tmp_path):
        clip_root = self._make_clip_dir(tmp_path, {
            ViewMode.INPUT: ["frame_001", "frame_002", "frame_003"],
            ViewMode.COMP: ["frame_001", "frame_002", "frame_003"],
        })
        idx = build_frame_index(clip_root)
        assert idx.frame_count == 3
        assert idx.stems == ["frame_001", "frame_002", "frame_003"]

    def test_cross_mode_holes(self, tmp_path):
        """Codex test: stem-based sync when one mode has missing frames."""
        clip_root = self._make_clip_dir(tmp_path, {
            ViewMode.INPUT: ["frame_001", "frame_002", "frame_003"],
            ViewMode.COMP: ["frame_001", "frame_003"],  # frame_002 missing
            ViewMode.FG: ["frame_001", "frame_002"],     # frame_003 missing
        })
        idx = build_frame_index(clip_root)
        # All stems should be present in the timeline
        assert idx.frame_count == 3
        assert idx.stems == ["frame_001", "frame_002", "frame_003"]

        # Check availability
        assert idx.has_frame(ViewMode.INPUT, 0)  # frame_001
        assert idx.has_frame(ViewMode.INPUT, 1)  # frame_002
        assert idx.has_frame(ViewMode.INPUT, 2)  # frame_003

        assert idx.has_frame(ViewMode.COMP, 0)   # frame_001 ✓
        assert not idx.has_frame(ViewMode.COMP, 1) # frame_002 ✗
        assert idx.has_frame(ViewMode.COMP, 2)   # frame_003 ✓

        assert idx.has_frame(ViewMode.FG, 0)      # frame_001 ✓
        assert idx.has_frame(ViewMode.FG, 1)      # frame_002 ✓
        assert not idx.has_frame(ViewMode.FG, 2)  # frame_003 ✗

    def test_natural_sort_order(self, tmp_path):
        """Verify natural sort in stem timeline."""
        clip_root = self._make_clip_dir(tmp_path, {
            ViewMode.INPUT: ["frame_1", "frame_10", "frame_2"],
        })
        idx = build_frame_index(clip_root)
        assert idx.stems == ["frame_1", "frame_2", "frame_10"]

    def test_available_modes(self, tmp_path):
        clip_root = self._make_clip_dir(tmp_path, {
            ViewMode.INPUT: ["frame_001"],
            ViewMode.COMP: ["frame_001"],
        })
        idx = build_frame_index(clip_root)
        modes = idx.available_modes()
        assert ViewMode.INPUT in modes
        assert ViewMode.COMP in modes
        assert ViewMode.FG not in modes

    def test_get_path(self, tmp_path):
        clip_root = self._make_clip_dir(tmp_path, {
            ViewMode.INPUT: ["frame_001"],
        })
        idx = build_frame_index(clip_root)
        path = idx.get_path(ViewMode.INPUT, 0)
        assert path is not None
        assert "frame_001.exr" in path

    def test_empty_clip(self, tmp_path):
        clip_root = os.path.join(tmp_path, "EmptyClip")
        os.makedirs(clip_root, exist_ok=True)
        idx = build_frame_index(clip_root)
        assert idx.frame_count == 0
        assert idx.stems == []

    def test_out_of_range(self, tmp_path):
        clip_root = self._make_clip_dir(tmp_path, {
            ViewMode.INPUT: ["frame_001"],
        })
        idx = build_frame_index(clip_root)
        assert not idx.has_frame(ViewMode.INPUT, -1)
        assert not idx.has_frame(ViewMode.INPUT, 1)
        assert idx.get_path(ViewMode.INPUT, 5) is None

    def test_output_dir_override(self, tmp_path):
        """Regression for GitHub #86: when clip.output_dir is overridden via
        a global 'Default Output Directory' preference, inference writes
        FG/Matte/Comp/Processed to that external location. The preview
        viewport must read from the same location, not from
        ``clip_root/Output``.
        """
        clip_root = os.path.join(tmp_path, "Clip")
        os.makedirs(os.path.join(clip_root, "Input"), exist_ok=True)
        for stem in ("frame_001", "frame_002"):
            open(os.path.join(clip_root, "Input", f"{stem}.exr"), "w").close()

        # Simulate frames written to an external output dir (e.g. F:/Tests/...).
        external_out = os.path.join(tmp_path, "ExternalOutputs", "Proj", "Clip")
        for sub in ("FG", "Matte", "Comp", "Processed"):
            os.makedirs(os.path.join(external_out, sub), exist_ok=True)
            for stem in ("frame_001", "frame_002"):
                ext = ".png" if sub == "Comp" else ".exr"
                open(os.path.join(external_out, sub, f"{stem}{ext}"), "w").close()

        # Put a stale empty Output/ next to the clip_root to make sure we are
        # NOT reading from it.
        for sub in ("FG", "Matte", "Comp", "Processed"):
            os.makedirs(os.path.join(clip_root, "Output", sub), exist_ok=True)

        idx = build_frame_index(clip_root, output_dir=external_out)

        # Input still resolved against clip_root
        assert idx.has_frame(ViewMode.INPUT, 0)

        # Output modes resolved against the override, not clip_root/Output
        for mode in (ViewMode.FG, ViewMode.MATTE, ViewMode.COMP, ViewMode.PROCESSED):
            path = idx.get_path(mode, 0)
            assert path is not None, f"{mode} missing from override dir"
            assert external_out in path
            assert os.path.join(clip_root, "Output") not in path

    def test_output_dir_defaults_to_clip_root_output(self, tmp_path):
        """When no output_dir is passed, falls back to clip_root/Output
        (legacy behavior for callers that pre-date the override parameter).
        """
        clip_root = self._make_clip_dir(tmp_path, {
            ViewMode.INPUT: ["frame_001"],
            ViewMode.COMP: ["frame_001"],
        })
        idx = build_frame_index(clip_root)  # no output_dir
        assert idx.has_frame(ViewMode.COMP, 0)
        assert os.path.join(clip_root, "Output", "Comp") in idx.get_path(ViewMode.COMP, 0)

"""Tests for OutputConfig, InferenceParams serialization, and manifest-based resume."""
import os
import json
import pytest

from backend.service import InferenceParams, OutputConfig


class TestInferenceParams:
    def test_to_dict_roundtrip(self):
        params = InferenceParams(
            input_is_linear=True,
            despill_strength=0.8,
            auto_despeckle=False,
            despeckle_size=200,
            refiner_scale=2.5,
        )
        d = params.to_dict()
        restored = InferenceParams.from_dict(d)
        assert restored.input_is_linear is True
        assert restored.despill_strength == 0.8
        assert restored.auto_despeckle is False
        assert restored.despeckle_size == 200
        assert restored.refiner_scale == 2.5

    def test_from_dict_ignores_unknown_keys(self):
        d = {"input_is_linear": True, "future_field": 42}
        params = InferenceParams.from_dict(d)
        assert params.input_is_linear is True
        assert params.despill_strength == 0.5  # default

    def test_defaults(self):
        params = InferenceParams()
        assert params.input_is_linear is False
        assert params.despill_strength == 0.5
        assert params.refiner_scale == 1.0
        assert params.despeckle_dilation == 25
        assert params.despeckle_blur == 5

    def test_despeckle_advanced_roundtrip(self):
        params = InferenceParams(despeckle_dilation=10, despeckle_blur=15)
        d = params.to_dict()
        restored = InferenceParams.from_dict(d)
        assert restored.despeckle_dilation == 10
        assert restored.despeckle_blur == 15

    def test_old_session_missing_advanced_fields(self):
        """Old session data without new fields gets defaults."""
        d = {"input_is_linear": False, "despill_strength": 1.0}
        params = InferenceParams.from_dict(d)
        assert params.despeckle_dilation == 25
        assert params.despeckle_blur == 5


class TestOutputConfig:
    def test_to_dict_roundtrip(self):
        cfg = OutputConfig(
            fg_enabled=False,
            fg_format="png",
            matte_enabled=True,
            matte_format="exr",
            comp_enabled=True,
            comp_format="png",
            processed_enabled=False,
            processed_format="exr",
        )
        d = cfg.to_dict()
        restored = OutputConfig.from_dict(d)
        assert restored.fg_enabled is False
        assert restored.fg_format == "png"
        assert restored.processed_enabled is False

    def test_enabled_outputs(self):
        cfg = OutputConfig(fg_enabled=True, matte_enabled=True, comp_enabled=False, processed_enabled=False)
        assert cfg.enabled_outputs == ["fg", "matte"]

    def test_all_enabled(self):
        cfg = OutputConfig()
        assert cfg.enabled_outputs == ["fg", "matte", "comp", "processed"]

    def test_from_dict_ignores_unknown(self):
        d = {"fg_enabled": False, "new_feature": True}
        cfg = OutputConfig.from_dict(d)
        assert cfg.fg_enabled is False
        assert cfg.matte_enabled is True  # default


class TestManifestResume:
    def test_completed_stems_uses_manifest(self, tmp_path):
        """When manifest exists, resume checks only enabled outputs."""
        from backend.clip_state import ClipEntry, ClipState

        clip = ClipEntry(name="TestClip", root_path=str(tmp_path))
        out = os.path.join(str(tmp_path), "Output")
        fg_dir = os.path.join(out, "FG")
        comp_dir = os.path.join(out, "Comp")
        os.makedirs(fg_dir)
        os.makedirs(comp_dir)

        # Write manifest with only FG+Comp enabled
        manifest = {
            "version": 1,
            "enabled_outputs": ["fg", "comp"],
            "formats": {"fg": "exr", "comp": "png"},
            "params": {},
        }
        with open(os.path.join(out, ".corridorkey_manifest.json"), 'w') as f:
            json.dump(manifest, f)

        # Create matching stems
        open(os.path.join(fg_dir, "frame_001.exr"), 'w').close()
        open(os.path.join(comp_dir, "frame_001.png"), 'w').close()

        stems = clip.completed_stems()
        assert "frame_001" in stems

    def test_completed_stems_fallback_no_manifest(self, tmp_path):
        """Without manifest, falls back to FG+Matte intersection."""
        from backend.clip_state import ClipEntry

        clip = ClipEntry(name="TestClip", root_path=str(tmp_path))
        out = os.path.join(str(tmp_path), "Output")
        fg_dir = os.path.join(out, "FG")
        matte_dir = os.path.join(out, "Matte")
        os.makedirs(fg_dir)
        os.makedirs(matte_dir)

        open(os.path.join(fg_dir, "frame_001.exr"), 'w').close()
        open(os.path.join(matte_dir, "frame_001.exr"), 'w').close()
        open(os.path.join(fg_dir, "frame_002.exr"), 'w').close()
        # frame_002 missing from matte — not complete

        stems = clip.completed_stems()
        assert "frame_001" in stems
        assert "frame_002" not in stems

    def test_completed_count_matches_stems(self, tmp_path):
        from backend.clip_state import ClipEntry

        clip = ClipEntry(name="TestClip", root_path=str(tmp_path))
        out = os.path.join(str(tmp_path), "Output")
        fg_dir = os.path.join(out, "FG")
        matte_dir = os.path.join(out, "Matte")
        os.makedirs(fg_dir)
        os.makedirs(matte_dir)

        for stem in ["frame_001", "frame_002", "frame_003"]:
            open(os.path.join(fg_dir, f"{stem}.exr"), 'w').close()
            open(os.path.join(matte_dir, f"{stem}.exr"), 'w').close()

        assert clip.completed_frame_count() == 3
        assert len(clip.completed_stems()) == 3

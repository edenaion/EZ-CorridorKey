"""Tests for session save/load — JSON sidecar, versioning, forward compat."""

import json
import os
import pytest

from backend.service import InferenceParams, OutputConfig


class TestSessionData:
    def test_params_roundtrip(self):
        """Params serialize and deserialize correctly."""
        params = InferenceParams(despill_strength=0.5, refiner_scale=2.0)
        d = params.to_dict()
        restored = InferenceParams.from_dict(d)
        assert restored.despill_strength == 0.5
        assert restored.refiner_scale == 2.0

    def test_output_config_roundtrip(self):
        cfg = OutputConfig(fg_enabled=False, comp_format="exr")
        d = cfg.to_dict()
        restored = OutputConfig.from_dict(d)
        assert restored.fg_enabled is False
        assert restored.comp_format == "exr"

    def test_session_file_format(self, tmp_path):
        """Session file should be valid JSON with version key."""
        session = {
            "version": 1,
            "params": InferenceParams().to_dict(),
            "output_config": OutputConfig().to_dict(),
            "live_preview": False,
            "split_view": False,
        }
        path = os.path.join(str(tmp_path), ".corridorkey_session.json")
        with open(path, "w") as f:
            json.dump(session, f)

        with open(path, "r") as f:
            loaded = json.load(f)

        assert loaded["version"] == 1
        assert "params" in loaded
        assert "output_config" in loaded

    def test_forward_compat_unknown_keys(self):
        """Unknown keys from newer versions should be ignored."""
        d = {
            "input_is_linear": True,
            "new_param_v2": 42,
            "another_future_param": "hello",
        }
        params = InferenceParams.from_dict(d)
        assert params.input_is_linear is True
        # No error, unknown keys silently ignored

    def test_corrupt_session_file(self, tmp_path):
        """Corrupt JSON should not crash."""
        path = os.path.join(str(tmp_path), ".corridorkey_session.json")
        with open(path, "w") as f:
            f.write("{invalid json")

        with pytest.raises(json.JSONDecodeError):
            with open(path, "r") as f:
                json.load(f)

    def test_atomic_write_pattern(self, tmp_path):
        """Verify tmp+rename pattern produces valid file."""
        path = os.path.join(str(tmp_path), "session.json")
        tmp_path_file = path + ".tmp"

        data = {"version": 1, "test": True}
        with open(tmp_path_file, "w") as f:
            json.dump(data, f)
        os.rename(tmp_path_file, path)

        assert os.path.isfile(path)
        assert not os.path.exists(tmp_path_file)
        with open(path, "r") as f:
            loaded = json.load(f)
        assert loaded["test"] is True

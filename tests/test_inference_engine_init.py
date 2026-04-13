"""Constructor-level guards for CorridorKeyEngine startup decisions."""

import pytest

timm = pytest.importorskip("timm", reason="timm not installed")

from CorridorKeyModule.inference_engine import CorridorKeyEngine


class TestCorridorKeyEngineInit:
    def test_explicit_speed_skips_vram_probe(self, monkeypatch):
        """Explicit speed mode must not touch VRAM probe."""
        monkeypatch.delenv("CORRIDORKEY_OPT_MODE", raising=False)

        def _boom() -> float:
            raise AssertionError("VRAM probe should not run for explicit mode")

        monkeypatch.setattr(CorridorKeyEngine, "_get_vram_gb", staticmethod(_boom))
        monkeypatch.setattr(CorridorKeyEngine, "_load_model", lambda self: object())

        engine = CorridorKeyEngine(
            checkpoint_path="dummy.pth",
            device="cpu",
            optimization_mode="speed",
        )

        assert engine.tile_size == 0

    def test_explicit_lowvram_skips_vram_probe(self, monkeypatch):
        """Explicit lowvram mode must not touch VRAM probe."""
        monkeypatch.delenv("CORRIDORKEY_OPT_MODE", raising=False)

        def _boom() -> float:
            raise AssertionError("VRAM probe should not run for explicit mode")

        monkeypatch.setattr(CorridorKeyEngine, "_get_vram_gb", staticmethod(_boom))
        monkeypatch.setattr(CorridorKeyEngine, "_load_model", lambda self: object())

        engine = CorridorKeyEngine(
            checkpoint_path="dummy.pth",
            device="cpu",
            optimization_mode="lowvram",
        )

        assert engine.tile_size == 512

    def test_auto_mode_probes_vram_and_selects_speed(self, monkeypatch):
        """Auto mode with >=12GB VRAM should select speed (no tiling)."""
        monkeypatch.delenv("CORRIDORKEY_OPT_MODE", raising=False)
        monkeypatch.setattr(CorridorKeyEngine, "_get_vram_gb", staticmethod(lambda: 24.0))
        monkeypatch.setattr(CorridorKeyEngine, "_load_model", lambda self: object())

        engine = CorridorKeyEngine(
            checkpoint_path="dummy.pth",
            device="cpu",
            optimization_mode="auto",
        )

        assert engine.tile_size == 0

    def test_auto_mode_probes_vram_and_selects_lowvram(self, monkeypatch):
        """Auto mode with <12GB VRAM should select tiled mode."""
        monkeypatch.delenv("CORRIDORKEY_OPT_MODE", raising=False)
        monkeypatch.setattr(CorridorKeyEngine, "_get_vram_gb", staticmethod(lambda: 8.0))
        monkeypatch.setattr(CorridorKeyEngine, "_load_model", lambda self: object())

        engine = CorridorKeyEngine(
            checkpoint_path="dummy.pth",
            device="cpu",
            optimization_mode="auto",
        )

        assert engine.tile_size == 512

    def test_auto_mode_uses_pynvml_not_torch_cuda(self, monkeypatch):
        """Auto mode VRAM probe should use pynvml (driver-level), not torch.cuda
        which can stall after GVM teardown."""
        monkeypatch.delenv("CORRIDORKEY_OPT_MODE", raising=False)
        monkeypatch.setattr(CorridorKeyEngine, "_load_model", lambda self: object())

        probe_called_with = []

        @staticmethod
        def _tracking_probe() -> float:
            probe_called_with.append(True)
            return 32.0

        monkeypatch.setattr(CorridorKeyEngine, "_get_vram_gb", _tracking_probe)

        engine = CorridorKeyEngine(
            checkpoint_path="dummy.pth",
            device="cpu",
            optimization_mode="auto",
        )

        assert len(probe_called_with) == 1, "auto mode must call _get_vram_gb exactly once"
        assert engine.tile_size == 0  # 32GB -> speed mode

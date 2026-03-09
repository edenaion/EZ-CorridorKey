"""Constructor-level guards for CorridorKeyEngine startup decisions."""

import pytest

timm = pytest.importorskip("timm", reason="timm not installed")

from CorridorKeyModule.inference_engine import CorridorKeyEngine


class TestCorridorKeyEngineInit:
    def test_explicit_mode_skips_vram_probe(self, monkeypatch):
        """Explicit lowvram mode must not touch VRAM probe calls."""
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

    def test_auto_mode_uses_deterministic_startup_without_probe(self, monkeypatch):
        monkeypatch.delenv("CORRIDORKEY_OPT_MODE", raising=False)

        def _boom() -> float:
            raise AssertionError("VRAM probe should not run in auto startup path")

        monkeypatch.setattr(CorridorKeyEngine, "_get_vram_gb", staticmethod(_boom))
        monkeypatch.setattr(CorridorKeyEngine, "_load_model", lambda self: object())

        engine = CorridorKeyEngine(
            checkpoint_path="dummy.pth",
            device="cpu",
            optimization_mode="auto",
        )

        assert engine.tile_size == 512

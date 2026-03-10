"""Regression tests for lazy torch.compile/Triton fallback behavior."""

from types import SimpleNamespace

import pytest
import torch

timm = pytest.importorskip("timm", reason="timm not installed")

from CorridorKeyModule.inference_engine import CorridorKeyEngine


class _FakeModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0
        self.refiner = None

    def __call__(self, inp_t, refiner_scale=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class _TileCompileModel:
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.refiner = SimpleNamespace(_compiled_process_tile=object())

    def __call__(self, inp_t, refiner_scale=None):
        self.calls += 1
        if self.refiner._compiled_process_tile is not None:
            raise FileNotFoundError(
                r"C:\Users\Johan\AppData\Local\Temp\torchinductor_Johan\triton\0\tmp.x\kernel.source"
            )
        return self.result


def _make_engine(model):
    engine = CorridorKeyEngine.__new__(CorridorKeyEngine)
    engine.device = torch.device("cpu")
    engine.model = model
    engine._eager_model = model
    engine._compiled_model = None
    engine._compile_error = None
    engine._use_compile = True
    engine._on_status = None
    return engine


class TestCompileFallback:
    def test_retries_eager_on_compiled_model_runtime_failure(self):
        eager_result = {"alpha": torch.tensor(1.0), "fg": torch.tensor(2.0)}
        eager_model = _FakeModel(result=eager_result)
        eager_model.refiner = SimpleNamespace(_compiled_process_tile=object())

        compiled_model = _FakeModel(
            error=FileNotFoundError(
                r"C:\Users\Johan\AppData\Local\Temp\torchinductor_Johan\triton\0\tmp.x\kernel.source"
            )
        )

        engine = _make_engine(eager_model)
        engine.model = compiled_model
        engine._compiled_model = compiled_model

        out = engine._forward_model(torch.tensor(1.0), torch.tensor(1.0))

        assert out is eager_result
        assert compiled_model.calls == 1
        assert eager_model.calls == 1
        assert engine.model is eager_model
        assert engine._compiled_model is None
        assert engine._use_compile is False
        assert eager_model.refiner._compiled_process_tile is None
        assert "FileNotFoundError" in engine._compile_error

    def test_tile_kernel_failure_retries_same_eager_model_after_clearing_tile_wrapper(self):
        result = {"alpha": torch.tensor(1.0), "fg": torch.tensor(2.0)}
        model = _TileCompileModel(result=result)
        engine = _make_engine(model)

        out = engine._forward_model(torch.tensor(1.0), torch.tensor(1.0))

        assert out is result
        assert model.calls == 2
        assert engine.model is model
        assert engine._use_compile is False
        assert model.refiner._compiled_process_tile is None

    def test_non_compile_error_still_propagates(self):
        eager_model = _FakeModel(result={"alpha": torch.tensor(1.0), "fg": torch.tensor(2.0)})
        compiled_model = _FakeModel(error=RuntimeError("CUDA out of memory"))

        engine = _make_engine(eager_model)
        engine.model = compiled_model
        engine._compiled_model = compiled_model

        with pytest.raises(RuntimeError, match="out of memory"):
            engine._forward_model(torch.tensor(1.0), torch.tensor(1.0))

        assert compiled_model.calls == 1
        assert eager_model.calls == 0
        assert engine.model is compiled_model
        assert engine._use_compile is True

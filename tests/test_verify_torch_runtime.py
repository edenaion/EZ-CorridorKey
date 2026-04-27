import sys
import types

import pytest

from scripts import verify_torch_runtime
from scripts.verify_torch_runtime import RuntimeInfo, evaluate_runtime


def _info(**overrides):
    base = RuntimeInfo(
        platform="Windows-11-10.0.22631-SP0",
        python_version="3.11.11",
        torch_version="2.9.1+cu128",
        torchvision_version="0.24.1+cu128",
        torch_cuda_version="12.8",
        cuda_available=True,
        cuda_device_count=1,
        mps_available=False,
        mps_built=False,
        nvidia_smi_path=r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        nvidia_smi_summary="| NVIDIA-SMI 581.57 Driver Version: 581.57 CUDA Version: 13.0 |",
    )
    return RuntimeInfo(**{**base.__dict__, **overrides})


def test_gpu_runtime_passes_when_cuda_is_available():
    ok, message = evaluate_runtime(_info())
    assert ok is True
    assert "CUDA verified" in message


def test_gpu_runtime_fails_when_torch_has_no_cuda_support():
    ok, message = evaluate_runtime(_info(torch_cuda_version="", cuda_available=False))
    assert ok is False
    assert "without CUDA support" in message


def test_gpu_runtime_fails_when_cuda_build_cannot_initialize():
    ok, message = evaluate_runtime(_info(cuda_available=False))
    assert ok is False
    assert "torch.cuda.is_available() is false" in message


def test_macos_runtime_passes_without_nvidia():
    ok, message = evaluate_runtime(
        _info(
            platform="macOS-15.0-arm64-arm-64bit",
            torch_cuda_version="",
            cuda_available=False,
            cuda_device_count=0,
            mps_available=True,
            mps_built=True,
            nvidia_smi_path="",
            nvidia_smi_summary="",
        )
    )
    assert ok is True
    assert "macOS runtime verified" in message


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="evaluate_runtime takes the macOS branch on Darwin hosts",
)
def test_cpu_runtime_passes_when_no_gpu_is_expected():
    ok, message = evaluate_runtime(
        _info(
            torch_cuda_version="",
            cuda_available=False,
            cuda_device_count=0,
            nvidia_smi_path="",
            nvidia_smi_summary="",
        )
    )
    assert ok is True
    assert "CPU runtime verified" in message


def test_collect_runtime_info_skips_torch_cuda_on_macos(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.9.1"
    fake_torch.version = types.SimpleNamespace(cuda="")
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: True, is_built=lambda: True)
    )

    def _module_getattr(name):
        if name == "cuda":
            raise AssertionError("torch.cuda should not be touched on macOS verification")
        raise AttributeError(name)

    fake_torch.__getattr__ = _module_getattr

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delitem(sys.modules, "torchvision", raising=False)
    monkeypatch.setattr(verify_torch_runtime.platform, "platform", lambda: "macOS-15.0-arm64-arm-64bit")
    monkeypatch.setattr(verify_torch_runtime, "find_nvidia_smi", lambda: "")

    info = verify_torch_runtime.collect_runtime_info()

    assert info.platform.startswith("macOS")
    assert info.cuda_available is False
    assert info.cuda_device_count == 0
    assert info.mps_available is True

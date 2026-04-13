"""Verify the installed torch runtime matches the expected accelerator path.

This runs inside the freshly created virtual environment at install time so the
installer can fail fast with actionable diagnostics instead of reporting a
"successful" install that only works on CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path


warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\..*",
    category=FutureWarning,
)


@dataclass
class RuntimeInfo:
    platform: str
    python_version: str
    torch_version: str
    torchvision_version: str
    torch_cuda_version: str
    cuda_available: bool
    cuda_device_count: int
    mps_available: bool
    mps_built: bool
    nvidia_smi_path: str
    nvidia_smi_summary: str


def _safe_getattr(obj: object, name: str, default: object) -> object:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def find_nvidia_smi() -> str:
    which = shutil.which("nvidia-smi")
    return which or ""


def read_nvidia_smi_summary(path: str) -> str:
    if not path:
        return ""
    try:
        result = subprocess.run(
            [path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive against platform oddities
        return f"failed to run nvidia-smi: {exc}"

    text = (result.stdout or "").strip()
    if not text:
        return ""
    return text.splitlines()[0][:400]


def should_probe_cuda(platform_name: str, nvidia_smi_path: str, torch_cuda_version: str) -> bool:
    if platform_name.lower().startswith("macos") or sys.platform == "darwin":
        return False
    return bool(nvidia_smi_path or torch_cuda_version)


def collect_runtime_info() -> RuntimeInfo:
    import torch  # Imported lazily so installer can surface import failures cleanly.

    torchvision_version = ""
    try:
        import torchvision  # type: ignore

        torchvision_version = getattr(torchvision, "__version__", "") or ""
    except Exception:
        torchvision_version = ""

    backends = _safe_getattr(torch, "backends", None)
    mps = _safe_getattr(backends, "mps", None) if backends is not None else None

    platform_name = platform.platform()
    nvidia_smi_path = find_nvidia_smi()
    torch_cuda_version = getattr(_safe_getattr(torch, "version", None), "cuda", "") or ""
    cuda = None
    cuda_available = False
    cuda_device_count = 0
    if should_probe_cuda(platform_name, nvidia_smi_path, torch_cuda_version):
        cuda = _safe_getattr(torch, "cuda", None)
        cuda_available = bool(cuda and cuda.is_available())
        cuda_device_count = int(cuda.device_count()) if cuda and cuda_available else 0

    return RuntimeInfo(
        platform=platform_name,
        python_version=sys.version.split()[0],
        torch_version=getattr(torch, "__version__", "") or "",
        torchvision_version=torchvision_version,
        torch_cuda_version=torch_cuda_version,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        mps_available=bool(mps and mps.is_available()),
        mps_built=bool(mps and mps.is_built()),
        nvidia_smi_path=nvidia_smi_path,
        nvidia_smi_summary=read_nvidia_smi_summary(nvidia_smi_path),
    )


def evaluate_runtime(info: RuntimeInfo, expect_gpu: str = "auto") -> tuple[bool, str]:
    if expect_gpu not in {"auto", "0", "1"}:
        return False, f"Invalid expect_gpu value: {expect_gpu}"

    gpu_expected = expect_gpu == "1" or (expect_gpu == "auto" and bool(info.nvidia_smi_path))
    if gpu_expected:
        if not info.torch_cuda_version:
            return (
                False,
                "NVIDIA GPU detected via nvidia-smi, but torch was installed without CUDA support. "
                "The driver may be too old for CorridorKey's current torch build, or the install fell back to CPU-only wheels.",
            )
        if not info.cuda_available:
            return (
                False,
                "Torch reports a CUDA build, but torch.cuda.is_available() is false. "
                "Check that the NVIDIA driver is working and that the installed torch build matches the system.",
            )
        return (
            True,
            f"CUDA verified: torch {info.torch_version} with CUDA {info.torch_cuda_version}.",
        )

    if info.platform.lower().startswith("macos") or sys.platform == "darwin":
        if info.mps_available:
            return True, f"macOS runtime verified: torch {info.torch_version} with MPS available."
        return True, f"macOS runtime verified: torch {info.torch_version} (CPU/MPS fallback)."

    return True, f"CPU runtime verified: torch {info.torch_version}."


def write_log(path: str | None, payload: dict[str, object]) -> None:
    if not path:
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the installed torch runtime for CorridorKey"
    )
    parser.add_argument("--log", default="", help="Optional JSON log path")
    parser.add_argument(
        "--expect-gpu",
        default=os.environ.get("CORRIDORKEY_EXPECT_GPU", "auto"),
        choices=("auto", "0", "1"),
        help="Whether GPU support is expected. 'auto' requires CUDA only when nvidia-smi is present.",
    )
    args = parser.parse_args()

    payload: dict[str, object] = {"ok": False}
    try:
        info = collect_runtime_info()
        ok, message = evaluate_runtime(info, expect_gpu=args.expect_gpu)
        payload.update(asdict(info))
        payload["ok"] = ok
        payload["message"] = message
    except Exception as exc:
        payload["message"] = f"Failed to import/inspect torch: {exc}"
        write_log(args.log, payload)
        print(payload["message"])
        return 1

    write_log(args.log, payload)
    print(payload["message"])
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

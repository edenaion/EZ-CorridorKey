"""Resolve the appropriate PyTorch wheel index for Windows installers.

This helper is intentionally stdlib-only so it can run before the project
environment exists. It prefers real `nvidia-smi` output, but supports mocked
output via environment variables for batch smoke tests.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

CU128_URL = "https://download.pytorch.org/whl/cu128"
CU126_URL = "https://download.pytorch.org/whl/cu126"
CPU_URL = "https://download.pytorch.org/whl/cpu"

# NOTE: cu130 wheels were dropped from the auto-installer matrix in
# 1.9.2 because they ship without sm_60/61/70 kernels — meaning Pascal
# (GTX 10-series) GPUs crash with "no kernel image is available" at
# inference time. cu128 wheels run on every NVIDIA driver from CUDA
# 12.6 up through CUDA 13.x and cover Pascal through Blackwell, so
# they are now the canonical wheel for any modern NVIDIA driver.

_VERSION_RE = re.compile(r"(\d+\.\d+)")


def _standard_nvidia_smi_paths() -> list[Path]:
    roots = []
    for env_name in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))
    if not roots:
        roots.append(Path(r"C:\Program Files"))
    seen: set[str] = set()
    candidates: list[Path] = []
    for root in roots:
        candidate = root / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def find_nvidia_smi() -> str | None:
    mock_file = os.environ.get("CORRIDORKEY_MOCK_NVIDIA_SMI_FILE")
    if mock_file:
        return mock_file

    explicit = os.environ.get("CORRIDORKEY_NVIDIA_SMI_PATH")
    if explicit and Path(explicit).is_file():
        return explicit

    which = shutil.which("nvidia-smi")
    if which:
        return which

    for candidate in _standard_nvidia_smi_paths():
        if candidate.is_file():
            return str(candidate)
    return None


def _read_mock_output() -> str | None:
    mock_file = os.environ.get("CORRIDORKEY_MOCK_NVIDIA_SMI_FILE")
    if not mock_file:
        return None
    return Path(mock_file).read_text(encoding="utf-8", errors="replace")


def _run_command(args: list[str]) -> str:
    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout or ""


def read_nvidia_smi_output(nvidia_smi_path: str | None) -> str:
    mock = _read_mock_output()
    if mock is not None:
        return mock
    if not nvidia_smi_path:
        return ""
    return _run_command([nvidia_smi_path])


def parse_driver_version(text: str) -> str | None:
    match = re.search(r"Driver\s*Version[^0-9]*([0-9]+(?:\.[0-9]+)+)", text, re.IGNORECASE)
    if match:
        return match.group(1)

    for line in text.splitlines():
        if "NVIDIA-SMI" not in line.upper():
            continue
        versions = _VERSION_RE.findall(line)
        if len(versions) >= 2:
            return versions[1]
        if versions:
            return versions[0]
    return None


def parse_cuda_line(text: str) -> str | None:
    for line in text.splitlines():
        upper = line.upper()
        if "CUDA" in upper and "NVIDIA-SMI" in upper:
            return line.strip()
        if "CUDA" in upper and _VERSION_RE.search(line):
            return line.strip()
    return None


def parse_cuda_version(text: str) -> str | None:
    line = parse_cuda_line(text)
    if not line:
        return None
    versions = _VERSION_RE.findall(line)
    if not versions:
        return None
    return versions[-1]


def choose_index_url(cuda_version: str | None) -> tuple[str, str, str]:
    """Pick the PyTorch wheel index for a given driver-reported CUDA version.

    cu128 is the canonical wheel: it ships kernels for sm_60 (Pascal)
    through sm_120 (Blackwell) and runs on any NVIDIA driver from CUDA
    12.6 through CUDA 13.x. cu130 was deliberately dropped here in
    1.9.2 because it omits Pascal kernels and breaks GTX 10-series
    users. cu126 stays as a fallback for older drivers (CUDA 12.0-12.5).
    """
    if not cuda_version:
        return CPU_URL, "CPU-only PyTorch", "Could not determine CUDA version from nvidia-smi; installing CPU-only PyTorch."

    try:
        major_s, minor_s = cuda_version.split(".", 1)
        major = int(major_s)
        minor = int(re.match(r"\d+", minor_s).group(0)) if re.match(r"\d+", minor_s) else 0
    except Exception:
        return CPU_URL, "CPU-only PyTorch", f"Could not parse CUDA version '{cuda_version}'; installing CPU-only PyTorch."

    # CUDA 13.x drivers: cu128 wheels are forward-compatible and
    # preserve Pascal support, so they win over cu130.
    if major >= 13:
        return CU128_URL, "PyTorch CUDA 12.8 wheels (covers Pascal-Blackwell)", ""
    # CUDA 12.6+ drivers: same wheel.
    if major == 12 and minor >= 6:
        return CU128_URL, "PyTorch CUDA 12.8 wheels", ""
    # CUDA 12.0-12.5 drivers: fall back to the older cu126 wheel.
    if major == 12:
        return CU126_URL, "PyTorch CUDA 12.6 wheels", ""
    return (
        CPU_URL,
        "CPU-only PyTorch",
        f"Detected CUDA {cuda_version}, which is below the supported PyTorch 2.9.1 GPU wheel range for the auto-installer; installing CPU-only PyTorch.",
    )


def detect() -> dict[str, str]:
    nvidia_smi_path = find_nvidia_smi()
    if not nvidia_smi_path:
        return {
            "CUDA_DETECT_MODE": "cpu",
            "CUDA_DETECT_REASON": "nvidia_smi_not_found",
            "INDEX_URL": CPU_URL,
            "CUDA_WHEEL_LABEL": "CPU-only PyTorch",
            "CUDA_NOTE": "No NVIDIA GPU detected, installing CPU-only PyTorch.",
            "DRIVER": "",
            "CUDA_VERSION": "",
            "CUDA_LINE": "",
            "NVIDIA_SMI_PATH": "",
        }

    output = read_nvidia_smi_output(nvidia_smi_path)
    driver = parse_driver_version(output) or os.environ.get("CORRIDORKEY_MOCK_DRIVER_VERSION", "")
    cuda_line = parse_cuda_line(output) or ""
    cuda_version = parse_cuda_version(output) or ""
    index_url, wheel_label, note = choose_index_url(cuda_version)
    mode = "nvidia" if index_url != CPU_URL else "cpu"
    reason = "ok" if mode == "nvidia" else "cuda_unsupported_or_unknown"

    if mode == "cpu" and not note:
        note = "Falling back to CPU-only PyTorch."

    return {
        "CUDA_DETECT_MODE": mode,
        "CUDA_DETECT_REASON": reason,
        "INDEX_URL": index_url,
        "CUDA_WHEEL_LABEL": wheel_label,
        "CUDA_NOTE": note,
        "DRIVER": driver,
        "CUDA_VERSION": cuda_version,
        "CUDA_LINE": cuda_line,
        "NVIDIA_SMI_PATH": nvidia_smi_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect Windows PyTorch wheel index from nvidia-smi output")
    parser.add_argument("--format", choices=("env",), default="env")
    args = parser.parse_args()

    data = detect()
    if args.format == "env":
        for key, value in data.items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

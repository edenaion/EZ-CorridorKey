#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def _cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def main() -> int:
    root = Path("logs/azure-gpu-installer-smoke")
    rows: list[dict[str, object]] = []
    if root.exists():
        for path in sorted(root.glob("*/summary.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))

    headers = [
        "Lane",
        "Status",
        "OS",
        "Python",
        "Region",
        "VM Size",
        "Est. Max USD",
        "Torch",
        "Torch CUDA",
        "CUDA OK",
        "Device",
        "Failure",
    ]

    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")

    for row in rows:
        result = row.get("result") or {}
        os_name = "windows" if row.get("windows_region") else "linux"
        python_version = row.get("windows_python") if os_name == "windows" else row.get("linux_python")
        region = row.get("windows_region") if os_name == "windows" else row.get("linux_region")
        size = row.get("windows_size") if os_name == "windows" else row.get("linux_size")
        est = row.get("windows_estimated_max_usd") if os_name == "windows" else row.get("linux_estimated_max_usd")
        values = [
            row.get("lane_name"),
            row.get("status"),
            os_name,
            python_version,
            region,
            size,
            est,
            result.get("torch_version"),
            result.get("torch_cuda"),
            result.get("cuda_available"),
            result.get("service_detect_device"),
            row.get("failure_context"),
        ]
        print("| " + " | ".join(_cell(value) for value in values) + " |")

    if not rows:
        print("| no-data | no-data | | | | | | | | | | |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

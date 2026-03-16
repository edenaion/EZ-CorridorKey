import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed")

from ui.workers.gpu_monitor import GPUMonitor


def test_apple_silicon_memory_report_uses_same_basis_for_text_and_meter(monkeypatch):
    monitor = GPUMonitor.__new__(GPUMonitor)
    monitor._apple_chip_name = "Apple M1 Max"

    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(
            total=16 * 1024**3,
            available=2 * 1024**3,
            used=6 * 1024**3,  # intentionally inconsistent with available
            percent=86.0,
        ),
    )

    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    info = GPUMonitor._query_apple_silicon(monitor)

    assert info["available"] is True
    assert info["name"] == "Apple M1 Max"
    assert info["total_gb"] == 16.0
    assert info["used_gb"] == 14.0
    assert info["free_gb"] == 2.0
    assert info["usage_pct"] == 87.5

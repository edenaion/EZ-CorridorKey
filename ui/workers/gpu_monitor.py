"""GPU monitor — polls VRAM usage and GPU info via QTimer.

Design decisions (from Codex review):
- Uses pynvml (NVML) for VRAM display, not torch.cuda calls
- Falls back to torch.cuda only if NVML unavailable
- Polling interval: 2000ms (not too aggressive on GPU queries)
- Emits lightweight dict signal, QPixmap/rendering stays on main thread
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QTimer, Signal

logger = logging.getLogger(__name__)


class GPUMonitor(QObject):
    """Polls GPU VRAM usage periodically and emits updates.

    Signal data dict keys:
        name: str — GPU name (e.g. "NVIDIA GeForce RTX 4090")
        total_gb: float — Total VRAM in GB
        used_gb: float — Used VRAM in GB
        free_gb: float — Free VRAM in GB
        usage_pct: float — Usage percentage (0-100)
        available: bool — Whether GPU monitoring is available
    """

    vram_updated = Signal(dict)  # GPU info dict
    gpu_name = Signal(str)       # emitted once on first successful poll

    def __init__(self, interval_ms: int = 2000, parent=None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._poll)
        self._nvml_available = False
        self._nvml_handle = None
        self._torch_fallback = False
        self._name_emitted = False
        self._init_nvml()

    def _init_nvml(self) -> None:
        """Try to initialize NVML for GPU monitoring."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_available = True
            logger.info("GPU monitor: using NVML")
        except Exception:
            self._nvml_available = False
            # Check torch fallback
            try:
                import torch
                if torch.cuda.is_available():
                    self._torch_fallback = True
                    logger.info("GPU monitor: NVML unavailable, using torch.cuda fallback")
                else:
                    logger.info("GPU monitor: no GPU available")
            except ImportError:
                logger.info("GPU monitor: no GPU monitoring available")

    def start(self) -> None:
        """Start polling."""
        self._poll()  # immediate first poll
        self._timer.start()

    def stop(self) -> None:
        """Stop polling."""
        self._timer.stop()

    def _poll(self) -> None:
        """Query GPU status and emit signal."""
        info = self._query_gpu()
        self.vram_updated.emit(info)

        if info.get("available") and not self._name_emitted:
            self.gpu_name.emit(info.get("name", "GPU"))
            self._name_emitted = True

    def _query_gpu(self) -> dict:
        """Get GPU VRAM info from NVML or torch fallback."""
        if self._nvml_available:
            return self._query_nvml()
        elif self._torch_fallback:
            return self._query_torch()
        return {"available": False}

    def _query_nvml(self) -> dict:
        """Query via NVML (preferred — no CUDA context interference)."""
        try:
            import pynvml
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            name = pynvml.nvmlDeviceGetName(self._nvml_handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            total_gb = mem.total / (1024 ** 3)
            used_gb = mem.used / (1024 ** 3)
            free_gb = mem.free / (1024 ** 3)
            return {
                "available": True,
                "name": name,
                "total_gb": round(total_gb, 1),
                "used_gb": round(used_gb, 1),
                "free_gb": round(free_gb, 1),
                "usage_pct": round(used_gb / total_gb * 100, 1) if total_gb > 0 else 0,
            }
        except Exception as e:
            logger.debug(f"NVML query failed: {e}")
            return {"available": False}

    def _query_torch(self) -> dict:
        """Fallback: query via torch.cuda (may contend with inference)."""
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory
            reserved = torch.cuda.memory_reserved(0)
            free = total - reserved
            total_gb = total / (1024 ** 3)
            used_gb = reserved / (1024 ** 3)
            free_gb = free / (1024 ** 3)
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "total_gb": round(total_gb, 1),
                "used_gb": round(used_gb, 1),
                "free_gb": round(free_gb, 1),
                "usage_pct": round(used_gb / total_gb * 100, 1) if total_gb > 0 else 0,
            }
        except Exception as e:
            logger.debug(f"Torch GPU query failed: {e}")
            return {"available": False}

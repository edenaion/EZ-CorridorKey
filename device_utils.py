"""ROCm / AMD GPU environment helpers (optional; safe on NVIDIA CUDA and CPU).

`setup_rocm_env()` is invoked from `main._try_setup_rocm_env()` at process start.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def _ensure_stderr_info_logging() -> None:
    """Emit HIP INFO lines to stderr before main.setup_logging() configures the root logger."""
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def setup_rocm_env() -> None:
    """Configure ROCm-related env when PyTorch is built with HIP.

    If ``torch.version.hip`` is missing, returns without side effects (CUDA or CPU builds).
    Uses :func:`os.environ.setdefault` for ``HSA_OVERRIDE_GFX_VERSION`` so runtime overrides win.
    """
    import torch

    hip = getattr(torch.version, "hip", None)
    if not hip:
        logger.debug("ROCm: HIP not present in this PyTorch build (torch.version.hip is None)")
        return

    _ensure_stderr_info_logging()

    logger.info("ROCm: HIP-enabled PyTorch (HIP version %s)", hip)
    try:
        if torch.cuda.is_available():
            logger.info("ROCm: using device %s", torch.cuda.get_device_name(0))
        else:
            logger.warning(
                "ROCm: HIP build but CUDA API reports no device (check drivers and ROCm stack)"
            )
    except Exception as exc:
        logger.debug("ROCm: device query skipped: %s", exc)

    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

"""Shared pytest configuration and fixtures for CorridorKey tests."""

import numpy as np
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: requires CUDA GPU (skipped in CI)")
    config.addinivalue_line("markers", "slow: long-running test")


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests when CUDA is not available."""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if not has_cuda:
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture
def sample_frame_rgb():
    """Small 64x64 RGB frame as float32 in [0, 1] (sRGB)."""
    rng = np.random.default_rng(42)
    return rng.random((64, 64, 3), dtype=np.float32)


@pytest.fixture
def sample_mask():
    """Matching 64x64 single-channel alpha mask as float32 in [0, 1]."""
    rng = np.random.default_rng(42)
    mask = rng.random((64, 64), dtype=np.float32)
    # Make it more mask-like: threshold to create distinct FG/BG regions
    return (mask > 0.5).astype(np.float32)

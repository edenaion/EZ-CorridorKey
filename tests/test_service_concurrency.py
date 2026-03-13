"""Behavioral tests for GPU lock serialization (Codex: not structural).

Uses threading.Barrier + time recording to prove process_frame calls
don't overlap, without asserting Lock.acquire directly.
"""
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.service import CorridorKeyService, InferenceParams, OutputConfig, _ActiveModel
from backend.clip_state import ClipAsset, ClipEntry, ClipState


def _make_slow_engine(delay: float = 0.1):
    """Create a mock engine whose process_frame takes `delay` seconds."""
    engine = MagicMock()
    call_log = []

    def slow_process(*args, **kwargs):
        start = time.monotonic()
        call_log.append(("start", start))
        time.sleep(delay)
        end = time.monotonic()
        call_log.append(("end", end))
        return {
            "fg": np.ones((4, 4, 3), dtype=np.float32),
            "alpha": np.ones((4, 4, 1), dtype=np.float32),
            "comp": np.ones((4, 4, 3), dtype=np.float32),
            "processed": np.ones((4, 4, 4), dtype=np.float32),
        }

    engine.process_frame.side_effect = slow_process
    return engine, call_log


class TestGPULockSerialization:
    """Two threads running reprocess_single_frame should not overlap."""

    def test_concurrent_reprocess_serialized(self, sample_clip):
        svc = CorridorKeyService()
        engine, call_log = _make_slow_engine(0.05)
        svc._engine_pool = [engine]
        svc._active_model = _ActiveModel.INFERENCE
        params = InferenceParams()

        results = [None, None]
        errors = [None, None]

        def worker(idx):
            try:
                results[idx] = svc.reprocess_single_frame(sample_clip, params, 0)
            except Exception as e:
                errors[idx] = e

        t1 = threading.Thread(target=worker, args=(0,))
        t2 = threading.Thread(target=worker, args=(1,))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert errors[0] is None, f"Thread 0 error: {errors[0]}"
        assert errors[1] is None, f"Thread 1 error: {errors[1]}"
        assert results[0] is not None
        assert results[1] is not None

        # Verify serialization: no overlapping process_frame calls
        starts = [t for label, t in call_log if label == "start"]
        ends = [t for label, t in call_log if label == "end"]
        assert len(starts) == 2
        assert len(ends) == 2

        # Second call must start after first call ends
        sorted_starts = sorted(starts)
        sorted_ends = sorted(ends)
        assert sorted_starts[1] >= sorted_ends[0], (
            f"Overlap detected: second start {sorted_starts[1]:.4f} "
            f"< first end {sorted_ends[0]:.4f}"
        )


class TestGPULockModelSwitch:
    """Concurrent inference + GVM must switch models without corruption."""

    def test_model_residency_consistent_after_concurrent_access(self):
        svc = CorridorKeyService()
        svc._engine_pool = [MagicMock()]
        svc._gvm_processor = MagicMock()
        svc._active_model = _ActiveModel.INFERENCE

        errors = []

        def switch_to_gvm():
            try:
                with svc._gpu_lock:
                    svc._ensure_model(_ActiveModel.GVM)
            except Exception as e:
                errors.append(e)

        def switch_to_inference():
            try:
                with svc._gpu_lock:
                    svc._ensure_model(_ActiveModel.INFERENCE)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=switch_to_gvm))
            threads.append(threading.Thread(target=switch_to_inference))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Errors during concurrent model switch: {errors}"
        # Model must be in a valid state (one of the enum values)
        assert svc._active_model in _ActiveModel

#!/usr/bin/env python3
"""Benchmark: Can two CorridorKey engines process frames simultaneously?

Tests three scenarios:
  1. Sequential: One engine processes N frames one at a time
  2. Two engines, sequential: Two engines take turns (same GPU, no overlap)
  3. Two engines, concurrent: Two engines process frames simultaneously via CUDA streams

Measures wall-clock time and VRAM usage for each scenario.

Usage:
    python scripts/bench_parallel_frames.py [--frames 10] [--resolution 2048]

Requirements:
    - CorridorKey checkpoint in CorridorKeyModule/checkpoints/
    - Enough VRAM for 2 engine instances (~4 GB minimum)
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
import time
import threading

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from CorridorKeyModule.inference_engine import CorridorKeyEngine


def get_vram_mb() -> float:
    """Current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def get_vram_peak_mb() -> float:
    """Peak GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def make_dummy_frame(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a random image + mask for benchmarking."""
    rng = np.random.default_rng(42)
    image = rng.random((h, w, 3), dtype=np.float32)
    mask = rng.random((h, w), dtype=np.float32)
    return image, mask


def find_checkpoint() -> str:
    ckpt_dir = os.path.join(PROJECT_ROOT, "CorridorKeyModule", "checkpoints")
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .pth checkpoint found in {ckpt_dir}")
    return ckpt_files[0]


def warmup_engine(engine: CorridorKeyEngine, image: np.ndarray, mask: np.ndarray) -> None:
    """Run one warmup frame to trigger torch.compile / CUDA init."""
    print("  Warmup frame...", end=" ", flush=True)
    t0 = time.monotonic()
    engine.process_frame(image, mask)
    print(f"done ({time.monotonic() - t0:.2f}s)")


def bench_sequential(
    engine: CorridorKeyEngine, image: np.ndarray, mask: np.ndarray, n_frames: int
) -> tuple[float, float]:
    """Scenario 1: One engine, N frames sequentially."""
    torch.cuda.reset_peak_memory_stats()

    t0 = time.monotonic()
    for i in range(n_frames):
        engine.process_frame(image, mask)
    elapsed = time.monotonic() - t0

    peak = get_vram_peak_mb()
    return elapsed, peak


def bench_two_engines_sequential(
    engine1: CorridorKeyEngine,
    engine2: CorridorKeyEngine,
    image: np.ndarray,
    mask: np.ndarray,
    n_frames: int,
) -> tuple[float, float]:
    """Scenario 2: Two engines, alternating frames (no overlap)."""
    torch.cuda.reset_peak_memory_stats()

    t0 = time.monotonic()
    for i in range(n_frames):
        if i % 2 == 0:
            engine1.process_frame(image, mask)
        else:
            engine2.process_frame(image, mask)
    elapsed = time.monotonic() - t0

    peak = get_vram_peak_mb()
    return elapsed, peak


def bench_two_engines_concurrent(
    engine1: CorridorKeyEngine,
    engine2: CorridorKeyEngine,
    image: np.ndarray,
    mask: np.ndarray,
    n_frames: int,
) -> tuple[float, float]:
    """Scenario 3: Two engines processing frames concurrently via threads.

    PyTorch releases the GIL during CUDA kernel execution, so two Python
    threads calling process_frame() will naturally overlap GPU work without
    needing explicit CUDA stream management.
    """
    torch.cuda.reset_peak_memory_stats()

    # Each engine processes n_frames // 2 frames
    half = n_frames // 2
    remainder = n_frames - half * 2

    results = {"t1": 0.0, "t2": 0.0, "err1": None, "err2": None}

    def worker(engine, count, key):
        try:
            t0 = time.monotonic()
            for _ in range(count):
                engine.process_frame(image, mask)
            torch.cuda.synchronize()
            results[f"t{key}"] = time.monotonic() - t0
        except Exception as e:
            results[f"err{key}"] = str(e)

    t0 = time.monotonic()

    t1 = threading.Thread(target=worker, args=(engine1, half, 1))
    t2 = threading.Thread(target=worker, args=(engine2, half + remainder, 2))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    elapsed = time.monotonic() - t0
    peak = get_vram_peak_mb()

    if results["err1"]:
        print(f"  WARNING: Engine 1 error: {results['err1']}")
    if results["err2"]:
        print(f"  WARNING: Engine 2 error: {results['err2']}")

    return elapsed, peak


def main():
    parser = argparse.ArgumentParser(description="Benchmark parallel frame processing")
    parser.add_argument(
        "--frames", type=int, default=10, help="Number of frames to process per scenario"
    )
    parser.add_argument("--resolution", type=int, default=2048, help="Input resolution (square)")
    parser.add_argument(
        "--opt-mode",
        type=str,
        default="speed",
        choices=["speed", "lowvram", "auto"],
        help="Optimization mode for engines",
    )
    args = parser.parse_args()

    n_frames = args.frames
    res = args.resolution

    print("=== CorridorKey Parallel Frame Benchmark ===")
    print(f"Frames: {n_frames}, Resolution: {res}x{res}, Opt mode: {args.opt_mode}")
    print()

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_properties(0).name
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_vram:.1f} GB)")
    print()

    # Create dummy frame
    image, mask = make_dummy_frame(res, res)
    ckpt = find_checkpoint()
    print(f"Checkpoint: {os.path.basename(ckpt)}")
    print()

    # --- Scenario 1: Single engine ---
    print("--- Scenario 1: Single engine, sequential ---")
    os.environ["CORRIDORKEY_OPT_MODE"] = args.opt_mode
    engine1 = CorridorKeyEngine(
        checkpoint_path=ckpt, device="cuda", img_size=2048, optimization_mode=args.opt_mode
    )
    vram_after_load1 = get_vram_mb()
    print(f"  VRAM after loading engine 1: {vram_after_load1:.0f} MB")

    warmup_engine(engine1, image, mask)

    elapsed1, peak1 = bench_sequential(engine1, image, mask, n_frames)
    fps1 = n_frames / elapsed1
    print(f"  Result: {n_frames} frames in {elapsed1:.2f}s = {fps1:.2f} fps")
    print(f"  Peak VRAM: {peak1:.0f} MB")
    print()

    # --- Load second engine ---
    print("--- Loading second engine ---")
    engine2 = CorridorKeyEngine(
        checkpoint_path=ckpt, device="cuda", img_size=2048, optimization_mode=args.opt_mode
    )
    vram_after_load2 = get_vram_mb()
    print(f"  VRAM after loading engine 2: {vram_after_load2:.0f} MB")
    print(f"  VRAM increase: {vram_after_load2 - vram_after_load1:.0f} MB")

    warmup_engine(engine2, image, mask)
    print()

    # --- Scenario 2: Two engines, sequential ---
    print("--- Scenario 2: Two engines, alternating (no overlap) ---")
    elapsed2, peak2 = bench_two_engines_sequential(engine1, engine2, image, mask, n_frames)
    fps2 = n_frames / elapsed2
    print(f"  Result: {n_frames} frames in {elapsed2:.2f}s = {fps2:.2f} fps")
    print(f"  Peak VRAM: {peak2:.0f} MB")
    print()

    # --- Scenario 3: Two engines, concurrent ---
    print("--- Scenario 3: Two engines, concurrent (CUDA streams + threads) ---")
    elapsed3, peak3 = bench_two_engines_concurrent(engine1, engine2, image, mask, n_frames)
    fps3 = n_frames / elapsed3
    print(f"  Result: {n_frames} frames in {elapsed3:.2f}s = {fps3:.2f} fps")
    print(f"  Peak VRAM: {peak3:.0f} MB")
    print()

    # --- Summary ---
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<40} {'FPS':>8} {'Speedup':>10} {'Peak VRAM':>12}")
    print("-" * 70)
    print(f"{'1. Single engine, sequential':<40} {fps1:>8.2f} {'1.00x':>10} {peak1:>10.0f} MB")
    print(
        f"{'2. Two engines, alternating':<40} {fps2:>8.2f} {fps2 / fps1:>9.2f}x {peak2:>10.0f} MB"
    )
    print(f"{'3. Two engines, concurrent':<40} {fps3:>8.2f} {fps3 / fps1:>9.2f}x {peak3:>10.0f} MB")
    print()

    if fps3 > fps1 * 1.15:
        print("VERDICT: Concurrent frames DOES help! Implement multi-engine.")
    elif fps3 > fps1 * 1.05:
        print("VERDICT: Marginal benefit (~5-15%). May not be worth the complexity.")
    else:
        print("VERDICT: GPU is already saturated. Concurrent frames don't help at this resolution.")
        print("         CPU/GPU pipeline overlap (already built) is the right approach.")

    # Cleanup
    del engine1, engine2
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

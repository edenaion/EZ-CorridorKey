#!/usr/bin/env python3
"""VRAM profiler — measures GPU memory at each stage of CorridorKey inference.

Reports a per-step breakdown: model weights, input tensors, forward pass,
post-processing, and cleanup.  Useful for identifying bottlenecks on
VRAM-constrained GPUs (e.g. RTX 3080 10 GB).

Usage:
    python scripts/profile_vram.py [--img-size 2048] [--resolution 1080] [--opt-mode auto]
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from CorridorKeyModule.inference_engine import CorridorKeyEngine


# ── helpers ──────────────────────────────────────────────────────────────────

def _mb(b: float) -> float:
    return b / (1024 * 1024)


def _alloc_mb() -> float:
    return _mb(torch.cuda.memory_allocated())


def _reserved_mb() -> float:
    return _mb(torch.cuda.memory_reserved())


def _peak_mb() -> float:
    return _mb(torch.cuda.max_memory_allocated())


def _snap(label: str, rows: list[dict], prev_alloc: float) -> float:
    alloc = _alloc_mb()
    peak = _peak_mb()
    delta = alloc - prev_alloc
    rows.append({
        "step": label,
        "alloc_mb": alloc,
        "peak_mb": peak,
        "delta_mb": delta,
        "reserved_mb": _reserved_mb(),
    })
    return alloc


def _print_table(rows: list[dict]) -> None:
    hdr = f"{'Step':<42} {'Alloc MB':>10} {'Peak MB':>10} {'Delta MB':>10} {'Rsv MB':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        sign = "+" if r["delta_mb"] >= 0 else ""
        print(
            f"{r['step']:<42} {r['alloc_mb']:>10.1f} {r['peak_mb']:>10.1f} "
            f"{sign}{r['delta_mb']:>9.1f} {r['reserved_mb']:>10.1f}"
        )
    print()


def find_checkpoint() -> str:
    ckpt_dir = os.path.join(PROJECT_ROOT, "CorridorKeyModule", "checkpoints")
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .pth checkpoint found in {ckpt_dir}")
    return ckpt_files[0]


# ── resolution presets ───────────────────────────────────────────────────────

_RESOLUTIONS = {
    720: (720, 1280),
    1080: (1080, 1920),
    1440: (1440, 2560),
    2160: (2160, 3840),
}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VRAM profiler for CorridorKey inference")
    parser.add_argument("--img-size", type=int, default=2048,
                        help="Backbone resolution (default: 2048)")
    parser.add_argument("--resolution", type=int, default=1080,
                        choices=sorted(_RESOLUTIONS), help="Input frame resolution preset")
    parser.add_argument("--opt-mode", type=str, default="auto",
                        choices=["speed", "lowvram", "auto"],
                        help="Optimization mode")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup frame (measures eager mode, not compiled)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    gpu = torch.cuda.get_device_properties(0)
    gpu_vram = gpu.total_memory / (1024 ** 3)
    h, w = _RESOLUTIONS[args.resolution]

    print(f"=== CorridorKey VRAM Profiler ===")
    print(f"GPU: {gpu.name} ({gpu_vram:.1f} GB)")
    print(f"Input: {w}x{h}, Backbone: {args.img_size}x{args.img_size}")
    print(f"Opt mode: {args.opt_mode}")
    print()

    rows: list[dict] = []

    # ── Step 0: Baseline ─────────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    prev = _snap("0. Baseline (empty GPU)", rows, 0.0)

    # ── Step 1: Load model ───────────────────────────────────────────────
    ckpt = find_checkpoint()
    os.environ["CORRIDORKEY_OPT_MODE"] = args.opt_mode
    t0 = time.monotonic()
    engine = CorridorKeyEngine(
        checkpoint_path=ckpt,
        device="cuda",
        img_size=args.img_size,
        optimization_mode=args.opt_mode,
    )
    load_time = time.monotonic() - t0
    prev = _snap(f"1. Model loaded ({load_time:.1f}s)", rows, prev)

    # ── Optional warmup (torch.compile first run) ────────────────────────
    if not args.no_warmup:
        torch.cuda.reset_peak_memory_stats()
        dummy_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        dummy_mask = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        t0 = time.monotonic()
        engine.process_frame(dummy_img, dummy_mask)
        warmup_time = time.monotonic() - t0
        prev = _snap(f"1b. Warmup frame ({warmup_time:.1f}s)", rows, prev)
        del dummy_img, dummy_mask
        gc.collect()
        torch.cuda.empty_cache()

    # ── Step 2-7: Instrumented inference ─────────────────────────────────
    # We manually replicate process_frame steps to measure each one.
    print("Running instrumented inference pass...")
    torch.cuda.reset_peak_memory_stats()
    torch.set_grad_enabled(False)
    prev = _snap("2. Pre-inference baseline", rows, _alloc_mb())

    image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    mask = np.random.randint(0, 255, (h, w), dtype=np.uint8)

    # 2a. Upload + resize to img_size
    import torch.nn.functional as F

    img_f = image.astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to("cuda")
    mask_f = mask.astype(np.float32) / 255.0
    mask_t = torch.from_numpy(mask_f).unsqueeze(0).unsqueeze(0).to("cuda")

    img_t = F.interpolate(img_t, size=(args.img_size, args.img_size),
                          mode="bilinear", align_corners=False)
    mask_t = F.interpolate(mask_t, size=(args.img_size, args.img_size),
                           mode="bilinear", align_corners=False)
    prev = _snap("3. Input upload + resize to backbone", rows, prev)

    # 2b. Normalize + concat
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    img_norm = (img_t - mean) / std
    inp = torch.cat([img_norm, mask_t], dim=1)
    prev = _snap("4. Normalize + concat (4ch input)", rows, prev)

    # 2c. Forward pass
    torch.cuda.reset_peak_memory_stats()
    refiner_scale = inp.new_tensor(1.0)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = engine._forward_model(inp, refiner_scale)
    torch.cuda.synchronize()
    fwd_peak = _peak_mb()
    prev = _snap(f"5. Forward pass (peak during: {fwd_peak:.0f} MB)", rows, prev)

    # 2d. Output resize
    pred_alpha = out["alpha"].float()
    pred_fg = out["fg"].float()
    res_alpha = F.interpolate(pred_alpha, size=(h, w), mode="bicubic",
                              align_corners=False).clamp(0, 1)
    res_fg = F.interpolate(pred_fg, size=(h, w), mode="bicubic",
                           align_corners=False).clamp(0, 1)
    prev = _snap("6. Output resize to original", rows, prev)

    # Free forward intermediates
    del inp, img_norm, img_t, mask_t, mean, std, out, pred_alpha, pred_fg
    gc.collect()
    torch.cuda.empty_cache()
    prev = _snap("7. After del intermediates + empty_cache", rows, prev)

    # 2e. GPU→CPU transfer
    _ = res_alpha.cpu().numpy()
    _ = res_fg.cpu().numpy()
    del res_alpha, res_fg
    gc.collect()
    torch.cuda.empty_cache()
    prev = _snap("8. After GPU->CPU transfer + cleanup", rows, prev)

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    _print_table(rows)

    # Recommendations
    model_mb = rows[1]["alloc_mb"] - rows[0]["alloc_mb"]
    fwd_peak_mb = fwd_peak
    total_vram_gb = gpu_vram

    print("=== Analysis ===")
    print(f"Model weights:      {model_mb:>8.0f} MB ({model_mb / 1024:.1f} GB)")
    print(f"Forward peak:       {fwd_peak_mb:>8.0f} MB ({fwd_peak_mb / 1024:.1f} GB)")
    print(f"GPU total:          {total_vram_gb * 1024:>8.0f} MB ({total_vram_gb:.1f} GB)")
    print(f"Headroom:           {total_vram_gb * 1024 - fwd_peak_mb:>8.0f} MB")
    print()

    if fwd_peak_mb > total_vram_gb * 1024 * 0.9:
        print("WARNING: Forward pass uses >90% of VRAM — OOM risk!")
        print("  Recommendations:")
        print(f"  - Try --img-size 1536 (current: {args.img_size})")
        print(f"  - Try --opt-mode lowvram (current: {args.opt_mode})")
    elif fwd_peak_mb > total_vram_gb * 1024 * 0.7:
        print("NOTE: Forward pass uses 70-90% of VRAM — tight but workable")
        print("  Multi-engine parallelism will likely OOM.")
    else:
        print("OK: Comfortable VRAM headroom for this configuration.")


if __name__ == "__main__":
    main()

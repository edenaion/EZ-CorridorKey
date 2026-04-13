"""Quick color analysis between upstream and our composite outputs.

Loads the same frame+mask, runs both engines, compares RGB channels in detail.
"""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import sys
import numpy as np
import cv2
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "CorridorKey-upstream")
sys.path.insert(0, PROJECT_ROOT)


def load_frame(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3 and img.shape[2] >= 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def load_mask(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def run_engine(engine_class, frame, mask, ckpt):
    engine = engine_class(checkpoint_path=ckpt, device="cuda", img_size=2048)
    result = engine.process_frame(frame, mask)
    del engine
    torch.cuda.empty_cache()
    return result


def channel_stats(name, img):
    """Print per-channel statistics."""
    if img.ndim == 2:
        print(f"  {name}: min={img.min():.6f} max={img.max():.6f} mean={img.mean():.6f}")
        return
    labels = ["R", "G", "B", "A"][: img.shape[2]]
    for i, ch in enumerate(labels):
        c = img[:, :, i]
        print(
            f"  {name}[{ch}]: min={c.min():.6f} max={c.max():.6f} "
            f"mean={c.mean():.6f} std={c.std():.6f}"
        )


def compare_channels(name, a, b):
    """Compare two images channel by channel."""
    if a.ndim == 2:
        a = a[:, :, np.newaxis]
    if b.ndim == 2:
        b = b[:, :, np.newaxis]
    labels = ["R", "G", "B", "A"][: a.shape[2]]
    for i, ch in enumerate(labels):
        diff = np.abs(a[:, :, i] - b[:, :, i])
        mse = float(np.mean((a[:, :, i] - b[:, :, i]) ** 2))
        psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
        print(f"  {name}[{ch}]: max_diff={diff.max():.8f} mae={diff.mean():.8f} psnr={psnr:.1f}dB")


def main():
    ckpt_dir = os.path.join(PROJECT_ROOT, "CorridorKeyModule", "checkpoints")
    ckpt = os.path.join(ckpt_dir, [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")][0])

    frame_path = "Projects/260308_181206_jhe-test/clips/Brunette_Plays_With_Hair_original_178131/Frames/frame_000000.exr"
    mask_path = "Projects/260308_181206_jhe-test/clips/Brunette_Plays_With_Hair_original_178131/AlphaHint/frame_000000.png"

    frame = load_frame(os.path.join(PROJECT_ROOT, frame_path))
    mask = load_mask(os.path.join(PROJECT_ROOT, mask_path))
    print(f"Frame: {frame.shape}, dtype={frame.dtype}")
    print(f"Mask:  {mask.shape}, dtype={mask.dtype}")
    print()

    # --- Upstream ---
    print("=" * 60)
    print("UPSTREAM ENGINE")
    print("=" * 60)
    sys.path.insert(0, UPSTREAM_ROOT)
    mods = [k for k in sys.modules if k.startswith("CorridorKeyModule")]
    for m in mods:
        del sys.modules[m]
    from CorridorKeyModule.inference_engine import CorridorKeyEngine as UpEngine

    sys.path.pop(0)

    res_up = run_engine(UpEngine, frame, mask, ckpt)
    print("\nUpstream outputs:")
    channel_stats("alpha", res_up["alpha"])
    channel_stats("fg", res_up["fg"])
    channel_stats("comp", res_up["comp"])
    print()

    # --- Ours ---
    print("=" * 60)
    print("OUR ENGINE")
    print("=" * 60)
    mods = [k for k in sys.modules if k.startswith("CorridorKeyModule")]
    for m in mods:
        del sys.modules[m]
    sys.path.insert(0, PROJECT_ROOT)
    from CorridorKeyModule.inference_engine import CorridorKeyEngine as OurEngine

    res_ours = run_engine(OurEngine, frame, mask, ckpt)
    torch._dynamo.reset()
    print("\nOur outputs:")
    channel_stats("alpha", res_ours["alpha"])
    channel_stats("fg", res_ours["fg"])
    channel_stats("comp", res_ours["comp"])
    print()

    # --- Comparison ---
    print("=" * 60)
    print("CHANNEL-BY-CHANNEL COMPARISON")
    print("=" * 60)

    print("\nAlpha comparison:")
    compare_channels("alpha", res_up["alpha"], res_ours["alpha"])

    print("\nFG comparison:")
    compare_channels("fg", res_up["fg"], res_ours["fg"])

    print("\nComp comparison:")
    compare_channels("comp", res_up["comp"], res_ours["comp"])

    # --- Saturation analysis ---
    print("\n" + "=" * 60)
    print("SATURATION / COLOR ANALYSIS")
    print("=" * 60)

    # Compare mean color of the subject region (where alpha > 0.5)
    a_up = res_up["alpha"][:, :, 0] if res_up["alpha"].ndim == 3 else res_up["alpha"]

    # Subject mask (high alpha = foreground)
    subj_mask = a_up > 0.5

    for name, comp in [("Upstream comp", res_up["comp"]), ("Our comp", res_ours["comp"])]:
        r = comp[:, :, 0][subj_mask]
        g = comp[:, :, 1][subj_mask]
        b = comp[:, :, 2][subj_mask]
        # Simple saturation: (max - min) / max per pixel
        rgb_stack = np.stack([r, g, b], axis=-1)
        mx = rgb_stack.max(axis=-1)
        mn = rgb_stack.min(axis=-1)
        sat = np.where(mx > 0, (mx - mn) / mx, 0)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        print(f"\n  {name} (subject region, alpha > 0.5):")
        print(f"    Mean R={r.mean():.6f} G={g.mean():.6f} B={b.mean():.6f}")
        print(f"    Mean saturation: {sat.mean():.6f}")
        print(f"    Mean luminance:  {lum.mean():.6f}")

    # Direct diff of comp in subject region
    comp_diff = res_up["comp"] - res_ours["comp"]
    print("\n  Comp diff (subject region):")
    for i, ch in enumerate(["R", "G", "B"]):
        d = comp_diff[:, :, i][subj_mask]
        print(f"    {ch}: mean_diff={d.mean():.8f} std={d.std():.8f} (positive=upstream brighter)")


if __name__ == "__main__":
    main()

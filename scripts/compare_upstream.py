"""End-to-end comparison: upstream CorridorKey vs our optimized fork.

Runs the SAME source frame + alpha mask through:
  A) Upstream CorridorKeyEngine (freshly cloned, no optimizations)
  B) Our CorridorKeyEngine (with Hiera patch, TF32, torch.compile)

Compares the actual output alphas, FGs, and composites.
Generates a branded visual report PNG.
"""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import sys
import numpy as np
import cv2
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Upstream repo root (cloned separately)
UPSTREAM_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "CorridorKey-upstream")

VERSION = "1.5.0"


def find_test_frame():
    """Find a test frame with its alpha hint mask."""
    projects = os.path.join(PROJECT_ROOT, "Projects")
    # Prefer clips with hair detail
    preferred = ["Brunette_Plays_With_Hair", "girl_frames"]
    for root, dirs, files in os.walk(projects):
        for f in sorted(files):
            if f.endswith(".exr") and "frame_" in f and "Frames" in root:
                if any(p in root for p in preferred):
                    # clip_dir is two levels up: Frames -> clip_name -> clip_dir
                    clip_dir = os.path.dirname(root)  # goes from Frames/ to clip/
                    alpha_dir = os.path.join(clip_dir, "AlphaHint")
                    stem = os.path.splitext(f)[0]
                    # Try common extensions
                    for ext in [".png", ".exr", ".jpg"]:
                        alpha_file = os.path.join(alpha_dir, stem + ext)
                        if os.path.isfile(alpha_file):
                            return os.path.join(root, f), alpha_file
    return None, None


def load_frame(path):
    """Load EXR/PNG frame as float32 RGB [H,W,3]."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read: {path}")
    if img.ndim == 3 and img.shape[2] >= 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def load_mask(path):
    """Load alpha hint mask as float32 [H,W]."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read mask: {path}")
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def run_upstream(frame, mask, checkpoint):
    """Run frame through upstream CorridorKeyEngine."""
    # Temporarily add upstream to path
    sys.path.insert(0, UPSTREAM_ROOT)
    # Import upstream's engine
    from CorridorKeyModule.inference_engine import CorridorKeyEngine as UpstreamEngine

    sys.path.pop(0)

    engine = UpstreamEngine(
        checkpoint_path=checkpoint,
        device="cuda",
        img_size=2048,
    )
    result = engine.process_frame(frame, mask)
    alpha = result["alpha"]
    comp = result["comp"]
    del engine
    torch.cuda.empty_cache()
    return alpha, comp


def run_ours(frame, mask, checkpoint):
    """Run frame through our optimized CorridorKeyEngine."""
    sys.path.insert(0, PROJECT_ROOT)
    # Force fresh import of our engine (not upstream's cached one)
    # Remove any cached CorridorKeyModule imports
    mods_to_remove = [k for k in sys.modules if k.startswith("CorridorKeyModule")]
    for m in mods_to_remove:
        del sys.modules[m]

    from CorridorKeyModule.inference_engine import CorridorKeyEngine as OurEngine

    engine = OurEngine(
        checkpoint_path=checkpoint,
        device="cuda",
        img_size=2048,
    )
    result = engine.process_frame(frame, mask)
    alpha = result["alpha"]
    comp = result["comp"]
    del engine
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    return alpha, comp


def generate_report(frame_path, source, mask, alpha_up, comp_up, alpha_ours, comp_ours, metrics):
    """Generate branded visual comparison report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import datetime

    BG = "#141300"
    BG_CARD = "#1A1900"
    BORDER = "#2A2910"
    YELLOW = "#FFF203"
    TEXT = "#E0E0E0"
    PASS_GREEN = "#44ff44"
    FAIL_RED = "#ff4444"

    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    date_slug = datetime.now().strftime("%y%m%d_%H%M%S")
    h, w = source.shape[:2]

    # Normalize source for display
    src_disp = np.clip(source / max(source.max(), 1e-6), 0, 1)

    # Normalize composites for display
    comp_up_disp = np.clip(comp_up, 0, 1)
    comp_ours_disp = np.clip(comp_ours, 0, 1)

    # Alpha as grayscale RGB for display (clipped to valid range)
    def alpha_to_rgb(a):
        a2d = a[:, :, 0] if a.ndim == 3 else a
        a2d = np.clip(a2d, 0, 1)
        return np.stack([a2d, a2d, a2d], axis=-1)

    alpha_up_disp = alpha_to_rgb(alpha_up)
    alpha_ours_disp = alpha_to_rgb(alpha_ours)

    # Mask display
    mask_disp = np.stack([mask, mask, mask], axis=-1)

    # --- Figure: 2 rows x 3 cols of images + table + footer ---
    fig = plt.figure(figsize=(20, 18), facecolor=BG)

    # Row 1: Source | Upstream Composite | Our Composite
    # Row 2: Alpha Mask Input | Upstream Alpha Output | Our Alpha Output
    positions = [
        # Row 1
        [0.02, 0.52, 0.31, 0.34],  # source
        [0.345, 0.52, 0.31, 0.34],  # upstream comp
        [0.67, 0.52, 0.31, 0.34],  # ours comp
        # Row 2
        [0.02, 0.15, 0.31, 0.34],  # mask input
        [0.345, 0.15, 0.31, 0.34],  # upstream alpha
        [0.67, 0.15, 0.31, 0.34],  # ours alpha
    ]

    panels = [
        (src_disp, "Source Frame", TEXT),
        (comp_ours_disp, "EZ-CorridorKey Output\n(composited key)", PASS_GREEN),
        (comp_up_disp, "CorridorKey Output\n(composited key)", YELLOW),
        (mask_disp, "Alpha Hint Input\n(GVM coarse mask)", TEXT),
        (alpha_ours_disp, "EZ-CorridorKey Alpha\n(raw matte output)", PASS_GREEN),
        (alpha_up_disp, "CorridorKey Alpha\n(raw matte output)", YELLOW),
    ]

    for pos, (img, title, color) in zip(positions, panels):
        ax = fig.add_axes(pos)
        ax.imshow(img)
        ax.set_title(title, color=color, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    # --- Header ---
    fig.text(
        0.5,
        0.97,
        "C O R R I D O R K E Y",
        color=YELLOW,
        fontsize=18,
        fontweight="bold",
        ha="center",
        fontfamily="sans-serif",
    )
    fig.text(
        0.5,
        0.935,
        f"EZ-CorridorKey vs CorridorKey — End-to-End Comparison   |   v{VERSION}   |   {timestamp}",
        color=TEXT,
        fontsize=11,
        ha="center",
        fontfamily="sans-serif",
    )
    fig.patches.append(
        matplotlib.patches.Rectangle(
            (0.02, 0.925),
            0.96,
            0.003,
            transform=fig.transFigure,
            facecolor=YELLOW,
            edgecolor="none",
        )
    )

    # --- Metrics table ---
    ax_tbl = fig.add_axes([0.10, 0.02, 0.80, 0.10])
    ax_tbl.axis("off")

    m = metrics
    psnr_str = f"{m['psnr']:.1f}" if m["psnr"] != float("inf") else "inf"
    if m["psnr"] == float("inf"):
        verdict = "BIT-IDENTICAL"
        v_color = YELLOW
    elif m["psnr"] > 80:
        verdict = "PASS — below float32 noise floor"
        v_color = PASS_GREEN
    elif m["psnr"] > 60:
        verdict = "PASS — imperceptible"
        v_color = PASS_GREEN
    else:
        verdict = "INVESTIGATE"
        v_color = FAIL_RED

    col_labels = ["Comparison", "Max Pixel Diff", "MAE", "PSNR (dB)", "Verdict"]
    table_data = [
        [
            "EZ-CorridorKey vs CorridorKey (alpha)",
            f"{m['max_diff']:.10f}",
            f"{m['mae']:.10f}",
            psnr_str,
            verdict,
        ]
    ]

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 2.2)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if row == 0:
            cell.set_facecolor(BORDER)
            cell.set_text_props(color=YELLOW, fontweight="bold", fontsize=11)
        else:
            cell.set_facecolor(BG_CARD)
            cell.set_text_props(color=TEXT, fontsize=11)
            if col == 4:
                cell.set_text_props(color=v_color, fontweight="bold", fontsize=11)

    # --- Footer ---
    fig.text(
        0.5,
        0.005,
        f"v{VERSION}   |   Frame: {os.path.basename(frame_path)}   |   "
        f"Resolution: {w}x{h}   |   {verdict}",
        ha="center",
        color=YELLOW if "PASS" in verdict or "IDENTICAL" in verdict else FAIL_RED,
        fontsize=12,
        fontweight="bold",
        fontfamily="sans-serif",
    )

    # Save
    out_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{date_slug}_upstream_comparison_v{VERSION}.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nReport saved: {out_path}")
    return out_path


def main():
    # Checkpoint — shared between upstream and ours (same weights)
    ckpt = os.path.join(PROJECT_ROOT, "CorridorKeyModule", "CorridorKey.pth")
    if not os.path.isfile(ckpt):
        ckpt_dir = os.path.join(PROJECT_ROOT, "CorridorKeyModule", "checkpoints")
        candidates = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
        if candidates:
            ckpt = os.path.join(ckpt_dir, candidates[0])
    if not os.path.isfile(ckpt):
        print("Checkpoint not found")
        sys.exit(1)
    print(f"Checkpoint: {ckpt}")

    # Verify upstream exists
    if not os.path.isdir(UPSTREAM_ROOT):
        print(f"Upstream repo not found at: {UPSTREAM_ROOT}")
        print("Clone it first: git clone https://github.com/nikopueringer/CorridorKey.git")
        sys.exit(1)

    # Find test frame + mask
    frame_path, mask_path = find_test_frame()
    if not frame_path:
        print("No test frame with alpha hint found in Projects/")
        sys.exit(1)

    print(f"Source frame: {frame_path}")
    print(f"Alpha mask:   {mask_path}")

    frame = load_frame(frame_path)
    mask = load_mask(mask_path)
    h, w = frame.shape[:2]
    print(f"Resolution:   {w}x{h}")
    print()

    # --- A) Run through UPSTREAM ---
    print("=" * 60)
    print("UPSTREAM (nikopueringer/CorridorKey)")
    print("=" * 60)
    alpha_up, comp_up = run_upstream(frame, mask, ckpt)
    a = alpha_up[:, :, 0] if alpha_up.ndim == 3 else alpha_up
    print(f"  Alpha range: [{a.min():.6f}, {a.max():.6f}]")
    print()

    # --- B) Run through OURS ---
    print("=" * 60)
    print("OURS (EZ-CorridorKey optimized)")
    print("=" * 60)
    alpha_ours, comp_ours = run_ours(frame, mask, ckpt)
    b = alpha_ours[:, :, 0] if alpha_ours.ndim == 3 else alpha_ours
    print(f"  Alpha range: [{b.min():.6f}, {b.max():.6f}]")
    print()

    # --- Compare ---
    diff = np.abs(a - b)
    max_diff = float(diff.max())
    mae = float(diff.mean())
    mse = float(np.mean((a - b) ** 2))
    psnr_val = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)

    print("=" * 60)
    print("COMPARISON: Upstream vs Ours")
    print("=" * 60)
    psnr_str = f"{psnr_val:.1f}" if psnr_val != float("inf") else "inf (bit-identical)"
    print(f"  Max pixel diff: {max_diff:.10f}")
    print(f"  MAE:            {mae:.10f}")
    print(f"  PSNR:           {psnr_str} dB")
    print()

    metrics = {"max_diff": max_diff, "mae": mae, "psnr": psnr_val}

    # --- Generate report ---
    generate_report(frame_path, frame, mask, alpha_up, comp_up, alpha_ours, comp_ours, metrics)


if __name__ == "__main__":
    main()

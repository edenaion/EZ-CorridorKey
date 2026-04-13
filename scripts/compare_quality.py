"""A/B quality comparison between two CorridorKey alpha output directories.

Usage:
    python scripts/compare_quality.py <dir_a> <dir_b>

Where dir_a and dir_b contain alpha EXR files from the same clip processed
with different engine configs (e.g., baseline vs optimized).

Metrics:
    - Max absolute pixel difference
    - Mean absolute error (MAE)
    - PSNR (Peak Signal-to-Noise Ratio) — >60dB is essentially identical
    - SSIM (Structural Similarity) — >0.999 is perceptually identical
"""

import sys
import os
import glob
import numpy as np

try:
    import OpenEXR
    import Imath

    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def load_exr_alpha(path: str) -> np.ndarray:
    """Load alpha channel from EXR file as float32 array."""
    if HAS_OPENEXR:
        exr = OpenEXR.InputFile(path)
        header = exr.header()
        dw = header["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        # Try A channel first, fall back to R
        channels = header["channels"]
        ch_name = "A" if "A" in channels else "R"
        raw = exr.channel(ch_name, Imath.PixelType(Imath.PixelType.FLOAT))
        return np.frombuffer(raw, dtype=np.float32).reshape(h, w)
    elif HAS_CV2:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot read: {path}")
        if img.ndim == 3 and img.shape[2] >= 4:
            return img[:, :, 3].astype(np.float32)
        elif img.ndim == 3:
            return img[:, :, 0].astype(np.float32)
        return img.astype(np.float32)
    else:
        raise ImportError("Need OpenEXR or cv2 to read EXR files")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Simplified SSIM (no windowing, whole-image)."""
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = np.mean((a - mu_a) * (b - mu_b))
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(num / den)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    dir_a, dir_b = sys.argv[1], sys.argv[2]

    files_a = sorted(glob.glob(os.path.join(dir_a, "*.exr")))
    files_b = sorted(glob.glob(os.path.join(dir_b, "*.exr")))

    if not files_a:
        print(f"No EXR files found in {dir_a}")
        sys.exit(1)
    if not files_b:
        print(f"No EXR files found in {dir_b}")
        sys.exit(1)

    # Match by filename
    names_a = {os.path.basename(f): f for f in files_a}
    names_b = {os.path.basename(f): f for f in files_b}
    common = sorted(set(names_a) & set(names_b))

    if not common:
        print("No matching filenames between directories")
        sys.exit(1)

    print(f"Comparing {len(common)} frames...")
    print(f"  A: {dir_a}")
    print(f"  B: {dir_b}")
    print()

    all_max_diff = []
    all_mae = []
    all_psnr = []
    all_ssim = []

    for name in common:
        a = load_exr_alpha(names_a[name])
        b = load_exr_alpha(names_b[name])

        if a.shape != b.shape:
            print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape} — SKIPPED")
            continue

        diff = np.abs(a - b)
        max_diff = float(diff.max())
        mae = float(diff.mean())
        p = psnr(a, b)
        s = ssim_simple(a, b)

        all_max_diff.append(max_diff)
        all_mae.append(mae)
        all_psnr.append(p)
        all_ssim.append(s)

    print(f"{'Metric':<25} {'Min':>12} {'Mean':>12} {'Max':>12}")
    print("-" * 63)
    print(
        f"{'Max pixel difference':<25} {min(all_max_diff):>12.8f} {np.mean(all_max_diff):>12.8f} {max(all_max_diff):>12.8f}"
    )
    print(
        f"{'Mean absolute error':<25} {min(all_mae):>12.8f} {np.mean(all_mae):>12.8f} {max(all_mae):>12.8f}"
    )
    print(
        f"{'PSNR (dB)':<25} {min(all_psnr):>12.2f} {np.mean(all_psnr):>12.2f} {max(all_psnr):>12.2f}"
    )
    print(f"{'SSIM':<25} {min(all_ssim):>12.8f} {np.mean(all_ssim):>12.8f} {max(all_ssim):>12.8f}")
    print()

    avg_psnr = np.mean(all_psnr)
    avg_max = np.mean(all_max_diff)

    if avg_psnr == float("inf"):
        print("RESULT: IDENTICAL — zero difference between A and B")
    elif avg_psnr > 60:
        print(f"RESULT: EFFECTIVELY IDENTICAL — PSNR {avg_psnr:.1f}dB (>60dB)")
    elif avg_psnr > 40:
        print(f"RESULT: VERY CLOSE — PSNR {avg_psnr:.1f}dB, max pixel diff {avg_max:.6f}")
    else:
        print(f"RESULT: SIGNIFICANT DIFFERENCE — PSNR {avg_psnr:.1f}dB — investigate!")


if __name__ == "__main__":
    main()

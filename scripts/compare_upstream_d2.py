"""D2 comparison: upstream CorridorKey vs EZ-CorridorKey with supplemental metrics.

Runs the same source frame and alpha hint through:
  A) Upstream CorridorKeyEngine
  B) Our optimized CorridorKeyEngine

Reports:
  - Pixel fidelity: max diff, MAE, PSNR
  - Perceptual similarity: SSIM, MS-SSIM, LPIPS
  - Color fidelity: saturation delta, luminance delta, DeltaE2000 on skin-like pixels

The default input behavior intentionally matches compare_upstream.py so the D2
report stays apples-to-apples with the original harness unless you override it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch

try:
    from skimage.color import deltaE_ciede2000, rgb2lab
    from skimage.metrics import structural_similarity as skimage_ssim
except ImportError:
    deltaE_ciede2000 = None
    rgb2lab = None
    skimage_ssim = None

try:
    from pytorch_msssim import ms_ssim as torch_ms_ssim
except ImportError:
    torch_ms_ssim = None

try:
    import lpips
except ImportError:
    lpips = None


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
UPSTREAM_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "CorridorKey-upstream")

VERSION = "1.5.0"

DEFAULT_IMG_SIZE = 2048
DEFAULT_SUBJECT_ALPHA = 0.50
DEFAULT_SKIN_ALPHA = 0.85
DEFAULT_MASK_PAD = 24
MIN_SKIN_PIXELS = 256

_LPIPS_MODEL = None
_LPIPS_DEVICE = None
_LPIPS_ERROR: str | None = None


@dataclass
class MetricRow:
    name: str
    pixel_count: int
    max_diff: float
    mae: float
    psnr: float
    verdict: str
    sat_delta: float | None = None
    lum_delta: float | None = None
    ssim: float | None = None
    ms_ssim: float | None = None
    lpips: float | None = None
    deltae_mean: float | None = None
    deltae_p95: float | None = None
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare upstream CorridorKey against EZ-CorridorKey with D2 supplemental metrics.",
    )
    parser.add_argument("--frame", help="Absolute or project-relative frame path.")
    parser.add_argument("--mask", help="Absolute or project-relative alpha hint path.")
    parser.add_argument("--checkpoint", help="Absolute or project-relative checkpoint path.")
    parser.add_argument(
        "--input-color-space",
        choices=("match-original", "auto", "linear", "srgb"),
        default="match-original",
        help=(
            "How to interpret the source frame before process_frame(). "
            "Default: match-original (preserves compare_upstream.py behavior)."
        ),
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help=f"Model input size for both engines. Default: {DEFAULT_IMG_SIZE}.",
    )
    parser.add_argument(
        "--subject-alpha",
        type=float,
        default=DEFAULT_SUBJECT_ALPHA,
        help=f"Alpha threshold for subject metrics. Default: {DEFAULT_SUBJECT_ALPHA}.",
    )
    parser.add_argument(
        "--skin-alpha",
        type=float,
        default=DEFAULT_SKIN_ALPHA,
        help=f"Alpha threshold for skin metrics. Default: {DEFAULT_SKIN_ALPHA}.",
    )
    return parser.parse_args()


def resolve_path(path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def find_test_frame() -> tuple[str | None, str | None]:
    projects = os.path.join(PROJECT_ROOT, "Projects")
    preferred = ["Brunette_Plays_With_Hair", "girl_frames"]
    for root, _dirs, files in os.walk(projects):
        for fname in sorted(files):
            if not fname.endswith(".exr"):
                continue
            if "frame_" not in fname or "Frames" not in root:
                continue
            if not any(tag in root for tag in preferred):
                continue
            clip_dir = os.path.dirname(root)
            alpha_dir = os.path.join(clip_dir, "AlphaHint")
            stem = os.path.splitext(fname)[0]
            for ext in (".png", ".exr", ".jpg"):
                alpha_file = os.path.join(alpha_dir, stem + ext)
                if os.path.isfile(alpha_file):
                    return os.path.join(root, fname), alpha_file
    return None, None


def load_frame(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"Cannot read frame: {path}")
    if img.ndim == 3 and img.shape[2] >= 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def load_mask(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"Cannot read mask: {path}")
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def find_checkpoint(user_path: str | None) -> str:
    resolved = resolve_path(user_path)
    if resolved and os.path.isfile(resolved):
        return resolved

    direct = os.path.join(PROJECT_ROOT, "CorridorKeyModule", "CorridorKey.pth")
    if os.path.isfile(direct):
        return direct

    ckpt_dir = os.path.join(PROJECT_ROOT, "CorridorKeyModule", "checkpoints")
    candidates = sorted(
        os.path.join(ckpt_dir, fname) for fname in os.listdir(ckpt_dir) if fname.endswith(".pth")
    )
    if not candidates:
        raise FileNotFoundError("Checkpoint not found in CorridorKeyModule/checkpoints")
    return candidates[0]


def input_is_linear_for(frame_path: str, color_space: str) -> tuple[bool, str]:
    if color_space == "match-original":
        return False, "sRGB (match-original compare_upstream.py behavior)"
    if color_space == "linear":
        return True, "linear (forced)"
    if color_space == "srgb":
        return False, "sRGB (forced)"
    if frame_path.lower().endswith(".exr"):
        return True, "linear (auto from .exr)"
    return False, "sRGB (auto from file extension)"


def clear_corridorkey_modules() -> None:
    mods_to_remove = [
        name
        for name in list(sys.modules)
        if name == "CorridorKeyModule" or name.startswith("CorridorKeyModule.")
    ]
    for name in mods_to_remove:
        del sys.modules[name]


def import_engine(repo_root: str):
    clear_corridorkey_modules()
    sys.path.insert(0, repo_root)
    try:
        from CorridorKeyModule.inference_engine import CorridorKeyEngine

        return CorridorKeyEngine
    finally:
        sys.path.pop(0)


def run_engine(
    repo_root: str,
    frame: np.ndarray,
    mask: np.ndarray,
    checkpoint: str,
    img_size: int,
    input_is_linear: bool,
) -> dict[str, np.ndarray]:
    Engine = import_engine(repo_root)
    engine = Engine(checkpoint_path=checkpoint, device="cuda", img_size=img_size)
    try:
        result = engine.process_frame(frame, mask, input_is_linear=input_is_linear)
        return {
            "alpha": np.asarray(result["alpha"], dtype=np.float32),
            "fg": np.asarray(result["fg"], dtype=np.float32),
            "comp": np.asarray(result["comp"], dtype=np.float32),
        }
    finally:
        del engine
        torch.cuda.empty_cache()
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()


def _select_values(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        if arr.ndim == 2:
            return arr.reshape(-1)
        return arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 2:
        return arr[mask]
    return arr[mask]


def verdict_for_psnr(psnr: float) -> str:
    if psnr == float("inf"):
        return "BIT-IDENTICAL"
    if psnr > 80.0:
        return "PASS - below float32 noise floor"
    if psnr > 60.0:
        return "PASS - imperceptible"
    return "INVESTIGATE"


def compute_metrics(
    name: str, a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None
) -> MetricRow:
    a_sel = np.asarray(_select_values(a, mask), dtype=np.float64)
    b_sel = np.asarray(_select_values(b, mask), dtype=np.float64)
    if a_sel.shape != b_sel.shape:
        raise ValueError(f"Shape mismatch for {name}: {a_sel.shape} vs {b_sel.shape}")
    if a_sel.size == 0:
        raise ValueError(f"No pixels selected for {name}")

    diff = a_sel - b_sel
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff**2))
    psnr = float("inf") if mse == 0.0 else 10.0 * np.log10(1.0 / mse)
    pixel_count = int(a_sel.shape[0]) if a_sel.ndim > 1 else int(a_sel.size)

    return MetricRow(
        name=name,
        pixel_count=pixel_count,
        max_diff=float(abs_diff.max()),
        mae=float(abs_diff.mean()),
        psnr=psnr,
        verdict=verdict_for_psnr(psnr),
    )


def compute_subject_mask(
    alpha_up: np.ndarray, alpha_ours: np.ndarray, threshold: float
) -> np.ndarray:
    a = alpha_up[:, :, 0] if alpha_up.ndim == 3 else alpha_up
    b = alpha_ours[:, :, 0] if alpha_ours.ndim == 3 else alpha_ours
    return np.maximum(np.clip(a, 0.0, 1.0), np.clip(b, 0.0, 1.0)) > threshold


def compute_saturation(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    return np.where(mx > 0.0, (mx - mn) / mx, 0.0)


def compute_luminance(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]


def attach_color_drift(
    row: MetricRow, comp_up: np.ndarray, comp_ours: np.ndarray, mask: np.ndarray
) -> None:
    up = np.clip(comp_up[mask], 0.0, 1.0).astype(np.float64)
    ours = np.clip(comp_ours[mask], 0.0, 1.0).astype(np.float64)
    if up.size == 0 or ours.size == 0:
        return

    sat_up = compute_saturation(up)
    sat_ours = compute_saturation(ours)
    lum_up = compute_luminance(up)
    lum_ours = compute_luminance(ours)

    row.sat_delta = float(sat_ours.mean() - sat_up.mean())
    row.lum_delta = float(lum_ours.mean() - lum_up.mean())


def detect_skin_mask(
    comp_up: np.ndarray,
    comp_ours: np.ndarray,
    subject_mask: np.ndarray,
    alpha_mask: np.ndarray,
) -> np.ndarray | None:
    comp_mean = np.clip((comp_up + comp_ours) * 0.5, 0.0, 1.0)
    comp_uint8 = (comp_mean * 255.0 + 0.5).astype(np.uint8)
    ycrcb = cv2.cvtColor(comp_uint8, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    skin_mask = (
        subject_mask & alpha_mask & (y > 40) & (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    )
    if int(skin_mask.sum()) < MIN_SKIN_PIXELS:
        return None
    return skin_mask


def alpha_to_rgb(alpha: np.ndarray) -> np.ndarray:
    a = alpha[:, :, 0] if alpha.ndim == 3 else alpha
    a = np.clip(a, 0.0, 1.0)
    return np.stack([a, a, a], axis=-1)


def _backend_status() -> list[str]:
    status = []
    status.append(
        "SSIM/DeltaE: ready" if skimage_ssim is not None else "SSIM/DeltaE: missing scikit-image"
    )
    status.append(
        "MS-SSIM: ready" if torch_ms_ssim is not None else "MS-SSIM: missing pytorch-msssim"
    )
    if lpips is not None:
        status.append("LPIPS: ready")
    else:
        status.append("LPIPS: missing lpips")
    return status


def _bbox_from_mask(
    mask: np.ndarray, pad: int = DEFAULT_MASK_PAD
) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, mask.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, mask.shape[1])
    return y0, y1, x0, x1


def _crop_and_fill(image: np.ndarray, mask: np.ndarray, fill: float) -> np.ndarray | None:
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return None
    y0, y1, x0, x1 = bbox
    cropped = np.clip(image[y0:y1, x0:x1], 0.0, 1.0).copy()
    cropped_mask = mask[y0:y1, x0:x1]
    if cropped.ndim == 2:
        cropped[~cropped_mask] = fill
    else:
        cropped[~cropped_mask] = fill
    return cropped


def _prepare_tensor(image: np.ndarray) -> torch.Tensor:
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.contiguous()


def _compute_ssim(a: np.ndarray, b: np.ndarray) -> float | None:
    if skimage_ssim is None:
        return None
    if a.ndim == 2:
        return float(skimage_ssim(a, b, data_range=1.0))
    return float(skimage_ssim(a, b, data_range=1.0, channel_axis=-1))


def _compute_ms_ssim(a: np.ndarray, b: np.ndarray) -> float | None:
    if torch_ms_ssim is None:
        return None
    if min(a.shape[0], a.shape[1]) < 32:
        return None
    t_a = _prepare_tensor(a)
    t_b = _prepare_tensor(b)
    try:
        value = torch_ms_ssim(t_a, t_b, data_range=1.0, size_average=True)
        return float(value.item())
    except Exception:
        return None


def _get_lpips_model():
    global _LPIPS_DEVICE
    global _LPIPS_ERROR
    global _LPIPS_MODEL

    if lpips is None:
        return None
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    if _LPIPS_ERROR is not None:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = lpips.LPIPS(net="alex")
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _LPIPS_MODEL = model
        _LPIPS_DEVICE = device
        return _LPIPS_MODEL
    except Exception as exc:
        _LPIPS_ERROR = f"{type(exc).__name__}: {exc}"
        logger.warning(f"LPIPS unavailable: {_LPIPS_ERROR}")
        return None


def _compute_lpips(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.ndim != 3 or a.shape[2] != 3:
        return None
    model = _get_lpips_model()
    if model is None or _LPIPS_DEVICE is None:
        return None
    t_a = _prepare_tensor(a).to(_LPIPS_DEVICE) * 2.0 - 1.0
    t_b = _prepare_tensor(b).to(_LPIPS_DEVICE) * 2.0 - 1.0
    with torch.inference_mode():
        value = model(t_a, t_b)
    return float(value.item())


def _compute_deltae(
    comp_up: np.ndarray, comp_ours: np.ndarray, mask: np.ndarray
) -> tuple[float, float] | tuple[None, None]:
    if rgb2lab is None or deltaE_ciede2000 is None:
        return None, None
    if not np.any(mask):
        return None, None
    up_lab = rgb2lab(np.clip(comp_up, 0.0, 1.0))
    ours_lab = rgb2lab(np.clip(comp_ours, 0.0, 1.0))
    delta = deltaE_ciede2000(up_lab, ours_lab)
    masked = delta[mask]
    if masked.size == 0:
        return None, None
    return float(masked.mean()), float(np.percentile(masked, 95))


def _attach_image_metrics(
    row: MetricRow, a: np.ndarray, b: np.ndarray, allow_lpips: bool = True
) -> None:
    row.ssim = _compute_ssim(a, b)
    row.ms_ssim = _compute_ms_ssim(a, b)
    row.lpips = _compute_lpips(a, b) if allow_lpips else None


def _attach_subject_patch_metrics(
    row: MetricRow, up: np.ndarray, ours: np.ndarray, mask: np.ndarray, fill: float = 0.5
) -> None:
    patch_up = _crop_and_fill(up, mask, fill)
    patch_ours = _crop_and_fill(ours, mask, fill)
    if patch_up is None or patch_ours is None:
        return
    _attach_image_metrics(row, patch_up, patch_ours)


def _apply_supplemental_verdict(row: MetricRow) -> None:
    if row.verdict == "SKIPPED":
        return
    if row.ssim is not None and row.ssim < 0.99:
        row.verdict = "INVESTIGATE"
        return
    if row.ms_ssim is not None and row.ms_ssim < 0.99:
        row.verdict = "INVESTIGATE"
        return
    if row.lpips is not None and row.lpips > 0.02:
        row.verdict = "INVESTIGATE"
        return
    if row.deltae_p95 is not None and row.deltae_p95 > 2.0:
        row.verdict = "INVESTIGATE"


def metric_row_to_diff_table(row: MetricRow) -> list[str]:
    psnr_str = "inf" if row.psnr == float("inf") else f"{row.psnr:.1f}"
    sat_str = "-" if row.sat_delta is None else f"{row.sat_delta:+.6f}"
    lum_str = "-" if row.lum_delta is None else f"{row.lum_delta:+.6f}"
    return [
        row.name,
        f"{row.pixel_count:,}",
        f"{row.max_diff:.10f}",
        f"{row.mae:.10f}",
        psnr_str,
        sat_str,
        lum_str,
        row.verdict,
    ]


def metric_row_to_perceptual_table(row: MetricRow) -> list[str]:
    return [
        row.name,
        "-" if row.ssim is None else f"{row.ssim:.6f}",
        "-" if row.ms_ssim is None else f"{row.ms_ssim:.6f}",
        "-" if row.lpips is None else f"{row.lpips:.6f}",
        "-" if row.deltae_mean is None else f"{row.deltae_mean:.6f}",
        "-" if row.deltae_p95 is None else f"{row.deltae_p95:.6f}",
    ]


def verdict_color(verdict: str, pass_green: str, fail_red: str, yellow: str) -> str:
    if verdict == "BIT-IDENTICAL":
        return yellow
    if verdict.startswith("PASS"):
        return pass_green
    if verdict == "SKIPPED":
        return yellow
    return fail_red


def summarize_overall(rows: list[MetricRow]) -> tuple[str, str]:
    if any(row.verdict == "INVESTIGATE" for row in rows):
        return "DIFFERENCES DETECTED - INVESTIGATE", "#ff4444"
    if any(row.verdict.startswith("PASS") for row in rows):
        return "PASS - pixel and perceptual deltas are below concern", "#44ff44"
    return "BIT-IDENTICAL", "#FFF203"


def _render_table(
    ax,
    rows: list[MetricRow],
    col_labels: list[str],
    cell_text: list[list[str]],
    highlight_verdict: bool,
) -> None:
    BG_CARD = "#1A1900"
    BORDER = "#2A2910"
    YELLOW = "#FFF203"
    TEXT = "#E0E0E0"
    PASS_GREEN = "#44ff44"
    FAIL_RED = "#ff4444"

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER)
        if row_idx == 0:
            cell.set_facecolor(BORDER)
            cell.set_text_props(color=YELLOW, fontweight="bold", fontsize=9)
            continue

        metric = rows[row_idx - 1]
        cell.set_facecolor(BG_CARD)
        if highlight_verdict and col_idx == len(col_labels) - 1:
            cell.set_text_props(
                color=verdict_color(metric.verdict, PASS_GREEN, FAIL_RED, YELLOW),
                fontweight="bold",
                fontsize=9,
            )
        else:
            cell.set_text_props(color=TEXT, fontsize=9)


def generate_report(
    frame_path: str,
    source: np.ndarray,
    mask: np.ndarray,
    res_up: dict[str, np.ndarray],
    res_ours: dict[str, np.ndarray],
    rows: list[MetricRow],
    input_mode_label: str,
    img_size: int,
    subject_alpha: float,
    skin_alpha: float,
) -> str:
    import base64
    from datetime import datetime
    from io import BytesIO
    from PIL import Image

    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    date_slug = datetime.now().strftime("%y%m%d_%H%M%S")
    h, w = source.shape[:2]

    def to_b64_png(arr: np.ndarray) -> str:
        img = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            pil = Image.fromarray(img, mode="L")
        else:
            pil = Image.fromarray(img)
        buf = BytesIO()
        pil.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode()

    comp_up_disp = np.clip(res_up["comp"], 0.0, 1.0)
    comp_ours_disp = np.clip(res_ours["comp"], 0.0, 1.0)
    alpha_up_disp = alpha_to_rgb(res_up["alpha"])
    alpha_ours_disp = alpha_to_rgb(res_ours["alpha"])

    comp_up_b64 = to_b64_png(comp_up_disp)
    comp_ours_b64 = to_b64_png(comp_ours_disp)
    alpha_up_b64 = to_b64_png(alpha_up_disp)
    alpha_ours_b64 = to_b64_png(alpha_ours_disp)

    overall_text, overall_color = summarize_overall(rows)

    def verdict_html_color(verdict: str) -> str:
        if verdict == "BIT-IDENTICAL":
            return "#FFF203"
        if verdict.startswith("PASS"):
            return "#44ff44"
        if verdict == "SKIPPED":
            return "#FFF203"
        return "#ff4444"

    diff_rows_html = ""
    for row in rows:
        psnr_str = "inf" if row.psnr == float("inf") else f"{row.psnr:.1f}"
        sat_str = "-" if row.sat_delta is None else f"{row.sat_delta:+.6f}"
        lum_str = "-" if row.lum_delta is None else f"{row.lum_delta:+.6f}"
        vc = verdict_html_color(row.verdict)
        diff_rows_html += f"""<tr>
            <td>{row.name}</td><td>{row.pixel_count:,}</td><td>{row.max_diff:.10f}</td>
            <td>{row.mae:.10f}</td><td>{psnr_str}</td><td>{sat_str}</td><td>{lum_str}</td>
            <td style="color:{vc};font-weight:bold">{row.verdict}</td></tr>\n"""

    perc_rows_html = ""
    for row in rows:
        perc_rows_html += f"""<tr>
            <td>{row.name}</td>
            <td>{"-" if row.ssim is None else f"{row.ssim:.6f}"}</td>
            <td>{"-" if row.ms_ssim is None else f"{row.ms_ssim:.6f}"}</td>
            <td>{"-" if row.lpips is None else f"{row.lpips:.6f}"}</td>
            <td>{"-" if row.deltae_mean is None else f"{row.deltae_mean:.6f}"}</td>
            <td>{"-" if row.deltae_p95 is None else f"{row.deltae_p95:.6f}"}</td></tr>\n"""

    # Inline logo SVG (from ui/theme/corridorkey_logo.svg)
    logo_svg = (
        '<svg width="40" height="40" viewBox="0 0 832 832" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="832" height="832" rx="400" fill="#101010"/>'
        '<path d="M462.509 294.801L463.586 209.304L530.556 144L527.813 361.772L462.509 294.801Z" fill="#2DC451"/>'
        '<path d="M620.364 372.557L534.955 368.531L472 299.348L689.547 309.602L620.364 372.557Z" fill="#2DC451"/>'
        '<path d="M400 369.241L460.46 429.701L554 429.701L400 275.701L400 369.241Z" fill="#2DC451"/>'
        '<path d="M269.54 474L330 413.54V320L176 474H269.54Z" fill="#FFF203"/>'
        '<path d="M330 562.46L269.54 502H176L330 656V562.46Z" fill="#FFF203"/>'
        '<path d="M418.46 502L358 562.46V656L512 502H418.46Z" fill="#FFF203"/>'
        "</svg>"
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background: linear-gradient(135deg, rgb(10,9,0) 0%, rgb(22,21,2) 100%); color:#E0E0E0; font-family:'Segoe UI',Arial,sans-serif; padding:20px 30px; min-height:100vh; }}
.header {{ text-align:center; margin-bottom:8px; }}
.header-row {{ display:flex; align-items:center; justify-content:center; gap:12px; }}
.title {{ color:#FFF203; font-size:24px; font-weight:bold; }}
.subtitle {{ color:#E0E0E0; font-size:13px; margin-top:4px; }}
.divider {{ height:3px; background:#FFF203; margin:8px 20px 16px; }}
.panels {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:16px; }}
.col {{ display:flex; flex-direction:column; gap:8px; }}
.panel {{ text-align:center; }}
.panel img {{ width:100%; display:block; }}
.panel-title {{ font-size:12px; font-weight:bold; margin-bottom:4px; }}
.col-left .panel-title {{ color:#FFF203; }}
.col-right .panel-title {{ color:#44ff44; }}
table {{ width:100%; border-collapse:collapse; font-size:11px; margin-bottom:10px; }}
th {{ background:#2A2910; color:#FFF203; font-weight:bold; padding:5px 8px; border:1px solid #2A2910; }}
td {{ background:#1A1900; padding:4px 8px; border:1px solid #2A2910; text-align:center; }}
.info {{ text-align:center; color:#E0E0E0; font-size:11px; margin-bottom:8px; }}
.verdict-bar {{ text-align:center; font-size:14px; font-weight:bold; color:{overall_color}; margin-top:8px; }}
</style></head><body>
<div class="header">
    <div class="header-row">{logo_svg}<span class="title">CORRIDOR<span style="color:#2DC451">KEY</span></span></div>
    <div class="subtitle">CorridorKey vs EZ-CorridorKey &mdash; Quality Comparison &nbsp;|&nbsp; v{VERSION} &nbsp;|&nbsp; {timestamp}</div>
</div>
<div class="divider"></div>
<div class="panels">
    <div class="col col-left">
        <div class="panel"><div class="panel-title">CorridorKey &mdash; Composite</div><img src="data:image/png;base64,{comp_up_b64}"></div>
        <div class="panel"><div class="panel-title">CorridorKey &mdash; Alpha Matte</div><img src="data:image/png;base64,{alpha_up_b64}"></div>
    </div>
    <div class="col col-right">
        <div class="panel"><div class="panel-title">EZ-CorridorKey &mdash; Composite</div><img src="data:image/png;base64,{comp_ours_b64}"></div>
        <div class="panel"><div class="panel-title">EZ-CorridorKey &mdash; Alpha Matte</div><img src="data:image/png;base64,{alpha_ours_b64}"></div>
    </div>
</div>
<div class="info">Input mode: {input_mode_label} &nbsp;|&nbsp; img_size={img_size} &nbsp;|&nbsp; subject alpha&gt;{subject_alpha:.2f} &nbsp;|&nbsp; skin alpha&gt;{skin_alpha:.2f}</div>
<table>
    <tr><th>Comparison</th><th>Pixels</th><th>Max Diff</th><th>MAE</th><th>PSNR (dB)</th><th>Sat Delta</th><th>Lum Delta</th><th>Verdict</th></tr>
    {diff_rows_html}
</table>
<table>
    <tr><th>Comparison</th><th>SSIM</th><th>MS-SSIM</th><th>LPIPS</th><th>DeltaE00 Mean</th><th>DeltaE00 P95</th></tr>
    {perc_rows_html}
</table>
<div class="verdict-bar">v{VERSION} &nbsp;|&nbsp; Frame: {os.path.basename(frame_path)} &nbsp;|&nbsp; Resolution: {w}x{h} &nbsp;|&nbsp; {overall_text}</div>
</body></html>"""

    out_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f"{date_slug}_upstream_d2_comparison_v{VERSION}.html")
    out_path = os.path.join(out_dir, f"{date_slug}_upstream_d2_comparison_v{VERSION}.png")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nHTML report saved: {html_path}")
    print("Open in a browser to preview, or use a screenshot tool to convert to PNG.")

    # Try headless screenshot via Playwright if available
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 900})
            page.goto(f"file:///{html_path.replace(os.sep, '/')}")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=out_path, full_page=True)
            browser.close()
        print(f"PNG report saved: {out_path}")
    except Exception as exc:
        print(f"Playwright screenshot skipped ({exc}). Use the HTML file directly.")
        out_path = html_path

    return out_path


def print_metric_rows(rows: list[MetricRow]) -> None:
    print("=" * 88)
    print(
        f"{'Comparison':<28} {'Pixels':>10} {'Max Diff':>14} {'MAE':>14} "
        f"{'PSNR':>10} {'Sat D':>12} {'Lum D':>12}"
    )
    print("=" * 88)
    for row in rows:
        psnr_str = "inf" if row.psnr == float("inf") else f"{row.psnr:.1f}"
        sat_str = "-" if row.sat_delta is None else f"{row.sat_delta:+.6f}"
        lum_str = "-" if row.lum_delta is None else f"{row.lum_delta:+.6f}"
        print(
            f"{row.name:<28} {row.pixel_count:>10,} {row.max_diff:>14.10f} "
            f"{row.mae:>14.10f} {psnr_str:>10} {sat_str:>12} {lum_str:>12}"
        )
        print(f"  Verdict: {row.verdict}")
        print(
            "  Supplemental: "
            f"SSIM={('-' if row.ssim is None else f'{row.ssim:.6f}')}  "
            f"MS-SSIM={('-' if row.ms_ssim is None else f'{row.ms_ssim:.6f}')}  "
            f"LPIPS={('-' if row.lpips is None else f'{row.lpips:.6f}')}  "
            f"DeltaE00(mean/p95)={('-' if row.deltae_mean is None else f'{row.deltae_mean:.6f}')}/"
            f"{('-' if row.deltae_p95 is None else f'{row.deltae_p95:.6f}')}"
        )
        if row.note:
            print(f"  Note:    {row.note}")
    print()


def build_rows(
    res_up: dict[str, np.ndarray],
    res_ours: dict[str, np.ndarray],
    subject_alpha: float,
    skin_alpha: float,
) -> list[MetricRow]:
    rows: list[MetricRow] = []

    alpha_row = compute_metrics("Alpha Matte", res_up["alpha"], res_ours["alpha"])
    _attach_image_metrics(
        alpha_row,
        alpha_to_rgb(res_up["alpha"]),
        alpha_to_rgb(res_ours["alpha"]),
        allow_lpips=False,
    )
    _apply_supplemental_verdict(alpha_row)
    rows.append(alpha_row)

    fg_row = compute_metrics("Foreground RGB", res_up["fg"], res_ours["fg"])
    _attach_image_metrics(fg_row, res_up["fg"], res_ours["fg"])
    _apply_supplemental_verdict(fg_row)
    rows.append(fg_row)

    comp_row = compute_metrics("Composite RGB", res_up["comp"], res_ours["comp"])
    _attach_image_metrics(comp_row, res_up["comp"], res_ours["comp"])
    _apply_supplemental_verdict(comp_row)
    rows.append(comp_row)

    subject_mask = compute_subject_mask(res_up["alpha"], res_ours["alpha"], subject_alpha)
    subject_row = compute_metrics(
        "Composite RGB (Subject)", res_up["comp"], res_ours["comp"], mask=subject_mask
    )
    attach_color_drift(subject_row, res_up["comp"], res_ours["comp"], subject_mask)
    _attach_subject_patch_metrics(subject_row, res_up["comp"], res_ours["comp"], subject_mask)
    subject_row.note = f"Mask = max(alpha_up, alpha_ours) > {subject_alpha:.2f}"
    _apply_supplemental_verdict(subject_row)
    rows.append(subject_row)

    alpha_mask = compute_subject_mask(res_up["alpha"], res_ours["alpha"], skin_alpha)
    skin_mask = detect_skin_mask(res_up["comp"], res_ours["comp"], subject_mask, alpha_mask)
    if skin_mask is not None:
        skin_row = compute_metrics(
            "Composite RGB (Skin-Like)", res_up["comp"], res_ours["comp"], mask=skin_mask
        )
        attach_color_drift(skin_row, res_up["comp"], res_ours["comp"], skin_mask)
        skin_row.deltae_mean, skin_row.deltae_p95 = _compute_deltae(
            res_up["comp"], res_ours["comp"], skin_mask
        )
        skin_row.note = (
            f"Mask = skin-like YCrCb thresholds on mean composite and alpha > {skin_alpha:.2f}"
        )
        _apply_supplemental_verdict(skin_row)
        rows.append(skin_row)
    else:
        rows.append(
            MetricRow(
                name="Composite RGB (Skin-Like)",
                pixel_count=0,
                max_diff=0.0,
                mae=0.0,
                psnr=float("inf"),
                verdict="SKIPPED",
                note=f"Fewer than {MIN_SKIN_PIXELS} robust skin-like pixels detected",
            )
        )

    return rows


def print_backend_status() -> None:
    print("Supplemental metric backends:")
    for line in _backend_status():
        print(f"  {line}")
    if _LPIPS_ERROR:
        print(f"  LPIPS init error: {_LPIPS_ERROR}")
    print()


def main() -> None:
    args = parse_args()
    checkpoint = find_checkpoint(args.checkpoint)

    if not os.path.isdir(UPSTREAM_ROOT):
        print(f"Upstream repo not found at: {UPSTREAM_ROOT}")
        print("Clone it first: git clone https://github.com/nikopueringer/CorridorKey.git")
        sys.exit(1)

    frame_path = resolve_path(args.frame)
    mask_path = resolve_path(args.mask)
    if frame_path is None or mask_path is None:
        auto_frame, auto_mask = find_test_frame()
        frame_path = frame_path or auto_frame
        mask_path = mask_path or auto_mask
    if not frame_path or not mask_path:
        print("No test frame with alpha hint found in Projects/")
        sys.exit(1)

    input_is_linear, input_mode_label = input_is_linear_for(frame_path, args.input_color_space)

    print(f"Checkpoint:   {checkpoint}")
    print(f"Source frame: {frame_path}")
    print(f"Alpha mask:   {mask_path}")
    print(f"Input mode:   {input_mode_label}")
    print(f"img_size:     {args.img_size}")
    print()
    print_backend_status()

    frame = load_frame(frame_path)
    mask = load_mask(mask_path)
    h, w = frame.shape[:2]
    print(f"Resolution:   {w}x{h}")
    print()

    print("=" * 60)
    print("UPSTREAM (nikopueringer/CorridorKey)")
    print("=" * 60)
    res_up = run_engine(
        UPSTREAM_ROOT,
        frame,
        mask,
        checkpoint,
        img_size=args.img_size,
        input_is_linear=input_is_linear,
    )
    print("Completed.")
    print()

    print("=" * 60)
    print("OURS (EZ-CorridorKey optimized)")
    print("=" * 60)
    res_ours = run_engine(
        PROJECT_ROOT,
        frame,
        mask,
        checkpoint,
        img_size=args.img_size,
        input_is_linear=input_is_linear,
    )
    print("Completed.")
    print()

    rows = build_rows(
        res_up=res_up,
        res_ours=res_ours,
        subject_alpha=args.subject_alpha,
        skin_alpha=args.skin_alpha,
    )
    print_metric_rows(rows)

    generate_report(
        frame_path=frame_path,
        source=frame,
        mask=mask,
        res_up=res_up,
        res_ours=res_ours,
        rows=rows,
        input_mode_label=input_mode_label,
        img_size=args.img_size,
        subject_alpha=args.subject_alpha,
        skin_alpha=args.skin_alpha,
    )


if __name__ == "__main__":
    main()

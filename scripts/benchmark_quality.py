"""Process the same frame through 4 optimization levels and compare quality.

Levels:
  0. Baseline (no patches, no TF32)
  1. Hiera FlashAttention patch only
  2. Hiera + TF32
  3. Hiera + TF32 + torch.compile

Outputs per-pixel difference metrics between each level and baseline.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_exr_frame(path: str):
    """Load EXR as float32 [H, W, C]."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read: {path}")
    # OpenCV loads BGR, convert to RGB, take first 3 channels
    if img.ndim == 3 and img.shape[2] >= 3:
        img = img[:, :, :3][:, :, ::-1].copy()  # BGR -> RGB
    return img.astype(np.float32)


def create_engine(checkpoint, device='cuda', img_size=2048):
    """Create a fresh CorridorKeyEngine without any patches."""
    from CorridorKeyModule.core.model_transformer import GreenFormer
    import math

    model = GreenFormer(img_size=img_size, use_refiner=True)
    model = model.to(device)
    model.eval()

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = ckpt.get('state_dict', ckpt)
    model_state = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            k = k[10:]
        if 'pos_embed' in k and k in model_state:
            if v.shape != model_state[k].shape:
                N_src, N_dst = v.shape[1], model_state[k].shape[1]
                C = v.shape[2]
                g_src, g_dst = int(math.sqrt(N_src)), int(math.sqrt(N_dst))
                v_img = v.permute(0, 2, 1).view(1, C, g_src, g_src)
                v = F.interpolate(v_img, size=(g_dst, g_dst), mode='bicubic',
                                  align_corners=False).flatten(2).transpose(1, 2)
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model


def patch_hiera(model):
    """Apply Hiera FlashAttention patch."""
    hiera = model.encoder.model
    patched = 0
    for blk in hiera.blocks:
        attn = blk.attn
        if attn.use_mask_unit_attn:
            continue

        def _make_patched_forward(original_attn):
            def _patched_forward(self, x: torch.Tensor) -> torch.Tensor:
                B, N, _ = x.shape
                qkv = self.qkv(x)
                qkv = qkv.reshape(B, N, 3, self.heads, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                if self.q_stride > 1:
                    q = q.view(B, self.heads, self.q_stride, -1, self.head_dim).amax(dim=2)
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()
                x = F.scaled_dot_product_attention(q, k, v)
                x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
                x = self.proj(x)
                return x
            return types.MethodType(_patched_forward, original_attn)

        attn.forward = _make_patched_forward(attn)
        patched += 1
    return patched


@torch.no_grad()
def run_inference(model, image: np.ndarray, img_size=2048):
    """Run model forward pass, return alpha as float32 numpy [H, W]."""
    h, w = image.shape[:2]

    # Create dummy green screen mask (all 1.0)
    mask = np.ones((h, w), dtype=np.float32)

    img_resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mask_resized = mask_resized[:, :, np.newaxis]

    img_t = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).cuda()
    mask_t = torch.from_numpy(mask_resized).permute(2, 0, 1).unsqueeze(0).cuda()
    inp = torch.cat([img_t, mask_t], dim=1)

    out = model(inp)
    pred_alpha = out['alpha']
    res = pred_alpha[0].permute(1, 2, 0).float().cpu().numpy()
    res = cv2.resize(res, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return res


def compare(name_a, alpha_a, name_b, alpha_b):
    """Compare two alpha outputs."""
    diff = np.abs(alpha_a - alpha_b)
    max_diff = float(diff.max())
    mae = float(diff.mean())
    mse = float(np.mean((alpha_a - alpha_b) ** 2))
    if mse == 0:
        psnr_val = float('inf')
    else:
        psnr_val = 10 * np.log10(1.0 / mse)

    return {
        'pair': f"{name_a} vs {name_b}",
        'max_diff': max_diff,
        'mae': mae,
        'psnr': psnr_val,
    }


def main():
    ckpt = "CorridorKeyModule/CorridorKey.pth"
    if not os.path.isfile(ckpt):
        # Try finding it
        for candidate in ["checkpoints/CorridorKey.pth",
                          "CorridorKeyModule/checkpoints/CorridorKey.pth"]:
            if os.path.isfile(candidate):
                ckpt = candidate
                break

    if not os.path.isfile(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    # Find a test frame
    frame_path = None
    for root, dirs, files in os.walk("Projects"):
        for f in sorted(files):
            if f.endswith(".exr") and "frame_" in f and "Frames" in root:
                candidate = os.path.join(root, f)
                # Prefer 4K frames
                img = cv2.imread(candidate, cv2.IMREAD_UNCHANGED)
                if img is not None and img.shape[0] > 1000:
                    frame_path = candidate
                    break
        if frame_path:
            break

    if not frame_path:
        print("No test frame found in Projects/")
        sys.exit(1)

    print(f"Test frame: {frame_path}")
    image = load_exr_frame(frame_path)
    print(f"Resolution: {image.shape[1]}x{image.shape[0]}")
    print()

    results = {}

    # Level 0: Baseline
    print("=" * 60)
    print("Level 0: BASELINE (no patches)")
    print("=" * 60)
    torch.set_float32_matmul_precision('highest')  # Reset to default
    model = create_engine(ckpt)
    alpha_baseline = run_inference(model, image)
    results['baseline'] = alpha_baseline
    del model
    torch.cuda.empty_cache()
    print(f"  Alpha range: [{alpha_baseline.min():.4f}, {alpha_baseline.max():.4f}]")
    print()

    # Level 1: Hiera FlashAttention
    print("=" * 60)
    print("Level 1: HIERA FlashAttention patch")
    print("=" * 60)
    torch.set_float32_matmul_precision('highest')
    model = create_engine(ckpt)
    n = patch_hiera(model)
    print(f"  Patched {n} blocks")
    alpha_hiera = run_inference(model, image)
    results['hiera'] = alpha_hiera
    del model
    torch.cuda.empty_cache()
    print(f"  Alpha range: [{alpha_hiera.min():.4f}, {alpha_hiera.max():.4f}]")
    print()

    # Level 2: Hiera + TF32
    print("=" * 60)
    print("Level 2: HIERA + TF32")
    print("=" * 60)
    torch.set_float32_matmul_precision('high')  # Enable TF32
    model = create_engine(ckpt)
    n = patch_hiera(model)
    print(f"  Patched {n} blocks, TF32 enabled")
    alpha_tf32 = run_inference(model, image)
    results['hiera_tf32'] = alpha_tf32
    del model
    torch.cuda.empty_cache()
    print(f"  Alpha range: [{alpha_tf32.min():.4f}, {alpha_tf32.max():.4f}]")
    print()

    # Level 3: Hiera + TF32 + torch.compile
    print("=" * 60)
    print("Level 3: HIERA + TF32 + torch.compile")
    print("=" * 60)
    torch.set_float32_matmul_precision('high')
    model = create_engine(ckpt)
    n = patch_hiera(model)
    try:
        import subprocess
        if sys.platform == 'win32':
            _orig = subprocess.Popen.__init__
            def _silent(self, *a, **kw):
                kw.setdefault('creationflags', subprocess.CREATE_NO_WINDOW)
                _orig(self, *a, **kw)
            subprocess.Popen.__init__ = _silent
        model = torch.compile(model)
        print(f"  Patched {n} blocks, TF32 enabled, torch.compile active")
        # Warmup run
        print("  Warmup (compiling)...")
        _ = run_inference(model, image)
        # Real run
        alpha_compile = run_inference(model, image)
        results['hiera_tf32_compile'] = alpha_compile
        print(f"  Alpha range: [{alpha_compile.min():.4f}, {alpha_compile.max():.4f}]")
    except Exception as e:
        print(f"  torch.compile FAILED: {e}")
        print("  Skipping Level 3")
    del model
    torch.cuda.empty_cache()
    print()

    # Comparison
    print("=" * 60)
    print("QUALITY COMPARISON (all vs baseline)")
    print("=" * 60)
    print()
    print(f"{'Config':<30} {'Max Diff':>12} {'MAE':>12} {'PSNR (dB)':>12}")
    print("-" * 68)

    for name, alpha in results.items():
        if name == 'baseline':
            continue
        c = compare('baseline', results['baseline'], name, alpha)
        psnr_str = f"{c['psnr']:.1f}" if c['psnr'] != float('inf') else "inf (identical)"
        print(f"{name:<30} {c['max_diff']:>12.8f} {c['mae']:>12.8f} {psnr_str:>15}")

    print()
    print("Interpretation:")
    print("  PSNR inf     = bit-identical output")
    print("  PSNR > 80 dB = difference below float32 noise floor")
    print("  PSNR > 60 dB = effectively identical (imperceptible)")
    print("  PSNR > 40 dB = very close (sub-pixel differences)")
    print("  PSNR < 40 dB = visible differences — investigate")


if __name__ == "__main__":
    main()

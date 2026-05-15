"""
MLX implementation of the CorridorKey GreenFormer model.

Faithfully reproduces the PyTorch model (Hiera backbone + DecoderHead + CNNRefiner)
using Apple MLX for native Apple Silicon inference.

Architecture:
  ☼ PatchEmbed: Conv2d(4, 112, 7x7, stride=4, pad=3)
  ☼ pos_embed: learnable [1, N, 112]
  ☼ Unroll/Reroll: spatial permutation for Hiera mask-unit attention
  ☼ 24 HieraBlocks across 4 stages: [112, 224, 448, 896] channels
  ☼ 2x DecoderHead (alpha 1ch, fg 3ch)
  ☼ CNNRefinerModule: dilated residual refiner (7ch -> 4ch)
"""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gelu(x: mx.array) -> mx.array:
    return nn.gelu(x)


def _interpolate_2d(x: mx.array, target_h: int, target_w: int) -> mx.array:
    """Bilinear interpolation for NHWC tensors. Matches PyTorch F.interpolate(align_corners=False)."""
    B, H, W, C = x.shape
    if H == target_h and W == target_w:
        return x

    # Use nn.Upsample when target is an integer multiple
    scale_h = target_h / H
    scale_w = target_w / W
    if scale_h == scale_w and scale_h == int(scale_h):
        up = nn.Upsample(scale_factor=scale_h, mode="linear", align_corners=False)
        return up(x)

    # For non-integer scales, use nn.Upsample with tuple scale factors
    up = nn.Upsample(scale_factor=(scale_h, scale_w), mode="linear", align_corners=False)
    return up(x)


# ---------------------------------------------------------------------------
# Hiera components
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Conv2d patch embedding: image [B, H, W, 4] -> tokens [B, N, C]."""

    def __init__(self, in_channels: int = 4, embed_dim: int = 112,
                 kernel_size: int = 7, stride: int = 4, padding: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=kernel_size, stride=stride, padding=padding)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, H, W, C_in] (MLX uses NHWC)
        x = self.proj(x)           # [B, H', W', C]
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)  # [B, N, C]
        return x


class Mlp(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        return x


class MaskUnitAttention(nn.Module):
    """Hiera's MaskUnitAttention ported to MLX.

    Supports windowed attention (use_mask_unit_attn=True) and global attention.
    Supports q_stride > 1 for spatial downsampling via max-pooling Q.
    """

    def __init__(self, dim: int, dim_out: int, heads: int,
                 q_stride: int = 1, window_size: int = 64,
                 use_mask_unit_attn: bool = True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride
        self.head_dim = dim_out // heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, _ = x.shape
        num_windows = (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1

        # QKV projection and reshape
        qkv = self.qkv(x)  # [B, N, 3*dim_out]
        # Reshape: [B, N_per_win, num_windows, 3, heads, head_dim]
        tokens_per_window = N // num_windows
        qkv = qkv.reshape(B, tokens_per_window, num_windows, 3, self.heads, self.head_dim)
        # Permute to [3, B, heads, num_windows, tokens_per_window, head_dim]
        qkv = qkv.transpose(3, 0, 4, 2, 1, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, heads, num_win, tokens, head_dim]

        if self.q_stride > 1:
            # Max-pool Q along stride dimension
            q = q.reshape(B, self.heads, num_windows, self.q_stride, -1, self.head_dim)
            q = mx.max(q, axis=3)  # [B, heads, num_win, tokens/stride, head_dim]

        # Scaled dot-product attention
        attn = (q * self.scale) @ k.transpose(0, 1, 2, 4, 3)  # [..., Nq, Nk]
        attn = mx.softmax(attn, axis=-1)
        x = attn @ v  # [B, heads, num_win, Nq, head_dim]

        # Reshape back: [B, heads, num_win, Nq, head_dim] -> [B, N_out, dim_out]
        x = x.transpose(0, 3, 2, 1, 4)  # [B, Nq, num_win, heads, head_dim]
        x = x.reshape(B, -1, self.dim_out)

        x = self.proj(x)
        return x


class HieraBlock(nn.Module):
    """Single Hiera transformer block."""

    def __init__(self, dim: int, dim_out: int, heads: int,
                 q_stride: int = 1, window_size: int = 64,
                 use_mask_unit_attn: bool = True,
                 do_expand: bool = False, mlp_ratio: float = 4.0):
        super().__init__()
        self.do_expand = do_expand
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MaskUnitAttention(
            dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn)
        self.norm2 = nn.LayerNorm(dim_out)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), dim_out)

        # Stage transition projection
        self.proj = nn.Linear(dim, dim_out) if (do_expand and q_stride > 1) else None
        self._q_stride = q_stride

    def __call__(self, x: mx.array) -> mx.array:
        x_norm = self.norm1(x)

        if self.do_expand:
            if self.proj is not None:
                x = self.proj(x_norm)
                # Max-pool: [B, N, C] -> [B, q_stride, N/q_stride, C] -> max dim=1
                B, N, C = x.shape
                x = x.reshape(B, self._q_stride, N // self._q_stride, C)
                x = mx.max(x, axis=1)
            else:
                # Concat max+avg pool (shouldn't happen for this model, all expand blocks have proj)
                B, N, C = x.shape
                xr = x.reshape(B, self._q_stride, N // self._q_stride, C)
                x = mx.concatenate([mx.max(xr, axis=1), mx.mean(xr, axis=1)], axis=-1)

        x = x + self.attn(x_norm)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Unroll / Reroll (spatial permutation for Hiera mask units)
# ---------------------------------------------------------------------------

def unroll_2d(x: mx.array, size: List[int],
              schedule: List[Tuple[int, int]]) -> mx.array:
    """Permute flattened [B, N, C] tokens for Hiera mask-unit attention.

    Implements the same logic as timm's Unroll.forward for 2D inputs.
    The final reshape merges expanded batch back using the original grid size.
    """
    B, _, C = x.shape
    orig_h, orig_w = size
    cur_h, cur_w = size
    x = x.reshape(B, cur_h, cur_w, C)

    for (sy, sx) in schedule:
        cur_h //= sy
        cur_w //= sx
        # [B, cur_h, sy, cur_w, sx, C]
        x = x.reshape(B, cur_h, sy, cur_w, sx, C)
        # [B, sy, sx, cur_h, cur_w, C]
        x = x.transpose(0, 2, 4, 1, 3, 5)
        # Flatten batch dims
        x = x.reshape(B * sy * sx, cur_h, cur_w, C)
        B = B * sy * sx

    # Merge expanded batch back: timm uses prod(self.size) = orig_h * orig_w
    x = x.reshape(-1, orig_h * orig_w, C)
    return x


def reroll_2d(x: mx.array, block_idx: int, size: List[int],
              schedule_for_block: Tuple[List[Tuple[int, int]], List[int]]) -> mx.array:
    """Inverse of unroll: restore spatial order from Hiera block output.

    Returns [B, H, W, C] in NHWC layout.
    """
    remaining_schedule, cur_size = schedule_for_block
    B, N, C = x.shape
    cur_h, cur_w = cur_size
    cur_mu_h, cur_mu_w = 1, 1

    for (sy, sx) in remaining_schedule:
        # x: [B, N, mu_h, mu_w, C] conceptually
        x = x.reshape(B, sy, sx, N // (sy * sx), cur_mu_h, cur_mu_w, C)
        # Permute: [B, N//(sy*sx), sy, mu_h, sx, mu_w, C]
        x = x.transpose(0, 3, 1, 4, 2, 5, 6)
        cur_mu_h *= sy
        cur_mu_w *= sx
        x = x.reshape(B, -1, cur_mu_h, cur_mu_w, C)
        N = x.shape[1]

    # x: [B, num_MUs, mu_h, mu_w, C]
    x = x.reshape(B, N, cur_mu_h, cur_mu_w, C)

    # Undo windowing: [B, #MUy, #MUx, mu_h, mu_w, C] -> [B, H, W, C]
    num_mu_y = cur_h // cur_mu_h
    num_mu_x = cur_w // cur_mu_w
    x = x.reshape(B, num_mu_y, num_mu_x, cur_mu_h, cur_mu_w, C)
    # [B, #MUy, mu_h, #MUx, mu_w, C]
    x = x.transpose(0, 1, 3, 2, 4, 5)
    x = x.reshape(B, cur_h, cur_w, C)
    return x


# ---------------------------------------------------------------------------
# Hiera encoder
# ---------------------------------------------------------------------------

# Block configurations for hiera_base_plus_224
# (dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn, do_expand)
HIERA_BASE_PLUS_BLOCKS = [
    # Stage 1: dim=112, 2 blocks
    (112, 112, 2, 1, 64, True, False),   # block 0
    (112, 112, 2, 1, 64, True, False),   # block 1
    # Stage 2: transition 112->224, 3 blocks
    (112, 224, 4, 4, 16, True, True),    # block 2 (expand)
    (224, 224, 4, 1, 16, True, False),   # block 3
    (224, 224, 4, 1, 16, True, False),   # block 4
    # Stage 3: transition 224->448, 16 blocks
    (224, 448, 8, 4, 4, True, True),     # block 5 (expand)
    (448, 448, 8, 1, 4, False, False),   # block 6
    (448, 448, 8, 1, 4, False, False),   # block 7
    (448, 448, 8, 1, 4, False, False),   # block 8
    (448, 448, 8, 1, 4, False, False),   # block 9
    (448, 448, 8, 1, 4, False, False),   # block 10
    (448, 448, 8, 1, 4, False, False),   # block 11
    (448, 448, 8, 1, 4, False, False),   # block 12
    (448, 448, 8, 1, 4, False, False),   # block 13
    (448, 448, 8, 1, 4, False, False),   # block 14
    (448, 448, 8, 1, 4, False, False),   # block 15
    (448, 448, 8, 1, 4, False, False),   # block 16
    (448, 448, 8, 1, 4, False, False),   # block 17
    (448, 448, 8, 1, 4, False, False),   # block 18
    (448, 448, 8, 1, 4, False, False),   # block 19
    (448, 448, 8, 1, 4, False, False),   # block 20
    # Stage 4: transition 448->896, 3 blocks
    (448, 896, 16, 4, 1, False, True),   # block 21 (expand)
    (896, 896, 16, 1, 1, False, False),  # block 22
    (896, 896, 16, 1, 1, False, False),  # block 23
]

# Stage ends (indices where features are extracted)
STAGE_ENDS = [1, 4, 20, 23]

# Unroll schedule: 3 rounds of (2,2) spatial folding
UNROLL_SCHEDULE = [(2, 2), (2, 2), (2, 2)]

# Reroll schedule per block (remaining unrolls to undo + spatial size at that point)
# Precomputed for the 4 stage-end blocks we actually use
def _build_reroll_schedule(img_size: int = 512):
    """Build reroll schedule for each block index."""
    patch_stride = 4
    grid = img_size // patch_stride  # 128 for 512, 512 for 2048

    # After each expand (q_stride=4 means 2x2 spatial pooling), grid halves
    # Stage 1 (blocks 0-1): grid
    # Stage 2 (blocks 2-4): grid/2
    # Stage 3 (blocks 5-20): grid/4
    # Stage 4 (blocks 21-23): grid/8

    schedules = {}
    # After unroll with schedule [(2,2),(2,2),(2,2)], the reroll needs to undo:
    # Block 1 (stage 1): undo all 3 -> size=[grid, grid]
    schedules[1] = ([(2, 2), (2, 2), (2, 2)], [grid, grid])
    # Block 4 (stage 2): after first expand, 2 remaining -> size=[grid//2, grid//2]
    schedules[4] = ([(2, 2), (2, 2)], [grid // 2, grid // 2])
    # Block 20 (stage 3): after second expand, 1 remaining -> size=[grid//4, grid//4]
    schedules[20] = ([(2, 2)], [grid // 4, grid // 4])
    # Block 23 (stage 4): after third expand, 0 remaining -> size=[grid//8, grid//8]
    schedules[23] = ([], [grid // 8, grid // 8])
    return schedules


class HieraEncoder(nn.Module):
    """Hiera backbone encoder for CorridorKey."""

    def __init__(self, img_size: int = 512, in_channels: int = 4):
        super().__init__()
        self.img_size = img_size
        patch_stride = 4
        self.grid_size = img_size // patch_stride

        self.patch_embed = PatchEmbed(in_channels, 112, 7, 4, 3)
        self.pos_embed = mx.zeros((1, self.grid_size * self.grid_size, 112))

        self.blocks = [
            HieraBlock(dim, dim_out, heads, q_stride, window_size, use_mu_attn, do_expand)
            for (dim, dim_out, heads, q_stride, window_size, use_mu_attn, do_expand)
            in HIERA_BASE_PLUS_BLOCKS
        ]

        self.reroll_schedules = _build_reroll_schedule(img_size)

    def __call__(self, x: mx.array) -> List[mx.array]:
        """Forward pass. x: [B, H, W, 4] (NHWC). Returns list of 4 feature maps in NHWC."""
        # Patch embed + pos embed
        x = self.patch_embed(x)       # [B, N, 112]
        x = x + self.pos_embed        # [B, N, 112]

        # Unroll
        x = unroll_2d(x, [self.grid_size, self.grid_size], UNROLL_SCHEDULE)

        # Run blocks, extract features at stage ends
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in STAGE_ENDS:
                feat = reroll_2d(x, i, [self.grid_size, self.grid_size],
                                self.reroll_schedules[i])
                features.append(feat)  # [B, H_i, W_i, C_i]

        return features  # 4 features in NHWC


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DecoderHead(nn.Module):
    """SegFormer-style decoder: unify multi-scale features, fuse, classify."""

    def __init__(self, feature_channels: List[int] = [112, 224, 448, 896],
                 embedding_dim: int = 256, output_dim: int = 1):
        super().__init__()
        self.linear_c1 = nn.Linear(feature_channels[0], embedding_dim)
        self.linear_c2 = nn.Linear(feature_channels[1], embedding_dim)
        self.linear_c3 = nn.Linear(feature_channels[2], embedding_dim)
        self.linear_c4 = nn.Linear(feature_channels[3], embedding_dim)

        # Conv2d fuse (1x1)
        self.linear_fuse = nn.Conv2d(embedding_dim * 4, embedding_dim,
                                     kernel_size=1, bias=False)
        self.bn = nn.BatchNorm(embedding_dim)
        self.classifier = nn.Conv2d(embedding_dim, output_dim, kernel_size=1)

    def __call__(self, features: List[mx.array]) -> mx.array:
        """features: list of 4 NHWC arrays. Returns [B, H1, W1, output_dim]."""
        c1, c2, c3, c4 = features
        B = c1.shape[0]
        target_h, target_w = c1.shape[1], c1.shape[2]

        # Project each to embedding_dim and upsample to c1 spatial size
        _c4 = self.linear_c4(c4)  # [B, H4, W4, embed]
        _c4 = _interpolate_2d(_c4, target_h, target_w)

        _c3 = self.linear_c3(c3)
        _c3 = _interpolate_2d(_c3, target_h, target_w)

        _c2 = self.linear_c2(c2)
        _c2 = _interpolate_2d(_c2, target_h, target_w)

        _c1 = self.linear_c1(c1)  # Already at target size

        # Concatenate: [B, H, W, 4*embed]
        cat = mx.concatenate([_c4, _c3, _c2, _c1], axis=-1)

        # Fuse (1x1 conv)
        x = self.linear_fuse(cat)  # [B, H, W, embed]
        x = self.bn(x)
        x = nn.relu(x)

        # Classify
        x = self.classifier(x)  # [B, H, W, output_dim]
        return x


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------

class RefinerBlock(nn.Module):
    """Residual block with dilation and GroupNorm."""

    def __init__(self, channels: int = 64, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=dilation, dilation=dilation, bias=True)
        self.gn1 = nn.GroupNorm(8, channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=dilation, dilation=dilation, bias=True)
        self.gn2 = nn.GroupNorm(8, channels, pytorch_compatible=True)
        self._dilation = dilation

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = nn.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = out + residual
        out = nn.relu(out)
        return out


class CNNRefiner(nn.Module):
    """Dilated residual refiner. Input: [B, H, W, 7], output: [B, H, W, 4]."""

    def __init__(self, in_channels: int = 7, hidden_channels: int = 64,
                 out_channels: int = 4):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.stem_gn = nn.GroupNorm(8, hidden_channels, pytorch_compatible=True)
        self.res1 = RefinerBlock(hidden_channels, dilation=1)
        self.res2 = RefinerBlock(hidden_channels, dilation=2)
        self.res3 = RefinerBlock(hidden_channels, dilation=4)
        self.res4 = RefinerBlock(hidden_channels, dilation=8)
        self.final = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.stem_conv(x)
        x = self.stem_gn(x)
        x = nn.relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return self.final(x) * 10.0


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class GreenFormerMLX(nn.Module):
    """Complete CorridorKey model in MLX.

    Input: [B, H, W, 4] (NHWC, channels = RGB + alpha hint)
    Output: dict with 'alpha' [B, H, W, 1] and 'fg' [B, H, W, 3]
    """

    def __init__(self, img_size: int = 2048, use_refiner: bool = True):
        super().__init__()
        self.img_size = img_size
        self.encoder = HieraEncoder(img_size=img_size, in_channels=4)
        self.alpha_decoder = DecoderHead([112, 224, 448, 896], 256, output_dim=1)
        self.fg_decoder = DecoderHead([112, 224, 448, 896], 256, output_dim=3)
        self.use_refiner = use_refiner
        if use_refiner:
            self.refiner = CNNRefiner(in_channels=7, hidden_channels=64, out_channels=4)
        else:
            self.refiner = None

    def __call__(self, x: mx.array, refiner_scale: Optional[float] = None) -> dict:
        """Forward pass. x: [B, H, W, 4] float32 NHWC."""
        input_h, input_w = x.shape[1], x.shape[2]

        # Encode
        features = self.encoder(x)

        # Decode
        alpha_logits = self.alpha_decoder(features)  # [B, H/4, W/4, 1]
        fg_logits = self.fg_decoder(features)        # [B, H/4, W/4, 3]

        # Upsample to full resolution
        alpha_logits_up = _interpolate_2d(alpha_logits, input_h, input_w)
        fg_logits_up = _interpolate_2d(fg_logits, input_h, input_w)

        # Coarse predictions
        alpha_coarse = mx.sigmoid(alpha_logits_up)
        fg_coarse = mx.sigmoid(fg_logits_up)

        # Refinement
        rgb = x[:, :, :, :3]
        coarse_pred = mx.concatenate([alpha_coarse, fg_coarse], axis=-1)  # [B, H, W, 4]

        if self.use_refiner and self.refiner is not None:
            refiner_input = mx.concatenate([rgb, coarse_pred], axis=-1)   # [B, H, W, 7]
            delta_logits = self.refiner(refiner_input)                     # [B, H, W, 4]
        else:
            delta_logits = mx.zeros_like(coarse_pred)

        delta_alpha = delta_logits[:, :, :, 0:1]
        delta_fg = delta_logits[:, :, :, 1:4]

        if refiner_scale is not None:
            delta_alpha = delta_alpha * refiner_scale

        alpha_final = mx.sigmoid(alpha_logits_up + delta_alpha)
        fg_final = mx.sigmoid(fg_logits_up + delta_fg)

        return {'alpha': alpha_final, 'fg': fg_final}

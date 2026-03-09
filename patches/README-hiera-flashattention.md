# Hiera FlashAttention Patch (Experimental)

Fixes a performance issue in timm's Hiera backbone where global attention blocks
create 5D non-contiguous tensors, causing PyTorch's SDPA to silently fall back
to the slow O(N^2) math backend instead of FlashAttention.

**Result:** ~30% faster inference on 4K footage (3.3s -> 2.27s per frame on RTX 5090).

Credit: Jhe Kimchi (Discord)

---

## How to Apply

### Option A: Git Patch (if you cloned the repo)

```bash
cd CorridorKey
git apply patches/hiera-flashattention-v1.patch
```

To revert:
```bash
git checkout -- CorridorKeyModule/inference_engine.py
```

### Option B: Manual Edit (any installation)

Open `CorridorKeyModule/inference_engine.py` and make two changes:

**1. Add imports at the top** (after `import time`):

```python
import types
```

And after `import torch`:

```python
import torch.nn as nn
```

**2. Add the patch function** (before the `class CorridorKeyEngine:` line):

```python
def _patch_hiera_global_attention(hiera_model: nn.Module) -> int:
    """Monkey-patch global attention blocks for FlashAttention."""
    patched = 0
    for blk in hiera_model.blocks:
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
```

**3. Hook it into model loading.** Find the `_load_model` method, and right before `return model`, add:

```python
        # Hiera FlashAttention patch
        try:
            hiera = model.encoder.model
            n = _patch_hiera_global_attention(hiera)
            print(f"Hiera patch: {n} global blocks patched for FlashAttention")
        except Exception as e:
            print(f"Hiera patch failed (non-fatal): {e}")
```

---

## Verification

After applying, check the console/log output when the model loads. You should see:

```
Hiera patch: 18 global blocks patched for FlashAttention
```

If you see `0 blocks patched` or the patch failed message, something went wrong.

## Quality Testing

To verify no quality regression:

1. Process the same frame/clip **with and without** the patch
2. Compare alpha mattes — they should be **pixel-identical**
   (the patch only changes tensor memory layout, not math)
3. In Nuke: load both alphas, use a Merge (difference) node
   - Result should be solid black (zero difference)
4. If testing custom alphas from another app, compare the keyed output
   before/after on fine detail (hair, motion blur, transparent edges)

## Requirements

- PyTorch >= 2.0 (needs `scaled_dot_product_attention`)
- CUDA GPU (FlashAttention requires CUDA)
- No additional packages needed

## Reverting

If anything looks off, just undo the changes to `inference_engine.py`.
Git users: `git checkout -- CorridorKeyModule/inference_engine.py`

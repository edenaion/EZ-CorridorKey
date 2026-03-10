# Local GPU QA

This is the repeatable manual QA harness for a real CUDA workstation.

It is designed to answer:

- Does the installed SAM2 tracker actually run on this machine?
- Does the installed VideoMaMa path actually run on this machine?
- Do they produce the expected number of output frames?
- Did performance or VRAM behavior regress in an obvious way?

It is **not** a final quality benchmark for real production footage.

## Command

From the repo root:

```bash
.venv\Scripts\python.exe scripts\local_gpu_qa.py --keep-dir .tmp\gpu-qa
```

On macOS/Linux:

```bash
.venv/bin/python scripts/local_gpu_qa.py --keep-dir .tmp/gpu-qa
```

Optional stricter run:

```bash
.venv\Scripts\python.exe scripts\local_gpu_qa.py ^
  --frames 8 ^
  --chunk-size 8 ^
  --sam2-model large ^
  --min-sam2-iou 0.75 ^
  --min-videomama-iou 0.95 ^
  --json-out logs\gpu-qa-summary.json ^
  --keep-dir .tmp\gpu-qa
```

Available SAM2 checkpoints:

- `--sam2-model small`
- `--sam2-model base-plus` (default)
- `--sam2-model large`

## What The Script Does

It creates a tiny synthetic green-screen clip in a temp directory:

- `Frames/frame_00000.png ...`
- `GroundTruthMask/frame_00000.png ...`
- `annotations.json`

The clip is a simple moving humanoid shape on a green background. Because the
script generated the clip, it also knows the exact ground-truth mask for every
frame.

Then it runs the real backend service path:

1. `CorridorKeyService.detect_device()`
2. `run_sam2_track()`
3. reads `VideoMamaMaskHint/*.png`
4. computes SAM2 IoU vs the known ground-truth masks
5. `run_videomama()`
6. reads `AlphaHint/*.png`
7. computes VideoMaMa IoU vs the known ground-truth masks

## Pass Criteria

### SAM2

This is the strict stage.

The script fails if:

- CUDA is not available
- `VideoMamaMaskHint/` is not created
- the mask frame count does not match the input frame count
- SAM2 mean IoU falls below the configured threshold

Default threshold:

- `SAM2 mean IoU >= 0.75`

### VideoMaMa

This is now a real regression gate on the synthetic fixture.

The script fails if:

- `AlphaHint/` is not created
- the alpha frame count does not match the input frame count
- VideoMaMa mean IoU falls below the configured threshold

Default threshold:

- `VideoMaMa mean IoU >= 0.90`

## Output

The script prints:

- temp/output directory
- device
- stage timings
- VRAM snapshots from `CorridorKeyService.get_vram_info()`
- SAM2 mean/min/max IoU
- VideoMaMa mean/min/max IoU
- non-zero coverage summaries for masks and alpha

VRAM snapshots may be empty on some setups if PyTorch cannot query the CUDA
allocator cleanly from that environment. That does not invalidate the run by
itself.

If `--json-out` is provided, the same summary is written to disk as JSON.

## What To Look At

If you pass `--keep-dir`, inspect:

- `Frames/`
- `GroundTruthMask/`
- `VideoMamaMaskHint/`
- `AlphaHint/`
- `annotations.json`

The most useful visual checks are:

- `VideoMamaMaskHint/` should be dense full-body masks, not sparse dots
- `AlphaHint/` should exist for every frame and roughly follow the subject

## What This Does Not Prove

It does **not** prove:

- real-world hair quality
- hard production-footage robustness
- GVM quality
- EXR extraction/import behavior

This harness is for repeatable **local GPU pipeline QA**, not final quality signoff.

# CorridorKeyModule

A self-contained, high-performance AI Chroma Keying engine. This module provides a simple API to access the `CorridorKey` architecture (Hiera Backbone + CNN Refiner) for processing green screen footage.

## Features
*   **Resolution Independent:** Automatically resizes input images to match the native training resolution of the model (2048x2048).
*   **High Fidelity:** Preserves original input resolution using Lanczos4 resampling for final output.
*   **Robust:** Supports explicit configurations for Linear (EXR) and sRGB (PNG/MP4) source inputs.
*   **Cross-Platform GPU Support (added by toowyred):** Native CUDA for NVIDIA, DirectML ORT split-engine for AMD (RX 6000+) and Intel Arc.

## Installation

Dependencies for the engine are managed in the main project root `requirements.txt`.  
*(Requires PyTorch, NumPy, OpenCV, Timm)*

**For AMD / non-NVIDIA GPUs (Windows)**, install the DirectML runtime instead of standard CUDA packages:
```bash
# Run the included helper or install manually:
pip install onnxruntime-directml onnx
# Do NOT have both onnxruntime and onnxruntime-directml installed — they conflict.
pip uninstall onnxruntime -y
```

Then run the one-time ONNX export (from the project root):
```bash
python export_to_onnx.py
```
This generates two files in `CorridorKeyModule/checkpoints/`:
- `CorridorKey_backbone.onnx` — Hiera encoder + decoders (runs on CPU)
- `CorridorKey_refiner.onnx` — CNN refiner (runs on AMD GPU via DirectML)

> **Why two files?** Hiera's windowed attention uses high-rank tensors that exceed
> DirectML's 8-dimension limit. The backbone runs on CPU via ORT, while the CNN
> refiner (pure 2D convolutions) runs on the GPU. The split is invisible to calling
> code — the interface is identical to `CorridorKeyEngine`.

---

## Usage (GUI Wizard)

For most users, the easiest way to interact with the module is through the included wizard:
`clip_manager.py` (or dragging and dropping folders onto the `.bat` / `.sh` scripts).
The wizard handles finding the latest `.pth` checkpoint automatically, prompting for configuration (gamma, despill strength, despeckling), and batch processing entire sequences.

---

## Usage (Python API)

### NVIDIA — Standard Engine

```python
from CorridorKeyModule import CorridorKeyEngine

engine = CorridorKeyEngine(
    checkpoint_path="models/latest_model.pth",
    device='cuda',
    img_size=2048
)
```

### AMD / Intel Arc — ORT Split Engine

Requires the ONNX export step above to be completed first.

```python
from CorridorKeyModule.inference_engine_amd import CorridorKeyEngineAMD

engine = CorridorKeyEngineAMD(
    backbone_onnx="CorridorKeyModule/checkpoints/CorridorKey_backbone.onnx",
    refiner_onnx="CorridorKeyModule/checkpoints/CorridorKey_refiner.onnx",
    img_size=1024,  # Must match the size used during export_to_onnx.py
)
```

Both engines share an **identical `process_frame` interface** — all downstream code works unchanged regardless of which engine is loaded.

---

### Processing a Frame

The engine expects inputs as Numpy Arrays (`H, W, Channels`).
*   It natively processes in **32-bit float** (`0.0 - 1.0`).
*   If you pass an **8-bit integer** (`0 - 255`) array, the engine will automatically normalize it to `0.0 - 1.0` floats for you.
*   If you pass a **16-bit or 32-bit float** array (like an EXR), it will process it at full precision without downgrading.

```python
import cv2
import os

# Enable EXR Support in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Load Image (Linear EXR - Read as 32-bit Float)
img_linear = cv2.imread("input.exr", cv2.IMREAD_UNCHANGED)
img_linear_rgb = cv2.cvtColor(img_linear, cv2.COLOR_BGR2RGB)

# Load Coarse Mask (Linear EXR - Read as 32-bit Float)
mask = cv2.imread("mask.exr", cv2.IMREAD_UNCHANGED)
if mask.ndim == 3:
    mask = mask[:,:,0]  # Keep single channel

# Process
result = engine.process_frame(
    img_linear_rgb,
    mask,
    input_is_linear=True,  # Critical: Tell the engine this is a Linear EXR
)

# Save Results (Preserving Float Precision as EXR)
# 'processed' contains the final RGBA composite (Linear 0-1 float)
proc_rgba = result['processed']
proc_bgra = cv2.cvtColor(proc_rgba, cv2.COLOR_RGBA2BGRA)

exr_flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
             cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PXR24]
cv2.imwrite("output_processed.exr", proc_bgra, exr_flags)
```

---

## Module Structure

| File | Description |
|---|---|
| `inference_engine.py` | Main engine for NVIDIA CUDA. Handles normalization, tensor ops, tiled refiner, torch.compile, and full post-processing pipeline. |
| `inference_engine_amd.py` | Drop-in AMD/DirectML engine. Identical interface. Uses ORT split sessions to work around DirectML tensor rank limits on Windows. |
| `core/model_transformer.py` | Architecture definition — Hiera backbone + CNN refiner head. |
| `core/color_utils.py` | Compositing math: despill, straight/premul compositing, sRGB↔linear conversions, matte cleaning. |

**Project root helpers (AMD only - added by toowyred):**

| File | Description |
|---|---|
| `export_to_onnx.py` | One-time export script — converts `.pth` checkpoint to two `.onnx` files for AMD. |
| `1-install-amd.bat` | Installs AMD dependencies into the local `.venv`. |
| `2-start.bat` | Launches the app with DirectML flags set. |
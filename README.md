# EZ-CorridorKey **[v1.6.0](https://github.com/edenaion/EZ-CorridorKey/releases/tag/v1.6.0)**

> **Latest:** Parallel inference (1.8-2x speedup), MatAnyone2, MLX Apple Silicon acceleration, macOS support — [full changelog](CHANGELOG.md)

A full desktop GUI for [Niko Pueringer's CorridorKey](https://github.com/nikopueringer/CorridorKey) — the AI green screen keyer by Corridor Digital that physically unmixes foreground from background, preserving hair, motion blur, and translucency.

This GUI replaces the CLI drag-and-drop workflow with a complete desktop application while preserving 100% backward compatibility (`python main.py --cli` still runs the original wizard).

![EZ-CorridorKey](dev-docs/guides/screenshots/EZ-CorridorKey_Inference_Done.jpg)

| Capability | CLI (Upstream) | GUI (This Project) |
|------------|---------------|-------------------|
| Import clips | Drag onto .bat file | Drag-drop into app, or File > Import |
| Configure inference | Terminal prompts | Sliders, dropdowns, checkboxes |
| Monitor progress | Terminal text output | Progress bars, frame counter, ETA |
| Preview results | Open output folder manually | Real-time dual viewer (input vs output) |
| Job management | One clip at a time | Queue with batch processing + full pipeline mode |
| GPU monitoring | None | Live VRAM meter in brand bar |
| Keyboard shortcuts | None | 20+ hotkeys |
| Sound feedback | None | 7 context-aware sound effects |
| Session persistence | None | Recent projects, auto-save |
| Paint / masking | Manual external tool | Built-in brush tool for VideoMaMa / MatAnyone2 masks |
| Alpha generators | None | GVM, VideoMaMa, MatAnyone2 (one-click) |
| Apple Silicon | MPS only | MLX acceleration (auto-detected) |

---

## Installation

**One-Click Install (Windows / macOS / Linux):**
1. Clone or download this repository.
2. The one-click path provisions and uses managed Python 3.11 automatically, so you do not need to pre-install Python just to use `1-install`.
3. Run the installer for your platform:
   - **Windows:** Double-click `1-install.bat`
   - **macOS / Linux:** `chmod +x 1-install.sh && ./1-install.sh`
4. The installer handles everything: managed Python, virtual environment, dependencies (including the correct PyTorch backend for your GPU when available), verification, and model downloads.
5. To launch: double-click `2-start.bat` (Windows) or `./2-start.sh` (macOS/Linux).

**Prerequisites:**
- For the one-click installer: no preinstalled Python required
- For manual installs: [Python 3.10–3.13](https://python.org) (3.14 is not yet supported)
- **Windows/Linux:** NVIDIA GPU with CUDA support (8 GB+ VRAM recommended). Keep your driver current — the installer verifies the torch runtime and will stop with diagnostics instead of silently leaving you on the wrong backend.
- **macOS:** Apple Silicon (M1+). CorridorKey inference runs natively via MLX (1.5–2x faster than MPS). GPU-intensive alpha generators (SAM2, GVM, VideoMaMa, MatAnyone2) run on MPS but are significantly slower — importing pre-made alpha mattes is recommended on Mac.

**What the installer does:**
- Checks for [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (C++ compiler needed by OpenEXR) — offers to auto-install if missing
- Installs [uv](https://docs.astral.sh/uv/) and provisions managed Python 3.11 for the installer path
- Creates a `.venv` virtual environment in the project folder
- Installs the correct PyTorch backend for your platform/GPU and verifies the resulting torch runtime before reporting success
- Downloads and installs [FFmpeg](https://ffmpeg.org/) locally if not found on PATH (used for video import)
- Downloads the CorridorKey model checkpoint (383 MB, required)
- Optionally installs SAM2 tracking support and pre-downloads the default Base+ checkpoint (324 MB)
- Optionally downloads GVM (~6 GB) and VideoMaMa (~37 GB) alpha hint generators
- Creates a desktop shortcut (optional)

**Updating:**
- **Windows:** Double-click `3-update.bat`
- **macOS / Linux:** `./3-update.sh`

---

## Application Layout

```
+--------------------------------------------------------------------+
| File | Edit | View | Help                             Volume ---->-|
| CORRIDORKEY                                        RTX 5090  ## 4GB|
+------+------------------------+---------------------+--------------+
|      |      241 frames - RAW | IN|FG|MATTE[COMP]PROC| ALPHA GEN    |
|  Q   |                       |                      |  GVM AUTO    |
|  U   +-----------------------+----------------------+  MATANYONE2  |
|      |                       |                      |  VIDEOMAMA   |
|  E   |                       |                      |  EXPORT MASK |
|  U   |                       |                      |--------------+
|  E   |   INPUT viewer        |   OUTPUT viewer      | INFERENCE    |
|      |   (left image)        |   (right image)      |  Color Space |
|      |                       |                      |  Despill     |
| tab  |                       |                      |  Despeckle   |
|      |                       |                      |  Refiner     |
|      |                       |                      |  Live Preview|
|      |                       |                      |--------------+
|      |                       |                      | OUTPUT       |
|      |                       |                      |  FG          |
|      |                       |                      |  Matte       |
|      |                       |                      |  Comp        |
|      |                    SCRUBBER                  |  Processed   |
|------+<<-<-------In=====================Out----->->>+--------------|
|INPUT (3)               + ADD | EXPORTS (2)                         |
| +-----+ +-----+ +-----+      | +-----+ +-----+                     |
| |thumb| |thumb| |thumb|      | |thumb| |thumb| ..                  |
| +-----+ +-----+ +-----+      | +-----+ +-----+                     |
+--------------------------------------------------------------------+
|  Status Bar                                         [RUN INFERENCE]|
+--------------------------------------------------------------------+
```

- **Brand bar + menu** — top row with logo, menu bar, GPU name + VRAM meter
- **Queue panel** — collapsible sidebar (left), toggle with **Q**
- **Dual viewer** — center, split into INPUT (left) and switchable output (right)
- **Parameter panel** — right sidebar with Alpha Generation, Inference, and Output controls
- **I/O tray** — horizontal thumbnail strip below the viewer
- **Status bar** — progress bar and RUN INFERENCE button

---

## Quick Start

### 1. Import

Drop a video file onto the welcome screen (or File > Import Clips > Import Video). The video is automatically extracted to a high-quality image sequence:

1. **FFmpeg extracts** each frame as **EXR half-float** (ZIP16 compression)
2. **Recompression pass** converts ZIP16 → **DWAB** (VFX-standard lossy compression, ~4× smaller)

This two-pass approach preserves full floating-point precision from the video decoder — no 8-bit quantization, no banding. Even 8-bit source video benefits because FFmpeg's internal YUV→RGB conversion stays in float, avoiding rounding errors that accumulate in integer pipelines. The DWAB-compressed EXR frames are typically comparable in size to PNG while retaining 16-bit half-float dynamic range.

The recompression runs in a separate process so the UI stays fully responsive during extraction.

### 2. Generate Alpha Hint

Your clip starts in **RAW** state (gray badge). You need an alpha hint before running inference.

**Option A — GVM Auto (one-click):**
Click **GVM AUTO** in the parameter panel. Works great for most green screen footage with people.

**Option B — Track Mask + MatAnyone2 / VideoMaMa:**
For difficult shots, use the paint brush as a prompt tool:
1. Press **1** to activate foreground mode (green)
2. Paint over the subject on a few key frames
3. Press **2** to switch to background mode (red)
4. Paint over background areas
5. Click **TRACK MASK** to generate a dense SAM2 mask track
6. Click **MATANYONE2** or **VIDEOMAMA** in the parameter panel

**Option C — Import Alpha (bring your own):**
If you already have alpha mattes from another tool (Rotobrush, Silhouette, etc.), click **IMPORT ALPHA** in the parameter panel and select the folder containing your images.

- Supported formats: **PNG, JPG, JPEG, TIF, TIFF, EXR**
- Images should be **grayscale** (white = foreground, black = background)
- Frame count should match your input sequence
- Non-PNG files are automatically converted to grayscale PNG on import
- Files are copied into the clip's `AlphaHint/` folder and the clip advances to **READY** state

You can re-import at any time — if the clip already has alpha hints, you'll be asked whether to overwrite them.

### 3. Run Inference

Your clip is now **READY** (yellow badge). Adjust parameters as needed, then click **RUN INFERENCE** (or **Ctrl+R**).

### Batch Pipeline (Unattended)

Select multiple clips in the I/O tray (Ctrl+click or Shift+click), then click **RUN PIPELINE** in the status bar. The system automatically:

1. Detects each clip's state and paint strokes
2. Runs **TRACK MASK** and then **VideoMaMa** for painted clips
3. Runs **GVM Auto** for unpainted RAW clips
4. Chains **inference** after each alpha generation completes

The entire pipeline is cancellable (Esc) and checkpointable — if interrupted, restarting picks up where each clip left off.

### 4. Review

Switch between view modes to inspect results:
- **COMP** — key over checkerboard
- **FG** — check for green fringing
- **MATTE** — inspect alpha quality
- **PROCESSED** — production RGBA

Outputs are written to the project's `Output/` subdirectories during inference.

---

## Keyboard Shortcuts

Viewable and rebindable in-app via Edit > Hotkeys.

### Global

| Shortcut | Action |
|----------|--------|
| **Ctrl+R** | Run inference on selected clip |
| **Ctrl+Shift+R** | Run all ready clips (batch) |
| **Esc** | Stop / cancel current job |
| **Ctrl+S** | Save session |
| **Ctrl+O** | Open project |
| **Ctrl+M** | Toggle mute (sound on/off) |
| **Home** | Return to welcome screen |
| **Del** | Remove selected clips |
| **Q** | Toggle queue panel |
| **F12** | Toggle debug console |

### Viewer

| Shortcut | Action |
|----------|--------|
| **Ctrl + scroll wheel** | Zoom in/out toward cursor (0.25x to 8.0x) |
| **Shift + scroll wheel** | Horizontal pan (left/right) |
| **Middle-click + drag** | Pan the image |
| **Double-click** | Reset zoom to 100% |

### Timeline

| Shortcut | Action |
|----------|--------|
| **Space** | Play / Pause |
| **I** | Set in-point marker |
| **O** | Set out-point marker |
| **Alt+I** | Clear in/out range |

### Paint

| Shortcut | Action |
|----------|--------|
| **1** | Foreground paint brush (green) |
| **2** | Background paint brush (red) |
| **C** | Cycle foreground brush color (green / blue) |
| **Shift + drag up/down** | Resize brush |
| **Alt + left-drag** | Draw straight line |
| **Ctrl+Z** | Undo last stroke on current frame |
| **Ctrl+C** | Clear all paint strokes |

---

## View Modes

The view mode bar at the top of each viewport switches what the right viewer displays:

| Mode | Source | What You See |
|------|--------|-------------|
| **INPUT** | `Input/` or `Frames/` | Original unprocessed footage |
| **FG** | `Output/FG/` | Foreground with green spill removed |
| **MATTE** | `Output/Matte/` | Alpha matte (white = opaque, black = transparent) |
| **COMP** | `Output/Comp/` | Final key composited over checkerboard |
| **PROCESSED** | `Output/Processed/` | Production RGBA — premultiplied linear for compositing |

---

## Inference Controls

| Control | Range | Default | Description |
|---------|-------|---------|-------------|
| **Color Space** | sRGB, Linear | sRGB | Working color space |
| **Despill Strength** | 0.0 – 1.0 | 1.0 | Green spill removal intensity |
| **Despeckle** | 50 – 2000 px | ON, 400 px | Removes isolated artifacts smaller than threshold |
| **Refiner Scale** | 0.0 – 3.0 | 1.0 | Edge refinement. 0 = disabled |
| **Live Preview** | — | OFF | Reprocess current frame when parameters change |

**Middle-click** any slider to reset it to default.

### Output Format

Each output channel can be individually enabled and set to EXR or PNG:

| Output | Default | Format | Description |
|--------|---------|--------|-------------|
| **FG** | ON | EXR | Foreground RGB with spill removed |
| **Matte** | ON | EXR | Single-channel alpha |
| **Comp** | ON | PNG | Key over checkerboard (for review) |
| **Processed** | ON | EXR | Full RGBA premultiplied linear (for VFX compositing) |

---

## Status Colors

| Color | State | Meaning |
|-------|-------|---------|
| Orange | EXTRACTING | Video being extracted to image sequence |
| Gray | RAW | Frames loaded, no alpha hint yet |
| Blue | MASKED | Paint masks created (ready for VideoMaMa) |
| Yellow | READY | Alpha hint available, ready for inference |
| Green | COMPLETE | Inference finished, outputs available |
| Red | ERROR | Processing failed — can retry |

---

## Frame Scrubber

The scrubber below the dual viewer provides:

- **Transport buttons:** First frame, step back, play/pause, step forward, last frame
- **Playback is CAPPED:** Pressing spacebar will play back footage at a hardcoded rate of 3FPS. This is intentional, the files are large.
- **Coverage bar:** Three color-coded lanes showing which frames have paint strokes (green), alpha hints (white), and inference output (yellow)
- **In/Out markers:** Press **I** / **O** to set a sub-range for processing. When set, the RUN button changes to "RUN SELECTED" and playback loops within the range.

---

## Project Structure

Each imported clip creates a project folder:

```
Projects/
  260301_093000_Woman_Jumps/
    Source/              # Original video
    Frames/              # Extracted EXR DWAB half-float image sequence
    AlphaHint/           # Generated alpha hints
    Output/
      FG/                # Foreground EXR/PNG
      Matte/             # Alpha matte EXR/PNG
      Comp/              # Checkerboard composite PNG
      Processed/         # Production RGBA EXR
    project.json         # Metadata and settings
```

---

## Preferences

Access via Edit > Preferences.

| Setting | Default | Description |
|---------|---------|-------------|
| **Show tooltips** | ON | Helpful tooltips on all controls |
| **UI sounds** | ON | Sound effects for actions |
| **Copy source videos** | ON | Copy imports into project folder (OFF = reference in place) |
| **Loop playback** | ON | Loop within in/out range during playback |

---

## Running from the Command Line

```bash
# GUI mode (default)
python main.py

# CLI mode (original terminal wizard)
python main.py --cli

# Verbose logging
python main.py --log-level DEBUG
```

Logs are written to `logs/backend/YYMMDD_HHMMSS_corridorkey.log`.

---

## Hardware Requirements

### Windows / Linux (NVIDIA CUDA)

| | Minimum | Recommended | Comfortable |
|---|---------|-------------|-------------|
| **VRAM** | 8 GB | 12 GB | 16 GB+ |
| **GPU** | NVIDIA (CUDA) | Ampere+ (RTX 30xx) | Ada/Blackwell (RTX 40xx/50xx) |

#### VRAM modes

| Mode | VRAM usage | 4K speed | How it works |
|------|-----------|----------|--------------|
| **Speed** (≥12 GB) | ~8.7 GB | ~1.5s/frame | Full-frame refiner, full torch.compile |
| **Low-VRAM** (<12 GB) | ~2.5 GB | ~1.6s/frame | 512×512 tiled refiner, selective compile |

Mode is auto-detected from available VRAM. Override with `CORRIDORKEY_OPT_MODE=speed|lowvram|auto`.

### macOS (Apple Silicon)

| | Minimum | Recommended |
|---|---------|-------------|
| **Chip** | M1 (8 GB) | M1 Pro+ (16 GB+) |
| **Backend** | MPS (PyTorch) | MLX (auto-detected if corridorkey-mlx installed) |

CorridorKey inference auto-selects the fastest available backend: MLX (1.5–2x faster) when `corridorkey-mlx` and a `.safetensors` checkpoint are present, otherwise PyTorch MPS. Override with `CORRIDORKEY_BACKEND=torch|mlx|auto`.

Alpha generators (SAM2, GVM, VideoMaMa, MatAnyone2) always run on PyTorch MPS — no MLX ports exist for these models. For best Mac experience, import pre-made alpha mattes from After Effects, DaVinci Resolve, or Nuke.

---

## Quality Verification

EZ-CorridorKey's optimizations (Hiera FlashAttention, TF32 tensor cores, torch.compile, tiled refiner) produce output identical to upstream CorridorKey within float32 noise floor — verified across PSNR, SSIM, MS-SSIM, LPIPS, and DeltaE 2000.

![Quality Comparison](dev-docs/guides/screenshots/quality_comparison_v1.5.0.png)

---

## Security

All installer scripts are open-source and readable in this repository. Independent VirusTotal scans for the current release:

- [**1-install.bat** (v1.6.0) — 0 detections](https://www.virustotal.com/gui/file/c88b68b2fdc429de8bd70a5dde182486c788fcdc34eb508a4a137373d1ddb1bc)

---

## Licensing & Attribution

This project wraps [Niko Pueringer's CorridorKey](https://github.com/nikopueringer/CorridorKey), licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

GUI/SFX/Workflow/QA/Maintenance by [Ed Zisk](https://www.edzisk.com).

Logo by [Sara Ann Stewart](https://www.instagram.com/sarastewartwork).
Hiera optimization by [Jhe Kim](https://github.com/Raiden129).
Tiling optimization by [MarcelLieb](https://github.com/MarcelLieb).
FX graph cache from [99oblivius](https://github.com/99oblivius) ([CorridorKey-Engine](https://github.com/99oblivius/CorridorKey-Engine)).

If you use or build on this project, please star this repo and credit the contributors <3

Optional modules:
- **SAM 2.1** ([facebookresearch/sam2](https://github.com/facebookresearch/sam2)) — Apache 2.0
- **GVM** ([aim-uofa/GVM](https://github.com/aim-uofa/GVM)) — CC BY-NC-SA 4.0
- **VideoMaMa** ([cvlab-kaist/VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)) — CC BY-NC 4.0, model weights under Stability AI Community License
- **MatAnyone2** ([pq-yang/MatAnyone2](https://github.com/pq-yang/MatAnyone2)) — Apache 2.0
- **corridorkey-mlx** ([nikopueringer/corridorkey-mlx](https://github.com/nikopueringer/corridorkey-mlx)) — CC BY-NC-SA 4.0

Join EZSCAPE Discord for EZ-CorridorKey troubleshooting: https://discord.gg/6kgxHUfA

Join the Corridor Creates Discord: https://discord.gg/zvwUrdWXJm






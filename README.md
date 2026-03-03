# EZ-CorridorKey

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
| Annotation / masking | Manual external tool | Built-in brush tool for VideoMaMa masks |

---

## Installation

**One-Click Install (Windows / macOS / Linux):**
1. Clone or download this repository.
2. Ensure you have [Python 3.10+](https://python.org) installed (check "Add to PATH" on Windows).
3. Run the installer for your platform:
   - **Windows:** Double-click `1-install.bat`
   - **macOS / Linux:** `chmod +x 1-install.sh && ./1-install.sh`
4. The installer handles everything: virtual environment, dependencies (including correct PyTorch for your GPU), and model downloads.
5. To launch: double-click `2-start.bat` (Windows) or `./2-start.sh` (macOS/Linux).

**What the installer does:**
- Installs [uv](https://docs.astral.sh/uv/) (fast Python package manager) — falls back to pip if needed
- Creates a `.venv` virtual environment in the project folder
- Auto-detects your GPU and installs the correct PyTorch variant (CUDA on NVIDIA, MPS on Apple Silicon, CPU fallback)
- Downloads the CorridorKey model checkpoint (383 MB, required)
- Optionally downloads GVM (~6 GB) and VideoMaMa (~37 GB) alpha hint generators
- Checks for FFmpeg and suggests install methods if missing

**Updating:**
- **Windows:** Double-click `3-update.bat`
- **macOS / Linux:** `./3-update.sh`

---

## Application Layout

```
+--------------------------------------------------------------------+
| File | Edit | View | Help                             Volume ---->-|
| CORRIDORKEY                                        RTX 5090  ## 4GB|
+------+------------------------+----------------------+-------------+
|      | 241 frames - RAW      | 241 frames - RAW     |  ALPHA GEN   |
|  Q   | [INPUT] FG MATTE COMP | INPUT FG MATTE [COMP]|  GVM AUTO    |
|  U   +-----------------------+----------------------+  VIDEOMAMA   |
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
|      +<<--------------------------------------- < >>|--------------|
|      | INPUT (8)       + ADD | EXPORTS (0)                         |
|      | +-----+ +-----+ +-----+ +-----+ +-----+                     |
|      | |thumb| |thumb| |thumb| |thumb| |thumb| ..                  |
|      | +-----+ +-----+ +-----+ +-----+ +-----+                     |
+------+-----------------------------------------------+-------------+
|  Status Bar                                            [RUN INF]   |
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

Drop a video file onto the welcome screen (or File > Import Clips > Import Video). The video is extracted to an image sequence automatically.

### 2. Generate Alpha Hint

Your clip starts in **RAW** state (gray badge). You need an alpha hint before running inference.

**Option A — GVM Auto (one-click):**
Click **GVM AUTO** in the parameter panel. Works great for most green screen footage with people.

**Option B — VideoMaMa (manual masking):**
For difficult shots, use the annotation brush:
1. Press **1** to activate foreground mode (green)
2. Paint over the subject on a few key frames
3. Press **2** to switch to background mode (red)
4. Paint over background areas
5. Click **VIDEOMAMA** in the parameter panel

### 3. Run Inference

Your clip is now **READY** (yellow badge). Adjust parameters as needed, then click **RUN INFERENCE** (or **Ctrl+R**).

### Batch Pipeline (Unattended)

Select multiple clips in the I/O tray (Ctrl+click or Shift+click), then click **RUN PIPELINE** in the status bar. The system automatically:

1. Detects each clip's state and annotations
2. Exports masks and runs **VideoMaMa** for annotated clips
3. Runs **GVM Auto** for unannotated RAW clips
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

### Annotation

| Shortcut | Action |
|----------|--------|
| **1** | Foreground annotation brush (green) |
| **2** | Background annotation brush (red) |
| **C** | Cycle foreground color (green / blue) |
| **Shift + drag up/down** | Resize brush |
| **Alt + left-drag** | Draw straight line |
| **Ctrl+Z** | Undo last stroke on current frame |
| **Ctrl+C** | Clear all annotations |

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
| Blue | MASKED | Annotation masks painted (ready for VideoMaMa) |
| Yellow | READY | Alpha hint available, ready for inference |
| Green | COMPLETE | Inference finished, outputs available |
| Red | ERROR | Processing failed — can retry |

---

## Frame Scrubber

The scrubber below the dual viewer provides:

- **Transport buttons:** First frame, step back, play/pause, step forward, last frame
- **Coverage bar:** Three color-coded lanes showing which frames have annotations (green), alpha hints (white), and inference output (yellow)
- **In/Out markers:** Press **I** / **O** to set a sub-range for processing. When set, the RUN button changes to "RUN SELECTED" and playback loops within the range.

---

## Project Structure

Each imported clip creates a project folder:

```
Projects/
  260301_093000_Woman_Jumps/
    Source/              # Original video
    Frames/              # Extracted image sequence
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

- **CorridorKey:** ~1.5 GB VRAM (8 GB GPU minimum, 12 GB recommended)
- **GVM (Optional):** ~4.5 GB VRAM
- **VideoMaMa (Optional):** ~6–8 GB VRAM

Only one model is loaded at a time. A 12 GB GPU handles all three models comfortably.

---

## Licensing & Attribution

This project wraps [Niko Pueringer's CorridorKey](https://github.com/nikopueringer/CorridorKey), licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

GUI/SFX/Workflow enhancements by [Ed Zisk](https://www.edzisk.com).
Logo by [Sara Ann Stewart](https://www.clade.design/).

If you use or build on this project, please star this repo and credit the contributors <3

Optional modules:
- **GVM** ([aim-uofa/GVM](https://github.com/aim-uofa/GVM)) — CC BY-NC-SA 4.0
- **VideoMaMa** ([cvlab-kaist/VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)) — CC BY-NC 4.0, model weights under Stability AI Community License

Join the Corridor Creates Discord: https://discord.gg/zvwUrdWXJm

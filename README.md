# EZ-CorridorKey **v1.10.0**

[![Release](https://img.shields.io/github/v/release/edenaion/EZ-CorridorKey?style=flat-square&label=Release&labelColor=141300&color=FFF203)](https://github.com/edenaion/EZ-CorridorKey/releases/latest)
[![Stars](https://img.shields.io/github/stars/edenaion/EZ-CorridorKey?style=flat-square&labelColor=141300&color=FFF203)](https://github.com/edenaion/EZ-CorridorKey/stargazers)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-FFF203?style=flat-square&labelColor=141300)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Discord](https://img.shields.io/badge/Discord-EZSCAPE-5865F2?style=flat-square&labelColor=000000&logo=discord&logoColor=50FF80)](https://discord.gg/TyxNjcWeF3)
[![EZSCAPE](https://img.shields.io/badge/EZSCAPE-Plugins/Tools-50FF80?style=flat-square&labelColor=000000)](https://www.ezscape.space)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-454430?style=flat-square&labelColor=141300)]()

> **Latest release: [v1.10.0](https://github.com/edenaion/EZ-CorridorKey/releases/tag/v1.10.0)** — Project output dirs, wizard overhaul, signed updates, frozen build fixes. See the [full changelog](CHANGELOG.md).

A full desktop GUI for [Niko Pueringer's CorridorKey](https://github.com/nikopueringer/CorridorKey) — the AI green screen keyer by Corridor Digital that physically unmixes foreground from background, preserving hair, motion blur, and translucency.

This GUI replaces the CLI drag-and-drop workflow with a complete desktop application while preserving 100% backward compatibility (`python main.py --cli` still runs the original wizard).

![EZ-CorridorKey](docs/screenshots/EZ-CorridorKey_Inference_Done.jpg)

### Contents

- [Installation](#installation) — Desktop installer, CLI setup, Docker
- [Application Layout](#application-layout) — UI overview
- [Quick Start](#quick-start) — Import, generate alpha, run inference
- [Keyboard Shortcuts](#keyboard-shortcuts) — Full hotkey reference
- [View Modes](#view-modes) — Output channel switching
- [Inference Controls](#inference-controls) — Parameters and output formats
- [Hardware Requirements](#hardware-requirements) — VRAM, GPU, and platform info
- [Security](#security) — Verified downloads, signed updates, checksums
- [Contributing](#contributing--support) — How to help and get help

[![Star History Chart](https://api.star-history.com/svg?repos=edenaion/EZ-CorridorKey&type=Date)](https://star-history.com/#edenaion/EZ-CorridorKey&Date)

| Capability          | CLI (Upstream)              | GUI (This Project)                                   |
| ------------------- | --------------------------- | ---------------------------------------------------- |
| Import clips        | Drag onto .bat file         | Drag-drop into app, or File > Import                 |
| Configure inference | Terminal prompts            | Sliders, dropdowns, checkboxes                       |
| Monitor progress    | Terminal text output        | Progress bars, frame counter, ETA                    |
| Preview results     | Open output folder manually | Real-time dual viewer (input vs output)              |
| Job management      | One clip at a time          | Queue with batch processing + full pipeline mode     |
| GPU monitoring      | None                        | Live VRAM meter in brand bar                         |
| Keyboard shortcuts  | None                        | 20+ hotkeys                                          |
| Sound feedback      | None                        | 7 context-aware sound effects                        |
| Session persistence | None                        | Recent projects, auto-save                           |
| Paint / masking     | Manual external tool        | Built-in brush tool for VideoMaMa / MatAnyone2 masks |
| Alpha generators    | None                        | GVM, BiRefNet, VideoMaMa, MatAnyone2 (one-click)     |
| Apple Silicon       | MPS only                    | MLX acceleration (auto-detected)                     |

---

## Installation

*Do NOT download the GitHub Zip, this is not a viable installation method. You will have issues. Do not rename the installation path to anything other than EZ-CorridorKey

### Desktop App Installer (recommended)

**Don't want to deal with Python, git, or the command line?** A full Windows installer, portable exe, and macOS `.pkg` are available. Entirely optional and free — donations help support active development.

[![Download on Gumroad](https://img.shields.io/badge/Download-Gumroad-ff90e8?style=for-the-badge&logo=gumroad&logoColor=white)](https://edenaion.gumroad.com/)

The installer includes everything — Python runtime, AI models, GPU libraries — no setup required. Just install and run.

### Terminal (CLI) Install (Windows / macOS / Linux)

1. Git Clone this repository:
git clone https://github.com/edenaion/EZ-CorridorKey
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
- Optionally downloads BiRefNet (~940 MB), MatAnyone2 (~135 MB), GVM (~6 GB), and VideoMaMa (~37 GB) alpha hint generators
- Creates a desktop shortcut (optional)

**Updating:**

- **Windows Desktop App Installer users:** The app checks for updates automatically. When a new version is available, click the update button in the app. It downloads a lightweight patch and relaunches.
- **macOS Desktop App users on 1.9.x:** 1.10.0 does **not** auto-update on Mac. You must download the new `.pkg` from Gumroad and run it manually. The installer replaces the existing `/Applications/EZ-CorridorKey.app` in place. Your preferences, projects, and downloaded models live in `~/Library/Application Support/EZ-CorridorKey/` and are preserved across the reinstall. Mac auto-update returns in 1.11.0 or later.
- **CLI users:** Double-click `3-update.bat` (Windows) or run `./3-update.sh` (macOS/Linux). This pulls the latest code via git, or downloads a ZIP if git isn't available.

> **Note:** The update ZIP on GitHub Releases (`EZ-CorridorKey-windows-x64.zip`) is for Windows Desktop App Installer users only. It patches an existing installation. CLI users should continue using `3-update.bat` / `3-update.sh`.

### Alternate Installation: Docker

For Linux users or remote/cloud setups, EZ-CorridorKey can run inside Docker with browser-based access via noVNC. See [docker/README.md](docker/README.md) for setup instructions. The native install above is recommended for Windows and macOS.

---

## Application Layout

```
+--------------------------------------------------------------------+
| File | Edit | View | Help                             Volume ---->-|
| CORRIDORKEY                                        RTX 5090  ## 4GB|
+------+------------------------+---------------------+--------------+
|      |      241 frames - RAW | IN|FG|MATTE[COMP]PROC| ALPHA GEN    |
|  Q   |                       |                      |  GVM/BIREFNET|
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

**Option A — One-click alpha generators:**
- **BiRefNet** (recommended) — click **BIREFNET** in the parameter panel. Fast, accurate, and works well on a wide range of footage.
- **GVM Auto** — click **GVM AUTO** in the parameter panel. Works great for most green screen footage with people.

**Option B — Track Mask + MatAnyone2 / VideoMaMa:**
For difficult shots, use the paint brush as a prompt tool:

1. Press **1** to activate foreground mode (green)
2. Paint over the subject on a few key frames
3. Press **2** to switch to background mode (red)
4. Paint over background areas
5. Click **TRACK MASK** to generate a dense SAM2 mask track
6. Click **MATANYONE2** or **VIDEOMAMA** in the parameter panel

**Option C — Import Alpha (bring your own):**
If you already have alpha mattes from another tool (Rotobrush, Silhouette, Resolve, Nuke, etc.), click **IMPORT ALPHA** in the parameter panel and choose either an image folder or a matte video file.

- Supported image formats: **PNG, JPG, JPEG, TIF, TIFF, EXR**
- Supported alpha-video path: standard video files accepted by the normal clip importer (for example **MOV** or **MP4**)
- Images and visible matte videos should be **grayscale** (white = foreground, black = background)
- Frame count should match your input sequence
- Non-PNG stills are automatically converted to grayscale PNG on import
- Imported alpha videos are decoded into grayscale `AlphaHint/*.png` frames so they follow the same downstream path as image-sequence hints
- Imported files are copied into the clip's `AlphaHint/` folder and the clip advances to **READY** state

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

- **FG** — check for green fringing
- **MATTE** — inspect alpha quality
- **COMP** — key over checkerboard
- **PROCESSED** — production RGBA

Outputs are written to the project's `Output/` subdirectories during inference (configurable — see [Custom Output Directory](#custom-output-directory)).

---

## Keyboard Shortcuts

Viewable and rebindable in-app via Edit > Hotkeys.

### Global

| Shortcut         | Action                         |
| ---------------- | ------------------------------ |
| **Ctrl+R**       | Run inference on selected clip |
| **Ctrl+Shift+R** | Run all ready clips (batch)    |
| **Esc**          | Stop / cancel current job      |
| **Ctrl+S**       | Save session                   |
| **Ctrl+O**       | Open project                   |
| **Ctrl+M**       | Toggle mute (sound on/off)     |
| **Home**         | Return to welcome screen       |
| **Del**          | Remove selected clips          |
| **Q**            | Toggle queue panel             |
| **Ctrl+,**       | Toggle Preferences             |
| **F12**          | Toggle debug console           |

### Viewer

| Shortcut                 | Action                                    |
| ------------------------ | ----------------------------------------- |
| **F1–F7**                | View modes: INPUT, MASK, ALPHA, FG, MATTE, COMP, PROC |
| **A**                    | Toggle A/B wipe comparison                |
| **Ctrl + scroll wheel**  | Zoom in/out toward cursor (0.25x to 8.0x) |
| **Shift + scroll wheel** | Horizontal pan (left/right)               |
| **Middle-click + drag**  | Pan the image                             |
| **Double-click**         | Reset zoom to 100%                        |

### Timeline

| Shortcut  | Action               |
| --------- | -------------------- |
| **Space** | Play / Pause         |
| **I**     | Set in-point marker  |
| **O**     | Set out-point marker |
| **Alt+I** | Clear in/out range   |

### Paint

| Shortcut                 | Action                                      |
| ------------------------ | ------------------------------------------- |
| **1**                    | Foreground paint brush (green)              |
| **2**                    | Background paint brush (red)                |
| **C**                    | Cycle foreground brush color (green / blue) |
| **Shift + drag up/down** | Resize brush                                |
| **Alt + left-drag**      | Draw straight line                          |
| **Ctrl+Z**               | Undo last stroke on current frame           |
| **Ctrl+C**               | Clear all paint strokes                     |

---

## View Modes

The view mode bar at the top of each viewport switches what the right viewer displays:

| Mode          | Source                | What You See                                              |
| ------------- | --------------------- | --------------------------------------------------------- |
| **INPUT**     | `Input/` or `Frames/` | Original unprocessed footage                              |
| **MASK**      | Paint strokes         | Foreground (green/blue) and background (red) paint masks  |
| **ALPHA**     | `AlphaHint/`          | Alpha hint used as input to inference                     |
| **FG**        | `Output/FG/`          | Foreground with green spill removed                       |
| **MATTE**     | `Output/Matte/`       | Alpha matte (white = opaque, black = transparent)         |
| **COMP**      | `Output/Comp/`        | Final key composited over checkerboard                    |
| **PROCESSED** | `Output/Processed/`   | Production RGBA — straight linear for Resolve/compositing |

---

## Inference Controls

| Control              | Range        | Default    | Description                                           |
| -------------------- | ------------ | ---------- | ----------------------------------------------------- |
| **Color Space**      | sRGB, Linear | sRGB       | How CorridorKey interprets the input before inference |
| **Despill Strength** | 0.0 – 1.0    | 1.0        | Green spill removal intensity                         |
| **Despeckle**        | 50 – 2000 px | ON, 400 px | Removes isolated artifacts smaller than threshold     |
| **Refiner Scale**    | 0.0 – 3.0    | 1.0        | Edge refinement. 0 = disabled                         |
| **Live Preview**     | —            | ON         | Reprocess current frame when parameters change        |

**Color Space behavior**

- The **left INPUT viewer** always shows CorridorKey's current interpretation of the source. If the input looks wrong there, your future inference results and exports will be based on that wrong interpretation too.
- Changing **Color Space** before clicking **RUN INFERENCE** affects how the next live preview and the next export run are generated.
- Changing **Color Space** after outputs already exist does **not** rewrite those files on disk. It only updates the viewer and live preview. To keep that new interpretation in saved files, rerun inference.
- CorridorKey auto-detects color space from file type and metadata when possible, but you can override it if the INPUT viewer does not look representative.
- With **Live Preview** enabled, the first adjustment after a fresh launch may pause briefly while the inference engine loads.

**Middle-click** any slider to reset it to default.

### Output Format

Each output channel can be individually enabled and set to EXR or PNG:

| Output        | Default | Format | Description                                             |
| ------------- | ------- | ------ | ------------------------------------------------------- |
| **FG**        | ON      | EXR    | Foreground RGB with spill removed                       |
| **Matte**     | ON      | EXR    | Single-channel alpha                                    |
| **Comp**      | ON      | PNG    | Key over checkerboard (for review)                      |
| **Processed** | ON      | EXR    | Full RGBA straight linear (for Resolve and compositing) |

---

## Status Colors

| Color  | State      | Meaning                                   |
| ------ | ---------- | ----------------------------------------- |
| Orange | EXTRACTING | Video being extracted to image sequence   |
| Gray   | RAW        | Frames loaded, no alpha hint yet          |
| Blue   | MASKED     | Paint masks created (ready for VideoMaMa) |
| Yellow | READY      | Alpha hint available, ready for inference |
| Green  | COMPLETE   | Inference finished, outputs available     |
| Red    | ERROR      | Processing failed — can retry             |

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

| Setting                      | Default            | Description                                                 |
| ---------------------------- | ------------------ | ----------------------------------------------------------- |
| **Show tooltips**            | ON                 | Helpful tooltips on all controls                            |
| **UI sounds**                | ON                 | Sound effects for actions                                   |
| **Copy source videos**       | ON                 | Copy imports into project folder (OFF = reference in place) |
| **Loop playback**            | ON                 | Loop within in/out range during playback                    |
| **Default output directory** | (inside project)   | Global output location — outputs go to `<dir>/<Project>/<Clip>/` |

### Custom Output Directory

By default, inference output is written to `Output/` inside each clip folder. You can redirect output at three levels:

- **Global**: Preferences > Output > Default output directory
- **Per-project**: File > Set Project Output Folder
- **Per-clip**: Right-click a clip > Set Output Directory

Priority: per-clip > per-project > global preference > default.

When using a non-default directory, outputs are organized as `<dir>/<ProjectName>/<ClipName>/FG/`, `Matte/`, etc. to prevent collisions between projects. The per-project setting persists to `project.json`.

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

|          | Minimum       | Recommended        | Comfortable                   |
| -------- | ------------- | ------------------ | ----------------------------- |
| **VRAM** | 8 GB          | 12 GB              | 16 GB+                        |
| **GPU**  | NVIDIA (CUDA) | Ampere+ (RTX 30xx) | Ada/Blackwell (RTX 40xx/50xx) |

#### VRAM modes

| Mode                  | VRAM usage | 4K speed    | How it works                             |
| --------------------- | ---------- | ----------- | ---------------------------------------- |
| **Speed** (≥12 GB)    | ~8.7 GB    | ~1.5s/frame | Full-frame refiner, full torch.compile   |
| **Low-VRAM** (<12 GB) | ~2.5 GB    | ~1.6s/frame | 512×512 tiled refiner, selective compile |

Mode is auto-detected from available VRAM. Override with `CORRIDORKEY_OPT_MODE=speed|lowvram|auto`.

### macOS (Apple Silicon)

|             | Minimum       | Recommended                                      |
| ----------- | ------------- | ------------------------------------------------ |
| **Chip**    | M1 (8 GB)     | M1 Pro+ (16 GB+)                                 |
| **Backend** | MPS (PyTorch) | MLX (auto-detected if corridorkey-mlx installed) |

CorridorKey inference auto-selects the fastest available backend: MLX (1.5–2x faster) when `corridorkey-mlx` and a `.safetensors` checkpoint are present, otherwise PyTorch MPS. Override with `CORRIDORKEY_BACKEND=torch|mlx|auto`.

Alpha generators (SAM2, GVM, VideoMaMa, MatAnyone2) always run on PyTorch MPS — no MLX ports exist for these models. For best Mac experience, import pre-made alpha mattes from After Effects, DaVinci Resolve, or Nuke.

> **Known issue:** MLX FG output can produce blocky/artifacted results. Disable FG output and use Matte + Processed outputs instead. CUDA users are not affected.

---

## Quality Verification

EZ-CorridorKey's optimizations (Hiera FlashAttention, TF32 tensor cores, torch.compile, tiled refiner) produce output identical to upstream CorridorKey within float32 noise floor — verified across PSNR, SSIM, MS-SSIM, LPIPS, and DeltaE 2000.

![Quality Comparison](docs/screenshots/quality_comparison_v1.5.0.png)

---

## Security

### Verified download sources

The only official sources for EZ-CorridorKey are:

☼ **GitHub:** [github.com/edenaion/EZ-CorridorKey/releases](https://github.com/edenaion/EZ-CorridorKey/releases)
☼ **Gumroad:** [edenaion.gumroad.com](https://edenaion.gumroad.com/)

Any other site hosting EZ-CorridorKey downloads is **unverified and potentially malware**. Do not download from third-party mirrors, repackaging sites, or file-sharing links. If you see EZ-CorridorKey hosted elsewhere, please report it in [GitHub Issues](https://github.com/edenaion/EZ-CorridorKey/issues) or in the [EZSCAPE Discord](https://discord.gg/TyxNjcWeF3).

### Code signing

☼ **Windows:** The installer (.exe) is signed via Azure Trusted Signing. Windows SmartScreen shows **EZscape Ventures LLC** as the verified publisher.
☼ **macOS:** The .pkg is code-signed and Apple-notarized under **Developer ID: Edward Zisk (UX6RDC39ZW)**. Gatekeeper verifies it on first launch.

### Signed updates

Starting with v1.10.0, every GitHub release ships a signed manifest (`manifest.json` + `manifest.json.sig`). When the in-app updater downloads a new version, it verifies the Ed25519 signature and SHA-256 hash before applying the update. If either check fails, the update is rejected and the user is warned.

This means even if the GitHub account were compromised, the attacker could not push a malicious update to installed users without also possessing the offline signing key.

The verification code is open source: [`backend/update_verify.py`](backend/update_verify.py).

### Checksums

Each release includes a `SHA256SUMS.txt` file listing the SHA-256 hash of every release artifact. Download it alongside the binary and verify manually:

```bash
# macOS / Linux
sha256sum -c SHA256SUMS.txt

# Windows PowerShell
(Get-FileHash EZ-CorridorKey-1.10.0-Windows-x64-Setup.exe -Algorithm SHA256).Hash
```

### Git integrity

All commits to `main` are signed with an Ed25519 SSH key. Branch protection enforces signed commits, pull request reviews, and blocks force pushes and branch deletion.

### VirusTotal

Independent scans:

- [**EZ-CorridorKey.exe** (v1.10.0, signed inner exe) — VirusTotal scan](https://www.virustotal.com/gui/file/82019d296fbc8064fcbac99e71699a0ee5d81ee2893b4d3dbbb25f265282ba0f?nocache=1)
- [**1-install.bat** (v1.6.0) — 0 detections](https://www.virustotal.com/gui/file/c88b68b2fdc429de8bd70a5dde182486c788fcdc34eb508a4a137373d1ddb1bc)

> **Third-party models:** The core CorridorKey checkpoint (`CorridorKey.pth`) is the only model we can vouch for. Optional models (SAM2, GVM, VideoMaMa, MatAnyone2, BiRefNet) are downloaded from their respective authors' repositories -- use them at your own discretion.

---

## Contributing & Support

EZ-CorridorKey is a labor of love — built and maintained by one man from Brooklyn for the VFX community. If this tool saves you time on a project, consider paying it forward:

[![Ko-Fi](https://img.shields.io/badge/Ko--fi-Support%20Development-FF5E5B?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/edenaion)
[![EZSCAPE Plugins](https://img.shields.io/badge/EZSCAPE-Plugins%20%26%20Tools-50FF80?style=for-the-badge&labelColor=000000)](https://www.ezscape.space)
[![RunPod](https://img.shields.io/badge/RunPod-Cloud%20GPU-673AB7?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJ3aGl0ZSI+PHBhdGggZD0iTTEyIDJMMiA3bDEwIDUgMTAtNS0xMC01ek0yIDE3bDEwIDUgMTAtNS0xMC01LTEwIDV6TTIgMTJsMTAgNSAxMC01LTEwLTUtMTAgNXoiLz48L3N2Zz4=&logoColor=white)](https://runpod.io?ref=2k18fmnh)

- **Star this repo** if you find it useful -- it helps others discover the project
- **Purchase a plugin** -- I pour my heart into [EZSCAPE plugins](https://www.ezscape.space)
- **Report bugs** via [GitHub Issues](https://github.com/edenaion/EZ-CorridorKey/issues)
- **Contribute code** -- see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
- **Security issues** -- see [SECURITY.md](SECURITY.md) for responsible disclosure

Need GPU compute for inference? [RunPod](https://runpod.io?ref=2k18fmnh) offers on-demand cloud GPUs — great for batch processing large shoots without tying up your local machine. Using my referral link will enable me to build more tools for all!

### Community

[![Discord - EZSCAPE](https://img.shields.io/badge/EZSCAPE-Support%20%26%20Chill-5865F2?style=flat-square&labelColor=000000&logo=discord&logoColor=50FF80)](https://discord.gg/TyxNjcWeF3)
[![Discord - Corridor Creates](https://img.shields.io/badge/Corridor%20Creates-Community-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/zvwUrdWXJm)

---

## Licensing & Attribution

This project wraps [Niko Pueringer's CorridorKey](https://github.com/nikopueringer/CorridorKey), licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

GUI/SFX/Workflow/QA/Maintenance by [Ed Zisk](https://www.edzisk.com).

<details>
<summary><strong>Contributors</strong></summary>

- Logo by [Sara Ann Stewart](https://www.instagram.com/sarastewartwork)
- Hiera optimization by [Jhe Kim](https://github.com/Raiden129)
- Tiling optimization by [MarcelLieb](https://github.com/MarcelLieb)
- MLX Apple Silicon backend by [Cristopher Yates](https://github.com/cmoyates) ([corridorkey-mlx](https://github.com/cmoyates/corridorkey-mlx))
- FX graph cache from [99oblivius](https://github.com/99oblivius) ([CorridorKey-Engine](https://github.com/99oblivius/CorridorKey-Engine))
- BiRefNet integration adapted from [Warwlock](https://github.com/Warwlock)'s [upstream PR](https://github.com/edenaion/EZ-CorridorKey/pull/10)
- Docker / noVNC browser mode by [DCRepublic](https://github.com/DCRepublic)

</details>

<details>
<summary><strong>Optional modules & licenses</strong></summary>

- **SAM 2.1** ([facebookresearch/sam2](https://github.com/facebookresearch/sam2)) — Apache 2.0
- **GVM** ([aim-uofa/GVM](https://github.com/aim-uofa/GVM)) — CC BY-NC-SA 4.0
- **VideoMaMa** ([cvlab-kaist/VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)) — CC BY-NC 4.0, model weights under Stability AI Community License
- **MatAnyone2** ([pq-yang/MatAnyone2](https://github.com/pq-yang/MatAnyone2)) — Apache 2.0
- **BiRefNet** ([ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)) — MIT

</details>

If you use or build on this project, please star this repo and credit the contributors.

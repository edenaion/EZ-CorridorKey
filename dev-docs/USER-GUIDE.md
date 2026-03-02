# EZ-CorridorKey GUI — User Guide

> A complete reference for the PySide6 desktop interface built on top of CorridorKey.
> Every feature documented here is traced to its source code location.

---

## What This GUI Adds

The upstream CorridorKey project is a CLI-only tool — you drag clips onto a `.bat` file, answer terminal prompts, and wait. This GUI replaces that workflow with a full desktop application while preserving 100% backward compatibility (`python main.py --cli` still runs the original wizard).

| Capability | CLI (Upstream) | GUI (This Project) |
|------------|---------------|-------------------|
| Import clips | Drag onto .bat file | Drag-drop into app, or File > Import |
| Configure inference | Terminal prompts | Sliders, dropdowns, checkboxes |
| Monitor progress | Terminal text output | Progress bars, frame counter, ETA |
| Preview results | Open output folder manually | Real-time dual viewer (input vs output) |
| Job management | One clip at a time | Queue with batch processing |
| GPU monitoring | None | Live VRAM meter in brand bar |
| Keyboard shortcuts | None | 18 hotkeys |
| Sound feedback | None | 7 context-aware sound effects |
| Session persistence | None | Recent projects, auto-save |
| Annotation / masking | Manual external tool | Built-in brush tool for VideoMaMa masks |

---

## Application Layout

When you launch the GUI, you'll see the **Welcome Screen**. After importing clips, the workspace has five main areas:

```
┌─────────────────────────────────────────────────────────────────────┐
│ CORRIDORKEY    File  Edit  View  Help              RTX 5090  ██ 4GB │
├──────┬───────────────────────┬──────────────────────┬───────────────┤
│      │ 241 frames · RAW      │ 241 frames · RAW     │ ALPHA GEN    │
│  Q   │ [INPUT] FG MATTE COMP │ INPUT FG MATTE [COMP]│  GVM AUTO    │
│  U   ├───────────────────────┼──────────────────────┤  VIDEOMAMA   │
│  E   │                       │                      │  EXPORT MASK │
│  U   │                       │                      │──────────────│
│  E   │   INPUT viewer        │   OUTPUT viewer      │ INFERENCE    │
│      │   (left image)        │   (right image)      │  Color Space │
│      │                       │                      │  Despill     │
│ tab  │                       │                      │  Despeckle   │
│      │                       │                      │  Refiner     │
│      │                       │                      │  Live Preview│
│      │                       │                      │──────────────│
│      │                       │                      │ OUTPUT       │
│      │                       │                      │  ☑ FG       │
│      │                       │                      │  ☑ Matte    │
│      │                       │                      │  ☑ Comp     │
│      │                       │                      │  ☑ Processed │
│      ├──────────────────────────────────────────────┤  ◀ ▶▶  100% │
│      │ INPUT (8)                + ADD   EXPORTS (0) │              │
│      │ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐    │              │
│      │ │MASKD│ │ RAW │ │ RAW │ │ RAW │ │ RAW │ ·· │              │
│      │ │thumb│ │thumb│ │thumb│ │thumb│ │thumb│    │              │
│      │ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘    │              │
├──────┴──────────────────────────────────────────────┴──────────────┤
│  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  [RUN INF] │
└───────────────────────────────────────────────────────────────────  ┘
```

**Layout breakdown:**
- **Brand bar + menu** — single top row with CORRIDORKEY logo (left), menu bar (center-left), GPU name + VRAM meter (right)
- **Queue panel** — collapsible narrow sidebar on far left, vertical "QUEUE" tab always visible (24px)
- **Dual viewer** — center area, split into left (INPUT locked) and right (switchable mode), each with its own view mode buttons above
- **Parameter panel** — full-height right sidebar (240px), scrollable: Alpha Generation → Inference → Output sections
- **I/O tray** — horizontal thumbnail strip below the dual viewer, with INPUT/EXPORTS tabs and +ADD button
- **Status bar** — minimal bottom row with progress bar and RUN INFERENCE button

**Source:** Layout constructed in [main_window.py:301-362](ui/main_window.py#L301-L362)

---

## Welcome Screen

**Source:** [welcome_screen.py](ui/widgets/welcome_screen.py)

The first thing you see when launching the app.

**Left panel (320px):** Recent projects — click to reopen a previous workspace. Each card shows the project name, path, and last-opened date. Click the folder icon to open in Explorer, or the × to remove from recents.

**Right panel:** Drop zone — drag video files or image sequence folders directly onto it. You can also click the Browse button or click anywhere in the drop zone.

**Accepted file types:**
- **Video:** `.mp4` `.mov` `.avi` `.mkv` `.mxf` `.webm` `.m4v`
- **Image sequences:** `.exr` `.png` `.tif` `.tiff` `.jpg` `.jpeg` `.dpx`

When you drop a video, it's automatically extracted to an image sequence via FFmpeg. Image sequence folders are loaded directly.

---

## Clip Browser (Left Sidebar)

**Source:** [clip_browser.py](ui/widgets/clip_browser.py)

The clip browser shows all clips in the current project. Each clip card displays:
- **Thumbnail** (60×40px) — first frame of the clip
- **State badge** — colored indicator showing processing state
- **Clip name** (bold) — editable via right-click > Rename
- **Processing indicator** — animated "..." when a job is running

### Adding Clips

Click the **+ADD** button at the top:
- **Import Folder...** — select a directory containing an image sequence
- **Import Video(s)...** — select one or more video files

You can also drag-and-drop files directly onto the clip browser or the I/O tray.

### Context Menu (Right-Click)

- **Rename Project** — change the display name
- **Open in Explorer** — open the project folder
- **Clear Outputs** — delete FG/Matte/Comp/Processed outputs (resets COMPLETE → READY)
- **Delete Project** — two-stage: "Remove from List" or "Delete from Disk"

### Collapsing

The clip browser can be collapsed to save space. When collapsed, a floating expand button (▶) appears at position (2, 2). Default width: 220px.

---

## Dual Viewer (Center)

**Source:** [dual_viewer.py](ui/widgets/dual_viewer.py), [split_view.py](ui/widgets/split_view.py)

The main preview area shows two images side by side:
- **Left viewer** — locked to INPUT mode (original footage)
- **Right viewer** — switchable between view modes (default: COMP)

Both viewers are synced to the same frame via the shared scrubber below.

### Zoom & Pan

| Action | Effect |
|--------|--------|
| **Ctrl + scroll wheel** | Zoom in/out (0.25× to 8.0×) |
| **Middle-click + drag** | Pan the image |
| **Double-click** | Reset zoom to 100% |

The current zoom level is shown in the bottom-right corner (e.g., "100%").

### Annotation Drawing

When annotation mode is active (press **1** or **2**), the viewer becomes a drawing canvas:

| Action | Effect |
|--------|--------|
| **Left-click + drag** | Paint brush stroke |
| **Shift + drag up/down** | Resize brush radius |
| **Alt + left-drag** | Draw straight line |
| **Ctrl+Z** | Undo last stroke on current frame |
| **Ctrl+C** | Clear all annotations |

Annotations are used for the VideoMaMa alpha generation workflow. Green strokes (hotkey **1**) mark foreground, red strokes (hotkey **2**) mark background.

**Source:** [annotation_overlay.py](ui/widgets/annotation_overlay.py), brush interactions in [split_view.py:64-86](ui/widgets/split_view.py#L64-L86)

---

## View Modes

**Source:** [view_mode_bar.py](ui/widgets/view_mode_bar.py), enum definition in [frame_index.py:27-42](ui/preview/frame_index.py#L27-L42)

The view mode bar appears at the top of each preview viewport. Five buttons let you switch what the right viewer displays:

| Mode | Source Directory | What You See |
|------|-----------------|-------------|
| **INPUT** | `Input/` or `Frames/` | Original unprocessed footage |
| **FG** | `Output/FG/` | Foreground with green spill removed |
| **MATTE** | `Output/Matte/` | Alpha matte (white = opaque, black = transparent) |
| **COMP** | `Output/Comp/` | Final key composited over checkerboard |
| **PROCESSED** | `Output/Processed/` | Production RGBA — premultiplied linear for compositing |

**COMP** is the default view. Buttons are disabled until that mode has output frames available. If the current mode becomes unavailable, the viewer falls back to COMP → INPUT → first available.

**Button colors:**
- Active: yellow (#FFF203) background, black text, bold
- Inactive: dark (#1A1900), gray text
- Disabled: muted text, dark border

---

## Frame Scrubber & Timeline

**Source:** [frame_scrubber.py](ui/widgets/frame_scrubber.py)

The scrubber sits below the dual viewer. From left to right:

### Frame Counter
Shows "X / Y" (1-indexed) — current frame and total frame count. Fixed width 90px.

### Transport Buttons

| Button | Action |
|--------|--------|
| **◀◀** | Jump to first frame |
| **◀** | Step back one frame |
| **▶ / ❚❚** | Play / Pause (shows ❚❚ during playback) |
| **▶** | Step forward one frame |
| **▶▶** | Jump to last frame |

Playback runs at **24 fps** (42ms interval) by default. When in/out markers are set and loop is enabled (Preferences), playback loops within that range.

### Coverage Bar

**Source:** [frame_scrubber.py:21-120](ui/widgets/frame_scrubber.py#L21-L120) (CoverageBar class)

A thin multi-lane bar above the slider showing which frames have data:

| Lane | Color | Meaning |
|------|-------|---------|
| **Top** | Green (#2CC350) | Frames with annotation brush strokes |
| **Middle** | White (#C8C8C8) | Frames with alpha hints (GVM or VideoMaMa) |
| **Bottom** | Yellow (#FFF203) | Frames with inference output |

Each lane is 3px tall with 1px gaps. The annotation lane only appears when annotations exist.

### In/Out Markers

**Source:** [frame_scrubber.py:122-318](ui/widgets/frame_scrubber.py#L122-L318) (MarkerOverlay class)

Set a sub-range to process only part of a clip:

| Action | Effect |
|--------|--------|
| Press **I** | Set in-point at current frame |
| Press **O** | Set out-point at current frame |
| Press **Alt+I** | Clear both markers |
| **Drag** marker handle | Adjust in/out position |
| **Middle-click** marker | Reset that marker to boundary |

When markers are set:
- Regions outside the range are dimmed (semi-transparent overlay)
- RUN button changes to "RUN SELECTED"
- Playback loops within the range (if loop enabled)
- Markers are yellow (#FFF203) brackets with 6×8px triangle handles

---

## Parameter Panel (Right Sidebar)

**Source:** [parameter_panel.py](ui/widgets/parameter_panel.py), minimum width 240px

The right sidebar contains all inference and output controls, organized into three sections.

### Alpha Generation

| Control | Description |
|---------|-------------|
| **GVM AUTO** button | Auto-generate alpha hints via GVM model. Available when clip is in RAW state. |
| **VIDEOMAMA** button | Generate alpha from painted annotation masks. Requires frames with brush annotations. |
| **EXPORT MASKS** button | Export annotation brush strokes as binary PNG masks for external VideoMaMa use. |

**Source:** [parameter_panel.py:56-102](ui/widgets/parameter_panel.py#L56-L102)

### Inference Controls

| Control | Type | Range | Default | Description |
|---------|------|-------|---------|-------------|
| **Color Space** | Dropdown | sRGB, Linear | sRGB | Working color space for inference |
| **Despill Strength** | Slider | 0.0 – 1.0 | 1.0 (slider value 10) | Green spill removal intensity. Higher = more aggressive cleanup of green fringing on edges. |
| **Despeckle** | Checkbox + spinner | 50 – 2000 px | ON, 400 px | Morphological cleanup. Removes isolated artifacts smaller than the threshold. |
| **Refiner Scale** | Slider | 0.0 – 3.0 | 1.0 (slider value 10) | Edge refinement pass intensity. 0 = disabled, higher = more cleanup. |
| **Live Preview** | Checkbox | — | OFF | Reprocess current frame when parameters change. |

**Middle-click** any slider to reset it to its default value.

The **Despeckle Advanced** section (Dilation + Blur) is collapsed by default. Click to expand.

**Source:** [parameter_panel.py:111-188](ui/widgets/parameter_panel.py#L111-L188)

### Output Format

Each output channel can be individually enabled/disabled and set to EXR or PNG:

| Output | Default | Default Format | Description |
|--------|---------|---------------|-------------|
| **FG** | ON | EXR | Foreground RGB with green spill removed |
| **Matte** | ON | EXR | Single-channel alpha (grayscale) |
| **Comp** | ON | PNG | Key composited over checkerboard (for review) |
| **Processed** | ON | EXR | Full RGBA premultiplied linear (for VFX compositing) |

**Source:** [parameter_panel.py:191-262](ui/widgets/parameter_panel.py#L191-L262)

---

## Queue Panel (Left Sidebar)

**Source:** [queue_panel.py](ui/widgets/queue_panel.py)

The queue panel shows all pending, active, and completed jobs. It's collapsible — when collapsed, only a 24px-wide vertical "QUEUE" tab is visible. Toggle with the **Q** hotkey.

### Per-Job Display

Each job row (60px height) shows:
- **Job type** — Inference, GVM Auto, VideoMaMa, or Preview
- **Progress bar** — frame-level progress for inference, pulsing for indeterminate jobs (GVM)
- **Status label** with color:

| Status | Color | Display Text |
|--------|-------|-------------|
| QUEUED | Gray (#808070) | "STARTING..." |
| RUNNING | Yellow (#FFF203) | "PROCESSING" |
| COMPLETED | Green (#22C55E) | "DONE" |
| CANCELLED | Gray (#808070) | "CANCELLED" (strikethrough) |
| FAILED | Red (#D10000) | "FAILED" |

- **Cancel button** (×) — cancels a running or queued job

The **Clear** button in the header removes completed and cancelled jobs from the list.

---

## I/O Tray Panel (Bottom)

**Source:** [io_tray_panel.py](ui/widgets/io_tray_panel.py)

A horizontal-scrolling strip at the bottom of the workspace showing clip thumbnails.

### Two Sections

| Tab | Shows | Purpose |
|-----|-------|---------|
| **INPUT** | All loaded clips | Select clips to view/process |
| **EXPORTS** | Only COMPLETE clips | Review finished output |

### Card Content

Each card (130px wide) displays:
- **Thumbnail** (110×62px) — first frame of the clip
- **State badge** (top-right) — colored text matching clip state
- **Clip name** (bold, elided if long)
- **Frame count** — "N frames" or "N frames (video)"

### Card Interaction

| Action | Effect |
|--------|--------|
| **Left-click** | Select clip (loads in preview) |
| **Ctrl+click** | Toggle multi-select |
| **Shift+click** | Range select |
| **Right-click** | Context menu (project options) |

Export cards show a tooltip with the inference settings used (output formats, color space, despill %, refiner %, despeckle settings).

---

## Status Bar (Bottom)

**Source:** [status_bar.py](ui/widgets/status_bar.py), height 44px

### Left Side
- **Progress bar** — 6px tall, 250px max width, shows inference progress
- **Frame counter** — "42 / 1024 frames · 3.2s/frame · ETA 52m"
- **Warning button** (orange) — shows warning count, click to view warning details

### Right Side
- **RUN INFERENCE** button (160×32px, primary CTA)
  - Changes to "RUN SELECTED" when in/out range is set
  - Disabled until clip is READY or COMPLETE
  - Hotkey: **Ctrl+R**
- **RESUME** button — appears only when partial outputs exist from a previous run
- **STOP** button — appears during processing, cancels current job (hotkey: **Esc**)

---

## Brand Bar (Top)

**Source:** [main_window.py:264-299](ui/main_window.py#L264-L299)

The top strip shows:
- **Left:** "CORRIDOR" (yellow #FFF203) + "KEY" (green #2CC350) in Gagarin brand font
- **Right:** GPU name (e.g., "RTX 5090"), VRAM usage bar (80px wide, 8px tall), VRAM text ("4.2 GB / 32.0 GB")

The VRAM meter updates during inference via the GPU monitor worker. Tooltip: "GPU video memory usage — updates during inference."

---

## Preferences

**Source:** [preferences_dialog.py](ui/widgets/preferences_dialog.py), access via Edit > Preferences

| Setting | Default | Key | Description |
|---------|---------|-----|-------------|
| **Show tooltips** | ON | `ui/show_tooltips` | Show helpful tooltips on all controls |
| **UI sounds** | ON | `ui/sounds_enabled` | Play sound effects for actions |
| **Copy source videos** | ON | `project/copy_source_videos` | Copy imported videos into the project folder (OFF = reference in place, saves disk space) |
| **Loop playback** | ON | `playback/loop` | Loop within in/out range during playback |

Settings persist via QSettings (platform-native storage — Windows Registry on Windows).

---

## Sound Effects

**Source:** [audio_manager.py](ui/sounds/audio_manager.py)

The GUI plays contextual sound cues for workflow feedback. Toggle via Ctrl+M or Edit > Preferences.

| Sound | Trigger |
|-------|---------|
| **Hover** | Mouse enters welcome screen buttons |
| **Click** | Any button press (auto-installed on all QPushButtons) |
| **Error** | Job failure or validation error |
| **Frame Extract Done** | Video extraction completes |
| **Mask Done** | VideoMaMa mask generation completes |
| **Inference Done** | Inference job finishes |
| **User Cancel** | Job cancelled |

---

## Keyboard Shortcuts

**Source:** [shortcut_registry.py](ui/widgets/shortcut_registry.py), viewable in-app via Edit > Hotkeys

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

### Timeline

| Shortcut | Action |
|----------|--------|
| **I** | Set in-point marker |
| **O** | Set out-point marker |
| **Alt+I** | Clear in/out range |

### Playback

| Shortcut | Action |
|----------|--------|
| **Space** | Play / Pause |

### Annotation

| Shortcut | Action |
|----------|--------|
| **1** | Green (foreground) annotation brush |
| **2** | Red (background) annotation brush |
| **Ctrl+Z** | Undo last annotation stroke |
| **Ctrl+C** | Clear all annotations |

Shortcuts are rebindable via Edit > Hotkeys. The dialog shows all shortcuts grouped by category, with click-to-rebind buttons and conflict detection.

---

## Status Colors

Clip processing states are color-coded consistently across the entire UI — clip browser, I/O tray, queue panel, and status bar.

| Color | Hex | State | Meaning |
|-------|-----|-------|---------|
| Orange | `#FF8C00` | EXTRACTING | Video being extracted to image sequence |
| Gray | `#808070` | RAW | Frames loaded, no alpha hint generated yet |
| Blue | `#009ADA` | MASKED | User annotation masks painted (ready for VideoMaMa) |
| Yellow | `#FFF203` | READY | Alpha hint available, ready for inference |
| Green | `#22C55E` | COMPLETE | Inference finished, all outputs available |
| Red | `#D10000` | ERROR | Processing failed — can retry |

---

## Typical Workflow

### 1. Import

Drop a video file onto the welcome screen (or File > Import Clips > Import Video).

The video is extracted to an image sequence automatically. You'll see the extraction progress in the status bar and hear a sound when it completes.

### 2. Generate Alpha Hint

Your clip is now in **RAW** state (gray badge). You need an alpha hint before running inference.

**Option A — GVM Auto (one-click):**
Click the **GVM AUTO** button in the parameter panel. GVM generates alpha hints automatically from the input frames. This works for most green screen footage.

**Option B — VideoMaMa (manual masking):**
For difficult shots, use the annotation brush:
1. Press **1** to activate foreground mode (green)
2. Paint over the subject on a few key frames
3. Press **2** to switch to background mode (red)
4. Paint over background areas
5. Click **VIDEOMAMA** in the parameter panel

VideoMaMa uses your painted masks to generate alpha hints that are more accurate for complex footage.

### 3. Run Inference

Your clip is now **READY** (yellow badge). Set your parameters:
- Adjust despill, despeckle, and refiner as needed
- Optionally set in/out markers to process a sub-range
- Click **RUN INFERENCE** (or press **Ctrl+R**)

Watch the progress in the status bar and queue panel. Use the dual viewer to compare input vs output in real-time.

### 4. Review

When complete (green badge), switch between view modes to inspect results:
- **COMP** — see the key over checkerboard
- **FG** — check for green fringing
- **MATTE** — inspect alpha quality
- **PROCESSED** — verify the production RGBA

### 5. Export

Outputs are written to the project's `Output/` subdirectories during inference. For video export, use File > Export Video.

---

## Project Structure

Each imported clip creates a project folder under `Projects/`:

```
Projects/
  260301_093000_Woman_Jumps/
    Source/              # Original video (copied or referenced)
    Frames/              # Extracted image sequence
    AlphaHint/           # Generated alpha hints (GVM or VideoMaMa)
    Output/
      FG/                # Foreground EXR/PNG
      Matte/             # Alpha matte EXR/PNG
      Comp/              # Checkerboard composite PNG
      Processed/         # Production RGBA EXR
    project.json         # Metadata (name, settings, in/out range)
```

**Source:** Project creation in [project.py:52-107](backend/project.py#L52-L107)

---

## Menu Reference

### File
| Item | Shortcut | Action |
|------|----------|--------|
| Import Clips > Import Folder... | — | Open folder browser for image sequences |
| Import Clips > Import Video(s)... | — | Open file browser for video files |
| Save Session | Ctrl+S | Save current workspace state |
| Open Project... | Ctrl+O | Open an existing project folder |
| Export Video... | — | Export composite video with audio |
| Return to Home | Home | Go back to welcome screen |
| Exit | — | Close the application |

### Edit
| Item | Action |
|------|--------|
| Preferences... | Open settings dialog |
| Hotkeys... | View and rebind keyboard shortcuts |
| Export Annotation Masks | Save brush annotations as PNG masks |
| Clear Annotations | Remove all annotation brush strokes |

### View
| Item | Action |
|------|--------|
| Reset Layout | Restore default panel sizes |
| Toggle Queue Panel | Show/hide queue sidebar |
| Reset Zoom | Reset viewer zoom to 100% |

### Help
| Item | Action |
|------|--------|
| About | Application info and version |

**Source:** Menu construction in [main_window.py:211-251](ui/main_window.py#L211-L251)

---

## Theme & Branding

The GUI uses a dark theme built around the Corridor Digital brand identity.

**Primary palette:**
| Role | Color | Hex |
|------|-------|-----|
| Brand accent | Yellow | `#FFF203` |
| Secondary accent | Green | `#2CC350` |
| Main background | Warm black | `#1A1900` |
| Deep background | Near black | `#0E0D00` |
| Card/panel surface | Dark | `#1E1D13` |
| Hover surface | Subtle | `#454430` |
| Border | Warm gray | `#2A2910` |
| Primary text | Light gray | `#E0E0E0` |
| Secondary text | Warm gray | `#CCCCAA` |
| Dimmed text | Gray | `#808070` |
| Link / interactive | Blue | `#009ADA` |
| Error | Red | `#D10000` |
| Warning | Orange | `#FFA500` |

**Fonts:**
- **Gagarin** — brand mark font (used only for "CORRIDOR KEY" in the brand bar)
- **Open Sans** (13px) — all UI text, labels, buttons

**Source:** [corridor_theme.qss](ui/theme/corridor_theme.qss), font loading in [app.py:37-88](ui/app.py#L37-L88)

---

## Background Workers

The GUI runs several background threads to keep the UI responsive:

| Worker | Source | Purpose |
|--------|--------|---------|
| **GPUJobWorker** | [gpu_job_worker.py](ui/workers/gpu_job_worker.py) | Runs inference jobs from the queue. Emits progress per frame, preview every 5th frame. |
| **ExtractWorker** | [extract_worker.py](ui/workers/extract_worker.py) | Extracts video to image sequences via FFmpeg. Multiple concurrent jobs supported. |
| **GPUMonitor** | [gpu_monitor.py](ui/workers/gpu_monitor.py) | Polls VRAM usage every 2 seconds via pynvml. Drives the brand bar VRAM meter. |
| **ThumbnailWorker** | [thumbnail_worker.py](ui/workers/thumbnail_worker.py) | Generates clip thumbnails asynchronously for the browser and I/O tray. |

All workers use Qt signals to communicate with the main thread — no direct UI access from worker threads.

---

## Running the Application

```bash
# GUI mode (default)
python main.py

# CLI mode (original terminal wizard)
python main.py --cli

# Verbose logging
python main.py --log-level DEBUG
```

Logs are written to `logs/backend/YYMMDD_HHMMSS_corridorkey.log` with Eastern Time timestamps.

**Source:** Entry point in [main.py](main.py)

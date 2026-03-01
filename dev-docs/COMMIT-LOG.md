# EZ-CorridorKey Commit Log

Running log of all commits for development history.

---

### 7aa22ee - 2026-03-01 18:29
**Add split RUN/RESUME buttons, draggable markers, middle-click reset, and EXR write fix**

Key changes:
- Replace resume modal dialog with contextual two-button layout (RUN/RESUME)
- Draggable in/out markers via MarkerOverlay with mouse-transparent pass-through
- Middle-click resets parameter sliders and markers to defaults
- Fix EXR write assertion: promote uint8 to float32 before half-float encoding
- Scrubber slider color changed from yellow to gray
- Tooltip forwarding through overlay via event filter
- Debounced marker drag to prevent frame loading flood

---

### d20201f - 2026-03-01
**Add I/O frame markers, coverage bar, project persistence, and UI polish**

Key changes:
- In/Out frame range markers (I/O/Alt+I hotkeys) with project.json persistence
- CoverageBar dual-lane painting (alpha + inference) with dim overlay and yellow brackets
- Frame range-aware inference (sub-clip processing, GVM always full clip)
- backend/project.py for per-clip project.json read/write
- ClipEntry.in_out_range field with InOutRange dataclass
- Clip browser polish: welcome screen, recent projects, ghost frame fix
- Parameter panel and status bar improvements
- 267 tests (expanded from 236)

---

### b9367d7 - 2026-03-01 09:39
**Add video extract pipeline, session persistence, cancel/stop, GVM progress, and brand assets**

---

### b18f30b - 2026-02-28 22:35
**Add Topaz-style welcome screen, brand polish, and transport controls**

---

### c346f7c - 2026-02-28 21:44
**Consolidate duplicated frame I/O into backend/frame_io.py, remove dead imports**

---

### 1b49aa1 - 2026-02-28 21:32
**Add comprehensive debug logging infrastructure**

---

### 1ab67eb - 2026-02-28 21:18
**Add comprehensive backend test suite (77 → 224 tests)**

---

### 8833736 - 2026-02-28 20:42
**Add Phase 4: GPU mutex, output config, live reprocess, session save/load, PyInstaller**

---

### 938008f - 2026-02-28 20:15
**Add preview polish: split view, frame scrubber, view modes, zoom/pan, thumbnails**

---

### 4970885 - 2026-02-28 20:03
**Add PySide6 GUI with 3-panel layout, job queue panel, and GPU worker**

---

### ef8e636 - 2026-02-28 20:03
**Add backend service layer, clip state machine, job queue, and validators**

---

### a29d8b3 - 2026-02-27 23:14
**Rename MaskHint to VideoMamaMaskHint across codebase and folders**

---

### f88fb2d - 2026-02-27 09:45
**Remove unused video from docs**

---

### 1125eb5 - 2026-02-27 01:40
**Update README.md**

---

### b70ae5e - 2026-02-27 01:36
**Update README.md**

---

### 37d2040 - 2026-02-27 09:32
**Change video embed to raw URL to trigger GitHub video player**

---

### 6fe5a81 - 2026-02-27 09:29
**Add demo video to docs directory**

---

### fd4cc32 - 2026-02-27 08:58
**Embed demo video directly into top of README**

---

### f35fffe - 2026-02-26 00:03
**Optimize inference VRAM with FP16 autocast and update README requirements**

---

### 30e147a - 2026-02-25 23:31
**Update README with explicit model download links and new Windows installer instructions**

---

### bc734f6 - 2026-02-25 23:29
**Update README with future training/dataset info and CorridorKey licensing**

---

### 5e5f8dc - 2026-02-25 23:15
**Add licensing and acknowledgements for GVM and VideoMaMa**

---

### 5b2ef1f - 2026-02-25 23:09
**Add HuggingFace model download links to Windows installers**

---

### cec7b85 - 2026-02-25 23:03
**Add Windows Auto-Installer scripts**

---

### c987163 - 2026-02-25 21:55
**Update README with Discord link and rename launcher scripts**

---

### a36ef2b - 2026-02-25 21:00
**Untrack CorridorKey_remote.bat and add to gitignore**

---

### 6a8a33c - 2026-02-25 20:59
**Update gitignore for Ignored empty directories**

---

### 4f6f5bb - 2026-02-25 20:57
**Add .gitkeep to maintain empty project directories**

---

### 0e4bbdc - 2026-02-25 08:39
**Added comprehensive Master README.md**

---

### 06260e1 - 2026-02-25 01:26
**Incorporated user feedback into LLM_HANDOVER.md for greater technical accuracy**

---

### d86ec87 - 2026-02-25 01:07
**Added technical handover document for future LLM assistants**

---

### 10b843c - 2026-02-25 00:31
**Removed lingering PointRend comments**

---

### 0f71aa0 - 2026-02-25 00:29
**Removed dead debug comments from model_transformer.py**

---

### ec6a0c9 - 2026-02-25 00:26
**Removed unused point_rend module from CorridorKeyModule**

---

### 38989bf - 2026-02-25 00:24
**Added true sRGB conversions to color_utils and added refiner scale to wizard**

---

### ee31d86 - 2026-02-25 00:23
**Added refiner strength prompt to wizard**

---

### 0faf09d - 2026-02-25 00:22
**Updated CorridorKeyModule README and removed redundant requirements.txt for open source release**

---

### 418a324 - 2026-02-23 21:48
**Added local Windows and Linux launcher scripts**

---

### 4f1dad6 - 2026-02-22 04:36
**Added luminance-preserving despill, configurable auto-despeckling garbage matte, and checkerboard composite background**

---

### d5559bc - 2026-02-15 06:22
**Initial Commit (Code Only): Smart Wizard, VideoMaMa Integration, Optional GVM**

---

# EZ-CorridorKey Commit Log

Running log of all commits for development history.

---

### 64b1dbd - 2026-03-12 16:28
**Refine SAM2 prompt frames iteratively**

Key changes:
- Change SAM2 point prompting from a single all-at-once `add_new_points_or_box()` call to iterative same-frame refinements in sparse batches (`6` foreground / `2` background prompts per step)
- Keep the prompt box only on the first refinement step and let later calls accumulate against prior `prev_sam_mask_logits`, matching the official SAM2 correction flow more closely
- Preserve full propagation behavior after the prompt frame; the change is isolated to conditioning-frame prompt application
- Add focused tests covering prompt refinement batching and wrapper call sequencing

Observed impact on the real failing clip:
- Frame `0` prompt result improved from `200` foreground components / `342` hole components to `1` foreground component / `0` holes
- Frame `12` prompt result improved from `114` foreground components / `535` hole components to `1` foreground component / `0` holes
- 5-frame smoke test now passes with stable fills around `0.238`

---

### 5f87f66 - 2026-03-12 15:46
**Fix preview display and SAM input color handling**

Key changes:
- Stop forcing extracted video EXR sequences to `Linear` during preview/live-reprocess when the UI says `sRGB`
- Make `PROC` display show the saved premultiplied processed image over black instead of a viewer-only unpremultiply path that made it look washed out
- Align SAM2 preview / Track Mask frame loading with the same input color truth as the viewer, including explicit user overrides
- Add regression coverage across clip default color-space selection, display transform behavior, and preview service paths

User-visible impact:
- live despill/refiner adjustments no longer cause the image to jump to a washed-out color interpretation
- `PROC` no longer looks brighter than the actual processed output contract
- Track Mask preview no longer brightens the frame before SAM2 runs

---

### b1a0a1b - 2026-03-12 15:06
**Make thumbnails match viewer decode**

Key changes:
- Route sequence thumbnails through the same `INPUT` display/decode path as the main viewer instead of a separate EXR gamma path
- Route video thumbnails through the same video decode helper used by the viewer
- Bump thumbnail cache version so stale brightened cached thumbnails are replaced automatically
- Add focused regression coverage for thumbnail decode parity

User-visible impact:
- Input and Export tray cards now show the same color/brightness the large viewer shows

---

### 426609d - 2026-03-12 14:20
**Fix live preview EXR color space override**

Key changes:
- Remove the hidden EXR→Linear override from `reprocess_single_frame()`
- Preserve the user’s explicit `sRGB` or `Linear` choice through live preview reruns
- Add a regression test proving EXR preview respects `input_is_linear=False`

User-visible impact:
- changing despill/refiner with live preview enabled no longer forces extracted-video EXRs into the washed-out Linear interpretation

---

### 70c51f6 - 2026-03-02 13:19
**Queue panel: floating overlay, vertical QUEUE tab, splitter alignment**

---

### a9df37a - 2026-03-02 13:06
**Move queue panel to collapsible left sidebar, UI polish pass**

---

### 9afd225 - 2026-03-02 12:42
**Add hotkeys dialog, remove split view, fix queue panel progress stutter**

---

### 15611e8 - 2026-03-02 10:22
**Add UI sound system, context-aware import, escape cancel, extraction controls**

---

### (uncommitted) - 2026-03-02
**Interactive annotation overlay, VideoMaMa pipeline fixes, and code hygiene cleanup**

- Interactive annotation overlay: green/red brush strokes (hotkeys 1/2), Shift+drag resize, Ctrl+Z undo, mask export to VideoMamaMaskHint
- VideoMaMa pipeline: fixed button enable after mask export (stale asset refs), relaxed guard condition to accept mask_asset
- Annotation markers on timeline scrubber (green lane in CoverageBar, auto-hides when empty)
- Downloaded SVD base model (stable-video-diffusion-img2vid-xt) and renamed VideoMaMa UNet checkpoint
- Added einops to requirements.txt
- Code hygiene: fixed _reset_layout splitter sizes, added empty availability guard, removed dead code and unused imports, hoisted shutil import

---

### 55a9c1e - 2026-03-01 19:31
**Add advanced despeckle controls (dilation/blur) and hide status bar on welcome screen**

---

### 56b451f - 2026-03-01 19:10
**Add play/pause transport button with Space hotkey, loop playback within in/out range**

---

### 81e603a - 2026-03-01 18:54
**Add copy-source preference toggle and deletion safety guard for Projects root**

---

### 73fb3e2 - 2026-03-01 18:50
**Add Preferences dialog (Edit > Preferences) with tooltips toggle via QSettings**

---

### 2270270 - 2026-03-01 18:47
**Enable live output mode switching during inference via FrameIndex rebuild on preview**

---

### 2422772 - 2026-03-01 18:45
**Add alpha coverage feedback: frame counts in status bar, 3-option partial alpha dialog**

---

### 2b2a3ef - 2026-03-01 18:41
**Add export settings tooltip on Exports cards in IO tray from manifest data**

---

### c21b0e9 - 2026-03-01 18:38
**Add dual import: ADD button supports folders or video files, drag-drop accepts videos**

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

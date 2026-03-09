# CorridorKey — Tasks

## Upstream PRs (nikopueringer/CorridorKey)

### PR #40 — Backend Service Layer
- **Status:** Submitted, CI passing (lint + test 3.10 + test 3.13)
- **Branch:** `ez/backend-service-v2`
- **URL:** https://github.com/nikopueringer/CorridorKey/pull/40
- **Waiting on:** Niko's review (getting second pair of eyes)
- **Next:** If changes requested, address feedback and push to same branch

### Bug Fixes PR (not yet submitted)
- **Status:** Waiting for PR #40 feedback first
- **Scope:**
  - [ ] cv2.imwrite() return value checking (silent write failures)
  - [ ] Mask channel normalization (2ch/4ch EXR alpha)
  - [ ] opencv-python-headless note (Qt5 conflict with PySide6)

### Test Suite PR (not yet submitted)
- **Status:** Depends on PR #40 being merged
- **Scope:**
  - [ ] 13 backend test files
  - [ ] conftest.py with shared fixtures
  - [ ] All tests pass with `uv run pytest`

---

## Plug & Play Release

Goal: Users clone, install, run. No fiddling. Windows/Mac/Linux.

### One-Click Installer (DONE)
- [x] `install.bat` — Windows one-click (uv-first, pip fallback, auto CUDA detection)
- [x] `install.sh` — macOS/Linux one-click (MPS for Mac, distro-specific FFmpeg tips)
- [x] `start.bat` / `start.sh` — fast launchers for subsequent runs
- [x] `scripts/setup_models.py` — model downloader (resume, progress, disk space checks)
- [x] `pyproject.toml` — added `[tool.uv] torch-backend = "auto"`
- [x] MPS device detection in `backend/service.py`
- [x] README updated with new install instructions

### Still To Verify
- [ ] Test `install.bat` on clean Windows machine (full flow)
- [ ] Test `install.sh` on macOS (MPS path)
- [ ] Test `install.sh` on Linux (CUDA path)
- [ ] PyInstaller build (corridorkey.spec) — verify frozen build works
- [ ] Welcome screen works with zero projects
- [ ] Drag-and-drop import works immediately
- [ ] Clear error messages if FFmpeg not found
- [ ] Clear error messages if CUDA not available (graceful fallback)

### GUI Repo (EZ-CorridorKey-GUI)
- [ ] Submodule points at correct upstream
- [ ] README install instructions verified
- [ ] CHANGELOG up to date

---

## Features In Progress

### Frame Timing
- [x] Per-frame timing in run_inference() (rolling avg, fps, ETA)
- [x] Progress callback passes fps, elapsed, eta_seconds as kwargs
- [x] DEBUG logging per frame, INFO summary with avg fps
- [ ] UI status bar displays fps and ETA during inference
- [ ] UI status bar displays fps and ETA during GVM/VideoMaMa

---

## Done

- Interactive annotation overlay — green/red brush strokes (hotkeys 1/2) for foreground/background marking, Shift+drag brush resize, Ctrl+Z undo, mask export to VideoMamaMaskHint, VideoMaMa pipeline integration, annotation markers on timeline scrubber (green lane)
- In/Out trim points per clip — I/O/Alt+I hotkeys, visual brackets, project.json persistence, frame range-aware inference
- Tooltips on all interactive controls — parameter panel, status bar, scrubber, clip browser, queue, view modes, GPU info
- Cancel/stop inference — signal chain + confirmation dialog
- GVM real progress bar with percentage and ETA (no more bouncing bar)
- Collapsible left clip browser sidebar
- Escape key no longer accidentally cancels during modal dialogs
- Middle-click resets to default — parameter sliders (despill, refiner, despeckle size) + in/out markers reset to boundaries
- WATCH button removed from clip browser sidebar
- Coverage bar aligned with slider + draggable in/out markers
- RUN/RESUME split buttons — contextual two-button layout replaces resume modal dialog
- Squished parameter fields — fixed layout widths so Color Space and Despeckle display properly
- Collapsed sidebar — floating chevron nub, 0px width when collapsed, full space reclaimed
- Cancel shows "Canceled" not "Failed" — separated cancel vs error signal paths
- ADD button supports folders or files — QMenu choice: "Import Folder..." or "Import Video(s)...", drag-drop also accepts video files
- Export settings tooltip — hover over Exports cards in IO tray shows manifest data
- Post-inference side-by-side scrub — auto-COMP switch, synced scrubbing, mode switching
- Alpha coverage feedback — status bar shows "X/Y alpha frames" after GVM/VideoMaMa
- Live output mode switching during inference — FrameIndex rebuilds on each preview update
- Welcome screen multi-select — QFileDialog.getOpenFileNames() handles Ctrl/Shift/Ctrl+A
- Preferences dialog (Edit > Preferences) — QSettings-based, tooltips toggle, copy source toggle
- Deletion safety — rmtree guarded by Projects root check

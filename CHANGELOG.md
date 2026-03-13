# Changelog

All notable changes to EZ-CorridorKey are documented here.

---

## [1.6.0] - 2026-03-13 — Parallel Inference, MatAnyone2, MLX Apple Silicon, macOS Support, 

### Parallel Inference
- **Multi-engine frame processing!** — inference engine pool processes multiple frames concurrently when VRAM allows, with user-configurable pool sizing based on available memory. 1.8-2x speed increase clocked on 4K footage via RTX 5090 at 4 parallel frames. Experimental up to 8, use at your own discretion if you have VRAM headroom.

### MatAnyone2 Integration
- **Fourth alpha generator** — MatAnyone2 joins GVM, VideoMaMa, and SAM2 as a one-click alpha hint generator. Make sure to paint a frame using hotkeys 1/2 on frame 1 first. Checkpoint auto-downloads from GitHub releases on first use (~390 MB)
- **Performance hardening** — robust Hydra config initialization, graceful error handling, and progress reporting during MatAnyone2 inference

### MLX Apple Silicon Acceleration
- **Native Apple Silicon backend** — on Macs with Apple Silicon and `corridorkey-mlx` installed, CorridorKey inference uses MLX instead of PyTorch MPS for 1.5–2× faster keying
- **Auto-detection** — backend resolves automatically: Apple Silicon + corridorkey-mlx + `.safetensors` checkpoint → MLX, otherwise Torch. Override with `CORRIDORKEY_BACKEND=torch|mlx|auto`
- **Output normalization** — adapter converts MLX uint8 output to float32 0–1 matching the Torch engine contract; despill/despeckle post-processing applied identically
- **Installer support** — `1-install.sh` and `3-update.sh` automatically install `.[mlx]` extras on Apple Silicon Macs

### macOS Support
- **Cross-platform FFmpeg** — macOS FFmpeg install via Homebrew, file manager opens via `open` instead of `explorer`
- **MPS performance warning** — one-time-per-session dialog warns Mac users before launching GPU-heavy features (SAM2, GVM, VideoMaMa, MatAnyone2) that may be extremely slow on MPS, recommending imported alpha mattes instead

### Installer Hardening
- **Managed runtime verification** — install now verifies the actual torch runtime before reporting success, with clearer diagnostics when Python or accelerator selection is wrong
- **macOS polish** — Apple Silicon installs automatically include MLX extras; macOS torch verification skips unnecessary CUDA probing; Homebrew FFmpeg install path now works cleanly in non-interactive runs
- **Windows/Linux robustness** — Windows CUDA detection smoke was hardened for locale/codepage issues; fresh installs now resolve the correct torch backend more reliably
- **Upgrade cleanup** — updater removes lingering legacy `pynvml` installs and suppresses the related non-actionable deprecation warning for older upgraded environments

### Added
- **ALPHA view mode** — preview the raw AlphaHint input before running inference
- **MASK view mode** — preview SAM2-tracked masks before feeding them to alpha generators
- **Clear Mask / Clear All** — right-click context menu on clips to clear tracked masks or all generated data
- **Global drag-and-drop** — drop video files anywhere on the main window to import them
- **Frame scrubber improvements** — scroll wheel scrubbing and larger slider hitbox for easier frame navigation
- **Live clip progress** — selected clip's progress bar updates in real time during inference jobs

### Fixed
- **Live preview EXR color-space override** — `PREVIEW_REPROCESS` no longer silently reinterprets extracted video EXR sequences as Linear on every rerun. When the user leaves the clip in `sRGB`, live preview now stays in `sRGB`; when the user explicitly switches to `Linear`, the preview follows that override
- **`Processed` viewer contract** — `PROC` preview now reflects the premultiplied RGBA data actually written on disk instead of unpremultiplying it in the viewer. `PROC` shows over black, `COMP` remains checkerboard-backed
- **Thumbnail decode mismatch** — clip thumbnails now use the same input decode path as the main viewer
- **SAM2 preview input color truth** — Track Mask preview and full SAM2 tracking load EXR frames with the same source-truth color interpretation as the viewer
- **SAM2 prompt-frame quality** — prompt points now applied as smaller same-frame refinements instead of one bulk call, eliminating boxy/hole-ridden prompt-frame masks
- **SAM2 Hydra reinitialization** — Hydra global state cleared before each tracker build, preventing config collision errors on repeated runs
- **ProRes 10-bit extraction** — fixed frame extraction failing on FFmpeg 8.x with 10-bit ProRes sources
- **In-out alpha alignment** — corrected frame alignment between input and alpha hint sequences during inference
- **Update button text** — fixed garbled button text when window is resized narrow
- **Clear Alpha state** — clearing alpha now properly refreshes the viewer, disables the ALPHA button and coverage bar
- **Windows EcoQoS throttling** — disabled Windows EcoQoS power policy that was silently throttling GPU inference in background
- **UI paint clipping** — fixed paint brush strokes clipping at widget boundaries, tooltip wrapping, and mask-ready state check
- **Report Issue version** — Report Issue dialog now reads app version dynamically instead of hardcoded string

### Changed
- **Azure GPU CI** — added installer smoke test matrix on Azure GPU runners
- **`pynvml` → `nvidia-ml-py`** — switched to canonical PyPI package name for NVIDIA Management Library

### Verification
- Regression tests for EXR live-preview color-space override, `PROC` display transform, thumbnail decode, and SAM2 prompt refinement batching
- macOS CUDA-skip test for `verify_torch_runtime.py`
- Paint prompt tests, split view tests, status bar tests, Windows CUDA detection tests
- Fresh install smoke passed on Windows, macOS, and Ubuntu
- Real upgrade smokes passed for `v1.5.0 -> 1.6.0` and `v1.5.2 -> 1.6.0`

---

## [1.5.1] - 2026-03-11 — Hotfix: FFmpeg Filter, Alpha Video Import

### Fixed
- **FFmpeg extraction broken on standard builds** — `in_primaries` and `in_transfer` are not valid FFmpeg scale filter options; removed them (only `in_color_matrix` and `in_range` are needed for YUV→RGB conversion)
- **FFmpeg startup check** — app now validates FFmpeg availability at launch and shows a clear diagnostic dialog if missing or outdated (v6.x users see fix instructions instead of cryptic extraction errors)
- **Reverted broken v1.4.0 updater auto-patch** that was shipped by mistake

### Added
- **Alpha hint video import** — users can now import ProRes MOV, MP4, and other video files as alpha hints (previously only image sequences were accepted)
- **Auto-detect companion alpha hint** — place `input_alphahint.mov` next to `input.mov` and it's automatically extracted to `AlphaHint/` as grayscale PNGs during source extraction — zero extra clicks
- **`--opt-mode` CLI flag** — override GPU optimization mode (`speed`, `lowvram`, `auto`) from the command line

---

## [1.5.0] - 2026-03-10 — SAM2 Tracking, FFmpeg Hardening, Quality Verification

### SAM2 Tracking Integration
- **Annotations are now prompts, not final masks** — painting foreground/background strokes feeds into SAM2 dense tracking before VideoMaMa, producing far better alpha hints than raw brush strokes
- **TRACK MASK button** — generates dense per-frame segmentation masks from sparse annotation prompts using SAM2
- **VideoMaMa pipeline updated** — now expects SAM2-tracked masks (chunking, VRAM, dtype defaults corrected toward official usage)
- **Tracker model selection** — Preferences > Tracking lets you choose Fast (184 MB), Base+ (324 MB, default), or Highest Quality (898 MB) SAM2 models
- **SAM2 pre-download** — installer optionally pre-downloads the default Base+ checkpoint (324 MB) during install

### FFmpeg Hardening
- **Validation requires both ffmpeg and ffprobe** — rejects installs missing ffprobe
- **Version >= 7 enforced** — rejects older FFmpeg builds at both install and runtime
- **Windows essentials build rejected** — Gyan "essentials" builds lack required filters; validation detects and blocks them
- **Bundled local FFmpeg preferred** — `tools/ffmpeg/bin` is checked first, before PATH, preventing stale system installs from interfering
- **Repair FFmpeg in Preferences** — Windows: one-click download of full BtbN GPL build into `tools/ffmpeg`. macOS/Linux: shows exact platform-specific install commands and copies to clipboard (no system mutation)
- **Open FFmpeg Folder** — quick access to the bundled FFmpeg directory from Preferences

### Cross-Platform Smoke CI
- **GitHub Actions workflow** — fresh install + headless startup smoke on Windows, Ubuntu, and macOS
- **Covers**: installer, dependency resolution, SAM2 pre-download, QApplication creation, MainWindow construction, Preferences dialog, FFmpeg repair button
- **Ubuntu runtime libs** — installs libegl1, libgl1, libpulse0, libxkbcommon-x11-0, libxcb-cursor0
- **Node 24 compatible** — uses actions/checkout@v6 and actions/setup-python@v6

### Quality Verification
- **Multi-metric upstream comparison** — same frame run through upstream CorridorKey and our fork, verified across PSNR (80–90 dB), SSIM (0.999996+), MS-SSIM (0.999999+), LPIPS (0.000001), and DeltaE 2000 (0.015 mean on skin) — all below float32 noise floor
- **Branded HTML report** — `scripts/compare_upstream_d2.py` generates a dated, versioned comparison PNG via Playwright with side-by-side composite and alpha matte panels

### Updater Bridge (master → main)
- **`3-update.bat` / `3-update.sh`** — automatically migrates local `master` checkouts to `main`, repoints tracking if needed
- **`origin/master` compatibility mirror** — kept in sync with `main` for the next two releases so older installs can update gracefully. Planned retirement: after `1.5.1`

### Installer Parity
- **`1-install.sh`** — added Python 3.14+ upper-bound check (was already in `.bat`)

### Fixed
- **UI sound clicks** — 50ms fade-in/out applied to all UI sound effects to eliminate playback pops
- **Windows updater crash** — `3-update.bat` restructured to avoid cmd.exe label-inside-parenthesized-block bug; replaced `::` comments with `REM`, removed UTF-8 em dashes
- **FFmpeg version regex on Windows** — `ffmpeg.exe version ...` now parsed correctly (was failing on `.exe` suffix)
- **VirusTotal security link** — README now links independent 0-detection scan for `1-install.bat`

---

## [1.4.0] - 2026-03-09 — Full Pipeline, Model Handoff, Force-Stop, Installer Overhaul

### Workflow Improvements
- **Import Images as sequence or folders** — images are now able to be imported! PNG, JPG/JPEG, EXR, TIF/TIFF, BMP, DPX
- **Alpha hint re-import** — importing alpha hints on a clip that already has them now asks to overwrite (previously showed "Already Imported")

### Model Residency & GPU Handoff
- **`torch._dynamo.reset()` after model switch** — clears stale Triton compilation cache when switching between GVM/VideoMaMa/inference models, fixing the hang where `torch.compile` would deadlock on cached CUDA state from the previous diffusion pipeline
- **pynvml VRAM detection** — `_get_vram_gb()` now uses NVML (driver-level, no CUDA context calls) instead of `torch.cuda.get_device_properties()` which stalled after GVM teardown. Falls back to torch.cuda if pynvml unavailable
- **Granular model-switch status callbacks** — status bar shows each phase: "Switching GVM → inference...", "Offloading GVM...", "Releasing Python references...", "Waiting for CUDA to finish...", "Clearing CUDA cache..."
- **Diagnostic logging in `_load_model()`** — force-flushed `_diag()` messages at every step (GreenFormer init, model.to(device), checkpoint load, state_dict, hiera patch, torch.compile) for hang diagnosis

### Force-Stop Escape Hatch
- **Double-click STOP to force-stop** — first STOP sends cooperative cancel; if the GPU job is stuck (e.g. in a non-interruptible CUDA operation), clicking STOP again triggers a force-stop that auto-saves the session and relaunches the app
- **Status bar FORCE STOP mode** — after first stop attempt, button changes to "FORCE STOP" with warning tooltip
- **Improved stop handling** — cooperative cancel now propagates through model load and CUDA teardown phases, not just frame processing

### Installer (`1-install.bat`)
- **Auto-install FFmpeg** — downloads from gyan.dev, extracts to `tools/ffmpeg/`, adds to PATH. No more manual FFmpeg install needed
- **VS Build Tools auto-install** — detects MSVC via `cl.exe` and `vswhere.exe`, offers auto-install via winget if missing (needed for OpenEXR compilation). Fully unattended — no user interaction required in the VS installer window
- **Python 3.10–3.13 supported** — installer accepts 3.10–3.13, blocks 3.14+ (PyTorch lacks wheels)
- **Git auto-install** — offers to install Git via winget, then links the extracted folder to the upstream repo for `git pull` updates
- **Desktop shortcut** — optional shortcut creation (no console window)

### UI Polish
- **Toast notifications** — animated overlay messages for transient events (alpha import complete, foreground color change, stop-while-chunking, etc.)
- **Update checker** — background thread checks for new versions on GitHub at startup
- **Despeckle spinbox arrows** — custom SVG up/down arrows (`arrow-up.svg`, `arrow-down.svg`) with themed hover/press states for QSpinBox and QDoubleSpinBox controls
- **QSS spinbox styling** — full restyle of spin box up/down buttons with proper hit areas, hover feedback, and brand-consistent sizing

### Fixed
- **GVM → inference hang (root cause)** — `torch.compile` deadlocked on stale Triton/dynamo state from GVM's diffusion pipeline. `torch._dynamo.reset()` in `_ensure_model()` clears the cache before loading the next model. Compile stays enabled in all modes (speed, lowvram, auto) — no performance loss
- **VRAM probe stall during model switch** — `torch.cuda.get_device_properties()` blocked after GVM teardown; replaced with pynvml (NVML driver API)
- **Auto mode resolved to speed but kept compile enabled** — explicit `speed` mode disabled compile (TEMP workaround) but `auto` on ≥12GB cards still ran compile without the dynamo reset. Now all paths are safe
- **FFmpeg bt470bg color space extraction failure** — Photo-JPEG `.mov` files (e.g. Apple cameras) report `color_space: bt470bg` which FFmpeg's scale filter doesn't accept. Built comprehensive mapping tables for matrix (`bt470bg→bt601`), primaries, and transfer (`bt470bg→gamma28`) identifiers. Unknown values now log a WARNING with the exact mapping file/line to update, so future color spaces are caught immediately instead of silently failing
- **EXR colour conversion accuracy** — replaced `setparams` filter with explicit `scale` filter providing input colour metadata (`in_color_matrix`, `in_primaries`, `in_transfer`, `in_range`) directly to swscale. Fixes washed-out or shifted colours on files with incomplete/missing colour tags (e.g. ProRes `.mov` from cameras)
- **Colour metadata fallbacks** — resolution-aware heuristics for missing matrix (BT.709/BT.601/BT.2020), primaries, transfer, and range. SD PAL (576p), NTSC (480p), HD, and BT.2020 sources all get correct defaults
- **Deleted clips reappear on import** — removing clips from the I/O tray was UI-only; adding a new video triggered `os.listdir()` rescan that rediscovered all folders. Now persists removals to `removed_clips` list in `project.json`, filtered during `scan_project_clips()`. Re-importing a removed clip clears it from the list
- **Clip identity mismatch** — `clip.name` was mutable (overwritten by `display_name`), causing removal tracking to fail. Added stable `folder_name` property using `os.path.basename(root_path)`
- **Corrupt project.json on removal** — `add_removed_clip()` now guards against missing `project.json` (returns early with warning instead of creating partial file)
- **Extraction retry fails on ERROR clips** — retrying extraction only removed the `.dwab_done` marker, leaving partial EXR frames that triggered resume logic. Now wipes the entire Frames/ directory on retry for a clean start
- **FFmpeg hardware decode fallback** — if NVDEC fails (e.g. unsupported codec in `.mov`), automatically retries with software decode instead of failing permanently
- **Re-import of removed clips blocked** — "Already Imported" dialog appeared for clips the user had removed. Duplicate check now skips removed clips; re-importing restores the original clip folder instead of creating duplicates
- **Despeckle spinbox arrows invisible** — default Qt arrows had no contrast against dark theme; replaced with explicit SVG icons
- **Installer winget quoting** — VS Build Tools `--override` flag now works correctly on all Windows versions (temp batch file avoids nested-quote escaping)

### Added
- **Extraction diagnostics** — `video_metadata.json` now includes `exr_vf` (exact FFmpeg filter used) and `source_probe` (raw colour metadata from ffprobe) for debugging colour issues
- **Version in About dialog** — Help > About now shows app version via `importlib.metadata`
- **`removed_clips` persistence** — new `get_removed_clips()`, `add_removed_clip()`, `clear_removed_clip()` functions in `backend/project.py`
- **Import Alpha documentation** — README now documents Option C (bring your own alpha mattes) with supported formats

### Installer & Scripts
- **`2-start.bat`** — adds local `tools/ffmpeg/bin` to PATH before launch
- **`3-update.bat` / `3-update.sh`** — migrates older git installs from local `master` checkouts to `main`, repoints `main` checkouts still tracking `origin/master`, and still preserves ZIP fallback for non-git installs
- **Temporary `master` compatibility mirror** — `origin/master` remains a compatibility branch for the next two releases so older installs can keep using their existing updater before being bridged onto `main`. Planned retirement: after `1.5.1`
- **`pyproject.toml`** — `requires-python` updated to `">=3.10,<3.14"`
- **`.gitignore`** — added `tools/` to prevent ffmpeg binary from being committed

### Debug Infrastructure
- **Debug log bible updated** — added GVM→Inference handoff chain, 3 bare `except:` clauses catalogued, 7 silent failures identified, 6 unlogged state initializations documented
- **Force-flush diagnostics** — `_load_model()` uses `_diag()` that flushes all log handlers immediately, ensuring last message is visible even on hang

### Tests
- 10 new tests in `TestRemovedClips` — empty default, missing JSON, add/clear, idempotency, scan filtering, sorted deterministic, re-import clears removed
- 8 new tests in `test_ffmpeg_tools.py` — probe colour metadata, `build_exr_vf` for RGB/YUV/SD/BT.2020, extract filter chain, metadata roundtrip

---

## [1.3.1] - 2026-03-09 — Low-VRAM compile fix, status callbacks

### Fixed
- **Low-VRAM mode hang on first frame** — `@torch.compiler.disable(recursive=False)` allowed Dynamo to re-enter the tile loop, causing O(N) graph breaks per tile. Changed to `@torch.compiler.disable` (recursive=True) so the tiled scheduler is a true eager island. First-frame compile now takes ~15s instead of hanging indefinitely.
- **Tile kernel compile** — added `fullgraph=True` for the pure CNN tile kernel (no graph breaks needed)
- **`total_mem` AttributeError** — corrected to `total_memory` in both inference engine and GPU monitor (PyTorch API)

### Added
- **Status callbacks** — `on_status` parameter on `run_inference()` for phase labels ("Loading model...", "Compiling...") piped to UI status bar via `status_update` signal

---

## [1.3.0] - 2026-03-09 — 2x Faster Inference, Low-VRAM Support

### Performance (4K: 3.3s → 1.5-1.8s per frame)
- **Hiera FlashAttention patch** — fixes timm's 5D non-contiguous Q/K/V tensors in 18 global attention blocks, enabling SDPA FlashAttention instead of O(N²) math fallback (credit: Jhe Kimchi)
- **TF32 tensor cores** — `torch.set_float32_matmul_precision('high')` for ~15% throughput boost on Ampere+ GPUs (no-op on older hardware)
- **torch.compile** — JIT compilation via Triton inductor backend with `fullgraph=False` for graph break support
- **Quality verified** — 157+ dB PSNR across all optimization levels (mathematically identical output)

### Low-VRAM Support (8GB GPUs)
- **Tiled CNN refiner** — 512×512 tiles with 128px overlap (> 65px receptive field = lossless blending) (credit: Marclie)
- **Boundary-aware blending** — tile edges at image boundaries keep full weight, only internal overlaps get ramped
- **Selective torch.compile** — `@torch.compiler.disable` excludes tile scheduler from Dynamo; tile CNN compiled separately with `fullgraph=True`
- **VRAM auto-detection** — ≥12GB uses speed mode (full compile), <12GB uses tiled mode
- **cuDNN benchmark disabled** — saves 2-5 GB workspace memory
- **Manual override** — `CORRIDORKEY_OPT_MODE=speed|lowvram|auto` environment variable

### Fixes
- **Zombie process on exit** — `os._exit()` kills Triton background threads that prevented clean shutdown (was holding ~5.5GB VRAM after window close)
- **Debug console** — F12 console now closes with main window
- **Batch launcher** — `2-start.bat` exits immediately after spawning app
- **GPU flush** — `gc.collect()` + `torch.cuda.synchronize()` before `empty_cache()` in model unload
- **Console window spam** — Triton compilation no longer opens flurry of terminal windows on Windows

### Dependencies
- Added `triton-windows>=3.5,<4` (Windows-only, enables torch.compile)
- Added `OpenEXR>=3.4` to requirements.txt (was in pyproject.toml only)

### Quality Scripts
- `scripts/benchmark_quality.py` — compare 4 optimization levels on same frame (PSNR/MAE/max diff)
- `scripts/compare_quality.py` — A/B comparison between two alpha output directories

---

## [1.2.4] - 2026-03-08 — In/Out State Fix, Extraction Retry

### In/Out Marker State Fix
- Setting or clearing in/out markers now re-evaluates clip state immediately
- Previously, partial alpha + in/out range wouldn't promote RAW to READY until app restart
- RUN SELECTED button now enables as soon as in/out covers the alpha'd range

### Extraction Retry Fix
- Re-running extraction after an error now works correctly
- Clears stale `.dwab_done` marker so `extract_frames()` doesn't skip the actual work
- Resets clip state from ERROR to EXTRACTING before re-queueing

---

## [1.2.3] - 2026-03-08 — Cross-Platform FFmpeg Discovery

- FFmpeg/FFprobe discovery now checks common macOS/Linux install paths (`/opt/homebrew/bin`, `/usr/local/bin`, etc.) — fixes "FFmpeg not found" when GUI is launched from Finder/Dock where `~/.zshrc` PATH isn't loaded

---

## [1.2.2] - 2026-03-08 — macOS Installer Fix

- Fixed `1-install.sh` failing on macOS due to Bash 4+ syntax (`${var,,}`) — macOS ships with Bash 3.2

---

## [1.2.1] - 2026-03-08 — Partial Alpha + In/Out Range Support

### Partial Alpha Hint
- Clips with partial alpha hints now promote to READY if in/out markers are set and alpha covers the in/out range
- Previously, alpha had to cover ALL input frames regardless of in/out markers — Run Inference stayed greyed out
- Without in/out markers, full-clip alpha coverage is still required (unchanged behavior)

---

## [1.2.0] - 2026-03-08 — Import Alpha, Export UX

### Import Alpha Hint
- New **IMPORT ALPHA** button in the Alpha Generation panel — import your own alpha hint images directly from the GUI
- Supports PNG, JPG, TIFF, and EXR input (non-PNG auto-converted to grayscale PNG)
- Files are automatically renamed to match input frame stems for correct 1:1 index matching
- Natural/numeric sort handles any zero-padding scheme (e.g. `alpha_1.png` through `alpha_10.png` sorts correctly)
- Frame count mismatch warning with option to proceed with partial pairing
- Clip auto-advances to READY state after import — no restart needed

---

## [1.1.3] - 2026-03-08 — VideoMaMa UX, Export Improvements

### VideoMaMa Status Feedback
- Status bar now shows real-time phase updates during VideoMaMa: Loading model, Loading frames, Loading masks, Running inference
- Eliminated silent 6+ minute gap where users had no feedback — each phase is now visible
- Added "Running inference (chunk N/M)..." status before GPU inference begins
- Cancel checks added during frame and mask loading phases (not just between chunks)

### Alpha Generation Panel
- Added "— or —" divider between GVM AUTO and VIDEOMAMA buttons for clarity
- VideoMaMa tooltip now references hotkeys (1 for green/foreground, 2 for red/background)
- Tooltip includes annotation strategy tip: annotate keyframes where subject changes shape

### Clear Annotations Fix
- Clearing annotations now removes exported `VideoMamaMaskHint/` directory and disables VideoMaMa button
- Previously, clearing annotations left stale masks on disk, allowing VideoMaMa to run on outdated data

### Export Card Improvements
- Folder icon on export thumbnails — click to open output folder in Explorer
- Double-click export card to load clip in the viewer
- Right-click context menu lists each output type (Comp, FG, BG, etc.) as separate export options
- Video exports now default to `_EXPORTS/` subdirectory in each clip's project folder
- Export filenames include output type (e.g. `MyClip_Comp_export.mp4`)

### VideoMaMa Cancel Toast
- Cancel notification now appears centered on screen instead of bottom-left

---

## [1.1.2] - 2026-03-03 — In-App Issue Reporter, UI Polish

### In-App Issue Reporter
- **Help > Report Issue...** opens a dialog to file GitHub issues directly from the app
- Auto-collects system info: app version, OS, Python, PyTorch, CUDA, GPU, VRAM, display resolution
- Recent WARNING/ERROR log lines (up to 20) included automatically
- Full report copied to clipboard before opening browser — survives GitHub login redirects
- Notice informs users a free GitHub account is required

### UI Polish
- Fixed Python's `platform.platform()` misreporting Windows 11 as Windows 10

---

## [1.1.1] - 2026-03-03 — Pipeline In/Out Fix, Reset I/O Button

### Pipeline Fix
- **Batch pipeline now respects per-clip in/out markers** — inference jobs use each clip's in/out range instead of always processing the full clip
- Fixed in all three batch paths: RUN PIPELINE (Phase 3 direct + auto-chain after alpha), and Ctrl+Shift+R (Run All Ready)
- Alpha generation (GVM / VideoMaMa) still always processes the full clip — only inference is scoped to in/out range

### Reset I/O Button
- New **RESET I/O** button in the I/O tray header (next to + ADD)
- Clears in/out markers on all clips at once — reverts to full-clip processing
- Double confirmation required: "Continue?" then "Are you sure? Cannot be undone."
- Shows count of affected clips; disabled message if no markers are set

---

## [1.1.0] - 2026-03-03 — EXR DWAB Half-Float Extraction

### Frame Extraction Overhaul
- Video frames now extracted as **EXR half-float** instead of PNG — preserves full floating-point precision from the video decoder, eliminating 8-bit quantization and banding
- Two-pass pipeline: FFmpeg extracts to EXR ZIP16, then a recompression pass converts to **DWAB** (VFX-standard lossy compression, ~4× smaller than ZIP)
- Even 8-bit source video benefits: FFmpeg's internal YUV→RGB conversion stays in float, avoiding rounding errors from integer pipelines
- Hardware-accelerated decode (NVDEC/DXVA2) with automatic fallback to software decode
- DWAB recompression runs in a **separate subprocess** — zero GIL contention, UI stays fully responsive

### UI Performance Fixes
- **Throttled progress signals**: Extraction progress emitted at most every 100ms (was per-frame ~500Hz), preventing main thread saturation
- **Cached thumbnails**: `ThumbnailCanvas` scales thumbnails once on load, not on every repaint — eliminates repeated `Qt.SmoothTransformation` during progress updates
- **Smart data-change handling**: Progress-only updates trigger lightweight `update()` instead of full `_rebuild()` with thumbnail rescaling

### Display & Theme
- **EXR input display fix**: Skip gamma correction and Reinhard tone mapping for INPUT mode frames (FFmpeg EXR output is sRGB-range float, not HDR linear)
- **Opaque context menus**: Fixed QSS `background` vs `background-color` causing transparent right-click menus

### GVM Compatibility
- `ImageSequenceReader` now filters files by image extension — `.dwab_done` marker file and other non-image files no longer cause "cannot identify image file" errors

### Version
- Bumped to 1.1.0

---

## [1.0.0] - 2026-03-03 — Batch Pipeline, Annotation Persistence, Installer

### Batch Pipeline (`fd35073`)
- **RUN PIPELINE** button appears when multiple clips are selected in the I/O tray
- Automatic per-clip route classification via `PipelineRoute` enum:
  - **RAW + no annotations** → GVM Auto → Inference
  - **RAW + annotations** → Export masks → VideoMaMa → Inference
  - **MASKED** → VideoMaMa → Inference
  - **READY / COMPLETE** → Inference only
  - **EXTRACTING / ERROR** → Skip
- Phase 0 (CPU): Headless mask export for annotated clips (`export_masks_headless()`)
- Phase 1–3 (GPU): Jobs queued in dependency order — alpha generation first, then auto-chain inference on completion
- Fully cancellable (Esc) and checkpointable — interrupted runs resume where each clip left off
- Extraction done sound plays only after the last batch clip finishes

### Annotation Persistence (`a81ee1f`)
- Annotation strokes now saved to `annotations.json` per clip and persist across app restarts
- Strokes restored on clip load — no need to re-annotate after closing the app

### Video Import & Extraction
- Dual import: ADD button supports folders (image sequences) or video files; drag-drop accepts videos
- Video extraction pipeline with FFmpeg: progress tracking, cancel support, resume detection
- Metadata sidecar (`video_metadata.json`) for stitching back later

### Viewer & Playback
- Cursor-centered zoom (Ctrl+scroll), shift+scroll horizontal pan (`25019f9`)
- Play/pause transport (Space hotkey) with loop playback within in/out range (`56b451f`)
- Live output mode switching during inference — `FrameIndex` rebuilds on the fly (`2270270`)
- Draggable in/out markers, split RUN/RESUME buttons, middle-click slider reset (`7aa22ee`)
- Alpha coverage feedback: frame counts in status bar, 3-option partial alpha dialog (`2422772`)

### Annotation Brush
- Cycle foreground color with **C** key (green / blue) (`c712a53`)

### Preferences
- Preferences dialog (Edit > Preferences) with tooltips toggle (`73fb3e2`)
- Copy-source preference: copy imported videos into project folder or reference in-place (`81e603a`)
- Deletion safety guard prevents removing the Projects root itself

### Installer & Packaging
- One-click installers (`1-install.bat` / `1-install.sh`) with auto GPU detection (`f5b2892`)
- Update scripts (`3-update.bat` / `3-update.sh`) for easy bug fix delivery (`97023be`)
- Desktop shortcut creation during install (no console window) (`98f7b63`)
- Skip GVM/VideoMaMa download on macOS (CUDA-only models) (`c2707ed`)
- Download SVD base model alongside VideoMaMa weights (`b4aa6a4`)
- Fix VideoMaMa unet folder rename after download (`7fc3dd1`)

### Debug & Logging
- Debug console captures all logs from session start (`32001ae`)
- VideoMaMa VAE decode logging with per-chunk timing (`06bbbbf`)
- System local time for log timestamps instead of hardcoded Eastern timezone (`77c6acb`)

### Fixes
- GVM unet rename bug, enable GVM button after extraction (`534b4d9`)
- Progress bar, queue panel, extraction cancel fixes (`8ac9530`)
- Subtler volume slider — thin groove, white handle (`477f81b`)
- Hide redundant clip info label on right viewport (`ebafdc8`)
- Cancel sound debounce, skip previews during annotation mode (`202b76d`)
- Crisp app icon from official logo SVG rendered at 1024px (`57977da`)

---

## [0.1.0] - 2026-03-02 — Release Prep

### Release Packaging
- Added `pyproject.toml` for uv/pip editable install support
- Added 9 UI sound effect WAV files (click, hover, error, done, cancel)
- Added `dev-docs/USER-GUIDE.md` with comprehensive feature documentation
- Updated `.gitignore` for dev-only artifacts
- Removed obsolete dev-docs (branding prompts, clip pipeline, LLM handover)
- Removed all `_BACKUPS/` directories

### UI Sound System (`15611e8`, `7c815af`, `9060279`, `840f4b3`)
- Audio feedback for UI interactions: click, hover, error, inference done, mask done, cancel
- `audio_manager.py` with debounced playback (200ms) to prevent double-fire
- Context-aware import: ADD button distinguishes folders vs video files
- Escape key cancels active extraction or inference job

### Queue Panel Overhaul (`a9df37a`, `70c51f6`)
- Moved queue to collapsible left sidebar with vertical "QUEUE" tab
- Floating overlay style with semi-transparent background
- Per-job progress bars with status color coding
- Splitter alignment with clip browser

### Hotkeys Dialog & Keyboard System (`9afd225`, `b699071`, `b089492`)
- New hotkeys dialog (Help > Keyboard Shortcuts) showing all bindings
- Removed split view (Ctrl+D) — dual viewer is now always-on
- Fixed `QKeyCombination` import for PySide6 compatibility
- Fixed queue panel progress bar stutter during active jobs

### Code Quality (`9a46c59`, `d7df7de`, `e420603`, `8448532`, `9381c09`)
- Deleted 3 dead files (clip_browser.py, clip_card.py, preview/natural_sort.py)
- Removed 9 dead functions from UI layer
- Removed unused imports across 12 files
- Unified image/video extension constants in `backend/project.py`
- Synced `backend/__init__.py` exports with actual module contents

### Interactive Annotation Overlay
- Green/red brush strokes (hotkeys 1/2) for VideoMaMa mask painting
- Shift+drag to resize brush, Ctrl+Z undo, mask export to VideoMamaMaskHint
- Annotation markers on timeline scrubber (green lane, auto-hides when empty)

---

## 2026-02-28 — Frame I/O Consolidation & Dead Code Removal

### New Module: `backend/frame_io.py`
- Unified frame reading functions: `read_image_frame()`, `read_video_frame_at()`, `read_video_frames()`, `read_mask_frame()`, `read_video_mask_at()`
- Handles EXR (linear float, BGRA stripping) and standard formats (uint8 → float32) in one place
- Optional `gamma_correct_exr` parameter for VideoMaMa's linear→sRGB conversion
- `read_video_frames()` accepts optional `processor` callable for custom per-frame transforms
- `EXR_WRITE_FLAGS` constant moved here from `service.py`

### `backend/service.py` (946 → 877 lines, -69 lines)
- `_read_input_frame()` — sequence path now delegates to `frame_io.read_image_frame()`
- `_read_alpha_frame()` — sequence path now delegates to `frame_io.read_mask_frame()`
- `reprocess_single_frame()` — replaced 50 lines of inline frame reading with `frame_io` calls
- `_load_frames_for_videomama()` — simplified from 30 lines to 7 using `frame_io`
- `_load_mask_frames_for_videomama()` — simplified using `read_video_frames()` with processor
- Removed `normalize_mask_channels`, `normalize_mask_dtype` imports (now internal to `frame_io`)

### Dead Code Removal
- `main.py` — removed unused `import time` (added in Step 2, never used)
- `ui/main_window.py` — removed unused `import tempfile`

### Test Adaptation
- `test_invalid_format.py` — updated `_EXR_FLAGS` reference to `frame_io.EXR_WRITE_FLAGS`
- All 224 tests pass, 0 regressions

---

## 2026-02-28 — Comprehensive Logging Infrastructure

### File-Based Session Logging
- Dual-handler logging: console (respects `--log-level`) + file (always DEBUG)
- Session-named log files: `logs/backend/YYMMDD_HHMMSS_corridorkey.log` (Eastern Time)
- `EasternFormatter` subclass forces America/New_York timezone on all timestamps
- `RotatingFileHandler` — 50MB per file, 3 backups (200MB max)
- Frozen build aware via `get_app_dir()` for log directory path

### Latency Tracking (5 GPU operations)
- `_get_engine()` — model load time
- `run_inference()` — total time + per-frame `process_frame` time + avg
- `run_gvm()` — total time
- `run_videomama()` — total time + per-chunk time
- `reprocess_single_frame()` — total time
- `process_frame()` in inference_engine.py — per-frame GPU time with resolution

### Silent Exception Fixes (6 locations)
- `service.py:201` — VRAM query failure now logged at DEBUG
- `service.py:227` — torch import in `_ensure_model` now logged at DEBUG
- `service.py:295` — torch import in `unload_engines` now logged at DEBUG
- `service.py:628` — state transition to COMPLETE failure now logged at WARNING
- `clip_state.py:88` — video frame count detection failure now logged at DEBUG
- `clip_state.py:215` — manifest JSON parse failure now logged at DEBUG

### inference_engine.py Modernization
- Replaced 4 `print()` calls with proper `logging` (model load, PosEmbed mismatch, missing/unexpected keys)
- Added `logger = logging.getLogger(__name__)` infrastructure

### Entry/Exit Logging
- `_read_input_frame()` — logs frame index at DEBUG
- `_write_outputs()` — logs clip name, frame index, stem at DEBUG
- `detect_device()` — logs selected device at INFO
- `scan_clips_dir()` — logs clip count at INFO

### Documentation
- Created `dev-docs/guides/debug-log-bible.md` — process chains, log format, debug queries, module logger names
- Added `logs/` to `.gitignore`

---

## 2026-02-28 — Comprehensive Backend Test Suite

### New Test Files (7 created/updated)
- `tests/conftest.py` — shared fixtures: `sample_frame`, `sample_mask`, `tmp_clip_dir` (real tiny PNGs), `sample_clip`
- `tests/test_service.py` — 45 tests covering CorridorKeyService (init, device detection, VRAM, model residency, engine loading, scan/filter, frame I/O, write image/manifest/outputs, run_inference full pipeline, reprocess, unload, GVM)
- `tests/test_service_videomama_contract.py` — 10 tests documenting VideoMaMa dtype/range contracts (float32 [0,1] input, binary mask threshold, output write, uint8 binarization bug, missing assets, cancellation)
- `tests/test_service_concurrency.py` — 2 behavioral GPU lock tests (serialization via timestamps, model switch under contention)
- `tests/test_job_queue_full.py` — 37 tests for GPUJobQueue lifecycle (submit→start→complete, failure, cancellation, mark_cancelled, callbacks, callback safety, deduplication, find_job, properties, clear_history, GPUJob)
- `tests/test_validators_edge.py` — 13 edge case tests (normalize_mask_dtype with int32/int64/bool/uint32, zero frame counts, 1D/4D mask arrays, empty arrays, idempotent dir creation)
- `tests/test_invalid_format.py` — 8 format validation tests (EXR flags, PNG conversion, unknown format fallback, OutputConfig with non-standard formats)

### Bug Fix
- **Manifest atomic write** (`service.py:416-419`): Replaced `os.remove()` + `os.rename()` with `os.replace()` — eliminates window where manifest file disappears between remove and rename

### Test Coverage Results
- Before: 77 tests (0 covering service.py)
- After: 224 tests passed, 1 skipped
- service.py: 0 → 45 tests (all 22 methods covered)
- job_queue.py: 4 → 41 tests (full lifecycle, cancellation, callbacks)
- validators.py: 20 → 33 tests (edge cases documented)

### Identified Issues Documented in Tests
- VideoMaMa dtype contract: `_load_frames_for_videomama()` returns float32 [0,1] but uint8 input to `clip(x, 0, 1)*255` binarizes all non-zero values to 255
- Callback exception safety: on_completion/on_error raising exceptions must not corrupt job queue state
- Zero-frame COMPLETE policy: `processed == num_frames` passes when both are 0
- normalize_mask_dtype: int32/int64 values cast without normalizing to [0,1] range

---

## [Phase 4] - 2026-02-28 — Advanced: GPU Safety, Output Config, Live Reprocess, Sessions, PyInstaller

`8833736` — 17 files changed, 1227 insertions, 106 deletions

### GPU Serialization & Thread Safety
- Added `threading.Lock` (`_gpu_lock`) to `CorridorKeyService` — wraps ALL model operations (`_ensure_model`, `process_frame`, `run_inference`, `run_gvm`, `run_videomama`, `reprocess_single_frame`)
- Prevents concurrent GPU access that could corrupt model state or OOM

### PREVIEW_REPROCESS Job Type
- New `JobType.PREVIEW_REPROCESS` with "latest-only" replacement semantics
- Submitting a new preview job cancels any queued preview for the same clip
- Routes through the same GPU queue as inference (no bypass)
- Rapid slider changes only keep the most recent request

### Output Configuration
- New `OutputConfig` dataclass: per-output enable/disable flags and format selectors (EXR/PNG) for FG, Matte, Comp, Processed
- `to_dict()` / `from_dict()` serialization on both `InferenceParams` and `OutputConfig`
- OUTPUT section added to parameter panel with checkboxes and format dropdowns

### Run Manifest
- `.corridorkey_manifest.json` written atomically (tmp+rename) after each inference run
- Records enabled outputs and parameters used
- `completed_stems()` reads manifest to determine which output dirs to check for resume
- Falls back to FG+Matte intersection when no manifest exists (backward compat)

### Live Preview Reprocess
- "Live Preview" checkbox in parameter panel
- 200ms debounced `QTimer` on `params_changed` signal
- Submits `PREVIEW_REPROCESS` through GPU queue (serialized, not parallel)
- `reprocess_single_frame()` on service: GPU-locked, in-memory only, no disk write
- Signal suppression (`_suppress_signals`) during session restore prevents event storms

### Session Save/Load
- JSON sidecar `.corridorkey_session.json` in clips directory
- `_SESSION_VERSION = 1` with forward compatibility (ignores unknown keys)
- Atomic write (tmp+rename pattern)
- Auto-save on window close, auto-load on directory change
- Ctrl+S / Ctrl+O keyboard shortcuts
- Captures: params, output config, live preview state, split view position, window geometry, splitter sizes, selected clip

### PyInstaller Packaging
- `corridorkey.spec` — bundles QSS theme and fonts, hidden imports for all modules, excludes matplotlib/tkinter/jupyter, Console=False
- `scripts/build_windows.ps1` — PowerShell build script with checkpoint copy and build summary

### Frozen Build Support
- `get_base_dir()` / `get_app_dir()` in main.py (sys._MEIPASS aware)
- Frozen-aware paths in `ui/theme/__init__.py`, `ui/app.py`, `backend/service.py`
- Fixed `run_cli()` with proper `hasattr` checks and ImportError handling

### Tests (20 new)
- `test_job_queue_phase4.py` — PREVIEW_REPROCESS replacement, no blocking inference, dedup unchanged, rapid requests
- `test_output_config.py` — InferenceParams roundtrip, OutputConfig roundtrip, enabled_outputs, manifest-based resume, fallback resume
- `test_session.py` — params roundtrip, session format, forward compat, corrupt file handling, atomic write

---

## [Phase 3] - 2026-02-28 — Preview Polish: Split View, Scrubber, View Modes, Zoom/Pan, Thumbnails

`938008f` — 19 files changed, 1809 insertions, 89 deletions

### Split View
- `SplitViewWidget` — before/after comparison with draggable yellow (#FFF203) divider
- Vertical split line, smooth drag, keyboard toggle

### Frame Scrubber
- `FrameScrubber` — timeline widget with frame-accurate seeking
- Click-to-seek, drag scrubbing, frame counter display

### View Modes
- `ViewModeBar` — toggle buttons for Input, Alpha, FG, Matte, Comp, Processed
- Each mode loads from corresponding output directory
- Keyboard shortcuts for quick switching

### Preview Viewport Overhaul
- Zoom (mouse wheel) + pan (middle-click drag)
- Fit-to-viewport on load, zoom-to-cursor behavior
- EXR → 8-bit display transform for HDR content

### Display Transform Pipeline
- `display_transform.py` — tone mapping, gamma correction, alpha compositing over checkerboard
- Handles EXR (float32), PNG (uint8), and mixed formats
- Checkerboard background for transparency visualization

### Frame Index System
- `FrameIndex` — stem-based navigation (not list index) prevents cross-mode misalignment
- `natural_sort.py` — handles non-zero-padded filenames (frame1, frame2, ..., frame10)

### Async Decoder
- `AsyncDecoder` — background frame loading with LRU cache
- Prevents UI blocking on large EXR files

### Thumbnails
- `ThumbnailWorker` — extracts first frame from each clip in background thread
- 60x40px thumbnails displayed in clip browser cards
- `ClipCardDelegate` — custom paint delegate with thumbnail, state badge, frame count

### Tests (15 new)
- `test_display_transform.py` — tone mapping, gamma, alpha compositing
- `test_frame_index.py` — stem navigation, boundary checks, natural sort integration
- `test_natural_sort.py` — zero-padded, non-padded, mixed filenames

---

## [Phase 1+2] - 2026-02-28 — GUI Shell + Batch Queue + GVM/VideoMaMa

`4970885` — 18 files changed, 2497 insertions

### Application Shell
- `ui/app.py` — QApplication with Corridor Digital dark theme, Open Sans font loading
- `main.py` — entry point with `--gui` (default) and `--cli` fallback

### 3-Panel Layout
- `MainWindow` — QSplitter-based 3-panel layout (clip browser | preview | parameters)
- Menu bar: File (Open, Save Session), View, Help
- Brand wordmark header

### Clip Browser (Left Panel)
- `ClipBrowser` — scrollable list with `ExtendedSelection` for batch ops
- `ClipCard` — card widget with state badge (RAW=gray, MASKED=blue, READY=yellow, COMPLETE=green, ERROR=red)
- [+ADD] button → QFileDialog for clips directory
- [WATCH] toggle → QFileSystemWatcher for auto-detection of new clips
- Drag-and-drop folder support
- Processing guards: watcher won't reclassify clips being processed

### Preview Viewport (Center Panel)
- `PreviewViewport` — QLabel + QPixmap display (CPU-rendered, zero VRAM)
- Downsample to viewport size for performance
- Frame navigation (left/right arrows)

### Parameter Panel (Right Panel)
- `ParameterPanel` — all inference controls
- Color Space dropdown (sRGB / Linear)
- Despill strength slider (0-10 → 0.0-1.0)
- Despeckle toggle + size spinbox
- Refiner scale slider (0-30 → 0.0-3.0)
- GVM AUTO and VIDEOMAMA alpha generation buttons (state-gated)

### Status Bar (Bottom)
- `StatusBar` — progress bar, frame counter, percentage, ETA
- VRAM usage bar with numeric readout
- GPU name badge
- [RUN INFERENCE] / [STOP] button with state toggle

### Queue Panel
- `QueuePanel` — visual job queue display
- Per-job progress, status badges, cancel buttons
- Batch queue management

### GPU Job Worker
- `GPUJobWorker` — single QThread processing jobs from `GPUJobQueue`
- Handles INFERENCE, GVM, VIDEOMAMA job types
- Per-frame progress signals, preview throttling (every 5th frame)
- Cancel/abort support (checks `_abort` flag between frames)
- Resume support (skips completed frames)
- Settings snapshot per job (params frozen at queue time)

### GPU Monitor
- `GPUMonitor` — QTimer polling `torch.cuda.memory_reserved()` every 2 seconds
- Reports VRAM usage, GPU name, temperature
- Fallback to nvidia-smi subprocess if torch unavailable

### Clip Model
- `ClipListModel` — QAbstractListModel wrapping ClipEntry list
- Custom roles for clip data, state, thumbnails
- Batch update support, count change signals

### Theme
- `corridor_theme.qss` — 398 lines, full brand stylesheet
- Dark-only: #141300 background, #1E1D13 panels, #FFF203 accents
- Zero border-radius on all widgets
- Styled scrollbars, tooltips, group boxes, menus

---

## [Phase 0] - 2026-02-28 — Backend Service Layer + Bug Fixes

`ef8e636` — Backend extraction from clip_manager.py

### Bug Fixes
- **BUG 1 (CRITICAL):** Fixed `cu.to_srgb()` → `cu.linear_to_srgb()` in inference_engine.py (linear mode was crashing)
- **BUG 2 (CRITICAL):** Fixed mask channel handling — always reduces to single channel regardless of input (2ch/4ch EXR no longer creates invalid tensors)
- **BUG 3 (HIGH):** Added logging for frame count mismatches and read failures (was silently truncating/skipping)
- **BUG 4 (HIGH):** Added cv2.imwrite() return value checking (disk full / permission errors no longer silent)
- **DEPENDENCY:** Changed `opencv-python` → `opencv-python-headless` (avoids Qt5 plugin conflict with PySide6)

### Backend Architecture
- `backend/service.py` — `CorridorKeyService` wrapping scan, validate, process, write operations
- `backend/clip_state.py` — `ClipEntry` dataclass + state machine (RAW → MASKED → READY → COMPLETE → ERROR)
- `backend/job_queue.py` — `GPUJobQueue` with mutual exclusion, deduplication, priority scheduling
- `backend/validators.py` — frame count validation, mask channel checks, write verification
- `backend/errors.py` — typed exceptions (FrameMismatchError, WriteFailureError, VRAMInsufficientError, etc.)

### Tests (32 tests)
- `test_validators.py` — frame parity, mask channel validation, write checks
- `test_clip_state.py` — state machine transitions, guard conditions
- `test_job_queue.py` — queue ordering, dedup, mutual exclusion

---

## Pre-GUI Releases

### 2026-02-27
- `a29d8b3` Rename MaskHint to VideoMamaMaskHint across codebase and folders

### 2026-02-26
- `f35fffe` Optimize inference VRAM with FP16 autocast

### 2026-02-25
- `cec7b85` Add Windows Auto-Installer scripts with HuggingFace model downloads
- `5e5f8dc` Add licensing and acknowledgements for GVM and VideoMaMa
- `0e4bbdc` Add comprehensive master README.md
- `d86ec87` Add technical handover document (LLM_HANDOVER.md)
- `ec6a0c9` Remove unused PointRend module from CorridorKeyModule
- `38989bf` Add true sRGB conversions to color_utils, refiner scale to wizard

### 2026-02-23
- `418a324` Add local Windows and Linux launcher scripts

### 2026-02-22
- `4f1dad6` Add luminance-preserving despill, configurable auto-despeckling garbage matte, checkerboard composite
- `d5559bc` Initial commit: Smart Wizard, VideoMaMa Integration, Optional GVM


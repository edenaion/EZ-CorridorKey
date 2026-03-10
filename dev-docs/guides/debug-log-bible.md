# Debug Log Bible — ez-CorridorKey

Last updated: 2026-03-02
Last audit: 2026-03-02

## Log Location

| Subdirectory | Contents | Written By |
|-------------|----------|------------|
| `logs/backend/` | All application logs (backend + UI + inference) | `main.py:setup_logging()` |

Session files are named: `YYMMDD_HHMMSS_corridorkey.log` (Eastern Time).

Rotation: 50MB per file, 3 backups (200MB max per session).

## Log Format

```
2026-02-28 14:05:23 [INFO   ] backend.service: Engine loaded in 4.2s
2026-02-28 14:05:24 [DEBUG  ] backend.service: Reading input frame 0 for 'Shot_01'
2026-02-28 14:05:25 [DEBUG  ] CorridorKeyModule.inference_engine: process_frame: 2160x3840 in 1.234s
```

- **Timestamps:** Eastern Time (America/New_York) via `zoneinfo.ZoneInfo`
- **Console:** Respects `--log-level` flag (default INFO), short time `HH:MM:SS`
- **File:** Always captures DEBUG, full date `YYYY-MM-DD HH:MM:SS`

## How to Enable Debug Logging

```bash
python main.py --log-level DEBUG
```

Console will show DEBUG messages. File always captures DEBUG regardless.

## Process Chain: Full Inference Run

```
MainWindow._on_run_inference()
└─ GPUJobWorker._run_inference()
   └─ CorridorKeyService.run_inference()
      ├─ _get_engine()            → "Engine loaded in Xs" (INFO)
      │                           → "Loading checkpoint: name.pth" (INFO)
      ├─ validate_frame_counts()  → warning if mismatch
      ├─ _write_manifest()        → atomic JSON write
      ├─ (per frame loop)
      │   ├─ _read_input_frame()  → "Reading input frame N" (DEBUG)
      │   ├─ _read_alpha_frame()
      │   ├─ process_frame()      → "process_frame: HxW in Xs" (DEBUG)
      │   ├─ _write_outputs()     → "Writing outputs for ... stem=..." (DEBUG)
      │   └─ (timing)             → "frame N: process_frame Xs" (DEBUG)
      └─ summary                  → "inference complete. N/M frames in Xs (Xs/frame avg)" (INFO)
```

## Process Chain: GVM Alpha Generation

```
CorridorKeyService.run_gvm()
├─ _get_gvm()
│   ├─ _ensure_model(GVM)       → model switch with VRAM reporting (see below)
│   ├─ "Loading GVM processor..." (INFO)
│   └─ "GVM loaded in Xs" (INFO)
├─ gvm.process_sequence()        → monolithic call
└─ summary                       → "GVM complete: N alpha frames in Xs" (INFO)
```

## Process Chain: VideoMaMa Alpha Generation

```
CorridorKeyService.run_videomama()
├─ _get_videomama_pipeline()
│   ├─ _ensure_model(VIDEOMAMA)  → model switch with VRAM reporting (see below)
│   ├─ "Loading VideoMaMa pipeline..." (INFO)
│   └─ "VideoMaMa loaded in Xs" (INFO)
├─ _load_frames_for_videomama()
├─ _load_mask_frames_for_videomama()
├─ (per chunk loop)
│   └─ run_inference()           → "chunk N: M frames in Xs" (DEBUG)
└─ summary                       → "VideoMaMa complete: N alpha frames in Xs" (INFO)
```

## Process Chain: Single-Frame Reprocess (Live Preview)

```
CorridorKeyService.reprocess_single_frame()
├─ _get_engine()        → cached (no load log if already loaded)
├─ read input frame
├─ read alpha frame
├─ process_frame()      → GPU-locked
└─ timing               → "frame N: reprocess Xs" (DEBUG)
```

## Process Chain: Model Residency Switch

Fires whenever the active model type changes (e.g., GVM → inference).

```
CorridorKeyService._ensure_model(needed)
├─ (if switching models)
│   ├─ "Unloading {old} model for {new} (VRAM before: NMB)" (INFO)
│   ├─ _safe_offload(model)
│   │   ├─ "Offloading model: ClassName" (DEBUG)
│   │   └─ model.to('cpu') / model.unload()
│   ├─ gc.collect()
│   ├─ torch.cuda.empty_cache()
│   └─ "VRAM after unload: NMB (freed NMB)" (INFO)
└─ (sets _active_model = needed)
```

**VRAM leak diagnosis:** If "freed" is 0MB or very small, the previous model's
tensors are still referenced somewhere (circular refs, stale local vars, etc.).

## Process Chain: Frame Extraction Pipeline

```
MainWindow._on_import_videos()
└─ ExtractWorker._process_job()
   ├─ find_ffmpeg()                → (checks PATH, no log)
   ├─ probe_video()                → (subprocess, no direct log)
   ├─ extract_frames()             → "Extracting N frames from video" (INFO)
   │   ├─ (progress callback)      → UI signal only
   │   └─ cancel check             → "Extraction cancelled" (INFO)
   ├─ write_video_metadata()       → (silent)
   └─ finished signal              → "Extraction complete: clip (N frames)" (INFO)
```

## Process Chain: Project Management

```
MainWindow._on_files_selected()
└─ project.create_project()
   ├─ projects_root()              → creates dir if needed (silent)
   ├─ _create_clip_folder()        → "Copied source video" (INFO)
   └─ write_project_json()         → (silent)

RecentProjectsPanel._on_delete_clicked()
├─ safety check                    → "Refusing to delete outside Projects folder" (WARNING)
├─ shutil.rmtree()                 → "Deleting project folder: path" (INFO)
└─ failure                         → "Failed to delete project: error" (ERROR)
```

## Silent Errors (Logged at DEBUG)

These are non-fatal graceful degradations:

| Pattern | Module | When It Fires |
|---------|--------|---------------|
| `VRAM query failed: ...` | service.py | Non-CUDA machine, driver issue |
| `torch not available for cache clear...` | service.py | CPU-only install (no torch) |
| `Video frame count detection failed...` | clip_state.py | Corrupt video, missing codec |
| `Failed to read manifest: ...` | clip_state.py | Missing/corrupt JSON manifest |
| `Model offload warning: ...` | service.py | Model `.to('cpu')` failed (rare) |
| `Preview save skipped: ...` | gpu_job_worker.py | Best-effort preview I/O failure |

## Warning-Level Issues

| Pattern | Module | Meaning |
|---------|--------|---------|
| `state transition to COMPLETE failed` | service.py | State machine rejected transition after successful inference |
| `PosEmbed shape mismatch: resizing...` | inference_engine.py | Checkpoint/model version discrepancy |
| `Missing keys in checkpoint` | inference_engine.py | Partial checkpoint load |
| `Unexpected keys in checkpoint` | inference_engine.py | Extra keys in checkpoint |
| `frame(s) skipped` | service.py | Frame read failures during batch |
| `Refusing to delete outside Projects folder` | recent_projects_panel.py | Safety guard blocked deletion |

## Error-Level Issues

| Pattern | Module | Meaning |
|---------|--------|---------|
| `Job failed [id]: clip — error` | gpu_job_worker.py | CorridorKeyError during job execution |
| `Unexpected error: ...` | gpu_job_worker.py | Uncaught exception (includes traceback) |
| `Failed to delete project: ...` | recent_projects_panel.py | `shutil.rmtree` failure |

## Common Debug Queries

Find slow frames:
```bash
grep "process_frame:" logs/backend/*.log | sort -t' ' -k2 -rn | head -20
```

Find all errors in a session:
```bash
grep "\[ERROR\]" logs/backend/YYMMDD_HHMMSS_corridorkey.log
```

Trace a specific clip:
```bash
grep "Shot_01" logs/backend/YYMMDD_HHMMSS_corridorkey.log
```

Find model load times (all three engines):
```bash
grep -E "(Engine|GVM|VideoMaMa) loaded in" logs/backend/*.log
```

Trace VRAM during model switches:
```bash
grep -E "VRAM (before|after)" logs/backend/*.log
```

Trace model offload actions:
```bash
grep "Offloading model:" logs/backend/*.log
```

## Module Logger Names

All modules use `logging.getLogger(__name__)`:

| Logger Name | File |
|-------------|------|
| `backend.service` | backend/service.py |
| `backend.clip_state` | backend/clip_state.py |
| `backend.job_queue` | backend/job_queue.py |
| `backend.validators` | backend/validators.py |
| `backend.ffmpeg_tools` | backend/ffmpeg_tools.py |
| `backend.project` | backend/project.py |
| `CorridorKeyModule.inference_engine` | CorridorKeyModule/inference_engine.py |
| `gvm_core.wrapper` | gvm_core/wrapper.py |
| `VideoMaMaInferenceModule.inference` | VideoMaMaInferenceModule/inference.py |
| `VideoMaMaInferenceModule.pipeline` | VideoMaMaInferenceModule/pipeline.py |
| `ui.app` | ui/app.py |
| `ui.main_window` | ui/main_window.py |
| `ui.recent_sessions` | ui/recent_sessions.py |
| `ui.shortcut_registry` | ui/shortcut_registry.py |
| `ui.workers.gpu_job_worker` | ui/workers/gpu_job_worker.py |
| `ui.workers.gpu_monitor` | ui/workers/gpu_monitor.py |
| `ui.workers.extract_worker` | ui/workers/extract_worker.py |
| `ui.workers.thumbnail_worker` | ui/workers/thumbnail_worker.py |
| `ui.preview.async_decoder` | ui/preview/async_decoder.py |
| `ui.preview.display_transform` | ui/preview/display_transform.py |
| `ui.widgets.clip_browser` | ui/widgets/clip_browser.py |
| `ui.widgets.dual_viewer` | ui/widgets/dual_viewer.py |
| `ui.widgets.io_tray_panel` | ui/widgets/io_tray_panel.py |
| `ui.widgets.hotkeys_dialog` | ui/widgets/hotkeys_dialog.py |
| `ui.widgets.annotation_overlay` | ui/widgets/annotation_overlay.py |
| `ui.widgets.preview_viewport` | ui/widgets/preview_viewport.py |
| `ui.widgets.recent_projects_panel` | ui/widgets/recent_projects_panel.py |

## Process Chain: GVM→Inference Model Handoff (Bug #3)

Critical path that caused hang when switching from GVM to inference in same session.

```
run_inference()
├─ "run_inference: waiting for _gpu_lock" (INFO)
├─ "run_inference: acquired _gpu_lock" (INFO)
├─ _get_engine()
│   ├─ _ensure_model(INFERENCE)
│   │   ├─ "Unloading gvm model for inference (VRAM before: NMB)" (INFO)
│   │   ├─ _safe_offload(gvm_processor)  → "Offloading model: GVMProcessor" (DEBUG)
│   │   │                                → "_safe_offload took Xs" (INFO)
│   │   ├─ gc.collect()                  → "gc.collect took Xs" (INFO)
│   │   ├─ torch.cuda.synchronize()      → "cuda.synchronize took Xs" (INFO)
│   │   ├─ torch.cuda.empty_cache()      → "cuda.empty_cache took Xs" (INFO)
│   │   └─ "VRAM after unload: NMB (freed NMB)" (INFO)
│   ├─ "Constructing CorridorKeyEngine..." (INFO)
│   ├─ CorridorKeyEngine.__init__
│   │   ├─ "__init__: begin" (INFO)
│   │   ├─ "Optimization: auto mode — probing VRAM..." (INFO)
│   │   ├─ _get_vram_gb() [pynvml first, torch.cuda fallback]
│   │   │   ├─ "_get_vram_gb: entering" (DEBUG)
│   │   │   ├─ "_get_vram_gb: pynvml reports N.N GB" (DEBUG)
│   │   │   └─ OR "_get_vram_gb: torch.cuda reports N.N GB" (DEBUG)
│   │   ├─ "Optimization: auto → speed mode (N GB ≥ 12 GB)" (INFO)
│   │   ├─ "__init__: entering _load_model" (INFO)
│   │   └─ _load_model()  → [see inference chain above]
│   └─ "CorridorKeyEngine construction complete" (INFO)
└─ (inference loop continues)
```

**Hang diagnosis:** If log stops at `__init__: begin` or `entering _load_model`, the
constructor or model load is blocking. Check VRAM probe path and torch.compile status.

## Known Debug Gaps (2026-03-09 Audit)

### Priority 1 — Error Handling
- 3 bare `except:` clauses: clip_manager.py:929, model_transformer.py:267, pipeline_gvm.py:212
- `set_error()` in clip_state.py accepts reason but never logs it
- 5 `print()` in model_transformer.py not integrated with logging system

### Priority 2 — State Visibility
- `_resolve_state()` sets initial clip state silently (6 branches at lines 446-478)
- Job queue tracks status changes but not elapsed wall-clock time per job
- UI widget enable/disable changes are completely unlogged

### Priority 3 — Latency Gaps
- gvm_core/wrapper.py: model load and per-file processing not timed
- VideoMaMaInferenceModule/inference.py: per-chunk inference duration not captured
- FFmpeg extraction: total operation duration not logged (only frame count)

## Audit History

| Date | Changes |
|------|---------|
| 2026-02-28 | Initial creation: dual-handler setup, Eastern Time, latency tracking, silent exception logging |
| 2026-03-02 | Audit update: VRAM reporting in _ensure_model, model load timing for GVM/VideoMaMa, error logging in gpu_job_worker CorridorKeyError branch, print→logger in VideoMaMa and GVM wrapper, root-logger fix in gvm_core, logging added to recent_projects_panel, 3 new process chains (model switch, extraction, project mgmt), 16 new module logger entries |
| 2026-03-09 | Full audit (8 parallel agents): 223 logging statements catalogued across 58 files. Added GVM→Inference handoff chain. Identified 3 bare except clauses, 7 silent failures, 6 unlogged state initializations, 5 print() calls in model_transformer.py. Log locations fully compliant (single system, YYMMDD_HHMMSS naming, ./logs/backend/). Coverage rated Complete for model lifecycle + inference engine, Partial for job queue (no elapsed time), GVM wrapper (no timing), clip state (silent init). Gaps documented above for future fix passes. |

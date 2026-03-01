# Debug Log Bible — ez-CorridorKey

Last updated: 2026-02-28
Last audit: 2026-02-28

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
├─ _get_gvm()              → model load
├─ gvm.process_sequence()  → monolithic call
└─ summary                 → "GVM complete: N alpha frames in Xs" (INFO)
```

## Process Chain: VideoMaMa Alpha Generation

```
CorridorKeyService.run_videomama()
├─ _get_videomama_pipeline()   → model load
├─ _load_frames_for_videomama()
├─ _load_mask_frames_for_videomama()
├─ (per chunk loop)
│   └─ run_inference()         → "chunk N: M frames in Xs" (DEBUG)
└─ summary                     → "VideoMaMa complete: N alpha frames in Xs" (INFO)
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

## Silent Errors (Logged at DEBUG)

These are non-fatal graceful degradations:

| Pattern | Module | When It Fires |
|---------|--------|---------------|
| `VRAM query failed: ...` | service.py | Non-CUDA machine, driver issue |
| `torch not available for cache clear...` | service.py | CPU-only install (no torch) |
| `Video frame count detection failed...` | clip_state.py | Corrupt video, missing codec |
| `Failed to read manifest: ...` | clip_state.py | Missing/corrupt JSON manifest |

## Warning-Level Issues

| Pattern | Module | Meaning |
|---------|--------|---------|
| `state transition to COMPLETE failed` | service.py | State machine rejected transition after successful inference |
| `PosEmbed shape mismatch: resizing...` | inference_engine.py | Checkpoint/model version discrepancy |
| `Missing keys in checkpoint` | inference_engine.py | Partial checkpoint load |
| `Unexpected keys in checkpoint` | inference_engine.py | Extra keys in checkpoint |
| `frame(s) skipped` | service.py | Frame read failures during batch |

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

Find model load times:
```bash
grep "Engine loaded in" logs/backend/*.log
```

## Module Logger Names

All modules use `logging.getLogger(__name__)`:

| Logger Name | File |
|-------------|------|
| `backend.service` | backend/service.py |
| `backend.clip_state` | backend/clip_state.py |
| `backend.job_queue` | backend/job_queue.py |
| `backend.validators` | backend/validators.py |
| `CorridorKeyModule.inference_engine` | CorridorKeyModule/inference_engine.py |
| `ui.main_window` | ui/main_window.py |
| `ui.workers.gpu_job_worker` | ui/workers/gpu_job_worker.py |
| `ui.workers.gpu_monitor` | ui/workers/gpu_monitor.py |
| `ui.widgets.clip_browser` | ui/widgets/clip_browser.py |
| `ui.preview.async_decoder` | ui/preview/async_decoder.py |
| `ui.preview.display_transform` | ui/preview/display_transform.py |

## Audit History

| Date | Changes |
|------|---------|
| 2026-02-28 | Initial creation: dual-handler setup, Eastern Time, latency tracking, silent exception logging |

# DVR Roundtrip QA

This is the repeatable EZCK-side harness for DaVinci Resolve roundtrip checks.

The goal is simple:

- prepare footage exactly through the same current EZCK backend path the app uses
- avoid "custom script math" that could muddy the result
- produce a JSON report that proves which params and outputs were used

## What This Mirrors

This harness intentionally follows the real app/backend path:

1. `backend.project.create_project_from_media()`
2. `backend.ffmpeg_tools.extract_frames()` plus `.video_metadata.json`
3. `ui.main_window._on_import_alpha()` rename/copy behavior
4. `backend.service.CorridorKeyService.run_inference()`
5. `backend.service.CorridorKeyService._write_outputs()`

That means:

- source video -> `Frames/` extraction uses the app's FFmpeg EXR path
- alpha import is renamed to input frame stems the same way the button does it
- `Processed`, `FG`, `Matte`, and `Comp` are written by the same service layer as the GUI

## What This Does Not Mirror

It does not drive the GUI itself.

That is intentional. For conclusive export QA, the important part is that the
rendered files come from the same backend functions the GUI dispatches to, not
that a person clicked the button by hand.

## Commands

From the repo root on Windows:

```bash
.venv\Scripts\python.exe scripts\dvr_roundtrip_qa.py ^
  --source-video Projects\DVR_Test\shot_lin.mov ^
  --alpha-dir Projects\DVR_Test\lin_alphahint ^
  --display-name EZCK_DVR_TEST ^
  --copy-video ^
  --input-linear
```

From the repo root on macOS/Linux:

```bash
.venv/bin/python scripts/dvr_roundtrip_qa.py \
  --source-video Projects/DVR_Test/shot_lin.mov \
  --alpha-dir Projects/DVR_Test/lin_alphahint \
  --display-name EZCK_DVR_TEST \
  --copy-video \
  --input-linear
```

If you already have an extracted EZCK project and only want to rerun inference:

```bash
.venv\Scripts\python.exe scripts\dvr_roundtrip_qa.py ^
  --project-dir Projects\260314_051239_shot_lin ^
  --input-linear
```

If you want the thinnest possible export for Resolve inspection:

```bash
.venv\Scripts\python.exe scripts\dvr_roundtrip_qa.py ^
  --project-dir Projects\260314_051239_shot_lin ^
  --input-linear ^
  --processed-only
```

If you want a quick single-frame smoke that still uses the app's real
`frame_range` path:

```bash
.venv\Scripts\python.exe scripts\dvr_roundtrip_qa.py ^
  --project-dir Projects\260314_051239_shot_lin ^
  --input-linear ^
  --processed-only ^
  --frame-start 0 ^
  --frame-end 0
```

## Report

By default the script writes:

```text
<clip_root>/dvr_roundtrip_report.json
```

The report records:

- project path
- clip path
- git short SHA if available
- whether the run created a fresh project or reused an existing one
- extraction details
- alpha-import details
- exact `InferenceParams`
- exact `OutputConfig`
- output directories written by the service

## Resolve-Side Check

After the harness completes:

1. Open the Resolve project `EZCK_DVR TEST`.
2. Import the rendered sequence from `Output/Processed/`.
3. Do not add a grade or CST just for the baseline check.
4. Compare against the original source using a wipe or side-by-side.
5. If needed, rerun with `--despill 0.0` to isolate keying controls from export math.

## Notes

- This harness is the trustworthy EZCK side of the test.
- If we later automate Resolve via MCP, that automation should start after these files already exist.
- That way, the Resolve automation only verifies import/interpretation/comparison and does not change the EZCK render path under test.

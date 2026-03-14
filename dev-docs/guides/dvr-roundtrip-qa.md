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

After the harness completes, use the same general Fusion A/B method the tester
used in the tutorial video.

### Fusion A/B Workflow

1. Open the Resolve project `EZCK_DVR TEST`.
2. Open a Fusion composition.
3. Add `Loader1` pointing at the original source clip, for example:
   `Projects/DVR_Test/shot_lin.mov`
4. Add `Loader2` pointing at the rendered EZCK output sequence, for example:
   `Output/Processed/frame_000000.exr`
5. Send `Loader1` to viewer buffer `A`.
6. Send `Loader2` to viewer buffer `B`.
7. In the Fusion viewer, enable the same viewer-level gamut normalization the
   tester showed:
   - open the viewer LUT menu
   - choose `GamutView LUT`
   - keep `Source Space = No Change`
   - set `Output Space = ITU-R BT.709 (scene)`
   - leave `Add Gamma` enabled
8. Switch the viewer compare mode to `Buffer Split Wipe`.
9. Scrub to a representative frame and inspect the split line directly across
   skin, fabric, and neutral areas.

### Pass / Fail

Pass:

- The processed result does not show a global brightness, gamma, or contrast
  shift relative to the source under the same viewer normalization.
- Any visible change is attributable to despill, refiner, matte cleanup, or
  edge handling rather than a whole-image darkening.

Fail:

- The processed side looks globally darker under the same Fusion viewer setup.
- The mismatch is broad and image-wide, not localized to expected keying
  changes.

### Isolation Runs

If the first comparison looks wrong:

1. Re-run the EZCK harness with `--despill 0.0`.
2. Keep `Processed` as `EXR`.
3. Re-import only the new output sequence in Fusion.
4. Repeat the exact same `Buffer Split Wipe` comparison.

That keeps the A/B stable while isolating export math from intentional keying
adjustments.

## Notes

- This harness is the trustworthy EZCK side of the test.
- If we later automate Resolve via MCP, that automation should start after these files already exist.
- That way, the Resolve automation only verifies import/interpretation/comparison and does not change the EZCK render path under test.

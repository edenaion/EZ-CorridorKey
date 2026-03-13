# Debug TODOs

## 2026-03-12

- SAM2 Track Mask preview brightness mismatch
  - Symptom: on the first `Track Mask` / `Track Preview`, the right-side SAM preview can appear visibly brighter / washed out than the source image.
  - Repro note: user sees this immediately on the previewed right pane during SAM preview, before accepting full track.
  - Scope note: treat this as a separate issue from the live `Processed` / reprocess color-space regression. Do not conflate the two until verified.
  - Investigation target: the SAM preview image/display path, specifically how `frame_rgb` is converted for the right viewer during `sam2_preview`.

- Live preview consistency across output modes
  - Symptom: a live preview update seen in `COMP` is not carried through when switching to `PROC` / `FG` / `MATTE`; the modes do not appear to share the same freshly reprocessed result.
  - Expected: one live reprocess result should be viewable consistently across all output modes without requiring scrub/re-run.
  - Investigation target: retain/reuse the latest reprocess result across output mode switches instead of only rendering the mode that was active when the preview job returned.

- `PROC` preview appears washed out relative to `COMP`
  - Symptom: `Processed` preview is slightly brighter / more washed out than `Comp`, even when `Comp` looks correct.
  - Scope note: investigate separately from the EXR input color-space override that was just fixed and committed.
  - Investigation target: the `ViewMode.PROCESSED` live-preview display transform, especially unpremultiply + gamma conversion versus the `COMP` preview path.

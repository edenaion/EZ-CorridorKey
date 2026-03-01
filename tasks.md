# CorridorKey GUI — Tasks

## Pending

- **Export settings tracking**: When inference produces exports, save the settings used (color space, despill, despeckle, refiner, output formats) alongside the output — either as a sidecar JSON file in the Output/ folder, or displayed in the GUI's Exports section so users can recall what settings produced each result.

- **Squished parameter fields**: Color Space dropdown and Despeckle spinbox get truncated/clipped in the right panel during inference. The 240px minimum panel width plus margins doesn't leave enough room. Need to fix layout so these fields display their full text.

- **Collapsed sidebar takes too much space**: When the left clip browser is collapsed, it still occupies a full-height 28px column. Should be reduced to ONLY a tiny chevron nub at the top-left corner, with ALL horizontal space reclaimed by the dual viewer panels.

- **In/Out trim points per clip**: Allow users to set in-point (I hotkey) and out-point (O hotkey) on a per-clip basis so inference only processes a subsection of the video. Visual brackets above the timeline scrubber (Topaz-style). Trim range saved per-clip in session data. Inference respects the trim range when processing frames.

- **Verify post-inference side-by-side scrub**: The dual viewer already supports input vs output comparison (left=Original, right=Comp/FG/Matte/Processed). Need to verify: (1) after inference completes, output viewer auto-switches to COMP mode showing keyed result, (2) scrubbing works frame-by-frame across full range, (3) mode switching (COMP/FG/Matte) works on the output side. The infrastructure exists — this is a QA/verification task.

- **Alpha coverage feedback & partial run detection**: After GVM or VideoMaMa completes, show the user how many alpha frames were generated vs total input frames (e.g., "Generated 105/470 alpha hints"). On inference start, if alpha count < input count, warn the user with a dialog: "Alpha hints cover 105 of 470 frames. Process available range or re-run GVM?" Options: (1) Process available, (2) Re-run GVM for full clip, (3) Cancel. Also detect partial alpha from interrupted runs — offer to clear and re-generate.

## Done

- Cancel/stop inference — signal chain + confirmation dialog
- GVM real progress bar with percentage and ETA (no more bouncing bar)
- Collapsible left clip browser sidebar
- Escape key no longer accidentally cancels during modal dialogs

# CorridorKey GUI — Tasks

## Pending

- **Export settings tracking**: When inference produces exports, save the settings used (color space, despill, despeckle, refiner, output formats) alongside the output — either as a sidecar JSON file in the Output/ folder, or displayed in the GUI's Exports section so users can recall what settings produced each result.

- **Squished parameter fields**: Color Space dropdown and Despeckle spinbox get truncated/clipped in the right panel during inference. The 240px minimum panel width plus margins doesn't leave enough room. Need to fix layout so these fields display their full text.

- **Collapsed sidebar takes too much space**: When the left clip browser is collapsed, it still occupies a full-height 28px column. Should be reduced to ONLY a tiny chevron nub at the top-left corner, with ALL horizontal space reclaimed by the dual viewer panels.

- **In/Out trim points per clip**: Allow users to set in-point (I hotkey) and out-point (O hotkey) on a per-clip basis so inference only processes a subsection of the video. Visual brackets above the timeline scrubber (Topaz-style). Trim range saved per-clip in session data. Inference respects the trim range when processing frames.

- **Welcome screen multi-select for batch import**: Video thumbnails and video files on the welcome screen should support multi-selection. Users should be able to: (1) Ctrl+click to toggle individual files, (2) Shift+click for range selection, (3) Ctrl+A to select all, (4) Click-drag for rubber-band/marquee selection across thumbnail grid. Selected files get imported as a batch when confirmed. This enables processing multiple videos together without repeated file dialog trips.

- **Verify post-inference side-by-side scrub**: The dual viewer already supports input vs output comparison (left=Original, right=Comp/FG/Matte/Processed). Need to verify: (1) after inference completes, output viewer auto-switches to COMP mode showing keyed result, (2) scrubbing works frame-by-frame across full range, (3) mode switching (COMP/FG/Matte) works on the output side. The infrastructure exists — this is a QA/verification task.

- **Alpha coverage feedback & partial run detection**: After GVM or VideoMaMa completes, show the user how many alpha frames were generated vs total input frames (e.g., "Generated 105/470 alpha hints"). On inference start, if alpha count < input count, warn the user with a dialog: "Alpha hints cover 105 of 470 frames. Process available range or re-run GVM?" Options: (1) Process available, (2) Re-run GVM for full clip, (3) Cancel. Also detect partial alpha from interrupted runs — offer to clear and re-generate.

- **Middle-click resets to default**: Middle-clicking on any adjustable value should reset it to its default. Applies to: (1) QSplitter divider lines — middle-click resets to 50/50 split, (2) Parameter sliders — middle-click resets the slider to its default value. Extend to other adjustable controls as discovered.

- **Live output mode switching during inference**: Users should be able to click between Input, FG, Matte, Comp, and Processed view modes on the right viewport while inference is still running. Since output frames are just images written to disk, switching modes should load whatever frames exist so far for that mode. Users can see partial FG, Matte, etc. results in real time as inference progresses, up to the last completed frame.

- **Comprehensive tooltips system**: Add descriptive tooltips to every interactive element and labeled area in the GUI. Behavior: 1-second delay before showing, black background with light gray text (#D0D0D0), persistent display while hovering, 600ms fade-out on mouse leave. Tooltips should fully explain each control's function in plain language (not just a label restatement). Cover: all toolbar buttons, parameter panel controls (sliders, dropdowns, checkboxes), view mode buttons (Input/FG/Matte/Comp/Processed), transport controls, coverage bar lanes, clip browser actions, welcome screen elements, status bar indicators, and GPU info. **Must be rendered inside the application window** — tooltips are custom in-window widgets (NOT native OS tooltips), so they clamp to window bounds and never go off-screen. If the tooltip would overflow any edge, reposition it to stay fully visible within the window rect. Implement as a custom QWidget overlay (not QToolTip) for full control over positioning, styling, and animation. Requires a **Settings panel** (gear icon or menu) where users can toggle tooltips on/off globally — setting persists across sessions via QSettings or project config.

## Done

- Cancel/stop inference — signal chain + confirmation dialog
- GVM real progress bar with percentage and ETA (no more bouncing bar)
- Collapsible left clip browser sidebar
- Escape key no longer accidentally cancels during modal dialogs

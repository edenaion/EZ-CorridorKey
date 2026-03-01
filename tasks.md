# CorridorKey GUI — Tasks

## Pending

- **Export settings display in GUI**: Manifest JSON is already saved alongside outputs (.corridorkey_manifest.json). Need to display these saved settings in the GUI — either in the IO tray's Exports section or as a tooltip/panel so users can recall what settings produced each result.

- **Welcome screen multi-select for batch import**: Video thumbnails and video files on the welcome screen should support multi-selection. Users should be able to: (1) Ctrl+click to toggle individual files, (2) Shift+click for range selection, (3) Ctrl+A to select all, (4) Click-drag for rubber-band/marquee selection across thumbnail grid. Selected files get imported as a batch when confirmed. This enables processing multiple videos together without repeated file dialog trips.

- **Verify post-inference side-by-side scrub**: The dual viewer already supports input vs output comparison (left=Original, right=Comp/FG/Matte/Processed). Need to verify: (1) after inference completes, output viewer auto-switches to COMP mode showing keyed result, (2) scrubbing works frame-by-frame across full range, (3) mode switching (COMP/FG/Matte) works on the output side. The infrastructure exists — this is a QA/verification task.

- **Alpha coverage feedback & partial run detection**: After GVM or VideoMaMa completes, show the user how many alpha frames were generated vs total input frames (e.g., "Generated 105/470 alpha hints"). On inference start, if alpha count < input count, warn the user with a dialog: "Alpha hints cover 105 of 470 frames. Process available range or re-run GVM?" Options: (1) Process available, (2) Re-run GVM for full clip, (3) Cancel. Also detect partial alpha from interrupted runs — offer to clear and re-generate.


- **Live output mode switching during inference**: Users should be able to click between Input, FG, Matte, Comp, and Processed view modes on the right viewport while inference is still running. Since output frames are just images written to disk, switching modes should load whatever frames exist so far for that mode. Users can see partial FG, Matte, etc. results in real time as inference progresses, up to the last completed frame.


- **Preferences dialog (Edit > Preferences)**: Add a settings/preferences panel accessible from the menu bar. Users can toggle options like tooltips on/off. Settings persist across sessions via QSettings.

## Done

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
- Cancel shows "Canceled" not "Failed" — already separated: cancel path uses warning signal + "Cancelled:" prefix, error path uses error signal + QMessageBox
- ADD button supports folders or files — QMenu choice: "Import Folder..." or "Import Video(s)...", drag-drop also accepts video files

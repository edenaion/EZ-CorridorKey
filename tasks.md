# CorridorKey GUI — Tasks

## Pending

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
- Export settings tooltip — hover over Exports cards in IO tray shows manifest data (outputs, formats, color space, despill, refiner, despeckle)
- Post-inference side-by-side scrub — verified working: auto-COMP switch, synced scrubbing, mode switching (COMP/FG/Matte/Processed)
- Alpha coverage feedback — status bar shows "X/Y alpha frames" after GVM/VideoMaMa; 3-option dialog (Process Available / Re-run GVM / Cancel) on partial alpha at inference start; partial alpha detection already in _resolve_state()
- Live output mode switching during inference — FrameIndex rebuilds on each preview update, mode buttons enable as FG/Matte/Comp/Processed outputs appear mid-inference
- Welcome screen multi-select — already supported: QFileDialog.getOpenFileNames() handles Ctrl/Shift/Ctrl+A natively, drag-drop accepts multiple files, _on_welcome_files() loops through batch
- Preferences dialog (Edit > Preferences) — QSettings-based, tooltips on/off toggle, copy source videos toggle, persists across sessions
- Deletion safety — rmtree guarded by Projects root check, never deletes outside project folder; non-copy mode references original video in place

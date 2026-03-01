# Clip Pipeline — What Happens Behind the Scenes

How a video goes from user import to keyed output.

---

## 1. Import

User drops a video file (MP4, MOV, etc.) or a folder of image sequences onto the welcome screen.

**Video file path:**
```
User drops: vacation_greenscreen.mp4
```

The app creates a workspace directory and a clip folder inside it:
```
WorkspaceName/
└── vacation_greenscreen/
    └── Input.mp4              ← original video, untouched
```

**Image sequence path:**
```
User drops folder: vacation_greenscreen/
└── Input/
    ├── frame_0001.png
    ├── frame_0002.png
    └── ...
```

If the folder already has an `Input/` subdirectory with frames, no extraction is needed.

**Code:** `backend/clip_state.py:scan_clips_dir()` scans the workspace and creates `ClipEntry` objects.

---

## 2. Frame Extraction

If the input is a video file (not an image sequence), the app extracts frames using FFmpeg:

```
vacation_greenscreen/
├── Input.mp4                  ← original video
└── Input/                     ← extracted frames (auto-created)
    ├── frame_0001.png
    ├── frame_0002.png
    └── ...
```

Clip state transitions: `EXTRACTING` → `RAW`

**Code:** `backend/ffmpeg_tools.py:extract_frames()` runs FFmpeg as a subprocess.

---

## 3. Clip States

Each clip progresses through a state machine:

| State | Meaning | What Exists |
|-------|---------|-------------|
| `EXTRACTING` | FFmpeg extracting frames from video | `Input.mp4` only |
| `RAW` | Frames extracted, no alpha hints | `Input/` with frames |
| `MASKED` | User-provided mask hints present | `Input/` + `VideoMamaMaskHint/` |
| `READY` | Alpha hints generated, ready for inference | `Input/` + `AlphaHint/` |
| `COMPLETE` | Inference done, outputs written | `Input/` + `AlphaHint/` + `Output/` |
| `ERROR` | Something failed | Partial state |

**Code:** `backend/clip_state.py:ClipEntry._resolve_state()`

---

## 4. Alpha Generation (Optional Pre-Step)

Before inference, the app can auto-generate alpha hints:

- **GVM Auto** — Generates alpha hints from raw input frames (RAW → READY)
- **VideoMaMa** — Generates alpha from user mask hints (MASKED → READY)

Output goes to:
```
vacation_greenscreen/
├── Input/
└── AlphaHint/                 ← generated alpha hints
    ├── frame_0001.png
    ├── frame_0002.png
    └── ...
```

**Code:** `backend/service.py:run_gvm()`, `run_videomama()`

---

## 5. Inference

The CorridorKey model processes each frame using the input + alpha hint to produce:

- **FG** — Clean foreground RGB (background removed)
- **Matte** — Alpha channel (grayscale mask)
- **Comp** — Composited result over checkerboard (sRGB uint8)
- **Processed** — RGBA premultiplied (full pipeline output)

Settings that affect output:
- Color Space (sRGB / Linear)
- Despill strength (0.0–1.0)
- Despeckle toggle + pixel size
- Refiner scale (0.0–3.0)

**Code:** `backend/service.py:run_inference()`, `_write_outputs()`

---

## 6. Output Directory Structure

All outputs go under the clip's `Output/` folder, organized by type:

```
vacation_greenscreen/
├── Input/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
├── AlphaHint/
│   ├── frame_0001.png
│   └── ...
└── Output/
    ├── FG/                    ← foreground RGB
    │   ├── frame_0001.exr
    │   └── ...
    ├── Matte/                 ← alpha channel
    │   ├── frame_0001.exr
    │   └── ...
    ├── Comp/                  ← composited preview
    │   ├── frame_0001.png
    │   └── ...
    ├── Processed/             ← RGBA premultiplied
    │   ├── frame_0001.exr
    │   └── ...
    └── .corridorkey_manifest.json  ← records which outputs were enabled
```

**Frame naming:** Output frames keep the same stem as input frames (`frame_0001` → `frame_0001.exr`). This allows stem-based alignment across all directories.

**Formats:** Each output type has a user-configurable format (EXR or PNG). Default is EXR for FG/Matte/Processed, PNG for Comp.

**Code:** `backend/validators.py:ensure_output_dirs()` creates the structure.

---

## 7. Workspace-Level Files

```
WorkspaceName/
├── .corridorkey_session.json  ← session state (params, UI, selected clip)
├── clip_1/
│   ├── Input/
│   ├── AlphaHint/
│   └── Output/
├── clip_2/
│   ├── Input/
│   └── Output/
└── ...
```

The session file stores:
- Inference parameters (despill, despeckle, refiner, color space)
- Output config (which outputs enabled, formats)
- UI state (window geometry, splitter sizes, selected clip)
- Live preview enabled state

**Code:** `ui/main_window.py:_build_session_data()`, `_apply_session_data()`

---

## 8. Resume Support

If inference is interrupted (cancel, crash, app close), it can resume:

1. The manifest (`.corridorkey_manifest.json`) records which output types were enabled
2. On restart, `ClipEntry.completed_stems()` checks which frames have ALL enabled outputs
3. Only incomplete frames are reprocessed

**Code:** `backend/clip_state.py:completed_stems()`

---

## 9. Post-Inference Viewing

After inference, the dual viewer shows:
- **Left panel** — Original input frames (locked to INPUT mode)
- **Right panel** — Output frames (user switches between COMP, FG, Matte, Processed)
- **Shared scrubber** — Both panels sync to the same frame index

During inference, the output viewer auto-updates with COMP previews as frames are written.

---

## 10. File Naming Convention

| Context | Pattern | Example |
|---------|---------|---------|
| Frame files | `{original_stem}.{format}` | `frame_0001.exr` |
| Session logs | `YYMMDD_HHMMSS_corridorkey.log` | `260301_090800_corridorkey.log` |
| Session file | `.corridorkey_session.json` | Fixed name per workspace |
| Manifest | `.corridorkey_manifest.json` | Fixed name per clip Output/ |

# Windows Installer Spec — CorridorKey .exe Installer

## Goal
Professional Windows installer (.exe) that installs CorridorKey to Program Files,
creates Start Menu/Desktop shortcuts, and handles uninstall. User runs one .exe,
clicks through a wizard, and launches CorridorKey from the Start Menu.

---

## Current State (What Already Works)

| Item | Status | Location |
|------|--------|----------|
| PyInstaller spec | Done | `corridorkey.spec` |
| Windows build script | Done | `scripts/build_windows.ps1` |
| Frozen detection (sys._MEIPASS) | Done | `main.py:52-73` |
| .ico icon | Done | `ui/theme/corridorkey.ico` |
| CUDA auto-detection | Done | `backend/service/core.py:251-267` |
| pynvml GPU monitoring | Done | `ui/workers/gpu_monitor.py` |
| torch.compile + Triton | Done | `CorridorKeyModule/inference_engine.py` |
| Model downloader | Done | `scripts/setup_models.py` |
| 7-step batch installer | Done | `1-install.bat` (dev-oriented, not end-user) |

---

## Current PyInstaller Build vs. Proper Installer

**What `corridorkey.spec` produces today:**
- `dist/CorridorKey/` folder with `CorridorKey.exe` + bundled DLLs/libs
- User must manually place checkpoints alongside exe
- No Start Menu shortcut, no Add/Remove Programs, no uninstall

**What we want:**
- Single `CorridorKey-Setup-1.8.0.exe` that wraps everything
- Install wizard: license → destination → shortcuts → install
- Registers in Add/Remove Programs (Programs and Features)
- Creates Start Menu + optional Desktop shortcut
- Uninstaller that cleanly removes everything
- Optional: bundled VC++ redistributable (if needed for CUDA/torch)

---

## Recommended Approach: Inno Setup

**Why Inno Setup over alternatives:**

| Tool | Pros | Cons |
|------|------|------|
| **Inno Setup** | Free, scriptable (.iss), widely used, small overhead | Windows-only compiler |
| NSIS | Very flexible, open source | More complex scripting |
| WiX | MSI format, enterprise-friendly | XML-heavy, steep learning curve |
| MSIX | Modern Windows packaging | Requires Store or sideloading config |

Inno Setup is the best fit: simple, produces professional installers, and the .iss script is easy to maintain.

---

## What Needs to Be Built

### Phase 1: PyInstaller Build (already done, minor tweaks)

The existing `corridorkey.spec` + `scripts/build_windows.ps1` handles this.

Minor improvements needed:
- [ ] Verify CUDA DLLs are collected properly (torch does this automatically)
- [ ] Ensure `nvidia-ml-py` pynvml DLL is included
- [ ] Test the frozen build launches correctly on a clean Windows machine

### Phase 2: Inno Setup Script

File: `scripts/windows/corridorkey-setup.iss`

```iss
[Setup]
AppName=EZ-CorridorKey
AppVersion=1.8.0
AppPublisher=Corridor Digital
AppPublisherURL=https://github.com/edenaion/EZ-CorridorKey
DefaultDirName={autopf}\CorridorKey
DefaultGroupName=CorridorKey
OutputBaseFilename=CorridorKey-Setup-1.8.0
SetupIconFile=..\..\ui\theme\corridorkey.ico
UninstallDisplayIcon={app}\CorridorKey.exe
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
LicenseFile=..\..\LICENSE
WizardStyle=modern
DisableProgramGroupPage=yes

[Files]
; PyInstaller output — entire dist/CorridorKey/ folder
Source: "..\..\dist\CorridorKey\*"; DestDir: "{app}"; Flags: recursesubdirs

; Checkpoint placeholder directory
Source: ""; DestDir: "{app}\CorridorKeyModule\checkpoints"; Flags: createallsubdirs

[Icons]
Name: "{group}\CorridorKey"; Filename: "{app}\CorridorKey.exe"
Name: "{autodesktop}\CorridorKey"; Filename: "{app}\CorridorKey.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
; Launch after install
Filename: "{app}\CorridorKey.exe"; Description: "Launch CorridorKey"; Flags: postinstall nowait skipifsilent
```

### Phase 3: Build Pipeline

File: `scripts/build_windows_installer.ps1`

```powershell
# 1. Run PyInstaller via existing build_windows.ps1
# 2. Optionally copy checkpoint into dist/ if available
# 3. Run Inno Setup compiler: iscc scripts/windows/corridorkey-setup.iss
# 4. Output: CorridorKey-Setup-1.8.0.exe
```

Steps:
```powershell
# Build the app
powershell -File scripts\build_windows.ps1

# Compile installer (requires Inno Setup installed)
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" scripts\windows\corridorkey-setup.iss
```

### Phase 4: VC++ Redistributable (if needed)

PyTorch/CUDA DLLs may require the Visual C++ Redistributable.
Two options:
1. **Bundle** vc_redist.x64.exe and run it silently during install
2. **Skip** if torch already ships the needed DLLs (test on clean machine first)

If bundling:
```iss
[Files]
Source: "redist\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; \
  StatusMsg: "Installing Visual C++ Runtime..."; Check: VCRedistNeeded
```

---

## Model Weights Strategy (same as Mac)

Weights are NOT bundled in the installer. Same options:
1. **First-launch download** — GUI detects missing checkpoints, triggers download
2. **Separate download** — companion .zip with weights
3. **HuggingFace Hub** — auto-download on first use

The existing `scripts/setup_models.py` handles all download logic.

---

## Phase 5: GitHub Actions CI (optional)

File: `.github/workflows/build-windows.yml`

```yaml
name: Build Windows Installer
on:
  workflow_dispatch:
  release:
    types: [published]

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m venv .venv
          .venv\Scripts\activate
          pip install -e .
          pip install pyinstaller
      - name: Build with PyInstaller
        run: |
          .venv\Scripts\activate
          pyinstaller corridorkey.spec --noconfirm
      - name: Install Inno Setup
        run: choco install innosetup -y
      - name: Build installer
        run: |
          & "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" scripts\windows\corridorkey-setup.iss
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: CorridorKey-Windows
          path: "Output/*.exe"
```

---

## Code Signing (optional but recommended)

Windows code signing prevents SmartScreen warnings ("Windows protected your PC").

Options:
- **EV Code Signing Certificate** (~$300-500/year) — immediate SmartScreen trust
- **Standard Code Signing Certificate** (~$70-200/year) — builds trust over downloads
- **Self-signed** — not useful, SmartScreen still warns

```powershell
# Sign with signtool (from Windows SDK)
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 \
  /f certificate.pfx /p PASSWORD \
  "Output\CorridorKey-Setup-1.8.0.exe"
```

---

## Checklist

- [ ] Test current PyInstaller build on clean Windows (no Python installed)
- [ ] Install Inno Setup 6
- [ ] Create `scripts/windows/corridorkey-setup.iss`
- [ ] Create `scripts/build_windows_installer.ps1` (full pipeline)
- [ ] Test installer on clean Windows VM
- [ ] Verify Add/Remove Programs entry + uninstall works
- [ ] Test that CorridorKey.exe launches from Program Files
- [ ] Check if VC++ redistributable is needed
- [ ] (Optional) Code sign the installer
- [ ] (Optional) Set up GitHub Actions CI
- [ ] (Optional) First-run model download in GUI

## Estimated Bundle Sizes

| Component | Size |
|-----------|------|
| Python runtime | ~30 MB |
| PySide6 | ~150 MB |
| PyTorch + CUDA libs | ~800 MB - 2 GB |
| Triton | ~100 MB |
| App code + assets | ~5 MB |
| **Total installer (compressed)** | **~500 MB - 1 GB** |
| CorridorKey checkpoint | +383 MB (separate) |
| GVM weights | +6 GB (separate) |

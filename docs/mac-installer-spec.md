# Mac Installer Spec — CorridorKey .app + .pkg

## Goal
One-click installer (.pkg) that puts CorridorKey.app into /Applications on macOS.
User double-clicks CorridorKey, it launches the PySide6 GUI. No terminal, no Python install, no pip.

---

## Current State (What Already Works)

| Item | Status | Location |
|------|--------|----------|
| MLX backend auto-detection | Done | `CorridorKeyModule/backend.py:29-51` |
| Apple Silicon GPU monitor (psutil) | Done | `ui/workers/gpu_monitor.py:165-194` |
| MPS fallback (non-MLX torch) | Done | `backend/service/core.py:251-267` |
| pynvml graceful failure on Mac | Done | `ui/workers/gpu_monitor.py:50-78` |
| torch.compile disabled on MPS | Done | `CorridorKeyModule/inference_engine.py:124-127` |
| PyInstaller frozen detection | Done | `main.py:52-73` |
| corridorkey-mlx optional dep | Done | `pyproject.toml:43` |
| macOS CI smoke test | Done | `.github/workflows/fresh-install-smoke.yml` |
| .ico icon (Windows) | Done | `ui/theme/corridorkey.ico` |
| .icns icon (Mac) | **MISSING** | Need to generate from corridorkey.png |

---

## What Needs to Be Built

### Phase 1: Prerequisites (one-time setup)

#### 1.1 Generate .icns icon
```bash
# On Mac — convert corridorkey.png to .icns
mkdir corridorkey.iconset
sips -z 16 16     corridorkey.png --out corridorkey.iconset/icon_16x16.png
sips -z 32 32     corridorkey.png --out corridorkey.iconset/icon_16x16@2x.png
sips -z 32 32     corridorkey.png --out corridorkey.iconset/icon_32x32.png
sips -z 64 64     corridorkey.png --out corridorkey.iconset/icon_32x32@2x.png
sips -z 128 128   corridorkey.png --out corridorkey.iconset/icon_128x128.png
sips -z 256 256   corridorkey.png --out corridorkey.iconset/icon_128x128@2x.png
sips -z 256 256   corridorkey.png --out corridorkey.iconset/icon_256x256.png
sips -z 512 512   corridorkey.png --out corridorkey.iconset/icon_256x256@2x.png
sips -z 512 512   corridorkey.png --out corridorkey.iconset/icon_512x512.png
sips -z 1024 1024 corridorkey.png --out corridorkey.iconset/icon_512x512@2x.png
iconutil -c icns corridorkey.iconset -o ui/theme/corridorkey.icns
rm -rf corridorkey.iconset
```

#### 1.2 Create entitlements file
File: `scripts/macos/CorridorKey.entitlements`
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- Required for Hardened Runtime + PyInstaller -->
    <key>com.apple.security.cs.allow-jit</key>
    <true/>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
    <!-- File access for reading clips / writing output -->
    <key>com.apple.security.files.user-selected.read-write</key>
    <true/>
    <!-- GPU (Metal) access for MLX inference -->
    <key>com.apple.security.device.gpu</key>
    <true/>
    <!-- Network for model downloads (HuggingFace Hub) -->
    <key>com.apple.security.network.client</key>
    <true/>
</dict>
</plist>
```

#### 1.3 Create Info.plist template
File: `scripts/macos/Info.plist`
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>CorridorKey</string>
    <key>CFBundleDisplayName</key>
    <string>EZ-CorridorKey</string>
    <key>CFBundleIdentifier</key>
    <string>com.corridordigital.corridorkey</string>
    <key>CFBundleVersion</key>
    <string>1.8.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.8.0</string>
    <key>CFBundleExecutable</key>
    <string>CorridorKey</string>
    <key>CFBundleIconFile</key>
    <string>corridorkey.icns</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeExtensions</key>
            <array>
                <string>mp4</string>
                <string>mov</string>
                <string>avi</string>
                <string>mkv</string>
                <string>webm</string>
            </array>
            <key>CFBundleTypeName</key>
            <string>Video File</string>
            <key>CFBundleTypeRole</key>
            <string>Editor</string>
        </dict>
    </array>
</dict>
</plist>
```

---

### Phase 2: PyInstaller Spec for macOS

File: `corridorkey-macos.spec`

Key differences from Windows spec:
- **BUNDLE()** target to create .app instead of bare executable
- **.icns** icon instead of .ico
- **Exclude** nvidia-ml-py, triton-windows, CUDA libs
- **Include** corridorkey-mlx (if installed) — dynamically imported
- **target_arch='arm64'** — Apple Silicon only (MLX requires it)
- **argv_emulation=True** — allows drag-and-drop file opening on macOS

Dependencies to EXCLUDE (not needed on Mac):
- `nvidia-ml-py` / `pynvml` (NVML is NVIDIA-only, already handled gracefully)
- `triton-windows` (Windows-only)
- CUDA-specific torch libs (torch ships Metal/MPS support built-in)

Dependencies to INCLUDE (Mac-specific):
- `corridorkey-mlx` (optional but recommended for Apple Silicon)
- `psutil` (used for Apple Silicon memory reporting)

---

### Phase 3: Build Script

File: `scripts/build_macos.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. Activate venv
# 2. Install pyinstaller + corridorkey-mlx
# 3. Run: pyinstaller corridorkey-macos.spec --noconfirm
# 4. Create checkpoint directory placeholder in dist/
# 5. Report .app location and size
```

---

### Phase 4: Code Signing + Notarization

Requires: Apple Developer ID Application certificate in Keychain.

#### 4.1 Sign the .app
```bash
codesign --deep --force --options runtime \
  --entitlements scripts/macos/CorridorKey.entitlements \
  --sign "Developer ID Application: YOUR NAME (TEAM_ID)" \
  dist/CorridorKey.app
```

#### 4.2 Notarize
```bash
# Create zip for notarization
ditto -c -k --keepParent dist/CorridorKey.app CorridorKey.zip

# Submit
xcrun notarytool submit CorridorKey.zip \
  --apple-id YOUR_APPLE_ID \
  --team-id TEAM_ID \
  --password @keychain:AC_PASSWORD \
  --wait

# Staple the ticket
xcrun stapler staple dist/CorridorKey.app
```

#### 4.3 Build .pkg installer
```bash
# Build component package
pkgbuild --component dist/CorridorKey.app \
  --install-location /Applications \
  --sign "Developer ID Installer: YOUR NAME (TEAM_ID)" \
  CorridorKey-1.8.0.pkg
```

Or use `productbuild` for a branded installer with license + intro:
```bash
productbuild --distribution scripts/macos/distribution.xml \
  --resources scripts/macos/resources \
  --sign "Developer ID Installer: YOUR NAME (TEAM_ID)" \
  CorridorKey-1.8.0.pkg
```

---

### Phase 5: Model Weights Strategy

Weights are NOT bundled (too large: 383MB CorridorKey + 6GB GVM/VideoMaMa).

Options (pick one):
1. **First-launch download** — app detects missing checkpoints, triggers `setup_models.py` logic
2. **Separate DMG/zip** — ship weights as a companion download
3. **HuggingFace Hub** — `huggingface_hub` already in deps, auto-download on first use

Recommendation: **Option 1** — the app already has `scripts/setup_models.py` with download logic.
Wire it into the GUI as a first-run setup wizard or progress dialog.

---

### Phase 6: GitHub Actions CI (optional but recommended)

File: `.github/workflows/build-macos.yml`

```yaml
name: Build macOS Installer
on:
  workflow_dispatch:
  release:
    types: [published]

jobs:
  build:
    runs-on: macos-latest  # Apple Silicon (M1) runner
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -e ".[mlx]"
          pip install pyinstaller
      - name: Build .app
        run: |
          source .venv/bin/activate
          pyinstaller corridorkey-macos.spec --noconfirm
      - name: Sign and notarize
        env:
          APPLE_CERT_BASE64: ${{ secrets.APPLE_CERT_BASE64 }}
          APPLE_CERT_PASSWORD: ${{ secrets.APPLE_CERT_PASSWORD }}
          APPLE_ID: ${{ secrets.APPLE_ID }}
          APPLE_TEAM_ID: ${{ secrets.APPLE_TEAM_ID }}
          APPLE_APP_PASSWORD: ${{ secrets.APPLE_APP_PASSWORD }}
        run: scripts/macos/sign_and_notarize.sh
      - name: Build .pkg
        run: |
          pkgbuild --component dist/CorridorKey.app \
            --install-location /Applications \
            CorridorKey-${{ github.ref_name }}.pkg
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: CorridorKey-macOS
          path: "*.pkg"
```

---

## Checklist

- [ ] Generate .icns from corridorkey.png
- [ ] Create `scripts/macos/` directory with entitlements + Info.plist
- [ ] Create `corridorkey-macos.spec` (PyInstaller)
- [ ] Create `scripts/build_macos.sh`
- [ ] Test build on Mac (local)
- [ ] Test .app launches and runs inference
- [ ] Code sign with Developer ID
- [ ] Notarize with Apple
- [ ] Build .pkg
- [ ] Test .pkg install on clean Mac
- [ ] (Optional) Set up GitHub Actions CI
- [ ] (Optional) First-run model download wizard in GUI

## Estimated Bundle Sizes

| Component | Size |
|-----------|------|
| Python runtime | ~30 MB |
| PySide6 | ~150 MB |
| PyTorch (CPU/MPS, no CUDA) | ~200 MB |
| corridorkey-mlx | ~20 MB |
| App code + assets | ~5 MB |
| **Total .app** | **~400-500 MB** |
| CorridorKey checkpoint | +383 MB (separate) |
| GVM weights | +6 GB (separate) |

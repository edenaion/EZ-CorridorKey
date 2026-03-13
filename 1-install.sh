#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo " ========================================"
echo "  EZ-CorridorKey - One-Click Installer"
echo " ========================================"
echo ""

PYTHON_SPEC="${CORRIDORKEY_PYTHON_VERSION:-3.11}"
INSTALL_PYTHON=""
INDEX_URL=""

resolve_system_python() {
    local candidate ver major minor
    for candidate in python3 python; do
        if ! command -v "$candidate" >/dev/null 2>&1; then
            continue
        fi
        ver="$("$candidate" --version 2>&1 | awk '{print $2}')"
        major="$(echo "$ver" | cut -d. -f1)"
        minor="$(echo "$ver" | cut -d. -f2)"
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -lt 14 ]; then
            INSTALL_PYTHON="$candidate"
            return 0
        fi
    done
    return 1
}

echo "[0/6] Installer runtime target..."
echo "  CorridorKey will provision and use Python ${PYTHON_SPEC} for this install."

# ── Step 2: Check for old venv ──
if [ -d "venv" ] && [ ! -d ".venv" ]; then
    echo ""
    echo "  [NOTE] Found old 'venv' directory from previous installer."
    echo "  The new installer uses '.venv'. You can safely delete 'venv' later."
    echo ""
fi

# ── Step 3: Install/locate uv ──
echo "[2/6] Setting up package manager..."
UV_AVAILABLE=0

if command -v uv &>/dev/null; then
    UV_AVAILABLE=1
    echo "  [OK] uv found"
elif [ -f "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
    UV_AVAILABLE=1
    echo "  [OK] uv found at ~/.local/bin"
else
    echo "  Installing uv (fast Python package manager)..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null; then
        export PATH="$HOME/.local/bin:$PATH"
        if command -v uv &>/dev/null; then
            UV_AVAILABLE=1
            echo "  [OK] uv installed"
        fi
    fi
    if [ "$UV_AVAILABLE" -eq 0 ]; then
        echo "  [WARN] uv install failed."
    fi
fi

OS_TYPE="linux"
case "$(uname -s)" in
    Darwin*) OS_TYPE="macos" ;;
esac

# ── Step 4: Provision Python ──
echo "[3/6] Provisioning Python..."
if [ "$UV_AVAILABLE" -eq 1 ]; then
    if uv python install --managed-python "$PYTHON_SPEC" >/dev/null 2>&1; then
        INSTALL_PYTHON="$(uv python find --managed-python "$PYTHON_SPEC")"
    else
        echo "  [WARN] Managed Python download failed, trying a supported system Python..."
    fi
fi

if [ -z "$INSTALL_PYTHON" ]; then
    if ! resolve_system_python; then
        echo "  [ERROR] Could not find a supported Python runtime."
        echo "  CorridorKey needs Python 3.10-3.13, and the one-click installer targets Python ${PYTHON_SPEC}."
        case "$OS_TYPE" in
            macos) echo "  Install via: brew install python@3.11" ;;
            *)     echo "  Install via: your package manager, then rerun this installer." ;;
        esac
        exit 1
    fi
fi
PYVER="$("$INSTALL_PYTHON" --version 2>&1 | awk '{print $2}')"
echo "  [OK] Using Python $PYVER ($INSTALL_PYTHON)"

# ── Step 5: Create venv + install dependencies ──
echo "[4/6] Installing dependencies..."
if [ "$UV_AVAILABLE" -eq 1 ]; then
    echo "  Creating virtual environment..."
    uv venv --clear --managed-python --python "$INSTALL_PYTHON" .venv >/dev/null 2>&1
    echo "  Installing packages (uv + auto CUDA detection)..."
    if uv pip install --python .venv/bin/python --torch-backend=auto -e . 2>&1; then
        echo "  [OK] Dependencies installed via uv"
    else
        echo "  [WARN] uv install failed, trying pip fallback..."
        UV_AVAILABLE=0
    fi
fi

if [ "$UV_AVAILABLE" -eq 0 ]; then
    echo "  Creating virtual environment..."
    rm -rf .venv
    "$INSTALL_PYTHON" -m venv .venv
    source .venv/bin/activate

    if [ "$OS_TYPE" = "macos" ]; then
        # macOS: default PyTorch wheels include MPS support, no special index needed
        echo "  macOS detected — PyTorch will use MPS (Apple Silicon) or CPU"
    elif command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
        if [ -n "$CUDA_VER" ]; then
            CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
            CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
            echo "  CUDA $CUDA_VER detected"
            if [ "$CUDA_MAJOR" -ge 13 ]; then
                INDEX_URL="https://download.pytorch.org/whl/cu130"
            elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 6 ]; then
                INDEX_URL="https://download.pytorch.org/whl/cu128"
            elif [ "$CUDA_MAJOR" -eq 12 ]; then
                INDEX_URL="https://download.pytorch.org/whl/cu126"
            fi
        fi
    fi

    if [ -z "$INDEX_URL" ] && [ "$OS_TYPE" != "macos" ]; then
        echo "  No NVIDIA GPU detected, installing CPU-only PyTorch"
        INDEX_URL="https://download.pytorch.org/whl/cpu"
    fi

    echo "  Installing packages via pip (this may take a few minutes)..."
    .venv/bin/python -m pip install --upgrade pip >/dev/null 2>&1
    if [ -n "$INDEX_URL" ]; then
        .venv/bin/python -m pip install --extra-index-url "$INDEX_URL" -e . 2>&1
    else
        .venv/bin/python -m pip install -e . 2>&1
    fi
    echo "  [OK] Dependencies installed via pip"
fi

echo "[4b/6] Verifying torch runtime..."
DIAGNOSTICS_DIR="logs/diagnostics"
TORCH_RUNTIME_LOG="${DIAGNOSTICS_DIR}/install-torch-runtime.json"
SUPPORT_REPORT_LOG="${DIAGNOSTICS_DIR}/install-support-report.md"
mkdir -p "$DIAGNOSTICS_DIR"
if .venv/bin/python scripts/verify_torch_runtime.py --log "$TORCH_RUNTIME_LOG"; then
    echo "  [OK] Torch runtime verified"
else
    echo "  [ERROR] Installed torch runtime did not validate."
    echo "  Preparing a pre-filled GitHub issue to help you report this."
    if ! .venv/bin/python scripts/open_installer_issue.py --json "$TORCH_RUNTIME_LOG" --body-out "$SUPPORT_REPORT_LOG" --stage "${OS_TYPE}-installer-runtime-verification"; then
        echo "  [WARN] Could not open the GitHub issue helper automatically."
    fi
    echo "  See $TORCH_RUNTIME_LOG and $SUPPORT_REPORT_LOG for details."
    exit 1
fi

# ── Step 4c: MLX acceleration for Apple Silicon ──
if [ "$OS_TYPE" = "macos" ] && [ "$(uname -m)" = "arm64" ]; then
    echo "[4c/6] Installing MLX acceleration for Apple Silicon..."
    MLX_OK=0
    if [ "$UV_AVAILABLE" -eq 1 ]; then
        if uv pip install --python .venv/bin/python -e ".[mlx]" 2>&1; then
            MLX_OK=1
        fi
    else
        if .venv/bin/python -m pip install -e ".[mlx]" 2>&1; then
            MLX_OK=1
        fi
    fi
    if [ "$MLX_OK" -eq 1 ]; then
        echo "  [OK] MLX acceleration installed (1.5-2x faster inference on Apple Silicon)"
        echo "  Downloading MLX checkpoint..."
        .venv/bin/python scripts/setup_models.py --corridorkey-mlx 2>&1 || \
            echo "  [WARN] MLX checkpoint download failed. You can retry: .venv/bin/python scripts/setup_models.py --corridorkey-mlx"
    else
        echo "  [WARN] MLX install failed. CorridorKey will use PyTorch MPS instead."
        echo "  You can retry later with: .venv/bin/python -m pip install -e '.[mlx]'"
    fi
else
    echo "[4c/6] MLX acceleration — skipped (Apple Silicon only)"
fi

echo "[4d/6] Optional SAM2 tracker..."
INSTALL_SAM2="y"
if [ "$OS_TYPE" = "macos" ]; then
    echo "  [NOTE] SAM2 tracking on macOS is experimental in CorridorKey."
fi
INSTALL_SAM2_INPUT="${CORRIDORKEY_INSTALL_SAM2:-}"
if [ -z "$INSTALL_SAM2_INPUT" ]; then
    read -rp "  Install SAM2 tracking support? [Y/n]: " INSTALL_SAM2_INPUT
fi
if [[ "$(echo "${INSTALL_SAM2_INPUT:-y}" | tr '[:upper:]' '[:lower:]')" == "n" ]]; then
    INSTALL_SAM2="n"
fi

if [ "$INSTALL_SAM2" = "y" ]; then
    SAM2_OK=0
    echo "  Installing SAM2 tracker package..."
    if [ "$UV_AVAILABLE" -eq 1 ]; then
        if uv pip install --python .venv/bin/python --torch-backend=auto -e ".[tracker]" 2>&1; then
            SAM2_OK=1
        fi
    else
        if [ -n "${INDEX_URL:-}" ]; then
            if .venv/bin/python -m pip install --extra-index-url "$INDEX_URL" -e ".[tracker]" 2>&1; then
                SAM2_OK=1
            fi
        else
            if .venv/bin/python -m pip install -e ".[tracker]" 2>&1; then
                SAM2_OK=1
            fi
        fi
    fi

    if [ "$SAM2_OK" -eq 1 ]; then
        echo "  [OK] SAM2 tracker support installed"
        DOWNLOAD_SAM2="${CORRIDORKEY_PREDOWNLOAD_SAM2:-}"
        if [ -z "$DOWNLOAD_SAM2" ]; then
            read -rp "  Pre-download default SAM2 Base+ model? (324MB) [Y/n]: " DOWNLOAD_SAM2
        fi
        if [[ "$(echo "${DOWNLOAD_SAM2:-y}" | tr '[:upper:]' '[:lower:]')" != "n" ]]; then
            .venv/bin/python scripts/setup_models.py --sam2
        fi
    else
        echo "  [WARN] SAM2 tracker install failed. CorridorKey will still run without Track Mask."
        echo "  You can retry later with:"
        echo "    .venv/bin/python -m pip install -e '.[tracker]'"
    fi
fi

# ── Step 6: Check FFmpeg ──
echo "[5/6] Checking FFmpeg..."
if .venv/bin/python scripts/check_ffmpeg.py; then
    :
else
    echo "  [WARN] Video import/export requires FFmpeg 7.0+ and FFprobe."
    case "$OS_TYPE" in
        macos)
            if command -v brew &>/dev/null; then
                INSTALL_FFMPEG="${CORRIDORKEY_INSTALL_FFMPEG:-}"
                if [ -z "$INSTALL_FFMPEG" ]; then
                    if [ -t 0 ]; then
                        read -rp "  FFmpeg not found. Install via Homebrew? [Y/n]: " INSTALL_FFMPEG
                    else
                        INSTALL_FFMPEG="y"
                        echo "  Non-interactive shell detected — defaulting to Homebrew FFmpeg install."
                    fi
                fi
                if [[ "$(echo "${INSTALL_FFMPEG:-y}" | tr '[:upper:]' '[:lower:]')" != "n" ]]; then
                    echo "  Installing FFmpeg via Homebrew..."
                    brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
                    if .venv/bin/python scripts/check_ffmpeg.py; then
                        echo "  [OK] FFmpeg installed successfully"
                    else
                        echo "  [WARN] FFmpeg installed but validation failed."
                        echo "  Try: brew reinstall ffmpeg"
                    fi
                else
                    echo "  Skipped. Install later via: brew install ffmpeg"
                fi
            else
                echo "  Homebrew not found. Install Homebrew first:"
                echo "    /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                echo "  Then: brew install ffmpeg"
            fi ;;
        *)
            if [ -f /etc/debian_version ]; then
                echo "  Install via: sudo apt install ffmpeg"
            elif [ -f /etc/fedora-release ]; then
                echo "  Install via: sudo dnf install ffmpeg"
            elif [ -f /etc/arch-release ]; then
                echo "  Install via: sudo pacman -S ffmpeg"
            else
                echo "  Install via your package manager or https://ffmpeg.org/download.html"
            fi ;;
    esac
    echo "  Verify both commands afterward:"
    echo "    ffmpeg -version"
    echo "    ffprobe -version"
    echo ""
fi

# ── Step 7: Download model weights ──
echo "[6/6] Checking model weights..."
.venv/bin/python scripts/setup_models.py --check
.venv/bin/python scripts/setup_models.py --corridorkey
if [ $? -ne 0 ]; then
    echo "  [WARN] CorridorKey model download failed. Retry later:"
    echo "    .venv/bin/python scripts/setup_models.py --corridorkey"
fi

echo ""
echo "[6/6] Optional models (can be downloaded later)"
echo ""

INSTALL_GVM="${CORRIDORKEY_INSTALL_GVM:-}"
if [ -z "$INSTALL_GVM" ]; then
    read -rp "  Download GVM alpha generator? (~6GB) [y/N]: " INSTALL_GVM
fi
if [[ "$(echo "$INSTALL_GVM" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
    .venv/bin/python scripts/setup_models.py --gvm
fi

if [ "$OS_TYPE" = "macos" ]; then
    echo ""
echo "  [NOTE] VideoMaMa runs on CPU on macOS (no MPS support yet)."
echo "  It works but will be slow. 37GB download — skip if unsure."
fi
INSTALL_VM="${CORRIDORKEY_INSTALL_VIDEOMAMA:-}"
if [ -z "$INSTALL_VM" ]; then
    read -rp "  Download VideoMaMa alpha generator? (~37GB) [y/N]: " INSTALL_VM
fi
if [[ "$(echo "$INSTALL_VM" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
    .venv/bin/python scripts/setup_models.py --videomama
fi

# ── Create desktop shortcut ──
echo ""
CREATE_SHORTCUT="${CORRIDORKEY_CREATE_SHORTCUT:-}"
if [ -z "$CREATE_SHORTCUT" ]; then
    read -rp "  Create desktop shortcut? [Y/n]: " CREATE_SHORTCUT
fi
if [[ "$(echo "$CREATE_SHORTCUT" | tr '[:upper:]' '[:lower:]')" != "n" ]]; then
    ICON_PATH="$SCRIPT_DIR/ui/theme/corridorkey.png"
    if [ "$OS_TYPE" = "macos" ]; then
        # macOS: create a minimal .app bundle on Desktop (no Terminal window)
        APP_DIR="$HOME/Desktop/CorridorKey.app/Contents/MacOS"
        mkdir -p "$APP_DIR"
        mkdir -p "$HOME/Desktop/CorridorKey.app/Contents/Resources"
        # Copy icon if available
        if [ -f "$ICON_PATH" ]; then
            cp "$ICON_PATH" "$HOME/Desktop/CorridorKey.app/Contents/Resources/corridorkey.png"
        fi
        cat > "$HOME/Desktop/CorridorKey.app/Contents/Info.plist" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>CorridorKey</string>
    <key>CFBundleExecutable</key><string>launch</string>
    <key>CFBundleIconFile</key><string>corridorkey</string>
    <key>LSUIElement</key><false/>
</dict>
</plist>
PLISTEOF
        cat > "$APP_DIR/launch" <<LAUNCHEOF
#!/usr/bin/env bash
cd "$SCRIPT_DIR"
.venv/bin/python main.py
LAUNCHEOF
        chmod +x "$APP_DIR/launch"
        if [ -d "$HOME/Desktop/CorridorKey.app" ]; then
            echo "  [OK] Desktop app created (CorridorKey.app — no Terminal window)"
            echo "  Tip: drag it to the Dock for quick access"
        else
            echo "  [WARN] App creation failed"
        fi
    else
        # Linux: create a .desktop file
        DESKTOP_FILE="$HOME/.local/share/applications/corridorkey.desktop"
        mkdir -p "$HOME/.local/share/applications"
        cat > "$DESKTOP_FILE" <<DSKEOF
[Desktop Entry]
Name=CorridorKey
Comment=AI Green Screen Keyer
Exec=$SCRIPT_DIR/2-start.sh
Icon=$ICON_PATH
Terminal=false
Type=Application
Categories=Graphics;Video;
DSKEOF
        # Also copy to Desktop if it exists
        if [ -d "$HOME/Desktop" ]; then
            cp "$DESKTOP_FILE" "$HOME/Desktop/CorridorKey.desktop"
            chmod +x "$HOME/Desktop/CorridorKey.desktop"
        fi
        if [ -f "$DESKTOP_FILE" ]; then
            echo "  [OK] Desktop shortcut created — also added to app menu"
        else
            echo "  [WARN] Shortcut creation failed"
        fi
    fi
fi

# ── Done ──
echo ""
echo " ========================================"
echo "  Installation complete!"
echo " ========================================"
echo ""
echo "  To launch: ./2-start.sh (or the desktop shortcut)"
echo ""
echo "  To download optional models later:"
echo "    .venv/bin/python scripts/setup_models.py --gvm"
echo "    .venv/bin/python scripts/setup_models.py --videomama"
echo ""

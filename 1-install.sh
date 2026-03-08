#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo " ========================================"
echo "  EZ-CorridorKey - One-Click Installer"
echo " ========================================"
echo ""

# ── Step 1: Check Python ──
echo "[1/6] Checking Python..."
PYTHON=""
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
fi

if [ -z "$PYTHON" ]; then
    echo "  [ERROR] Python not found."
    case "$(uname -s)" in
        Darwin*) echo "  Install via: brew install python@3.11" ;;
        *)       echo "  Install via: sudo apt install python3 python3-venv (Debian/Ubuntu)"
                 echo "           or: sudo dnf install python3 (Fedora)" ;;
    esac
    exit 1
fi

PYVER=$($PYTHON --version 2>&1 | awk '{print $2}')
PYMAJOR=$(echo "$PYVER" | cut -d. -f1)
PYMINOR=$(echo "$PYVER" | cut -d. -f2)

if [ "$PYMAJOR" -lt 3 ] || ([ "$PYMAJOR" -eq 3 ] && [ "$PYMINOR" -lt 10 ]); then
    echo "  [ERROR] Python 3.10+ required, found $PYVER"
    exit 1
fi
echo "  [OK] Python $PYVER ($PYTHON)"

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
        echo "  [WARN] uv install failed, falling back to pip (slower)"
    fi
fi

# ── Step 4: Create venv + install dependencies ──
echo "[3/6] Installing dependencies..."

OS_TYPE="linux"
case "$(uname -s)" in
    Darwin*) OS_TYPE="macos" ;;
esac

if [ "$UV_AVAILABLE" -eq 1 ]; then
    if [ ! -d ".venv" ]; then
        echo "  Creating virtual environment..."
        uv venv .venv >/dev/null 2>&1
    fi
    echo "  Installing packages (uv + auto CUDA detection)..."
    if uv pip install --python .venv/bin/python --torch-backend=auto -e . 2>&1; then
        echo "  [OK] Dependencies installed via uv"
    else
        echo "  [WARN] uv install failed, trying pip fallback..."
        UV_AVAILABLE=0
    fi
fi

if [ "$UV_AVAILABLE" -eq 0 ]; then
    if [ ! -d ".venv" ]; then
        echo "  Creating virtual environment..."
        $PYTHON -m venv .venv
    fi
    source .venv/bin/activate

    INDEX_URL=""
    if [ "$OS_TYPE" = "macos" ]; then
        # macOS: default PyTorch wheels include MPS support, no special index needed
        echo "  macOS detected — PyTorch will use MPS (Apple Silicon) or CPU"
    elif command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
        if [ -n "$CUDA_VER" ]; then
            CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
            CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
            echo "  CUDA $CUDA_VER detected"
            if [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
                INDEX_URL="https://download.pytorch.org/whl/cu128"
            elif [ "$CUDA_MAJOR" -eq 12 ]; then
                INDEX_URL="https://download.pytorch.org/whl/cu121"
            elif [ "$CUDA_MAJOR" -eq 11 ]; then
                INDEX_URL="https://download.pytorch.org/whl/cu118"
            fi
        fi
    fi

    if [ -z "$INDEX_URL" ] && [ "$OS_TYPE" != "macos" ]; then
        echo "  No NVIDIA GPU detected, installing CPU-only PyTorch"
        INDEX_URL="https://download.pytorch.org/whl/cpu"
    fi

    echo "  Installing packages via pip (this may take a few minutes)..."
    pip install --upgrade pip >/dev/null 2>&1
    if [ -n "$INDEX_URL" ]; then
        pip install --extra-index-url "$INDEX_URL" -e . 2>&1
    else
        pip install -e . 2>&1
    fi
    echo "  [OK] Dependencies installed via pip"
fi

# ── Step 5: Check FFmpeg ──
echo "[4/6] Checking FFmpeg..."
if command -v ffmpeg &>/dev/null; then
    echo "  [OK] FFmpeg found"
else
    echo "  [WARN] FFmpeg not found. Video import requires FFmpeg."
    case "$OS_TYPE" in
        macos)
            echo "  Install via: brew install ffmpeg" ;;
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
    echo ""
fi

# ── Step 6: Download model weights ──
echo "[5/6] Checking model weights..."
.venv/bin/python scripts/setup_models.py --check
.venv/bin/python scripts/setup_models.py --corridorkey
if [ $? -ne 0 ]; then
    echo "  [WARN] CorridorKey model download failed. Retry later:"
    echo "    .venv/bin/python scripts/setup_models.py --corridorkey"
fi

echo ""
echo "[6/6] Optional models (can be downloaded later)"
echo ""

read -rp "  Download GVM alpha generator? (~6GB) [y/N]: " INSTALL_GVM
if [[ "$(echo "$INSTALL_GVM" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
    .venv/bin/python scripts/setup_models.py --gvm
fi

if [ "$OS_TYPE" = "macos" ]; then
    echo ""
    echo "  [NOTE] VideoMaMa runs on CPU on macOS (no MPS support yet)."
    echo "  It works but will be slow. 37GB download — skip if unsure."
fi
read -rp "  Download VideoMaMa alpha generator? (~37GB) [y/N]: " INSTALL_VM
if [[ "$(echo "$INSTALL_VM" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
    .venv/bin/python scripts/setup_models.py --videomama
fi

# ── Create desktop shortcut ──
echo ""
read -rp "  Create desktop shortcut? [Y/n]: " CREATE_SHORTCUT
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

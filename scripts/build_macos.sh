#!/usr/bin/env bash
# CorridorKey macOS Build Script
# Usage: bash scripts/build_macos.sh
#
# Prerequisites:
#   - macOS with Apple Silicon (M1/M2/M3/M4)
#   - Python 3.11+ (brew install python@3.11)
#   - Xcode Command Line Tools (xcode-select --install)
#
# Output: dist/CorridorKey.app
#
# Post-build: checkpoints are downloaded on first launch via setup_models.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo ""
echo "=== CorridorKey macOS Build ==="
echo "Project root: $ROOT"
echo ""

# --- Check architecture ---
ARCH="$(uname -m)"
if [ "$ARCH" != "arm64" ]; then
    echo "WARNING: Building on $ARCH — MLX requires Apple Silicon (arm64)"
    echo "The build will proceed but MLX backend won't be available."
fi

# --- Find Python 3.11+ ---
PYTHON=""
for candidate in python3.11 python3.12 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver="$("$candidate" --version 2>&1 | awk '{print $2}')"
        major="$(echo "$ver" | cut -d. -f1)"
        minor="$(echo "$ver" | cut -d. -f2)"
        if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && [ "$minor" -lt 14 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.11-3.13 not found."
    echo "  Install with: brew install python@3.11"
    exit 1
fi
echo "Using Python: $PYTHON ($($PYTHON --version))"

# --- Create/activate venv ---
VENV_DIR="$ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"
echo "Activated venv: $VENV_DIR"

# --- Install dependencies ---
echo ""
echo "Installing dependencies..."
"$PIP" install --upgrade pip >/dev/null

# Install torch (CPU/MPS — no CUDA on macOS)
"$PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
    "$PIP" install torch torchvision

# Install project with MLX extra
"$PIP" install -e ".[mlx]" 2>/dev/null || "$PIP" install -e .

# Install PyInstaller
"$PIP" install pyinstaller

# --- Generate .icns if missing ---
ICNS="$ROOT/ui/theme/corridorkey.icns"
if [ ! -f "$ICNS" ]; then
    echo ""
    echo "Generating .icns icon..."
    bash "$ROOT/scripts/macos/generate_icns.sh"
fi

# --- Clean previous build ---
if [ -d "$ROOT/dist/CorridorKey.app" ]; then
    echo "Cleaning previous build..."
    rm -rf "$ROOT/dist/CorridorKey.app"
fi
if [ -d "$ROOT/build/CorridorKey" ]; then
    rm -rf "$ROOT/build/CorridorKey"
fi

# --- Build ---
echo ""
echo "Building with PyInstaller..."
"$VENV_DIR/bin/pyinstaller" corridorkey-macos.spec --noconfirm

if [ ! -d "$ROOT/dist/CorridorKey.app" ]; then
    echo "ERROR: Build failed — dist/CorridorKey.app not found"
    exit 1
fi

# --- Create checkpoint directory placeholder ---
CKPT_DIR="$ROOT/dist/CorridorKey.app/Contents/MacOS/CorridorKeyModule/checkpoints"
mkdir -p "$CKPT_DIR"

# --- Copy checkpoint if available locally ---
SRC_CKPT="$ROOT/CorridorKeyModule/checkpoints"
if [ -d "$SRC_CKPT" ]; then
    pth_count=$(find "$SRC_CKPT" -name "*.pth" -o -name "*.safetensors" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$pth_count" -gt 0 ]; then
        echo "Copying checkpoint(s)..."
        cp "$SRC_CKPT"/*.pth "$CKPT_DIR/" 2>/dev/null || true
        cp "$SRC_CKPT"/*.safetensors "$CKPT_DIR/" 2>/dev/null || true
    else
        echo "WARNING: No checkpoint found in $SRC_CKPT"
        echo "  Checkpoints will be downloaded on first launch."
    fi
else
    echo "WARNING: Checkpoint directory not found: $SRC_CKPT"
    echo "  Checkpoints will be downloaded on first launch."
fi

# --- Summary ---
APP_PATH="$ROOT/dist/CorridorKey.app"
APP_SIZE=$(du -sh "$APP_PATH" | awk '{print $1}')

echo ""
echo "=== Build Complete ==="
echo "  App: $APP_PATH ($APP_SIZE)"
echo "  Checkpoints: $CKPT_DIR"
echo ""
echo "To test:  open dist/CorridorKey.app"
echo "To sign:  bash scripts/macos/sign_and_notarize.sh"
echo ""

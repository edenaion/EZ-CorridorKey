#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo ""
echo "  ========================================"
echo "   EZ-CorridorKey — Update"
echo "  ========================================"
echo ""

# ── Step 1: Pull latest code ──
echo "[1/3] Pulling latest changes..."
if ! command -v git &>/dev/null; then
    echo "  [ERROR] Git not found. Install git first."
    exit 1
fi

if git pull --recurse-submodules 2>&1; then
    echo "  [OK] Code updated"
else
    echo "  [WARN] Git pull had issues. You may have local changes."
    echo "  If stuck, try: git stash && git pull && git stash pop"
fi

# ── Step 2: Update dependencies ──
echo "[2/3] Updating dependencies..."

UV_AVAILABLE=0
if command -v uv &>/dev/null; then
    UV_AVAILABLE=1
elif [ -f "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
    UV_AVAILABLE=1
elif [ -f "$HOME/.cargo/bin/uv" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
    UV_AVAILABLE=1
fi

if [ "$UV_AVAILABLE" = "1" ]; then
    if uv pip install --python .venv/bin/python --torch-backend=auto -e . 2>&1; then
        echo "  [OK] Dependencies updated via uv"
    else
        echo "  [WARN] uv update failed, trying pip..."
        UV_AVAILABLE=0
    fi
fi

if [ "$UV_AVAILABLE" = "0" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        pip install -e . 2>&1
        echo "  [OK] Dependencies updated via pip"
    else
        echo "  [ERROR] No .venv found. Run 1-install.sh first."
        exit 1
    fi
fi

# ── Step 3: Check for new model weights ──
echo "[3/3] Checking model weights..."
.venv/bin/python scripts/setup_models.py --check

# ── Done ──
echo ""
echo "  ========================================"
echo "   Update complete!"
echo "  ========================================"
echo ""
echo "  To launch: ./2-start.sh"
echo ""

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

CURRENT_BRANCH="$(git branch --show-current 2>/dev/null || true)"
CURRENT_UPSTREAM="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"

if [ "$CURRENT_BRANCH" = "master" ]; then
    echo "  [MIGRATE] Moving local branch from master to main..."
    git fetch origin main --recurse-submodules >/dev/null 2>&1 || true
    if git show-ref --verify --quiet refs/heads/main; then
        if git checkout main >/dev/null 2>&1; then
            :
        else
            echo "  [WARN] Could not switch to local main branch automatically."
            echo "  Finish this update, then run: git fetch origin && git checkout main"
        fi
    elif git branch -m master main >/dev/null 2>&1 || git checkout -b main origin/main >/dev/null 2>&1; then
        :
    else
        echo "  [WARN] Could not auto-migrate this checkout to main."
        echo "  Finish this update, then run: git fetch origin && git checkout main"
    fi
    if [ "$(git branch --show-current 2>/dev/null || true)" = "main" ]; then
        git branch --set-upstream-to=origin/main main >/dev/null 2>&1 || true
        echo "  [OK] Now tracking origin/main"
    fi
elif [ "$CURRENT_BRANCH" = "main" ] && [ "$CURRENT_UPSTREAM" = "origin/master" ]; then
    echo "  [MIGRATE] Repointing main to track origin/main..."
    git fetch origin main --recurse-submodules >/dev/null 2>&1 || true
    if git branch --set-upstream-to=origin/main main >/dev/null 2>&1; then
        echo "  [OK] Now tracking origin/main"
    fi
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
INSTALL_TARGET="-e ."
if command -v uv &>/dev/null; then
    UV_AVAILABLE=1
elif [ -f "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
    UV_AVAILABLE=1
elif [ -f "$HOME/.cargo/bin/uv" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
    UV_AVAILABLE=1
fi

EXTRAS=""
if [ -f ".venv/bin/python" ] && .venv/bin/python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('sam2') else 1)" >/dev/null 2>&1; then
    EXTRAS="tracker"
    echo "  SAM2 tracker detected — updating tracker extras too"
fi

# On Apple Silicon, include MLX extra if corridorkey_mlx is installed or platform matches
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    if [ -n "$EXTRAS" ]; then
        EXTRAS="${EXTRAS},mlx"
    else
        EXTRAS="mlx"
    fi
    echo "  Apple Silicon detected — including MLX acceleration"
fi

if [ -n "$EXTRAS" ]; then
    INSTALL_TARGET="-e .[${EXTRAS}]"
fi

if [ "$UV_AVAILABLE" = "1" ]; then
    if uv pip install --python .venv/bin/python --torch-backend=auto --reinstall-package corridorkey $INSTALL_TARGET 2>&1; then
        echo "  [OK] Dependencies updated via uv"
    else
        echo "  [WARN] uv update failed, trying pip..."
        UV_AVAILABLE=0
    fi
fi

if [ "$UV_AVAILABLE" = "0" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        pip install $INSTALL_TARGET 2>&1
        echo "  [OK] Dependencies updated via pip"
    else
        echo "  [ERROR] No .venv found. Run 1-install.sh first."
        exit 1
    fi
fi

# Clean up retired distributions that can survive editable upgrades.
if [ -f ".venv/bin/python" ] && .venv/bin/python -m pip show pynvml >/dev/null 2>&1; then
    echo "  Removing legacy pynvml package..."
    .venv/bin/python -m pip uninstall -y pynvml >/dev/null 2>&1 || true
fi

# ── Step 3: Check for new model weights ──
echo "[3/3] Checking model weights..."
.venv/bin/python scripts/setup_models.py --check

# Auto-download MLX checkpoint on Apple Silicon if missing
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    if [ ! -f "CorridorKeyModule/checkpoints/corridorkey_mlx.safetensors" ]; then
        echo "  MLX checkpoint missing — downloading..."
        .venv/bin/python scripts/setup_models.py --corridorkey-mlx 2>&1 || \
            echo "  [WARN] MLX checkpoint download failed. Run: .venv/bin/python scripts/setup_models.py --corridorkey-mlx"
    fi
fi

# ── Done ──
echo ""
echo "  ========================================"
echo "   Update complete!"
echo "  ========================================"
echo ""
echo "  To launch: ./2-start.sh"
echo ""

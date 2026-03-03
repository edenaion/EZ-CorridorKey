#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found. Run 1-install.sh first!"
    exit 1
fi

source .venv/bin/activate
python main.py "$@"

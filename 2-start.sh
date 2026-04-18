#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"



if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found. "
    echo "Attempting to run installation script."
    chmod +x "$SCRIPT_DIR/1-install.sh"
    $SCRIPT_DIR/1-install.sh
    if [ ! -d ".venv" ]; then
        echo "[ERROR] Installation failed. Please run 1-install.sh manually and check for errors."
        exit 1
    fi
fi

echo "[INFO] Checking for updates."

chmod +x "$SCRIPT_DIR/3-update.sh"
$SCRIPT_DIR/3-update.sh

source .venv/bin/activate
python main.py "$@"
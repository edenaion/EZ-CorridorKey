#!/usr/bin/env bash
# ============================================================
#  EZ-CorridorKey container entrypoint
#
#  First start:
#    1. Extracts app source from image tarball into /app (volume)
#    2. Runs 1-install.sh — venv, packages, and models all go
#       into /app on the volume
#  After that:
#    Sentinel skips straight to launching the app
#
#  To force a clean reinstall:
#    docker exec -it corridorkey-gpu rm /app/.install_complete
#    docker compose restart corridorkey-gpu
#
#  To update after a git pull + rebuild:
#    docker exec -it corridorkey-gpu rm /app/.install_complete
#    docker compose up corridorkey-gpu --build
# ============================================================
set -euo pipefail

SENTINEL="/app/.install_complete"

echo ""
echo "  ======================================================"
echo "   EZ-CorridorKey  —  starting up"
echo "  ======================================================"
echo ""
echo "  noVNC        →  http://localhost:6080  (auto-connects)"
echo "  File upload  →  http://localhost:6081"
echo "  VNC          →  localhost:5900  (no password)"
echo ""

if [ ! -f "$SENTINEL" ]; then
    echo "  [install] First start — extracting source and installing..."
    echo "  [install] This takes a few minutes. Subsequent starts are instant."
    echo ""

    # Copy source from image into /app on the volume
    cp -r /opt/corridorkey-src/. /app/

    cd /app
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    CORRIDORKEY_PYTHON_VERSION=${CORRIDORKEY_PYTHON_VERSION:-3.11} \
    CORRIDORKEY_INSTALL_SAM2=${CORRIDORKEY_INSTALL_SAM2:-y} \
    CORRIDORKEY_PREDOWNLOAD_SAM2=${CORRIDORKEY_PREDOWNLOAD_SAM2:-y} \
    CORRIDORKEY_INSTALL_GVM=${CORRIDORKEY_INSTALL_GVM:-n} \
    CORRIDORKEY_INSTALL_VIDEOMAMA=${CORRIDORKEY_INSTALL_VIDEOMAMA:-n} \
    CORRIDORKEY_CREATE_SHORTCUT=n \
    bash /app/1-install.sh

    touch "$SENTINEL"
    echo ""
    echo "  [install] Done. Everything is on the volume — reinstall won't be needed unless deps change."
else
    echo "  [install] Already installed — skipping."
    echo "  [install] (Delete $SENTINEL to reinstall)"
fi

echo ""

# ── Fix venv python symlink ───────────────────────────────────
# The venv's python symlinks into /opt/uv-python in the image.
# Rewrite them to absolute real paths so supervisord can exec
# them directly without needing to resolve the symlink chain.
VENV_BIN="/app/.venv/bin"
for name in python python3 python3.11; do
    link="$VENV_BIN/$name"
    if [ -L "$link" ]; then
        real=$(readlink -f "$link" 2>/dev/null || true)
        if [ -n "$real" ] && [ -f "$real" ]; then
            ln -sf "$real" "$link"
            echo "  [python] $name -> $real"
        fi
    fi
done

# ── filebrowser database init ────────────────────────────────
# Recreate the db every start to ensure --noauth and permissions
# are always correctly set. The db is just config — not user data.
rm -f /app/.filebrowser.db
filebrowser config init --database /app/.filebrowser.db
filebrowser config set --database /app/.filebrowser.db     --address 0.0.0.0     --port 6081     --root /app/ClipsForInference     --auth.method=noauth
echo "  [filebrowser] Database initialised (no auth)." 

# Patch supervisord with runtime resolution
RESOLUTION="${CORRIDORKEY_RESOLUTION:-2560x1440x24}"
sed "s|CORRIDORKEY_RESOLUTION_PLACEHOLDER|${RESOLUTION}|g" \
    /etc/supervisor/conf.d/corridorkey.conf.tpl \
    > /etc/supervisor/conf.d/corridorkey.conf
echo "  [display] Resolution: ${RESOLUTION}"
echo ""

exec /usr/bin/supervisord -c /etc/supervisor/conf.d/corridorkey.conf

#!/usr/bin/env bash
# ROCm VNC image — starts noVNC + x11vnc + filebrowser + CorridorKey GUI (baked .venv).
# Does not run 1-install.sh (Python 3.10 + deps are fixed at image build time).
set -euo pipefail

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export CORRIDORKEY_RESOLUTION="${CORRIDORKEY_RESOLUTION:-1920x1080x24}"
export CORRIDORKEY_CONTAINER_MODE="${CORRIDORKEY_CONTAINER_MODE:-1}"

mkdir -p /opt/corridorkey/ClipsForInference

echo "noVNC: http://localhost:6080"
echo "Upload: http://localhost:6081"
echo "VNC: localhost:5900 (password: EZ-CorridorKey — see docker/supervisord.conf)"
echo ""

exec /usr/bin/supervisord -c /etc/supervisor/conf.d/corridorkey-vnc.conf

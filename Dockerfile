# ============================================================
#  EZ-CorridorKey — Docker + noVNC (browser GUI)
#
#  http://localhost:6080  — GUI (auto-connects)
#  http://localhost:6081  — File browser (upload clips here)
#
#  Image contains only OS packages + app source.
#  On first start entrypoint.sh copies source into /app (volume),
#  runs 1-install.sh, and launches the app.
#
#  CPU:  docker compose up --build
#  GPU:  docker compose up corridorkey-gpu --build
# ============================================================

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────
# Split into two RUN steps: first add the FFmpeg PPA, then install
# everything in one pass. Keeps layer count low while ensuring
# the PPA is available before ffmpeg is requested.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gpg-agent \
    ca-certificates \
    && add-apt-repository -y ppa:ubuntuhandbook1/ffmpeg7 \
    && apt-get update && apt-get install -y --no-install-recommends \
    # Virtual display + VNC stack
    xvfb \
    x11vnc \
    novnc \
    websockify \
    # Qt6 / PySide6 platform plugin dependencies
    libgl1 \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libxcb-cursor0 \
    libdbus-1-3 \
    libfontconfig1 \
    libfreetype6 \
    libegl1 \
    libgles2 \
    # Audio
    libpulse0 \
    libasound2-dev \
    # OpenEXR
    libopenexr-dev \
    pkg-config \
    # FFmpeg 7.x from PPA
    ffmpeg \
    # build tools needed by some Python packages
    # Tools
    curl \
    git \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# ── uv ────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# Install uv-managed Python into a fixed image path so venv symlinks
# on the volume always resolve regardless of image rebuilds.
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_PYTHON_INSTALL_DIR="/opt/uv-python"
RUN uv python install 3.11

# ── filebrowser ───────────────────────────────────────────────
RUN curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash

# ── noVNC: auto-connect + upload button ──────────────────────
#RUN sed -i 's|</body>|<a href="http://localhost:6081" target="_blank" style="position:fixed;bottom:20px;right:20px;z-index:9999;background:#4caf50;color:white;padding:10px 18px;border-radius:6px;font-size:14px;text-decoration:none;box-shadow:0 2px 6px rgba(0,0,0,0.5)">& Upload Clips</a></body>|' /usr/share/novnc/vnc.html \ 
RUN echo '<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0;url=/vnc.html?autoconnect=true&reconnect=true&reconnect_delay=2000"></head></html>'  > /usr/share/novnc/index.html

# ── App source ────────────────────────────────────────────────
COPY . /opt/corridorkey-src

# ── Supervisor + entrypoint ───────────────────────────────────
COPY docker/supervisord.conf /etc/supervisor/conf.d/corridorkey.conf.tpl
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 6080 6081 5900

ENV DISPLAY=:1
ENV QT_QPA_PLATFORM=xcb
ENV LIBGL_ALWAYS_SOFTWARE=0
ENV QT_LOGGING_RULES="*.debug=false"
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV CORRIDORKEY_OPT_MODE=""


WORKDIR /app

ENTRYPOINT ["/entrypoint.sh"]

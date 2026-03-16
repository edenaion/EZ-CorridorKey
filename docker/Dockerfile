FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gpg-agent \
    ca-certificates \
    && add-apt-repository -y ppa:ubuntuhandbook1/ffmpeg7 \
    && apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    xvfb \
    python3-pil \
    python3-netifaces \
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
    libpulse0 \
    libasound2-dev \
    libopenexr-dev \
    ffmpeg \
    curl \
    git \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_PYTHON_INSTALL_DIR="/opt/uv-python"
RUN uv python install 3.11
RUN sed -i 's|</body>|<a href="http://localhost:6081" target="_blank" style="position:fixed;bottom:20px;right:20px;z-index:9999;background:#4caf50;color:white;padding:10px 18px;border-radius:6px;font-size:14px;text-decoration:none;box-shadow:0 2px 6px rgba(0,0,0,0.5)">& Upload Clips</a></body>|' /usr/share/novnc/vnc.html 
RUN echo '<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0;url=/vnc.html?autoconnect=true&reconnect=true&reconnect_delay=2000&resize=scale"></head></html>' > /usr/share/novnc/index.html

RUN curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash

COPY . /opt/corridorkey-src
COPY docker/supervisord.conf /etc/supervisor/conf.d/corridorkey-vnc.conf
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 6080 6081 5900

ENV DISPLAY=:1
ENV QT_QPA_PLATFORM=xcb
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV CORRIDORKEY_RESOLUTION=1920x1080x24

ENTRYPOINT ["/entrypoint.sh"]

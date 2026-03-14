# EZ-CorridorKey — Docker + Browser GUI

Runs the PySide6 GUI inside Docker via **Xvfb → x11vnc → noVNC**,
accessible from any browser at **http://localhost:6080**.

`1-install.sh` is the single source of truth for installation. The Dockerfile
contains only OS packages — on first container start, `entrypoint.sh` copies
the app source into the volume and runs `1-install.sh`. Everything (venv,
models, outputs) lives on the volume and persists across rebuilds.

---

## Quick Start

### CPU (no GPU)
```bash
docker compose up --build
```

### NVIDIA GPU
```bash
docker compose up corridorkey-gpu --build
```

Open **http://localhost:6080** — the GUI connects automatically.

To upload video clips, open **http://localhost:6081** — drag and drop files
directly in the browser. Uploads are saved to a separate volume so they
survive even if the main app volume is wiped.

> **First start takes a few minutes** — `1-install.sh` runs inside the
> container to set up Python, the venv, and model weights. Subsequent
> starts are instant.

---

## Updating

The venv and model weights live on the `corridorkey_app` volume and persist
across rebuilds — you won't need to reinstall unless dependencies changed.

### 1. Pull the latest source

```bash
git pull
```

### 2. Rebuild the image

```bash
docker compose up corridorkey-gpu --build   # GPU
docker compose up --build                   # CPU
```

The image is small (OS packages + source copy only) so rebuilds are fast.
On next start, the updated source is copied into the volume automatically.

### 3. If dependencies changed (`1-install.sh` or `pyproject.toml` updated)

Remove the sentinel so `1-install.sh` re-runs on next start and updates
the venv in place. Model weights are not affected.

```bash
docker exec -it corridorkey-gpu rm /app/.install_complete
docker compose restart corridorkey-gpu
```

### 4. Full clean slate (if something is badly broken)

```bash
docker compose down
docker volume rm corridorkey_app    # WARNING: deletes venv AND model weights
docker compose up corridorkey-gpu --build
```

> **Uploads are safe** — `corridorkey_uploads` is a separate volume and is
> not removed by the above command. Your clips are always preserved.

---

## File structure

```
your-repo/
├── Dockerfile
├── docker-compose.yml
├── .env                 ← all options live here
├── docker/
│   ├── entrypoint.sh
│   ├── supervisord.conf
│   └── README.md
├── 1-install.sh         ← single source of truth for install logic
├── main.py
└── ...
```

---

## Configuration

All options live in `.env`. No changes to `docker-compose.yml` or
`Dockerfile` needed.

### Install options
Read by `1-install.sh` on first start. To re-apply, delete the sentinel
and restart (see Updating → step 3).

| Variable | Default | Description |
|----------|---------|-------------|
| `CORRIDORKEY_PYTHON_VERSION` | `3.11` | Python version — uv downloads and manages this, any version 3.10–3.13 |
| `CORRIDORKEY_INSTALL_SAM2` | `y` | Install SAM2 tracker package. Set `n` to skip (faster first start, disables Track Mask) |
| `CORRIDORKEY_PREDOWNLOAD_SAM2` | `y` | Download SAM2 Base+ model weights (324 MB). Ignored if `CORRIDORKEY_INSTALL_SAM2=n` |
| `CORRIDORKEY_INSTALL_GVM` | `n` | Download GVM alpha generator (~6 GB) — opt-in |
| `CORRIDORKEY_INSTALL_VIDEOMAMA` | `n` | Download VideoMaMa model (~37 GB) — opt-in |

### Runtime options
Take effect on restart — no reinstall needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `CORRIDORKEY_RESOLUTION` | `2560x1440x24` | Virtual display resolution. Format: `WIDTHxHEIGHTxDEPTH`. The `x24` is color depth (24-bit RGB), not framerate. Common: `3840x2160x24`, `1920x1080x24` |
| `CORRIDORKEY_OPT_MODE` | *(blank)* | GPU optimization. `auto` = detect VRAM \| `speed` = force `torch.compile` \| `lowvram` = tiled refiner for ~8 GB cards. Blank = app auto-detects |

`CORRIDORKEY_CREATE_SHORTCUT` is always forced to `n` — meaningless inside a container.

---

## Volumes

| Volume | Mount | Purpose |
|--------|-------|---------|
| `corridorkey_app` | `/app` | Everything — source, venv, models, outputs, sentinel |
| `corridorkey_uploads` | `/app/ClipsForInference` | Uploaded clips — separate volume so they survive if `corridorkey_app` is wiped |

---

## Ports

| Port | Service | URL |
|------|---------|-----|
| `6080` | noVNC GUI | http://localhost:6080 |
| `6081` | File browser (upload clips) | http://localhost:6081 |
| `5900` | Raw VNC (native VNC clients, no password) | — |

---

## Logs

```bash
docker exec -it corridorkey-gpu tail -f /var/log/corridorkey.log
docker exec -it corridorkey-gpu tail -f /var/log/xvfb.log
docker exec -it corridorkey-gpu tail -f /var/log/x11vnc.log
docker exec -it corridorkey-gpu tail -f /var/log/novnc.log
docker exec -it corridorkey-gpu tail -f /var/log/filebrowser.log
```

Replace `corridorkey-gpu` with `corridorkey` for the CPU service.

---

## How it works

On every start, `entrypoint.sh`:

1. **Copies** app source from `/opt/corridorkey-src` (image) → `/app` (volume)
2. **Runs `1-install.sh`** on first start only — gated by `/app/.install_complete`
3. **Patches** the supervisord config template with the runtime resolution
4. **Launches supervisord**, which starts five processes in order:

| Priority | Process | Description |
|----------|---------|-------------|
| 10 | `xvfb` | Headless virtual X display at `:1` |
| 20 | `x11vnc` | VNC server on port `5900` |
| 30 | `novnc` | WebSocket bridge to VNC at port `6080` |
| 35 | `filebrowser` | Drag-and-drop upload UI at port `6081` |
| 40 | `corridorkey` | PySide6 GUI (`main.py`) |

---

## Notes

- **uv manages Python** — installs it to `/opt/uv-python` inside the image so
  venv symlinks on the volume always resolve after a rebuild. No system Python
  required.
- **FFmpeg 7.x** is installed from the `ubuntuhandbook1/ffmpeg7` PPA. Ubuntu
  24.04's default repo only ships 6.1.1, below CorridorKey's 7.0+ requirement.
- **The VNC session has no password.** Do not expose ports `5900`, `6080`, or
  `6081` to the public internet without authentication (e.g. nginx + basic
  auth, or a VPN).

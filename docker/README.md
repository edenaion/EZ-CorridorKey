# EZ-CorridorKey Docker Guide

Run the GUI in a browser via noVNC, with persistent volumes for projects, models, and runtime data.

## Prerequisites

- Docker Engine + Docker Compose plugin
- NVIDIA users: working NVIDIA Container Toolkit (for GPU service)

## Start

From the `docker/` directory:

CPU:

```bash
docker compose up -d corridorkey-cpu --build
```

GPU:

```bash
docker compose --profile gpu up -d corridorkey-gpu --build
```

First run will take several minutes (installs Python dependencies + downloads models). Subsequent starts are fast.

To see logs during first run: omit `-d` to run in the foreground.

## Access the app

All three endpoints are bound to `127.0.0.1` only (never exposed to your LAN) and require authentication. The password for every endpoint is the repository name: **`EZ-CorridorKey`**.

| Endpoint | URL | User | Password |
|---|---|---|---|
| Web UI (noVNC) | http://localhost:6080 | — | `EZ-CorridorKey` |
| Upload UI (file browser) | http://localhost:6081 | `admin` | `EZ-CorridorKey` |
| Raw VNC (optional) | localhost:5900 | — | `EZ-CorridorKey` |

> **Changing the password:** edit the VNC password in [`supervisord.conf`](supervisord.conf) (`x11vnc -passwd ...`) and the filebrowser admin creds in the same file (`filebrowser users add admin <new-password>`), then recreate the container with `docker compose --profile gpu up -d corridorkey-gpu --force-recreate`. The filebrowser user only gets created on the first boot of a fresh `.filebrowser.db`; if you change the password after first boot you'll also need to `docker exec` in and run the `filebrowser users update admin --password <new>` command, or wipe the volume.

## Upload files

1. Open http://localhost:6081
2. Upload clips into `ClipsForInference`
3. In the app, import from that folder

Uploads are persisted in the `corridorkey_uploads` volume.

## Stop / restart

Stop services (keep all volumes/data):

```bash
docker compose stop
```

Start again:

```bash
docker compose start
```

Stop and remove containers (keep volumes/data):

```bash
docker compose down
```

## Update project

```bash
git pull
docker compose --profile gpu up -d corridorkey-gpu --build --force-recreate
```

For CPU, replace service with `corridorkey-cpu`.

## Environment changes

If you change environment values in `docker-compose.yml`, recreate the service:

```bash
docker compose --profile gpu up -d corridorkey-gpu --force-recreate
```

The startup script re-checks install/model env flags on every start and applies changes when needed.

## Logs

GPU service:

```bash
docker compose logs -f corridorkey-gpu
```

CPU service:

```bash
docker compose logs -f corridorkey-cpu
```

## Volumes (persistent)

- `corridorkey_install` → app source + `.venv`
- `corridorkey_projects` → project data
- `corridorkey_models_*` → model checkpoints
- `corridorkey_hf_cache` → Hugging Face cache
- `corridorkey_config` → app-level config
- `corridorkey_logs` → logs
- `corridorkey_uploads` → uploaded files

## Security note

All three service ports (6080, 6081, 5900) bind to `127.0.0.1` only, so they are never reachable from your LAN or the internet out of the box. Every endpoint also requires a password (`EZ-CorridorKey` by default — see the table above). If you need remote access, tunnel over SSH rather than re-binding the ports to `0.0.0.0`:

```bash
ssh -L 6080:localhost:6080 -L 6081:localhost:6081 user@your-host
```

**Before publishing a Docker image publicly or handing the default container to untrusted users, change the password** in [`supervisord.conf`](supervisord.conf). The default is deliberately well-known — it exists so local/single-user setups don't have to guess a randomized password, not as real access control against a determined attacker.

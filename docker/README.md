# EZ-CorridorKey Docker Guide

Run the GUI in a browser via noVNC, with persistent volumes for projects, models, and runtime data.

## Prerequisites

- Docker Engine + Docker Compose plugin
- NVIDIA users: working NVIDIA Container Toolkit (for GPU service)

## Start

From the repository root:

CPU:

```bash
docker compose up -d corridorkey-cpu --build
```

GPU:

```bash
docker compose --profile gpu up -d corridorkey-gpu --build
```

This starts the container in the background (to stop `docker compose down`). If you would like to start in the foreground to be able see logs ommit `-d` when starting the container.

## Access the app

- Web UI (noVNC): http://localhost:6080
- Upload UI (file browser): http://localhost:6081
- Raw VNC (optional): localhost:5900

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

The web and VNC endpoints are unauthenticated by default. Do not expose these ports directly to the public internet.

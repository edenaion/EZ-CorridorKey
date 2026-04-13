# EZ-CorridorKey Docker Guide

Run the GUI in a browser via noVNC, with persistent volumes for projects, models, and runtime data.

## Prerequisites

- Docker Engine + Docker Compose plugin
- NVIDIA users: working NVIDIA Container Toolkit (for GPU service)

### AMD GPU (ROCm)

Compose below defaults to **CPU**; **NVIDIA** uses the **`gpu`** profile. For **AMD** hosts, use the **`rocm`** profile (or build from the repo root) — see **[docs/ROCm_Setup.md](../docs/ROCm_Setup.md)** (and **[docs/ROCm_PR_handoff.md](../docs/ROCm_PR_handoff.md)** for a full change list and review checklist).

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

AMD (ROCm) — builds **`rocm-vnc`** (browser UI + **baked** `.venv` at image build time; no `1-install.sh`). Requires `/dev/kfd` and `/dev/dri` on the host. See **[docs/ROCm_Setup.md](../docs/ROCm_Setup.md)** for differences vs the NVIDIA volume-based install.

```bash
docker compose --profile rocm up -d corridorkey-rocm --build
```

**CPU / GPU (`corridorkey-cpu`, `corridorkey-gpu`):** first start can take several minutes (`entrypoint.sh` runs `1-install.sh`, may pull models per env flags). **ROCm (`corridorkey-rocm`):** dependencies are baked at **image build** time; container start is usually quick unless you trigger a heavy rebuild.

To see logs during startup: omit `-d` to run in the foreground.

## Access the app

- Web UI (noVNC): http://localhost:6080
- Upload UI (file browser): http://localhost:6081
- Raw VNC (optional): localhost:5900 (password **EZ-CorridorKey**)

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

**GPU** and **ROCm** services use Compose **profiles**. To start a stopped profiled container, pass the same profile (and service name) you used when creating it:

```bash
docker compose --profile gpu start corridorkey-gpu
docker compose --profile rocm start corridorkey-rocm
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

For CPU, replace the service name with `corridorkey-cpu` (omit `--profile gpu`). For AMD:

```bash
docker compose --profile rocm up -d corridorkey-rocm --build --force-recreate
```

## Environment changes

If you change environment values in `docker-compose.yml`, recreate the service. Examples:

```bash
docker compose up -d corridorkey-cpu --force-recreate
docker compose --profile gpu up -d corridorkey-gpu --force-recreate
```

For **CPU/GPU**, `entrypoint.sh` re-checks install/model env flags on each start. For **`corridorkey-rocm`**, optional install env vars are **ignored** by the stock image; change Python deps by **rebuilding** the image (or extend `Dockerfile.rocm`).

## Logs

GPU service:

```bash
docker compose logs -f corridorkey-gpu
```

CPU service:

```bash
docker compose logs -f corridorkey-cpu
```

AMD (ROCm):

```bash
docker compose --profile rocm logs -f corridorkey-rocm
```

## Volumes (persistent)

- `corridorkey_install` → app source + `.venv` (used by **CPU/GPU**; **not** mounted for **`corridorkey-rocm`**, which ships a baked `.venv`)
- `corridorkey_projects` → project data
- `corridorkey_models_*` → model checkpoints
- `corridorkey_hf_cache` → Hugging Face cache
- `corridorkey_config` → app-level config
- `corridorkey_logs` → logs
- `corridorkey_uploads` → uploaded files

## Security note

**VNC / noVNC:** x11vnc is configured with password **EZ-CorridorKey** (see `docker/supervisord.conf`). Use that when the viewer prompts for a password.

**filebrowser (uploads):** listens on **127.0.0.1** inside the container; Compose publishes **6081** on **127.0.0.1** only. There is no shared VNC password for filebrowser—behavior depends on the installed **filebrowser** version (first-run setup is possible). The image build uses a **pinned** [filebrowser/get](https://github.com/filebrowser/get) `get.sh` commit (see `docker/Dockerfile`), not a floating **`master`** URL.

Do not publish **6080**, **6081**, or **5900** beyond localhost unless you understand the exposure.

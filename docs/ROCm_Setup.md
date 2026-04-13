# ROCm (AMD GPU) Docker setup

## Overview

This project’s default path targets **NVIDIA CUDA** (installer, manual venv, and [`docker/Dockerfile`](../docker/Dockerfile) / Compose). **AMD GPUs** are supported through a **separate** image, [`docker/Dockerfile.rocm`](../docker/Dockerfile.rocm). The default **`docker/Dockerfile`** is unchanged; [`docker/supervisord.conf`](../docker/supervisord.conf) is **shared** between the noVNC stacks (NVIDIA and ROCm).

The **noVNC** targets in [`docker/Dockerfile`](../docker/Dockerfile) and [`docker/Dockerfile.rocm`](../docker/Dockerfile.rocm) install **filebrowser** via a **pinned** `get.sh` commit from [filebrowser/get](https://github.com/filebrowser/get) (not `master`), so the installer script is reproducible; bump the SHA in both Dockerfiles together when updating.

[`docker/Dockerfile.rocm`](../docker/Dockerfile.rocm) defines **two build targets**:

| Target | Purpose |
|--------|---------|
| **`rocm-runtime`** (default) | Single-process app: `uv run python main.py`. Use for direct runs, host X11 forwarding, or automation. |
| **`rocm-vnc`** | Same baked app as above plus **noVNC**, **x11vnc**, **filebrowser**, and **supervisord** — aligned with the interactive stack in [`docker/Dockerfile`](../docker/Dockerfile). |

## Critical configurations

### Why not `pyproject.toml` / `uv sync --extra rocm`?

To avoid shared lockfile churn and merge conflicts, **ROCm PyTorch wheels are not wired through optional extras**. Inside `Dockerfile.rocm` only:

1. `uv sync --no-dev --python "$(command -v python3)"` installs the normal project dependency graph (conda `python3` on this base image; may briefly resolve the default Linux PyTorch build).
2. `uv pip install --python /opt/corridorkey/.venv/bin/python torch==2.9.1 torchvision==0.24.1 pytorch-triton-rocm --index-url https://download.pytorch.org/whl/rocm6.1 --reinstall` **replaces** that stack with AMD-compatible wheels **inside the container** (Triton is included for `torch.compile` / inductor paths on ROCm). The **`--python …/.venv/bin/python`** pin matches [`docker/Dockerfile.rocm`](../docker/Dockerfile.rocm) so wheels land in the project venv, not another interpreter on the base image.

### Runtime environment

- **`HSA_OVERRIDE_GFX_VERSION`**: The image sets a default (`10.3.0`) suitable for many consumer RDNA2 cards. Override at run time if your GPU needs a different value, for example:

  ```bash
  docker run -e HSA_OVERRIDE_GFX_VERSION=11.0.0 ...
  ```

- **Application code**: [`main.py`](../main.py) calls **`device_utils.setup_rocm_env()`** at the start of **`main()`** (via **`_try_setup_rocm_env()`**) before argument parsing and logging setup. When HIP is present, **`device_utils`** attaches a **stderr** `StreamHandler` so INFO lines appear **before** `setup_logging()` runs, then uses `os.environ.setdefault` for `HSA_OVERRIDE_GFX_VERSION`. PyTorch on ROCm still uses the **CUDA API surface** (`torch.cuda.*`), so [`backend/service/core.py`](../backend/service/core.py) device detection does not need ROCm-specific branches.

### Python version

The **`rocm/pytorch:rocm6.1-py3.10-ubuntu22.04`** base uses **Python 3.10** (conda). The project allows `requires-python = ">=3.10,<3.14"`; the one-click installer and default Docker image typically use **3.11**. If you hit subtle version-only bugs, compare against a 3.11 environment.

### Baked image vs NVIDIA Docker entrypoint

| | Default [`docker/Dockerfile`](../docker/Dockerfile) + Compose | `Dockerfile.rocm` |
|---|------------------|-------------------|
| **Install** | `entrypoint.sh` runs `1-install.sh` into a volume when signatures change | Dependencies fixed at **`docker build`** (`uv sync` + ROCm `uv pip`) |
| **`CORRIDORKEY_INSTALL_*`** | Drives optional SAM2 / GVM / VideoMaMa steps at container start | **Ignored** unless you add your own entrypoint or rebuild with a modified Dockerfile |
| **`corridorkey_install` volume** | Mounts over `/opt/corridorkey` | **Not used** for ROCm (would overwrite the baked `.venv`) |

### Optional: SAM2 / tracker extras

To bake **SAM2** (`tracker` extra) into a ROCm image, extend `rocm-base` with an extra `RUN` (for example `uv sync --no-dev --extra tracker` before the ROCm `uv pip reinstall`, or `uv pip install -e ".[tracker]"` with the same Python), then rebuild. There is no runtime hook equivalent to `CORRIDORKEY_INSTALL_SAM2=y` on the stock ROCm images.

### Host devices and groups

Most AMD hosts need **`--device=/dev/kfd`** and **`--device=/dev/dri`** plus **`--group-add video`**. Some distributions also require the **`render`** group; if you get permission errors on `/dev/dri/*`, try `--group-add render` (numeric GID varies by distro).

## Build

From the **repository root**:

**Default (minimal runtime image):**

```bash
docker build -t ez-corridorkey-rocm -f docker/Dockerfile.rocm .
```

**Browser / noVNC (parity with default Docker UX):**

```bash
docker build -t ez-corridorkey-rocm-vnc -f docker/Dockerfile.rocm --target rocm-vnc .
```

## Run (hardware passthrough)

AMD access requires passing the render/KFD devices into the container (adjust group IDs if your distro differs).

**`rocm-runtime`:**

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  ez-corridorkey-rocm
```

For GUI sessions that expect the same window sizing behavior as the default Docker stack, set **`CORRIDORKEY_CONTAINER_MODE=1`** (optional for headless or pure CLI use).

**`rocm-vnc`** (same device flags; publish ports like the NVIDIA Compose services). The image defaults **`CORRIDORKEY_CONTAINER_MODE=1`**; [`docker/entrypoint-rocm-vnc.sh`](../docker/entrypoint-rocm-vnc.sh) and [`docker/supervisord.conf`](../docker/supervisord.conf) pass **`%(ENV_CORRIDORKEY_CONTAINER_MODE)s`** into the GUI process, so **`docker run -e CORRIDORKEY_CONTAINER_MODE=0`** disables container/fullscreen mode when you want desktop-style sizing.

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -p 127.0.0.1:6080:6080 \
  -p 127.0.0.1:6081:6081 \
  -p 127.0.0.1:5900:5900 \
  ez-corridorkey-rocm-vnc
```

### Docker Compose (profile `rocm`)

From the `docker/` directory:

```bash
docker compose --profile rocm up -d corridorkey-rocm --build
```

This builds **`rocm-vnc`** and applies the same **127.0.0.1** port bindings as the CPU/GPU services. **Do not** mount a volume over all of `/opt/corridorkey` — the image ships a baked `.venv`; only data directories are mounted (projects, checkpoints, uploads, etc.).

## CI

Two workflows matter for this workstream:

| Workflow | Role |
|----------|------|
| [`.github/workflows/ruff.yml`](../.github/workflows/ruff.yml) | **Merge gate** on **`pull_request`** / **`push`** to **`main`** when **`**/*.py`**, **`pyproject.toml`**, or the workflow file changes: **`uv run ruff check .`** and **`uv run ruff format --check .`**. Configure **branch protection** to require **`Ruff / Lint & format`** if merges must be blocked on lint. |
| [`.github/workflows/rocm-ci.yml`](../.github/workflows/rocm-ci.yml) | **ROCm-focused**: mocked **`pytest tests/test_rocm_setup.py -m rocm --noconftest`**, optional **`docker build … --target rocm-runtime`** with **`continue-on-error: true`**. Runs only when paths in the YAML match (Dockerfiles, entrypoints, Compose, **`device_utils.py`**, **`main.py`**, tests, docs, **`CHANGELOG.md`**, etc.). |

Lint rules live under **`[tool.ruff]`** / **`[tool.ruff.lint]`** in [`pyproject.toml`](../pyproject.toml) (defaults plus **`E402`** / **`E701`** ignored for legacy import layout). That config is **not** the ROCm wheel index; ROCm PyTorch stays **Docker-only** as above.

## Troubleshooting

- **`tests/test_rocm_setup.py` missing in CI / not in PR** — ensure [`.gitignore`](../.gitignore) uses **`/test_*.py`** (repo root only), not a bare **`test_*.py`** (that ignores every `tests/test_*.py`). Verify: **`git check-ignore -v tests/test_rocm_setup.py`** should print nothing; **`git ls-files tests/test_rocm_setup.py`** should list the file.
- **`pytest` / `cv2` / `conftest` errors** — run ROCm tests with **`--noconftest`** (see [CONTRIBUTING.md](../CONTRIBUTING.md)) so the shared [`tests/conftest.py`](../tests/conftest.py) is not loaded.
- **Port `6080` / `6081` already in use** — the **`rocm`** Compose profile uses the same localhost ports as **`corridorkey-cpu`** / **`corridorkey-gpu`**. Stop the other stack first, or change published ports in a local override file.
- **`/dev/kfd` or `/dev/dri` missing inside WSL2 / VM** — ROCm needs a Linux host with AMD drivers and device nodes passed through; nested or Windows-only environments often cannot see these devices.
- **VNC password** — same as the default Docker stack: **EZ-CorridorKey** in [`docker/supervisord.conf`](../docker/supervisord.conf) (`x11vnc -passwd`).

## Cross-references

- [docker/README.md](../docker/README.md) — default Docker / Compose (NVIDIA-oriented)
- [CONTRIBUTING.md](../CONTRIBUTING.md) — dev setup, **Ruff** / pytest commands, **ROCm** prerequisite note
- [ROCm PR and handoff](ROCm_PR_handoff.md) — full change list, test commands, and copy-paste PR text for reviewers
- [AI full-pass review instructions](AI_full_pass_review_instructions.md) — how to review every changed file and line (for a larger AI or maintainer audit)

## Conclusion

Use **`docker/Dockerfile.rocm`** for AMD (`rocm-runtime` or `rocm-vnc`). Keep using the existing installer and **`docker/Dockerfile`** for NVIDIA. ROCm torch resolution stays **inside the Docker build**; there is still **no** `rocm` optional extra for `uv sync`.

Repo-wide **Python style** is enforced by **Ruff** in CI (see **`[tool.ruff]`** in [`pyproject.toml`](../pyproject.toml)); that is separate from ROCm packaging. Other **`pyproject.toml`** tweaks (**`[tool.uv.pip] torch-backend`**, hatch **`force-include`** for **`device_utils.py`**) fix **`uv`** parsing and ship the startup helper in wheels—they are **not** ROCm wheel indexes.

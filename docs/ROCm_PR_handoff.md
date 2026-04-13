# ROCm AMD support — implementation progress and PR package

This document is for **you and the upstream maintainers**: it tracks delivery from **0% to 100%**, lists **exact verification commands**, and includes a **ready-to-paste GitHub PR description**.

---

## Implementation progress (0% → 100%)

| Step | Weight | Status | Notes |
|------|--------|--------|--------|
| 1. `docker/Dockerfile.rocm` — `rocm-base`, **`rocm-runtime`** (default), **`rocm-vnc`** (noVNC parity), `pytorch-triton-rocm` | 25% | **Done** | Does not modify [`docker/Dockerfile`](../docker/Dockerfile). |
| 2. `docker/entrypoint-rocm-vnc.sh` + Compose **`corridorkey-rocm`** (`profile: rocm`) | 5% | **Done** | Baked `.venv`; no `1-install.sh` in ROCm VNC path. |
| 3. `device_utils.setup_rocm_env()` (HIP, stderr INFO before `setup_logging`, `HSA_*` `setdefault`) | 15% | **Done** | [`device_utils.py`](../device_utils.py). |
| 4. `main.py` (`_try_setup_rocm_env`, stderr DEBUG on failure before root logging) | 10% | **Done** | Start of `main()`. |
| 5. Tests + logging assertions + autouse logger reset | 15% | **Done** | [`tests/test_rocm_setup.py`](../tests/test_rocm_setup.py) (6 tests). |
| 6. `pyproject.toml` — `[tool.uv.pip] torch-backend`, hatch **`force-include`** `device_utils.py` | 10% | **Done** | Fixes `uv` parse warning; ships `device_utils` in wheel/sdist. **No** ROCm optional extra. |
| 7. CI `.github/workflows/rocm-ci.yml` (pytest + Docker build, `continue-on-error` on image) | 10% | **Done** | |
| 8. Docs + `CHANGELOG.md` | 10% | **Done** | |

**Overall: 100%** — ready for maintainer review.

### Follow-on (post-review hardening)

These landed after the core ROCm table but belong in the same PR / branch story:

| Item | Notes |
|------|--------|
| **`.github/workflows/ruff.yml`** | Merge gate: **`uv run ruff check .`** + **`uv run ruff format --check .`** on **`**/*.py`** / **`pyproject.toml`** changes. |
| **`[tool.ruff]` / `[tool.ruff.lint]`** in **`pyproject.toml`** | Default Ruff rules; **`ignore = ["E402", "E701"]`** (lazy imports + legacy one-liners). |
| **`.gitignore`** | **`/test_*.py`** (repo root only). A bare **`test_*.py`** rule ignored **`tests/test_rocm_setup.py`** and broke CI checkout. |
| **filebrowser** | Both Dockerfiles use a **pinned** [filebrowser/get](https://github.com/filebrowser/get) **`get.sh`** commit (not **`master`**). |
| **[`docs/AI_full_pass_review_instructions.md`](AI_full_pass_review_instructions.md)** | Playbook for a full-file / line-by-line AI or human review. |

Set **branch protection** on **`main`** to require the **`Ruff / Lint & format`** check name from Actions if merges should be blocked on lint.

---

## Verification evidence (run before submit)

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/test_rocm_setup.py -m rocm -v --tb=short --noconftest
```

- Tests are marked **`@pytest.mark.rocm`** (see [`pyproject.toml`](../pyproject.toml)); the **file path** limits collection time.
- **`--noconftest`** skips [`tests/conftest.py`](../tests/conftest.py) (imports `cv2`) for a fast, isolated run.
- **`torch-backend`** now lives under **`[tool.uv.pip]`** — current `uv` should parse `pyproject.toml` without the old `unknown field` warning.
- Confirm **`git ls-files tests/test_rocm_setup.py`** lists the file and **`git check-ignore -v tests/test_rocm_setup.py`** prints nothing.

### Test coverage summary

| Test | What it proves |
|------|----------------|
| `test_setup_rocm_env_no_hip_no_env_change` | HIP absent → no `HSA_OVERRIDE_GFX_VERSION` added. |
| `test_setup_rocm_env_with_hip_sets_default` | HIP present → `setdefault` applies `10.3.0`. |
| `test_setup_rocm_env_hip_preserves_existing_override` | Pre-set env wins. |
| `test_try_setup_rocm_env_import_error_is_silent` | `ImportError` on `device_utils` → no crash. |
| `test_try_setup_rocm_env_inner_failure_logged` | Inner failure → log record **"ROCm environment setup skipped"** (caplog). |
| `test_setup_rocm_env_hip_writes_to_stderr` | HIP path → **stderr** contains `ROCm:` before `main.setup_logging()`. |

---

## Files changed (inventory)

| Area | Files |
|------|--------|
| Docker | [`docker/Dockerfile.rocm`](../docker/Dockerfile.rocm), [`docker/entrypoint.sh`](../docker/entrypoint.sh), [`docker/entrypoint-rocm-vnc.sh`](../docker/entrypoint-rocm-vnc.sh), [`docker/docker-compose.yml`](../docker/docker-compose.yml), [`docker/supervisord.conf`](../docker/supervisord.conf) |
| App | [`device_utils.py`](../device_utils.py), [`main.py`](../main.py) |
| Packaging | [`pyproject.toml`](../pyproject.toml) — `[tool.uv.pip]`, hatch `force-include` |
| Tests | [`tests/test_rocm_setup.py`](../tests/test_rocm_setup.py) |
| CI | [`.github/workflows/rocm-ci.yml`](../.github/workflows/rocm-ci.yml), [`.github/workflows/ruff.yml`](../.github/workflows/ruff.yml) |
| Docs | [`docs/ROCm_Setup.md`](ROCm_Setup.md), [`docs/AI_full_pass_review_instructions.md`](AI_full_pass_review_instructions.md), this file, [`README.md`](../README.md), [`docker/README.md`](../docker/README.md), [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
| Changelog | [`CHANGELOG.md`](../CHANGELOG.md) |
| Repo hygiene | [`.gitignore`](../.gitignore) (`/test_*.py` for root scratch scripts only) |

**Unchanged by design:** [`docker/Dockerfile`](../docker/Dockerfile) remains the default NVIDIA-oriented image (it shares **noVNC** / **filebrowser** patterns with ROCm); [`backend/service/core.py`](../backend/service/core.py) needs no ROCm branch. **No** `rocm` optional dependency group for `uv sync`.

---

## Suggested PR title

```
feat(docker): ROCm AMD image (runtime + noVNC), device_utils, CI
```

---

## Copy-paste PR description (GitHub)

```markdown
## Summary

Adds **optional AMD GPU (ROCm) support** via [`docker/Dockerfile.rocm`](docker/Dockerfile.rocm):

- **`rocm-runtime`** (default build target): `uv run python main.py`.
- **`rocm-vnc`**: same baked environment plus noVNC / x11vnc / filebrowser / supervisord (parity with the default `docker/Dockerfile` UX).

ROCm PyTorch + **`pytorch-triton-rocm`** are applied **only inside the image** (`uv pip install … --reinstall` after `uv sync`). **No** `rocm` optional extra in `pyproject.toml`.

Also adds **`device_utils.setup_rocm_env()`** (HIP detection, `HSA_OVERRIDE_GFX_VERSION`, stderr logging before `setup_logging`), **`main._try_setup_rocm_env()`**, **mocked tests**, **Compose profile `rocm`**, **ROCm CI workflow**, **`device_utils` in wheel/sdist** via hatch `force-include`, moves **`torch-backend = "auto"`** to **`[tool.uv.pip]`**, a **Ruff** merge gate (**`ruff.yml`**), **`[tool.ruff]`** lint config, **pinned filebrowser** `get.sh`, **`.gitignore`** fix so **`tests/test_rocm_setup.py`** is tracked, and **[`docs/AI_full_pass_review_instructions.md`](AI_full_pass_review_instructions.md)** for deep reviews.

## How to test

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/test_rocm_setup.py -m rocm -v --tb=short --noconftest
```

Docker (optional; large pull):

```bash
docker build -f docker/Dockerfile.rocm --target rocm-runtime -t ez-corridorkey-rocm .
docker build -f docker/Dockerfile.rocm --target rocm-vnc -t ez-corridorkey-rocm-vnc .
```

## Docs

- [`docs/ROCm_Setup.md`](docs/ROCm_Setup.md)
- [`docs/ROCm_PR_handoff.md`](docs/ROCm_PR_handoff.md)
- [`docs/AI_full_pass_review_instructions.md`](docs/AI_full_pass_review_instructions.md)
```

---

## Cross-references

- [ROCm_Setup.md](ROCm_Setup.md)
- [docker/README.md](../docker/README.md)
- [AI full-pass review instructions](AI_full_pass_review_instructions.md) — detailed playbook for line-by-line / full-file review of every changed path (for a larger AI or human reviewer)

## Conclusion

All previously identified gaps are addressed: **triton**, **verified `uv` binary path** (documented in Dockerfile), **VNC parity target + Compose**, **ROCm CI**, **Ruff CI + `[tool.ruff]`**, **CHANGELOG**, **wheel/sdist include**, **early logging**, **stronger tests**, **`[tool.uv.pip]`** fix, **pinned filebrowser installer**, **`.gitignore`** safe for **`tests/`**, and an **optional full-pass review playbook**.

# Azure GPU Installer Smoke

This supplements the existing cross-platform hosted CI in
[fresh-install-smoke.yml](/mnt/n/AI/EZ-CorridorKey/.github/workflows/fresh-install-smoke.yml).

The goal is narrow and pragmatic:

- validate the real Windows GPU installer path on Azure
- validate the real Linux GPU installer path on Azure
- keep cost low enough to run on startup credits

The primary failure class this harness targets is the one users keep reporting:

- installer picks CPU-only torch on a machine that has an NVIDIA GPU
- `nvidia-smi` works, but the app venv still lands on `torch+cpu`
- `torch.cuda.is_available()` is false inside CorridorKey's own `.venv`

It does **not** try to brute-force every CUDA version in the cloud. For CUDA
branch coverage, keep the mocked parser matrix in CI. Use Azure for real
end-to-end truth:

- VM provisioning
- NVIDIA driver extension install
- `nvidia-smi`
- Python bootstrap
- `1-install.bat` / `1-install.sh`
- installer log classification (`CPU-only` vs GPU wheel path)
- `pip show torch` inside `.venv`
- `torch.cuda.is_available()`
- `torch.version.cuda`
- `CorridorKeyService().detect_device()` inside `.venv`
- headless smoke startup

## Recommended Matrix

Use the smallest useful GPU lane first:

- Windows: `Standard_NC4as_T4_v3`
- Linux: `Standard_NC4as_T4_v3`

Suggested region order:

- `eastus,eastus2,westus2,westus`

This keeps the first pass focused on one real NVIDIA class instead of spending
credits on a giant matrix before the harness is proven.

Current workflow profiles:

- `issue-cluster-v1`
  - `win-py310`
  - `win-py311`
  - `win-py312`
  - `win-py313`
  - `linux-py310`
- `single`
  - uses the manual `run_windows` / `run_linux` and Python inputs

Recommended order:

1. `issue-cluster-v1` with `estimate_only=true`
2. `issue-cluster-v1` real run
3. If one lane fails, switch to `single` and iterate on just that lane
4. Optional later: add more regions or VM classes only after the first matrix is stable

That order is deliberate. We first validate the harness on the most stable
Windows path, then spend credits on the Python versions that match the actual
user reports.

## GitHub Secret

Create one repository secret named `AZURE_CREDENTIALS` with JSON in this form:

```json
{
  "clientId": "...",
  "clientSecret": "...",
  "subscriptionId": "...",
  "tenantId": "..."
}
```

The existing Azure service principal created for this project already matches
that shape.

## Manual Run

Workflow:

- [azure-gpu-installer-smoke.yml](/mnt/n/AI/EZ-CorridorKey/.github/workflows/azure-gpu-installer-smoke.yml)

Run it from GitHub Actions with:

- `repo_ref=main` or the release branch
- `matrix_profile=issue-cluster-v1`
- `max_parallel=5`
- default region list
- default `Standard_NC4as_T4_v3`
- `max_budget_usd=150`
- `estimate_only=true` for a dry run, then `false` for the real run
- `keep_resources=false`
- `spot=false` for reliability on the first run

Artifacts uploaded:

- `logs/azure-gpu-installer-smoke/`
- aggregated Markdown table: `azure-gpu-installer-summary-*`

## Direct Cloud Shell Run

You can also run the harness directly from Azure Cloud Shell after cloning the
repo:

```bash
bash scripts/azure/cloud_gpu_installer_smoke.sh \
  --ref main \
  --windows-regions eastus,eastus2,westus2,westus \
  --linux-regions eastus,eastus2,westus2,westus
```

## Cost Guidance

Default behavior is intentionally conservative:

- one workflow run at a time (`concurrency` in GitHub Actions)
- bounded matrix concurrency (`max_parallel`, default `5`)
- no public IPs on the test VMs
- a fresh per-lane resource group, deleted at the end
- live Azure retail price lookup before provisioning
- hard budget cap before the first VM is created
- per-lane summary JSON, plus a final Markdown summary table

Default behavior deletes the per-run resource group at the end of the run.

Use `--keep-resources` only for debugging. That is the main lever that can
accidentally burn credits.

Use `--spot` only after the standard path is stable. Spot VMs are cheaper, but
they are less reliable for installer validation.

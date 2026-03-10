# Self-Hosted GPU Runner

This is the practical setup for automating CorridorKey GPU QA on your own
Windows NVIDIA workstation.

It is useful for automation. It is **not** new hardware coverage. If your only
GPU box is the same 5090 workstation you already test on manually, the value
here is:

- rerunning the same GPU QA after changes
- getting pass/fail history in GitHub
- uploading a JSON summary and fixture outputs for later inspection

## What The Workflow Runs

The GPU workflow is [.github/workflows/local-gpu-qa.yml](/mnt/n/AI/EZ-CorridorKey/.github/workflows/local-gpu-qa.yml).

It does this on a labeled self-hosted runner:

1. checks out the repo
2. runs the real Windows installer non-interactively
3. ensures the requested SAM2 checkpoint and VideoMaMa weights are cached
4. runs [scripts/local_gpu_qa.py](/mnt/n/AI/EZ-CorridorKey/scripts/local_gpu_qa.py)
5. uploads:
   - `.tmp/gpu-qa/`
   - `.tmp/gpu-qa-run.log`
   - `logs/gpu-qa-summary.json`

## Recommended Labels

GitHub adds these labels automatically:

- `self-hosted`
- `windows`
- `x64`

Add these custom labels when you configure the runner:

- `nvidia`
- `corridorkey-gpu`

The workflow requires all five labels so it does not accidentally land on the
wrong machine.

## One-Time Setup

1. Open your GitHub repo.
2. Go to `Settings > Actions > Runners`.
3. Click `New self-hosted runner`.
4. Choose `Windows` and `x64`.
5. Download the runner package onto the 5090 box.
6. Extract it somewhere stable, for example:
   `C:\actions-runner\corridorkey-5090`
7. Open `cmd.exe` in that folder and run the GitHub-provided config command.
8. When prompted for labels, add:
   `nvidia,corridorkey-gpu`
9. Install it as a service if you want it always available.

## Machine Requirements

For the GPU QA workflow, the machine should already have:

- NVIDIA drivers working
- a usable CUDA-capable PyTorch environment
- enough free disk for VideoMaMa weights and caches
- enough free time to let GitHub use the workstation during the job

`ffmpeg` is not required for the synthetic GPU QA fixture.

## First Run

After the runner shows up as `Idle` in GitHub:

1. Open the `Actions` tab.
2. Choose `Local GPU QA`.
3. Click `Run workflow`.
4. Pick the SAM2 model you want to validate.
5. Leave `frames=8` and `chunk_size=8` unless you are intentionally stress-testing.

The workflow will reuse cached models on later runs. It still reruns the
installer so the dependency set stays aligned with the checked-out branch.

## Operational Notes

- This runner executes on your real workstation.
- Do not treat it like free cloud hardware.
- If you are rendering, editing, or debugging locally, pause the runner first.
- Start with manual dispatch only. Add automatic triggers later if the workflow
  proves stable and non-disruptive.

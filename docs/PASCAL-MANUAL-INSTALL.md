# Pascal GPU (GTX 10-series) — Manual PyTorch Install

EZ-CorridorKey ships a PyTorch build that does **not** include GPU kernels for Pascal-generation cards (GeForce GTX 10-series, Titan X/Xp, older Quadro). If you launch the app on one of these cards you will see a "no kernel image is available for execution on the device" error the first time you run AI keying.

This guide explains how to replace the bundled PyTorch with a Pascal-compatible wheel by editing the app's internal folder directly. It is a **one-time, in-place swap** — you do not need to reinstall the app, and your projects, settings, and model weights are preserved.

Supported cards:

- GeForce GTX 1050, 1050 Ti, 1060, 1070, 1070 Ti, 1080, 1080 Ti
- Titan X (Pascal), Titan Xp
- Quadro P-series
- Tesla P-series

If your card is Turing (RTX 20-series, GTX 16-series) or newer, **you do not need this guide** — the stock install already supports you.

---

## What you're changing

- **torch 2.9.1** → **torch 2.8.0+cu128** (the last release that still ships kernels for sm_61 Pascal through sm_120 Blackwell in a single binary)
- **torchvision 0.24.1** → **torchvision 0.23.0** (pair-pinned to torch 2.8.0)
- **triton-windows 3.5.x** → **triton-windows 3.4.x** (pair-pinned to torch 2.8.0 — `torch.compile` breaks if this is not updated in lockstep)

You need a Python 3.11 interpreter available on your PATH, OR you can use the Python bundled inside the installed app if you prefer.

---

## Step-by-step (Windows, using system Python)

Open **Windows Terminal** or **Command Prompt** and run:

```powershell
cd "%LOCALAPPDATA%\EZ-CorridorKey\_internal"

python -m pip install --target . --upgrade ^
  torch==2.8.0 torchvision==0.23.0 "triton-windows<3.5" ^
  --index-url https://download.pytorch.org/whl/cu128
```

If your install is in a different location, substitute the correct path to the `_internal` folder.

The `--target .` flag tells pip to install the packages directly into the current directory, overwriting the bundled torch. This is exactly the effect you want — it replaces the broken wheel with the Pascal-compatible one without touching anything else.

Once the install finishes, restart EZ-CorridorKey. The "Older NVIDIA GPU Detected" diagnostic should no longer appear.

---

## Verifying it worked

Launch the app and open **Help → Diagnostics**. The "Older NVIDIA GPU Detected" entry should be gone. If it still appears:

1. Make sure you ran the pip command inside the `_internal` folder (not the app root).
2. Confirm no errors during the pip install — you must see `Successfully installed torch-2.8.0 torchvision-0.23.0 ...`.
3. Check that `_internal/torch/version.py` now reads `__version__ = '2.8.0+cu128'`. If it still reads `2.9.1`, the pip install did not land in the right folder — re-run it with the full absolute path to `_internal`.

You can also verify the GPU arch list from the terminal:

```powershell
cd "%LOCALAPPDATA%\EZ-CorridorKey\_internal"
python -c "import torch; print(torch.cuda.get_arch_list())"
```

You should see `sm_61` in the list. If you see only `sm_70` and above, the old wheel is still in place.

---

## Reverting the manual install

If the swap causes issues, you can roll back by running the full installer for the same version of EZ-CorridorKey you currently have. The installer will overwrite `_internal/torch/` with the stock wheel, restoring the original state. Your user data is not touched.

---

## Why doesn't the app do this automatically?

Shipping a Pascal-compatible PyTorch as the default would force every user on a modern GPU to re-download the ~2 GB PyTorch runtime on every release, because the in-app skinny updater intentionally does not replace heavy runtime files (torch, triton, nvidia). We chose to keep the fast in-app update path for the majority and offer this manual swap for the Pascal minority.

If your card is Turing or newer and you still see a CUDA error, this is not the fix you need — open an issue at https://github.com/edenaion/EZ-CorridorKey/issues with your error log.

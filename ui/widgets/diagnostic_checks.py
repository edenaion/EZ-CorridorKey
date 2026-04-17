"""Diagnostic check data — known error patterns and startup environment checks.

This module holds the ``Diagnostic`` / ``StartupIssue`` dataclasses, the
registry of known error patterns (``_DIAGNOSTICS``), and the
``run_startup_diagnostics`` routine.  The UI dialog classes that *display*
these results live in ``diagnostic_dialog.py``.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Known error patterns ─────────────────────────────────────────────

@dataclass
class Diagnostic:
    """A single known-error diagnosis with user-facing fix steps."""
    id: str
    title: str
    pattern: re.Pattern[str]
    explanation: str
    steps: list[str]
    tags: list[str] = field(default_factory=list)


_DIAGNOSTICS: list[Diagnostic] = [
    # ── GPU / CUDA ────────────────────────────────────────────────
    Diagnostic(
        id="gpu-required",
        title="GPU Required — CPU-Only PyTorch Detected",
        pattern=re.compile(
            r"requires a CUDA GPU but only CPU is available|"
            r"float16.*cannot run with.*cpu.*device|"
            r"No GPU acceleration available",
            re.IGNORECASE,
        ),
        explanation=(
            "Your PyTorch installation does not include CUDA support, so "
            "GPU-accelerated pipelines (GVM, VideoMaMa) cannot run.  This "
            "usually means the installer picked the CPU-only wheels, or "
            "PyTorch was installed separately without the CUDA index."
        ),
        steps=[
            "Open a terminal in the EZ-CorridorKey folder.",
            "Activate the virtual environment:\n"
            "    .venv\\Scripts\\activate",
            "Reinstall PyTorch with CUDA:\n"
            "    pip install torch torchvision --index-url\n"
            "    https://download.pytorch.org/whl/cu128",
            "Restart EZ-CorridorKey.",
            "If you don't have an NVIDIA GPU, GVM and VideoMaMa\n"
            "are not supported — use manual alpha hints instead.",
        ],
        tags=["gpu", "cuda", "cpu", "float16"],
    ),
    # ── Missing checkpoint ────────────────────────────────────────
    Diagnostic(
        id="missing-checkpoint",
        title="Model Checkpoint Not Found",
        pattern=re.compile(
            r"No \.pth checkpoint found|"
            r"checkpoints.*not found|"
            r"CorridorKey\.pth.*missing",
            re.IGNORECASE,
        ),
        explanation=(
            "The CorridorKey model weights (.pth file) are missing from the "
            "checkpoints folder.  The model must be downloaded separately."
        ),
        steps=[
            "Download CorridorKey.pth from the project release page\n"
            "or the link in the README.",
            "Place the file in:\n"
            "    CorridorKeyModule/checkpoints/CorridorKey.pth",
            "Restart EZ-CorridorKey.",
        ],
        tags=["checkpoint", "model", "pth"],
    ),
    # ── FFmpeg ────────────────────────────────────────────────────
    Diagnostic(
        id="ffmpeg-invalid",
        title="FFmpeg Install Unsupported",
        pattern=re.compile(
            r"FFmpeg 7\.0 or newer is required|"
            r"FFprobe 7\.0 or newer is required|"
            r"FFmpeg and FFprobe major versions do not match|"
            r"Could not determine ffmpeg version|"
            r"Could not determine ffprobe version|"
            r"CorridorKey requires a full FFmpeg build",
            re.IGNORECASE,
        ),
        explanation=(
            "CorridorKey found FFmpeg, but the install is too old, incomplete, "
            "or using a stripped Windows build. Video import/export requires "
            "FFmpeg 7.0+ plus FFprobe."
        ),
        steps=[
            "Go to Edit > Preferences > Repair FFmpeg.\n"
            "This will automatically download a compatible FFmpeg build.",
            "If that doesn't work:\n"
            "  Windows: re-run 1-install.bat.\n"
            "  macOS: brew install ffmpeg\n"
            "  Linux: install both ffmpeg and ffprobe from your package manager (version 7.0+).",
            "Verify both commands work:\n"
            "    ffmpeg -version\n"
            "    ffprobe -version",
        ],
        tags=["ffmpeg", "ffprobe", "video", "version"],
    ),
    Diagnostic(
        id="ffmpeg-missing",
        title="FFmpeg Not Found",
        pattern=re.compile(
            r"FFmpeg not found|"
            r"ffmpeg.*not.*found|"
            r"ffprobe.*not.*found",
            re.IGNORECASE,
        ),
        explanation=(
            "FFmpeg and FFprobe are required for video import/export but were "
            "not found on your system."
        ),
        steps=[
            "Go to Edit > Preferences > Repair FFmpeg.\n"
            "This will automatically download and install FFmpeg for you.",
            "If that doesn't work:\n"
            "  Windows: re-run 1-install.bat.\n"
            "  macOS: brew install ffmpeg\n"
            "  Linux: install ffmpeg from your package manager.",
            "Restart EZ-CorridorKey.",
        ],
        tags=["ffmpeg", "video"],
    ),
    # ── Triton / torch.compile ────────────────────────────────────
    Diagnostic(
        id="triton-missing",
        title="Triton Not Available (torch.compile Disabled)",
        pattern=re.compile(
            r"triton.*not.*available|"
            r"triton.*import.*failed|"
            r"ModuleNotFoundError.*triton",
            re.IGNORECASE,
        ),
        explanation=(
            "Triton is required for torch.compile optimizations on Windows. "
            "Without it, inference will still work but will be slower."
        ),
        steps=[
            "Open a terminal in the EZ-CorridorKey folder.",
            "Activate the virtual environment:\n"
            "    .venv\\Scripts\\activate",
            "Install triton-windows:\n"
            "    pip install triton-windows",
            "Restart EZ-CorridorKey.",
        ],
        tags=["triton", "compile"],
    ),
    Diagnostic(
        id="msvc-compiler-missing",
        title="MSVC Compiler Not Found (torch.compile Disabled)",
        pattern=re.compile(
            r"Compiler: cl is not found|"
            r"Compiler: cl\.exe is not found",
            re.IGNORECASE,
        ),
        explanation=(
            "torch.compile requires the MSVC C++ compiler (cl.exe) on Windows. "
            "Without it, inference still works but falls back to slower eager mode."
        ),
        steps=[
            "Install Visual Studio Build Tools from:\n"
            "    https://visualstudio.microsoft.com/visual-cpp-build-tools/",
            'Select "Desktop development with C++" workload.',
            "Restart your computer, then restart EZ-CorridorKey.",
            "Alternatively, this is safe to ignore — inference just runs a bit slower.",
        ],
        tags=["compile", "msvc"],
    ),
    # ── CPU-only PyTorch installed (wrong wheel) ─────────────────
    Diagnostic(
        id="pytorch-cpu-wheel",
        title="CPU-Only PyTorch Installed (Missing CUDA Support)",
        pattern=re.compile(
            r"torch.*\+cpu|"
            r"PyTorch.*\+cpu",
            re.IGNORECASE,
        ),
        explanation=(
            "You have the CPU-only build of PyTorch installed even though "
            "an NVIDIA GPU is present. This means none of the GPU-accelerated "
            "models (CorridorKey, GVM, VideoMaMa) can use your GPU. "
            "This usually happens when PyTorch was installed without the "
            "CUDA index URL."
        ),
        steps=[
            "Open a terminal in the EZ-CorridorKey folder.",
            "Activate the virtual environment:\n"
            "    .venv\\Scripts\\activate  (or venv\\Scripts\\activate)",
            "Uninstall the CPU-only build:\n"
            "    pip uninstall torch torchvision -y",
            "Reinstall with CUDA support:\n"
            "    pip install torch torchvision --index-url\n"
            "    https://download.pytorch.org/whl/cu128",
            "Restart EZ-CorridorKey.",
            "Tip: re-running 1-install.bat will also fix this automatically.",
        ],
        tags=["gpu", "cuda", "cpu", "wheel"],
    ),
    # ── six / protobuf import error (GVM) ──────────────────────
    Diagnostic(
        id="six-metapath-importer",
        title="GVM Import Error (_SixMetaPathImporter)",
        pattern=re.compile(
            r"_SixMetaPathImporter.*object has no attribute|"
            r"SixMetaPathImporter",
            re.IGNORECASE,
        ),
        explanation=(
            "The GVM pipeline hit a compatibility error in the 'six' library, "
            "which is used by protobuf/grpc. This typically occurs when "
            "PyTorch is CPU-only (+cpu) or there is a version conflict between "
            "protobuf and other packages."
        ),
        steps=[
            "First, check if you have CPU-only PyTorch:\n"
            "    python -c \"import torch; print(torch.__version__)\"",
            "If the version ends in '+cpu', reinstall with CUDA:\n"
            "    pip install torch torchvision --index-url\n"
            "    https://download.pytorch.org/whl/cu128",
            "If the version already has '+cu...', update protobuf:\n"
            "    pip install -U protobuf grpcio",
            "Restart EZ-CorridorKey.",
        ],
        tags=["gvm", "six", "protobuf", "import"],
    ),
    # ── GVM weights ───────────────────────────────────────────────
    Diagnostic(
        id="gvm-weights-missing",
        title="GVM Model Weights Not Found",
        pattern=re.compile(
            r"Base model path not found.*stable-video-diffusion|"
            r"gvm_core.*weights.*not found|"
            r"GVM.*model.*not found",
            re.IGNORECASE,
        ),
        explanation=(
            "The GVM (Green-screen Video Matting) model weights are missing.  "
            "They must be downloaded separately from the model files."
        ),
        steps=[
            "Run the GVM weight downloader:\n"
            "    python -m gvm_core.download",
            "Or download manually from the README link and place\n"
            "in gvm_core/weights/",
            "Restart EZ-CorridorKey.",
        ],
        tags=["gvm", "weights"],
    ),
    # ── Python version ────────────────────────────────────────────
    Diagnostic(
        id="python-version",
        title="Unsupported Python Version",
        pattern=re.compile(
            r"Python.*3\.\d+.*not.*supported|"
            r"requires.*Python.*3\.1[0-3]|"
            r"python_requires",
            re.IGNORECASE,
        ),
        explanation=(
            "EZ-CorridorKey requires Python 3.10–3.13.  Your current "
            "Python version is not compatible."
        ),
        steps=[
            "Download Python 3.11 from https://python.org",
            "During installation, check 'Add Python to PATH'.",
            "Delete the existing .venv folder.",
            "Re-run 1-install.bat to create a fresh environment.",
        ],
        tags=["python", "version"],
    ),
    # ── Pascal GPU on cu130 wheel (1.9.1 → 1.9.2 upgrade path) ───
    Diagnostic(
        id="pascal-cu130-mismatch",
        title="Older NVIDIA GPU Detected — Reinstall Required",
        # Pattern is unused at runtime — this diagnostic is only fired
        # by run_startup_diagnostics, not by error-text matching. The
        # regex is intentionally a no-match so a stray error message
        # can never trigger it accidentally.
        pattern=re.compile(r"(?!.*)", re.DOTALL),
        explanation=(
            "Your NVIDIA GPU is from the Pascal generation (GeForce GTX "
            "10-series, Titan X/Xp, or older Quadro). The PyTorch build "
            "currently installed in this app does not include the "
            "GPU kernels needed for your card, so AI keying will fail "
            "at inference time with a CUDA error. EZ-CorridorKey 1.9.2 "
            "ships a new PyTorch build that supports your GPU, but the "
            "in-app updater intentionally does not replace the heavy "
            "runtime files — you need to download the full installer "
            "once to upgrade them."
        ),
        steps=[
            "Download the latest installer from:\n"
            "    https://github.com/edenaion/EZ-CorridorKey/releases/latest",
            "Run the installer over your existing install —\n"
            "your projects, settings, and model weights are preserved.",
            "After reinstalling, future updates will work normally\n"
            "via Help > Check for Updates.",
            "If you do not want to reinstall, EZ-CorridorKey cannot\n"
            "use your GPU on this version. AI keying will not work.",
        ],
        tags=["pascal", "gpu", "cuda", "upgrade"],
    ),
    # ── CUDA out of memory ────────────────────────────────────────
    Diagnostic(
        id="cuda-oom",
        title="GPU Out of Memory (VRAM)",
        pattern=re.compile(
            r"CUDA out of memory|"
            r"OutOfMemoryError|"
            r"torch\.cuda\.OutOfMemoryError",
            re.IGNORECASE,
        ),
        explanation=(
            "Your GPU ran out of VRAM during processing.  This can happen "
            "with high-resolution clips or when other applications are "
            "using GPU memory."
        ),
        steps=[
            "Close other GPU-heavy applications (games, other AI tools,\n"
            "browser hardware acceleration).",
            "Try processing at a lower resolution first.",
            "Set the environment variable for low-VRAM mode:\n"
            "    set CORRIDORKEY_OPT_MODE=lowvram",
            "If the problem persists, your GPU may not have enough\n"
            "VRAM for this clip resolution.",
        ],
        tags=["vram", "memory", "oom"],
    ),
]


def match_diagnostic(error_msg: str) -> Diagnostic | None:
    """Return the first matching Diagnostic for *error_msg*, or ``None``."""
    for diag in _DIAGNOSTICS:
        if diag.pattern.search(error_msg):
            return diag
    return None


# ── Startup diagnostics ──────────────────────────────────────────────

@dataclass
class StartupIssue:
    """A non-fatal issue detected during application startup."""
    diagnostic: Diagnostic
    detail: str  # extra context (e.g. detected PyTorch version)


def _pascal_cu130_mismatch() -> tuple[bool, str]:
    """Detect a Pascal GPU paired with a cu13x torch wheel.

    Returns ``(is_mismatch, detail_string)``. ``is_mismatch`` is True
    when *all* of the following are true:

      * torch can be imported,
      * torch was built against CUDA 13.x (i.e. ``torch.version.cuda``
        starts with "13."),
      * an NVIDIA GPU is present and reachable,
      * the device's compute capability is below (7, 0) — i.e. Pascal
        (sm_60 / sm_61) or older.

    Every step is wrapped in try/except so a broken or missing torch
    can never crash startup. The function is the safety guard for the
    1.9.1 → 1.9.2 in-app upgrade path: existing Pascal users keep
    their cu130 wheel after a code-only update and would otherwise
    silently crash the next time they run inference.
    """
    try:
        import torch
    except Exception:
        return False, ""

    try:
        cuda_str = (torch.version.cuda or "").strip()
    except Exception:
        return False, ""
    if not cuda_str.startswith("13."):
        return False, ""

    try:
        if not torch.cuda.is_available():
            return False, ""
    except Exception:
        return False, ""

    try:
        device_count = torch.cuda.device_count()
    except Exception:
        return False, ""
    if device_count <= 0:
        return False, ""

    # Inspect device 0 — if a user has multiple GPUs and one is Pascal
    # and another isn't, we still want to flag it because inference
    # picks device 0 by default.
    try:
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
    except Exception:
        return False, ""

    if not isinstance(cap, tuple) or len(cap) != 2:
        return False, ""

    if cap >= (7, 0):
        # Turing or newer — fully supported by cu130 wheels.
        return False, ""

    detail = (
        f"Detected {name} (compute capability {cap[0]}.{cap[1]}) "
        f"with PyTorch CUDA {cuda_str}. cu13x wheels do not include "
        f"sm_{cap[0]}{cap[1]} kernels."
    )
    return True, detail


def run_startup_diagnostics(device: str) -> list[StartupIssue]:
    """Check the runtime environment and return any issues found.

    Called once after ``detect_device()`` during MainWindow init.
    """
    issues: list[StartupIssue] = []

    # 1. CPU-only device
    if device == "cpu":
        diag = next((d for d in _DIAGNOSTICS if d.id == "gpu-required"), None)
        if diag:
            detail = _get_torch_detail()
            issues.append(StartupIssue(diag, detail))

    # 1b. GPU present but PyTorch is CPU-only wheel (+cpu)
    if device == "cpu":
        try:
            import torch
            if "+cpu" in torch.__version__:
                has_nvidia = False
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    has_nvidia = pynvml.nvmlDeviceGetCount() > 0
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
                if has_nvidia:
                    diag = next(
                        (d for d in _DIAGNOSTICS if d.id == "pytorch-cpu-wheel"),
                        None,
                    )
                    if diag:
                        issues.append(StartupIssue(
                            diag,
                            f"PyTorch {torch.__version__} (CPU-only) with NVIDIA GPU detected",
                        ))
        except ImportError:
            pass

    # 2. Python version outside supported range
    import sys
    vi = sys.version_info
    if vi.major != 3 or vi.minor < 10 or vi.minor > 13:
        diag = next((d for d in _DIAGNOSTICS if d.id == "python-version"), None)
        if diag:
            issues.append(StartupIssue(
                diag,
                f"Detected Python {vi.major}.{vi.minor}.{vi.micro}",
            ))

    # 2b. Pascal GPU detected with a cu13x torch wheel — the 1.9.1 →
    # 1.9.2 upgrade path leaves existing cu130 runtimes in place, so a
    # GTX 10-series user who clicks "Check for Updates" will still
    # crash later when inference runs. Catch them at startup with a
    # clear "download the full installer" advisory rather than letting
    # them hit "no kernel image is available" mid-job.
    #
    # The detection helper is fully defensive — wrapped in try/except
    # at every layer — but we belt-and-suspenders the call site too so
    # any unexpected failure here can't break startup for any user.
    try:
        is_mismatch, mismatch_detail = _pascal_cu130_mismatch()
        if is_mismatch:
            diag = next(
                (d for d in _DIAGNOSTICS if d.id == "pascal-cu130-mismatch"),
                None,
            )
            if diag:
                issues.append(StartupIssue(diag, mismatch_detail))
    except Exception as exc:
        logger.warning("Pascal/cu130 startup check failed: %s", exc)

    # 3. FFmpeg missing, too old, or invalid build
    try:
        from backend.ffmpeg_tools import validate_ffmpeg_install
        result = validate_ffmpeg_install()
        if not result.ok:
            # Pick the right diagnostic based on whether FFmpeg was found at all
            diag_id = "ffmpeg-missing"
            if result.ffmpeg_path:
                diag_id = "ffmpeg-invalid"
            diag = next((d for d in _DIAGNOSTICS if d.id == diag_id), None)
            if diag:
                issues.append(StartupIssue(diag, result.message))
    except Exception as exc:
        logger.warning("FFmpeg startup check failed: %s", exc)

    return issues


def _get_torch_detail() -> str:
    """Build a one-line detail string about the PyTorch install."""
    try:
        import torch
        ver = torch.__version__
        cuda = torch.version.cuda or "none"
        return f"PyTorch {ver}, CUDA toolkit: {cuda}"
    except ImportError:
        return "PyTorch is not installed"


# ── Context-aware diagnostic step resolution ────────────────────────
#
# The static ``Diagnostic.steps`` lists were written for developers
# running from a git clone with a venv. That guidance is actively wrong
# for users on the Windows / macOS installer (they have no ``.venv`` to
# activate) and for users without an NVIDIA GPU (no amount of pip-
# installing a different torch wheel will help). This resolver returns
# the right steps for the current runtime instead of always echoing the
# dev-mode defaults.


def _runtime_context() -> dict:
    """Snapshot the bits of runtime state that affect diagnostic advice."""
    import sys

    ctx = {
        "is_frozen": bool(getattr(sys, "frozen", False)),
        "platform": sys.platform,
        "has_nvidia": False,
    }
    # NVIDIA GPU presence check — intentionally uses pynvml (driver-level)
    # rather than torch.cuda.is_available() because we may be handling a
    # broken torch install where cuda.is_available() returns False for
    # reasons unrelated to the actual hardware.
    try:
        import pynvml
        pynvml.nvmlInit()
        ctx["has_nvidia"] = pynvml.nvmlDeviceGetCount() > 0
        pynvml.nvmlShutdown()
    except Exception:
        pass
    return ctx


def _steps_no_nvidia_gpu() -> list[str]:
    """Shared step block for any torch/CUDA issue when no NVIDIA card is
    present. Stops the user from chasing pip reinstalls that cannot
    possibly help them."""
    return [
        "EZ-CorridorKey requires an NVIDIA GPU.\n"
        "There is no CPU fallback for the keying models.",
        "Check that your computer has an NVIDIA graphics card\n"
        "(GeForce RTX, GTX, Quadro, or similar).",
        "Install the latest NVIDIA driver from:\n"
        "    https://www.nvidia.com/Download/index.aspx",
        "Open a terminal and run:\n"
        "    nvidia-smi\n"
        "If this command is not found, the driver is not installed.",
        "If you do not have an NVIDIA GPU, EZ-CorridorKey's\n"
        "AI pipelines cannot run on this machine.",
    ]


def _steps_frozen_reinstall(reason: str) -> list[str]:
    """Shared step block for installer-build users: reinstalling the
    official build is the correct remedy, not pip."""
    return [
        f"{reason}",
        "Download the latest installer from:\n"
        "    https://github.com/edenaion/EZ-CorridorKey/releases/latest",
        "Run the installer over your existing install —\n"
        "your projects and settings will be preserved.",
        "If the problem persists after reinstalling, please\n"
        "click Report Issue on GitHub so we can investigate.",
    ]


def _resolve_gpu_required(ctx: dict, base: list[str]) -> list[str]:
    if not ctx["has_nvidia"]:
        return _steps_no_nvidia_gpu()
    if ctx["is_frozen"]:
        return _steps_frozen_reinstall(
            "Your installer build could not detect CUDA, but an NVIDIA\n"
            "GPU is present. This usually means the bundled PyTorch is\n"
            "corrupted or the NVIDIA driver is too old."
        )
    return base  # dev mode + NVIDIA → original venv instructions


def _resolve_pytorch_cpu_wheel(ctx: dict, base: list[str]) -> list[str]:
    if not ctx["has_nvidia"]:
        return _steps_no_nvidia_gpu()
    if ctx["is_frozen"]:
        return _steps_frozen_reinstall(
            "A CPU-only PyTorch wheel was detected in your installer\n"
            "build. This should not happen in a shipped release."
        )
    return base


def _resolve_triton_missing(ctx: dict, base: list[str]) -> list[str]:
    if ctx["is_frozen"]:
        # Triton is optional (torch.compile speedup); in a frozen build
        # this is a build-time packaging miss, not something the user
        # can fix themselves. Downgrade the instructions to "safe to
        # ignore" + Report Issue.
        return [
            "This warning is safe to ignore — inference will still\n"
            "work, it just falls back to eager mode (slightly slower).",
            "If you see this in an installer build, please click\n"
            "Report Issue on GitHub so we can investigate the package.",
        ]
    return base  # dev mode → original venv instructions


def _resolve_six_metapath(ctx: dict, base: list[str]) -> list[str]:
    if not ctx["has_nvidia"]:
        return _steps_no_nvidia_gpu()
    if ctx["is_frozen"]:
        return _steps_frozen_reinstall(
            "A protobuf/grpc compatibility error was hit inside the\n"
            "frozen build. This should not happen in a shipped release."
        )
    return base


def _resolve_missing_checkpoint(ctx: dict, base: list[str]) -> list[str]:
    if ctx["is_frozen"]:
        return [
            "Model weights are downloaded by the first-run Setup\n"
            "Wizard. If you dismissed it, reopen the app and follow\n"
            "the setup prompts to fetch CorridorKey.pth (~383 MB).",
            "If the wizard does not appear, go to:\n"
            "    Edit > Preferences > Re-run Setup Wizard",
            "If the download fails, please click Report Issue on\n"
            "GitHub with your system info.",
        ]
    return base


# Maps diagnostic id → resolver function. Entries are optional: if a
# diagnostic has no resolver we fall through to ``Diagnostic.steps``.
_STEP_RESOLVERS = {
    "gpu-required": _resolve_gpu_required,
    "pytorch-cpu-wheel": _resolve_pytorch_cpu_wheel,
    "triton-missing": _resolve_triton_missing,
    "six-metapath-importer": _resolve_six_metapath,
    "missing-checkpoint": _resolve_missing_checkpoint,
}


def resolve_steps(diag: Diagnostic) -> list[str]:
    """Return the fix steps for *diag* tailored to the current runtime.

    Dialogs should call this instead of reading ``diag.steps`` directly
    so that users on the installer build never see dev-mode guidance
    like "activate the virtual environment" (which fails on frozen
    builds) and users without an NVIDIA GPU never see pip install
    commands (which cannot fix their situation).
    """
    handler = _STEP_RESOLVERS.get(diag.id)
    if handler is None:
        return diag.steps
    try:
        ctx = _runtime_context()
        resolved = handler(ctx, diag.steps)
        return resolved or diag.steps
    except Exception as exc:
        logger.warning("Diagnostic step resolver failed for %s: %s", diag.id, exc)
        return diag.steps

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
            "Activate the virtual environment:\n    .venv\\Scripts\\activate",
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
            "Download CorridorKey.pth from the project release page\nor the link in the README.",
            "Place the file in:\n    CorridorKeyModule/checkpoints/CorridorKey.pth",
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
            "Verify both commands work:\n    ffmpeg -version\n    ffprobe -version",
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
            "Activate the virtual environment:\n    .venv\\Scripts\\activate",
            "Install triton-windows:\n    pip install triton-windows",
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
            "Uninstall the CPU-only build:\n    pip uninstall torch torchvision -y",
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
            '    python -c "import torch; print(torch.__version__)"',
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
            "Run the GVM weight downloader:\n    python -m gvm_core.download",
            "Or download manually from the README link and place\nin gvm_core/weights/",
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
            "Set the environment variable for low-VRAM mode:\n    set CORRIDORKEY_OPT_MODE=lowvram",
            "If the problem persists, your GPU may not have enough\nVRAM for this clip resolution.",
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
                        issues.append(
                            StartupIssue(
                                diag,
                                f"PyTorch {torch.__version__} (CPU-only) with NVIDIA GPU detected",
                            )
                        )
        except ImportError:
            pass

    # 2. Python version outside supported range
    import sys

    vi = sys.version_info
    if vi.major != 3 or vi.minor < 10 or vi.minor > 13:
        diag = next((d for d in _DIAGNOSTICS if d.id == "python-version"), None)
        if diag:
            issues.append(
                StartupIssue(
                    diag,
                    f"Detected Python {vi.major}.{vi.minor}.{vi.micro}",
                )
            )

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

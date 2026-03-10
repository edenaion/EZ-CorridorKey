"""Diagnostic dialog — pattern-matches known errors to actionable solutions.

When the backend raises an error that matches a known pattern, this dialog
presents the user with a clear explanation and step-by-step fix instructions
instead of a raw traceback.  If none of the steps resolve the issue, the
user can click through to file a GitHub issue with system info pre-filled.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

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
        id="ffmpeg-missing",
        title="FFmpeg Not Found",
        pattern=re.compile(
            r"FFmpeg not found|"
            r"ffmpeg.*not.*found|"
            r"ffprobe.*not.*found",
            re.IGNORECASE,
        ),
        explanation=(
            "FFmpeg is required for video import/export but was not found "
            "on your system PATH."
        ),
        steps=[
            "Download FFmpeg from https://ffmpeg.org/download.html",
            "Extract and place ffmpeg.exe in one of:\n"
            "    • C:\\Program Files\\ffmpeg\\bin\\\n"
            "    • Or any folder on your system PATH",
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


# ── Dialog ────────────────────────────────────────────────────────────

class DiagnosticDialog(QDialog):
    """Shows a known-error diagnosis with fix steps and a Report Issue fallback."""

    def __init__(
        self,
        diagnostic: Diagnostic,
        error_msg: str,
        *,
        detail: str = "",
        gpu_info: dict | None = None,
        recent_errors: list[str] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Diagnostic: {diagnostic.title}")
        self.setMinimumWidth(540)
        self.setMinimumHeight(320)
        self.setModal(True)

        self._diagnostic = diagnostic
        self._error_msg = error_msg
        self._gpu_info = gpu_info
        self._recent_errors = recent_errors

        root = QVBoxLayout(self)
        root.setSpacing(12)

        # ── Title ──
        title_lbl = QLabel(diagnostic.title)
        title_lbl.setStyleSheet(
            "QLabel { font-size: 16px; font-weight: bold; color: #FFF203; }"
        )
        root.addWidget(title_lbl)

        # ── Explanation ──
        explain_lbl = QLabel(diagnostic.explanation)
        explain_lbl.setWordWrap(True)
        explain_lbl.setStyleSheet("QLabel { color: #ccc; }")
        root.addWidget(explain_lbl)

        # ── Detail (optional runtime context) ──
        if detail:
            detail_lbl = QLabel(detail)
            detail_lbl.setStyleSheet(
                "QLabel { color: #999; font-style: italic; font-size: 12px; }"
            )
            root.addWidget(detail_lbl)

        # ── Steps (scrollable) ──
        steps_area = QScrollArea()
        steps_area.setWidgetResizable(True)
        steps_area.setFrameShape(QScrollArea.Shape.NoFrame)
        steps_widget = QWidget()
        steps_layout = QVBoxLayout(steps_widget)
        steps_layout.setContentsMargins(8, 4, 8, 4)
        steps_layout.setSpacing(8)

        for i, step in enumerate(diagnostic.steps, 1):
            step_lbl = QLabel(f"{i}.  {step}")
            step_lbl.setWordWrap(True)
            step_lbl.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            step_lbl.setStyleSheet(
                "QLabel { color: #E0E0E0; font-family: 'Consolas', monospace; "
                "font-size: 12px; background: #1a1a1a; border-radius: 4px; "
                "padding: 6px 8px; }"
            )
            steps_layout.addWidget(step_lbl)

        steps_layout.addStretch()
        steps_area.setWidget(steps_widget)
        root.addWidget(steps_area, stretch=1)

        # ── Error detail (collapsed) ──
        error_lbl = QLabel(f"Error: {error_msg}")
        error_lbl.setWordWrap(True)
        error_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        error_lbl.setStyleSheet(
            "QLabel { color: #888; font-size: 11px; padding: 4px; }"
        )
        root.addWidget(error_lbl)

        # ── Buttons ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        report_btn = QPushButton("Report Issue on GitHub")
        report_btn.setStyleSheet(
            "QPushButton { background: #333; color: #ccc; padding: 6px 14px; }"
        )
        report_btn.clicked.connect(self._on_report)
        btn_row.addWidget(report_btn)

        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)

        root.addLayout(btn_row)

    def _on_report(self) -> None:
        """Open the Report Issue dialog pre-filled with diagnostic context."""
        from ui.widgets.report_issue_dialog import ReportIssueDialog

        dlg = ReportIssueDialog(
            gpu_info=self._gpu_info,
            recent_errors=self._recent_errors or [self._error_msg],
            parent=self,
        )
        # Pre-fill title with diagnostic name
        dlg._title_edit.setText(self._diagnostic.title)
        dlg.exec()


class StartupDiagnosticDialog(QDialog):
    """Non-blocking startup warning showing one or more environment issues."""

    def __init__(
        self,
        issues: list[StartupIssue],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Startup Diagnostics")
        self.setMinimumWidth(560)
        self.setMinimumHeight(300)
        self.setModal(True)

        root = QVBoxLayout(self)
        root.setSpacing(12)

        header = QLabel(
            "EZ-CorridorKey detected issues with your environment that "
            "may prevent some features from working correctly."
        )
        header.setWordWrap(True)
        header.setStyleSheet("QLabel { color: #ccc; font-size: 13px; }")
        root.addWidget(header)

        # ── Scrollable issue list ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(4, 4, 4, 4)
        inner_layout.setSpacing(16)

        for issue in issues:
            card = self._build_issue_card(issue)
            inner_layout.addWidget(card)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll, stretch=1)

        # ── Buttons ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("Continue Anyway")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)
        root.addLayout(btn_row)

    @staticmethod
    def _build_issue_card(issue: StartupIssue) -> QWidget:
        card = QWidget()
        card.setStyleSheet(
            "QWidget { background: #1a1a1a; border: 1px solid #333; "
            "border-radius: 6px; }"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        title = QLabel(issue.diagnostic.title)
        title.setStyleSheet(
            "QLabel { font-size: 14px; font-weight: bold; color: #FFF203; "
            "background: transparent; border: none; }"
        )
        layout.addWidget(title)

        if issue.detail:
            det = QLabel(issue.detail)
            det.setStyleSheet(
                "QLabel { color: #999; font-size: 12px; font-style: italic; "
                "background: transparent; border: none; }"
            )
            layout.addWidget(det)

        explain = QLabel(issue.diagnostic.explanation)
        explain.setWordWrap(True)
        explain.setStyleSheet(
            "QLabel { color: #bbb; font-size: 12px; "
            "background: transparent; border: none; }"
        )
        layout.addWidget(explain)

        steps_text = "\n".join(
            f"  {i}. {s}" for i, s in enumerate(issue.diagnostic.steps, 1)
        )
        steps = QLabel(steps_text)
        steps.setWordWrap(True)
        steps.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        steps.setStyleSheet(
            "QLabel { color: #E0E0E0; font-family: 'Consolas', monospace; "
            "font-size: 11px; background: transparent; border: none; "
            "padding: 4px 0; }"
        )
        layout.addWidget(steps)

        return card

"""Report Issue dialog — Help > Report Issue.

Collects a user description, auto-gathers system info and recent error logs,
then opens the default browser to a pre-filled GitHub issue page.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from urllib.parse import quote

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)

_GITHUB_ISSUES_URL = "https://github.com/edenaion/EZ-CorridorKey/issues/new"
_MAX_URL_LENGTH = 7500  # stay well under 8192 to avoid 414 errors


def _get_app_version() -> str:
    """Read app version from pyproject.toml (single source of truth)."""
    import tomllib

    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(sys._MEIPASS, "pyproject.toml"))
    candidates.append(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pyproject.toml")
    )
    for path in candidates:
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)["project"]["version"]
        except Exception:
            continue
    return "unknown"


class ReportIssueDialog(QDialog):
    """Dialog that collects bug info and opens a pre-filled GitHub issue."""

    def __init__(
        self,
        gpu_info: dict | None = None,
        recent_errors: list[str] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Report Issue")
        self.setMinimumWidth(500)
        self.setMinimumHeight(420)
        self.setModal(True)

        self._gpu_info = gpu_info or {}
        self._recent_errors = recent_errors or []

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # --- Title ---
        layout.addWidget(QLabel("Issue title:"))
        self._title_edit = QLineEdit()
        self._title_edit.setPlaceholderText("Brief summary of the problem")
        self._title_edit.setStyleSheet(
            "QLineEdit { background: #1a1a1a; color: #E0E0E0; "
            "border: 1px solid #555; border-radius: 3px; padding: 4px; }"
        )
        layout.addWidget(self._title_edit)

        # --- Description ---
        layout.addWidget(QLabel("What happened?"))
        self._desc_edit = QTextEdit()
        self._desc_edit.setPlaceholderText(
            "Describe what you were doing and what went wrong.\n"
            "Steps to reproduce are very helpful."
        )
        self._desc_edit.setMaximumHeight(120)
        self._desc_edit.setStyleSheet(
            "QTextEdit { background: #1a1a1a; color: #E0E0E0; "
            "border: 1px solid #555; border-radius: 3px; padding: 4px; }"
        )
        layout.addWidget(self._desc_edit)

        # --- System info preview ---
        info_group = QGroupBox("System info (auto-collected, included in report)")
        info_layout = QVBoxLayout(info_group)
        self._info_preview = QTextEdit()
        self._info_preview.setReadOnly(True)
        self._info_preview.setMaximumHeight(140)
        self._info_preview.setStyleSheet(
            "QTextEdit { background: #1a1a1a; color: #aaa; font-size: 11px; }"
        )
        self._info_preview.setPlainText(self._build_system_info_text())
        info_layout.addWidget(self._info_preview)
        layout.addWidget(info_group)

        # --- GitHub account notice ---
        notice = QLabel(
            "This will open GitHub in your browser. A free GitHub account is "
            "required to submit issues. Your report is also copied to the "
            "clipboard in case you need to paste it after logging in."
        )
        notice.setWordWrap(True)
        notice.setStyleSheet("QLabel { color: #888; font-size: 11px; }")
        layout.addWidget(notice)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        open_btn = QPushButton("Open GitHub")
        open_btn.setDefault(True)
        open_btn.clicked.connect(self._on_open)
        btn_layout.addWidget(open_btn)

        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # System info
    # ------------------------------------------------------------------

    @staticmethod
    def _os_string() -> str:
        """Return OS string, fixing Python's Win11-reports-as-Win10 bug."""
        os_str = platform.platform()
        if os_str.startswith("Windows-10"):
            try:
                build = int(platform.version().split(".")[-1])
                if build >= 22000:
                    os_str = os_str.replace("Windows-10", "Windows-11", 1)
            except (ValueError, IndexError):
                pass
        return os_str

    def _gather_info_pairs(self) -> list[tuple[str, str]]:
        """Collect all system info as (label, value) pairs."""
        pairs: list[tuple[str, str]] = [
            ("App Version", _get_app_version()),
            ("OS", self._os_string()),
            ("Python", platform.python_version()),
        ]

        # PyTorch + CUDA
        try:
            import torch

            pairs.append(("PyTorch", torch.version.__version__))
            if torch.cuda.is_available():
                pairs.append(("CUDA", torch.version.cuda or "N/A"))
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                pairs.append(("Compute", "MPS (Apple Metal)"))
            else:
                pairs.append(("Compute", "CPU"))
        except ImportError:
            pairs.append(("PyTorch", "not installed"))

        # GPU + VRAM
        gpu = self._gpu_info
        if gpu.get("name"):
            pairs.append(("GPU", gpu["name"]))
            if "total_gb" in gpu and "used_gb" in gpu:
                pairs.append(
                    ("VRAM", f"{gpu['total_gb']:.1f} GB total, {gpu['used_gb']:.1f} GB used")
                )
        else:
            pairs.append(("GPU", "N/A"))

        # Display resolution
        screen = QApplication.primaryScreen()
        if screen:
            sz = screen.size()
            pairs.append(("Display", f"{sz.width()}x{sz.height()}"))

        return pairs

    def _build_system_info_text(self) -> str:
        """Plain-text preview of system info for the user to see."""
        lines = [f"{label}: {value}" for label, value in self._gather_info_pairs()]
        if self._recent_errors:
            lines.append("")
            lines.append(f"Recent errors/warnings ({len(self._recent_errors)}):")
            for entry in self._recent_errors[:20]:
                lines.append(f"  {entry}")
        return "\n".join(lines)

    def _build_system_info_md(self) -> str:
        """Markdown-formatted system info for the GitHub issue body."""
        return "\n".join(f"- **{label}:** {value}" for label, value in self._gather_info_pairs())

    # ------------------------------------------------------------------
    # URL builder
    # ------------------------------------------------------------------

    def _build_body(self) -> str:
        """Build the full markdown issue body."""
        desc = self._desc_edit.toPlainText().strip() or "(no description provided)"
        parts = [
            "## Description",
            desc,
            "",
            "## System Info",
            self._build_system_info_md(),
        ]
        if self._recent_errors:
            parts.append("")
            parts.append("## Recent Errors")
            parts.append("```")
            parts.extend(self._recent_errors[:20])
            parts.append("```")
        return "\n".join(parts)

    def _build_url(self, body: str) -> str:
        title = self._title_edit.text().strip() or "Bug Report"
        url = f"{_GITHUB_ISSUES_URL}?title={quote(title)}&body={quote(body)}"
        if len(url) > _MAX_URL_LENGTH:
            # Trim error log lines to fit; keep description + system info
            short = body.split("## Recent Errors")[0].rstrip()
            short += "\n\n_Log truncated to fit URL limit. Full report is in your clipboard._"
            url = f"{_GITHUB_ISSUES_URL}?title={quote(title)}&body={quote(short)}"
        return url

    def _on_open(self) -> None:
        body = self._build_body()

        # Copy full body to clipboard so it survives GitHub login redirects
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(body)
            logger.info("Issue body copied to clipboard (%d chars)", len(body))

        url = self._build_url(body)
        logger.info("Opening GitHub issue URL (%d chars)", len(url))
        QDesktopServices.openUrl(QUrl(url))
        self.accept()

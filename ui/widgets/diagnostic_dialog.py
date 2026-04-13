"""Diagnostic dialog — pattern-matches known errors to actionable solutions.

When the backend raises an error that matches a known pattern, this dialog
presents the user with a clear explanation and step-by-step fix instructions
instead of a raw traceback.  If none of the steps resolve the issue, the
user can click through to file a GitHub issue with system info pre-filled.
"""

from __future__ import annotations

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

# Re-export data types and helpers so existing imports keep working.
from ui.widgets.diagnostic_checks import (  # noqa: F401
    Diagnostic,
    StartupIssue,
    match_diagnostic,
    run_startup_diagnostics,
)


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
        title_lbl.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; color: #FFF203; }")
        root.addWidget(title_lbl)

        # ── Explanation ──
        explain_lbl = QLabel(diagnostic.explanation)
        explain_lbl.setWordWrap(True)
        explain_lbl.setStyleSheet("QLabel { color: #ccc; }")
        root.addWidget(explain_lbl)

        # ── Detail (optional runtime context) ──
        if detail:
            detail_lbl = QLabel(detail)
            detail_lbl.setStyleSheet("QLabel { color: #999; font-style: italic; font-size: 12px; }")
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
            step_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
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
        error_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        error_lbl.setStyleSheet("QLabel { color: #888; font-size: 11px; padding: 4px; }")
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
            "QWidget { background: #1a1a1a; border: 1px solid #333; border-radius: 6px; }"
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
            "QLabel { color: #bbb; font-size: 12px; background: transparent; border: none; }"
        )
        layout.addWidget(explain)

        steps_text = "\n".join(f"  {i}. {s}" for i, s in enumerate(issue.diagnostic.steps, 1))
        steps = QLabel(steps_text)
        steps.setWordWrap(True)
        steps.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        steps.setStyleSheet(
            "QLabel { color: #E0E0E0; font-family: 'Consolas', monospace; "
            "font-size: 11px; background: transparent; border: none; "
            "padding: 4px 0; }"
        )
        layout.addWidget(steps)

        return card

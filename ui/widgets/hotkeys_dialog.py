"""Hotkeys dialog — Edit > Hotkeys.

Displays all keyboard shortcuts grouped by category. Users can click
a key binding button to rebind, with conflict detection and reset-to-default.
"""

from __future__ import annotations

import logging

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QWidget,
    QLineEdit,
    QMessageBox,
)
from PySide6.QtCore import QKeyCombination, Qt
from PySide6.QtGui import QKeySequence, QKeyEvent

from ui.shortcut_registry import ShortcutRegistry, CATEGORY_ORDER

logger = logging.getLogger(__name__)


class KeyBindButton(QPushButton):
    """Button that captures a key sequence when clicked."""

    _STYLE_DEFAULT = (
        "QPushButton { background: #1A1900; color: #CCCCAA; "
        "border: 1px solid #2A2910; padding: 4px 12px; font-size: 12px; }"
        "QPushButton:hover { border-color: #454430; background: #252413; }"
    )
    _STYLE_CUSTOM = (
        "QPushButton { background: #1A1900; color: #FFF203; font-weight: 700; "
        "border: 1px solid #454430; padding: 4px 12px; font-size: 12px; }"
        "QPushButton:hover { border-color: #FFF203; background: #252413; }"
    )
    _STYLE_RECORDING = (
        "QPushButton { background: #2A2910; color: #FFF203; "
        "border: 1px solid #FFF203; padding: 4px 12px; font-size: 12px; }"
    )

    def __init__(
        self,
        action_id: str,
        key_str: str,
        registry: ShortcutRegistry,
        dialog: HotkeysDialog,
        parent=None,
    ):
        super().__init__(parent)
        self.action_id = action_id
        self._registry = registry
        self._dialog = dialog
        self._recording = False
        self.setFixedWidth(160)
        self.setFixedHeight(28)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self._start_recording)
        self.display_key(key_str)

    def display_key(self, key_str: str) -> None:
        """Show the key sequence, highlighting non-default bindings."""
        self.setText(key_str if key_str else "(none)")
        is_default = self._registry.is_default(self.action_id)
        self.setStyleSheet(self._STYLE_DEFAULT if is_default else self._STYLE_CUSTOM)

    def _start_recording(self) -> None:
        self._recording = True
        self.setText("Press a key...")
        self.setStyleSheet(self._STYLE_RECORDING)
        self.grabKeyboard()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not self._recording:
            super().keyPressEvent(event)
            return

        key = event.key()
        # Ignore bare modifier presses
        if key in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta):
            return

        # Escape during recording = cancel (not rebind)
        if key == Qt.Key_Escape and event.modifiers() == Qt.NoModifier:
            self._cancel_recording()
            return

        # Build QKeySequence via QKeyCombination (PySide6-safe)
        combo = QKeyCombination(event.modifiers(), Qt.Key(key))
        seq = QKeySequence(combo)
        key_str = seq.toString()

        # Check conflicts
        conflicts = self._registry.find_conflicts(self.action_id, key_str)
        if conflicts:
            conflict_names = ", ".join(
                d.display_name for d in self._registry.definitions() if d.action_id in conflicts
            )
            reply = QMessageBox.warning(
                self,
                "Shortcut Conflict",
                f'"{key_str}" is already assigned to:\n{conflict_names}\n\n'
                "Reassign anyway? The conflicting binding will be cleared.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self._cancel_recording()
                return
            # Clear conflicting bindings
            for cid in conflicts:
                self._registry.set_key(cid, "")
                self._dialog.refresh_button(cid)

        self._recording = False
        self.releaseKeyboard()
        self._registry.set_key(self.action_id, key_str)
        self.display_key(key_str)

    def _cancel_recording(self) -> None:
        self._recording = False
        self.releaseKeyboard()
        self.display_key(self._registry.get_key(self.action_id))

    def focusOutEvent(self, event) -> None:
        if self._recording:
            self._cancel_recording()
        super().focusOutEvent(event)


class HotkeysDialog(QDialog):
    """Keyboard shortcut configuration dialog."""

    def __init__(self, registry: ShortcutRegistry, parent=None):
        super().__init__(parent)
        self._registry = registry
        self._buttons: dict[str, KeyBindButton] = {}
        # Snapshot for cancel/revert
        self._original_overrides = registry.snapshot_overrides()

        self.setWindowTitle("Hotkeys")
        self.setMinimumSize(540, 480)
        self.setModal(True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # Filter bar
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter shortcuts...")
        self._filter.setStyleSheet(
            "QLineEdit { background: #1A1900; border: 1px solid #2A2910; "
            "color: #CCCCAA; padding: 6px 8px; font-size: 12px; }"
            "QLineEdit:focus { border-color: #FFF203; }"
        )
        self._filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self._filter)

        # Scrollable shortcut list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: #0E0D00; border: 1px solid #2A2910; }")
        content = QWidget()
        content.setStyleSheet("background: #0E0D00;")
        self._content_layout = QVBoxLayout(content)
        self._content_layout.setContentsMargins(8, 8, 8, 8)
        self._content_layout.setSpacing(2)

        # Build rows grouped by category
        # category -> (header_label, [(row_widget, action_id), ...])
        self._cat_groups: dict[str, tuple[QLabel, list[tuple[QWidget, str]]]] = {}
        self._rows: list[tuple[QWidget, str]] = []  # flat list for filter

        for cat in CATEGORY_ORDER:
            defs_in_cat = [d for d in self._registry.definitions() if d.category == cat]
            if not defs_in_cat:
                continue

            header = QLabel(cat.upper())
            header.setStyleSheet(
                "color: #808070; font-size: 10px; font-weight: 700; "
                "letter-spacing: 1px; padding: 8px 0 4px 0; background: transparent;"
            )
            self._content_layout.addWidget(header)
            cat_rows: list[tuple[QWidget, str]] = []
            self._cat_groups[cat] = (header, cat_rows)

            for defn in defs_in_cat:
                row_widget = QWidget()
                row_widget.setFixedHeight(34)
                row_widget.setStyleSheet("background: transparent;")
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(4, 0, 4, 0)
                row_layout.setSpacing(8)

                name_label = QLabel(defn.display_name)
                name_label.setStyleSheet(
                    "color: #CCCCAA; font-size: 12px; background: transparent;"
                )
                name_label.setMinimumWidth(240)
                row_layout.addWidget(name_label)

                row_layout.addStretch()

                btn = KeyBindButton(
                    defn.action_id,
                    self._registry.get_key(defn.action_id),
                    self._registry,
                    self,
                )
                self._buttons[defn.action_id] = btn
                row_layout.addWidget(btn)

                reset_btn = QPushButton("Reset")
                reset_btn.setFixedWidth(50)
                reset_btn.setFixedHeight(24)
                reset_btn.setStyleSheet(
                    "QPushButton { background: transparent; color: #555540; "
                    "border: 1px solid #2A2910; font-size: 10px; padding: 2px 6px; }"
                    "QPushButton:hover { color: #CCCCAA; border-color: #454430; }"
                )
                reset_btn.setToolTip(f"Reset to default: {defn.default_key}")
                reset_btn.setCursor(Qt.PointingHandCursor)
                aid = defn.action_id
                reset_btn.clicked.connect(lambda checked, a=aid: self._reset_single(a))
                row_layout.addWidget(reset_btn)

                self._content_layout.addWidget(row_widget)
                self._rows.append((row_widget, defn.action_id))
                cat_rows.append((row_widget, defn.action_id))

        self._content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        # Bottom button bar
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        reset_all_btn = QPushButton("Reset All to Defaults")
        reset_all_btn.setStyleSheet(
            "QPushButton { background: #1A1900; color: #999980; "
            "border: 1px solid #2A2910; padding: 6px 14px; font-size: 11px; }"
            "QPushButton:hover { color: #CCCCAA; border-color: #454430; }"
        )
        reset_all_btn.setCursor(Qt.PointingHandCursor)
        reset_all_btn.clicked.connect(self._reset_all)
        btn_layout.addWidget(reset_all_btn)

        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        cancel_btn.setStyleSheet(
            "QPushButton { background: #1A1900; color: #999980; "
            "border: 1px solid #2A2910; padding: 6px 14px; font-size: 11px; }"
            "QPushButton:hover { color: #CCCCAA; border-color: #454430; }"
        )
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("OK")
        ok_btn.setFixedWidth(80)
        ok_btn.setDefault(True)
        ok_btn.setStyleSheet(
            "QPushButton { background: #2A2910; color: #FFF203; "
            "border: 1px solid #454430; padding: 6px 14px; "
            "font-size: 11px; font-weight: 700; }"
            "QPushButton:hover { background: #3A3920; border-color: #FFF203; }"
        )
        ok_btn.setCursor(Qt.PointingHandCursor)
        ok_btn.clicked.connect(self._on_accept)
        btn_layout.addWidget(ok_btn)

        layout.addLayout(btn_layout)

    def refresh_button(self, action_id: str) -> None:
        """Refresh a specific button's display (called after conflict resolution)."""
        btn = self._buttons.get(action_id)
        if btn:
            btn.display_key(self._registry.get_key(action_id))

    def _apply_filter(self, text: str) -> None:
        """Show/hide rows based on filter text matching action name or key."""
        text = text.lower()
        for row_widget, action_id in self._rows:
            defn = self._registry.get_def(action_id)
            if defn is None:
                continue
            key_str = self._registry.get_key(action_id).lower()
            visible = (
                text in defn.display_name.lower()
                or text in key_str
                or text in defn.category.lower()
            )
            row_widget.setVisible(visible)
        # Hide category headers when all their rows are hidden
        for cat, (header, cat_rows) in self._cat_groups.items():
            any_visible = any(rw.isVisible() for rw, _ in cat_rows)
            header.setVisible(any_visible)

    def _reset_single(self, action_id: str) -> None:
        self._registry.reset_key(action_id)
        self.refresh_button(action_id)

    def _reset_all(self) -> None:
        reply = QMessageBox.question(
            self,
            "Reset All Shortcuts",
            "Reset all shortcuts to their default values?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._registry.reset_all()
            for action_id in self._buttons:
                self.refresh_button(action_id)

    def _on_cancel(self) -> None:
        """Revert all in-memory changes and close."""
        self._registry.restore_overrides(self._original_overrides)
        self.reject()

    def _on_accept(self) -> None:
        """Save overrides and close."""
        self._registry.save_overrides()
        self.accept()

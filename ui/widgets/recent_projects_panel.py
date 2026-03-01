"""Recent projects panel for the welcome screen.

Shows a scrollable list of project cards representing recently-opened
workspaces. Each card shows the project name, path, clip count, and
last opened date, with a delete button.
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QPushButton, QScrollArea, QMessageBox,
)
from PySide6.QtCore import Qt, Signal

from ui.recent_sessions import RecentSessionsStore, RecentSession


class RecentProjectCard(QFrame):
    """Single project card — clickable row with delete button."""

    clicked = Signal(str)   # workspace_path
    delete_clicked = Signal(str)  # workspace_path

    def __init__(self, session: RecentSession, parent=None):
        super().__init__(parent)
        self._workspace_path = session.workspace_path
        self.setObjectName("projectCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(56)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 8, 6)
        layout.setSpacing(8)

        # Left: text info
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)

        # Top row: name + clip count
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)

        name_label = QLabel(session.display_name)
        name_label.setObjectName("projectCardName")
        top_row.addWidget(name_label)

        top_row.addStretch()

        meta_parts = []
        if session.clip_count > 0:
            meta_parts.append(f"{session.clip_count} clip{'s' if session.clip_count != 1 else ''}")
        try:
            dt = datetime.fromtimestamp(session.last_opened)
            meta_parts.append(dt.strftime("%b %d, %Y"))
        except (ValueError, OSError):
            pass
        if meta_parts:
            meta_label = QLabel("  \u00B7  ".join(meta_parts))
            meta_label.setObjectName("projectCardMeta")
            top_row.addWidget(meta_label)

        text_layout.addLayout(top_row)

        # Bottom row: path (elided)
        path_label = QLabel(session.workspace_path)
        path_label.setObjectName("projectCardPath")
        text_layout.addWidget(path_label)

        layout.addLayout(text_layout, 1)

        # Delete button
        delete_btn = QPushButton("\u00D7")  # ×
        delete_btn.setObjectName("projectDeleteBtn")
        delete_btn.setFixedSize(24, 24)
        delete_btn.setToolTip("Remove project")
        delete_btn.clicked.connect(self._on_delete)
        layout.addWidget(delete_btn)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._workspace_path)
        super().mousePressEvent(event)

    def _on_delete(self):
        self.delete_clicked.emit(self._workspace_path)


class RecentProjectsPanel(QWidget):
    """Scrollable list of recent project cards."""

    project_selected = Signal(str)   # workspace_path
    project_deleted = Signal(str)    # workspace_path

    def __init__(self, store: RecentSessionsStore, parent=None):
        super().__init__(parent)
        self._store = store
        self.setObjectName("recentProjectsPanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("RECENT PROJECTS")
        header.setObjectName("recentProjectsHeader")
        layout.addWidget(header)

        # Scroll area for cards
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        layout.addWidget(self._scroll, 1)

        # Container for card widgets
        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._card_layout = QVBoxLayout(self._container)
        self._card_layout.setContentsMargins(0, 0, 0, 0)
        self._card_layout.setSpacing(2)
        self._card_layout.addStretch()
        self._scroll.setWidget(self._container)

        # Empty hint
        self._empty_label = QLabel("No recent projects")
        self._empty_label.setObjectName("recentProjectsEmpty")
        self._empty_label.setAlignment(Qt.AlignCenter)

        self.refresh()

    def refresh(self) -> None:
        """Rebuild the card list from the store."""
        # Clear existing cards (keep stretch at end)
        while self._card_layout.count() > 1:
            item = self._card_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        sessions = self._store.get_all()

        if not sessions:
            self._card_layout.insertWidget(0, self._empty_label)
            self._empty_label.show()
            return

        self._empty_label.hide()
        self._empty_label.setParent(None)

        for session in sessions:
            card = RecentProjectCard(session)
            card.clicked.connect(self.project_selected.emit)
            card.delete_clicked.connect(self._on_delete_requested)
            self._card_layout.insertWidget(self._card_layout.count() - 1, card)

    def _on_delete_requested(self, workspace_path: str) -> None:
        """Handle delete button click with confirmation dialog."""
        name = os.path.basename(workspace_path)

        msg = QMessageBox(self)
        msg.setWindowTitle("Remove Project")
        msg.setText(f"Remove \"{name}\" from recent projects?")
        msg.setInformativeText("You can remove it from the list only, or also delete the workspace files from disk.")

        remove_btn = msg.addButton("Remove from List", QMessageBox.AcceptRole)
        delete_btn = msg.addButton("Delete Files", QMessageBox.DestructiveRole)
        msg.addButton(QMessageBox.Cancel)
        msg.setDefaultButton(remove_btn)

        msg.exec()
        clicked = msg.clickedButton()

        if clicked == remove_btn:
            self._store.remove(workspace_path)
            self.project_deleted.emit(workspace_path)
            self.refresh()
        elif clicked == delete_btn:
            self._store.remove(workspace_path)
            try:
                if os.path.isdir(workspace_path):
                    shutil.rmtree(workspace_path)
            except OSError as e:
                QMessageBox.warning(self, "Delete Failed", f"Could not delete workspace:\n{e}")
            self.project_deleted.emit(workspace_path)
            self.refresh()

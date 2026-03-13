"""Recent projects panel for the welcome screen.

Shows a scrollable list of project cards representing recently-opened
projects. Each card shows the project name, path, and last opened date,
with an always-visible delete button. Right-click for rename/delete.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys

logger = logging.getLogger(__name__)

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QPushButton, QScrollArea, QMessageBox,
    QMenu, QInputDialog,
)
from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QAction, QDesktopServices

from ui.recent_sessions import RecentSessionsStore, RecentSession


class RecentProjectCard(QFrame):
    """Single project card — clickable row with always-visible X button."""

    clicked = Signal(str)         # workspace_path
    delete_clicked = Signal(str)  # workspace_path
    rename_clicked = Signal(str)  # workspace_path

    def __init__(self, session: RecentSession, parent=None):
        super().__init__(parent)
        self._workspace_path = session.workspace_path
        self.setObjectName("projectCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(72)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 8, 4)
        layout.setSpacing(6)

        # Left: two icon buttons, evenly distributed in full card height
        btn_layout = QVBoxLayout()
        btn_layout.setContentsMargins(2, 0, 2, 0)
        btn_layout.setSpacing(0)

        btn_layout.addStretch(1)

        folder_btn = QPushButton("\uD83D\uDCC2")  # open folder icon
        folder_btn.setObjectName("projectFolderBtn")
        folder_btn.setFixedSize(16, 16)
        folder_btn.setToolTip("Open in Finder" if sys.platform == "darwin" else "Open in Explorer")
        folder_btn.clicked.connect(self._on_open_folder)
        btn_layout.addWidget(folder_btn, 0, Qt.AlignHCenter)

        btn_layout.addStretch(2)

        delete_btn = QPushButton("\u00D7")  # ×
        delete_btn.setObjectName("projectDeleteBtn")
        delete_btn.setFixedSize(16, 16)
        delete_btn.setToolTip("Remove project")
        delete_btn.clicked.connect(self._on_delete)
        btn_layout.addWidget(delete_btn, 0, Qt.AlignHCenter)

        btn_layout.addStretch(1)

        layout.addLayout(btn_layout)

        # Right: text info
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)

        name_label = QLabel(session.display_name)
        name_label.setObjectName("projectCardName")
        text_layout.addWidget(name_label)

        path_label = QLabel(session.workspace_path)
        path_label.setObjectName("projectCardPath")
        text_layout.addWidget(path_label)

        layout.addLayout(text_layout, 1)

    def enterEvent(self, event):
        from ui.sounds.audio_manager import UIAudio
        UIAudio.hover()
        super().enterEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            from ui.sounds.audio_manager import UIAudio
            UIAudio.click()  # ProjectCard is not QPushButton, manual click needed
            self.clicked.emit(self._workspace_path)
        super().mousePressEvent(event)

    def _on_delete(self):
        self.delete_clicked.emit(self._workspace_path)

    def _on_open_folder(self):
        if os.path.isdir(self._workspace_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._workspace_path))

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        rename_action = QAction("Rename Project", self)
        rename_action.triggered.connect(lambda: self.rename_clicked.emit(self._workspace_path))
        menu.addAction(rename_action)

        menu.addSeparator()

        delete_action = QAction("Delete Project", self)
        delete_action.triggered.connect(lambda: self.delete_clicked.emit(self._workspace_path))
        menu.addAction(delete_action)

        menu.exec(self.mapToGlobal(pos))


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

    @staticmethod
    def _is_projects_root(path: str) -> bool:
        """Check if a path is the Projects root directory."""
        try:
            from backend.project import projects_root
            return os.path.normcase(os.path.abspath(path)) == os.path.normcase(
                os.path.abspath(projects_root())
            )
        except Exception:
            return False

    def refresh(self) -> None:
        """Rebuild the card list from the store, filtering out the Projects root."""
        # Clear existing cards (keep stretch at end)
        while self._card_layout.count() > 1:
            item = self._card_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        sessions = self._store.get_all()
        # Filter out the Projects root — it's not a project, just the container
        sessions = [s for s in sessions if not self._is_projects_root(s.workspace_path)]

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
            card.rename_clicked.connect(self._on_rename_requested)
            self._card_layout.insertWidget(self._card_layout.count() - 1, card)

    def _on_rename_requested(self, workspace_path: str) -> None:
        """Rename a project via dialog — updates project.json display_name."""
        from backend.project import get_display_name, set_display_name

        current_name = get_display_name(workspace_path)
        new_name, ok = QInputDialog.getText(
            self, "Rename Project", "Project name:", text=current_name,
        )
        if ok and new_name.strip() and new_name.strip() != current_name:
            new_name = new_name.strip()
            set_display_name(workspace_path, new_name)
            # Update the recent sessions store entry
            self._store.add_or_update(workspace_path, new_name, 1, force=True)
            self.refresh()

    def _on_delete_requested(self, workspace_path: str) -> None:
        """Handle delete button click with two-step confirmation.

        Step 1: Remove from List / Delete from Disk / Cancel
        Step 2 (disk only): Confirm with path shown

        Safety: refuses to delete the Projects root directory.
        """
        from backend.project import get_display_name
        name = get_display_name(workspace_path)

        # Safety: never allow disk-delete of the Projects root
        if self._is_projects_root(workspace_path):
            self._store.remove(workspace_path)
            self.refresh()
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("Remove Project")
        msg.setText(f"Remove \"{name}\" from recent projects?")
        msg.setInformativeText(
            "Remove from List: hides it from recents (files stay on disk).\n"
            "Delete from Disk: permanently deletes the project folder."
        )

        remove_btn = msg.addButton("Remove from List", QMessageBox.AcceptRole)
        delete_btn = msg.addButton("Delete from Disk", QMessageBox.DestructiveRole)
        msg.addButton(QMessageBox.Cancel)
        msg.setDefaultButton(remove_btn)

        msg.exec()
        clicked = msg.clickedButton()

        if clicked == remove_btn:
            self._store.remove(workspace_path)
            self.project_deleted.emit(workspace_path)
            self.refresh()
        elif clicked == delete_btn:
            # Second confirmation with path shown
            confirm = QMessageBox.warning(
                self, "Confirm Delete",
                f"Permanently delete this project folder?\n\n{workspace_path}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if confirm == QMessageBox.Yes:
                self._store.remove(workspace_path)
                try:
                    # Safety: only delete if path is inside the Projects root
                    from backend.project import projects_root
                    root = os.path.realpath(projects_root())
                    target = os.path.realpath(workspace_path)
                    if not target.startswith(root + os.sep):
                        logger.warning(f"Refusing to delete outside Projects folder: {target}")
                        raise OSError(
                            f"Refusing to delete outside Projects folder: {target}"
                        )
                    if os.path.isdir(workspace_path):
                        logger.info(f"Deleting project folder: {workspace_path}")
                        shutil.rmtree(workspace_path)
                except OSError as e:
                    logger.error(f"Failed to delete project: {e}")
                    QMessageBox.warning(
                        self, "Delete Failed",
                        f"Could not delete project:\n{e}",
                    )
                self.project_deleted.emit(workspace_path)
                self.refresh()

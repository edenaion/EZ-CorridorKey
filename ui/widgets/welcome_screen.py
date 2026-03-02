"""Welcome screen — full-area drop zone with small recents sidebar.

The entire center area is a clickable/droppable zone (brand logo, prompt,
browse button) — exactly like the original welcome screen. A small panel
on the left shows recent projects as clickable cards. Clicks on the
drop zone open the file dialog; clicks on project cards open that project.
"""
from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtGui import QPixmap, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

from ui.recent_sessions import RecentSessionsStore
from ui.widgets.recent_projects_panel import RecentProjectsPanel
from backend.project import is_video_file, is_image_file, VIDEO_FILE_FILTER


def _load_brand_logo(size: int) -> QPixmap | None:
    """Load brand logo from SVG (crisp at any size) with PNG fallback."""
    if getattr(sys, 'frozen', False):
        base = os.path.join(sys._MEIPASS, "ui", "theme")
    else:
        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "theme")

    # Prefer SVG — renders at native resolution, no scaling artifacts
    svg_path = os.path.join(base, "corridorkey.svg")
    if os.path.isfile(svg_path):
        renderer = QSvgRenderer(svg_path)
        if renderer.isValid():
            img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
            img.fill(Qt.transparent)
            painter = QPainter(img)
            painter.setRenderHint(QPainter.Antialiasing)
            renderer.render(painter)
            painter.end()
            return QPixmap.fromImage(img)

    # Fallback to PNG
    png_path = os.path.join(base, "corridorkey.png")
    if os.path.isfile(png_path):
        px = QPixmap(png_path)
        if not px.isNull():
            return px.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return None


class _DropZone(QWidget):
    """Full-area clickable drop zone — the main content of the welcome screen.

    Contains the brand logo, prompt text, and browse button. Clicking
    anywhere on this widget opens the file dialog.
    """

    browse_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(False)  # Parent handles DnD
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)

        # Brand logo (PNG scaled up)
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet("background: transparent;")
        logo_px = _load_brand_logo(160)
        if logo_px:
            logo_label.setPixmap(logo_px)
        layout.addWidget(logo_label, alignment=Qt.AlignCenter)

        layout.addSpacing(20)

        # Prompt text
        prompt = QLabel("Drop Videos or Click to Import")
        prompt.setAlignment(Qt.AlignCenter)
        prompt.setObjectName("welcomePrompt")
        layout.addWidget(prompt)

        layout.addSpacing(12)

        # Browse button
        browse_btn = QPushButton("Browse...")
        browse_btn.setObjectName("welcomeBrowse")
        browse_btn.setFixedWidth(200)
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.clicked.connect(self.browse_requested.emit)
        browse_btn.installEventFilter(self)
        layout.addWidget(browse_btn, alignment=Qt.AlignCenter)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter and obj.objectName() == "welcomeBrowse":
            from ui.sounds.audio_manager import UIAudio
            UIAudio.hover()
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.browse_requested.emit()
        else:
            super().mousePressEvent(event)


class WelcomeScreen(QWidget):
    """Full-window welcome screen: small recents sidebar + full drop zone.

    Layout:
    ┌─────────────┬────────────────────────────────────────────┐
    │ RECENT      │                                            │
    │ PROJECTS    │          [Brand Logo]                      │
    │             │  Drop Videos or Click to Import             │
    │ [Card 1]   │          [Browse...]                        │
    │ [Card 2]   │                                            │
    │ [Card 3]   │                                            │
    │             │                                            │
    └─────────────┴────────────────────────────────────────────┘
    """

    folder_selected = Signal(str)     # directory path
    files_selected = Signal(list)     # list of file paths
    recent_project_opened = Signal(str)  # workspace path from recents

    def __init__(self, store: RecentSessionsStore, parent=None):
        super().__init__(parent)
        self._store = store
        self.setAcceptDrops(True)
        self.setObjectName("welcomeScreen")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Left: recent projects sidebar (fixed width)
        self._recents_panel = RecentProjectsPanel(store)
        self._recents_panel.setFixedWidth(320)
        self._recents_panel.project_selected.connect(self.recent_project_opened.emit)
        self._recents_panel.project_deleted.connect(self._on_project_deleted)
        layout.addWidget(self._recents_panel)

        # Divider line
        divider = QWidget()
        divider.setFixedWidth(1)
        divider.setStyleSheet("background-color: #2A2910;")
        layout.addWidget(divider)

        # Right: full drop zone (fills all remaining space)
        self._drop_zone = _DropZone()
        self._drop_zone.browse_requested.connect(self._on_browse)
        layout.addWidget(self._drop_zone, 1)

    def refresh_recents(self) -> None:
        """Refresh the recent projects list from the store."""
        self._recents_panel.refresh()

    def _on_browse(self) -> None:
        """Open file dialog — users can pick video files."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "",
            VIDEO_FILE_FILTER,
        )
        if paths:
            self.files_selected.emit(paths)

    def _on_project_deleted(self, workspace_path: str) -> None:
        """Handle project deletion — already removed from store by the panel."""
        pass  # Panel handles store removal and refresh itself

    # ── Drag and drop ──

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isdir(path):
                    event.acceptProposedAction()
                    return
                if is_video_file(path) or is_image_file(path):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event) -> None:
        folders = []
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                folders.append(path)
            elif os.path.isfile(path):
                if is_video_file(path) or is_image_file(path):
                    files.append(path)

        # Prefer folder if dropped
        if folders:
            self.folder_selected.emit(folders[0])
        elif files:
            self.files_selected.emit(files)

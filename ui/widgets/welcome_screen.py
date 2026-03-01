"""Welcome screen — Topaz-style drop zone shown before any clips are loaded.

Centered layout with dashed border, brand clapperboard icon, drop prompt, and Browse button.
Accepts drag-and-drop of video files or folders. Entire area is clickable.
"""
from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QRect, QPoint, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QPolygon, QImage

# Video extensions accepted for drag-and-drop
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v"}
# Image sequence extensions
_IMAGE_EXTS = {".exr", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".dpx"}

_YELLOW = QColor("#FFF203")
_BLACK = QColor("#0A0A00")
_DARK = QColor("#1A1900")


class ClapperIcon(QWidget):
    """Custom-painted clapperboard icon in brand colors (black + yellow).

    Renders the Corridor Crew logo on the board body.
    """

    def __init__(self, size: int = 96, parent=None):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Load brand logo for the board body
        self._logo: QImage | None = None
        if getattr(sys, 'frozen', False):
            base = os.path.join(sys._MEIPASS, "ui", "theme")
        else:
            base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "theme")
        logo_path = os.path.join(base, "corridorkey.png")
        if os.path.isfile(logo_path):
            self._logo = QImage(logo_path)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        s = self._size

        # Scale factors
        def sx(v): return int(v * s / 96)
        def sy(v): return int(v * s / 96)

        # Board body (bottom portion) — black with yellow border
        board = QRect(sx(12), sy(38), sx(72), sy(46))
        p.setPen(QPen(_YELLOW, 2))
        p.setBrush(_BLACK)
        p.drawRect(board)

        # Draw logo centered on the board body
        if self._logo and not self._logo.isNull():
            logo_size = sx(34)
            logo_x = sx(12) + (sx(72) - logo_size) // 2
            logo_y = sy(38) + (sy(46) - logo_size) // 2
            p.drawImage(QRectF(logo_x, logo_y, logo_size, logo_size),
                        self._logo, QRectF(self._logo.rect()))

        # Clapper top (hinged part) — yellow with black stripes
        clapper = QPolygon([
            QPoint(sx(8), sy(36)),
            QPoint(sx(18), sy(12)),
            QPoint(sx(86), sy(12)),
            QPoint(sx(86), sy(36)),
        ])
        p.setPen(QPen(_BLACK, 2))
        p.setBrush(_YELLOW)
        p.drawPolygon(clapper)

        # Diagonal stripes on clapper (the classic clapperboard pattern)
        p.setPen(Qt.NoPen)
        p.setBrush(_BLACK)
        stripe_positions = [24, 42, 60, 78]
        for x_pos in stripe_positions:
            stripe = QPolygon([
                QPoint(sx(x_pos), sy(12)),
                QPoint(sx(x_pos + 6), sy(12)),
                QPoint(sx(x_pos + 2), sy(36)),
                QPoint(sx(x_pos - 4), sy(36)),
            ])
            p.drawPolygon(stripe)

        p.end()


class WelcomeScreen(QWidget):
    """Full-window clickable drop zone shown on startup before a clips folder is opened.

    The entire area is clickable (opens file dialog). Also accepts drag-and-drop.
    """

    folder_selected = Signal(str)   # emitted with a directory path
    files_selected = Signal(list)   # emitted with a list of file paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("welcomeScreen")
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # Brand clapperboard icon
        icon = ClapperIcon(96)
        layout.addWidget(icon, alignment=Qt.AlignCenter)

        layout.addSpacing(16)

        # Prompt text
        prompt = QLabel("Drop Videos or Click to Import")
        prompt.setAlignment(Qt.AlignCenter)
        prompt.setObjectName("welcomePrompt")
        layout.addWidget(prompt)

        layout.addSpacing(12)

        # Browse button (also clickable, but entire area works too)
        browse_btn = QPushButton("Browse...")
        browse_btn.setObjectName("welcomeBrowse")
        browse_btn.setFixedWidth(200)
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.clicked.connect(self._on_browse)
        layout.addWidget(browse_btn, alignment=Qt.AlignCenter)

    def _on_browse(self) -> None:
        """Open file dialog — users can pick video files or a folder."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.mxf *.webm *.m4v);;All Files (*)",
        )
        if paths:
            self.files_selected.emit(paths)

    def mousePressEvent(self, event) -> None:
        """Clicking anywhere on the welcome screen opens the file dialog."""
        if event.button() == Qt.LeftButton:
            self._on_browse()
        else:
            super().mousePressEvent(event)

    # ── Drag and drop ──

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isdir(path):
                    event.acceptProposedAction()
                    return
                ext = os.path.splitext(path)[1].lower()
                if ext in _VIDEO_EXTS or ext in _IMAGE_EXTS:
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
                ext = os.path.splitext(path)[1].lower()
                if ext in _VIDEO_EXTS or ext in _IMAGE_EXTS:
                    files.append(path)

        # Prefer folder if dropped
        if folders:
            self.folder_selected.emit(folders[0])
        elif files:
            self.files_selected.emit(files)

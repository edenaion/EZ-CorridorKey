"""Center panel — frame preview viewport.

Phase 1: QLabel + QPixmap (CPU rendering, zero GPU overhead).
Displays the latest preview frame, downsized to viewport dimensions.
QPixmap is ONLY created/updated on the main thread (Codex finding).

Phase 3 will add: before/after split, frame scrubber, zoom/pan.
Phase 4 will upgrade to QOpenGLWidget.
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage


class PreviewViewport(QWidget):
    """Center panel frame preview using QLabel + QPixmap."""

    frame_changed = Signal(int)  # current frame index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Preview image label
        self._preview_label = QLabel()
        self._preview_label.setObjectName("previewLabel")
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setScaledContents(False)
        self._preview_label.setText("No clip selected")
        self._preview_label.setStyleSheet(
            "QLabel { background-color: #0A0A00; color: #808070; font-size: 16px; }"
        )
        layout.addWidget(self._preview_label, 1)

        self._current_pixmap: QPixmap | None = None
        self._current_frame: int = -1
        self._clip_name: str = ""

    def load_preview_from_file(self, file_path: str, clip_name: str, frame_index: int) -> None:
        """Load a preview image from a temp file path.

        Called on the MAIN THREAD when the worker emits preview_ready.
        QPixmap creation happens here, never in the worker thread.
        """
        if not os.path.isfile(file_path):
            return

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            return

        self._current_pixmap = pixmap
        self._current_frame = frame_index
        self._clip_name = clip_name
        self._display_pixmap()

    def load_frame_from_dir(self, output_dir: str, frame_index: int) -> None:
        """Load a specific frame from the output comp directory.

        Used for browsing completed clips.
        """
        comp_dir = os.path.join(output_dir, "Comp")
        if not os.path.isdir(comp_dir):
            self.show_placeholder("No output frames")
            return

        frames = sorted(os.listdir(comp_dir))
        if not frames or frame_index >= len(frames):
            self.show_placeholder("Frame not available")
            return

        path = os.path.join(comp_dir, frames[frame_index])
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return

        self._current_pixmap = pixmap
        self._current_frame = frame_index
        self._display_pixmap()

    def show_placeholder(self, text: str = "No clip selected") -> None:
        """Show placeholder text instead of a frame."""
        self._preview_label.setPixmap(QPixmap())
        self._preview_label.setText(text)
        self._current_pixmap = None
        self._current_frame = -1

    def _display_pixmap(self) -> None:
        """Scale and display the current pixmap to fit the viewport."""
        if self._current_pixmap is None:
            return

        scaled = self._current_pixmap.scaled(
            self._preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._preview_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        """Re-scale preview when viewport resizes."""
        super().resizeEvent(event)
        if self._current_pixmap is not None:
            self._display_pixmap()

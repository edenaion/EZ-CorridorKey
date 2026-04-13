"""Bottom I/O tray panel — Topaz-style Input/Exports thumbnail strips.

Shows two horizontal-scrolling rows:
- INPUT (N): All loaded clips with thumbnails, + ADD button
- EXPORTS (N): Only COMPLETE clips with output thumbnails

Clicking a card selects the clip and loads it in the preview viewport.
Ctrl+click to multi-select clips for batch operations.
Right-click on INPUT cards shows project context menu.
Supports drag-and-drop of video files and folders.
"""

from __future__ import annotations

import logging
import os
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QPushButton,
    QMenu,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal

from backend import ClipEntry, ClipState
from backend.project import VIDEO_FILE_FILTER, is_image_file, is_video_file
from ui.models.clip_model import ClipListModel
from ui.widgets.thumbnail_canvas import ThumbnailCanvas, _STATE_COLORS  # noqa: F401
from ui.widgets.io_tray_actions import IOTrayActionsMixin

logger = logging.getLogger(__name__)


class IOTrayPanel(IOTrayActionsMixin, QWidget):
    """Bottom panel with Input and Exports thumbnail strips.

    Input section shows all loaded clips with + ADD button.
    Exports section shows only COMPLETE clips.
    Clicking a card emits clip_clicked. Right-click shows context menu.
    Supports drag-and-drop of video files and folders.
    """

    clip_clicked = Signal(object)  # ClipEntry
    selection_changed = Signal(list)  # list[ClipEntry] — multi-select changed
    clips_dir_changed = Signal(str)  # folder path (import folder)
    files_imported = Signal(list)  # list of video file paths
    sequence_folder_imported = Signal(str)  # folder path containing image sequence
    image_files_dropped = Signal(list)  # list of image file paths (for <5 popup)
    extract_requested = Signal(list)  # list[ClipEntry] — re-run extraction
    export_video_requested = Signal(object, str)  # ClipEntry, source_dir — export as video
    reset_in_out_requested = Signal()  # clear all in/out markers

    def __init__(self, model: ClipListModel, parent=None):
        super().__init__(parent)
        self.setObjectName("ioTrayPanel")
        self._model = model
        self._select_anchor: str | None = (
            None  # last single-clicked clip name for Shift+click range
        )
        self.setMinimumHeight(80)
        self.setMaximumHeight(600)
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Content: two strips in a splitter (synced with dual viewer divider)
        self._tray_splitter = QSplitter(Qt.Horizontal)
        self._tray_splitter.setHandleWidth(1)
        self._tray_splitter.setStyleSheet("QSplitter::handle { background-color: #2A2910; }")

        # Input section
        input_widget = QWidget()
        input_section = QVBoxLayout(input_widget)
        input_section.setContentsMargins(0, 0, 0, 0)
        input_section.setSpacing(0)

        # Header row: INPUT (N) label + stretch + ADD button
        input_header_row = QHBoxLayout()
        input_header_row.setContentsMargins(0, 0, 4, 0)
        input_header_row.setSpacing(0)

        self._input_header = QLabel("INPUT (0)")
        self._input_header.setObjectName("trayHeader")
        input_header_row.addWidget(self._input_header)
        input_header_row.addStretch()

        self._reset_io_btn = QPushButton("RESET I/O")
        self._reset_io_btn.setObjectName("trayAddBtn")
        self._reset_io_btn.setToolTip("Clear in/out markers on all clips")
        self._reset_io_btn.clicked.connect(self._on_reset_in_out)
        input_header_row.addWidget(self._reset_io_btn)

        self._add_btn = QPushButton("+ ADD")
        self._add_btn.setObjectName("trayAddBtn")
        self._add_btn.setToolTip("Import clips — choose a folder or video file(s)")
        self._add_btn.clicked.connect(self._on_add_clicked)
        input_header_row.addWidget(self._add_btn)

        input_section.addLayout(input_header_row)

        self._input_scroll = QScrollArea()
        self._input_scroll.setObjectName("trayScroll")
        self._input_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._input_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._input_scroll.setWidgetResizable(True)

        self._input_canvas = ThumbnailCanvas(thumbnail_kind="input")
        self._input_canvas.card_clicked.connect(self._on_single_click)
        self._input_canvas.multi_select_toggled.connect(self._on_multi_select_toggle)
        self._input_canvas.shift_select_requested.connect(self._on_shift_select)
        self._input_canvas.context_menu_requested.connect(self._on_context_menu)
        self._input_scroll.setWidget(self._input_canvas)

        input_section.addWidget(self._input_scroll, 1)
        self._tray_splitter.addWidget(input_widget)

        # Exports section
        export_widget = QWidget()
        export_section = QVBoxLayout(export_widget)
        export_section.setContentsMargins(0, 0, 0, 0)
        export_section.setSpacing(0)

        self._export_header = QLabel("EXPORTS (0)")
        self._export_header.setObjectName("trayHeader")
        export_section.addWidget(self._export_header)

        self._export_scroll = QScrollArea()
        self._export_scroll.setObjectName("trayScroll")
        self._export_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._export_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._export_scroll.setWidgetResizable(True)

        self._export_canvas = ThumbnailCanvas(
            show_manifest_tooltip=True,
            thumbnail_kind="export",
        )
        self._export_canvas.card_clicked.connect(self._on_export_click)
        self._export_canvas.card_double_clicked.connect(self._on_export_click)
        self._export_canvas.context_menu_requested.connect(self._on_export_context_menu)
        self._export_canvas.folder_icon_clicked.connect(self._open_export_folder)
        self._export_scroll.setWidget(self._export_canvas)

        export_section.addWidget(self._export_scroll, 1)
        self._tray_splitter.addWidget(export_widget)

        # Equal split by default (synced from main window)
        self._tray_splitter.setSizes([500, 500])
        self._tray_splitter.setStretchFactor(0, 1)
        self._tray_splitter.setStretchFactor(1, 1)
        self._tray_splitter.splitterMoved.connect(self._on_splitter_moved)

        layout.addWidget(self._tray_splitter)

        # Connect to model signals for auto-rebuild
        self._model.modelReset.connect(self._rebuild)
        self._model.dataChanged.connect(self._on_data_changed)
        self._model.layoutChanged.connect(self._rebuild)
        self._model.clip_count_changed.connect(lambda _: self._rebuild())

    def _on_splitter_moved(self, pos: int, index: int) -> None:
        """Force canvas reflow when the INPUT/EXPORTS splitter moves."""
        self._input_canvas._reflow()
        self._export_canvas._reflow()

    # ── + ADD button ──

    def _on_add_clicked(self) -> None:
        """Show import menu below the +ADD button."""
        menu = QMenu(self)
        menu.addAction("Import Folder...", self._import_folder)
        menu.addAction("Import Video(s)...", self._import_videos)
        menu.addAction("Import Image Sequence...", self._import_image_sequence)
        menu.exec(self._add_btn.mapToGlobal(self._add_btn.rect().bottomLeft()))

    def _on_reset_in_out(self) -> None:
        """Reset all in/out markers with double confirmation."""
        # Count clips that actually have in/out markers
        clips_with_range = [c for c in self._model.clips if c.in_out_range is not None]
        if not clips_with_range:
            QMessageBox.information(
                self,
                "No Markers",
                "No clips have in/out markers set.",
            )
            return

        n = len(clips_with_range)
        # First confirmation
        result = QMessageBox.question(
            self,
            "Reset In/Out Markers",
            f"This will clear in/out markers on {n} clip{'s' if n > 1 else ''}.\n\n"
            "All clips will revert to full-clip processing.\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if result != QMessageBox.Yes:
            return

        # Second confirmation
        result2 = QMessageBox.warning(
            self,
            "Confirm Reset",
            f"Are you sure? This cannot be undone.\n\n"
            f"Clearing in/out markers on {n} clip{'s' if n > 1 else ''}.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if result2 != QMessageBox.Yes:
            return

        self.reset_in_out_requested.emit()
        logger.info(f"Reset in/out markers requested for {n} clips")

    def _import_folder(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Clips Directory",
            "",
            QFileDialog.ShowDirsOnly,
        )
        if dir_path:
            self.clips_dir_changed.emit(dir_path)

    def _import_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            VIDEO_FILE_FILTER,
        )
        if paths:
            self.files_imported.emit(paths)

    def _import_image_sequence(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Image Sequence Folder",
            "",
            QFileDialog.ShowDirsOnly,
        )
        if dir_path:
            self.sequence_folder_imported.emit(dir_path)

    # ── Drag-and-drop ──

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        from backend.project import folder_has_image_sequence

        folders = []
        video_files = []
        image_files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                folders.append(path)
            elif os.path.isfile(path):
                if is_video_file(path):
                    video_files.append(path)
                elif is_image_file(path):
                    image_files.append(path)

        if folders:
            # Check if the folder contains images (sequence) vs subdirectories (project)
            folder = folders[0]
            if folder_has_image_sequence(folder):
                self.sequence_folder_imported.emit(folder)
            else:
                self.clips_dir_changed.emit(folder)
        elif video_files and not image_files:
            self.files_imported.emit(video_files)
        elif image_files:
            # Image files dropped — emit for popup handling in main_window
            self.image_files_dropped.emit(image_files)
        elif video_files:
            self.files_imported.emit(video_files)

    # ── Single / Multi-select management ──

    def _on_single_click(self, clip: ClipEntry) -> None:
        """Plain left-click on INPUT card — single-select, sync both strips."""
        self._select_anchor = clip.name
        self._input_canvas.set_selected(clip.name)
        self._export_canvas.set_selected(clip.name)
        self.clip_clicked.emit(clip)

    def _on_export_click(self, clip: ClipEntry) -> None:
        """Click on EXPORT card — select same clip in both strips."""
        self._select_anchor = clip.name
        self._input_canvas.set_selected(clip.name)
        self._export_canvas.set_selected(clip.name)
        self.clip_clicked.emit(clip)

    def _on_multi_select_toggle(self, clip: ClipEntry) -> None:
        """Ctrl+click — toggle clip in/out of multi-selection set."""
        self._select_anchor = clip.name
        names = set(self._input_canvas._selected_names)
        if clip.name in names:
            names.discard(clip.name)
        else:
            names.add(clip.name)
        self._input_canvas.set_multi_selected(names)
        # Emit the full list of selected ClipEntry objects
        self.selection_changed.emit(self.get_selected_clips())
        # Also load the clicked clip in the viewer
        self.clip_clicked.emit(clip)

    def _on_shift_select(self, clip: ClipEntry) -> None:
        """Shift+click — select range from anchor to clicked clip."""
        all_clips = self._model.clips
        if not all_clips:
            return

        # Find anchor index (fall back to first clip if no anchor)
        anchor_idx = 0
        if self._select_anchor:
            for i, c in enumerate(all_clips):
                if c.name == self._select_anchor:
                    anchor_idx = i
                    break

        # Find clicked clip index
        click_idx = 0
        for i, c in enumerate(all_clips):
            if c.name == clip.name:
                click_idx = i
                break

        # Select everything between anchor and click (inclusive)
        lo = min(anchor_idx, click_idx)
        hi = max(anchor_idx, click_idx)
        names = {all_clips[i].name for i in range(lo, hi + 1)}
        self._input_canvas.set_multi_selected(names)
        self.selection_changed.emit(self.get_selected_clips())
        self.clip_clicked.emit(clip)

    def get_selected_clips(self) -> list[ClipEntry]:
        """Return all clips whose names are in the selection set."""
        names = self._input_canvas._selected_names
        return [c for c in self._model.clips if c.name in names]

    # ── Selection highlight ──

    def set_selected(self, name: str | None) -> None:
        """Set single-selected clip (clears multi-select, highlights in both strips)."""
        self._input_canvas.set_selected(name)
        self._export_canvas.set_selected(name)

    def set_multi_selected(self, names: set[str]) -> None:
        """Set multi-selected clips (for external callers)."""
        self._input_canvas.set_multi_selected(names)

    def selected_count(self) -> int:
        """Return count of selected clips."""
        return len(self._input_canvas._selected_names)

    # ── Rebuild ──

    def _rebuild(self) -> None:
        """Rebuild both strips from current model data."""
        all_clips = self._model.clips
        complete_clips = [c for c in all_clips if c.state == ClipState.COMPLETE]

        self._input_canvas.set_clips(all_clips, self._model)
        self._export_canvas.set_clips(complete_clips, self._model)

        self._input_header.setText(f"INPUT ({len(all_clips)})")
        self._export_header.setText(f"EXPORTS ({len(complete_clips)})")

    def _on_data_changed(self, top_left, bottom_right, roles) -> None:
        """Handle model data changes — lightweight repaint for progress,
        full rebuild only when clip set or states change."""
        current_clips = self._model.clips if self._model else []
        current_key = {(c.name, c.state) for c in current_clips}
        cached_key = {(c.name, c.state) for c in self._input_canvas._clips}
        thumbnail_roles = {
            ClipListModel.ThumbnailRole,
            ClipListModel.ExportThumbnailRole,
        }
        # Full rebuild when clip set or any clip state changes
        if current_key != cached_key:
            self._rebuild()
        else:
            if roles and any(role in thumbnail_roles for role in roles):
                for row in range(top_left.row(), bottom_right.row() + 1):
                    clip = self._model.get_clip(row)
                    if clip is None:
                        continue
                    self._input_canvas._thumb_cache.pop(clip.name, None)
                    self._export_canvas._thumb_cache.pop(clip.name, None)
            # Lightweight: just repaint the canvases (no thumbnail rescale)
            self._input_canvas.update()
            self._export_canvas.update()

    def refresh(self) -> None:
        """Force rebuild (called after worker completes a clip)."""
        self._rebuild()

    def sync_divider(self, left_px: int) -> None:
        """Set the IO tray divider position in pixels from the left edge."""
        total = self._tray_splitter.width()
        right = max(1, total - left_px)
        self._tray_splitter.setSizes([left_px, right])

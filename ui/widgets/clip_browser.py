"""Left panel — clip browser with list view, drag-drop, and folder selection.

Displays clips from ClipListModel. Supports:
- Click to select (loads in preview)
- Multi-select (ExtendedSelection) for batch operations
- [+ADD] button → QFileDialog to add clips directory
- Drag-and-drop folders
- Collapsible: hides entirely, shows floating expand button on viewer edge
"""
from __future__ import annotations

import os
import shutil
import logging

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListView, QFileDialog, QAbstractItemView, QStyledItemDelegate, QStyle,
    QMenu, QInputDialog, QMessageBox,
)
from PySide6.QtCore import Qt, Signal, QModelIndex
from PySide6.QtGui import QColor, QImage, QAction

from ui.models.clip_model import ClipListModel
from backend import ClipEntry, ClipState

logger = logging.getLogger(__name__)


class ClipCardDelegate(QStyledItemDelegate):
    """Custom delegate that renders clip cards in the list view."""

    def paint(self, painter, option, index):
        clip = index.data(ClipListModel.ClipEntryRole)
        if clip is None:
            return super().paint(painter, option, index)

        painter.save()
        rect = option.rect

        # Thumbnail area (left side)
        thumb_w, thumb_h = 60, 40
        thumb_left = rect.x() + 4
        text_left = thumb_left + thumb_w + 6  # text starts after thumbnail

        # Selected background
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(rect, QColor("#252413"))
            painter.setPen(QColor("#FFF203"))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
        elif option.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(rect, QColor("#1E1D13"))
            painter.setPen(QColor("#454430"))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
        else:
            painter.fillRect(rect, QColor("#1E1D13"))

        # Thumbnail
        thumb = index.data(ClipListModel.ThumbnailRole)
        thumb_y = rect.y() + (rect.height() - thumb_h) // 2
        if isinstance(thumb, QImage) and not thumb.isNull():
            painter.drawImage(thumb_left, thumb_y, thumb)
        else:
            # Placeholder dark rect
            painter.fillRect(thumb_left, thumb_y, thumb_w, thumb_h, QColor("#0A0A00"))
            painter.setPen(QColor("#3A3A30"))
            painter.drawRect(thumb_left, thumb_y, thumb_w - 1, thumb_h - 1)

        # State badge (after thumbnail)
        state_colors = {
            ClipState.EXTRACTING: "#FF8C00",
            ClipState.RAW: "#808070",
            ClipState.MASKED: "#009ADA",
            ClipState.READY: "#FFF203",
            ClipState.COMPLETE: "#22C55E",
            ClipState.ERROR: "#D10000",
        }
        badge_color = QColor(state_colors.get(clip.state, "#808070"))
        painter.setPen(badge_color)
        badge_font = painter.font()
        badge_font.setPointSize(9)
        badge_font.setBold(True)
        painter.setFont(badge_font)
        badge_rect = rect.adjusted(text_left - rect.x(), 6, -rect.width() + text_left - rect.x() + 70, 0)
        painter.drawText(badge_rect, Qt.AlignLeft | Qt.AlignTop, clip.state.value)

        # Processing indicator
        if clip.is_processing:
            painter.setPen(QColor("#FFF203"))
            proc_rect = rect.adjusted(rect.width() - 16, 6, -4, -rect.height() + 18)
            painter.drawText(proc_rect, Qt.AlignRight | Qt.AlignTop, "...")

        # Clip name
        painter.setPen(QColor("#E0E0E0"))
        name_font = painter.font()
        name_font.setPointSize(11)
        name_font.setBold(True)
        painter.setFont(name_font)
        name_rect = rect.adjusted(text_left - rect.x() + 48, 4, -8, -16)
        painter.drawText(name_rect, Qt.AlignLeft | Qt.AlignVCenter, clip.name)

        # Detail line (frame count or extraction progress)
        detail_font = painter.font()
        detail_font.setPointSize(9)
        detail_font.setBold(False)
        painter.setFont(detail_font)
        detail_rect = rect.adjusted(text_left - rect.x() + 48, 18, -8, -2)

        if clip.state == ClipState.EXTRACTING and clip.extraction_total > 0:
            # Show extraction progress text
            pct = int(clip.extraction_progress * 100)
            current = int(clip.extraction_progress * clip.extraction_total)
            painter.setPen(QColor("#FF8C00"))
            painter.drawText(
                detail_rect, Qt.AlignLeft | Qt.AlignVCenter,
                f"Extracting: {pct}% ({current}/{clip.extraction_total} frames)",
            )

            # Progress bar at bottom of card (4px tall)
            bar_h = 4
            bar_y = rect.y() + rect.height() - bar_h
            bar_x = rect.x()
            bar_w = rect.width()

            # Track
            painter.fillRect(bar_x, bar_y, bar_w, bar_h, QColor("#0A0A00"))
            # Fill
            fill_w = int(bar_w * clip.extraction_progress)
            if fill_w > 0:
                painter.fillRect(bar_x, bar_y, fill_w, bar_h, QColor("#FFF203"))
        elif clip.input_asset:
            painter.setPen(QColor("#808070"))
            detail_text = f"{clip.input_asset.frame_count} frames"
            if clip.input_asset.asset_type == "video":
                detail_text += " (video)"
            painter.drawText(detail_rect, Qt.AlignLeft | Qt.AlignVCenter, detail_text)

        # Warning indicator
        if clip.warnings:
            painter.setPen(QColor("#FFA500"))
            warn_rect = rect.adjusted(rect.width() - 30, 6, -4, 0)
            painter.drawText(warn_rect, Qt.AlignRight | Qt.AlignTop, f"({len(clip.warnings)})")

        painter.restore()

    def sizeHint(self, option, index):
        return option.rect.size() if option.rect.isValid() else super().sizeHint(option, index)


class ClipBrowser(QWidget):
    """Left panel clip browser with folder watching."""

    clip_selected = Signal(ClipEntry)
    clips_dir_changed = Signal(str)
    files_imported = Signal(list)   # list of video file paths to import
    clip_deleted = Signal(str)     # clip root_path — project folder removed
    clip_renamed = Signal(str, str)  # clip_name, new_display_name

    def __init__(self, model: ClipListModel, parent=None):
        super().__init__(parent)
        self.setObjectName("clipPanel")
        self._model = model
        self._clips_dir: str | None = None
        self._watcher = None
        self._watching = False

        self._expanded = True
        self._expanded_width = 220  # remember width for restore
        self._expand_btn: QPushButton | None = None  # floating expand button

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar
        header_widget = QWidget()
        header = QHBoxLayout(header_widget)
        header.setContentsMargins(4, 4, 4, 4)
        header.setSpacing(4)

        self._chevron = QPushButton("\u25C0")  # ◀
        self._chevron.setFixedSize(20, 20)
        self._chevron.setStyleSheet(
            "QPushButton { background: transparent; color: #808070; border: none; "
            "font-size: 10px; padding: 0; }"
            "QPushButton:hover { color: #FFF203; }"
        )
        self._chevron.setToolTip("Collapse clip panel")
        self._chevron.clicked.connect(self.toggle_collapse)
        header.addWidget(self._chevron)

        self._title = QLabel("CLIPS")
        self._title.setObjectName("sectionHeader")
        header.addWidget(self._title)
        header.addStretch()

        self._count_label = QLabel("0")
        self._count_label.setStyleSheet("color: #808070; font-size: 11px;")
        header.addWidget(self._count_label)
        layout.addWidget(header_widget)

        # List view — ExtendedSelection for batch ops
        self._list_view = QListView()
        self._list_view.setModel(model)
        self._list_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list_view.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._list_view.setUniformItemSizes(True)
        self._list_view.setSpacing(1)
        self._list_view.clicked.connect(self._on_item_clicked)
        self._list_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list_view.customContextMenuRequested.connect(self._on_context_menu)

        # Custom delegate for rendering
        delegate = ClipCardDelegate(self._list_view)
        self._list_view.setItemDelegate(delegate)

        # Fixed row height (48px to fit 40px thumbnail + padding)
        self._list_view.setStyleSheet("QListView::item { height: 48px; }")

        layout.addWidget(self._list_view, 1)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(10, 6, 10, 8)

        add_btn = QPushButton("+ ADD")
        add_btn.setToolTip("Import clips — choose a folder (image sequences) or video file(s)")
        add_btn.clicked.connect(self._on_add_clicked)
        btn_layout.addWidget(add_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Accept drops
        self.setAcceptDrops(True)

        # Connect model count signal
        model.clip_count_changed.connect(
            lambda count: self._count_label.setText(str(count))
        )

    def _get_or_create_expand_btn(self) -> QPushButton:
        """Create the floating expand button, parented to the center viewer panel."""
        if self._expand_btn is not None:
            return self._expand_btn

        from PySide6.QtWidgets import QSplitter
        splitter = self.parent()
        if isinstance(splitter, QSplitter) and splitter.count() >= 2:
            # Parent to the center panel (dual viewer) so it floats over it
            overlay_parent = splitter.widget(1)
        else:
            overlay_parent = self.parent() or self

        btn = QPushButton("\u25B6", overlay_parent)  # ▶
        btn.setFixedSize(24, 24)
        btn.setStyleSheet(
            "QPushButton { background: #1E1D13; color: #808070; border: 1px solid #454430; "
            "font-size: 10px; padding: 0; }"
            "QPushButton:hover { color: #FFF203; border-color: #FFF203; }"
        )
        btn.setToolTip("Expand clip panel")
        btn.clicked.connect(self.toggle_collapse)
        btn.hide()
        self._expand_btn = btn
        return btn

    def toggle_collapse(self) -> None:
        """Toggle between expanded and collapsed states.

        Collapsed: ClipBrowser fully hidden, floating expand button on viewer edge.
        Expanded: full clip browser with list, buttons, etc.
        """
        from PySide6.QtWidgets import QSplitter
        splitter = self.parent()
        is_splitter = isinstance(splitter, QSplitter)

        self._expanded = not self._expanded
        expand_btn = self._get_or_create_expand_btn()

        if self._expanded:
            # Show the browser, hide the floating button
            expand_btn.hide()
            self.show()
            self.setMinimumWidth(140)
            self.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX
            if is_splitter:
                sizes = splitter.sizes()
                if len(sizes) >= 2:
                    delta = self._expanded_width - sizes[0]
                    sizes[0] = self._expanded_width
                    sizes[1] = max(100, sizes[1] - delta)
                    splitter.setSizes(sizes)
        else:
            # Save current width, then fully hide
            if is_splitter:
                sizes = splitter.sizes()
                if len(sizes) >= 2 and sizes[0] > 40:
                    self._expanded_width = sizes[0]
            self.setMinimumWidth(0)
            self.setMaximumWidth(0)
            self.hide()
            if is_splitter:
                sizes = splitter.sizes()
                if len(sizes) >= 2:
                    freed = sizes[0]
                    sizes[0] = 0
                    sizes[1] = sizes[1] + freed
                    splitter.setSizes(sizes)
            # Show the floating expand button at top-left of the splitter
            expand_btn.move(2, 2)
            expand_btn.raise_()
            expand_btn.show()

    def _on_item_clicked(self, index: QModelIndex) -> None:
        clip = self._model.get_clip(index.row())
        if clip:
            self.clip_selected.emit(clip)

    def _on_context_menu(self, pos) -> None:
        """Show right-click context menu for the clip under cursor."""
        index = self._list_view.indexAt(pos)
        if not index.isValid():
            return
        clip = self._model.get_clip(index.row())
        if clip is None:
            return

        menu = QMenu(self)

        rename_action = QAction("Rename Project", self)
        rename_action.triggered.connect(lambda: self._rename_clip(clip))
        menu.addAction(rename_action)

        explorer_action = QAction("Open in Explorer", self)
        explorer_action.triggered.connect(lambda: self._open_in_explorer(clip))
        menu.addAction(explorer_action)

        menu.addSeparator()

        clear_action = QAction("Clear Outputs", self)
        clear_action.setEnabled(clip.has_outputs)
        clear_action.triggered.connect(lambda: self._clear_outputs(clip))
        menu.addAction(clear_action)

        delete_action = QAction("Delete Project", self)
        delete_action.triggered.connect(lambda: self._delete_clip(clip))
        menu.addAction(delete_action)

        menu.exec(self._list_view.viewport().mapToGlobal(pos))

    def _rename_clip(self, clip: ClipEntry) -> None:
        """Rename project display name via dialog."""
        from backend.project import set_display_name
        new_name, ok = QInputDialog.getText(
            self, "Rename Project", "Display name:", text=clip.name,
        )
        if ok and new_name.strip():
            new_name = new_name.strip()
            set_display_name(clip.root_path, new_name)
            old_name = clip.name
            clip.name = new_name
            self._model.layoutChanged.emit()
            self.clip_renamed.emit(old_name, new_name)
            logger.info(f"Renamed clip: {old_name} → {new_name}")

    def _open_in_explorer(self, clip: ClipEntry) -> None:
        """Open the clip's root folder in the system file explorer."""
        if os.path.isdir(clip.root_path):
            os.startfile(clip.root_path)

    def _clear_outputs(self, clip: ClipEntry) -> None:
        """Remove all files from Output/ subdirectories."""
        confirm = QMessageBox.question(
            self, "Clear Outputs",
            f"Remove all output files for \"{clip.name}\"?\n\n"
            "This will delete FG, Matte, Comp, and Processed frames.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        output_dir = clip.output_dir
        cleared = 0
        for subdir in ("FG", "Matte", "Comp", "Processed"):
            d = os.path.join(output_dir, subdir)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    fpath = os.path.join(d, f)
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                        cleared += 1
        # Remove manifest too
        manifest = os.path.join(output_dir, ".corridorkey_manifest.json")
        if os.path.isfile(manifest):
            os.remove(manifest)

        # Transition back to READY if was COMPLETE
        if clip.state == ClipState.COMPLETE:
            clip.state = ClipState.READY
            self._model.update_clip_state(clip.name, ClipState.READY)

        self._model.layoutChanged.emit()
        logger.info(f"Cleared {cleared} output files for {clip.name}")

    def _delete_clip(self, clip: ClipEntry) -> None:
        """Delete project from disk with two-step confirmation."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Delete Project")
        msg.setText(f"Delete \"{clip.name}\"?")
        msg.setInformativeText(
            "Remove from List: hides it (files stay on disk).\n"
            "Delete from Disk: permanently deletes the project folder."
        )

        remove_btn = msg.addButton("Remove from List", QMessageBox.AcceptRole)
        delete_btn = msg.addButton("Delete from Disk", QMessageBox.DestructiveRole)
        msg.addButton(QMessageBox.Cancel)
        msg.setDefaultButton(remove_btn)

        msg.exec()
        clicked = msg.clickedButton()

        if clicked == remove_btn:
            self._remove_clip_from_model(clip)
            self.clip_deleted.emit(clip.root_path)
        elif clicked == delete_btn:
            confirm = QMessageBox.warning(
                self, "Confirm Delete",
                f"Permanently delete this project folder?\n\n{clip.root_path}",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if confirm == QMessageBox.Yes:
                self._remove_clip_from_model(clip)
                try:
                    if os.path.isdir(clip.root_path):
                        shutil.rmtree(clip.root_path)
                except OSError as e:
                    QMessageBox.warning(
                        self, "Delete Failed",
                        f"Could not delete project:\n{e}",
                    )
                self.clip_deleted.emit(clip.root_path)

    def _remove_clip_from_model(self, clip: ClipEntry) -> None:
        """Find clip in model by name and remove it."""
        for i, c in enumerate(self._model.clips):
            if c.name == clip.name:
                self._model.remove_clip(i)
                return

    def _on_add_clicked(self) -> None:
        menu = QMenu(self)
        menu.addAction("Import Folder...", self._import_folder)
        menu.addAction("Import Video(s)...", self._import_videos)
        # Show menu below the +ADD button
        btn = self.sender()
        if btn:
            menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))
        else:
            menu.exec(self.mapToGlobal(self.rect().bottomLeft()))

    def _import_folder(self) -> None:
        """Open folder dialog for image sequence directories."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Clips Directory", "",
            QFileDialog.ShowDirsOnly,
        )
        if dir_path:
            self._clips_dir = dir_path
            self.clips_dir_changed.emit(dir_path)

    def _import_videos(self) -> None:
        """Open file dialog for individual video files."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.mxf *.webm *.m4v);;All Files (*)",
        )
        if paths:
            self.files_imported.emit(paths)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        from backend.project import is_video_file
        folders = []
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                folders.append(path)
            elif os.path.isfile(path) and is_video_file(path):
                files.append(path)
        # Prefer folder if dropped, otherwise import video files
        if folders:
            self._clips_dir = folders[0]
            self.clips_dir_changed.emit(folders[0])
        elif files:
            self.files_imported.emit(files)

    def select_first(self) -> None:
        """Select the first clip in the list."""
        if self._model.rowCount() > 0:
            self.select_by_index(0)

    def select_by_index(self, row: int) -> None:
        """Select a clip by row index."""
        if 0 <= row < self._model.rowCount():
            idx = self._model.index(row)
            self._list_view.setCurrentIndex(idx)
            self._on_item_clicked(idx)

    def get_selected_clips(self) -> list[ClipEntry]:
        """Return all currently selected clips."""
        clips = []
        for index in self._list_view.selectionModel().selectedIndexes():
            clip = self._model.get_clip(index.row())
            if clip:
                clips.append(clip)
        return clips

    # ── Folder Watch (legacy, disabled) ──

    def stop_watching(self) -> None:
        """Stop folder watching if active."""
        if self._watcher:
            self._watcher.deleteLater()
            self._watcher = None
        self._watching = False

"""Left panel — clip browser with list view, drag-drop, folder selection, and watcher.

Displays clips from ClipListModel. Supports:
- Click to select (loads in preview)
- Multi-select (ExtendedSelection) for batch operations
- [+ADD] button → QFileDialog to add clips directory
- [WATCH] toggle → QFileSystemWatcher for auto-detection
- Drag-and-drop folders
- Processing guards: watcher won't reclassify clips being processed
"""
from __future__ import annotations

import os
import time
import logging

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListView, QFileDialog, QAbstractItemView, QStyledItemDelegate, QStyle,
)
from PySide6.QtCore import Qt, Signal, QModelIndex, QTimer
from PySide6.QtGui import QColor, QImage

from ui.models.clip_model import ClipListModel
from backend import ClipEntry, ClipState, scan_clips_dir

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

        # Frame count
        if clip.input_asset:
            painter.setPen(QColor("#808070"))
            detail_font = painter.font()
            detail_font.setPointSize(9)
            detail_font.setBold(False)
            painter.setFont(detail_font)
            detail_text = f"{clip.input_asset.frame_count} frames"
            if clip.input_asset.asset_type == "video":
                detail_text += " (video)"
            detail_rect = rect.adjusted(text_left - rect.x() + 48, 18, -8, -2)
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

    def __init__(self, model: ClipListModel, parent=None):
        super().__init__(parent)
        self.setObjectName("clipPanel")
        self._model = model
        self._clips_dir: str | None = None
        self._watcher = None
        self._watching = False
        self._last_rescan_time = 0.0
        self._rescan_cooldown = 2.0  # seconds

        # Debounce timer for watcher events
        self._rescan_timer = QTimer(self)
        self._rescan_timer.setSingleShot(True)
        self._rescan_timer.setInterval(2000)
        self._rescan_timer.timeout.connect(self._do_rescan)

        self._expanded = True
        self._expanded_width = 220  # remember width for restore

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with collapse chevron
        self._header_widget = QWidget()
        header = QHBoxLayout(self._header_widget)
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
        layout.addWidget(self._header_widget)

        # Content area (hidden when collapsed)
        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # List view — ExtendedSelection for batch ops
        self._list_view = QListView()
        self._list_view.setModel(model)
        self._list_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list_view.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._list_view.setUniformItemSizes(True)
        self._list_view.setSpacing(1)
        self._list_view.clicked.connect(self._on_item_clicked)

        # Custom delegate for rendering
        delegate = ClipCardDelegate(self._list_view)
        self._list_view.setItemDelegate(delegate)

        # Fixed row height (48px to fit 40px thumbnail + padding)
        self._list_view.setStyleSheet("QListView::item { height: 48px; }")

        content_layout.addWidget(self._list_view, 1)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(10, 6, 10, 8)

        add_btn = QPushButton("+ ADD")
        add_btn.clicked.connect(self._on_add_clicked)
        btn_layout.addWidget(add_btn)

        self._watch_btn = QPushButton("WATCH")
        self._watch_btn.setCheckable(True)
        self._watch_btn.setToolTip("Auto-detect new clips in the folder")
        self._watch_btn.clicked.connect(self._toggle_watch)
        btn_layout.addWidget(self._watch_btn)

        btn_layout.addStretch()
        content_layout.addLayout(btn_layout)

        layout.addWidget(self._content, 1)

        # Accept drops
        self.setAcceptDrops(True)

        # Connect model count signal
        model.clip_count_changed.connect(
            lambda count: self._count_label.setText(str(count))
        )

    _COLLAPSED_WIDTH = 28

    def toggle_collapse(self) -> None:
        """Toggle between expanded and collapsed states.

        Collapsed: tiny nub with just the chevron at top-left.
        Expanded: full clip browser with list, buttons, etc.
        """
        from PySide6.QtWidgets import QSplitter
        splitter = self.parent()
        is_splitter = isinstance(splitter, QSplitter)

        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._title.setVisible(self._expanded)
        self._count_label.setVisible(self._expanded)

        if self._expanded:
            self._chevron.setText("\u25C0")  # ◀
            self._chevron.setToolTip("Collapse clip panel")
            self._header_widget.layout().setContentsMargins(4, 4, 4, 4)
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
            if is_splitter:
                sizes = splitter.sizes()
                if len(sizes) >= 2 and sizes[0] > 40:
                    self._expanded_width = sizes[0]
                    freed = sizes[0] - self._COLLAPSED_WIDTH
                    sizes[0] = self._COLLAPSED_WIDTH
                    sizes[1] = sizes[1] + freed
                    splitter.setSizes(sizes)
            self._chevron.setText("\u25B6")  # ▶
            self._chevron.setToolTip("Expand clip panel")
            self._header_widget.layout().setContentsMargins(4, 4, 0, 0)
            self.setMinimumWidth(self._COLLAPSED_WIDTH)
            self.setMaximumWidth(self._COLLAPSED_WIDTH)

    def _on_item_clicked(self, index: QModelIndex) -> None:
        clip = self._model.get_clip(index.row())
        if clip:
            self.clip_selected.emit(clip)

    def _on_add_clicked(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Clips Directory", "",
            QFileDialog.ShowDirsOnly,
        )
        if dir_path:
            self._clips_dir = dir_path
            self.clips_dir_changed.emit(dir_path)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                self._clips_dir = path
                self.clips_dir_changed.emit(path)
                break

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

    # ── Folder Watch ──

    def _toggle_watch(self, checked: bool) -> None:
        """Toggle folder watching on/off."""
        if checked and self._clips_dir:
            self._start_watching()
        else:
            self._stop_watching()

    def _start_watching(self) -> None:
        """Start QFileSystemWatcher on the clips directory."""
        if not self._clips_dir or not os.path.isdir(self._clips_dir):
            self._watch_btn.setChecked(False)
            return

        from PySide6.QtCore import QFileSystemWatcher

        self._watcher = QFileSystemWatcher([self._clips_dir], self)
        self._watcher.directoryChanged.connect(self._on_dir_changed)
        self._watching = True
        self._watch_btn.setStyleSheet("color: #FFF203; font-weight: 700;")
        logger.info(f"Watching: {self._clips_dir}")

    def _stop_watching(self) -> None:
        """Stop folder watching."""
        if self._watcher:
            self._watcher.deleteLater()
            self._watcher = None
        self._watching = False
        self._watch_btn.setChecked(False)
        self._watch_btn.setStyleSheet("")
        logger.info("Watch stopped")

    def _on_dir_changed(self, path: str) -> None:
        """Handle filesystem change — debounced rescan.

        Uses a 2-second debounce timer to avoid event storms from
        worker writing Output/AlphaHint files (Codex finding).
        """
        # Debounce: restart the timer on each event
        self._rescan_timer.start()

    def _do_rescan(self) -> None:
        """Perform the actual rescan after debounce cooldown.

        Skips reclassification of clips that are currently being processed
        (Codex: watcher non-authoritative during active processing).
        """
        if not self._clips_dir:
            return

        now = time.time()
        if now - self._last_rescan_time < self._rescan_cooldown:
            return
        self._last_rescan_time = now

        try:
            new_clips = scan_clips_dir(self._clips_dir)
            existing_names = {c.name for c in self._model.clips}
            processing_names = {c.name for c in self._model.clips if c.is_processing}

            for clip in new_clips:
                if clip.name not in existing_names:
                    # Genuinely new clip — add it
                    self._model.add_clip(clip)
                    logger.info(f"Watcher: new clip detected: {clip.name}")
                elif clip.name not in processing_names:
                    # Existing clip, not processing — safe to update state
                    self._model.update_clip_state(clip.name, clip.state)

        except Exception as e:
            logger.warning(f"Watcher rescan failed: {e}")

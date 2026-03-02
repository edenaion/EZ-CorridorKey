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
import shutil
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QSplitter,
    QToolTip, QPushButton, QMenu, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, Signal, QRect, QSize, QEvent
from PySide6.QtGui import QPainter, QColor, QImage, QMouseEvent, QAction

from backend import ClipEntry, ClipState
from backend.project import VIDEO_FILE_FILTER
from ui.models.clip_model import ClipListModel

logger = logging.getLogger(__name__)

# State → color mapping (matches brand palette)
_STATE_COLORS: dict[ClipState, str] = {
    ClipState.EXTRACTING: "#FF8C00",
    ClipState.RAW: "#808070",
    ClipState.MASKED: "#009ADA",
    ClipState.READY: "#FFF203",
    ClipState.COMPLETE: "#22C55E",
    ClipState.ERROR: "#D10000",
}


class ThumbnailCanvas(QWidget):
    """Custom-painted horizontal strip of clip thumbnail cards.

    Each card is CARD_WIDTH wide and shows: thumbnail, clip name, state badge,
    and frame count. Mouse clicks emit card_clicked with the ClipEntry.
    """

    card_clicked = Signal(object)  # ClipEntry (single left-click)
    multi_select_toggled = Signal(object)  # ClipEntry (Ctrl+click toggle)
    shift_select_requested = Signal(object)  # ClipEntry (Shift+click range)
    context_menu_requested = Signal(object)  # ClipEntry (right-click)

    CARD_WIDTH = 130
    CARD_SPACING = 4
    CARD_PADDING = 6
    THUMB_W = 110
    THUMB_H = 62

    def __init__(self, parent=None, show_manifest_tooltip: bool = False):
        super().__init__(parent)
        self._clips: list[ClipEntry] = []
        self._model: ClipListModel | None = None
        self._show_manifest_tooltip = show_manifest_tooltip
        self._selected_names: set[str] = set()
        self._hovered_name: str | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(100)

    def set_clips(self, clips: list[ClipEntry], model: ClipListModel) -> None:
        """Update the displayed clips and trigger repaint."""
        self._clips = list(clips)
        self._model = model
        total_w = max(1, len(clips) * (self.CARD_WIDTH + self.CARD_SPACING))
        self.setFixedWidth(total_w)
        self.update()

    def set_selected(self, name: str | None) -> None:
        """Set single-selected clip (clears multi-select, draws highlight border)."""
        new_set = {name} if name else set()
        if self._selected_names != new_set:
            self._selected_names = new_set
            self.update()

    def set_multi_selected(self, names: set[str]) -> None:
        """Set the multi-selected clip names."""
        if self._selected_names != names:
            self._selected_names = set(names)
            self.update()

    def paintEvent(self, event) -> None:
        if not self._clips:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)

        for i, clip in enumerate(self._clips):
            x = i * (self.CARD_WIDTH + self.CARD_SPACING)
            card_rect = QRect(x, 0, self.CARD_WIDTH, self.height())

            # Skip cards not in the visible region
            if not card_rect.intersects(event.rect()):
                continue

            self._paint_card(p, card_rect, clip)

        p.end()

    def _paint_card(self, p: QPainter, rect: QRect, clip: ClipEntry) -> None:
        pad = self.CARD_PADDING
        is_selected = clip.name in self._selected_names
        is_hovered = clip.name == self._hovered_name and not is_selected

        # Card background
        if is_selected:
            bg = QColor("#252413")
        elif is_hovered:
            bg = QColor("#1E1D0A")
        else:
            bg = QColor("#1A1900")
        p.fillRect(rect, bg)

        # Border — yellow for selected, subtle yellow for hover, default otherwise
        if is_selected:
            p.setPen(QColor("#FFF203"))
            p.drawRect(rect.adjusted(0, 0, -1, -1))
            p.drawRect(rect.adjusted(1, 1, -2, -2))  # 2px border
        elif is_hovered:
            p.setPen(QColor(255, 242, 3, 100))  # subtle yellow glow
            p.drawRect(rect.adjusted(0, 0, -1, -1))
        else:
            p.setPen(QColor("#2A2910"))
            p.drawRect(rect.adjusted(0, 0, -1, -1))

        # Thumbnail
        thumb_rect = QRect(
            rect.x() + (self.CARD_WIDTH - self.THUMB_W) // 2,
            rect.y() + pad,
            self.THUMB_W,
            self.THUMB_H,
        )
        thumb = self._model.get_thumbnail(clip.name) if self._model else None
        if isinstance(thumb, QImage) and not thumb.isNull():
            scaled = thumb.scaled(
                self.THUMB_W, self.THUMB_H,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            dx = thumb_rect.x() + (self.THUMB_W - scaled.width()) // 2
            dy = thumb_rect.y() + (self.THUMB_H - scaled.height()) // 2
            p.drawImage(dx, dy, scaled)
        else:
            p.fillRect(thumb_rect, QColor("#0A0A00"))
            p.setPen(QColor("#3A3A30"))
            p.drawRect(thumb_rect.adjusted(0, 0, -1, -1))

        # State badge (top-right over thumbnail, with background pill)
        badge_color = QColor(_STATE_COLORS.get(clip.state, "#808070"))
        font = p.font()
        font.setPointSize(8)
        font.setBold(True)
        p.setFont(font)
        badge_text = clip.state.value
        metrics = p.fontMetrics()
        text_w = metrics.horizontalAdvance(badge_text)
        bg_rect = QRect(
            rect.x() + self.CARD_WIDTH - pad - text_w - 6,
            rect.y() + pad,
            text_w + 6, 14,
        )
        p.fillRect(bg_rect, QColor(0, 0, 0, 128))
        p.setPen(badge_color)
        p.drawText(bg_rect, Qt.AlignCenter, badge_text)

        # Clip name (below thumbnail, with background)
        text_y = rect.y() + pad + self.THUMB_H + 4
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)
        name_rect = QRect(rect.x() + pad, text_y, self.CARD_WIDTH - pad * 2, 16)
        metrics = p.fontMetrics()
        elided = metrics.elidedText(clip.name, Qt.ElideRight, name_rect.width())
        p.fillRect(name_rect, QColor(0, 0, 0, 128))
        p.setPen(QColor("#E0E0E0"))
        p.drawText(name_rect, Qt.AlignLeft | Qt.AlignVCenter, elided)

        # Frame count (below name, with background)
        if clip.input_asset:
            font.setPointSize(8)
            font.setBold(False)
            p.setFont(font)
            info_rect = QRect(rect.x() + pad, text_y + 14, self.CARD_WIDTH - pad * 2, 14)
            info_text = f"{clip.input_asset.frame_count} frames"
            if clip.input_asset.asset_type == "video":
                info_text += " (video)"
            p.fillRect(info_rect, QColor(0, 0, 0, 128))
            p.setPen(QColor("#808070"))
            p.drawText(info_rect, Qt.AlignLeft | Qt.AlignVCenter, info_text)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        clip = self._card_at(event.position().x())
        name = clip.name if clip else None
        if name != self._hovered_name:
            self._hovered_name = name
            self.update()

    def leaveEvent(self, event) -> None:
        if self._hovered_name is not None:
            self._hovered_name = None
            self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._clips:
            clip = self._card_at(event.position().x())
            if clip:
                from ui.sounds.audio_manager import UIAudio
                UIAudio.click()
                if event.modifiers() & Qt.ShiftModifier:
                    # Shift+click: range select from anchor to this clip
                    self.shift_select_requested.emit(clip)
                elif event.modifiers() & Qt.ControlModifier:
                    # Ctrl+click: toggle in/out of multi-selection
                    self.multi_select_toggled.emit(clip)
                else:
                    # Plain click: single-select
                    self.card_clicked.emit(clip)
        elif event.button() == Qt.RightButton and self._clips:
            clip = self._card_at(event.position().x())
            if clip:
                self.context_menu_requested.emit(clip)

    def sizeHint(self) -> QSize:
        w = max(1, len(self._clips) * (self.CARD_WIDTH + self.CARD_SPACING))
        return QSize(w, 100)

    def _card_at(self, x: float) -> ClipEntry | None:
        """Return the ClipEntry under the given x position, or None."""
        if not self._clips:
            return None
        idx = int(x // (self.CARD_WIDTH + self.CARD_SPACING))
        if 0 <= idx < len(self._clips):
            return self._clips[idx]
        return None

    def event(self, ev: QEvent) -> bool:
        if ev.type() == QEvent.ToolTip and self._show_manifest_tooltip:
            clip = self._card_at(ev.position().x())
            tip = _format_manifest_tooltip(clip) if clip else ""
            if tip:
                QToolTip.showText(ev.globalPosition().toPoint(), tip, self)
            else:
                QToolTip.hideText()
            return True
        return super().event(ev)


def _format_manifest_tooltip(clip: ClipEntry) -> str:
    """Build a tooltip string from the clip's .corridorkey_manifest.json."""
    manifest = clip._read_manifest()
    if manifest is None:
        return ""

    lines: list[str] = [f"<b>{clip.name}</b> — Export Settings"]

    # Outputs + formats
    enabled = manifest.get("enabled_outputs", [])
    formats = manifest.get("formats", {})
    if enabled:
        out_parts = []
        for name in enabled:
            fmt = formats.get(name, "?").upper()
            out_parts.append(f"{name.upper()} ({fmt})")
        lines.append(f"<b>Outputs:</b> {', '.join(out_parts)}")

    # Params
    params = manifest.get("params", {})
    if params:
        cs = "Linear" if params.get("input_is_linear") else "sRGB"
        lines.append(f"<b>Color Space:</b> {cs}")
        ds = params.get("despill_strength", 1.0)
        lines.append(f"<b>Despill:</b> {ds:.0%}")
        rs = params.get("refiner_scale", 1.0)
        lines.append(f"<b>Refiner:</b> {rs:.0%}")
        if params.get("auto_despeckle"):
            sz = params.get("despeckle_size", 400)
            lines.append(f"<b>Despeckle:</b> On (size {sz})")
        else:
            lines.append(f"<b>Despeckle:</b> Off")

    return "<br>".join(lines)


class IOTrayPanel(QWidget):
    """Bottom panel with Input and Exports thumbnail strips.

    Input section shows all loaded clips with + ADD button.
    Exports section shows only COMPLETE clips.
    Clicking a card emits clip_clicked. Right-click shows context menu.
    Supports drag-and-drop of video files and folders.
    """

    clip_clicked = Signal(object)   # ClipEntry
    selection_changed = Signal(list)  # list[ClipEntry] — multi-select changed
    clips_dir_changed = Signal(str)  # folder path (import folder)
    files_imported = Signal(list)    # list of video file paths
    extract_requested = Signal(list) # list[ClipEntry] — re-run extraction

    def __init__(self, model: ClipListModel, parent=None):
        super().__init__(parent)
        self.setObjectName("ioTrayPanel")
        self._model = model
        self._select_anchor: str | None = None  # last single-clicked clip name for Shift+click range
        self.setMinimumHeight(80)
        self.setMaximumHeight(600)
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Content: two strips in a splitter (synced with dual viewer divider)
        self._tray_splitter = QSplitter(Qt.Horizontal)
        self._tray_splitter.setHandleWidth(1)
        self._tray_splitter.setStyleSheet(
            "QSplitter::handle { background-color: #2A2910; }"
        )

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

        self._add_btn = QPushButton("+ ADD")
        self._add_btn.setObjectName("trayAddBtn")
        self._add_btn.setToolTip("Import clips — choose a folder or video file(s)")
        self._add_btn.clicked.connect(self._on_add_clicked)
        input_header_row.addWidget(self._add_btn)

        input_section.addLayout(input_header_row)

        self._input_scroll = QScrollArea()
        self._input_scroll.setObjectName("trayScroll")
        self._input_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._input_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._input_scroll.setWidgetResizable(False)

        self._input_canvas = ThumbnailCanvas()
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
        self._export_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._export_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._export_scroll.setWidgetResizable(False)

        self._export_canvas = ThumbnailCanvas(show_manifest_tooltip=True)
        self._export_canvas.card_clicked.connect(self.clip_clicked.emit)
        self._export_canvas.context_menu_requested.connect(self._on_export_context_menu)
        self._export_scroll.setWidget(self._export_canvas)

        export_section.addWidget(self._export_scroll, 1)
        self._tray_splitter.addWidget(export_widget)

        # Equal split by default (synced from main window)
        self._tray_splitter.setSizes([500, 500])
        self._tray_splitter.setStretchFactor(0, 1)
        self._tray_splitter.setStretchFactor(1, 1)

        layout.addWidget(self._tray_splitter)

        # Connect to model signals for auto-rebuild
        self._model.modelReset.connect(self._rebuild)
        self._model.dataChanged.connect(self._on_data_changed)
        self._model.clip_count_changed.connect(lambda _: self._rebuild())

    # ── + ADD button ──

    def _on_add_clicked(self) -> None:
        """Show import menu below the +ADD button."""
        menu = QMenu(self)
        menu.addAction("Import Folder...", self._import_folder)
        menu.addAction("Import Video(s)...", self._import_videos)
        menu.exec(self._add_btn.mapToGlobal(self._add_btn.rect().bottomLeft()))

    def _import_folder(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Clips Directory", "",
            QFileDialog.ShowDirsOnly,
        )
        if dir_path:
            self.clips_dir_changed.emit(dir_path)

    def _import_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "",
            VIDEO_FILE_FILTER,
        )
        if paths:
            self.files_imported.emit(paths)

    # ── Drag-and-drop ──

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
        if folders:
            self.clips_dir_changed.emit(folders[0])
        elif files:
            self.files_imported.emit(files)

    # ── Single / Multi-select management ──

    def _on_single_click(self, clip: ClipEntry) -> None:
        """Plain left-click — single-select, clear multi-select."""
        self._select_anchor = clip.name
        self._input_canvas.set_selected(clip.name)
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

    # ── Context menu (INPUT cards) ──

    def _on_context_menu(self, clip: ClipEntry) -> None:
        """Show right-click context menu for a clip card.

        If the right-clicked clip is not in the current selection,
        single-select it first (standard behaviour).
        """
        if clip.name not in self._input_canvas._selected_names:
            self._input_canvas.set_selected(clip.name)
            self.clip_clicked.emit(clip)
            self.selection_changed.emit([clip])

        selected = self.get_selected_clips()
        n = len(selected)
        multi = n > 1

        menu = QMenu(self)

        # Run Extraction — for clips that have a video source and need frames
        from backend.clip_state import ClipState
        needs_extract = [
            c for c in selected
            if c.state == ClipState.EXTRACTING
            or (c.input_asset and c.input_asset.asset_type == "video")
        ]
        if needs_extract:
            label_ext = (f"Run Extraction ({len(needs_extract)} clips)"
                         if len(needs_extract) > 1 else "Run Extraction")
            extract_action = QAction(label_ext, self)
            extract_action.triggered.connect(
                lambda: self.extract_requested.emit(needs_extract))
            menu.addAction(extract_action)
            menu.addSeparator()

        # Open in Explorer — single only
        explorer_action = QAction("Open in Explorer", self)
        explorer_action.setEnabled(not multi)
        explorer_action.triggered.connect(lambda: self._open_in_explorer(clip))
        menu.addAction(explorer_action)

        menu.addSeparator()

        # Clear Outputs — only show when there are outputs to clear
        any_outputs = any(c.has_outputs for c in selected)
        if any_outputs:
            label_clear = f"Clear Outputs ({n} clips)" if multi else "Clear Outputs"
            clear_action = QAction(label_clear, self)
            clear_action.triggered.connect(lambda: self._clear_outputs_batch(selected))
            menu.addAction(clear_action)

        # Remove...
        label_remove = f"Remove ({n} clips)..." if multi else "Remove..."
        remove_action = QAction(label_remove, self)
        remove_action.triggered.connect(lambda: self._remove_dialog(selected))
        menu.addAction(remove_action)

        from PySide6.QtGui import QCursor
        menu.exec(QCursor.pos())

    def _on_export_context_menu(self, clip: ClipEntry) -> None:
        """Show right-click context menu for an export card."""
        menu = QMenu(self)

        # Open containing folder (Output directory)
        output_dir = os.path.join(clip.root_path, "Output")
        if not os.path.isdir(output_dir):
            output_dir = clip.root_path

        open_action = QAction("Open Containing Folder", self)
        open_action.triggered.connect(lambda: os.startfile(output_dir))
        menu.addAction(open_action)

        from PySide6.QtGui import QCursor
        menu.exec(QCursor.pos())

    def _open_in_explorer(self, clip: ClipEntry) -> None:
        if os.path.isdir(clip.root_path):
            os.startfile(clip.root_path)

    def _clear_outputs_batch(self, clips: list[ClipEntry]) -> None:
        """Clear output files for one or more clips."""
        names = ", ".join(c.name for c in clips[:3])
        if len(clips) > 3:
            names += f" (+{len(clips) - 3} more)"
        confirm = QMessageBox.question(
            self, "Clear Outputs",
            f"Remove all output files for {len(clips)} clip(s)?\n{names}\n\n"
            "This will delete FG, Matte, Comp, and Processed frames.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        total_cleared = 0
        for clip in clips:
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
            manifest = os.path.join(output_dir, ".corridorkey_manifest.json")
            if os.path.isfile(manifest):
                os.remove(manifest)

            if clip.state == ClipState.COMPLETE:
                clip.state = ClipState.READY
                self._model.update_clip_state(clip.name, ClipState.READY)
            total_cleared += cleared

        self._model.layoutChanged.emit()
        logger.info(f"Cleared {total_cleared} output files across {len(clips)} clip(s)")

    def _remove_dialog(self, clips: list[ClipEntry]) -> None:
        """Show remove confirmation dialog with Remove from List / Delete from Disk options."""
        n = len(clips)
        title = f"Remove {n} clip{'s' if n > 1 else ''}?"

        paths_text = "\n".join(c.root_path for c in clips[:5])
        if n > 5:
            paths_text += f"\n... and {n - 5} more"

        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setIcon(QMessageBox.Warning)
        msg.setText(
            f"How would you like to remove {n} clip{'s' if n > 1 else ''}?"
        )
        msg.setInformativeText(paths_text)

        btn_list = msg.addButton("Remove from List", QMessageBox.AcceptRole)
        btn_disk = msg.addButton("Delete from Disk", QMessageBox.DestructiveRole)
        msg.addButton(QMessageBox.Cancel)

        msg.exec()
        clicked = msg.clickedButton()

        if clicked == btn_list:
            self._remove_clips_from_list(clips)
        elif clicked == btn_disk:
            self._delete_clips_from_disk(clips)

    def _remove_clips_from_list(self, clips: list[ClipEntry]) -> None:
        """Remove clips from the list (files stay on disk)."""
        names = {c.name for c in clips}
        # Remove in reverse index order to avoid index shifting
        indices = [i for i, c in enumerate(self._model.clips) if c.name in names]
        for i in sorted(indices, reverse=True):
            self._model.remove_clip(i)
        logger.info(f"Removed {len(clips)} clip(s) from list")

    def _delete_clips_from_disk(self, clips: list[ClipEntry]) -> None:
        """Delete clip project folders from disk."""
        for clip in clips:
            if os.path.isdir(clip.root_path):
                shutil.rmtree(clip.root_path, ignore_errors=True)
                logger.info(f"Deleted from disk: {clip.root_path}")
        # Then remove from model
        self._remove_clips_from_list(clips)

    # ── Selection highlight ──

    def set_selected(self, name: str | None) -> None:
        """Set single-selected clip (clears multi-select, highlights in INPUT strip)."""
        self._input_canvas.set_selected(name)

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
        """Handle model data changes — rebuild to catch state transitions."""
        self._rebuild()

    def refresh(self) -> None:
        """Force rebuild (called after worker completes a clip)."""
        self._rebuild()

    def sync_divider(self, left_px: int) -> None:
        """Set the IO tray divider position in pixels from the left edge."""
        total = self._tray_splitter.width()
        right = max(1, total - left_px)
        self._tray_splitter.setSizes([left_px, right])

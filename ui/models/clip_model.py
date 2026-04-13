"""QAbstractListModel wrapping a list of ClipEntry objects.

Provides the data layer for the clip browser list view. The model
emits standard Qt signals when clips are added/removed/changed, so
the view updates automatically.
"""

from __future__ import annotations

from collections import OrderedDict

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor, QImage

from backend import ClipEntry, ClipState

# State → display color mapping (matches brand palette)
_STATE_COLORS: dict[ClipState, str] = {
    ClipState.EXTRACTING: "#FF8C00",
    ClipState.RAW: "#808070",
    ClipState.MASKED: "#009ADA",
    ClipState.READY: "#FFF203",
    ClipState.COMPLETE: "#22C55E",
    ClipState.ERROR: "#D10000",
}


class ClipListModel(QAbstractListModel):
    """List model for clip entries in the browser panel.

    Custom roles:
        ClipEntryRole (Qt.UserRole) — returns the ClipEntry object
        StateColorRole (Qt.UserRole+1) — returns QColor for the state badge
    """

    ClipEntryRole = Qt.UserRole
    StateColorRole = Qt.UserRole + 1
    ThumbnailRole = Qt.UserRole + 2
    ExportThumbnailRole = Qt.UserRole + 3

    clip_count_changed = Signal(int)  # emitted when clip count changes

    # Max cached thumbnails (LRU, Codex: avoid unbounded memory)
    _THUMB_CACHE_MAX = 100

    def __init__(self, parent=None):
        super().__init__(parent)
        self._clips: list[ClipEntry] = []
        self._input_thumbnails: OrderedDict[str, QImage] = OrderedDict()
        self._export_thumbnails: OrderedDict[str, QImage] = OrderedDict()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._clips)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._clips):
            return None

        clip = self._clips[index.row()]

        if role == Qt.DisplayRole:
            return clip.name
        elif role == self.ClipEntryRole:
            return clip
        elif role == self.StateColorRole:
            return QColor(_STATE_COLORS.get(clip.state, "#808070"))
        elif role == self.ThumbnailRole:
            return self._input_thumbnails.get(clip.name)
        elif role == self.ExportThumbnailRole:
            return self._export_thumbnails.get(clip.name)
        elif role == Qt.ToolTipRole:
            lines = [f"State: {clip.state.value}"]
            if clip.input_asset:
                lines.append(
                    f"Input: {clip.input_asset.frame_count} frames ({clip.input_asset.asset_type})"
                )
            if clip.alpha_asset:
                lines.append(f"Alpha: {clip.alpha_asset.frame_count} frames")
            if clip.warnings:
                lines.append(f"Warnings: {len(clip.warnings)}")
            if clip.error_message:
                lines.append(f"Error: {clip.error_message}")
            return "\n".join(lines)

        return None

    def set_clips(self, clips: list[ClipEntry]) -> None:
        """Replace all clips. Called after scan_clips()."""
        self.beginResetModel()
        self._clips = list(clips)
        self._input_thumbnails.clear()  # Prevent stale thumbnails from previous project
        self._export_thumbnails.clear()
        self.endResetModel()
        self.clip_count_changed.emit(len(self._clips))

    def remove_clip(self, row: int) -> None:
        """Remove clip at row index."""
        if 0 <= row < len(self._clips):
            self.beginRemoveRows(QModelIndex(), row, row)
            self._clips.pop(row)
            self.endRemoveRows()
            self.clip_count_changed.emit(len(self._clips))

    def get_clip(self, row: int) -> ClipEntry | None:
        """Get clip by row index."""
        if 0 <= row < len(self._clips):
            return self._clips[row]
        return None

    def update_clip_state(self, clip_name: str, new_state: ClipState) -> None:
        """Update a clip's state by name and notify the view."""
        for i, clip in enumerate(self._clips):
            if clip.name == clip_name:
                clip.state = new_state
                idx = self.index(i)
                self.dataChanged.emit(idx, idx, [Qt.DisplayRole, self.StateColorRole])
                return

    def clips_by_state(self, state: ClipState) -> list[ClipEntry]:
        """Filter clips by state."""
        return [c for c in self._clips if c.state == state]

    def set_thumbnail(self, clip_name: str, kind: str, qimage: QImage) -> None:
        """Store an input/export thumbnail QImage for a clip."""
        cache = self._export_thumbnails if kind == "export" else self._input_thumbnails
        role = self.ExportThumbnailRole if kind == "export" else self.ThumbnailRole

        cache[clip_name] = qimage
        cache.move_to_end(clip_name)
        # Evict oldest if over limit
        while len(cache) > self._THUMB_CACHE_MAX:
            cache.popitem(last=False)
        # Notify view to repaint
        for i, clip in enumerate(self._clips):
            if clip.name == clip_name:
                idx = self.index(i)
                self.dataChanged.emit(idx, idx, [role])
                return

    def get_thumbnail(self, clip_name: str, kind: str = "input") -> QImage | None:
        """Get cached input/export thumbnail QImage for a clip, or None."""
        cache = self._export_thumbnails if kind == "export" else self._input_thumbnails
        return cache.get(clip_name)

    @property
    def clips(self) -> list[ClipEntry]:
        return list(self._clips)

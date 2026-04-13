"""Background thumbnail generator using QThreadPool.

Generates thumbnails lazily in worker threads, emits QImage to main
thread for storage in the model. Uses QStandardPaths for cache location
(Codex: clips dir may be read-only/network). Checks mtime for invalidation.

Codex finding: don't create QPixmap off main thread. Generate QImage
in worker, promote to QPixmap on main thread only if needed.
"""

from __future__ import annotations

import os
import hashlib
import logging

from PySide6.QtCore import QObject, QRunnable, Signal, QThreadPool, QStandardPaths, Qt
from PySide6.QtGui import QImage

from ui.preview.display_transform import decode_frame, decode_video_frame
from ui.preview.frame_index import ViewMode

logger = logging.getLogger(__name__)

# Thumbnail dimensions
THUMB_WIDTH = 60
THUMB_HEIGHT = 40
_THUMB_CACHE_VERSION = "v4"


def _cache_dir() -> str:
    """Get or create the thumbnail cache directory."""
    base = QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
    cache = os.path.join(base, "corridorkey", "thumbnails")
    os.makedirs(cache, exist_ok=True)
    return cache


def _cache_path(
    clip_root: str,
    *,
    kind: str,
    input_exr_is_linear: bool = False,
    export_mode: ViewMode = ViewMode.COMP,
) -> str:
    """Generate cache file path for a clip thumbnail."""
    h = hashlib.md5(clip_root.encode()).hexdigest()[:12]
    export_tag = export_mode.value.lower() if kind == "export" else "input"
    linear_tag = "lin" if input_exr_is_linear else "srgb"
    return os.path.join(
        _cache_dir(),
        f"{h}_{kind}_{linear_tag}_{export_tag}_{_THUMB_CACHE_VERSION}.jpg",
    )


class _ThumbSignals(QObject):
    finished = Signal(str, str, str, object)  # clip_name, kind, request_id, QImage|None


class _ThumbTask(QRunnable):
    """Generate a thumbnail for a single clip."""

    def __init__(
        self,
        clip_name: str,
        clip_root: str,
        *,
        kind: str,
        input_path: str | None = None,
        asset_type: str = "sequence",
        input_exr_is_linear: bool = False,
        export_mode: ViewMode = ViewMode.COMP,
        request_id: str = "",
    ):
        super().__init__()
        self.signals = _ThumbSignals()
        self._clip_name = clip_name
        self._clip_root = clip_root
        self._input_path = input_path
        self._asset_type = asset_type
        self._kind = kind
        self._input_exr_is_linear = input_exr_is_linear
        self._export_mode = export_mode
        self._request_id = request_id
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            qimg = self._generate()
            self.signals.finished.emit(self._clip_name, self._kind, self._request_id, qimg)
        except Exception as e:
            logger.debug(f"Thumbnail generation failed for {self._clip_name}: {e}")
            self.signals.finished.emit(self._clip_name, self._kind, self._request_id, None)

    def _generate(self) -> QImage | None:
        cache = _cache_path(
            self._clip_root,
            kind=self._kind,
            input_exr_is_linear=self._input_exr_is_linear,
            export_mode=self._export_mode,
        )

        # Check cache validity (mtime-based)
        if os.path.isfile(cache):
            cache_mtime = os.path.getmtime(cache)
            source_mtime = self._source_mtime()
            if source_mtime and cache_mtime >= source_mtime:
                cached = QImage(cache)
                if not cached.isNull():
                    return cached

        # Generate fresh
        frame = self._read_first_frame()
        if frame is None or frame.isNull():
            return None

        small = frame.scaled(
            THUMB_WIDTH,
            THUMB_HEIGHT,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        # Save to cache
        try:
            small.save(cache, "JPG")
        except Exception:
            pass  # Cache write is best-effort

        return small

    def _read_first_frame(self) -> QImage | None:
        """Read the representative frame using the same display transform as the viewer."""
        if self._kind == "export":
            return self._read_first_export_frame()
        return self._read_first_input_frame()

    def _read_first_input_frame(self) -> QImage | None:
        """Read the first input frame using the same display transform as the viewer."""
        if self._asset_type == "video":
            if self._input_path is None:
                return None
            return decode_video_frame(
                self._input_path,
                0,
                input_exr_is_linear=self._input_exr_is_linear,
            )
        else:
            if self._input_path is None or not os.path.isdir(self._input_path):
                return None
            from backend.natural_sort import natsorted

            files = natsorted(
                [
                    f
                    for f in os.listdir(self._input_path)
                    if f.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp")
                    )
                ]
            )
            if not files:
                return None
            path = os.path.join(self._input_path, files[0])
            return decode_frame(
                path,
                ViewMode.INPUT,
                input_exr_is_linear=self._input_exr_is_linear,
            )

    def _read_first_export_frame(self) -> QImage | None:
        """Read the first output frame using the same display transform as the viewer."""
        from backend.natural_sort import natsorted

        preferred_modes = [self._export_mode]
        for fallback in (ViewMode.COMP, ViewMode.PROCESSED, ViewMode.FG, ViewMode.MATTE):
            if fallback not in preferred_modes:
                preferred_modes.append(fallback)

        mode_dirs = {
            ViewMode.COMP: os.path.join(self._clip_root, "Output", "Comp"),
            ViewMode.PROCESSED: os.path.join(self._clip_root, "Output", "Processed"),
            ViewMode.FG: os.path.join(self._clip_root, "Output", "FG"),
            ViewMode.MATTE: os.path.join(self._clip_root, "Output", "Matte"),
        }

        for mode in preferred_modes:
            dir_path = mode_dirs.get(mode)
            if dir_path is None or not os.path.isdir(dir_path):
                continue
            files = natsorted(
                [
                    f
                    for f in os.listdir(dir_path)
                    if f.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp")
                    )
                ]
            )
            if not files:
                continue
            path = os.path.join(dir_path, files[0])
            return decode_frame(path, mode)
        return None

    def _source_mtime(self) -> float | None:
        """Get modification time of the source for cache invalidation."""
        try:
            if self._kind == "export":
                output_dir = os.path.join(self._clip_root, "Output")
                if not os.path.isdir(output_dir):
                    return None
                mtimes = [os.path.getmtime(output_dir)]
                for subdir in ("Comp", "Processed", "FG", "Matte"):
                    d = os.path.join(output_dir, subdir)
                    if os.path.isdir(d):
                        mtimes.append(os.path.getmtime(d))
                return max(mtimes)
            if self._asset_type == "video":
                if self._input_path is None:
                    return None
                return os.path.getmtime(self._input_path)
            elif self._input_path is not None and os.path.isdir(self._input_path):
                return os.path.getmtime(self._input_path)
        except Exception:
            pass
        return None


class ThumbnailGenerator(QObject):
    """Manages background thumbnail generation for clips.

    Usage:
        gen = ThumbnailGenerator()
        gen.thumbnail_ready.connect(model.set_thumbnail)
        gen.generate(clip)
    """

    thumbnail_ready = Signal(str, str, object)  # clip_name, kind, QImage

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._pending: set[tuple[str, str, str]] = set()
        self._latest_request: dict[tuple[str, str], str] = {}

    def generate(
        self,
        clip_name: str,
        clip_root: str,
        *,
        kind: str = "input",
        input_path: str | None = None,
        asset_type: str = "sequence",
        input_exr_is_linear: bool = False,
        export_mode: ViewMode = ViewMode.COMP,
    ) -> None:
        """Queue thumbnail generation for a clip."""
        request_id = f"{int(input_exr_is_linear)}:{export_mode.value}"
        pending_key = (clip_name, kind, request_id)
        if pending_key in self._pending:
            return
        self._latest_request[(clip_name, kind)] = request_id
        self._pending.add(pending_key)

        task = _ThumbTask(
            clip_name,
            clip_root,
            kind=kind,
            input_path=input_path,
            asset_type=asset_type,
            input_exr_is_linear=input_exr_is_linear,
            export_mode=export_mode,
            request_id=request_id,
        )
        task.signals.finished.connect(self._on_finished)
        self._pool.start(task)

    def _on_finished(self, clip_name: str, kind: str, request_id: str, qimage: object) -> None:
        self._pending.discard((clip_name, kind, request_id))
        if self._latest_request.get((clip_name, kind)) != request_id:
            return
        if qimage is not None:
            self.thumbnail_ready.emit(clip_name, kind, qimage)

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

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal, QThreadPool, QStandardPaths
from PySide6.QtGui import QImage

logger = logging.getLogger(__name__)

# Thumbnail dimensions
THUMB_WIDTH = 60
THUMB_HEIGHT = 40


def _cache_dir() -> str:
    """Get or create the thumbnail cache directory."""
    base = QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
    cache = os.path.join(base, "corridorkey", "thumbnails")
    os.makedirs(cache, exist_ok=True)
    return cache


def _cache_path(clip_root: str) -> str:
    """Generate cache file path for a clip, based on root path hash."""
    h = hashlib.md5(clip_root.encode()).hexdigest()[:12]
    return os.path.join(_cache_dir(), f"{h}.jpg")


class _ThumbSignals(QObject):
    finished = Signal(str, object)  # clip_name, QImage|None


class _ThumbTask(QRunnable):
    """Generate a thumbnail for a single clip."""

    def __init__(self, clip_name: str, clip_root: str, input_path: str,
                 asset_type: str):
        super().__init__()
        self.signals = _ThumbSignals()
        self._clip_name = clip_name
        self._clip_root = clip_root
        self._input_path = input_path
        self._asset_type = asset_type
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            qimg = self._generate()
            self.signals.finished.emit(self._clip_name, qimg)
        except Exception as e:
            logger.debug(f"Thumbnail generation failed for {self._clip_name}: {e}")
            self.signals.finished.emit(self._clip_name, None)

    def _generate(self) -> QImage | None:
        cache = _cache_path(self._clip_root)

        # Check cache validity (mtime-based)
        if os.path.isfile(cache):
            cache_mtime = os.path.getmtime(cache)
            source_mtime = self._source_mtime()
            if source_mtime and cache_mtime >= source_mtime:
                # Load from cache
                img = cv2.imread(cache)
                if img is not None:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return self._to_qimage(rgb)

        # Generate fresh
        frame = self._read_first_frame()
        if frame is None:
            return None

        # Resize to thumbnail
        h, w = frame.shape[:2]
        scale = min(THUMB_WIDTH / w, THUMB_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Save to cache (BGR for cv2)
        try:
            cv2.imwrite(cache, small if small.shape[2] == 3 else cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        except Exception:
            pass  # Cache write is best-effort

        # Convert to RGB for QImage
        if small.shape[2] == 3:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        else:
            rgb = small
        return self._to_qimage(rgb)

    def _read_first_frame(self) -> np.ndarray | None:
        """Read the first frame from input."""
        if self._asset_type == "video":
            try:
                cap = cv2.VideoCapture(self._input_path)
                if not cap.isOpened():
                    return None
                ret, frame = cap.read()
                cap.release()
                return frame if ret else None
            except Exception:
                return None
        else:
            # Image sequence — get first file
            if not os.path.isdir(self._input_path):
                return None
            from ui.preview.natural_sort import natsorted
            files = natsorted([f for f in os.listdir(self._input_path)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.exr', '.tif', '.tiff', '.bmp'))])
            if not files:
                return None
            path = os.path.join(self._input_path, files[0])

            if path.lower().endswith('.exr'):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    return None
                # Handle float EXR
                if img.dtype != np.uint8:
                    # Strip alpha if 4ch
                    if img.ndim == 3 and img.shape[2] == 4:
                        img = img[:, :, :3]
                    elif img.ndim == 2:
                        img = np.stack([img, img, img], axis=2)
                    # Clamp and convert
                    clamped = np.clip(img.astype(np.float32), 0.0, 1.0)
                    img = (np.power(clamped, 1.0 / 2.2) * 255).astype(np.uint8)
                return img
            else:
                return cv2.imread(path)

    def _source_mtime(self) -> float | None:
        """Get modification time of the source for cache invalidation."""
        try:
            if self._asset_type == "video":
                return os.path.getmtime(self._input_path)
            elif os.path.isdir(self._input_path):
                return os.path.getmtime(self._input_path)
        except Exception:
            pass
        return None

    @staticmethod
    def _to_qimage(rgb: np.ndarray) -> QImage:
        """Convert RGB uint8 numpy array to QImage (deep copy)."""
        h, w, ch = rgb.shape
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        return qimg.copy()


class ThumbnailGenerator(QObject):
    """Manages background thumbnail generation for clips.

    Usage:
        gen = ThumbnailGenerator()
        gen.thumbnail_ready.connect(model.set_thumbnail)
        gen.generate(clip)
    """

    thumbnail_ready = Signal(str, object)  # clip_name, QImage

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._pending: set[str] = set()

    def generate(self, clip_name: str, clip_root: str,
                 input_path: str, asset_type: str) -> None:
        """Queue thumbnail generation for a clip."""
        if clip_name in self._pending:
            return
        self._pending.add(clip_name)

        task = _ThumbTask(clip_name, clip_root, input_path, asset_type)
        task.signals.finished.connect(self._on_finished)
        self._pool.start(task)

    def _on_finished(self, clip_name: str, qimage: object) -> None:
        self._pending.discard(clip_name)
        if qimage is not None:
            self.thumbnail_ready.emit(clip_name, qimage)

"""Async frame decoder using QThreadPool.

Decodes frames off the main thread with request coalescing — only the
latest request is honored when rapid scrubbing generates many requests.

Codex finding: synchronous decode freezes UI on scrub/mode change.
Pattern: worker generates QImage, signals main thread, main thread paints.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QRunnable, Signal, Slot, QThreadPool

from .frame_index import ViewMode
from .display_transform import decode_frame, decode_video_frame

logger = logging.getLogger(__name__)


class _DecodeSignals(QObject):
    """Signals for the decode worker (QRunnable can't have signals directly)."""
    finished = Signal(int, str, object)  # stem_index, mode_value, QImage|None


class _DecodeTask(QRunnable):
    """Decode a single frame in the thread pool."""

    def __init__(self, request_id: int, path: str, mode: ViewMode,
                 stem_index: int, video_path: str | None = None,
                 video_frame_index: int = 0):
        super().__init__()
        self.signals = _DecodeSignals()
        self._request_id = request_id
        self._path = path
        self._mode = mode
        self._stem_index = stem_index
        self._video_path = video_path
        self._video_frame_index = video_frame_index
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            if self._video_path:
                qimg = decode_video_frame(self._video_path, self._video_frame_index)
            else:
                qimg = decode_frame(self._path, self._mode)
            self.signals.finished.emit(self._stem_index, self._mode.value, qimg)
        except RuntimeError:
            pass  # Signal source deleted — task outlived its decoder
        except Exception as e:
            logger.warning(f"Async decode failed: {e}")
            try:
                self.signals.finished.emit(self._stem_index, self._mode.value, None)
            except RuntimeError:
                pass  # Signal source deleted


class AsyncDecoder(QObject):
    """Manages async frame decoding with request coalescing.

    Only the latest decode request is honored — older requests are
    discarded when their results arrive (stale check via request_id).

    Keeps references to in-flight signal objects to prevent premature GC.
    """

    frame_decoded = Signal(int, str, object)  # stem_index, mode_value, QImage|None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(2)  # limit decode concurrency
        self._current_request_id = 0
        # Hold references to in-flight signal objects so they aren't GC'd
        # before the callback fires. Keyed by request_id.
        self._pending_signals: dict[int, _DecodeSignals] = {}

    def request_decode(self, path: str, mode: ViewMode, stem_index: int,
                       video_path: str | None = None,
                       video_frame_index: int = 0) -> None:
        """Submit a decode request. Supersedes any pending request."""
        self._current_request_id += 1
        req_id = self._current_request_id

        task = _DecodeTask(req_id, path, mode, stem_index, video_path, video_frame_index)

        # Keep a strong reference to the signals object
        self._pending_signals[req_id] = task.signals

        task.signals.finished.connect(
            lambda si, mv, qi, rid=req_id: self._on_decoded(rid, si, mv, qi)
        )
        self._pool.start(task)

    def _on_decoded(self, request_id: int, stem_index: int,
                    mode_value: str, qimage: object) -> None:
        """Handle decode completion — discard if stale."""
        # Release the signal reference
        self._pending_signals.pop(request_id, None)

        if request_id != self._current_request_id:
            return  # Stale request, newer one superseded it
        self.frame_decoded.emit(stem_index, mode_value, qimage)

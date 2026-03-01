"""Center panel — frame preview viewport.

Phase 3: Composites SplitViewWidget, ViewModeBar, FrameScrubber, and
AsyncDecoder for a professional preview experience.

Architecture (Codex findings incorporated):
- QImage as internal currency, not QPixmap (guaranteed CPU-only)
- Stem-based navigation via FrameIndex (no index misalignment)
- Async frame decoding with request coalescing
- Worker preview gated: only applied if viewing same clip + COMP mode + latest frame
- Frame list snapshot built once per clip selection (atomic)
"""
from __future__ import annotations

import os
import logging

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage

from backend import ClipEntry
from ui.preview.frame_index import FrameIndex, ViewMode, build_frame_index
from ui.preview.display_transform import decode_frame, decode_video_frame, clear_cache
from ui.preview.async_decoder import AsyncDecoder
from ui.widgets.split_view import SplitViewWidget
from ui.widgets.frame_scrubber import FrameScrubber
from ui.widgets.view_mode_bar import ViewModeBar

logger = logging.getLogger(__name__)


class PreviewViewport(QWidget):
    """Center panel frame preview with split view, scrubber, and mode switching."""

    frame_changed = Signal(int)  # current stem index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)

        # State
        self._clip: ClipEntry | None = None
        self._clip_name: str = ""
        self._frame_index: FrameIndex | None = None
        self._current_stem_idx: int = -1
        self._current_mode: ViewMode = ViewMode.COMP

        # Async decoder
        self._decoder = AsyncDecoder(self)
        self._decoder.frame_decoded.connect(self._on_frame_decoded)

        # Build layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # View mode bar (top)
        self._mode_bar = ViewModeBar()
        self._mode_bar.mode_changed.connect(self._on_mode_changed)
        layout.addWidget(self._mode_bar)

        # Split view widget (center, fills space)
        self._split_view = SplitViewWidget()
        self._split_view.zoom_changed.connect(self._on_zoom_changed)
        layout.addWidget(self._split_view, 1)

        # Bottom bar: scrubber + zoom indicator
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(4)

        self._scrubber = FrameScrubber()
        self._scrubber.frame_changed.connect(self._on_scrubber_frame)
        bottom.addWidget(self._scrubber, 1)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(50)
        self._zoom_label.setAlignment(Qt.AlignCenter)
        self._zoom_label.setStyleSheet("color: #808070; font-size: 10px;")
        bottom.addWidget(self._zoom_label)

        layout.addLayout(bottom)

    # ── Public API ──

    def set_clip(self, clip: ClipEntry) -> None:
        """Load a clip and build its frame index.

        Called when user selects a clip in the browser.
        Builds frame index atomically (Codex: snapshot, don't re-read per scrub).
        """
        self._clip = clip
        self._clip_name = clip.name
        clear_cache()

        # Build frame index
        asset_type = clip.input_asset.asset_type if clip.input_asset else "sequence"
        self._frame_index = build_frame_index(clip.root_path, asset_type)

        # Configure mode bar
        available = self._frame_index.available_modes()
        self._mode_bar.set_available_modes(available)

        # Use mode bar's selected mode (it auto-selects best available)
        self._current_mode = self._mode_bar.current_mode()

        # Configure scrubber
        self._scrubber.set_range(self._frame_index.frame_count)

        if self._frame_index.frame_count > 0:
            self._navigate_to(0)
        else:
            self._split_view.set_placeholder(f"Selected: {clip.name}\nState: {clip.state.value}")

    def load_preview_from_file(self, file_path: str, clip_name: str, frame_index: int) -> None:
        """Load a worker preview image.

        Gated: only applied if viewing the same clip AND in COMP mode
        AND at the latest frame (Codex: don't override user browsing).
        """
        if clip_name != self._clip_name:
            return
        if self._current_mode != ViewMode.COMP:
            return

        # Only update if user is at or near the latest frame
        if self._frame_index and self._current_stem_idx >= 0:
            at_latest = self._current_stem_idx >= self._frame_index.frame_count - 2
            if not at_latest:
                return

        if not os.path.isfile(file_path):
            return

        qimg = decode_frame(file_path, ViewMode.COMP)
        if qimg:
            self._split_view.set_image(qimg)
            # Update scrubber to reflect new frame count during live inference
            if self._clip:
                asset_type = self._clip.input_asset.asset_type if self._clip.input_asset else "sequence"
                self._frame_index = build_frame_index(self._clip.root_path, asset_type)
                self._scrubber.set_range(self._frame_index.frame_count)
                self._scrubber.set_frame(self._frame_index.frame_count - 1)
                self._current_stem_idx = self._frame_index.frame_count - 1

    def show_placeholder(self, text: str = "No clip selected") -> None:
        """Show placeholder text."""
        self._split_view.set_placeholder(text)
        self._scrubber.set_range(0)
        self._clip = None
        self._clip_name = ""
        self._frame_index = None
        self._current_stem_idx = -1

    def set_split_mode(self, enabled: bool) -> None:
        """Toggle split view on/off."""
        self._split_view.set_split_enabled(enabled)
        if enabled and self._frame_index and self._current_stem_idx >= 0:
            self._load_split_images()

    def reset_zoom(self) -> None:
        """Reset zoom to fit."""
        self._split_view.reset_zoom()

    # ── Navigation ──

    def _navigate_to(self, stem_index: int) -> None:
        """Navigate to a stem index and load the frame for current mode."""
        if not self._frame_index or stem_index < 0:
            return
        stem_index = min(stem_index, self._frame_index.frame_count - 1)
        self._current_stem_idx = stem_index
        self._scrubber.set_frame(stem_index)
        self.frame_changed.emit(stem_index)
        self._request_frame(stem_index, self._current_mode)

        # If split view, also load input frame for left side
        if self._split_view.split_enabled:
            self._load_split_images()

    def _request_frame(self, stem_index: int, mode: ViewMode) -> None:
        """Request async decode of a frame."""
        if not self._frame_index:
            return

        # Video input mode
        if self._frame_index.is_video_mode(mode):
            video_path = self._frame_index.video_modes.get(mode)
            if video_path:
                self._decoder.request_decode(
                    "", mode, stem_index,
                    video_path=video_path,
                    video_frame_index=stem_index,
                )
            return

        # Image sequence mode
        path = self._frame_index.get_path(mode, stem_index)
        if path:
            self._decoder.request_decode(path, mode, stem_index)
        else:
            # Frame not available in this mode for this stem
            self._split_view.set_placeholder(
                f"No {mode.value} frame for stem {stem_index}"
            )

    def _load_split_images(self) -> None:
        """Load both input and current-mode images for split view."""
        if not self._frame_index or self._current_stem_idx < 0:
            return

        # Left = Input
        idx = self._current_stem_idx
        if self._frame_index.is_video_mode(ViewMode.INPUT):
            video_path = self._frame_index.video_modes.get(ViewMode.INPUT)
            if video_path:
                qimg = decode_video_frame(video_path, idx)
                if qimg:
                    self._split_view.set_left_image(qimg)
        else:
            path = self._frame_index.get_path(ViewMode.INPUT, idx)
            if path:
                qimg = decode_frame(path, ViewMode.INPUT)
                if qimg:
                    self._split_view.set_left_image(qimg)

    # ── Signal Handlers ──

    @Slot(str)
    def _on_mode_changed(self, mode_value: str) -> None:
        """Handle view mode switch."""
        try:
            self._current_mode = ViewMode(mode_value)
        except ValueError:
            return

        if self._current_stem_idx >= 0:
            self._request_frame(self._current_stem_idx, self._current_mode)

    @Slot(int)
    def _on_scrubber_frame(self, stem_index: int) -> None:
        """Handle scrubber navigation (debounced)."""
        self._navigate_to(stem_index)

    @Slot(int, str, object)
    def _on_frame_decoded(self, stem_index: int, mode_value: str, qimage: object) -> None:
        """Handle async decode completion — display the image."""
        if qimage is None:
            return

        if not isinstance(qimage, QImage):
            return

        if self._split_view.split_enabled:
            # Determine if this is left or right
            if mode_value == ViewMode.INPUT.value:
                self._split_view.set_left_image(qimage)
            else:
                self._split_view.set_right_image(qimage)
        else:
            self._split_view.set_image(qimage)

    @Slot(float)
    def _on_zoom_changed(self, zoom: float) -> None:
        self._zoom_label.setText(f"{int(zoom * 100)}%")

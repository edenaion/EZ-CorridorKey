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
from ui.widgets.annotation_overlay import AnnotationModel

logger = logging.getLogger(__name__)


class PreviewViewport(QWidget):
    """Center panel frame preview with split view, scrubber, and mode switching.

    When used inside DualViewerPanel, pass show_scrubber=False to hide the
    internal scrubber (the dual viewer provides a shared one).
    """

    frame_changed = Signal(int)  # current stem index
    view_mode_changed = Signal(str)

    def __init__(self, show_scrubber: bool = True, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 150)

        # State
        self._clip: ClipEntry | None = None
        self._clip_name: str = ""
        self._frame_index: FrameIndex | None = None
        self._current_stem_idx: int = -1
        self._current_mode: ViewMode = ViewMode.COMP
        self._locked_mode: ViewMode | None = None
        self._input_exr_is_linear: bool = False

        # Async decoder
        self._decoder = AsyncDecoder(self)
        self._decoder.frame_decoded.connect(self._on_frame_decoded)

        # Build layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top bar: view modes (left) + clip info (right)
        self._top_bar = QWidget()
        self._top_bar.setFixedHeight(30)
        self._top_bar.setStyleSheet("background: #0E0D00;")
        top_bar = QHBoxLayout(self._top_bar)
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(0)

        self._mode_bar = ViewModeBar()
        self._mode_bar.mode_changed.connect(self._on_mode_changed)
        top_bar.addWidget(self._mode_bar)

        self._clip_info = QLabel("")
        self._clip_info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._clip_info.setStyleSheet(
            "color: #808070; font-size: 11px; padding-right: 8px; background: #0E0D00;"
        )
        top_bar.addWidget(self._clip_info)

        layout.addWidget(self._top_bar)

        # Annotation model (shared across frames, persists during scrubbing)
        self._annotation_model = AnnotationModel()

        # Split view widget (center, fills space)
        self._split_view = SplitViewWidget()
        self._split_view.zoom_changed.connect(self._on_zoom_changed)
        self._split_view.set_annotation_model(self._annotation_model)
        layout.addWidget(self._split_view, 1)

        # Bottom bar: scrubber + zoom indicator (optional)
        self._has_scrubber = show_scrubber
        if show_scrubber:
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
        else:
            self._scrubber = None
            self._zoom_label = None

    @property
    def current_stem_index(self) -> int:
        """Current stem index for reprocess targeting."""
        return self._current_stem_idx

    # ── Public API ──

    def lock_mode(self, mode: ViewMode) -> None:
        """Lock this viewport to a specific view mode and hide the mode buttons.

        Used by DualViewerPanel to lock the input viewer to INPUT mode.
        The top bar stays visible (clip info label) so both viewers align.
        """
        self._locked_mode = mode
        self._current_mode = mode
        self._mode_bar.hide()

    def set_input_exr_is_linear(self, enabled: bool) -> None:
        """Set how INPUT-mode source frames should be interpreted for display."""
        if self._input_exr_is_linear == enabled:
            return
        self._input_exr_is_linear = enabled

        if self._current_stem_idx < 0:
            return

        if self._current_mode == ViewMode.INPUT:
            self._request_frame(self._current_stem_idx, self._current_mode)
        if self._split_view.split_enabled:
            self._load_split_images()

    # ── Annotation API ──

    def set_annotation_mode(self, mode: str | None) -> None:
        """Set annotation mode: 'fg', 'bg', or None to disable."""
        self._split_view.set_annotation_mode(mode)

    @property
    def annotation_mode(self) -> str | None:
        return self._split_view.annotation_mode

    @property
    def annotation_model(self) -> AnnotationModel:
        return self._annotation_model

    def clear_annotations(self) -> None:
        """Clear all annotations for the current clip."""
        self._annotation_model.clear()
        if self._clip is not None:
            self._annotation_model.save(self._clip.root_path)
        self._split_view.update()

    def set_clip(self, clip: ClipEntry) -> None:
        """Load a clip and build its frame index.

        Called when user selects a clip in the browser.
        Builds frame index atomically (Codex: snapshot, don't re-read per scrub).
        EXTRACTING clips show a placeholder — no frames to scrub yet.
        """
        from backend import ClipState

        # Save annotations for previous clip before switching
        if self._clip is not None:
            self._annotation_model.save(self._clip.root_path)
        self._clip = clip
        self._clip_name = clip.name
        # Preserve the caller-managed input interpretation when reloading a clip.
        # MainWindow reapplies the remembered per-clip override after set_clip();
        # resetting here causes tray clicks and clear actions to snap back to the
        # auto-detected default before the user asked for it.
        clear_cache()
        self._annotation_model.load(clip.root_path)

        # EXTRACTING clips: show placeholder, no frame index
        if clip.state == ClipState.EXTRACTING:
            self._frame_index = None
            self._current_stem_idx = -1
            if self._scrubber:
                self._scrubber.set_range(0)
            self._update_clip_info(clip)
            # Clear any cached images from previous clip
            self._split_view._single_image = None
            self._split_view._left_image = None
            self._split_view._right_image = None
            self._split_view.set_placeholder(
                f"Extracting frames...\n{clip.name}"
            )
            return

        # Build frame index
        self._frame_index = self._build_frame_index(clip)

        # Configure mode bar (unless locked)
        if self._locked_mode is None:
            available = self._frame_index.available_modes()
            self._mode_bar.set_available_modes(available)
            self._current_mode = self._mode_bar.current_mode()
        else:
            self._current_mode = self._locked_mode

        # Configure scrubber (if we have one)
        if self._scrubber:
            self._scrubber.set_range(self._frame_index.frame_count)

        # Update clip info label
        self._update_clip_info(clip)

        if self._frame_index.frame_count > 0:
            self._navigate_to(0)
        else:
            self._split_view.set_placeholder(f"Selected: {clip.name}\nState: {clip.state.value}")

    def refresh_available_assets(self) -> None:
        """Refresh mode availability and frame coverage without resetting navigation.

        Used by live progress updates while a job is writing frames. This keeps the
        current stem stable instead of calling set_clip(), which would jump back to frame 0.
        """
        if self._clip is None:
            return

        old_stem: str | None = None
        if self._frame_index and 0 <= self._current_stem_idx < self._frame_index.frame_count:
            old_stem = self._frame_index.stems[self._current_stem_idx]

        self._frame_index = self._build_frame_index(self._clip)

        if self._locked_mode is None:
            self._mode_bar.set_available_modes(self._frame_index.available_modes())
            self._current_mode = self._mode_bar.current_mode()
        else:
            self._current_mode = self._locked_mode

        if self._scrubber:
            self._scrubber.set_range(self._frame_index.frame_count)

        self._update_clip_info(self._clip)

        if self._frame_index.frame_count <= 0:
            self._current_stem_idx = -1
            self._split_view.set_annotation_stem_index(-1)
            return

        if old_stem and old_stem in self._frame_index.stems:
            new_idx = self._frame_index.stems.index(old_stem)
        elif self._current_stem_idx >= 0:
            new_idx = min(self._current_stem_idx, self._frame_index.frame_count - 1)
        else:
            new_idx = 0

        self._current_stem_idx = new_idx
        self._split_view.set_annotation_stem_index(new_idx)
        if self._scrubber:
            self._scrubber.set_frame(new_idx)

    def load_preview_from_file(self, file_path: str, clip_name: str, frame_index: int) -> None:
        """Load a worker preview image.

        Gated: only applied if viewing the same clip AND at the latest
        frame (Codex: don't override user browsing).  Always rebuilds
        FrameIndex so mode buttons enable as new outputs appear.
        """
        if clip_name != self._clip_name:
            return

        # Don't interrupt annotation mode with heavy I/O (frame index rebuild + decode)
        if self._split_view.is_annotating:
            return

        # Rebuild frame index so newly written outputs are discoverable
        if self._clip:
            self._frame_index = self._build_frame_index(self._clip)
            # Enable mode buttons as new output types appear during inference
            self._mode_bar.set_available_modes(self._frame_index.available_modes())

        # Only update the displayed image if in COMP mode at the latest frame
        if self._current_mode != ViewMode.COMP:
            return

        if self._frame_index and self._current_stem_idx >= 0:
            at_latest = self._current_stem_idx >= self._frame_index.frame_count - 2
            if not at_latest:
                return

        if not os.path.isfile(file_path):
            return

        qimg = decode_frame(file_path, ViewMode.COMP)
        if qimg:
            self._split_view.set_image(qimg)
            if self._frame_index:
                if self._scrubber:
                    self._scrubber.set_range(self._frame_index.frame_count)
                    self._scrubber.set_frame(self._frame_index.frame_count - 1)
                self._current_stem_idx = self._frame_index.frame_count - 1

    def _build_frame_index(self, clip: ClipEntry) -> FrameIndex:
        """Build a fresh FrameIndex snapshot for the current clip assets."""
        asset_type = clip.input_asset.asset_type if clip.input_asset else "sequence"
        video_path = clip.input_asset.path if (clip.input_asset and asset_type == "video") else None
        seq_dir = clip.input_asset.path if (clip.input_asset and asset_type == "sequence") else None
        return build_frame_index(
            clip.root_path, asset_type, video_path=video_path,
            input_sequence_dir=seq_dir,
        )

    def show_placeholder(self, text: str = "No clip selected") -> None:
        """Show placeholder text."""
        self._split_view.set_placeholder(text)
        if self._scrubber:
            self._scrubber.set_range(0)
        self._clip_info.setText("")
        self._clip = None
        self._clip_name = ""
        self._frame_index = None
        self._current_stem_idx = -1

    def hide_clip_info(self) -> None:
        """Hide the clip info label (used on right panel to avoid duplication)."""
        self._clip_info.hide()

    def navigate_to_frame(self, stem_index: int) -> None:
        """Public method for external scrubber to drive navigation."""
        self._navigate_to(stem_index)

    def _update_clip_info(self, clip: ClipEntry) -> None:
        """Update the clip info label with resolution, frame count, and type."""
        parts = []
        asset = clip.input_asset
        if asset:
            # Frame count
            parts.append(f"{asset.frame_count} frames")
            # Asset type
            if asset.asset_type == "video":
                parts.append("video")
            else:
                parts.append("sequence")
        parts.append(clip.state.value)
        self._clip_info.setText("  \u00B7  ".join(parts))  # middle dot separator

    def set_split_mode(self, enabled: bool) -> None:
        """Toggle split view on/off."""
        self._split_view.set_split_enabled(enabled)
        if enabled and self._frame_index and self._current_stem_idx >= 0:
            self._load_split_images()

    def reset_zoom(self) -> None:
        """Reset zoom to fit."""
        self._split_view.reset_zoom()

    def show_reprocess_preview(self, qimage: QImage) -> None:
        """Show a live reprocess result image in the viewport."""
        if self._split_view.split_enabled:
            self._split_view.set_right_image(qimage)
        else:
            self._split_view.set_image(qimage)

    # ── Navigation ──

    def _navigate_to(self, stem_index: int) -> None:
        """Navigate to a stem index and load the frame for current mode."""
        if not self._frame_index or stem_index < 0:
            return
        stem_index = min(stem_index, self._frame_index.frame_count - 1)
        self._current_stem_idx = stem_index
        self._split_view.set_annotation_stem_index(stem_index)
        if self._scrubber:
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
                    input_exr_is_linear=(self._input_exr_is_linear if mode == ViewMode.INPUT else False),
                )
            return

        # Image sequence mode
        path = self._frame_index.get_path(mode, stem_index)
        if path:
            self._decoder.request_decode(
                path,
                mode,
                stem_index,
                input_exr_is_linear=(self._input_exr_is_linear if mode == ViewMode.INPUT else False),
            )
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
                qimg = decode_video_frame(
                    video_path,
                    idx,
                    input_exr_is_linear=self._input_exr_is_linear,
                )
                if qimg:
                    self._split_view.set_left_image(qimg)
        else:
            path = self._frame_index.get_path(ViewMode.INPUT, idx)
            if path:
                qimg = decode_frame(
                    path,
                    ViewMode.INPUT,
                    input_exr_is_linear=self._input_exr_is_linear,
                )
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
        self.view_mode_changed.emit(mode_value)

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
        if self._zoom_label:
            self._zoom_label.setText(f"{int(zoom * 100)}%")

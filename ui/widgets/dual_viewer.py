"""Dual viewer panel — side-by-side input/output comparison with synced scrubbing.

Left viewer shows the original input, right viewer shows the keyed output
(COMP by default, user-switchable). Both viewers share a single frame scrubber
at the bottom that keeps them in sync.

When no output exists yet (clip not processed), both viewers show the input.
"""
from __future__ import annotations

import logging
import os

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage

from backend import ClipEntry
from ui.preview.frame_index import FrameIndex, ViewMode
from ui.widgets.preview_viewport import PreviewViewport
from ui.widgets.frame_scrubber import FrameScrubber

logger = logging.getLogger(__name__)


class DualViewerPanel(QWidget):
    """Side-by-side input/output viewers with synced scrubbing.

    The left viewer is locked to INPUT mode. The right viewer defaults to
    COMP mode but can be switched via its ViewModeBar.
    A single shared scrubber drives both viewers.
    """

    frame_changed = Signal(int)  # current stem index (for external listeners)
    output_mode_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._clip: ClipEntry | None = None

        self._wipe_active = False
        self._saved_splitter_sizes: list[int] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Horizontal splitter for the two viewers
        self._viewer_splitter = QSplitter(Qt.Horizontal)

        # Left viewer — locked to INPUT
        self._input_viewer = PreviewViewport(show_scrubber=False)
        self._input_viewer.lock_mode(ViewMode.INPUT)
        self._viewer_splitter.addWidget(self._input_viewer)

        # Right viewer — user-switchable modes (COMP default)
        self._output_viewer = PreviewViewport(show_scrubber=False)
        self._output_viewer.hide_clip_info()  # redundant with left panel
        self._viewer_splitter.addWidget(self._output_viewer)

        # Equal split
        self._viewer_splitter.setSizes([500, 500])
        self._viewer_splitter.setStretchFactor(0, 1)
        self._viewer_splitter.setStretchFactor(1, 1)

        layout.addWidget(self._viewer_splitter, 1)

        # A/B wipe button — floats in the top-center area above the splitter handle
        self._ab_button = QPushButton("A/B")
        self._ab_button.setCheckable(True)
        self._ab_button.setFixedSize(40, 20)
        self._ab_button.setToolTip(
            "Toggle A/B wipe comparison (hotkey: A)\n\n"
            "Overlays input (A) and output (B) in one viewer\n"
            "with a diagonal divider line.\n\n"
            "Drag the center handle to move the line.\n"
            "Drag on the line to rotate it."
        )
        self._ab_button.setStyleSheet(
            "QPushButton { background: #1A1900; color: #808070; border: 1px solid #333320; "
            "font-size: 10px; font-weight: bold; border-radius: 2px; }"
            "QPushButton:checked { background: #332E00; color: #FFF203; border-color: #FFF203; }"
            "QPushButton:hover { border-color: #666650; }"
        )
        self._ab_button.clicked.connect(self.toggle_wipe_mode)
        # Position the button absolutely — will be placed in resizeEvent
        self._ab_button.setParent(self)
        self._ab_button.raise_()

        # Shared scrubber at the bottom
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

        # Wire zoom from output viewer (primary)
        self._output_viewer._split_view.zoom_changed.connect(self._on_zoom_changed)
        self._output_viewer.view_mode_changed.connect(self.output_mode_changed.emit)

    @property
    def current_stem_index(self) -> int:
        """Current stem index for reprocess targeting."""
        return self._output_viewer.current_stem_index

    @property
    def current_output_mode(self) -> ViewMode:
        """Current output viewer mode."""
        return self._output_viewer._current_mode

    @property
    def input_viewer(self) -> PreviewViewport:
        """Access the input (left) viewer."""
        return self._input_viewer

    def set_input_exr_is_linear(self, enabled: bool) -> None:
        """Keep both viewers aligned on INPUT-mode EXR display interpretation."""
        self._input_viewer.set_input_exr_is_linear(enabled)
        self._output_viewer.set_input_exr_is_linear(enabled)

    # ── Public API ──

    def set_clip(self, clip: ClipEntry) -> None:
        """Load a clip into both viewers and configure the scrubber."""
        self._clip = clip

        # Load into both viewers
        self._input_viewer.set_clip(clip)
        self._output_viewer.set_clip(clip)

        # Configure shared scrubber from the output viewer's frame index
        fi = self._output_viewer._frame_index
        if fi:
            self._scrubber.set_range(fi.frame_count)
            if fi.frame_count > 0:
                self._scrubber.set_frame(0)
            self._update_coverage(clip, fi)

        # Restore in/out markers from clip data, default to full range
        if clip.in_out_range:
            self._scrubber.set_in_out(clip.in_out_range.in_point, clip.in_out_range.out_point)
        else:
            total = fi.frame_count if fi else 0
            if total > 0:
                self._scrubber.set_in_out(0, total - 1)
            else:
                self._scrubber.clear_in_out()

    def load_preview_from_file(self, file_path: str, clip_name: str, frame_index: int) -> None:
        """Forward worker preview to the output viewer."""
        self._output_viewer.load_preview_from_file(file_path, clip_name, frame_index)
        # Also update shared scrubber range and coverage (output may have new frames)
        fi = self._output_viewer._frame_index
        if fi:
            self._scrubber.set_range(fi.frame_count)
            if self._clip:
                self._update_coverage(self._clip, fi)

    def refresh_generated_assets(self) -> None:
        """Refresh output availability in place while preserving the current frame.

        Used for live progress updates so the scrubber coverage and mode buttons
        stay current without forcing a full clip reselection.
        """
        if self._clip is None:
            return

        current_frame = self._scrubber.current_frame()
        self._output_viewer.refresh_available_assets()

        fi = self._output_viewer._frame_index
        if fi is None:
            return

        self._scrubber.set_range(fi.frame_count)
        clamped_frame = min(current_frame, max(0, fi.frame_count - 1))
        if fi.frame_count > 0:
            self._scrubber.set_frame(clamped_frame)
            if self._input_viewer.current_stem_index != clamped_frame:
                self._input_viewer.navigate_to_frame(clamped_frame)
        self._update_coverage(self._clip, fi)

    def show_placeholder(self, text: str = "No clip selected") -> None:
        """Show placeholder on both viewers."""
        self._input_viewer.show_placeholder(text)
        self._output_viewer.show_placeholder(text)
        self._scrubber.set_range(0)
        self._clip = None

    def show_reprocess_preview(self, qimage: QImage) -> None:
        """Show a live reprocess result on the output viewer."""
        self._output_viewer.show_reprocess_preview(qimage)

    def reset_zoom(self) -> None:
        """Reset zoom on both viewers."""
        self._input_viewer.reset_zoom()
        self._output_viewer.reset_zoom()

    def set_extraction_progress(self, progress: float, total: int) -> None:
        """Forward extraction progress to the input viewer's split view."""
        self._input_viewer._split_view.set_extraction_progress(progress, total)

    # ── In/Out Markers ──

    def set_in_out(self, in_point: int | None, out_point: int | None) -> None:
        """Forward in/out markers to the scrubber."""
        self._scrubber.set_in_out(in_point, out_point)

    def get_in_out(self) -> tuple[int | None, int | None]:
        """Return current (in_point, out_point) from the scrubber."""
        return self._scrubber.get_in_out()

    # ── Coverage ──

    def _update_coverage(self, clip: ClipEntry, fi: FrameIndex) -> None:
        """Compute alpha and inference coverage arrays and update the scrubber."""
        stems = fi.stems
        if not stems:
            self._scrubber.set_coverage([], [])
            return

        # Alpha coverage: use frame index availability for ALPHA mode
        alpha_stems = fi.availability.get(ViewMode.ALPHA, set())

        # Inference coverage: any output mode (FG, Matte, Comp, Processed) has this stem
        output_modes = (ViewMode.FG, ViewMode.MATTE, ViewMode.COMP, ViewMode.PROCESSED)
        inference_stems: set[str] = set()
        for mode in output_modes:
            inference_stems |= fi.availability.get(mode, set())

        # Build boolean arrays aligned to stems
        alpha_coverage = [s in alpha_stems for s in stems]
        inference_coverage = [s in inference_stems for s in stems]

        self._scrubber.set_coverage(alpha_coverage, inference_coverage)

    # ── Navigation ──

    @Slot(int)
    def _on_scrubber_frame(self, stem_index: int) -> None:
        """Shared scrubber drives both viewers to the same frame."""
        self._input_viewer.navigate_to_frame(stem_index)
        self._output_viewer.navigate_to_frame(stem_index)
        self.frame_changed.emit(stem_index)
        # In wipe mode, reload both layers after a short delay to let async decode finish
        if self._wipe_active:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(100, self._load_wipe_images)

    def toggle_playback(self) -> None:
        """Forward play/pause to the scrubber."""
        self._scrubber.toggle_playback()

    @Slot(float)
    def _on_zoom_changed(self, zoom: float) -> None:
        self._zoom_label.setText(f"{int(zoom * 100)}%")

    # ── A/B Wipe ──

    def resizeEvent(self, event) -> None:
        """Reposition the floating A/B button at the top-center of the splitter."""
        super().resizeEvent(event)
        if hasattr(self, '_ab_button'):
            # Place at the top, centered on the splitter handle
            splitter_sizes = self._viewer_splitter.sizes()
            if splitter_sizes and sum(splitter_sizes) > 0:
                left_w = splitter_sizes[0]
            else:
                left_w = self.width() // 2
            btn_x = left_w - self._ab_button.width() // 2
            btn_y = 5  # small top margin
            self._ab_button.move(btn_x, btn_y)

    def toggle_wipe_mode(self) -> None:
        """Toggle A/B wipe comparison mode."""
        self._wipe_active = not self._wipe_active
        self._ab_button.setChecked(self._wipe_active)

        if self._wipe_active:
            # Save splitter state and hide input viewer
            self._saved_splitter_sizes = self._viewer_splitter.sizes()

            # Enable wipe on the output viewer's split view
            sv = self._output_viewer._split_view
            sv.set_wipe_mode(True)

            # Load current INPUT frame as left image for wipe
            self._load_wipe_images()

            # Hide input viewer, give output viewer full width
            self._input_viewer.hide()
            self._viewer_splitter.setSizes([0, self.width()])

            # Show clip info on output viewer (normally hidden in dual mode)
            self._output_viewer.show_clip_info()
        else:
            # Disable wipe
            sv = self._output_viewer._split_view
            sv.set_wipe_mode(False)

            # Restore side-by-side
            self._input_viewer.show()
            if self._saved_splitter_sizes:
                self._viewer_splitter.setSizes(self._saved_splitter_sizes)
            else:
                self._viewer_splitter.setSizes([500, 500])

            self._output_viewer.hide_clip_info()

        # Reposition button
        self.resizeEvent(None)

    def _load_wipe_images(self) -> None:
        """Load INPUT and current output mode images into the wipe viewer."""
        if not self._wipe_active:
            return
        # Get the input image from the input viewer's decode cache
        input_img = self._input_viewer._split_view._single_image
        output_img = self._output_viewer._split_view._single_image

        sv = self._output_viewer._split_view
        if input_img:
            sv.set_left_image(input_img)
        if output_img:
            sv.set_right_image(output_img)

from __future__ import annotations

import logging
import os

from PySide6.QtCore import Slot, QTimer

from backend import ClipEntry, ClipState, PipelineRoute, classify_pipeline_route

logger = logging.getLogger(__name__)


class ClipMixin:
    """Clip selection and navigation methods for MainWindow."""

    def _sync_selected_clip_view(self, clip: ClipEntry) -> None:
        """Reload the selected clip while preserving its remembered input interpretation."""
        self._dual_viewer.set_clip(clip)

        if clip.input_asset is not None:
            input_is_linear = self._input_is_linear_for_clip(clip)
            self._param_panel.set_input_is_linear(input_is_linear)
            self._dual_viewer.set_input_exr_is_linear(input_is_linear)
            self._refresh_input_thumbnail(
                clip,
                input_is_linear=input_is_linear,
            )

        self._refresh_export_thumbnail(clip)

    @Slot(int)
    def _on_clip_count_changed(self, count: int) -> None:
        """Handle clip added/removed — clear viewer if selected clip is gone.

        When the deleted clip was at position N, select the clip to its left
        (position N-1) rather than jumping to the first clip.
        """
        if self._current_clip is None:
            return
        # Check if current clip still exists in the model
        current_name = self._current_clip.name
        remaining = self._clip_model.clips
        if any(c.name == current_name for c in remaining):
            return  # Selected clip still exists, nothing to do

        # Selected clip was removed — select left neighbor
        if remaining:
            # Deleted clip was at _last_clip_index. After removal the list
            # shifted left, so the left neighbor is now at index-1 (clamped).
            pick = max(0, min(self._last_clip_index - 1, len(remaining) - 1))
            self._on_clip_selected(remaining[pick])
        else:
            self._current_clip = None
            self._dual_viewer.show_placeholder("No clip selected")
            self._refresh_button_state()

    @Slot(object)
    def _on_tray_clip_clicked(self, clip: ClipEntry) -> None:
        """Handle clip clicked in I/O tray — select it and load preview."""
        self._on_clip_selected(clip)

    @Slot(list)
    def _on_selection_changed(self, clips: list) -> None:
        """Handle multi-select change in I/O tray — update button state."""
        batch_count = len(clips)
        if batch_count > 1:
            # Check if any clip needs alpha generation (pipeline mode)
            needs_pipeline = any(
                classify_pipeline_route(c) not in (
                    PipelineRoute.INFERENCE_ONLY, PipelineRoute.SKIP)
                for c in clips
            )
            self._status_bar.update_button_state(
                can_run=True,
                has_partial=False,
                has_in_out=False,
                batch_count=batch_count,
                needs_pipeline=needs_pipeline,
            )
        elif batch_count == 1:
            # Single clip: use normal button state
            self._refresh_button_state()
        else:
            self._status_bar.update_button_state(
                can_run=False, has_partial=False, has_in_out=False,
            )

    # ── Clip Selection ──

    @Slot(ClipEntry)
    def _on_clip_selected(self, clip: ClipEntry) -> None:
        if self._current_clip is not None and self._current_clip is not clip:
            self._remember_current_clip_input_color_space()
        self._current_clip = clip
        # Track position for left-neighbor selection on delete
        for i, c in enumerate(self._clip_model.clips):
            if c.name == clip.name:
                self._last_clip_index = i
                break
        logger.debug(f"Clip selected: '{clip.name}' state={clip.state.value}")

        # Highlight in I/O tray (single-select unless multi-select is active)
        batch_count = self._io_tray.selected_count()
        if batch_count <= 1:
            self._io_tray.set_selected(clip.name)

        # Load clip into both viewers while preserving the remembered input
        # interpretation for the selected clip.
        self._sync_selected_clip_view(clip)

        # Refresh annotation coverage bar (annotations loaded from disk above)
        self._update_annotation_info()

        # Ensure run/stop buttons are in correct visibility state
        # (guards against stale running state from crashed jobs)
        if not self._gpu_worker.isRunning():
            self._status_bar.set_running(False)

        # Enable run button only for READY or COMPLETE (reprocess) clips
        can_run = clip.state in (ClipState.READY, ClipState.COMPLETE)
        needs_extraction = clip.state == ClipState.EXTRACTING
        logger.debug(f"Run button enabled: {can_run} (state={clip.state.value})")
        self._status_bar.update_button_state(
            can_run=can_run,
            has_partial=clip.completed_frame_count() > 0,
            has_in_out=clip.in_out_range is not None,
            batch_count=batch_count if batch_count > 1 else 0,
            needs_extraction=needs_extraction,
        )

        # Enable GVM/BiRefNet/VideoMaMa/MatAnyone2/Import Alpha buttons based on state
        self._param_panel.set_gvm_enabled(clip.state in (ClipState.RAW, ClipState.MASKED))
        self._param_panel.set_birefnet_enabled(clip.state in (ClipState.RAW, ClipState.MASKED))
        has_mask = self._clip_has_videomama_ready_mask(clip)
        self._param_panel.set_videomama_enabled(has_mask)
        self._param_panel.set_matanyone2_enabled(has_mask)
        self._param_panel.set_import_alpha_enabled(
            clip.state in (ClipState.RAW, ClipState.MASKED, ClipState.READY)
        )

    @Slot(str)
    def _on_clips_dir_changed(
        self, dir_path: str, *,
        skip_session_restore: bool = False,
        select_clip: str | None = None,
    ) -> None:
        logger.info(f"Scanning clips directory: {dir_path}")
        previous_clips_dir = self._clips_dir
        self._clips_dir = dir_path
        # Reset status bar state on project load (no active job)
        self._status_bar.set_running(False)
        self._status_bar.update_button_state(
            can_run=False, has_partial=False, has_in_out=False,
        )
        # Ensure workspace is visible (may come from welcome screen or menu)
        self._switch_to_workspace()
        try:
            # Detect if this is the Projects root (no standalone videos there)
            from backend.project import projects_root as _projects_root, get_display_name
            is_projects = os.path.normcase(os.path.abspath(dir_path)) == os.path.normcase(
                os.path.abspath(_projects_root())
            )
            clips = self._service.scan_clips(
                dir_path, allow_standalone_videos=not is_projects,
            )
            if (
                previous_clips_dir is None
                or os.path.normcase(os.path.abspath(previous_clips_dir))
                != os.path.normcase(os.path.abspath(dir_path))
            ):
                self._clip_input_is_linear = {}
            else:
                current_names = {clip.name for clip in clips}
                self._clip_input_is_linear = {
                    name: value
                    for name, value in self._clip_input_is_linear.items()
                    if name in current_names
                }
            self._clip_model.set_clips(clips)

            # Generate thumbnails for all clips (background)
            for clip in clips:
                self._refresh_input_thumbnail(clip)
                self._refresh_export_thumbnail(clip)

            # Auto-submit EXTRACTING clips to extract worker
            self._auto_extract_clips(clips)

            if clips:
                # Select the requested clip (e.g. newly imported), fall back to last
                target = None
                if select_clip:
                    for clip in clips:
                        if clip.name == select_clip:
                            target = clip
                            break
                if target is None:
                    target = clips[-1]  # newest (timestamps sort ascending)
                self._on_clip_selected(target)
            logger.info(f"Found {len(clips)} clips")

            # Register in recent sessions store — per-project, not per-clip
            if is_projects:
                from backend.project import is_v2_project, get_clip_dirs
                # Group clips by their project container
                registered: set[str] = set()
                for clip in clips:
                    # Find the project dir this clip belongs to
                    # v2: clip.root_path is .../project_dir/clips/clip_name
                    # v1: clip.root_path IS the project dir
                    clip_parent = os.path.dirname(clip.root_path)
                    if os.path.basename(clip_parent) == "clips":
                        project_path = os.path.dirname(clip_parent)
                    else:
                        project_path = clip.root_path
                    norm_proj = os.path.normcase(os.path.abspath(project_path))
                    if norm_proj not in registered:
                        registered.add(norm_proj)
                        proj_name = get_display_name(project_path)
                        clip_count = len(get_clip_dirs(project_path))
                        self._recent_store.add_or_update(
                            project_path, proj_name, clip_count,
                        )
            else:
                from backend.project import is_v2_project as _is_v2
                if _is_v2(dir_path):
                    display_name = get_display_name(dir_path)
                else:
                    display_name = os.path.basename(dir_path)
                self._recent_store.add_or_update(dir_path, display_name, len(clips))

            # Auto-load session if exists (Codex: block signals during restore)
            # Skip restore when creating new projects — prevents old session
            # from overriding the newly-added clip selection.
            if not skip_session_restore:
                self._try_auto_load_session(dir_path)

            # Refresh project-level menu state (output folder actions)
            self._refresh_project_output_actions()

        except Exception as e:
            logger.error(f"Failed to scan clips: {e}")
            from ui.sounds.audio_manager import UIAudio
            from PySide6.QtWidgets import QMessageBox
            UIAudio.error()
            QMessageBox.critical(self, "Scan Error", f"Failed to scan clips directory:\n{e}")

    def _switch_to_workspace(self) -> None:
        """Switch from welcome screen to the 3-panel workspace."""
        self._stack.setCurrentIndex(1)
        self._status_bar.show()
        QTimer.singleShot(0, self._ensure_window_mode)
        # Sync IO tray divider and position queue overlay after layout settles
        QTimer.singleShot(0, self._sync_io_divider)
        QTimer.singleShot(0, self._position_queue_panel)

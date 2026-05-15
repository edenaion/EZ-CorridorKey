"""Batch Pipeline mixin — File > Batch Pipeline menu action and runner."""
from __future__ import annotations

import logging
import os

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QDialog, QInputDialog, QMessageBox

from . import _tr

logger = logging.getLogger(__name__)


class BatchPipelineMixin:
    """Adds batch pipeline functionality to MainWindow."""

    def _init_batch_pipeline(self) -> None:
        """Initialize batch pipeline state. Call from MainWindow.__init__."""
        self._batch_dialog = None
        self._batch_pending_extractions: dict[str, "BatchClipConfig"] = {}
        self._batch_configs: dict[str, "BatchClipConfig"] = {}

    # ------------------------------------------------------------------
    # Menu action
    # ------------------------------------------------------------------

    @Slot()
    def _on_batch_pipeline(self) -> None:
        """Show the Batch Pipeline dialog."""
        from ui.widgets.batch_pipeline_dialog import BatchPipelineDialog

        # If a batch is already running, just re-show the progress dialog
        if self._batch_configs and self._batch_dialog is not None:
            self._batch_dialog.show()
            self._batch_dialog.raise_()
            self._batch_dialog.activateWindow()
            return

        dialog = BatchPipelineDialog(self)
        dialog.run_requested.connect(lambda: self._on_batch_run_requested(dialog))
        dialog.clear_requested.connect(lambda: self._on_batch_clear_requested(dialog))
        self._batch_dialog = dialog
        dialog.show()

    @Slot()
    def _on_batch_run_requested(self, dialog) -> None:
        """Handle Run Batch button from the dialog."""
        configs = dialog.get_batch_config()
        folder_path = dialog.get_folder_path()
        if not configs or not folder_path:
            return

        # Ensure we have a project open
        if not self._clips_dir:
            name, ok = QInputDialog.getText(
                self, _tr("New Project"),
                _tr("Project name for this batch:"),
                text=os.path.basename(folder_path),
            )
            if not ok or not name.strip():
                return
            self._create_batch_project(name.strip(), configs, dialog)
            return

        self._run_batch_import_and_process(configs, dialog)

    @Slot()
    def _on_batch_clear_requested(self, dialog) -> None:
        """Handle Clear Pipeline button. Always confirms before clearing."""
        from PySide6.QtWidgets import QMessageBox

        msg = (_tr("Cancel all pending batch jobs and clear the pipeline?")
               if self._batch_configs
               else _tr("Clear the current batch folder and clip list?"))
        reply = QMessageBox.question(
            dialog, _tr("Clear Batch Pipeline"), msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        if self._batch_configs:
            for clip_name in list(self._batch_configs.keys()):
                self._batch_pending_extractions.pop(clip_name, None)
                self._pipeline_steps.pop(clip_name, None)
            self._batch_configs.clear()
            logger.info("Batch pipeline cleared")

        dialog.reset_to_initial()

    # ------------------------------------------------------------------
    # Project creation (when no project is open)
    # ------------------------------------------------------------------

    def _create_batch_project(self, name: str, configs: list, dialog=None) -> None:
        """Create a new project and import batch clips into it."""
        from backend.project import create_project, filter_companion_hints, get_clip_dirs

        source_paths = [c.clip_info.source_path for c in configs]
        filtered = filter_companion_hints(source_paths)

        from ui.widgets.preferences_dialog import KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE
        from ui.widgets.preferences_dialog import get_setting_bool
        copy_source = get_setting_bool(KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE)

        project_dir = create_project(
            filtered, copy_source=copy_source, display_name=name,
        )
        logger.info(f"Created batch project: {project_dir}")

        # Map actual clip folder names to configs using source path matching
        self._batch_configs = self._map_configs_to_clips(
            configs, get_clip_dirs(project_dir),
        )

        # Open the project (triggers _on_clips_dir_changed -> rescan -> extraction)
        self._on_clips_dir_changed(project_dir, skip_session_restore=True)

        # After rescan, submit batch jobs
        self._submit_batch_jobs()

    # ------------------------------------------------------------------
    # Import into existing project
    # ------------------------------------------------------------------

    def _run_batch_import_and_process(self, configs: list, dialog=None) -> None:
        """Import batch clips into the current project and run pipelines."""
        from backend.project import add_clips_to_project, filter_companion_hints
        from ui.widgets.preferences_dialog import KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE
        from ui.widgets.preferences_dialog import get_setting_bool

        source_paths = [c.clip_info.source_path for c in configs]
        filtered = filter_companion_hints(source_paths)

        copy_source = get_setting_bool(KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE)
        new_clip_paths = add_clips_to_project(
            self._clips_dir, filtered, copy_source=copy_source,
        )

        # Map actual clip folder names to configs
        self._batch_configs = self._map_configs_to_clips(configs, new_clip_paths)

        # Rescan to pick up new clips
        self._on_clips_dir_changed(self._clips_dir, skip_session_restore=True)

        # Submit batch jobs
        self._submit_batch_jobs()

    # ------------------------------------------------------------------
    # Config-to-clip name mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _map_configs_to_clips(configs: list, clip_paths: list[str]) -> dict:
        """Map batch configs to actual clip folder names.

        Clip folder names may be deduplicated (e.g., Shot01 -> Shot01_2) so
        we match by reading clip.json source path from each clip folder.
        """
        from backend.project import read_clip_json

        # Build source_path -> config lookup (normalized)
        source_to_config = {}
        for c in configs:
            norm = os.path.normcase(os.path.abspath(c.clip_info.source_path))
            source_to_config[norm] = c

        result = {}
        for clip_path in clip_paths:
            clip_name = os.path.basename(clip_path)
            clip_json = read_clip_json(clip_path)
            if not clip_json:
                continue
            original = clip_json.get("source", {}).get("original_path", "")
            if not original:
                continue
            norm = os.path.normcase(os.path.abspath(original))
            config = source_to_config.get(norm)
            if config:
                result[clip_name] = config
                config.clip_info.name = clip_name

        return result

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    def _submit_batch_jobs(self) -> None:
        """Submit pipeline jobs for all batch clips.

        Clips needing extraction (EXTRACTING state) are tracked and their
        GPU jobs will be submitted when extraction completes via
        _on_batch_extract_finished.
        """
        from backend.clip_state import ClipState
        from backend.job_queue import JobType
        from ui.workers.job_helpers import create_job_snapshot

        if not self._batch_configs:
            return

        params = self._param_panel.get_params()
        output_config = self._param_panel.get_output_config()

        self._batch_pending_extractions.clear()
        self._pipeline_steps.clear()

        # Switch dialog to processing mode (it stays open in place)
        if self._batch_dialog:
            self._batch_dialog.enter_processing_mode(
                [c.clip_info for c in self._batch_configs.values()]
            )

        queued = 0
        first_job_id = None

        for clip in self._clip_model.clips:
            config = self._batch_configs.get(clip.name)
            if config is None:
                continue

            # Video clips need extraction first
            if clip.state == ClipState.EXTRACTING:
                self._batch_pending_extractions[clip.name] = config
                if self._batch_dialog:
                    self._batch_dialog.set_clip_running(clip.name)
                continue

            job, next_steps = self._create_batch_job(clip, config, params, output_config)
            if job is None:
                continue

            if self._service.job_queue.submit(job):
                clip.set_processing(True)
                if next_steps:
                    self._pipeline_steps[clip.name] = next_steps
                if first_job_id is None:
                    first_job_id = job.id
                queued += 1
                if self._batch_dialog:
                    self._batch_dialog.set_clip_running(clip.name)

        if queued > 0:
            self._start_worker_if_needed(first_job_id, job_label="Batch Pipeline")

        logger.info(
            f"Batch pipeline: {queued} jobs queued, "
            f"{len(self._batch_pending_extractions)} awaiting extraction"
        )

    def _create_batch_job(self, clip, config, params, output_config):
        """Create a GPU job + next_steps for a single batch clip.

        Returns (job, next_steps) or (None, []) if the clip can't be processed.
        """
        from backend.clip_state import ClipState
        from backend.job_queue import JobType
        from ui.workers.job_helpers import create_job_snapshot

        job = None
        next_steps: list[JobType] = []

        if config.alpha_job_type == JobType.INFERENCE:
            # AlphaHint already present, go straight to CK
            if clip.state == ClipState.COMPLETE:
                clip.transition_to(ClipState.READY)
            job = create_job_snapshot(clip, params)
            job.params["_output_config"] = output_config
        elif config.alpha_job_type in (JobType.VIDEOMAMA_ALPHA, JobType.MATANYONE2_ALPHA):
            # MaskHint path: run mask refinement, then inference
            job = create_job_snapshot(clip, job_type=config.alpha_job_type)
            next_steps = [JobType.INFERENCE]
        elif config.alpha_job_type == JobType.GVM_ALPHA:
            # No hint: GVM -> inference
            job = create_job_snapshot(clip, job_type=JobType.GVM_ALPHA)
            next_steps = [JobType.INFERENCE]
        elif config.alpha_job_type == JobType.BIREFNET_ALPHA:
            # No hint: BiRefNet -> inference
            job = create_job_snapshot(
                clip, job_type=JobType.BIREFNET_ALPHA,
                birefnet_usage=config.birefnet_usage,
            )
            next_steps = [JobType.INFERENCE]

        return job, next_steps

    # ------------------------------------------------------------------
    # Signal handlers for batch progress
    # ------------------------------------------------------------------

    def _on_batch_extract_finished(self, clip_name: str) -> None:
        """Called when a batch clip finishes extraction. Submit its GPU job."""
        config = self._batch_pending_extractions.pop(clip_name, None)
        if config is None:
            return

        from backend.clip_state import ClipState
        from ui.workers.job_helpers import create_job_snapshot

        params = self._param_panel.get_params()
        output_config = self._param_panel.get_output_config()

        # Find the updated clip
        clip = None
        for c in self._clip_model.clips:
            if c.name == clip_name:
                clip = c
                break
        if clip is None or clip.state not in (ClipState.RAW, ClipState.READY):
            logger.warning(f"Batch: clip {clip_name} not ready after extraction")
            if self._batch_dialog:
                self._batch_dialog.set_clip_error(clip_name, "Not ready after extraction")
            return

        job, next_steps = self._create_batch_job(clip, config, params, output_config)
        if job and self._service.job_queue.submit(job):
            clip.set_processing(True)
            if next_steps:
                self._pipeline_steps[clip.name] = next_steps
            self._start_worker_if_needed(job.id, job_label="Batch Pipeline")
            logger.info(f"Batch: submitted GPU job for {clip_name} after extraction")

    def _on_batch_clip_finished(self, job_id: str, clip_name: str, job_type: str) -> None:
        """Update batch dialog when a clip's final job completes."""
        from backend.job_queue import JobType
        if not self._batch_dialog or clip_name not in self._batch_configs:
            return

        # Only mark done when the INFERENCE job completes (the final step)
        if job_type == JobType.INFERENCE.value:
            self._batch_dialog.set_clip_done(clip_name)
            # Check if all batch clips are done
            self._check_batch_complete()
        else:
            # Intermediate job (GVM, BiRefNet, VideoMaMa, etc.) — still running
            self._batch_dialog.set_clip_running(clip_name)

    def _on_batch_clip_error(self, job_id: str, clip_name: str, error_msg: str) -> None:
        """Update batch dialog when a clip errors."""
        if not self._batch_dialog or clip_name not in self._batch_configs:
            return
        self._batch_dialog.set_clip_error(clip_name, error_msg)
        self._check_batch_complete()

    def _on_batch_progress(self, job_id: str, clip_name: str, current: int, total: int, fps: float) -> None:
        """Update batch dialog progress bar."""
        if not self._batch_dialog or clip_name not in self._batch_configs:
            return
        self._batch_dialog.set_clip_progress(clip_name, current, total)

    def _check_batch_complete(self) -> None:
        """Check if all batch clips are done or errored."""
        if not self._batch_dialog or not self._batch_configs:
            return

        for clip_name in self._batch_configs:
            row = self._batch_dialog._find_clip_row(clip_name)
            if row < 0:
                continue
            # Check if this clip has a progress bar still (running) or is pending extraction
            if self._batch_dialog._progress_bars[row] is not None:
                return  # still running
            if clip_name in self._batch_pending_extractions:
                return  # still extracting
            # Check status widget text
            widget = self._batch_dialog._table.cellWidget(row, 3)
            if widget and isinstance(widget, type(self._batch_dialog._table.cellWidget(row, 3))):
                text = widget.text() if hasattr(widget, 'text') else ""
                if not text:
                    return  # not started yet

        self._batch_dialog.set_batch_complete()
        self._batch_configs.clear()
        logger.info("Batch pipeline complete")

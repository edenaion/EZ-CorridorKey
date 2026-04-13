from __future__ import annotations

import logging

from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import Slot

from backend import ClipState, JobType
from ui.workers.gpu_job_worker import create_job_snapshot

logger = logging.getLogger(__name__)


class WorkerMixin:
    """GPU worker signal handlers for MainWindow."""

    @Slot(str, str, int, int)
    def _on_worker_progress(
        self, job_id: str, clip_name: str, current: int, total: int, fps: float = 0.0
    ) -> None:
        if self._cancel_requested_job_id == job_id:
            self._queue_panel.refresh()
            return

        # Set active_job_id only on first progress of a new job (not every event)
        if self._active_job_id != job_id:
            # Only update if this is genuinely a new running job
            current_job = self._service.job_queue.current_job
            if current_job and current_job.id == job_id:
                self._active_job_id = job_id

        if job_id == self._active_job_id:
            self._status_bar.update_progress(current, total, fps)

        if self._current_clip and self._current_clip.name == clip_name:
            self._schedule_live_asset_refresh(clip_name, current, total)

        self._queue_panel.refresh()

    def _schedule_live_asset_refresh(self, clip_name: str, current: int, total: int) -> None:
        """Coalesce progress-driven asset refreshes for the selected clip.

        This keeps the feedback live without rescanning on every frame tick or
        touching any GPU-side work.
        """
        if current <= 0 or total <= 0:
            return

        step = max(1, total // 50)
        should_refresh = current <= 3 or current >= total or total <= 30 or current % step == 0
        if not should_refresh:
            return

        self._pending_live_asset_refresh_clip = clip_name
        if not self._live_asset_refresh_timer.isActive():
            self._live_asset_refresh_timer.start()

    @Slot()
    def _refresh_selected_clip_live_assets(self) -> None:
        """Refresh the selected clip's coverage/modes without resetting navigation."""
        clip_name = self._pending_live_asset_refresh_clip
        self._pending_live_asset_refresh_clip = None

        if clip_name is None or self._current_clip is None:
            return
        if self._current_clip.name != clip_name:
            return

        self._dual_viewer.refresh_generated_assets()

    @Slot(str, str)
    def _on_worker_status(self, job_id: str, message: str) -> None:
        """Phase status from long-running jobs (e.g. VideoMaMa loading phases).

        Always update — status signals fire during loading phases before
        progress signals arrive, so _active_job_id may not match yet.
        Also set _active_job_id if not already set.
        """
        if self._cancel_requested_job_id == job_id:
            return
        if self._active_job_id is None:
            self._active_job_id = job_id
        self._status_bar.set_phase(message)

    @Slot(str, str, int, str)
    def _on_worker_preview(self, job_id: str, clip_name: str, frame_index: int, path: str) -> None:
        if self._cancel_requested_job_id == job_id:
            return
        # Only update preview if this is the active job
        if job_id == self._active_job_id:
            self._dual_viewer.load_preview_from_file(path, clip_name, frame_index)

    @Slot(str, str, str)
    def _on_worker_clip_finished(self, job_id: str, clip_name: str, job_type: str) -> None:
        self._force_stop_armed = False
        self._status_bar.set_stop_button_mode(force=False)
        # Map job type to correct next state.
        if job_type == JobType.SAM2_TRACK.value:
            target_state = ClipState.MASKED
        elif job_type in (
            JobType.GVM_ALPHA.value,
            JobType.VIDEOMAMA_ALPHA.value,
            JobType.MATANYONE2_ALPHA.value,
            JobType.BIREFNET_ALPHA.value,
        ):
            target_state = ClipState.READY
        else:
            target_state = ClipState.COMPLETE

        self._clip_model.update_clip_state(clip_name, target_state)

        # Clear processing lock and rescan assets for pipeline steps
        for clip in self._clip_model.clips:
            if clip.name == clip_name:
                clip.set_processing(False)
                if target_state in (ClipState.MASKED, ClipState.READY):
                    try:
                        clip.find_assets()
                    except Exception:
                        pass
                    clip.state = target_state
                    self._clip_model.update_clip_state(clip_name, target_state)
                elif target_state == ClipState.COMPLETE:
                    try:
                        clip.find_assets()
                    except Exception:
                        pass
                break

        finished_clip = next((c for c in self._clip_model.clips if c.name == clip_name), None)
        if finished_clip is not None:
            self._refresh_input_thumbnail(finished_clip)
            self._refresh_export_thumbnail(finished_clip)

        # Pipeline auto-chain: queue the next stage, if any.
        if clip_name in self._pipeline_steps and self._pipeline_steps[clip_name]:
            next_step = self._pipeline_steps[clip_name].pop(0)
            queued_next = False
            for clip in self._clip_model.clips:
                if clip.name != clip_name:
                    continue
                if next_step == JobType.INFERENCE:
                    params = self._param_panel.get_params()
                    job = create_job_snapshot(clip, params)
                    job.params["_output_config"] = self._param_panel.get_output_config()
                    if clip.in_out_range:
                        job.params["_frame_range"] = (
                            clip.in_out_range.in_point,
                            clip.in_out_range.out_point,
                        )
                else:
                    job = create_job_snapshot(clip, job_type=next_step)
                clip.set_processing(True)
                if self._service.job_queue.submit(job):
                    queued_next = True
                    logger.info(
                        "Pipeline auto-chain: queued %s for %s",
                        next_step.value,
                        clip_name,
                    )
                else:
                    clip.set_processing(False)
                break
            if not self._pipeline_steps[clip_name]:
                self._pipeline_steps.pop(clip_name, None)
            elif not queued_next:
                logger.warning("Pipeline auto-chain failed to queue next step for %s", clip_name)

        # Stop timer; only exit running state if no more jobs are pending
        has_more = self._service.job_queue.has_pending or self._pipeline_steps
        self._status_bar.stop_job_timer()
        if not has_more:
            self._status_bar.set_running(False)
        else:
            # Reset progress for next job — show descriptive label
            self._status_bar.reset_progress()
            next_job = self._service.job_queue.next_job()
            if next_job:
                _label_map = {
                    JobType.GVM_ALPHA: "GVM Auto",
                    JobType.BIREFNET_ALPHA: "BiRefNet",
                    JobType.SAM2_PREVIEW: "Track Preview",
                    JobType.SAM2_TRACK: "Track Mask",
                    JobType.VIDEOMAMA_ALPHA: "VideoMaMa",
                    JobType.MATANYONE2_ALPHA: "MatAnyone2",
                    JobType.INFERENCE: "Inference",
                }
                next_label = _label_map.get(next_job.job_type, "Pipeline")
            else:
                next_label = "Pipeline"
            self._status_bar.start_job_timer(label=next_label)

        from ui.sounds.audio_manager import UIAudio

        if target_state == ClipState.MASKED:
            UIAudio.mask_done()
            self._status_bar.set_message(f"Track Mask complete for {clip_name}")
        elif target_state == ClipState.READY:
            UIAudio.mask_done()
            type_label = {
                JobType.GVM_ALPHA.value: "GVM Auto",
                JobType.BIREFNET_ALPHA.value: "BiRefNet",
                JobType.VIDEOMAMA_ALPHA.value: "VideoMaMa",
                JobType.MATANYONE2_ALPHA.value: "MatAnyone2",
            }.get(job_type, "Alpha")
            # Show alpha coverage count
            alpha_info = ""
            for c in self._clip_model.clips:
                if c.name == clip_name and c.alpha_asset and c.input_asset:
                    alpha_info = (
                        f" ({c.alpha_asset.frame_count}/{c.input_asset.frame_count} alpha frames)"
                    )
                    break
            self._status_bar.set_message(
                f"{type_label} complete for {clip_name}{alpha_info} -- Ready to Run Inference"
            )
        elif target_state == ClipState.COMPLETE:
            UIAudio.inference_done()
            self._status_bar.set_message(f"Inference complete: {clip_name}")

        # Refresh views
        self._pending_live_asset_refresh_clip = None
        self._live_asset_refresh_timer.stop()
        self._queue_panel.refresh()
        self._io_tray.refresh()

        # If selected clip, reload preview to show new assets
        if self._current_clip and self._current_clip.name == clip_name:
            self._sync_selected_clip_view(self._current_clip)
            self._refresh_button_state()
            _has_mask = self._clip_has_videomama_ready_mask(self._current_clip)
            self._param_panel.set_videomama_enabled(_has_mask)
            self._param_panel.set_matanyone2_enabled(_has_mask)
            self._param_panel.set_import_alpha_enabled(
                self._current_clip.state in (ClipState.RAW, ClipState.MASKED, ClipState.READY)
            )

        logger.info(f"Clip finished ({job_type}): {clip_name} -> {target_state.value}")

    @Slot(str, str)
    def _on_worker_warning(self, job_id: str, message: str) -> None:
        if message.startswith("Cancelled:"):
            self._cancel_requested_job_id = None
            self._active_job_id = None
            self._force_stop_armed = False
            self._status_bar.set_stop_button_mode(force=False)
            # Job was cancelled — clear processing lock on the clip
            clip_name = message.removeprefix("Cancelled:").strip()
            self._pipeline_steps.pop(clip_name, None)
            for clip in self._clip_model.clips:
                if clip.name == clip_name:
                    clip.set_processing(False)
                    # Refresh viewer to show any partial output from before cancel
                    if self._current_clip and self._current_clip.name == clip_name:
                        clip.find_assets()
                        self._sync_selected_clip_view(clip)
                        self._refresh_button_state()
                    break
            self._status_bar.stop_job_timer()
            self._status_bar.set_running(False)
            self._status_bar.set_message(f"Cancelled: {clip_name}")
            self._pending_live_asset_refresh_clip = None
            self._live_asset_refresh_timer.stop()
            self._queue_panel.refresh()
            logger.info(f"Job cancelled: {clip_name}")
        else:
            self._status_bar.add_warning(message)
            logger.warning(f"Worker warning: {message}")

    @Slot(str, str, str)
    def _on_worker_error(self, job_id: str, clip_name: str, error_msg: str) -> None:
        if self._cancel_requested_job_id == job_id:
            self._cancel_requested_job_id = None
            self._active_job_id = None
        self._force_stop_armed = False
        self._status_bar.set_stop_button_mode(force=False)
        self._status_bar.stop_job_timer()
        self._status_bar.set_running(False)
        self._pending_live_asset_refresh_clip = None
        self._live_asset_refresh_timer.stop()
        self._pipeline_steps.pop(clip_name, None)
        self._clip_model.update_clip_state(clip_name, ClipState.ERROR)
        # Clear processing lock
        for clip in self._clip_model.clips:
            if clip.name == clip_name:
                clip.set_processing(False)
                # Refresh viewer to show any partial output
                if self._current_clip and self._current_clip.name == clip_name:
                    clip.find_assets()
                    self._sync_selected_clip_view(clip)
                break
        self._queue_panel.refresh()
        logger.error(f"Worker error for {clip_name}: {error_msg}")
        from ui.sounds.audio_manager import UIAudio

        UIAudio.error()

        # Try to match a known error pattern for actionable diagnostics
        from ui.widgets.diagnostic_dialog import match_diagnostic, DiagnosticDialog

        diag = match_diagnostic(error_msg)
        if diag:
            dlg = DiagnosticDialog(
                diag,
                error_msg,
                gpu_info=self._service.get_vram_info(),
                recent_errors=self._debug_console.recent_errors()
                if hasattr(self, "_debug_console")
                else None,
                parent=self,
            )
            dlg.exec()
        else:
            QMessageBox.critical(self, "Processing Error", f"Clip: {clip_name}\n\n{error_msg}")

    @Slot()
    def _on_queue_empty(self) -> None:
        if self._service.job_queue.has_pending:
            return
        self._cancel_requested_job_id = None
        self._force_stop_armed = False
        self._status_bar.set_stop_button_mode(force=False)
        self._status_bar.set_running(False)
        self._status_bar.stop_job_timer()
        self._active_job_id = None
        self._pending_live_asset_refresh_clip = None
        self._live_asset_refresh_timer.stop()
        self._pipeline_steps.clear()
        self._queue_panel.refresh()
        # NOTE: Do NOT call unload_engines() here — it kills the inference
        # engine that live preview depends on.  Model switching is already
        # handled by _ensure_model() when a different model type is needed.
        logger.info("All jobs completed, queue idle")

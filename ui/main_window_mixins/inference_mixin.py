from __future__ import annotations

import logging
import sys

import numpy as np
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import Slot, QThread
from PySide6.QtGui import QImage

from backend import (
    ClipState, JobType,
    PipelineRoute, classify_pipeline_route,
)
from ui.preview.frame_index import ViewMode
from ui.preview.display_transform import processed_rgba_to_qimage
from ui.workers.gpu_job_worker import create_job_snapshot

logger = logging.getLogger(__name__)


class InferenceMixin:
    """Inference control, parameter changes, and GPU job submission for MainWindow."""

    @Slot()
    def _on_params_changed(self) -> None:
        """Handle parameter change — debounce before reprocess."""
        self._remember_current_clip_input_color_space()
        self._dual_viewer.set_input_exr_is_linear(
            self._param_panel.get_params().input_is_linear
        )
        if self._current_clip is not None:
            self._refresh_input_thumbnail(
                self._current_clip,
                input_is_linear=self._param_panel.get_params().input_is_linear,
            )
        if self._param_panel.live_preview_enabled:
            self._reprocess_timer.start()

    @Slot(str)
    def _on_output_mode_changed(self, mode_value: str) -> None:
        """Keep export-strip thumbnails aligned with the current output viewer mode."""
        try:
            ViewMode(mode_value)
        except ValueError:
            return

        for clip in self._clip_model.clips:
            if clip.state == ClipState.COMPLETE:
                self._refresh_export_thumbnail(clip)

    @Slot(bool)
    def _on_live_preview_toggled(self, checked: bool) -> None:
        """When live preview is re-enabled, immediately reprocess current frame."""
        if checked:
            self._reprocess_timer.start()

    def _do_reprocess(self) -> None:
        """Submit a PREVIEW_REPROCESS job through the GPU queue (Codex: no bypass)."""
        if self._current_clip is None:
            return
        clip = self._current_clip
        if clip.state not in (ClipState.READY, ClipState.COMPLETE):
            return

        frame_idx = max(0, self._dual_viewer.current_stem_index)
        params = self._param_panel.get_params()
        job = create_job_snapshot(clip, params, job_type=JobType.PREVIEW_REPROCESS)
        job.params["_frame_index"] = frame_idx

        self._service.job_queue.submit(job)
        self._start_worker_if_needed()

    @Slot(str, object)
    def _on_reprocess_result(self, job_id: str, result: object) -> None:
        """Handle live reprocess result — display preview matching current view mode."""
        if not isinstance(result, dict):
            return

        if result.get("kind") == "sam2_preview":
            clip_name = result.get("clip_name")
            if not isinstance(clip_name, str):
                return
            clip = next((c for c in self._clip_model.clips if c.name == clip_name), None)
            if clip is None or self._current_clip is None or self._current_clip.name != clip_name:
                return

            frame_rgb = result.get("frame_rgb")
            mask = result.get("mask")
            if not isinstance(frame_rgb, np.ndarray) or not isinstance(mask, np.ndarray):
                return

            qimg = self._sam2_preview_qimage(frame_rgb, mask)
            self._dual_viewer.show_reprocess_preview(qimg)

            frame_number = int(result.get("frame_index", 0)) + 1
            fill_pct = float(result.get("fill", 0.0)) * 100.0
            reply = QMessageBox.question(
                self,
                "Track Mask Preview",
                f"SAM2 preview on frame {frame_number} covers {fill_pct:.1f}% of the frame.\n\n"
                "If this looks right, continue with full Track Mask.\n"
                "If not, keep painting corrections on this frame and run Track Mask again.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._submit_sam2_track_job(clip)
            else:
                self._status_bar.set_message("Track preview ready. Refine paint strokes and run Track Mask again.")
            return

        if 'comp' not in result:
            return

        mode = self._dual_viewer.current_output_mode

        # INPUT, MASK, and ALPHA are source/guide views, not inference outputs.
        # Don't replace them with a COMP preview under the wrong mode label.
        if mode in (ViewMode.INPUT, ViewMode.MASK, ViewMode.ALPHA):
            return

        # Pick the array that matches current view mode
        if mode == ViewMode.MATTE and 'alpha' in result:
            arr = result['alpha']
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            # Matte display: clamp, light gamma lift, grayscale -> RGB
            display = np.power(np.clip(arr, 0.0, 1.0), 0.85)
            gray8 = (display * 255.0).astype(np.uint8)
            rgb = np.stack([gray8, gray8, gray8], axis=2)
        elif mode == ViewMode.FG and 'fg' in result:
            # FG is already sRGB float [H,W,3]
            rgb = (np.clip(result['fg'], 0.0, 1.0) * 255.0).astype(np.uint8)
        elif mode == ViewMode.PROCESSED and 'processed' in result:
            qimg = processed_rgba_to_qimage(result['processed'])
            self._dual_viewer.show_reprocess_preview(qimg)
            return
        else:
            # Default: COMP (also for INPUT/MASK/ALPHA which don't change on reprocess)
            rgb = (np.clip(result['comp'], 0.0, 1.0) * 255.0).astype(np.uint8)

        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._dual_viewer.show_reprocess_preview(qimg)

    # ── Inference Control ──

    @Slot()
    def _on_run_inference(self) -> None:
        # If multiple clips are selected, dispatch to pipeline
        if self._io_tray.selected_count() > 1:
            self._on_run_pipeline()
            return

        if self._current_clip is None:
            return

        clip = self._current_clip
        if clip.state not in (ClipState.READY, ClipState.COMPLETE):
            QMessageBox.warning(
                self, "Not Ready",
                f"Clip '{clip.name}' is in {clip.state.value} state.\n"
                "Only READY or COMPLETE clips can be processed.",
            )
            return

        # Warn if alpha doesn't cover all input frames
        if clip.alpha_asset and clip.input_asset:
            alpha_count = clip.alpha_asset.frame_count
            input_count = clip.input_asset.frame_count
            if 0 < alpha_count < input_count:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Incomplete Alpha")
                msg.setText(
                    f"Alpha hints cover {alpha_count} of {input_count} frames.\n\n"
                    "You can process the available range, re-run GVM to\n"
                    "regenerate all alpha frames, or cancel."
                )
                btn_process = msg.addButton("Process Available", QMessageBox.AcceptRole)
                btn_rerun = msg.addButton("Re-run GVM", QMessageBox.ActionRole)
                msg.addButton(QMessageBox.Cancel)
                msg.exec()
                clicked = msg.clickedButton()
                if clicked == btn_rerun:
                    # Transition back to RAW and submit GVM job
                    clip.transition_to(ClipState.RAW)
                    self._clip_model.update_clip_state(clip.name, ClipState.RAW)
                    gvm_job = create_job_snapshot(clip, job_type=JobType.GVM_ALPHA)
                    if self._service.job_queue.submit(gvm_job):
                        clip.set_processing(True)
                        self._start_worker_if_needed(
                            gvm_job.id, job_label="GVM Auto",
                        )
                    return
                elif clicked != btn_process:
                    return  # Cancel

        # For COMPLETE clips wanting reprocess, transition back to READY
        if clip.state == ClipState.COMPLETE:
            clip.transition_to(ClipState.READY)

        params = self._param_panel.get_params()
        job = create_job_snapshot(clip, params, resume=False)

        # Store output config in job params
        output_config = self._param_panel.get_output_config()
        job.params["_output_config"] = output_config

        # Pass in/out frame range (GVM always processes full clip)
        if clip.in_out_range:
            job.params["_frame_range"] = (
                clip.in_out_range.in_point,
                clip.in_out_range.out_point,
            )

        if not self._service.job_queue.submit(job):
            QMessageBox.information(self, "Duplicate", f"'{clip.name}' is already queued.")
            return

        self._start_worker_if_needed(job.id)

    @Slot()
    def _on_resume_inference(self) -> None:
        """Resume inference — skip already-processed frames, process full clip."""
        if self._current_clip is None:
            return

        clip = self._current_clip
        if clip.state not in (ClipState.READY, ClipState.COMPLETE):
            return

        # For COMPLETE clips, transition back to READY
        if clip.state == ClipState.COMPLETE:
            clip.transition_to(ClipState.READY)

        params = self._param_panel.get_params()
        job = create_job_snapshot(clip, params, resume=True)

        output_config = self._param_panel.get_output_config()
        job.params["_output_config"] = output_config
        # Resume always processes full clip — no in/out range

        if not self._service.job_queue.submit(job):
            QMessageBox.information(self, "Duplicate", f"'{clip.name}' is already queued.")
            return

        self._start_worker_if_needed(job.id)

    def _refresh_button_state(self) -> None:
        """Update run/resume button state based on current clip."""
        clip = self._current_clip
        batch_count = self._io_tray.selected_count()
        if clip is None:
            self._status_bar.update_button_state(
                can_run=False, has_partial=False, has_in_out=False,
            )
            return
        can_run = clip.state in (ClipState.READY, ClipState.COMPLETE)
        self._status_bar.update_button_state(
            can_run=can_run,
            has_partial=clip.completed_frame_count() > 0,
            has_in_out=clip.in_out_range is not None,
            batch_count=batch_count if batch_count > 1 else 0,
        )

    @Slot()
    def _on_run_all_ready(self) -> None:
        """Queue all READY clips for inference."""
        ready_clips = self._clip_model.clips_by_state(ClipState.READY)
        if not ready_clips:
            QMessageBox.information(self, "No Clips", "No READY clips to process.")
            return

        params = self._param_panel.get_params()
        output_config = self._param_panel.get_output_config()
        queued = 0
        for clip in ready_clips:
            job = create_job_snapshot(clip, params)
            job.params["_output_config"] = output_config
            if clip.in_out_range:
                job.params["_frame_range"] = (
                    clip.in_out_range.in_point,
                    clip.in_out_range.out_point,
                )
            if self._service.job_queue.submit(job):
                queued += 1

        if queued > 0:
            first_job = self._service.job_queue.next_job()
            self._start_worker_if_needed(first_job.id if first_job else None)
            logger.info(f"Batch queued: {queued} clips")

    def _on_run_pipeline(self) -> None:
        """Full pipeline: classify each selected clip and queue appropriate jobs.

        Routes per clip (see PipelineRoute):
        - INFERENCE_ONLY: queue inference directly
        - GVM_PIPELINE: queue GVM, auto-chain inference on completion
        - VIDEOMAMA_PIPELINE: track dense masks, queue VideoMaMa, auto-chain inference
        - VIDEOMAMA_INFERENCE: queue VideoMaMa, auto-chain inference
        - SKIP: skip clips in EXTRACTING/ERROR state
        """
        selected = self._io_tray.get_selected_clips()
        if not selected:
            return
        self._auto_save_annotations()

        params = self._param_panel.get_params()
        output_config = self._param_panel.get_output_config()

        # Classify all clips
        routes: dict[str, PipelineRoute] = {}
        for clip in selected:
            route = classify_pipeline_route(clip)
            if route != PipelineRoute.SKIP:
                routes[clip.name] = route

        if not routes:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Nothing to Process",
                "No selected clips are in a processable state.",
            )
            return

        self._pipeline_steps.clear()

        queued = 0
        first_job_id = None

        for clip in selected:
            route = routes.get(clip.name)
            if route is None:
                continue
            job = None
            next_steps: list[JobType] = []

            if route == PipelineRoute.GVM_PIPELINE:
                job = create_job_snapshot(clip, job_type=JobType.GVM_ALPHA)
                next_steps = [JobType.INFERENCE]
            elif route == PipelineRoute.VIDEOMAMA_PIPELINE:
                job = create_job_snapshot(clip, params, job_type=JobType.SAM2_TRACK)
                next_steps = [JobType.VIDEOMAMA_ALPHA, JobType.INFERENCE]
            elif route == PipelineRoute.VIDEOMAMA_INFERENCE:
                job = create_job_snapshot(clip, job_type=JobType.VIDEOMAMA_ALPHA)
                next_steps = [JobType.INFERENCE]
            elif route == PipelineRoute.INFERENCE_ONLY:
                if clip.state == ClipState.COMPLETE:
                    clip.transition_to(ClipState.READY)
                job = create_job_snapshot(clip, params)
                job.params["_output_config"] = output_config
                if clip.in_out_range:
                    job.params["_frame_range"] = (
                        clip.in_out_range.in_point,
                        clip.in_out_range.out_point,
                    )

            if job is None:
                continue
            if self._service.job_queue.submit(job):
                clip.set_processing(True)
                if next_steps:
                    self._pipeline_steps[clip.name] = next_steps
                if first_job_id is None:
                    first_job_id = job.id
                queued += 1

        if queued > 0:
            self._start_worker_if_needed(first_job_id, job_label="Pipeline")
            gvm_n = sum(1 for r in routes.values() if r == PipelineRoute.GVM_PIPELINE)
            track_n = sum(1 for r in routes.values() if r == PipelineRoute.VIDEOMAMA_PIPELINE)
            vm_n = sum(1 for r in routes.values() if r == PipelineRoute.VIDEOMAMA_INFERENCE)
            inf_n = sum(1 for r in routes.values() if r == PipelineRoute.INFERENCE_ONLY)
            logger.info(
                f"Pipeline queued: {gvm_n} GVM + {track_n} Track Mask + {vm_n} VideoMaMa + "
                f"{inf_n} inference = {queued} initial jobs "
                f"(+{sum(len(v) for v in self._pipeline_steps.values())} auto-chain pending)"
            )

    def _start_worker_if_needed(
        self,
        first_job_id: str | None = None,
        job_label: str = "Inference",
    ) -> None:
        """Ensure GPU worker is running and show queue panel."""
        if first_job_id and self._active_job_id is None:
            self._active_job_id = first_job_id

        if not self._gpu_worker.isRunning():
            self._gpu_worker.start(QThread.HighPriority)
        else:
            self._gpu_worker.wake()

        self._force_stop_armed = False
        self._status_bar.set_running(True)
        self._status_bar.set_stop_button_mode(force=False)
        self._status_bar.reset_progress()
        self._status_bar.start_job_timer(label=job_label)
        self._queue_panel.refresh()
        if self._queue_panel._collapsed:
            self._queue_panel.toggle_collapsed()  # auto-expand when job starts

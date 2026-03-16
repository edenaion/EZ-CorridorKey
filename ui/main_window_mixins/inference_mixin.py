from __future__ import annotations

import glob as glob_module
import logging
import os
import shutil
import sys

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox, QFileDialog
from PySide6.QtCore import Slot, QThread
from PySide6.QtGui import QImage

from backend import (
    ClipAsset, ClipEntry, ClipState, JobType,
    PipelineRoute, classify_pipeline_route,
)
from backend.project import VIDEO_FILE_FILTER, is_video_file
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

    @Slot()
    def _on_import_alpha(self) -> None:
        """Import user-provided alpha hints into AlphaHint/*.png.

        Image folders and alpha videos are both normalized into 8-bit PNG
        frames named to match input frame stems so index-based matching in
        the inference loop works correctly (frame 0 -> frame 0, etc.).
        """
        from ui.main_window import _remove_alpha_hint_assets, _import_alpha_video_as_sequence, _Toast

        clip = self._current_clip
        if clip is None or clip.state not in (ClipState.RAW, ClipState.MASKED, ClipState.READY):
            return
        if clip.input_asset is None:
            return

        # If AlphaHint already exists, ask before replacing
        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        alpha_video_candidates = [
            c for c in glob_module.glob(os.path.join(clip.root_path, "AlphaHint.*"))
            if os.path.isfile(c) and is_video_file(c)
        ]
        has_existing_alpha = (
            (os.path.isdir(alpha_dir) and os.listdir(alpha_dir))
            or bool(alpha_video_candidates)
        )
        if has_existing_alpha:
            result = QMessageBox.question(
                self, "Replace Alpha Hints?",
                f"Clip '{clip.name}' already has alpha hint images.\n\n"
                "Do you want to replace them with new ones?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                return

        picker = QMessageBox(self)
        picker.setWindowTitle("Import Alpha")
        picker.setText("Import alpha from an image folder or a video file?")
        folder_btn = picker.addButton("Image Folder", QMessageBox.AcceptRole)
        video_btn = picker.addButton("Video File", QMessageBox.ActionRole)
        picker.addButton(QMessageBox.Cancel)
        picker.setDefaultButton(folder_btn)
        picker.exec()

        source_kind: str | None = None
        source_path = ""
        clicked = picker.clickedButton()
        if clicked == folder_btn:
            source_path = QFileDialog.getExistingDirectory(
                self, "Select Alpha Hint Folder",
                "",
                QFileDialog.ShowDirsOnly,
            )
            if source_path:
                source_kind = "folder"
        elif clicked == video_btn:
            source_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Alpha Hint Video",
                "",
                VIDEO_FILE_FILTER,
            )
            if source_path:
                source_kind = "video"

        if not source_kind or not source_path:
            return

        n_src = 0
        src_files: list[str] = []

        if source_kind == "folder":
            # Find image files in the selected folder (natural/numeric sort)
            patterns = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.exr")
            for pat in patterns:
                src_files.extend(glob_module.glob(os.path.join(source_path, pat)))

            if not src_files:
                QMessageBox.warning(
                    self, "No Images",
                    "No image files found in the selected folder.\n"
                    "Expected grayscale images (white=foreground, black=background).",
                )
                return

            n_src = len(src_files)
        else:
            alpha_video = ClipAsset(source_path, "video")
            n_src = alpha_video.frame_count
            if n_src <= 0:
                QMessageBox.warning(
                    self, "Unreadable Video",
                    "Could not read frame count from the selected alpha video.",
                )
                return

        import re as re_module

        def _natural_key(path: str):
            """Sort key that handles any zero-padding scheme correctly."""
            name = os.path.basename(path)
            return [int(c) if c.isdigit() else c.lower()
                    for c in re_module.split(r'(\d+)', name)]

        src_files.sort(key=_natural_key)

        # Get input frame stems for renaming
        input_files = clip.input_asset.get_frame_files()
        n_input = len(input_files)

        if n_src != n_input:
            result = QMessageBox.warning(
                self, "Frame Count Mismatch",
                f"Clip '{clip.name}' has {n_input} input frames but you "
                f"selected {n_src} alpha hints.\n\n"
                f"Each input frame needs a matching alpha hint.\n"
                f"Only {min(n_src, n_input)} frames will be paired.",
                QMessageBox.Ok | QMessageBox.Cancel,
            )
            if result == QMessageBox.Cancel:
                return

        # Confirm import
        n_paired = min(n_src, n_input)
        if source_kind == "video":
            msg = (
                f"Import alpha video ({n_src} frames) into '{clip.name}'?\n\n"
                "The video will be converted to 8-bit PNG alpha frames in AlphaHint/."
            )
        else:
            msg = f"Import {n_paired} alpha hint images into '{clip.name}'?"
        if n_src != n_input:
            msg += f"\n({abs(n_src - n_input)} frames will have no alpha hint)"
        if QMessageBox.question(self, "Import Alpha", msg) != QMessageBox.Yes:
            return

        imported_count = 0
        try:
            _remove_alpha_hint_assets(clip.root_path)
            alpha_dir = os.path.join(clip.root_path, "AlphaHint")

            if source_kind == "video":
                imported_count = _import_alpha_video_as_sequence(
                    source_path,
                    alpha_dir,
                    input_files[:n_paired],
                )
                if imported_count == 0:
                    shutil.rmtree(alpha_dir, ignore_errors=True)
                logger.info(
                    "Imported %d/%d alpha frames from video into %s",
                    imported_count, n_paired, alpha_dir,
                )
            else:
                os.makedirs(alpha_dir, exist_ok=True)

                for i in range(n_paired):
                    src_path = src_files[i]
                    input_stem = os.path.splitext(input_files[i])[0]
                    dst_path = os.path.join(alpha_dir, f"{input_stem}.png")

                    src_ext = os.path.splitext(src_path)[1].lower()
                    if src_ext == ".png":
                        shutil.copy2(src_path, dst_path)
                        imported_count += 1
                        continue

                    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None and cv2.imwrite(dst_path, img):
                        imported_count += 1
                    else:
                        logger.warning("Failed to import alpha image: %s", src_path)

                if imported_count == 0:
                    shutil.rmtree(alpha_dir, ignore_errors=True)
                logger.info(
                    "Imported %d/%d alpha hints into %s (renamed to match input stems)",
                    imported_count, n_paired, alpha_dir,
                )
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Import Alpha Failed",
                f"Failed to import alpha hints:\n{exc}",
            )
            return

        # Refresh clip state
        clip.find_assets()
        self._io_tray.refresh()

        # Reload preview and button states
        if self._current_clip and self._current_clip.name == clip.name:
            self._sync_selected_clip_view(clip)
            self._refresh_button_state()
            self._param_panel.set_import_alpha_enabled(
                clip.state in (ClipState.RAW, ClipState.MASKED, ClipState.READY)
            )

        if source_kind == "video":
            toast_msg = (
                f"Imported {imported_count}/{n_paired} alpha frames from video.\n"
                f"Clip is now {clip.state.value}."
            )
        else:
            toast_msg = (
                f"Imported {imported_count}/{n_paired} alpha hints.\n"
                f"Clip is now {clip.state.value}."
            )
        _Toast(self, toast_msg)

    def _on_run_gvm(self) -> None:
        """Run GVM alpha generation on the selected clip."""
        if self._current_clip is None or self._current_clip.state not in (ClipState.RAW, ClipState.MASKED):
            return

        if not self._warn_mps_slow("GVM Auto Alpha"):
            return

        # Detect partial alpha from a previous interrupted run
        alpha_dir = os.path.join(self._current_clip.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir):
            existing = [f for f in os.listdir(alpha_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if existing:
                total = (self._current_clip.input_asset.frame_count
                         if self._current_clip.input_asset else 0)
                msg = QMessageBox(self)
                msg.setWindowTitle("Partial Alpha Found")
                msg.setText(
                    f"Found {len(existing)}/{total} alpha frames from a previous run."
                )
                msg.setInformativeText(
                    "Resume will skip completed frames.\n"
                    "Regenerate will redo all frames from scratch."
                )
                resume_btn = msg.addButton("Resume", QMessageBox.AcceptRole)
                regen_btn = msg.addButton("Regenerate", QMessageBox.DestructiveRole)
                msg.addButton(QMessageBox.Cancel)
                msg.setDefaultButton(resume_btn)
                msg.exec()
                clicked = msg.clickedButton()
                if clicked == regen_btn:
                    shutil.rmtree(alpha_dir, ignore_errors=True)
                elif clicked != resume_btn:
                    return  # cancelled

        job = create_job_snapshot(self._current_clip, job_type=JobType.GVM_ALPHA)
        if not self._service.job_queue.submit(job):
            return

        self._current_clip.set_processing(True)
        self._start_worker_if_needed(job.id, job_label="GVM Auto")

    @Slot(str)
    def _on_run_birefnet(self, usage: str) -> None:
        """Run BiRefNet alpha generation on the selected clip."""
        if self._current_clip is None or self._current_clip.state not in (ClipState.RAW, ClipState.MASKED):
            return

        # Detect partial alpha from a previous interrupted run
        alpha_dir = os.path.join(self._current_clip.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir):
            existing = [f for f in os.listdir(alpha_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if existing:
                total = (self._current_clip.input_asset.frame_count
                         if self._current_clip.input_asset else 0)
                msg = QMessageBox(self)
                msg.setWindowTitle("Partial Alpha Found")
                msg.setText(
                    f"Found {len(existing)}/{total} alpha frames from a previous run."
                )
                msg.setInformativeText(
                    "Resume will skip completed frames.\n"
                    "Regenerate will redo all frames from scratch."
                )
                resume_btn = msg.addButton("Resume", QMessageBox.AcceptRole)
                regen_btn = msg.addButton("Regenerate", QMessageBox.DestructiveRole)
                msg.addButton(QMessageBox.Cancel)
                msg.setDefaultButton(resume_btn)
                msg.exec()
                clicked = msg.clickedButton()
                if clicked == regen_btn:
                    shutil.rmtree(alpha_dir, ignore_errors=True)
                elif clicked != resume_btn:
                    return  # cancelled

        job = create_job_snapshot(
            self._current_clip,
            job_type=JobType.BIREFNET_ALPHA,
            birefnet_usage=usage,
        )
        if not self._service.job_queue.submit(job):
            return

        self._current_clip.set_processing(True)
        self._start_worker_if_needed(job.id, job_label=f"BiRefNet ({usage})")

    @Slot()
    def _on_run_videomama(self) -> None:
        """Run VideoMaMa alpha generation on the selected clip."""
        if self._current_clip is None:
            return
        if not self._clip_has_videomama_ready_mask(self._current_clip):
            QMessageBox.information(
                self,
                "Track Mask First",
                "Paint prompts and run Track Mask before using VideoMaMa.",
            )
            return

        if not self._warn_mps_slow("VideoMaMa Auto Alpha"):
            return

        job = create_job_snapshot(self._current_clip, job_type=JobType.VIDEOMAMA_ALPHA)
        if not self._service.job_queue.submit(job):
            return

        self._current_clip.set_processing(True)
        self._start_worker_if_needed(job.id, job_label="VideoMaMa")

    @Slot()
    def _on_run_matanyone2(self) -> None:
        """Run MatAnyone2 video matting alpha generation on the selected clip."""
        if self._current_clip is None:
            return
        if not self._clip_has_videomama_ready_mask(self._current_clip):
            QMessageBox.information(
                self,
                "Track Mask First",
                "MatAnyone2 requires a tracked mask on frame 0.\n\n"
                "Paint prompts and run Track Mask before using MatAnyone2.",
            )
            return

        if not self._warn_mps_slow("MatAnyone2"):
            return

        job = create_job_snapshot(self._current_clip, job_type=JobType.MATANYONE2_ALPHA)
        if not self._service.job_queue.submit(job):
            return

        self._current_clip.set_processing(True)
        self._start_worker_if_needed(job.id, job_label="MatAnyone2")

    def _start_worker_if_needed(
        self,
        first_job_id: str | None = None,
        job_label: str = "Inference",
    ) -> None:
        """Ensure GPU worker is running and show queue panel."""
        if first_job_id and self._active_job_id is None:
            self._active_job_id = first_job_id

        if not self._gpu_worker.isRunning():
            if sys.platform == "win32":
                self._gpu_worker.start(QThread.LowPriority)
            else:
                self._gpu_worker.start()
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

    def _cancel_extraction(self) -> None:
        """Stop frame extraction and reset the worker."""
        from ui.workers.extract_worker import ExtractWorker

        old = self._extract_worker
        # Disconnect old signals to prevent stale deliveries
        try:
            old.progress.disconnect(self._on_extract_progress)
            old.finished.disconnect(self._on_extract_finished)
            old.error.disconnect(self._on_extract_error)
        except RuntimeError:
            pass  # already disconnected
        old.stop()
        if not old.wait(5000):
            logger.warning("Extract worker did not stop in 5s — terminating")
            old.terminate()
            old.wait(2000)
        # Restart the worker thread (stop kills the thread loop)
        self._extract_worker = ExtractWorker(parent=self)
        self._extract_worker.progress.connect(self._on_extract_progress)
        self._extract_worker.finished.connect(self._on_extract_finished)
        self._extract_worker.error.connect(self._on_extract_error)
        self._extract_progress.clear()
        self._status_bar.reset_progress()
        # Clear extraction overlay on the viewer
        self._dual_viewer.set_extraction_progress(0.0, 0)
        # Reset any EXTRACTING clips back to their original state
        for clip in self._clip_model.clips:
            if clip.state == ClipState.EXTRACTING:
                clip.extraction_progress = 0.0
                clip.extraction_total = 0
                # Check if frames already exist -> RAW, else back to VIDEO
                frames_dir = os.path.join(clip.root_path, "Frames")
                input_dir = os.path.join(clip.root_path, "Input")
                has_frames = (os.path.isdir(frames_dir) and os.listdir(frames_dir)) or \
                             (os.path.isdir(input_dir) and os.listdir(input_dir))
                new_state = ClipState.RAW if has_frames else ClipState.ERROR
                self._clip_model.update_clip_state(clip.name, new_state)
        self._io_tray.refresh()
        self._refresh_button_state()
        logger.info("Frame extraction cancelled by user")

    def _cancel_inference(self) -> None:
        """Cancel all inference/GPU jobs."""
        from ui.main_window import _Toast

        queue = self._service.job_queue
        current_job = queue.current_job
        is_videomama = (queue.current_job
                        and queue.current_job.job_type == JobType.VIDEOMAMA_ALPHA)
        self._cancel_requested_job_id = current_job.id if current_job is not None else None
        queue.cancel_all()
        self._pipeline_steps.clear()
        self._status_bar.stop_job_timer()
        if current_job is not None:
            self._force_stop_armed = True
            self._status_bar.set_running(True)
            self._status_bar.set_stop_button_mode(force=True)
            self._status_bar.set_message(
                "Stop requested — waiting for current GPU step. "
                "Press FORCE STOP to relaunch if it stays stuck."
            )
        else:
            self._force_stop_armed = False
            self._status_bar.set_running(False)
            self._status_bar.set_message("Cancelled queued work.")
        self._queue_panel.refresh()
        logger.info("Processing cancelled by user")
        if is_videomama:
            _Toast(self, "GPU is finishing the current chunk.\n"
                         "VideoMaMa will stop after it completes.",
                   center=True)

    def _force_restart_app(self) -> None:
        """Hard-stop a blocked GPU phase by relaunching the app process."""
        import subprocess
        from PySide6.QtWidgets import QApplication

        try:
            self._auto_save_annotations()
        except Exception:
            logger.exception("Force stop: failed to auto-save annotations")

        try:
            self._auto_save_session()
        except Exception:
            logger.exception("Force stop: failed to auto-save session")

        if getattr(sys, "frozen", False):
            cmd = [sys.executable, *sys.argv[1:]]
            cwd = os.path.dirname(sys.executable)
        else:
            cmd = [sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]]
            cwd = os.path.dirname(os.path.abspath(sys.argv[0]))

        kwargs: dict = {"cwd": cwd}
        if os.name == "nt":
            kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0)
            )
        else:
            kwargs["start_new_session"] = True

        logger.warning("Force stop: relaunching application to break blocked GPU job")
        try:
            subprocess.Popen(cmd, **kwargs)
        except Exception as e:
            logger.exception(f"Force stop relaunch failed: {e}")
            QMessageBox.critical(
                self,
                "Force Stop Failed",
                "Could not relaunch the app automatically.\n\n"
                "Please close and reopen EZ-CorridorKey manually.",
            )
            return

        self._skip_shutdown_cleanup = True
        self._status_bar.set_message("Force restarting...")
        QApplication.instance().quit()

    @Slot()
    def _on_stop_inference(self) -> None:
        """STOP button handler — confirms and cancels inference."""
        if not self._status_bar._stop_btn.isVisible():
            return
        queue = self._service.job_queue
        if not queue.current_job and not queue.has_pending:
            return

        if self._force_stop_armed:
            reply = QMessageBox.question(
                self, "Force Stop",
                "The current GPU step has not returned to Python.\n\n"
                "Force Stop will auto-save the session and relaunch the app "
                "to break the stuck job immediately.\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            self._force_restart_app()
            return

        reply = QMessageBox.question(
            self, "Cancel",
            "Cancel processing?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        from ui.sounds.audio_manager import UIAudio
        UIAudio.user_cancel()
        self._cancel_inference()

    @Slot(str)
    def _on_cancel_job(self, job_id: str) -> None:
        """Cancel a specific job from the queue panel."""
        job = self._service.job_queue.find_job_by_id(job_id)
        if job:
            self._service.job_queue.cancel_job(job)
            self._queue_panel.refresh()

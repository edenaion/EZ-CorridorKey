from __future__ import annotations

import os
import shutil
import logging

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage

from backend import ClipEntry

logger = logging.getLogger(__name__)


class AnnotationMixin:
    """Annotation painting, SAM2 tracking, and mask management for MainWindow."""

    def _toggle_annotation_fg(self) -> None:
        """Hotkey 1: toggle green (foreground) annotation brush."""
        iv = self._dual_viewer.input_viewer
        if iv.annotation_mode == "fg":
            iv.set_annotation_mode(None)
        else:
            iv.set_annotation_mode("fg")

    def _toggle_annotation_bg(self) -> None:
        """Hotkey 2: toggle red (background) annotation brush."""
        iv = self._dual_viewer.input_viewer
        if iv.annotation_mode == "bg":
            iv.set_annotation_mode(None)
        else:
            iv.set_annotation_mode("bg")

    def _cycle_fg_color(self) -> None:
        """Hotkey C: cycle foreground annotation color (green/blue)."""
        from ui.widgets.annotation_overlay import cycle_fg_color

        name = cycle_fg_color()
        self._dual_viewer.input_viewer.update()
        self._show_toast(f"Foreground color: {name}")

    def _auto_save_annotations(self) -> None:
        """Auto-save annotation strokes to disk after changes."""
        if self._current_clip is not None:
            iv = self._dual_viewer.input_viewer
            iv.annotation_model.save(self._current_clip.root_path)
            manifest_path = os.path.join(
                self._current_clip.root_path,
                ".corridorkey_mask_manifest.json",
            )
            if os.path.isfile(manifest_path):
                os.remove(manifest_path)
            _has_mask = self._clip_has_videomama_ready_mask(self._current_clip)
            self._param_panel.set_videomama_enabled(_has_mask)
            self._param_panel.set_matanyone2_enabled(_has_mask)

    def _clip_has_videomama_ready_mask(self, clip: ClipEntry | None) -> bool:
        """True when the clip has a dense mask track that alpha generators can consume."""
        if clip is None or clip.mask_asset is None:
            return False
        # mask_asset is set only when VideoMamaMaskHint/ has actual frames —
        # that's sufficient to enable alpha generators regardless of manifest.
        return True

    def _undo_annotation(self) -> None:
        """Ctrl+Z: undo last annotation stroke on current frame."""
        iv = self._dual_viewer.input_viewer
        if iv.annotation_mode and iv.current_stem_index >= 0:
            if iv.annotation_model.undo(iv.current_stem_index):
                iv._split_view.update()
                self._auto_save_annotations()

    def _on_track_masks(self) -> None:
        """Preview SAM2 on the annotated frame, then confirm full tracking."""
        from ui.workers.gpu_job_worker import create_job_snapshot
        from backend import JobType

        clip = self._current_clip
        if clip is None:
            return
        self._auto_save_annotations()
        iv = self._dual_viewer.input_viewer
        model = iv.annotation_model
        if not model.has_annotations():
            QMessageBox.information(
                self,
                "No Paint Strokes",
                "Paint green (1) and red (2) strokes on frames first.",
            )
            return

        if not self._warn_mps_slow("SAM2 Track Mask"):
            return

        job = create_job_snapshot(
            clip, self._param_panel.get_params(), job_type=JobType.SAM2_PREVIEW
        )
        job.params["_frame_index"] = max(0, self._dual_viewer.current_stem_index)
        if not self._service.job_queue.submit(job):
            return

        self._start_worker_if_needed(job.id, job_label="Track Preview")

    def _submit_sam2_track_job(self, clip: ClipEntry) -> bool:
        """Queue the full SAM2 tracking job after preview confirmation."""
        from ui.workers.gpu_job_worker import create_job_snapshot
        from backend import JobType

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir):
            reply = QMessageBox.question(
                self,
                "Replace Existing Alpha?",
                "This clip already has an AlphaHint (from GVM or a previous run).\n\n"
                "Tracking a new mask sequence will replace that alpha hint.\n\n"
                "Remove existing AlphaHint and proceed?",
            )
            if reply != QMessageBox.Yes:
                return False
            shutil.rmtree(alpha_dir, ignore_errors=True)
            clip.alpha_asset = None
            clip.find_assets()
            self._refresh_button_state()
            logger.info("Removed existing AlphaHint/ before SAM2 tracking")

        job = create_job_snapshot(clip, self._param_panel.get_params(), job_type=JobType.SAM2_TRACK)
        if not self._service.job_queue.submit(job):
            return False

        clip.set_processing(True)
        self._start_worker_if_needed(job.id, job_label="Track Mask")
        return True

    @staticmethod
    def _sam2_preview_qimage(frame_rgb: np.ndarray, mask: np.ndarray) -> QImage:
        """Render a contour-only SAM2 preview for the output viewer."""
        overlay = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (0, 230, 255), 4)
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()

    def _confirm_clear_annotations(self) -> None:
        """Ctrl+C: choose to clear annotations on this frame, entire clip, or cancel."""
        iv = self._dual_viewer.input_viewer
        model = iv.annotation_model
        if not model or not model.has_annotations():
            return

        stem_idx = iv._split_view._annotation_stem_idx
        has_frame = model.has_annotations(stem_idx)

        box = QMessageBox(self)
        box.setWindowTitle("Clear Paint Strokes")
        box.setText("What would you like to clear?")
        frame_btn = box.addButton("This Frame", QMessageBox.AcceptRole)
        clip_btn = box.addButton("Entire Clip", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)

        # Disable "This Frame" if current frame has no annotations
        if not has_frame:
            frame_btn.setEnabled(False)

        box.exec()
        clicked = box.clickedButton()

        if clicked == frame_btn:
            model.clear(stem_idx)
            iv._split_view.update()
            self._update_annotation_info()
        elif clicked == clip_btn:
            self._on_clear_annotations()

    def _on_clear_annotations(self) -> None:
        """Clear all annotations on the current clip and remove tracked masks."""
        iv = self._dual_viewer.input_viewer
        iv.clear_annotations()
        self._update_annotation_info()

        # Remove exported mask directory so VideoMaMa button disables
        clip = self._current_clip
        if clip is not None:
            mask_dir = os.path.join(clip.root_path, "VideoMamaMaskHint")
            if os.path.isdir(mask_dir):
                shutil.rmtree(mask_dir, ignore_errors=True)
                logger.info(f"Removed mask hints: {mask_dir}")
            manifest_path = os.path.join(clip.root_path, ".corridorkey_mask_manifest.json")
            if os.path.isfile(manifest_path):
                os.remove(manifest_path)
            clip.mask_asset = None
            clip.find_assets()
            _has_mask = self._clip_has_videomama_ready_mask(clip)
            self._param_panel.set_videomama_enabled(_has_mask)
            self._param_panel.set_matanyone2_enabled(_has_mask)

    def _update_annotation_info(self) -> None:
        """Update parameter panel with current annotation count and scrubber markers."""
        iv = self._dual_viewer.input_viewer
        model = iv.annotation_model
        fi = iv._frame_index
        total = fi.frame_count if fi else 0
        self._param_panel.set_annotation_info(model.annotated_frame_count(), total)
        # Push annotation coverage to the scrubber timeline
        if fi and total > 0:
            annotated = [model.has_annotations(i) for i in range(total)]
            self._dual_viewer._scrubber.set_annotation_markers(annotated)
        else:
            self._dual_viewer._scrubber.set_annotation_markers([])

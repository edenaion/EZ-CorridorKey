from __future__ import annotations

import logging
import os
import shutil

import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QMessageBox

from . import _tr

from backend import ClipState, JobType
from ui.workers.job_helpers import create_job_snapshot

logger = logging.getLogger(__name__)


class ChromaKeyMixin:
    """Chroma key alpha hint generation handlers for MainWindow."""

    _ck_preview_timer: QTimer | None = None

    def _ensure_ck_preview_timer(self) -> QTimer:
        """Lazy-init the chroma key preview debounce timer."""
        if self._ck_preview_timer is None:
            self._ck_preview_timer = QTimer(self)
            self._ck_preview_timer.setSingleShot(True)
            self._ck_preview_timer.setInterval(80)  # 80ms debounce (fast, CPU-only)
            self._ck_preview_timer.timeout.connect(self._do_chroma_key_preview)
        return self._ck_preview_timer

    def _on_run_chroma_key(self, chroma_params: dict) -> None:
        """Generate alpha hints via chroma key for the selected clip."""
        clip = self._current_clip
        if clip is None or clip.state not in (ClipState.RAW, ClipState.MASKED):
            logger.warning("Chroma key generate: no clip or wrong state")
            return

        # Resolve "auto" screen type from clip's detected color
        if chroma_params.get("screen_type") == "auto":
            chroma_params["screen_type"] = getattr(clip, '_screen_color_cache', None) or "green"

        logger.info(f"Chroma key generate for '{clip.name}': {chroma_params}")

        # Check for existing alpha hints (reuse the same pattern as GVM/BiRefNet)
        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir) and os.listdir(alpha_dir):
            result = QMessageBox.question(
                self, _tr("Replace Alpha Hints?"),
                _tr("Clip '%s' already has alpha hint images.\n\n"
                    "Do you want to replace them with chroma key hints?") % clip.name,
                QMessageBox.Yes | QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                return
            shutil.rmtree(alpha_dir, ignore_errors=True)

        job = create_job_snapshot(
            clip,
            job_type=JobType.CHROMA_KEY_ALPHA,
            chroma_params=chroma_params,
        )
        if not self._service.job_queue.submit(job):
            return

        clip.set_processing(True)
        self._start_worker_if_needed(job.id, job_label="Chroma Key")

    def _on_eyedropper_toggle(self, enabled: bool) -> None:
        """Toggle eyedropper mode on both viewports."""
        logger.debug(f"Eyedropper mode: {enabled}")
        self._dual_viewer.set_eyedropper_mode(enabled)

    def _on_color_sampled(self, r: int, g: int, b: int) -> None:
        """Handle a sampled screen color from the eyedropper."""
        logger.info(f"Eyedropper sampled color: ({r}, {g}, {b})")
        # set_sampled_screen_color unchecks the eyedropper button,
        # which triggers _on_eyedropper_toggled(False) -> set_eyedropper_mode(False)
        # through the signal chain. No need to call set_eyedropper_mode directly.
        self._param_panel.set_sampled_screen_color(r, g, b)
        # Trigger preview with the new color and persist
        self._schedule_chroma_key_preview()
        self._save_chroma_params()

    def _on_chroma_key_param_changed(self) -> None:
        """Any chroma key parameter changed - schedule a preview update and save."""
        self._schedule_chroma_key_preview()
        self._save_chroma_params()

    def _save_chroma_params(self) -> None:
        """Persist current chroma key params to clip.json."""
        clip = self._current_clip
        if clip is None:
            return
        from backend.project import save_chroma_params
        params = self._param_panel.get_chroma_params()
        save_chroma_params(clip.root_path, params)

    def _load_chroma_params_for_clip(self, clip) -> None:
        """Load saved chroma key params from clip.json and apply to UI."""
        from backend.project import load_chroma_params
        params = load_chroma_params(clip.root_path)
        if params:
            self._param_panel.set_chroma_params(params)
        else:
            self._param_panel.reset_chroma_params()

    def _schedule_chroma_key_preview(self) -> None:
        """Debounced chroma key preview on the current frame."""
        if not self._param_panel._chroma_key_btn.isChecked():
            return
        self._ensure_ck_preview_timer().start()

    def _do_chroma_key_preview(self) -> None:
        """Run chroma key on the current frame and show result in B viewer."""
        from CorridorKeyModule.core.chroma_key import chroma_key_matte

        clip = self._current_clip
        if clip is None or clip.input_asset is None:
            logger.debug("Chroma key preview: no clip or input asset")
            return
        if not self._param_panel._chroma_key_btn.isChecked():
            return

        # Get the current frame path from the input viewer
        iv = self._dual_viewer.input_viewer
        fi = iv._frame_index
        if fi is None or fi.frame_count == 0:
            logger.debug("Chroma key preview: no frame index")
            return
        stem_idx = iv.current_stem_index
        if stem_idx < 0 or stem_idx >= fi.frame_count:
            logger.debug(f"Chroma key preview: invalid stem_idx={stem_idx}")
            return

        from ui.preview.frame_index import ViewMode
        input_path = fi.get_path(ViewMode.INPUT, stem_idx)
        if input_path is None or not os.path.isfile(input_path):
            logger.debug(f"Chroma key preview: no input path for stem {stem_idx}, "
                         f"path={input_path}")
            return

        logger.debug(f"Chroma key preview: reading {input_path}")

        # Read frame (handle EXR float data properly)
        is_exr = input_path.lower().endswith(".exr")
        if is_exr:
            frame_bgr = cv2.imread(input_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        else:
            frame_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            logger.warning(f"Chroma key preview: cv2.imread failed for {input_path}")
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # EXR comes as float32 in 0-1 range; convert to uint8 for display
        if frame_rgb.dtype != np.uint8:
            frame_rgb = (np.clip(frame_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Resolve screen type: use auto-detected color if BG Color is "auto"
        params = self._param_panel.get_chroma_params()
        screen_type = params.get("screen_type", "green")
        if screen_type == "auto" and hasattr(clip, '_screen_color_cache'):
            screen_type = clip._screen_color_cache or "green"

        logger.debug(f"Chroma key preview: screen_type={screen_type}, "
                     f"screen_color={params.get('screen_color')}, "
                     f"strength={params.get('strength')}")

        # Run chroma key with current params
        matte = chroma_key_matte(
            frame_rgb,
            screen_color=params.get("screen_color"),
            screen_type=screen_type,
            strength=params.get("strength", 1.0),
            clip_black=params.get("clip_black", 0.0),
            clip_white=params.get("clip_white", 1.0),
            shrink_grow=params.get("shrink_grow", 0),
            edge_blur=params.get("edge_blur", 0),
        )

        logger.debug(f"Chroma key preview: matte shape={matte.shape}, "
                     f"min={matte.min()}, max={matte.max()}")

        # Convert grayscale matte to RGB QImage for display
        h, w = matte.shape
        rgb = np.stack([matte, matte, matte], axis=2).copy()
        qimage = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()

        self._dual_viewer.show_reprocess_preview(qimage)
        logger.debug("Chroma key preview: displayed")

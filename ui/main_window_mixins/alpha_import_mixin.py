from __future__ import annotations

import glob as glob_module
import logging
import os
import shutil

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox, QFileDialog

from . import _tr

from backend import ClipAsset, ClipState
from backend.project import VIDEO_FILE_FILTER, is_video_file
from backend.frame_io import imread_unicode, imwrite_unicode

logger = logging.getLogger(__name__)


class AlphaImportMixin:
    """Alpha import dialog and validation pipeline for MainWindow."""

    def _on_import_vmama_mask(self) -> None:
        """Import masks directly into VideoMamaMaskHint/, bypassing SAM2 tracking.

        Sets a flag so the shared import flow writes to VideoMamaMaskHint/
        instead of AlphaHint/. The clip transitions to MASKED, ready for
        VideoMaMa to refine.
        """
        self._import_as_vmama_mask = True
        self._on_import_alpha()

    def _on_import_alpha(self) -> None:
        """Import user-provided alpha hints into AlphaHint.

        Image folders are renamed to match input frame stems. Final AlphaHint
        imports preserve EXR files as EXR; video imports and VideoMaMa mask
        imports are normalized into 8-bit PNG frame sequences.
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
                self, _tr("Replace Alpha Hints?"),
                _tr("Clip '%s' already has alpha hint images.\n\n"
                    "Do you want to replace them with new ones?") % clip.name,
                QMessageBox.Yes | QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                return

        picker = QMessageBox(self)
        picker.setWindowTitle(_tr("Import Alpha"))
        picker.setText(_tr("Import alpha from an image folder or a video file?"))
        folder_btn = picker.addButton(_tr("Image Folder"), QMessageBox.AcceptRole)
        video_btn = picker.addButton(_tr("Video File"), QMessageBox.ActionRole)
        picker.addButton(QMessageBox.Cancel)
        picker.setDefaultButton(folder_btn)
        picker.exec()

        source_kind: str | None = None
        source_path = ""
        clicked = picker.clickedButton()
        if clicked == folder_btn:
            source_path = QFileDialog.getExistingDirectory(
                self, _tr("Select Alpha Hint Folder"),
                "",
                QFileDialog.ShowDirsOnly,
            )
            if source_path:
                source_kind = "folder"
        elif clicked == video_btn:
            source_path, _ = QFileDialog.getOpenFileName(
                self,
                _tr("Select Alpha Hint Video"),
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
                    self, _tr("No Images"),
                    _tr("No image files found in the selected folder.\n"
                        "Expected grayscale images (white=foreground, black=background)."),
                )
                return

            n_src = len(src_files)
        else:
            alpha_video = ClipAsset(source_path, "video")
            n_src = alpha_video.frame_count
            if n_src <= 0:
                QMessageBox.warning(
                    self, _tr("Unreadable Video"),
                    _tr("Could not read frame count from the selected alpha video."),
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
                self, _tr("Frame Count Mismatch"),
                _tr("Clip '%s' has %d input frames but you "
                    "selected %d alpha hints.\n\n"
                    "Each input frame needs a matching alpha hint.\n"
                    "Only %d frames will be paired.") % (clip.name, n_input, n_src, min(n_src, n_input)),
                QMessageBox.Ok | QMessageBox.Cancel,
            )
            if result == QMessageBox.Cancel:
                return

        import_as_vmama_mask = getattr(self, '_import_as_vmama_mask', False)
        self._import_as_vmama_mask = False  # reset flag

        # Confirm import
        n_paired = min(n_src, n_input)
        target_name = "VideoMamaMaskHint" if import_as_vmama_mask else "AlphaHint"
        if source_kind == "video":
            msg = (
                f"Import alpha video ({n_src} frames) into '{clip.name}'?\n\n"
                f"The video will be converted to 8-bit PNG frames in {target_name}/."
            )
        else:
            exr_count = sum(
                1 for p in src_files[:n_paired]
                if os.path.splitext(p)[1].lower() == ".exr"
            )
            if import_as_vmama_mask or exr_count == 0:
                msg = f"Import {n_paired} alpha images into '{clip.name}' as {target_name}?"
            else:
                msg = (
                    f"Import {n_paired} alpha images into '{clip.name}' as {target_name}?\n\n"
                    f"{exr_count} EXR file(s) will be preserved as EXR."
                )
        if n_src != n_input:
            msg += f"\n({abs(n_src - n_input)} frames will have no alpha)"
        if QMessageBox.question(self, _tr("Import Alpha"), msg) != QMessageBox.Yes:
            return

        imported_count = 0
        try:
            if import_as_vmama_mask:
                # Clean existing mask hint directory
                mask_dir = os.path.join(clip.root_path, "VideoMamaMaskHint")
                if os.path.isdir(mask_dir):
                    for f in os.listdir(mask_dir):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            os.remove(os.path.join(mask_dir, f))
            else:
                _remove_alpha_hint_assets(clip.root_path)
            alpha_dir = os.path.join(clip.root_path, target_name)

            if source_kind == "video":
                imported_count = _import_alpha_video_as_sequence(
                    source_path,
                    alpha_dir,
                    input_files[:n_paired],
                )
                if imported_count == 0:
                    shutil.rmtree(alpha_dir, ignore_errors=True)
                logger.info(
                    "Imported %d/%d alpha frames from video %s into %s",
                    imported_count, n_paired, source_path, alpha_dir,
                )
            else:
                os.makedirs(alpha_dir, exist_ok=True)
                preserved_exr_count = 0

                for i in range(n_paired):
                    src_path = src_files[i]
                    input_stem = os.path.splitext(input_files[i])[0]

                    src_ext = os.path.splitext(src_path)[1].lower()
                    dst_ext = ".png"
                    if not import_as_vmama_mask and src_ext == ".exr":
                        dst_ext = ".exr"
                    dst_path = os.path.join(alpha_dir, f"{input_stem}{dst_ext}")

                    if not import_as_vmama_mask and src_ext == ".png":
                        # Fast path: copy PNG as-is for final alpha
                        shutil.copy2(src_path, dst_path)
                        imported_count += 1
                        continue

                    if not import_as_vmama_mask and src_ext == ".exr":
                        # Preserve float EXR alpha hints instead of quantizing to PNG.
                        img = imread_unicode(
                            src_path,
                            cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED,
                        )
                        if img is None:
                            logger.warning("Failed to import alpha EXR: %s", src_path)
                            continue
                        shutil.copy2(src_path, dst_path)
                        imported_count += 1
                        preserved_exr_count += 1
                        continue

                    if src_ext == ".exr":
                        img = imread_unicode(
                            src_path,
                            cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED,
                        )
                        if img is not None:
                            if img.ndim == 3:
                                img = img[:, :, 0]
                            if img.dtype != np.uint8:
                                img = np.clip(
                                    img.astype(np.float32), 0.0, 1.0,
                                )
                                img = (img * 255.0).astype(np.uint8)
                    else:
                        img = imread_unicode(src_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        logger.warning("Failed to import alpha image: %s", src_path)
                        continue

                    # VideoMaMa needs binary masks (0 or 255).
                    # Crush any grayscale values to pure black/white.
                    if import_as_vmama_mask:
                        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

                    if imwrite_unicode(dst_path, img):
                        imported_count += 1
                    else:
                        logger.warning("Failed to write alpha image: %s", dst_path)

                if imported_count == 0:
                    shutil.rmtree(alpha_dir, ignore_errors=True)
                logger.info(
                    "Imported %d/%d alpha hints from %s into %s "
                    "(renamed to match input stems, preserved %d EXR)",
                    imported_count, n_paired, source_path, alpha_dir,
                    preserved_exr_count,
                )
        except OSError as exc:
            QMessageBox.critical(
                self,
                _tr("Import Alpha Failed"),
                _tr("Failed to import alpha hints:\n%s") % exc,
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

        from ui.state_labels import state_display_name
        label = _tr("VideoMaMa masks") if import_as_vmama_mask else _tr("alpha hints")
        state_label = state_display_name(clip.state)
        if source_kind == "video":
            toast_msg = (
                _tr("Imported %d/%d %s from video.\nClip is now %s.")
                % (imported_count, n_paired, label, state_label)
            )
        else:
            toast_msg = (
                _tr("Imported %d/%d %s.\nClip is now %s.")
                % (imported_count, n_paired, label, state_label)
            )
        _Toast(self, toast_msg)

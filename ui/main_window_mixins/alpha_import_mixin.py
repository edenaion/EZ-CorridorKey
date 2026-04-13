from __future__ import annotations

import glob as glob_module
import logging
import os
import shutil

import cv2
from PySide6.QtWidgets import QMessageBox, QFileDialog

from backend import ClipAsset, ClipState
from backend.project import VIDEO_FILE_FILTER, is_video_file

logger = logging.getLogger(__name__)


class AlphaImportMixin:
    """Alpha import dialog and validation pipeline for MainWindow."""

    def _on_import_alpha(self) -> None:
        """Import user-provided alpha hints into AlphaHint/*.png.

        Image folders and alpha videos are both normalized into 8-bit PNG
        frames named to match input frame stems so index-based matching in
        the inference loop works correctly (frame 0 -> frame 0, etc.).
        """
        from ui.main_window import (
            _remove_alpha_hint_assets,
            _import_alpha_video_as_sequence,
            _Toast,
        )

        clip = self._current_clip
        if clip is None or clip.state not in (ClipState.RAW, ClipState.MASKED, ClipState.READY):
            return
        if clip.input_asset is None:
            return

        # If AlphaHint already exists, ask before replacing
        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        alpha_video_candidates = [
            c
            for c in glob_module.glob(os.path.join(clip.root_path, "AlphaHint.*"))
            if os.path.isfile(c) and is_video_file(c)
        ]
        has_existing_alpha = (os.path.isdir(alpha_dir) and os.listdir(alpha_dir)) or bool(
            alpha_video_candidates
        )
        if has_existing_alpha:
            result = QMessageBox.question(
                self,
                "Replace Alpha Hints?",
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
                self,
                "Select Alpha Hint Folder",
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
                    self,
                    "No Images",
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
                    self,
                    "Unreadable Video",
                    "Could not read frame count from the selected alpha video.",
                )
                return

        import re as re_module

        def _natural_key(path: str):
            """Sort key that handles any zero-padding scheme correctly."""
            name = os.path.basename(path)
            return [int(c) if c.isdigit() else c.lower() for c in re_module.split(r"(\d+)", name)]

        src_files.sort(key=_natural_key)

        # Get input frame stems for renaming
        input_files = clip.input_asset.get_frame_files()
        n_input = len(input_files)

        if n_src != n_input:
            result = QMessageBox.warning(
                self,
                "Frame Count Mismatch",
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
                    imported_count,
                    n_paired,
                    alpha_dir,
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
                    imported_count,
                    n_paired,
                    alpha_dir,
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

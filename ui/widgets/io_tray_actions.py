"""Batch action mixin for IOTrayPanel — context menus, clear/delete/rename ops.

Extracted from io_tray_panel.py for maintainability.
This mixin expects to be mixed into a class that has:
  - self._model (ClipListModel)
  - self._input_canvas (ThumbnailCanvas)
  - self._export_canvas (ThumbnailCanvas)
  - self.clip_clicked (Signal)
  - self.selection_changed (Signal)
  - self.extract_requested (Signal)
  - self.export_video_requested (Signal)
  - self.get_selected_clips() -> list[ClipEntry]
  - self._rebuild() -> None
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import glob as glob_module

from PySide6.QtWidgets import QMenu, QMessageBox, QFileDialog
from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QDesktopServices

from backend import ClipEntry, ClipState
from backend.project import is_video_file

logger = logging.getLogger(__name__)


class IOTrayActionsMixin:
    """Mixin providing context menu and batch operations for IOTrayPanel."""

    # ── Context menu (INPUT cards) ──

    def _on_context_menu(self, clip: ClipEntry) -> None:
        """Show right-click context menu for a clip card.

        If the right-clicked clip is not in the current selection,
        single-select it first (standard behaviour).
        """
        if clip.name not in self._input_canvas._selected_names:
            self._input_canvas.set_selected(clip.name)
            self.clip_clicked.emit(clip)
            self.selection_changed.emit([clip])

        selected = self.get_selected_clips()
        n = len(selected)
        multi = n > 1

        menu = QMenu(self)

        # Run Extraction — for clips that have a video source and need frames
        from backend.clip_state import ClipState

        needs_extract = [
            c
            for c in selected
            if c.state == ClipState.EXTRACTING
            or (c.input_asset and c.input_asset.asset_type == "video")
        ]
        if needs_extract:
            label_ext = (
                f"Run Extraction ({len(needs_extract)} clips)"
                if len(needs_extract) > 1
                else "Run Extraction"
            )
            extract_action = QAction(label_ext, self)
            extract_action.triggered.connect(lambda: self.extract_requested.emit(needs_extract))
            menu.addAction(extract_action)
            menu.addSeparator()

        # Rename — single only
        rename_action = QAction("Rename...", self)
        rename_action.setEnabled(not multi)
        rename_action.triggered.connect(lambda: self._rename_clip(clip))
        menu.addAction(rename_action)

        # Open in file manager — single only
        _fm = "Finder" if sys.platform == "darwin" else "Explorer"
        explorer_action = QAction(f"Open in {_fm}", self)
        explorer_action.setEnabled(not multi)
        explorer_action.triggered.connect(lambda: self._open_in_explorer(clip))
        menu.addAction(explorer_action)

        menu.addSeparator()

        # Clear Mask — only show when there's a VideoMamaMaskHint to clear
        any_mask = any(c.mask_asset is not None for c in selected)
        if any_mask:
            label_mask = f"Clear Mask ({n} clips)" if multi else "Clear Mask"
            clear_mask_action = QAction(label_mask, self)
            clear_mask_action.triggered.connect(lambda: self._clear_mask_batch(selected))
            menu.addAction(clear_mask_action)

        # Clear Alpha — only show when there's an AlphaHint to clear
        any_alpha = any(c.alpha_asset is not None for c in selected)
        if any_alpha:
            label_alpha = f"Clear Alpha ({n} clips)" if multi else "Clear Alpha"
            clear_alpha_action = QAction(label_alpha, self)
            clear_alpha_action.triggered.connect(lambda: self._clear_alpha_batch(selected))
            menu.addAction(clear_alpha_action)

        # Clear Outputs — only show when there are outputs to clear
        any_outputs = any(c.has_outputs for c in selected)
        if any_outputs:
            label_clear = f"Clear Outputs ({n} clips)" if multi else "Clear Outputs"
            clear_action = QAction(label_clear, self)
            clear_action.triggered.connect(lambda: self._clear_outputs_batch(selected))
            menu.addAction(clear_action)

        # Clear All — show when there's any generated data to clear
        if any_mask or any_alpha or any_outputs:
            menu.addSeparator()
            label_all = f"Clear All ({n} clips)" if multi else "Clear All"
            clear_all_action = QAction(label_all, self)
            clear_all_action.triggered.connect(lambda: self._clear_all_batch(selected))
            menu.addAction(clear_all_action)

        # Set Output Directory — single clip only
        menu.addSeparator()
        output_dir_action = QAction("Set Output Directory...", self)
        output_dir_action.setEnabled(not multi)
        output_dir_action.triggered.connect(lambda: self._set_output_dir(clip))
        menu.addAction(output_dir_action)

        if clip.custom_output_dir:
            clear_dir_action = QAction("Clear Output Directory Override", self)
            clear_dir_action.setEnabled(not multi)
            clear_dir_action.triggered.connect(lambda: self._clear_output_dir(clip))
            menu.addAction(clear_dir_action)

        # Remove...
        menu.addSeparator()
        label_remove = f"Remove ({n} clips)..." if multi else "Remove..."
        remove_action = QAction(label_remove, self)
        remove_action.triggered.connect(lambda: self._remove_dialog(selected))
        menu.addAction(remove_action)

        from PySide6.QtGui import QCursor

        menu.exec(QCursor.pos())

    def _on_export_context_menu(self, clip: ClipEntry) -> None:
        """Show right-click context menu for an export card."""
        menu = QMenu(self)

        # Export Video — list each available output subdirectory
        if clip.state == ClipState.COMPLETE and hasattr(clip, "output_dir"):
            output_dir = clip.output_dir
            if os.path.isdir(output_dir):
                subdirs = sorted(
                    d
                    for d in os.listdir(output_dir)
                    if os.path.isdir(os.path.join(output_dir, d))
                    and os.listdir(os.path.join(output_dir, d))
                )
                if subdirs:
                    for subdir in subdirs:
                        src = os.path.join(output_dir, subdir)
                        action = QAction(f"Export {subdir} as Video...", self)
                        action.triggered.connect(
                            lambda checked=False, c=clip, s=src: self.export_video_requested.emit(
                                c, s
                            )
                        )
                        menu.addAction(action)
                    menu.addSeparator()

        # Open containing folder (Output directory)
        output_dir = clip.output_dir
        if not os.path.isdir(output_dir):
            output_dir = clip.root_path

        open_action = QAction("Open Containing Folder", self)
        open_action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))
        )
        menu.addAction(open_action)

        from PySide6.QtGui import QCursor

        menu.exec(QCursor.pos())

    def _set_output_dir(self, clip: ClipEntry) -> None:
        """Prompt user to pick a custom output directory for this clip."""
        from backend.project import save_custom_output_dir

        start = clip.custom_output_dir or clip.output_dir
        path = QFileDialog.getExistingDirectory(
            self,
            f"Output Directory for '{clip.name}'",
            start,
            QFileDialog.ShowDirsOnly,
        )
        if not path:
            return
        clip.custom_output_dir = path
        save_custom_output_dir(clip.root_path, path)
        logger.info(f"Set custom output dir for '{clip.name}': {path}")

    def _clear_output_dir(self, clip: ClipEntry) -> None:
        """Remove per-clip output directory override."""
        from backend.project import save_custom_output_dir

        clip.custom_output_dir = ""
        save_custom_output_dir(clip.root_path, None)
        logger.info(f"Cleared custom output dir for '{clip.name}'")

    def _open_export_folder(self, clip: ClipEntry) -> None:
        """Open the export/output folder for a clip."""
        output_dir = clip.output_dir
        if not os.path.isdir(output_dir):
            output_dir = clip.root_path
        if os.path.isdir(output_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))

    def _rename_clip(self, clip: ClipEntry) -> None:
        """Prompt user to rename a clip's display name."""
        from PySide6.QtWidgets import QInputDialog
        from backend.project import set_display_name

        current = clip.name
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Clip",
            "New name:",
            text=current,
        )
        if not ok or not new_name.strip() or new_name.strip() == current:
            return
        set_display_name(clip.root_path, new_name.strip())
        clip.find_assets()  # re-reads display_name into clip.name
        self._rebuild()

    def _open_in_explorer(self, clip: ClipEntry) -> None:
        if os.path.isdir(clip.root_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(clip.root_path))

    def _clear_mask_batch(self, clips: list[ClipEntry]) -> None:
        """Delete VideoMamaMaskHint folder from disk for one or more clips."""
        names = ", ".join(c.name for c in clips[:3])
        if len(clips) > 3:
            names += f" (+{len(clips) - 3} more)"
        confirm = QMessageBox.question(
            self,
            "Clear Mask",
            f"Delete tracked masks for {len(clips)} clip(s)?\n{names}\n\n"
            "This will remove all SAM2 mask frames from disk.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        for clip in clips:
            mask_dir = os.path.join(clip.root_path, "VideoMamaMaskHint")
            if os.path.isdir(mask_dir):
                shutil.rmtree(mask_dir, ignore_errors=True)
            for candidate in glob_module.glob(os.path.join(clip.root_path, "VideoMamaMaskHint.*")):
                if os.path.isfile(candidate) and is_video_file(candidate):
                    os.remove(candidate)
            clip.mask_asset = None
            clip.find_assets()
            self._model.update_clip_state(clip.name, clip.state)

        self._model.layoutChanged.emit()
        if clips:
            self.clip_clicked.emit(clips[0])
        logger.info(f"Cleared masks for {len(clips)} clip(s)")

    @staticmethod
    def _all_output_dirs(clip: ClipEntry) -> list[str]:
        """Return all possible output directories for a clip (current + default).

        Ensures clearing/checking covers both the resolved output_dir and the
        built-in default location, so frames are never orphaned when the user
        changes the output directory setting.
        """
        dirs = [clip.output_dir]
        default = os.path.join(clip.root_path, "Output")
        if default not in dirs:
            dirs.append(default)
        return dirs

    @staticmethod
    def _clear_output_files(output_dir: str) -> int:
        """Remove all output frames and manifest from a single output directory.

        Returns the number of files deleted.
        """
        cleared = 0
        for subdir in ("FG", "Matte", "Comp", "Processed"):
            d = os.path.join(output_dir, subdir)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    fpath = os.path.join(d, f)
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                        cleared += 1
        manifest = os.path.join(output_dir, ".corridorkey_manifest.json")
        if os.path.isfile(manifest):
            os.remove(manifest)
        return cleared

    def _clear_all_batch(self, clips: list[ClipEntry]) -> None:
        """Delete masks, alpha hints, and outputs for one or more clips."""
        names = ", ".join(c.name for c in clips[:3])
        if len(clips) > 3:
            names += f" (+{len(clips) - 3} more)"
        confirm = QMessageBox.question(
            self,
            "Clear All",
            f"Remove ALL generated data for {len(clips)} clip(s)?\n{names}\n\n"
            "This will delete masks, alpha hints, and all output frames.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        for clip in clips:
            # Masks
            mask_dir = os.path.join(clip.root_path, "VideoMamaMaskHint")
            if os.path.isdir(mask_dir):
                shutil.rmtree(mask_dir, ignore_errors=True)
            for candidate in glob_module.glob(os.path.join(clip.root_path, "VideoMamaMaskHint.*")):
                if os.path.isfile(candidate) and is_video_file(candidate):
                    os.remove(candidate)
            clip.mask_asset = None

            # Alpha
            alpha_dir = os.path.join(clip.root_path, "AlphaHint")
            if os.path.isdir(alpha_dir):
                shutil.rmtree(alpha_dir, ignore_errors=True)
            for candidate in glob_module.glob(os.path.join(clip.root_path, "AlphaHint.*")):
                if os.path.isfile(candidate) and is_video_file(candidate):
                    os.remove(candidate)
            clip.alpha_asset = None

            # Outputs — clear all possible locations (current + default)
            for output_dir in self._all_output_dirs(clip):
                self._clear_output_files(output_dir)

            clip.find_assets()
            self._model.update_clip_state(clip.name, clip.state)

        self._model.layoutChanged.emit()
        if clips:
            self.clip_clicked.emit(clips[0])
        logger.info(f"Cleared all generated data for {len(clips)} clip(s)")

    def _clear_alpha_batch(self, clips: list[ClipEntry]) -> None:
        """Delete AlphaHint folder from disk for one or more clips."""
        names = ", ".join(c.name for c in clips[:3])
        if len(clips) > 3:
            names += f" (+{len(clips) - 3} more)"
        confirm = QMessageBox.question(
            self,
            "Clear Alpha",
            f"Delete AlphaHint for {len(clips)} clip(s)?\n{names}\n\n"
            "This will remove all generated alpha hint frames from disk.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        for clip in clips:
            alpha_dir = os.path.join(clip.root_path, "AlphaHint")
            if os.path.isdir(alpha_dir):
                shutil.rmtree(alpha_dir, ignore_errors=True)
            for candidate in glob_module.glob(os.path.join(clip.root_path, "AlphaHint.*")):
                if os.path.isfile(candidate) and is_video_file(candidate):
                    os.remove(candidate)
            clip.alpha_asset = None
            clip.find_assets()  # re-scan disk, updates alpha_asset and state
            self._model.update_clip_state(clip.name, clip.state)

        self._model.layoutChanged.emit()
        # Re-select first affected clip so the viewer rebuilds its FrameIndex
        # (clears stale ALPHA button + scrubber coverage bar)
        if clips:
            self.clip_clicked.emit(clips[0])
        logger.info(f"Cleared AlphaHint for {len(clips)} clip(s)")

    def _clear_outputs_batch(self, clips: list[ClipEntry]) -> None:
        """Clear output files for one or more clips."""
        names = ", ".join(c.name for c in clips[:3])
        if len(clips) > 3:
            names += f" (+{len(clips) - 3} more)"
        confirm = QMessageBox.question(
            self,
            "Clear Outputs",
            f"Remove all output files for {len(clips)} clip(s)?\n{names}\n\n"
            "This will delete FG, Matte, Comp, and Processed frames.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        total_cleared = 0
        for clip in clips:
            # Clear all possible locations (current + default) so nothing is orphaned
            for output_dir in self._all_output_dirs(clip):
                total_cleared += self._clear_output_files(output_dir)

            if clip.state == ClipState.COMPLETE:
                clip.state = ClipState.READY
                self._model.update_clip_state(clip.name, ClipState.READY)

        self._model.layoutChanged.emit()
        # Re-select first affected clip so the viewer rebuilds its FrameIndex
        if clips:
            self.clip_clicked.emit(clips[0])
        logger.info(f"Cleared {total_cleared} output files across {len(clips)} clip(s)")

    def _remove_dialog(self, clips: list[ClipEntry]) -> None:
        """Show remove confirmation dialog with Remove from List / Delete from Disk options."""
        n = len(clips)
        title = f"Remove {n} clip{'s' if n > 1 else ''}?"

        paths_text = "\n".join(c.root_path for c in clips[:5])
        if n > 5:
            paths_text += f"\n... and {n - 5} more"

        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setIcon(QMessageBox.Warning)
        msg.setText(f"How would you like to remove {n} clip{'s' if n > 1 else ''}?")
        msg.setInformativeText(paths_text)

        btn_list = msg.addButton("Remove from List", QMessageBox.AcceptRole)
        btn_disk = msg.addButton("Delete from Disk", QMessageBox.DestructiveRole)
        msg.addButton(QMessageBox.Cancel)

        msg.exec()
        clicked = msg.clickedButton()

        if clicked == btn_list:
            self._remove_clips_from_list(clips)
        elif clicked == btn_disk:
            self._delete_clips_from_disk(clips)

    def _remove_clips_from_list(self, clips: list[ClipEntry]) -> None:
        """Remove clips from the list (files stay on disk).

        Records removed clip folder names in project.json so they don't
        reappear on the next project rescan.  Uses folder_name (stable
        on-disk identity) rather than display name which is mutable.
        """
        from backend.project import add_removed_clip, is_v2_project

        names = {c.name for c in clips}

        # Persist removal in project.json for v2 projects
        for clip in clips:
            # v2 clip root_path: .../project_dir/clips/clip_folder
            parent = os.path.dirname(clip.root_path)
            if os.path.basename(parent) == "clips":
                project_dir = os.path.dirname(parent)
                if is_v2_project(project_dir):
                    add_removed_clip(project_dir, clip.folder_name)

        # Remove in reverse index order to avoid index shifting
        indices = [i for i, c in enumerate(self._model.clips) if c.name in names]
        for i in sorted(indices, reverse=True):
            self._model.remove_clip(i)
        logger.info(f"Removed {len(clips)} clip(s) from list")

    def _delete_clips_from_disk(self, clips: list[ClipEntry]) -> None:
        """Delete clip project folders from disk."""
        for clip in clips:
            if os.path.isdir(clip.root_path):
                shutil.rmtree(clip.root_path, ignore_errors=True)
                logger.info(f"Deleted from disk: {clip.root_path}")
        # Remove from model (skip persistence — folder is gone, won't reappear)
        names = {c.name for c in clips}
        indices = [i for i, c in enumerate(self._model.clips) if c.name in names]
        for i in sorted(indices, reverse=True):
            self._model.remove_clip(i)
        logger.info(f"Deleted {len(clips)} clip(s) from disk")

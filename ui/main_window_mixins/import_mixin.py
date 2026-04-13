from __future__ import annotations

import logging
import os
import re

from PySide6.QtWidgets import QMessageBox, QFileDialog, QInputDialog
from PySide6.QtCore import Slot

from backend.project import VIDEO_FILE_FILTER
from ui.widgets.preferences_dialog import (
    KEY_COPY_SOURCE,
    DEFAULT_COPY_SOURCE,
    KEY_COPY_SEQUENCES,
    DEFAULT_COPY_SEQUENCES,
    get_setting_bool,
)

logger = logging.getLogger(__name__)


class ImportMixin:
    """Clip import, folder handling, and welcome-screen methods for MainWindow."""

    @Slot(str)
    def _on_welcome_folder(self, dir_path: str) -> None:
        """Handle folder selected from welcome screen."""
        self._switch_to_workspace()
        self._on_clips_dir_changed(dir_path)

    @Slot(str)
    def _on_recent_project_opened(self, workspace_path: str) -> None:
        """Open a workspace from the recent projects list.

        Opens the specific project folder directly — NOT the Projects root —
        so only that project's clips appear in the browser (project isolation).
        """
        if not os.path.isdir(workspace_path):
            QMessageBox.warning(self, "Missing", f"Workspace no longer exists:\n{workspace_path}")
            self._recent_store.remove(workspace_path)
            self._welcome.refresh_recents()
            return
        self._switch_to_workspace()
        self._on_clips_dir_changed(workspace_path)

    def _on_delete_selected_clips(self) -> None:
        """Delete key — open remove dialog for selected clips."""
        if not self._clips_dir:
            return
        selected = self._io_tray.get_selected_clips()
        if not selected:
            return
        self._io_tray._remove_dialog(selected)

    def _return_to_welcome(self) -> None:
        """Save session and return to the welcome screen."""
        from PySide6.QtCore import QTimer

        if self._clips_dir:
            try:
                self._on_save_session()
            except Exception:
                pass
        self._stack.setCurrentIndex(0)
        self._status_bar.hide()
        QTimer.singleShot(0, self._ensure_window_mode)
        self._welcome.refresh_recents()
        self._clips_dir = None
        self._current_clip = None
        self._clip_input_is_linear = {}

    @Slot(list)
    def _on_tray_folder_imported(self, dir_path: str) -> None:
        """Handle folder from I/O tray +ADD button — context-aware."""
        if self._clips_dir:
            self._add_folder_to_project(dir_path)
        else:
            self._on_clips_dir_changed(dir_path)

    def _on_tray_files_imported(self, file_paths: list) -> None:
        """Handle files from I/O tray +ADD button — context-aware."""
        if self._clips_dir:
            self._add_videos_to_project(file_paths)
        else:
            self._on_welcome_files(file_paths)

    def _on_welcome_files(self, file_paths: list) -> None:
        """Handle files selected from welcome screen — creates a new project.

        Creates ONE project folder for all selected media (videos and/or images),
        with each video as a separate clip nested inside clips/.
        Image files are handled via the image_files_dropped flow.
        """
        if not file_paths:
            return

        from backend.project import create_project, is_video_file, is_image_file

        # Separate videos from images
        video_paths = [f for f in file_paths if is_video_file(f)]
        image_paths = [f for f in file_paths if is_image_file(f)]

        # If only images were selected, route to image handler
        if not video_paths and image_paths:
            self._on_image_files_dropped(image_paths)
            return

        if not video_paths:
            return

        copy_source = get_setting_bool(KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE)

        # Multi-video: ask user to name the project
        display_name = None
        if len(video_paths) > 1:
            name, ok = QInputDialog.getText(
                self,
                "Name Your Project",
                "Give your project a name:",
            )
            if not ok:
                return  # user cancelled
            display_name = name.strip() or None

        # Create ONE project with all videos as clips
        project_dir = create_project(
            video_paths,
            copy_source=copy_source,
            display_name=display_name,
        )
        logger.info(
            f"Created project with {len(video_paths)} clip(s): "
            f"{os.path.basename(project_dir)} (copy={copy_source})"
        )

        # Open the new project (not the Projects root — each project is isolated)
        self._switch_to_workspace()
        self._on_clips_dir_changed(
            project_dir,
            skip_session_restore=True,
            select_clip=None,
        )

    def _on_import_folder(self) -> None:
        """File -> Import Clips -> Import Folder — context-aware."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Clips Directory",
            "",
            QFileDialog.ShowDirsOnly,
        )
        if not dir_path:
            return
        if self._clips_dir:
            # Already in a project — add folder contents as clips
            self._add_folder_to_project(dir_path)
        else:
            # Welcome screen — create a project named after the folder
            self._create_project_from_folder(dir_path)

    def _on_import_videos(self) -> None:
        """File -> Import Clips -> Import Video(s) — context-aware."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            VIDEO_FILE_FILTER,
        )
        if not paths:
            return
        if self._clips_dir:
            # Already in a project — add videos to current project
            self._add_videos_to_project(paths)
        else:
            # Welcome screen — create a new project
            self._on_welcome_files(paths)

    def _on_import_image_sequence(self) -> None:
        """File -> Import Clips -> Import Image Sequence — context-aware."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Image Sequence Folder",
            "",
            QFileDialog.ShowDirsOnly,
        )
        if not dir_path:
            return
        # When no project is open, prompt for a name (pre-filled with folder name)
        display_name = None
        if not self._clips_dir:
            folder_name = os.path.basename(dir_path.rstrip("/\\"))
            name, ok = QInputDialog.getText(
                self,
                "Name Your Project",
                "Give your project a name:",
                text=folder_name,
            )
            if not ok:
                return
            display_name = name.strip() or folder_name
        self._on_sequence_folder_imported(dir_path, display_name=display_name)

    def _add_folder_to_project(self, dir_path: str) -> None:
        """Import all videos and image sequences from a folder into the current project."""
        from backend.project import (
            is_video_file,
            add_clips_to_project,
            folder_has_image_sequence,
            add_sequences_to_project,
        )

        videos = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if is_video_file(os.path.join(dir_path, f))
        ]
        is_seq = folder_has_image_sequence(dir_path)

        if not videos and not is_seq:
            QMessageBox.information(
                self, "No Media", "No video files or image sequences found in that folder."
            )
            return

        if videos:
            copy_source = get_setting_bool(KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE)
            add_clips_to_project(self._clips_dir, videos, copy_source=copy_source)
            logger.info(f"Added {len(videos)} video clip(s) from folder to project")

        if is_seq:
            copy_seq = get_setting_bool(KEY_COPY_SEQUENCES, DEFAULT_COPY_SEQUENCES)
            add_sequences_to_project(
                self._clips_dir,
                [dir_path],
                copy_source=copy_seq,
            )
            logger.info("Added image sequence from folder to project")

        self._on_clips_dir_changed(self._clips_dir, skip_session_restore=True)

    def _add_videos_to_project(self, file_paths: list) -> None:
        """Import selected video files into the current project."""
        from backend.project import (
            is_video_file,
            add_clips_to_project,
            find_clip_by_source,
            find_removed_clip_by_source,
            clear_removed_clip,
        )

        videos = [f for f in file_paths if is_video_file(f)]
        if not videos:
            return

        # Categorise: already active, removed (restore), or genuinely new
        new_videos = []
        skipped = []
        restored = []
        project_dir = self._clips_dir  # project root (contains clips/ subdir)
        for v in videos:
            existing = find_clip_by_source(self._clips_dir, v)
            if existing:
                skipped.append(existing)
                continue
            # Check if this was previously removed — restore instead of
            # creating a duplicate folder
            removed_folder = find_removed_clip_by_source(project_dir, v)
            if removed_folder:
                clear_removed_clip(project_dir, removed_folder)
                restored.append(removed_folder)
                continue
            new_videos.append(v)

        if skipped and not new_videos and not restored:
            names = ", ".join(f'"{s}"' for s in skipped[:3])
            QMessageBox.information(
                self,
                "Already Imported",
                f"All selected videos are already in the project ({names}).",
            )
            return

        if new_videos:
            copy_source = get_setting_bool(KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE)
            add_clips_to_project(self._clips_dir, new_videos, copy_source=copy_source)

        parts = []
        if new_videos:
            parts.append(f"{len(new_videos)} added")
        if restored:
            parts.append(f"{len(restored)} restored")
        if skipped:
            parts.append(f"{len(skipped)} duplicate(s) skipped")
        logger.info(f"Import: {', '.join(parts)}")

        if new_videos or restored:
            self._on_clips_dir_changed(self._clips_dir, skip_session_restore=True)

    @Slot(str)
    def _on_sequence_folder_imported(
        self,
        folder_path: str,
        display_name: str | None = None,
    ) -> None:
        """Handle image sequence folder from +ADD menu or drag-drop."""
        from backend.project import (
            folder_has_image_sequence,
            validate_sequence_stems,
            count_sequence_frames,
            add_sequences_to_project,
            create_project_from_media,
            find_clip_by_source,
        )

        if not folder_has_image_sequence(folder_path):
            QMessageBox.information(
                self,
                "No Images",
                "No image files found in that folder.\n\n"
                "Supported formats: PNG, JPG, EXR, TIF, TIFF, BMP, DPX",
            )
            return

        # Check if this source is already in the project (skip removed clips)
        if self._clips_dir:
            existing = find_clip_by_source(self._clips_dir, folder_path)
            if existing:
                QMessageBox.information(
                    self,
                    "Already Imported",
                    f'This sequence is already in the project as "{existing}".',
                )
                return
            # Restore if it was previously removed
            from backend.project import find_removed_clip_by_source, clear_removed_clip

            removed_folder = find_removed_clip_by_source(self._clips_dir, folder_path)
            if removed_folder:
                clear_removed_clip(self._clips_dir, removed_folder)
                logger.info(f"Restored removed sequence: {removed_folder}")
                self._on_clips_dir_changed(self._clips_dir, skip_session_restore=True)
                return

        # Check for duplicate stems (e.g. frame.png + frame.exr)
        dupes = validate_sequence_stems(folder_path)
        if dupes:
            sample = ", ".join(dupes[:5])
            if len(dupes) > 5:
                sample += f" ... ({len(dupes)} total)"
            QMessageBox.warning(
                self,
                "Duplicate Filenames",
                f"Found files with the same name but different extensions:\n"
                f"{sample}\n\n"
                f"This would cause output file conflicts. Please use one format "
                f"per sequence folder.",
            )
            return

        n_frames = count_sequence_frames(folder_path)
        logger.info(f"Importing image sequence: {folder_path} ({n_frames} frames)")

        copy_seq = get_setting_bool(KEY_COPY_SEQUENCES, DEFAULT_COPY_SEQUENCES)

        if self._clips_dir:
            add_sequences_to_project(
                self._clips_dir,
                [folder_path],
                copy_source=copy_seq,
            )
            logger.info(f"Added image sequence to project: {folder_path}")
            self._on_clips_dir_changed(self._clips_dir, skip_session_restore=True)
        else:
            proj_name = display_name or os.path.basename(folder_path.rstrip("/\\"))
            project_dir = create_project_from_media(
                sequence_folders=[folder_path],
                copy_sequences=copy_seq,
                display_name=proj_name,
            )
            logger.info(f"Created project from image sequence: {folder_path}")
            self._switch_to_workspace()
            self._on_clips_dir_changed(project_dir, skip_session_restore=True)

    @Slot(list)
    def _on_image_files_dropped(self, file_paths: list) -> None:
        """Handle individual image files dropped — popup for <5 or auto-detect."""
        if not file_paths:
            return

        n = len(file_paths)
        parent_folder = os.path.dirname(file_paths[0])

        # When no project is open, prompt for a project name (loose files
        # may come from generic folders like "Downloads").
        display_name = None
        if not self._clips_dir:
            folder_name = os.path.basename(parent_folder.rstrip("/\\"))
            name, ok = QInputDialog.getText(
                self,
                "Name Your Project",
                "Give your project a name:",
                text=folder_name,
            )
            if not ok:
                return  # user cancelled
            display_name = name.strip() or folder_name

        if n < 5:
            # Show popup: "Just these N frames" or "Scan folder for full sequence?"
            from backend.project import count_sequence_frames

            folder_count = count_sequence_frames(parent_folder)

            msg = QMessageBox(self)
            msg.setWindowTitle("Import Image Frames")
            msg.setText(
                f"You dropped {n} image file(s).\n"
                f"The source folder contains {folder_count} image(s) total."
            )
            msg.setInformativeText("How would you like to import?")

            btn_just_these = msg.addButton(
                f"Copy Just These {n}",
                QMessageBox.AcceptRole,
            )
            btn_full_seq = msg.addButton(
                "Import Full Sequence",
                QMessageBox.ActionRole,
            )
            msg.addButton(QMessageBox.Cancel)
            msg.setDefaultButton(btn_full_seq)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked == btn_just_these:
                self._import_specific_frames(
                    parent_folder,
                    file_paths,
                    display_name=display_name,
                )
            elif clicked == btn_full_seq:
                self._on_sequence_folder_imported(
                    parent_folder,
                    display_name=display_name,
                )
            # else: Cancel — do nothing
        else:
            # >= 5 files: auto-detect as full sequence from parent folder
            self._on_sequence_folder_imported(
                parent_folder,
                display_name=display_name,
            )

    def _import_specific_frames(
        self,
        source_folder: str,
        file_paths: list[str],
        display_name: str | None = None,
    ) -> None:
        """Import specific frames (always copies into Frames/)."""
        from backend.project import (
            create_clip_from_sequence,
            projects_root,
            write_project_json,
        )

        filenames = [os.path.basename(f) for f in file_paths]

        if self._clips_dir:
            clips_dir = os.path.join(self._clips_dir, "clips")
            os.makedirs(clips_dir, exist_ok=True)
            create_clip_from_sequence(
                clips_dir,
                source_folder,
                copy_source=True,
                specific_files=filenames,
            )
            logger.info(f"Imported {len(filenames)} specific frame(s)")
            self._on_clips_dir_changed(self._clips_dir, skip_session_restore=True)
        else:
            # No project open — create one manually with specific files
            from datetime import datetime

            proj_name = display_name or os.path.basename(source_folder.rstrip("/\\"))
            name_stem = re.sub(r"[^\w\-]", "_", proj_name)
            name_stem = re.sub(r"_+", "_", name_stem).strip("_")[:60]
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            project_dir = os.path.join(projects_root(), f"{timestamp}_{name_stem}")
            clips_dir = os.path.join(project_dir, "clips")
            os.makedirs(clips_dir, exist_ok=True)
            clip_name = create_clip_from_sequence(
                clips_dir,
                source_folder,
                copy_source=True,
                specific_files=filenames,
            )
            write_project_json(
                project_dir,
                {
                    "version": 2,
                    "created": datetime.now().isoformat(),
                    "display_name": proj_name,
                    "clips": [clip_name],
                },
            )
            logger.info(f"Created project with {len(filenames)} specific frame(s)")
            self._switch_to_workspace()
            self._on_clips_dir_changed(project_dir, skip_session_restore=True)

    def _create_project_from_folder(self, dir_path: str) -> None:
        """Create a new project from a folder of videos/sequences (welcome screen path).

        Scans the folder for video files and image sequences, creates a project
        named after the folder, and opens it.
        """
        from backend.project import (
            is_video_file,
            create_project,
            folder_has_image_sequence,
            create_project_from_media,
        )

        videos = sorted(
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if is_video_file(os.path.join(dir_path, f))
        )

        # Check if folder itself is an image sequence (no subdirectories)
        is_seq = folder_has_image_sequence(dir_path)

        if not videos and not is_seq:
            QMessageBox.information(
                self, "No Media", "No video files or image sequences found in that folder."
            )
            return

        folder_name = os.path.basename(dir_path.rstrip("/\\"))
        copy_video = get_setting_bool(KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE)
        copy_seq = get_setting_bool(KEY_COPY_SEQUENCES, DEFAULT_COPY_SEQUENCES)

        if videos and not is_seq:
            # Videos only — use existing path
            project_dir = create_project(
                videos,
                copy_source=copy_video,
                display_name=folder_name,
            )
        elif is_seq and not videos:
            # Image sequence only
            project_dir = create_project_from_media(
                sequence_folders=[dir_path],
                copy_sequences=copy_seq,
                display_name=folder_name,
            )
        else:
            # Mixed: videos + image sequence
            project_dir = create_project_from_media(
                video_paths=videos,
                sequence_folders=[dir_path],
                copy_video=copy_video,
                copy_sequences=copy_seq,
                display_name=folder_name,
            )

        media_count = len(videos) + (1 if is_seq else 0)
        logger.info(f"Created project '{folder_name}' with {media_count} source(s) from folder")
        self._switch_to_workspace()
        self._on_clips_dir_changed(project_dir, skip_session_restore=True)

"""Clip scanning from media — image sequences and mixed-media project creation.

Extracted from project.py to keep that module focused on core project
management (creation, JSON helpers, path utilities).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from datetime import datetime

from .project import (
    is_image_file,
    projects_root,
    sanitize_stem,
    write_clip_json,
    write_project_json,
    read_project_json,
    _create_clip_folder,
)

logger = logging.getLogger(__name__)


def create_clip_from_sequence(
    clips_dir: str,
    source_folder: str,
    *,
    copy_source: bool = False,
    specific_files: list[str] | None = None,
    display_name: str | None = None,
) -> str:
    """Create a clip folder from an image sequence directory.

    When *copy_source* is False (default), the clip references the original
    folder in place — no files are copied. The external path is stored in
    clip.json and find_assets() resolves it at scan time.

    When *copy_source* is True, image files are copied into the clip's
    Frames/ directory.

    When *specific_files* is provided, only those files (basenames) from
    *source_folder* are used. This forces copy_source=True since we can't
    reference a subset of a directory.

    Args:
        clips_dir: Parent clips directory (e.g., project/clips/).
        source_folder: Absolute path to the folder containing image files.
        copy_source: Whether to copy images into the clip folder.
        specific_files: Optional list of specific filenames to import.
        display_name: Optional display name for the clip.

    Returns:
        The clip folder name (not full path).
    """
    # Derive clip name from folder name or display_name
    if display_name and display_name.strip():
        clean = display_name.strip()
        clip_name = re.sub(r"[^\w\-]", "_", clean)
        clip_name = re.sub(r"_+", "_", clip_name).strip("_")[:60]
    else:
        clip_name = sanitize_stem(os.path.basename(source_folder))

    # Deduplicate clip folder names within same project
    clip_dir = os.path.join(clips_dir, clip_name)
    if os.path.exists(clip_dir):
        for i in range(2, 100):
            candidate = os.path.join(clips_dir, f"{clip_name}_{i}")
            if not os.path.exists(candidate):
                clip_dir = candidate
                clip_name = f"{clip_name}_{i}"
                break

    os.makedirs(clip_dir, exist_ok=True)

    # If specific files requested, always copy (can't reference a subset)
    if specific_files:
        copy_source = True

    if copy_source:
        frames_dir = os.path.join(clip_dir, "Frames")
        os.makedirs(frames_dir, exist_ok=True)
        files_to_copy = specific_files or [
            f
            for f in os.listdir(source_folder)
            if os.path.isfile(os.path.join(source_folder, f)) and is_image_file(f)
        ]
        for f in files_to_copy:
            src = os.path.join(source_folder, f)
            dst = os.path.join(frames_dir, f)
            if os.path.isfile(src) and not os.path.isfile(dst):
                shutil.copy2(src, dst)
        logger.info(f"Copied {len(files_to_copy)} frame(s) from {source_folder} -> {frames_dir}")
    else:
        logger.info(f"Referencing image sequence in place: {source_folder}")

    # Write clip.json with sequence source metadata
    clip_display = display_name or os.path.basename(source_folder).replace("_", " ")
    write_clip_json(
        clip_dir,
        {
            "source": {
                "type": "sequence",
                "original_path": os.path.abspath(source_folder),
                "copied": copy_source,
            },
            "display_name": clip_display,
        },
    )

    return clip_name


def add_sequences_to_project(
    project_dir: str,
    sequence_folders: list[str],
    *,
    copy_source: bool = False,
) -> list[str]:
    """Add image sequence clips to an existing project.

    Args:
        project_dir: Absolute path to the project folder.
        sequence_folders: List of folder paths containing image sequences.
        copy_source: Whether to copy images into clip folders.

    Returns:
        List of new clip subfolder paths (absolute).
    """
    clips_dir = os.path.join(project_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    new_paths: list[str] = []
    for folder in sequence_folders:
        clip_name = create_clip_from_sequence(
            clips_dir,
            folder,
            copy_source=copy_source,
        )
        new_paths.append(os.path.join(clips_dir, clip_name))

    # Update project.json clips list + clear from removed_clips if re-imported
    data = read_project_json(project_dir) or {}
    existing = data.get("clips", [])
    removed = set(data.get("removed_clips", []))
    for p in new_paths:
        name = os.path.basename(p)
        existing.append(name)
        removed.discard(name)
    data["clips"] = existing
    data["removed_clips"] = sorted(removed)
    write_project_json(project_dir, data)

    return new_paths


def create_project_from_media(
    video_paths: list[str] | None = None,
    sequence_folders: list[str] | None = None,
    *,
    copy_video: bool = True,
    copy_sequences: bool = False,
    display_name: str | None = None,
) -> str:
    """Create a new project from a mix of videos and/or image sequences.

    Generalizes create_project() to handle both media types.

    Args:
        video_paths: Optional list of video file paths.
        sequence_folders: Optional list of image sequence folder paths.
        copy_video: Whether to copy video files into clip folders.
        copy_sequences: Whether to copy image sequences into clip folders.
        display_name: Optional project name.

    Returns:
        Absolute path to the new project folder.
    """
    video_paths = video_paths or []
    sequence_folders = sequence_folders or []

    if not video_paths and not sequence_folders:
        raise ValueError("At least one video or image sequence is required")

    root = projects_root()

    if display_name and display_name.strip():
        clean = display_name.strip()
        name_stem = re.sub(r"[^\w\-]", "_", clean)
        name_stem = re.sub(r"_+", "_", name_stem).strip("_")[:60]
        project_display_name = clean
    elif video_paths:
        first_filename = os.path.basename(video_paths[0])
        name_stem = sanitize_stem(first_filename)
        project_display_name = name_stem.replace("_", " ")
    else:
        first_folder = os.path.basename(sequence_folders[0])
        name_stem = sanitize_stem(first_folder)
        project_display_name = name_stem.replace("_", " ")

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{name_stem}"

    project_dir = os.path.join(root, folder_name)
    if os.path.exists(project_dir):
        for i in range(2, 100):
            candidate = os.path.join(root, f"{folder_name}_{i}")
            if not os.path.exists(candidate):
                project_dir = candidate
                break

    clips_dir = os.path.join(project_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    clip_names: list[str] = []

    # Add video clips
    for video_path in video_paths:
        clip_name = _create_clip_folder(
            clips_dir,
            video_path,
            copy_source=copy_video,
        )
        clip_names.append(clip_name)

    # Add sequence clips
    for seq_folder in sequence_folders:
        clip_name = create_clip_from_sequence(
            clips_dir,
            seq_folder,
            copy_source=copy_sequences,
        )
        clip_names.append(clip_name)

    write_project_json(
        project_dir,
        {
            "version": 2,
            "created": datetime.now().isoformat(),
            "display_name": project_display_name,
            "clips": clip_names,
        },
    )

    return project_dir

"""Project folder management — creation, scanning, and metadata.

A project is a timestamped folder inside the Projects root:
    Projects/
        260301_093000_Woman_Jumps/
            Source/
                Woman_Jumps_For_Joy.mp4
            Frames/
            AlphaHint/
            Output/
                FG/ Matte/ Comp/ Processed/
            project.json
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)

_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v"})


def projects_root() -> str:
    """Return the Projects root directory, creating it if needed.

    In dev mode: {repo_root}/Projects/
    In frozen mode: {exe_dir}/Projects/
    """
    from main import get_app_dir
    root = os.path.join(get_app_dir(), "Projects")
    os.makedirs(root, exist_ok=True)
    return root


def sanitize_stem(filename: str, max_len: int = 60) -> str:
    """Clean a filename stem for use in folder names.

    Strips extension, replaces non-alphanumeric chars with underscores,
    collapses runs, and truncates.
    """
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r"[^\w\-]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem[:max_len]


def create_project(source_video_path: str, *, copy_source: bool = True) -> str:
    """Create a new project folder for a source video.

    When *copy_source* is True (default), the video file is copied into
    the project's ``Source/`` directory.  When False, the project stores
    a reference to the original file and frames are extracted in-place.

    Creates: Projects/YYMMDD_HHMMSS_{stem}/

    Args:
        source_video_path: Absolute path to the source video file.
        copy_source: Whether to copy the video into the project folder.

    Returns:
        Absolute path to the new project folder.
    """
    root = projects_root()
    filename = os.path.basename(source_video_path)
    stem = sanitize_stem(filename)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{stem}"

    # Deduplicate if folder already exists (e.g. rapid imports)
    project_dir = os.path.join(root, folder_name)
    if os.path.exists(project_dir):
        for i in range(2, 100):
            candidate = os.path.join(root, f"{folder_name}_{i}")
            if not os.path.exists(candidate):
                project_dir = candidate
                break

    source_dir = os.path.join(project_dir, "Source")
    os.makedirs(source_dir, exist_ok=True)

    if copy_source:
        # Copy video preserving original filename
        target = os.path.join(source_dir, filename)
        if not os.path.isfile(target):
            shutil.copy2(source_video_path, target)
            logger.info(f"Copied source video: {source_video_path} -> {target}")
    else:
        logger.info(f"Referencing source video in place: {source_video_path}")

    # Write initial project.json
    write_project_json(project_dir, {
        "version": 1,
        "created": datetime.now().isoformat(),
        "display_name": sanitize_stem(filename).replace("_", " "),
        "source": {
            "original_path": os.path.abspath(source_video_path),
            "filename": filename,
            "copied": copy_source,
        },
    })

    return project_dir


def write_project_json(project_root: str, data: dict) -> None:
    """Atomic write of project.json."""
    path = os.path.join(project_root, "project.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def read_project_json(project_root: str) -> dict | None:
    """Read project.json, returning None if missing or corrupt."""
    path = os.path.join(project_root, "project.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read project.json at {path}: {e}")
        return None


def get_display_name(project_root: str) -> str:
    """Get the user-visible project name.

    Returns display_name from project.json, falling back to the folder name.
    """
    data = read_project_json(project_root)
    if data and data.get("display_name"):
        return data["display_name"]
    return os.path.basename(project_root)


def set_display_name(project_root: str, name: str) -> None:
    """Update the display_name in project.json."""
    data = read_project_json(project_root) or {}
    data["display_name"] = name
    write_project_json(project_root, data)


def save_in_out_range(project_root: str, in_out) -> None:
    """Persist in/out range to project.json. Pass None to clear."""
    data = read_project_json(project_root) or {}
    if in_out is not None:
        data["in_out_range"] = in_out.to_dict()
    else:
        data.pop("in_out_range", None)
    write_project_json(project_root, data)


def load_in_out_range(project_root: str):
    """Load in/out range from project.json, or None if not set."""
    data = read_project_json(project_root)
    if data and "in_out_range" in data:
        try:
            from .clip_state import InOutRange
            return InOutRange.from_dict(data["in_out_range"])
        except (KeyError, TypeError):
            return None
    return None


def is_video_file(filename: str) -> bool:
    """Check if a filename has a video extension."""
    return os.path.splitext(filename)[1].lower() in _VIDEO_EXTS

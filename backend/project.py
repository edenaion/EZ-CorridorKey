"""Project folder management — creation, scanning, and metadata.

A project is a timestamped container holding one or more clips:
    Projects/
        260301_093000_Woman_Jumps/
            project.json                    (v2 — project-level metadata)
            clips/
                Woman_Jumps/                (ClipEntry.root_path → here)
                    Source/
                        Woman_Jumps_For_Joy.mp4
                    Frames/
                    AlphaHint/
                    Output/FG/ Matte/ Comp/ Processed/
                    clip.json               (per-clip metadata)
                Man_Walks/
                    Source/...

Legacy v1 format (no clips/ dir) is still supported for backward compat.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime

logger = logging.getLogger(__name__)

_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v", ".gif"})
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp", ".dpx"})
VIDEO_FILE_FILTER = "Video Files (*.mp4 *.mov *.avi *.mkv *.mxf *.webm *.m4v *.gif);;All Files (*)"

_app_dir: str | None = None


def set_app_dir(path: str) -> None:
    """Set the application directory. Called at startup by main.py.

    In frozen builds, called twice: first with exe dir, then with
    get_data_dir() (user-chosen install path) so projects_root() resolves correctly.
    """
    global _app_dir
    _app_dir = path


def get_data_dir() -> str:
    """Return the user-data root for models and projects.

    Portable mode: everything lives next to the .exe.
    Dev mode: project root (same as _app_dir).
    Frozen: QSettings app/install_path, falling back to platform default.
    """
    if not getattr(sys, "frozen", False):
        if _app_dir:
            return _app_dir
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Portable: all data next to the exe
    exe_dir = os.path.dirname(sys.executable)
    if os.path.isfile(os.path.join(exe_dir, "portable.txt")):
        return exe_dir
    # Frozen: read user-chosen install path
    try:
        from PySide6.QtCore import QSettings

        saved = QSettings().value("app/install_path", "", type=str)
        if saved and os.path.isdir(saved):
            return saved
    except Exception:
        pass
    # Platform default
    if sys.platform == "win32":
        return os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "EZ-CorridorKey")
    elif sys.platform == "darwin":
        return os.path.join(
            os.path.expanduser("~"), "Library", "Application Support", "EZ-CorridorKey"
        )
    return os.path.join(os.path.expanduser("~"), ".local", "share", "EZ-CorridorKey")


def projects_root() -> str:
    """Return the Projects root directory, creating it if needed.

    In dev mode: {repo_root}/Projects/
    In frozen mode: {install_path}/Projects/ (set via get_data_dir() at startup)
    """
    if _app_dir:
        root = os.path.join(_app_dir, "Projects")
    elif getattr(sys, "frozen", False):
        root = os.path.join(os.path.dirname(sys.executable), "Projects")
    else:
        # Fallback: two levels up from this file (backend/ -> repo root)
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Projects")
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


def create_project(
    source_video_paths: str | list[str],
    *,
    copy_source: bool = True,
    display_name: str | None = None,
) -> str:
    """Create a new project folder for one or more source videos.

    Creates a v2 project with a ``clips/`` subdirectory.  Each video
    gets its own clip subfolder inside ``clips/``.

    When *copy_source* is True (default), video files are copied into
    each clip's ``Source/`` directory.  When False, the clip stores a
    reference to the original file path.

    Creates: Projects/YYMMDD_HHMMSS_{stem}/clips/{clip_stem}/Source/...

    Args:
        source_video_paths: Single video path (str) or list of paths.
        copy_source: Whether to copy video files into clip folders.
        display_name: Optional project name. If provided, used for both
            the folder name stem and display_name in project.json.
            If None, derived from the first video filename.

    Returns:
        Absolute path to the new project folder.
    """
    # Accept single path for backward compat
    if isinstance(source_video_paths, str):
        source_video_paths = [source_video_paths]
    if not source_video_paths:
        raise ValueError("At least one source video path is required")

    root = projects_root()

    if display_name and display_name.strip():
        clean = display_name.strip()
        # Sanitize for folder name (no splitext — it's not a filename)
        name_stem = re.sub(r"[^\w\-]", "_", clean)
        name_stem = re.sub(r"_+", "_", name_stem).strip("_")[:60]
        project_display_name = clean
    else:
        first_filename = os.path.basename(source_video_paths[0])
        name_stem = sanitize_stem(first_filename)
        project_display_name = name_stem.replace("_", " ")

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{name_stem}"

    # Deduplicate if folder already exists (e.g. rapid imports)
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
    for video_path in source_video_paths:
        clip_name = _create_clip_folder(
            clips_dir,
            video_path,
            copy_source=copy_source,
        )
        clip_names.append(clip_name)

    # Write project.json (v2 — project-level metadata only)
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


def add_clips_to_project(
    project_dir: str,
    source_video_paths: list[str],
    *,
    copy_source: bool = True,
) -> list[str]:
    """Add new clips to an existing project.

    Args:
        project_dir: Absolute path to the project folder.
        source_video_paths: List of video file paths to add.
        copy_source: Whether to copy videos into clip folders.

    Returns:
        List of new clip subfolder paths (absolute).
    """
    clips_dir = os.path.join(project_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    new_paths: list[str] = []
    for video_path in source_video_paths:
        clip_name = _create_clip_folder(
            clips_dir,
            video_path,
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


def _copy_companion_alphahint(video_path: str, source_dir: str) -> None:
    """Copy a companion ``{stem}_alphahint.*`` video into *source_dir* if found."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    parent = os.path.dirname(video_path)
    for f in os.listdir(parent):
        f_stem, f_ext = os.path.splitext(f.lower())
        if f_ext in {e.lower() for e in _VIDEO_EXTS} and f_stem == f"{stem.lower()}_alphahint":
            src = os.path.join(parent, f)
            dst = os.path.join(source_dir, f)
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)
                logger.info("Copied companion alpha hint: %s -> %s", src, dst)
            return


def _create_clip_folder(
    clips_dir: str,
    video_path: str,
    *,
    copy_source: bool = True,
) -> str:
    """Create a single clip subfolder inside clips_dir.

    Returns the clip folder name (not full path).
    """
    filename = os.path.basename(video_path)
    clip_name = sanitize_stem(filename)

    # Deduplicate clip folder names within same project
    clip_dir = os.path.join(clips_dir, clip_name)
    if os.path.exists(clip_dir):
        for i in range(2, 100):
            candidate = os.path.join(clips_dir, f"{clip_name}_{i}")
            if not os.path.exists(candidate):
                clip_dir = candidate
                clip_name = f"{clip_name}_{i}"
                break

    source_dir = os.path.join(clip_dir, "Source")
    os.makedirs(source_dir, exist_ok=True)

    if copy_source:
        target = os.path.join(source_dir, filename)
        if not os.path.isfile(target):
            shutil.copy2(video_path, target)
            logger.info(f"Copied source video: {video_path} -> {target}")

        # Auto-copy companion alpha hint video if present
        # Convention: {stem}_alphahint.{ext} next to the source video
        _copy_companion_alphahint(video_path, source_dir)
    else:
        logger.info(f"Referencing source video in place: {video_path}")

    # Write clip.json (per-clip metadata)
    write_clip_json(
        clip_dir,
        {
            "source": {
                "original_path": os.path.abspath(video_path),
                "filename": filename,
                "copied": copy_source,
            },
        },
    )

    return clip_name


def get_clip_dirs(project_dir: str) -> list[str]:
    """Return absolute paths to all clip subdirectories in a project.

    For v2 projects (with clips/ dir), scans clips/ subdirectories.
    For v1 projects (no clips/ dir), returns [project_dir] as a single clip.
    """
    clips_dir = os.path.join(project_dir, "clips")
    if os.path.isdir(clips_dir):
        return sorted(
            os.path.join(clips_dir, d)
            for d in os.listdir(clips_dir)
            if os.path.isdir(os.path.join(clips_dir, d))
            and not d.startswith(".")
            and not d.startswith("_")
        )
    # v1 fallback: project dir itself is the clip
    return [project_dir]


def get_removed_clips(project_dir: str) -> set[str]:
    """Return the set of clip folder basenames that the user has removed."""
    data = read_project_json(project_dir)
    if data and "removed_clips" in data:
        return set(data["removed_clips"])
    return set()


def add_removed_clip(project_dir: str, clip_name: str) -> None:
    """Record a clip folder as removed so it won't reappear on rescan.

    Uses clip folder basename (stable identity), not display name.
    Only updates the removed_clips key — never overwrites a missing file.
    """
    data = read_project_json(project_dir)
    if data is None:
        logger.warning(f"Cannot persist removal: project.json missing at {project_dir}")
        return
    removed = set(data.get("removed_clips", []))
    removed.add(clip_name)
    data["removed_clips"] = sorted(removed)
    write_project_json(project_dir, data)
    logger.info(f"Marked clip folder as removed: {clip_name}")


def clear_removed_clip(project_dir: str, clip_name: str) -> None:
    """Un-remove a clip (e.g. when user explicitly re-imports it).

    Uses clip folder basename (stable identity), not display name.
    """
    data = read_project_json(project_dir)
    if data is None:
        return
    removed = set(data.get("removed_clips", []))
    if clip_name in removed:
        removed.discard(clip_name)
        data["removed_clips"] = sorted(removed)
        write_project_json(project_dir, data)
        logger.info(f"Restored removed clip folder: {clip_name}")


def is_v2_project(project_dir: str) -> bool:
    """Check if a project uses the v2 nested clips structure."""
    return os.path.isdir(os.path.join(project_dir, "clips"))


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


def write_clip_json(clip_root: str, data: dict) -> None:
    """Atomic write of clip.json (per-clip metadata)."""
    path = os.path.join(clip_root, "clip.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def read_clip_json(clip_root: str) -> dict | None:
    """Read clip.json, returning None if missing or corrupt."""
    path = os.path.join(clip_root, "clip.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read clip.json at {path}: {e}")
        return None


def _read_clip_or_project_json(root: str) -> dict | None:
    """Read clip.json first, falling back to project.json for v1 compat."""
    data = read_clip_json(root)
    if data is not None:
        return data
    return read_project_json(root)


def get_display_name(root: str) -> str:
    """Get the user-visible name for a clip or project.

    Checks clip.json first, then project.json, falling back to folder name.
    """
    data = _read_clip_or_project_json(root)
    if data and data.get("display_name"):
        return data["display_name"]
    return os.path.basename(root)


def set_display_name(root: str, name: str) -> None:
    """Update display_name. Writes to clip.json if it exists, else project.json."""
    if os.path.isfile(os.path.join(root, "clip.json")):
        data = read_clip_json(root) or {}
        data["display_name"] = name
        write_clip_json(root, data)
    else:
        data = read_project_json(root) or {}
        data["display_name"] = name
        write_project_json(root, data)


def save_in_out_range(clip_root: str, in_out) -> None:
    """Persist in/out range to clip.json (v2) or project.json (v1).

    Pass None to clear.
    """
    if os.path.isfile(os.path.join(clip_root, "clip.json")):
        data = read_clip_json(clip_root) or {}
        if in_out is not None:
            data["in_out_range"] = in_out.to_dict()
        else:
            data.pop("in_out_range", None)
        write_clip_json(clip_root, data)
    else:
        data = read_project_json(clip_root) or {}
        if in_out is not None:
            data["in_out_range"] = in_out.to_dict()
        else:
            data.pop("in_out_range", None)
        write_project_json(clip_root, data)


def load_in_out_range(clip_root: str):
    """Load in/out range from clip.json or project.json, or None if not set."""
    data = _read_clip_or_project_json(clip_root)
    if data and "in_out_range" in data:
        try:
            from .clip_state import InOutRange

            return InOutRange.from_dict(data["in_out_range"])
        except (KeyError, TypeError):
            return None
    return None


def save_custom_output_dir(clip_root: str, output_dir: str | None) -> None:
    """Persist a custom output directory to clip.json.

    Pass None or empty string to clear the override (revert to default).
    """
    data = read_clip_json(clip_root) or {}
    if output_dir:
        data["output_dir"] = output_dir
    else:
        data.pop("output_dir", None)
    write_clip_json(clip_root, data)


def load_custom_output_dir(clip_root: str) -> str:
    """Load custom output directory from clip.json, or empty string if not set."""
    data = _read_clip_or_project_json(clip_root)
    if data and "output_dir" in data:
        return str(data["output_dir"])
    return ""


def is_video_file(filename: str) -> bool:
    """Check if a filename has a video extension."""
    return os.path.splitext(filename)[1].lower() in _VIDEO_EXTS


def is_image_file(filename: str) -> bool:
    """Check if a filename has an image extension."""
    return os.path.splitext(filename)[1].lower() in _IMAGE_EXTS


def folder_has_image_sequence(folder_path: str) -> bool:
    """Check if a folder contains image files (potential image sequence).

    Returns True if the folder contains at least one file with a recognized
    image extension. Does NOT check subdirectories.
    """
    if not os.path.isdir(folder_path):
        return False
    return any(
        is_image_file(f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    )


def count_sequence_frames(folder_path: str) -> int:
    """Count image files in a folder (potential sequence frame count)."""
    if not os.path.isdir(folder_path):
        return 0
    return sum(
        1
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and is_image_file(f)
    )


def validate_sequence_stems(folder_path: str) -> list[str]:
    """Check for duplicate stems in a folder (e.g. frame.png + frame.exr).

    Returns list of duplicate stem names. Empty list means no conflicts.
    """
    if not os.path.isdir(folder_path):
        return []
    stems: dict[str, list[str]] = {}
    for f in os.listdir(folder_path):
        if not os.path.isfile(os.path.join(folder_path, f)):
            continue
        if not is_image_file(f):
            continue
        stem = os.path.splitext(f)[0]
        stems.setdefault(stem, []).append(f)
    return [stem for stem, files in stems.items() if len(files) > 1]


def find_clip_by_source(
    project_dir: str,
    source_path: str,
    *,
    include_removed: bool = False,
) -> str | None:
    """Check if any clip in the project already references the given source.

    Compares against source.original_path in each clip's clip.json.
    Works for both video and sequence sources.

    By default, skips clips that the user has removed (still on disk but
    hidden). Set *include_removed=True* to search all clips.

    Returns the clip display name if a duplicate is found, or None.
    """
    normalised = os.path.normcase(os.path.abspath(source_path))
    removed = get_removed_clips(project_dir) if not include_removed else set()
    for clip_dir in get_clip_dirs(project_dir):
        if os.path.basename(clip_dir) in removed:
            continue
        data = read_clip_json(clip_dir)
        if not data:
            continue
        source = data.get("source", {})
        existing = source.get("original_path", "")
        if existing and os.path.normcase(os.path.abspath(existing)) == normalised:
            return data.get("display_name", os.path.basename(clip_dir))
    return None


def find_removed_clip_by_source(project_dir: str, source_path: str) -> str | None:
    """Find a removed clip folder that references the given source.

    Returns the clip folder basename if found in removed_clips, or None.
    Used to restore removed clips instead of creating duplicates.
    """
    normalised = os.path.normcase(os.path.abspath(source_path))
    removed = get_removed_clips(project_dir)
    if not removed:
        return None
    for clip_dir in get_clip_dirs(project_dir):
        folder = os.path.basename(clip_dir)
        if folder not in removed:
            continue
        data = read_clip_json(clip_dir)
        if not data:
            continue
        source = data.get("source", {})
        existing = source.get("original_path", "")
        if existing and os.path.normcase(os.path.abspath(existing)) == normalised:
            return folder
    return None


# --- Backward-compat re-exports (moved to project_media.py) ---
from .project_media import (  # noqa: F401, E402
    create_clip_from_sequence,
    add_sequences_to_project,
    create_project_from_media,
)

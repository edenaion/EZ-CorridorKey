"""Clip directory scanning — scan_clips_dir and scan_project_clips.

Extracted from clip_state.py to keep that module focused on dataclasses
and state machine logic.
"""

from __future__ import annotations

import logging
import os

from .clip_state import (
    ClipAsset,
    ClipEntry,
    ClipState,
)
from .errors import ClipScanError
from .project import is_video_file as _is_video_file

logger = logging.getLogger(__name__)


def scan_project_clips(project_dir: str) -> list[ClipEntry]:
    """Scan a single project directory for its clips.

    v2 projects (with ``clips/`` subdir): each subdirectory inside clips/ is a clip.
    v1 projects (no ``clips/`` subdir): the project dir itself is a single clip.

    Clips that the user previously removed from the list (tracked in
    project.json ``removed_clips``) are skipped.

    Args:
        project_dir: Absolute path to a project folder.

    Returns:
        List of ClipEntry objects with root_path pointing to clip subdirectories.
    """
    from .project import is_v2_project, get_removed_clips

    if is_v2_project(project_dir):
        clips_dir = os.path.join(project_dir, "clips")
        removed = get_removed_clips(project_dir)
        entries: list[ClipEntry] = []
        for item in sorted(os.listdir(clips_dir)):
            item_path = os.path.join(clips_dir, item)
            if item.startswith(".") or item.startswith("_"):
                continue
            if not os.path.isdir(item_path):
                continue
            if item in removed:
                logger.debug(f"Skipping removed clip: {item}")
                continue
            clip = ClipEntry(name=item, root_path=item_path)
            try:
                clip.find_assets()
                entries.append(clip)
            except ClipScanError as e:
                logger.debug(str(e))
        logger.info(f"Scanned v2 project {project_dir}: {len(entries)} clip(s)")
        return entries

    # v1 fallback: project_dir is itself a single clip
    clip = ClipEntry(name=os.path.basename(project_dir), root_path=project_dir)
    try:
        clip.find_assets()
        return [clip]
    except ClipScanError as e:
        logger.debug(str(e))
        return []


def scan_clips_dir(
    clips_dir: str,
    allow_standalone_videos: bool = True,
) -> list[ClipEntry]:
    """Scan a directory for clip folders and optionally standalone video files.

    For the Projects root: iterates project subdirectories and delegates to
    scan_project_clips() for each, flattening results.

    For non-Projects directories: scans subdirectories directly as clips
    (legacy behavior for drag-and-dropped folders).

    Folders without valid input assets are skipped (not added as broken clips).

    Args:
        clips_dir: Path to scan.
        allow_standalone_videos: If False, loose video files at top level are ignored.
            Set False for the Projects root where videos live inside Source/ subdirs.
    """
    entries: list[ClipEntry] = []
    if not os.path.isdir(clips_dir):
        logger.warning(f"Clips directory not found: {clips_dir}")
        return entries

    # If the directory itself is a v2 project, scan its clips directly
    from .project import is_v2_project

    if is_v2_project(clips_dir):
        return scan_project_clips(clips_dir)

    seen_names: set[str] = set()

    for item in sorted(os.listdir(clips_dir)):
        item_path = os.path.join(clips_dir, item)

        # Skip hidden and special items
        if item.startswith(".") or item.startswith("_"):
            continue

        if os.path.isdir(item_path):
            # Check if this is a v2 project container (has clips/ subdir)
            from .project import is_v2_project

            if is_v2_project(item_path):
                # v2 project: scan its clips/ subdirectory
                for clip in scan_project_clips(item_path):
                    if clip.name not in seen_names:
                        entries.append(clip)
                        seen_names.add(clip.name)
            else:
                # Flat clip dir or v1 project
                clip = ClipEntry(name=item, root_path=item_path)
                try:
                    clip.find_assets()
                    entries.append(clip)
                    seen_names.add(clip.name)
                except ClipScanError as e:
                    # Skip folders without valid input assets
                    logger.debug(str(e))

        elif allow_standalone_videos and os.path.isfile(item_path) and _is_video_file(item_path):
            # Standalone video file → treat as a clip needing extraction
            stem = os.path.splitext(item)[0]
            if stem in seen_names:
                continue  # folder clip already exists with this name
            clip = ClipEntry(name=stem, root_path=clips_dir)
            clip.input_asset = ClipAsset(item_path, "video")
            clip.state = ClipState.EXTRACTING
            entries.append(clip)
            seen_names.add(stem)

    logger.info(f"Scanned {clips_dir}: {len(entries)} clip(s) found")
    return entries

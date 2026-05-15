"""Batch folder scanner — detect clips and companion hint files.

Scans a flat folder for video files, pairs them with companion hints
(alphahint / maskhint), and returns structured info for the batch
pipeline dialog.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .project import _HINT_KEYWORDS, _VIDEO_EXTS, is_video_file


@dataclass
class BatchClipInfo:
    """One source clip detected in a batch folder."""

    source_path: str  # Absolute path to the source video
    name: str  # Display name (filename stem)
    hint_type: str  # "none" | "alphahint" | "maskhint"
    hint_path: str | None = None  # Absolute path to companion hint file


def scan_batch_folder(folder: str) -> list[BatchClipInfo]:
    """Scan a flat folder for video files and pair with companion hints.

    Returns a list of BatchClipInfo sorted by name. Companion hint files
    are matched by keyword ("alphahint" or "maskhint" anywhere in the
    filename, case insensitive) and excluded from the source list.

    If a file contains both keywords, "maskhint" takes priority.
    """
    if not os.path.isdir(folder):
        return []

    # Collect all video files
    all_videos: list[str] = []
    for f in os.listdir(folder):
        full = os.path.join(folder, f)
        if os.path.isfile(full) and is_video_file(f):
            all_videos.append(full)

    if not all_videos:
        return []

    # Separate sources from hints
    sources: list[str] = []
    hints: list[tuple[str, str]] = []  # (path, keyword)

    for path in all_videos:
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        # Check for hint keywords (maskhint first for priority)
        if "maskhint" in stem:
            hints.append((path, "maskhint"))
        elif "alphahint" in stem:
            hints.append((path, "alphahint"))
        else:
            sources.append(path)

    # Build result — pair each source with its best-matching hint
    results: list[BatchClipInfo] = []
    used_hints: set[str] = set()

    for src in sources:
        stem = os.path.splitext(os.path.basename(src))[0]
        hint_type = "none"
        hint_path = None

        # Find a hint that belongs to this source
        for h_path, h_keyword in hints:
            if h_path in used_hints:
                continue
            h_stem = os.path.splitext(os.path.basename(h_path))[0].lower()
            # The hint filename should contain the source stem
            # (e.g., "Shot01_MaskHint" contains "shot01")
            if stem.lower() in h_stem:
                hint_type = h_keyword
                hint_path = h_path
                used_hints.add(h_path)
                break

        results.append(BatchClipInfo(
            source_path=src,
            name=stem,
            hint_type=hint_type,
            hint_path=hint_path,
        ))

    results.sort(key=lambda c: c.name.lower())
    return results

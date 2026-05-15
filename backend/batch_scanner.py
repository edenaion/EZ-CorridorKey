"""Batch folder scanner — detect clips and companion hint files.

Scans a folder for video files and image sequence subdirectories, pairs
them with companion hints (alphahint / maskhint), and returns structured
info for the batch pipeline dialog.

Reuses _HINT_KEYWORDS from backend.project for hint classification.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .project import (
    _HINT_KEYWORDS, is_video_file, folder_has_image_sequence,
)


@dataclass
class BatchClipInfo:
    """One source clip detected in a batch folder."""

    source_path: str  # Absolute path to the source video or sequence dir
    name: str  # Display name (filename stem or folder name)
    hint_type: str  # "none" | "alphahint" | "maskhint"
    hint_path: str | None = None  # Absolute path to alphahint file/dir
    mask_hint_path: str | None = None  # Absolute path to maskhint file/dir
    is_sequence: bool = False  # True if source_path is a frame directory


def _classify_hint(name_lower: str) -> str | None:
    """Return the hint keyword found in *name_lower*, or None.

    Uses _HINT_KEYWORDS. Maskhint wins if both are present.
    """
    for kw in reversed(_HINT_KEYWORDS):  # maskhint before alphahint
        if kw in name_lower:
            return kw
    return None


def _entry_stem(path: str, is_dir: bool = False) -> str:
    """Get the comparable stem from a path (no extension for files)."""
    base = os.path.basename(path)
    return base if is_dir else os.path.splitext(base)[0]


def scan_batch_folder(folder: str) -> list[BatchClipInfo]:
    """Scan a folder for video files and image sequence subdirs.

    Returns a list of BatchClipInfo sorted by name. Companion hints
    (files or dirs with "alphahint"/"maskhint" in the name) are paired
    with their source clip by stem matching and excluded from the list.
    """
    if not os.path.isdir(folder):
        return []

    # Collect all entries: (path, is_dir)
    entries: list[tuple[str, bool]] = []
    for item in os.listdir(folder):
        full = os.path.join(folder, item)
        if os.path.isfile(full) and is_video_file(item):
            entries.append((full, False))
        elif (os.path.isdir(full)
              and not item.startswith(('.', '_'))
              and folder_has_image_sequence(full)):
            entries.append((full, True))

    if not entries:
        return []

    # Separate sources from hints
    sources: list[tuple[str, bool]] = []
    hints: list[tuple[str, str, bool]] = []  # (path, keyword, is_dir)

    for path, is_dir in entries:
        stem = _entry_stem(path, is_dir).lower()
        kw = _classify_hint(stem)
        if kw:
            hints.append((path, kw, is_dir))
        else:
            sources.append((path, is_dir))

    # Pair each source with matching hints (stem containment)
    used_hints: set[str] = set()
    results: list[BatchClipInfo] = []

    for src_path, is_dir in sources:
        stem = _entry_stem(src_path, is_dir)
        stem_lower = stem.lower()
        alpha_path = None
        mask_path = None

        for h_path, h_kw, _h_is_dir in hints:
            if h_path in used_hints:
                continue
            h_stem = _entry_stem(h_path, _h_is_dir).lower()
            if stem_lower not in h_stem:
                continue
            if h_kw == "alphahint" and alpha_path is None:
                alpha_path = h_path
                used_hints.add(h_path)
            elif h_kw == "maskhint" and mask_path is None:
                mask_path = h_path
                used_hints.add(h_path)
            if alpha_path and mask_path:
                break

        if alpha_path:
            hint_type = "alphahint"
        elif mask_path:
            hint_type = "maskhint"
        else:
            hint_type = "none"

        results.append(BatchClipInfo(
            source_path=src_path, name=stem,
            hint_type=hint_type,
            hint_path=alpha_path,
            mask_hint_path=mask_path,
            is_sequence=is_dir,
        ))

    results.sort(key=lambda c: c.name.lower())
    return results

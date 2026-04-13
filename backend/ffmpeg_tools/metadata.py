"""Video metadata sidecar read/write."""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_METADATA_FILENAME = ".video_metadata.json"


def write_video_metadata(clip_root: str, metadata: dict) -> None:
    """Write video metadata sidecar JSON to clip root.

    Metadata typically includes: source_path, fps, width, height,
    frame_count, codec, duration, plus optional diagnostic fields such
    as source_probe and exr_vf for extraction bug reports.
    """
    path = os.path.join(clip_root, _METADATA_FILENAME)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.debug(f"Video metadata written: {path}")


def read_video_metadata(clip_root: str) -> dict | None:
    """Read video metadata sidecar from clip root. Returns None if not found."""
    path = os.path.join(clip_root, _METADATA_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"Failed to read video metadata: {e}")
        return None

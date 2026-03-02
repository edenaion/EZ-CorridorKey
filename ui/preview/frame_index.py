"""Stem-based frame index for cross-mode navigation.

Builds an ordered list of stems from the input asset, then maps each
view mode to which stems are available. Navigation is by stem position,
not file index — so index 42 is the same shot across all modes even
when some modes have holes.

Codex critical finding: index-based navigation across directories with
different file counts will misalign. Stem-based is the only correct approach.
"""
from __future__ import annotations

import os
from enum import Enum
from dataclasses import dataclass, field

from backend.natural_sort import natsorted
from backend.project import is_image_file as _is_image


class ViewMode(str, Enum):
    """Available preview modes. Each maps to a specific source directory."""
    INPUT = "Input"
    FG = "FG"
    MATTE = "Matte"
    COMP = "Comp"
    PROCESSED = "Processed"


# Maps ViewMode to the subdirectory relative to clip root
_MODE_DIRS: dict[ViewMode, str] = {
    ViewMode.INPUT: "Input",
    ViewMode.FG: os.path.join("Output", "FG"),
    ViewMode.MATTE: os.path.join("Output", "Matte"),
    ViewMode.COMP: os.path.join("Output", "Comp"),
    ViewMode.PROCESSED: os.path.join("Output", "Processed"),
}


@dataclass
class FrameIndex:
    """Ordered stem timeline with per-mode availability.

    Attributes:
        stems: Naturally sorted list of unique frame stems across all modes.
        availability: Maps ViewMode → set of stems present in that mode.
        stem_files: Maps (ViewMode, stem) → full file path.
    """
    stems: list[str] = field(default_factory=list)
    availability: dict[ViewMode, set[str]] = field(default_factory=dict)
    stem_files: dict[tuple[ViewMode, str], str] = field(default_factory=dict)
    video_modes: dict[ViewMode, str] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return len(self.stems)

    def available_modes(self) -> list[ViewMode]:
        """Return modes that have at least one frame."""
        return [m for m in ViewMode if self.availability.get(m)]

    def has_frame(self, mode: ViewMode, stem_index: int) -> bool:
        """Check if a specific stem index has a frame in the given mode."""
        if stem_index < 0 or stem_index >= len(self.stems):
            return False
        stem = self.stems[stem_index]
        return stem in self.availability.get(mode, set())

    def get_path(self, mode: ViewMode, stem_index: int) -> str | None:
        """Get the full file path for a mode and stem index, or None."""
        if stem_index < 0 or stem_index >= len(self.stems):
            return None
        stem = self.stems[stem_index]
        return self.stem_files.get((mode, stem))

    def is_video_mode(self, mode: ViewMode) -> bool:
        """Check if a mode uses video source (not image sequence)."""
        return mode in self.video_modes


def build_frame_index(
    clip_root: str,
    input_asset_type: str = "sequence",
    video_path: str | None = None,
) -> FrameIndex:
    """Build a FrameIndex by scanning all relevant directories.

    Scans Input/ and Output/{FG,Matte,Comp,Processed} for image files,
    extracts stems, and builds a unified timeline.

    Args:
        clip_root: Path to the clip's root directory.
        input_asset_type: 'sequence' or 'video' (affects Input mode).
        video_path: Direct path to video file (for standalone video clips
                    where the video isn't at clip_root/Input.*).
    """
    index = FrameIndex()
    all_stems: set[str] = set()

    for mode, rel_dir in _MODE_DIRS.items():
        # Resolve INPUT directory — try Frames/ (new) then Input/ (legacy)
        if mode == ViewMode.INPUT:
            # Handle video input
            if input_asset_type == "video":
                if video_path and os.path.isfile(video_path):
                    index.video_modes[mode] = video_path
                else:
                    # Check Source/ first (new format), then Input.* (legacy)
                    source_dir = os.path.join(clip_root, "Source")
                    if os.path.isdir(source_dir):
                        video_exts = ('.mp4', '.mov', '.avi', '.mkv', '.mxf', '.webm', '.m4v')
                        for f in os.listdir(source_dir):
                            if f.lower().endswith(video_exts):
                                index.video_modes[mode] = os.path.join(source_dir, f)
                                break
                    if mode not in index.video_modes:
                        import glob as glob_module
                        video_candidates = glob_module.glob(os.path.join(clip_root, "Input.*"))
                        video_exts = ('.mp4', '.mov', '.avi', '.mkv')
                        for vc in video_candidates:
                            if vc.lower().endswith(video_exts):
                                index.video_modes[mode] = vc
                                break
                continue

            # Image sequence — try Frames/ then Input/
            dir_path = None
            for candidate in ("Frames", "Input"):
                candidate_path = os.path.join(clip_root, candidate)
                if os.path.isdir(candidate_path):
                    dir_path = candidate_path
                    break
            if dir_path is None:
                continue
        else:
            dir_path = os.path.join(clip_root, rel_dir)

        if not os.path.isdir(dir_path):
            continue

        mode_stems: set[str] = set()
        for fname in os.listdir(dir_path):
            if _is_image(fname):
                stem = os.path.splitext(fname)[0]
                mode_stems.add(stem)
                index.stem_files[(mode, stem)] = os.path.join(dir_path, fname)

        if mode_stems:
            index.availability[mode] = mode_stems
            all_stems |= mode_stems

    # Build ordered stem list using natural sort
    index.stems = natsorted(list(all_stems))

    # For video input with no image stems, generate stems from frame count
    if ViewMode.INPUT in index.video_modes and not index.stems:
        vpath = index.video_modes[ViewMode.INPUT]
        try:
            import cv2
            cap = cv2.VideoCapture(vpath)
            if cap.isOpened():
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if count > 0:
                    # Generate synthetic stem names for video-only clips
                    index.stems = [f"frame_{i:06d}" for i in range(count)]
        except Exception:
            pass

    # For video input, mark all stems as available (seekable)
    if ViewMode.INPUT in index.video_modes:
        index.availability[ViewMode.INPUT] = set(index.stems)

    return index

"""Clip entry data model and state machine.

State Machine:
    RAW       — Input asset found, no alpha hint yet
    MASKED    — User mask provided (for VideoMaMa workflow)
    READY     — Alpha hint available (from GVM or VideoMaMa), ready for inference
    COMPLETE  — Inference outputs written
    ERROR     — Processing failed (can retry)

Transitions:
    RAW → MASKED      (user provides VideoMaMa mask)
    RAW → READY       (GVM auto-generates alpha)
    RAW → ERROR       (GVM/scan fails)
    MASKED → READY    (VideoMaMa generates alpha from user mask)
    MASKED → ERROR    (VideoMaMa fails)
    READY → COMPLETE  (inference succeeds)
    READY → ERROR     (inference fails)
    ERROR → RAW       (retry from scratch)
    ERROR → MASKED    (retry with mask)
    ERROR → READY     (retry inference)
    COMPLETE → READY  (reprocess with different params)
"""
from __future__ import annotations

import os
import glob as glob_module
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .errors import InvalidStateTransitionError, ClipScanError

logger = logging.getLogger(__name__)


class ClipState(Enum):
    RAW = "RAW"
    MASKED = "MASKED"
    READY = "READY"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


# Valid transitions: from_state -> set of allowed to_states
_TRANSITIONS: dict[ClipState, set[ClipState]] = {
    ClipState.RAW: {ClipState.MASKED, ClipState.READY, ClipState.ERROR},
    ClipState.MASKED: {ClipState.READY, ClipState.ERROR},
    ClipState.READY: {ClipState.COMPLETE, ClipState.ERROR},
    ClipState.COMPLETE: {ClipState.READY},  # reprocess with different params
    ClipState.ERROR: {ClipState.RAW, ClipState.MASKED, ClipState.READY},
}


def _is_image_file(filename: str) -> bool:
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.exr', '.tif', '.tiff', '.bmp'))


def _is_video_file(filename: str) -> bool:
    return filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))


@dataclass
class ClipAsset:
    """Represents an input source — either an image sequence directory or a video file."""
    path: str
    asset_type: str  # 'sequence' or 'video'
    frame_count: int = 0

    def __post_init__(self):
        self._calculate_length()

    def _calculate_length(self):
        if self.asset_type == 'sequence':
            if os.path.isdir(self.path):
                files = [f for f in os.listdir(self.path) if _is_image_file(f)]
                self.frame_count = len(files)
            else:
                self.frame_count = 0
        elif self.asset_type == 'video':
            try:
                import cv2
                cap = cv2.VideoCapture(self.path)
                if cap.isOpened():
                    self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except Exception:
                self.frame_count = 0

    def get_frame_files(self) -> list[str]:
        """Return naturally sorted list of frame filenames for sequence assets.

        Uses natural sort so frame_2 sorts before frame_10 (not lexicographic).
        """
        if self.asset_type != 'sequence' or not os.path.isdir(self.path):
            return []
        from ui.preview.natural_sort import natsorted
        return natsorted([f for f in os.listdir(self.path) if _is_image_file(f)])


@dataclass
class ClipEntry:
    """A single shot/clip with its assets and processing state."""
    name: str
    root_path: str
    state: ClipState = ClipState.RAW
    input_asset: Optional[ClipAsset] = None
    alpha_asset: Optional[ClipAsset] = None
    mask_asset: Optional[ClipAsset] = None  # User-provided VideoMaMa mask
    warnings: list[str] = field(default_factory=list)
    error_message: Optional[str] = None
    _processing: bool = field(default=False, repr=False)  # lock: watcher must not reclassify

    @property
    def is_processing(self) -> bool:
        """True while a GPU job is actively working on this clip."""
        return self._processing

    def set_processing(self, value: bool) -> None:
        """Set processing lock. Watcher skips reclassification while True."""
        self._processing = value

    def transition_to(self, new_state: ClipState) -> None:
        """Attempt a state transition. Raises InvalidStateTransitionError if not allowed."""
        if new_state not in _TRANSITIONS.get(self.state, set()):
            raise InvalidStateTransitionError(self.name, self.state.value, new_state.value)
        old = self.state
        self.state = new_state
        if new_state != ClipState.ERROR:
            self.error_message = None
        logger.debug(f"Clip '{self.name}': {old.value} → {new_state.value}")

    def set_error(self, message: str) -> None:
        """Transition to ERROR state with a message.

        Works from any state that allows ERROR transition
        (RAW, MASKED, READY — all can error now).
        """
        self.transition_to(ClipState.ERROR)
        self.error_message = message

    @property
    def output_dir(self) -> str:
        return os.path.join(self.root_path, "Output")

    @property
    def has_outputs(self) -> bool:
        """Check if output directory exists with content."""
        out = self.output_dir
        if not os.path.isdir(out):
            return False
        for subdir in ("FG", "Matte", "Comp", "Processed"):
            d = os.path.join(out, subdir)
            if os.path.isdir(d) and os.listdir(d):
                return True
        return False

    def completed_frame_count(self) -> int:
        """Count existing output frames for resume support.

        Uses stem intersection across FG and Matte directories
        (both must exist for a frame to be considered complete).
        Returns 0 if no valid outputs found.
        """
        fg_dir = os.path.join(self.output_dir, "FG")
        matte_dir = os.path.join(self.output_dir, "Matte")

        if not os.path.isdir(fg_dir) or not os.path.isdir(matte_dir):
            return 0

        fg_stems = {os.path.splitext(f)[0] for f in os.listdir(fg_dir) if _is_image_file(f)}
        matte_stems = {os.path.splitext(f)[0] for f in os.listdir(matte_dir) if _is_image_file(f)}

        # A frame is complete only if both FG and Matte exist
        complete_stems = fg_stems & matte_stems
        return len(complete_stems)

    def completed_stems(self) -> set[str]:
        """Return set of frame stems that have both FG and Matte outputs.

        Used for resume: skip frames whose stems are already in this set.
        """
        fg_dir = os.path.join(self.output_dir, "FG")
        matte_dir = os.path.join(self.output_dir, "Matte")

        if not os.path.isdir(fg_dir) or not os.path.isdir(matte_dir):
            return set()

        fg_stems = {os.path.splitext(f)[0] for f in os.listdir(fg_dir) if _is_image_file(f)}
        matte_stems = {os.path.splitext(f)[0] for f in os.listdir(matte_dir) if _is_image_file(f)}
        return fg_stems & matte_stems

    def find_assets(self) -> None:
        """Scan the clip directory for Input, AlphaHint, and mask assets.

        Updates state accordingly. Supports both directory and video file
        formats for VideoMamaMaskHint.
        """
        # Input asset
        input_dir = os.path.join(self.root_path, "Input")
        if os.path.isdir(input_dir):
            if not os.listdir(input_dir):
                raise ClipScanError(f"Clip '{self.name}': 'Input' directory is empty.")
            self.input_asset = ClipAsset(input_dir, 'sequence')
        else:
            candidates = glob_module.glob(os.path.join(self.root_path, "[Ii]nput.*"))
            candidates = [c for c in candidates if _is_video_file(c)]
            if candidates:
                self.input_asset = ClipAsset(candidates[0], 'video')
            else:
                raise ClipScanError(f"Clip '{self.name}': no Input directory or video found.")

        # Alpha hint asset
        alpha_dir = os.path.join(self.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir) and os.listdir(alpha_dir):
            self.alpha_asset = ClipAsset(alpha_dir, 'sequence')

        # VideoMaMa mask hint — directory OR video file
        mask_dir = os.path.join(self.root_path, "VideoMamaMaskHint")
        if os.path.isdir(mask_dir) and os.listdir(mask_dir):
            self.mask_asset = ClipAsset(mask_dir, 'sequence')
        else:
            # Check for mask video file (VideoMamaMaskHint.mp4 etc.)
            mask_candidates = glob_module.glob(
                os.path.join(self.root_path, "VideoMamaMaskHint.*")
            )
            mask_candidates = [c for c in mask_candidates if _is_video_file(c)]
            if mask_candidates:
                self.mask_asset = ClipAsset(mask_candidates[0], 'video')

        # Determine initial state
        self._resolve_state()

    def _resolve_state(self) -> None:
        """Set state based on what assets are present.

        Does NOT set COMPLETE purely from has_outputs — that caused
        premature state flips during folder watch rescans. Instead,
        only the service layer sets COMPLETE after verified processing.
        """
        if self.alpha_asset is not None:
            self.state = ClipState.READY
        elif self.mask_asset is not None:
            self.state = ClipState.MASKED
        else:
            self.state = ClipState.RAW


def scan_clips_dir(clips_dir: str) -> list[ClipEntry]:
    """Scan a directory for clip folders and return ClipEntry instances."""
    entries: list[ClipEntry] = []
    if not os.path.isdir(clips_dir):
        logger.warning(f"Clips directory not found: {clips_dir}")
        return entries

    for item in sorted(os.listdir(clips_dir)):
        item_path = os.path.join(clips_dir, item)
        if not os.path.isdir(item_path):
            continue
        # Skip hidden and special directories
        if item.startswith('.') or item.startswith('_'):
            continue

        clip = ClipEntry(name=item, root_path=item_path)
        try:
            clip.find_assets()
            entries.append(clip)
        except ClipScanError as e:
            logger.warning(str(e))
            clip.warnings.append(str(e))
            entries.append(clip)

    return entries

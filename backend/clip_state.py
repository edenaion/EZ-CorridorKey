"""Clip entry data model and state machine.

State Machine:
    EXTRACTING — Video input being extracted to image sequence
    RAW        — Input asset found, no alpha hint yet
    MASKED     — User mask provided (for VideoMaMa workflow)
    READY      — Alpha hint available (from GVM or VideoMaMa), ready for inference
    COMPLETE   — Inference outputs written
    ERROR      — Processing failed (can retry)

Transitions:
    EXTRACTING → RAW   (extraction completes)
    EXTRACTING → ERROR (extraction fails)
    RAW → MASKED       (user provides VideoMaMa mask)
    RAW → READY        (GVM auto-generates alpha)
    RAW → ERROR        (GVM/scan fails)
    MASKED → READY     (VideoMaMa generates alpha from user mask)
    MASKED → ERROR     (VideoMaMa fails)
    READY → COMPLETE   (inference succeeds)
    READY → ERROR      (inference fails)
    ERROR → RAW        (retry from scratch)
    ERROR → MASKED     (retry with mask)
    ERROR → READY      (retry inference)
    ERROR → EXTRACTING (retry extraction)
    COMPLETE → READY   (reprocess with different params)
"""
from __future__ import annotations

import json
import os
import glob as glob_module
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .errors import InvalidStateTransitionError, ClipScanError
from .project import is_image_file as _is_image_file, is_video_file as _is_video_file

logger = logging.getLogger(__name__)

MASK_TRACK_MANIFEST = ".corridorkey_mask_manifest.json"


class ClipState(Enum):
    EXTRACTING = "EXTRACTING"
    RAW = "RAW"
    MASKED = "MASKED"
    READY = "READY"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


# Valid transitions: from_state -> set of allowed to_states
_TRANSITIONS: dict[ClipState, set[ClipState]] = {
    ClipState.EXTRACTING: {ClipState.RAW, ClipState.ERROR},
    ClipState.RAW: {ClipState.MASKED, ClipState.READY, ClipState.ERROR},
    ClipState.MASKED: {ClipState.READY, ClipState.ERROR},
    ClipState.READY: {ClipState.COMPLETE, ClipState.ERROR},
    ClipState.COMPLETE: {ClipState.READY},  # reprocess with different params
    ClipState.ERROR: {ClipState.RAW, ClipState.MASKED, ClipState.READY, ClipState.EXTRACTING},
}


class PipelineRoute(Enum):
    """Pipeline route for a clip in batch processing."""
    INFERENCE_ONLY = "inference_only"       # READY/COMPLETE → inference
    GVM_PIPELINE = "gvm_pipeline"           # RAW, no annotations → GVM → inference
    VIDEOMAMA_PIPELINE = "videomama_pipeline"  # Annotations → track dense masks → VideoMaMa → inference
    VIDEOMAMA_INFERENCE = "videomama_inference" # MASKED → VideoMaMa → inference
    SKIP = "skip"                           # EXTRACTING or ERROR — cannot process


def mask_sequence_is_videomama_ready(root_path: str) -> bool:
    """Return True when the mask sequence is known to be dense/track-generated."""
    manifest_path = os.path.join(root_path, MASK_TRACK_MANIFEST)
    if not os.path.isfile(manifest_path):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    return data.get("source") in {"sam2", "imported"}


def classify_pipeline_route(clip: "ClipEntry") -> PipelineRoute:
    """Determine the full-pipeline route for a clip.

    For RAW clips, checks for annotations.json on disk to decide
    between GVM (automatic) and VideoMaMa (annotation-based) paths.
    """
    ann_path = os.path.join(clip.root_path, "annotations.json")
    has_annotations = os.path.isfile(ann_path) and os.path.getsize(ann_path) > 2
    mask_ready = mask_sequence_is_videomama_ready(clip.root_path)

    if clip.state in (ClipState.EXTRACTING, ClipState.ERROR):
        return PipelineRoute.SKIP
    if has_annotations and not mask_ready:
        return PipelineRoute.VIDEOMAMA_PIPELINE
    if clip.state in (ClipState.READY, ClipState.COMPLETE):
        return PipelineRoute.INFERENCE_ONLY
    if clip.mask_asset is not None and (mask_ready or not has_annotations):
        return PipelineRoute.VIDEOMAMA_INFERENCE
    if clip.state == ClipState.MASKED:
        return PipelineRoute.VIDEOMAMA_INFERENCE
    if clip.state == ClipState.RAW:
        return PipelineRoute.GVM_PIPELINE
    return PipelineRoute.SKIP


@dataclass
class ClipAsset:
    """Represents an input source — either an image sequence directory or a video file."""
    path: str
    asset_type: str  # 'sequence' or 'video'
    frame_count: int = 0
    # Cached (width, height) after first probe. None until probed.
    _dimensions: Optional[tuple] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self._calculate_length()

    def get_dimensions(self) -> Optional[tuple]:
        """Return (width, height) of the asset, or None if it can't be probed.

        Results are cached after the first successful probe. For videos, opens
        via OpenCV to read frame dimensions. For image sequences, reads the
        first frame's header. Probe failures are cached as None and retried
        on subsequent calls only if the asset object is re-created.
        """
        if self._dimensions is not None:
            return self._dimensions
        try:
            if self.asset_type == 'video':
                import cv2
                cap = cv2.VideoCapture(self.path)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    if w > 0 and h > 0:
                        self._dimensions = (w, h)
                        return self._dimensions
            elif self.asset_type == 'sequence':
                files = self.get_frame_files()
                if not files:
                    return None
                first = os.path.join(self.path, files[0])
                # Use cv2.imread with IMREAD_UNCHANGED to support EXR/PNG/TIFF.
                import cv2
                img = cv2.imread(first, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    h, w = img.shape[:2]
                    self._dimensions = (int(w), int(h))
                    return self._dimensions
        except Exception as e:
            logger.debug(f"ClipAsset.get_dimensions failed for {self.path}: {e}")
        return None

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
            except Exception as e:
                logger.debug(f"Video frame count detection failed for {self.path}: {e}")
                self.frame_count = 0

    def get_frame_files(self) -> list[str]:
        """Return naturally sorted list of frame filenames for sequence assets.

        Uses natural sort so frame_2 sorts before frame_10 (not lexicographic).
        """
        if self.asset_type != 'sequence' or not os.path.isdir(self.path):
            return []
        from .natural_sort import natsorted
        return natsorted([f for f in os.listdir(self.path) if _is_image_file(f)])

    def is_exr_sequence(self) -> bool:
        """True if this is an image sequence where the first frame is EXR."""
        if self.asset_type != 'sequence':
            return False
        files = self.get_frame_files()
        if not files:
            return False
        return files[0].lower().endswith('.exr')


@dataclass
class InOutRange:
    """In/out frame range for sub-clip processing. Both indices inclusive, 0-based."""
    in_point: int
    out_point: int

    @property
    def frame_count(self) -> int:
        return self.out_point - self.in_point + 1

    def contains(self, index: int) -> bool:
        return self.in_point <= index <= self.out_point

    def to_dict(self) -> dict:
        return {"in_point": self.in_point, "out_point": self.out_point}

    @classmethod
    def from_dict(cls, d: dict) -> InOutRange:
        return cls(in_point=d["in_point"], out_point=d["out_point"])


@dataclass
class ClipEntry:
    """A single shot/clip with its assets and processing state."""
    name: str
    root_path: str
    state: ClipState = ClipState.RAW
    input_asset: Optional[ClipAsset] = None
    alpha_asset: Optional[ClipAsset] = None
    mask_asset: Optional[ClipAsset] = None  # User-provided VideoMaMa mask
    in_out_range: Optional[InOutRange] = None  # Per-clip in/out markers (None = full clip)
    source_type: str = "unknown"  # "video", "sequence", or "unknown" (legacy)
    warnings: list[str] = field(default_factory=list)
    error_message: Optional[str] = None
    extraction_progress: float = 0.0  # 0.0 to 1.0 during EXTRACTING
    extraction_total: int = 0         # total frames expected during extraction
    custom_output_dir: str = ""       # Per-clip output dir override (from clip.json)
    _processing: bool = field(default=False, repr=False)  # lock: watcher must not reclassify

    @property
    def folder_name(self) -> str:
        """The on-disk folder basename (stable identity, unlike display name)."""
        return os.path.basename(self.root_path)

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
        logger.debug(f"Clip '{self.name}': {old.value} -> {new_state.value}")

    def set_error(self, message: str) -> None:
        """Transition to ERROR state with a message.

        Works from any state that allows ERROR transition
        (RAW, MASKED, READY — all can error now).
        """
        self.transition_to(ClipState.ERROR)
        self.error_message = message

    @property
    def _project_root(self) -> str | None:
        """Derive the v2 project root from this clip's root_path, or None for v1."""
        parent = os.path.dirname(self.root_path)
        if os.path.basename(parent) == "clips":
            return os.path.dirname(parent)
        return None

    @property
    def output_dir(self) -> str:
        """Resolved output directory.

        Priority: per-clip > per-project > global pref > default.
        """
        # 1. Per-clip override (from clip.json)
        if self.custom_output_dir:
            return self.custom_output_dir

        # 2. Per-project override (from project.json)
        project_root = self._project_root
        if project_root:
            from .project import load_project_output_dir
            project_dir = load_project_output_dir(project_root)
            if project_dir:
                return os.path.join(project_dir, self.name)

        # 3. Global default from QSettings (Qt Core — no UI dependency)
        try:
            from PySide6.QtCore import QSettings
            global_dir = QSettings().value("output/default_directory", "", type=str)
            if global_dir:
                # Use project/clip subfolders to prevent cross-project collision.
                if project_root:
                    project_name = os.path.basename(project_root)
                else:
                    # v1 layout — root_path is the project dir itself
                    project_name = os.path.basename(self.root_path)
                return os.path.join(global_dir, project_name, self.name)
        except Exception:
            pass
        # 4. Default: Output/ inside clip folder
        return os.path.join(self.root_path, "Output")

    def has_video_metadata(self) -> bool:
        """True when this clip's Frames/ were extracted from a source video."""
        return os.path.isfile(os.path.join(self.root_path, ".video_metadata.json"))

    def _video_source_transfer(self) -> str:
        """Best-effort source transfer string from extraction sidecar metadata."""
        metadata_path = os.path.join(self.root_path, ".video_metadata.json")
        if not os.path.isfile(metadata_path):
            return ""
        try:
            import json
            with open(metadata_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return ""
        source_probe = payload.get("source_probe", {})
        return str(source_probe.get("color_transfer", "") or "").strip().lower()

    def should_default_input_linear(self) -> bool:
        """Default Linear for standalone EXR or video-derived EXR tagged as linear."""
        return bool(
            self.input_asset is not None
            and self.input_asset.is_exr_sequence()
            and (
                not self.has_video_metadata()
                or self._video_source_transfer() == "linear"
            )
        )

    @property
    def has_outputs(self) -> bool:
        """Check if any output location has content (current or default)."""
        dirs_to_check = [self.output_dir]
        default = os.path.join(self.root_path, "Output")
        if default not in dirs_to_check:
            dirs_to_check.append(default)
        for out in dirs_to_check:
            if not os.path.isdir(out):
                continue
            for subdir in ("FG", "Matte", "Comp", "Processed"):
                d = os.path.join(out, subdir)
                if os.path.isdir(d) and os.listdir(d):
                    return True
        return False

    def completed_frame_count(self) -> int:
        """Count existing output frames for resume support.

        Manifest-aware: reads .corridorkey_manifest.json to determine which
        outputs were enabled. Falls back to FG+Matte intersection if no manifest.
        """
        return len(self.completed_stems())

    def completed_stems(self) -> set[str]:
        """Return set of frame stems that have all enabled outputs complete.

        Reads the run manifest to determine which outputs to check.
        Falls back to FG+Matte intersection if no manifest exists.
        """
        manifest = self._read_manifest()
        if manifest:
            enabled = manifest.get("enabled_outputs", [])
        else:
            enabled = ["fg", "matte"]

        dir_map = {
            "fg": os.path.join(self.output_dir, "FG"),
            "matte": os.path.join(self.output_dir, "Matte"),
            "comp": os.path.join(self.output_dir, "Comp"),
            "processed": os.path.join(self.output_dir, "Processed"),
        }

        stem_sets = []
        for output_name in enabled:
            d = dir_map.get(output_name)
            if d and os.path.isdir(d):
                stems = {os.path.splitext(f)[0] for f in os.listdir(d) if _is_image_file(f)}
                stem_sets.append(stems)
            else:
                # Required dir missing → no complete frames
                return set()

        if not stem_sets:
            return set()

        # Intersection: frame complete only if ALL enabled outputs exist
        result = stem_sets[0]
        for s in stem_sets[1:]:
            result &= s
        return result

    def _read_manifest(self) -> Optional[dict]:
        """Read the run manifest if it exists."""
        manifest_path = os.path.join(self.output_dir, ".corridorkey_manifest.json")
        if not os.path.isfile(manifest_path):
            return None
        try:
            import json
            with open(manifest_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Failed to read manifest at {manifest_path}: {e}")
            return None

    def _resolve_original_path(self) -> Optional[str]:
        """Resolve the original video path from clip.json or project.json."""
        from .project import _read_clip_or_project_json
        data = _read_clip_or_project_json(self.root_path)
        if not data:
            return None
        source = data.get("source", {})
        path = source.get("original_path")
        if path and os.path.isfile(path):
            return path
        return None

    def _resolve_external_sequence(self) -> Optional[str]:
        """Resolve an externally-referenced image sequence folder from clip.json.

        For imported image sequences that are referenced in-place (not copied),
        clip.json stores source.type='sequence' and source.original_path pointing
        to the external folder.

        Returns the folder path if valid, or None. Sets ERROR state with a clear
        message if the referenced folder no longer exists.
        """
        from .project import _read_clip_or_project_json
        data = _read_clip_or_project_json(self.root_path)
        if not data:
            return None
        source = data.get("source", {})
        if source.get("type") != "sequence":
            return None
        path = source.get("original_path")
        if not path:
            return None
        if os.path.isdir(path):
            return path
        # External folder is gone — set error state instead of raising
        self.state = ClipState.ERROR
        self.error_message = f"Source folder missing: {path}"
        self.source_type = "sequence"
        logger.warning(
            f"Clip '{self.name}': external sequence folder no longer exists: {path}"
        )
        return None

    def _resolve_source_type(self) -> str:
        """Determine source_type from clip.json metadata.

        Returns 'video', 'sequence', or 'unknown' (legacy clips without metadata).
        """
        from .project import _read_clip_or_project_json
        data = _read_clip_or_project_json(self.root_path)
        if not data:
            return "unknown"
        source = data.get("source", {})
        src_type = source.get("type")
        if src_type in ("video", "sequence"):
            return src_type
        # Legacy: has source.filename but no type field → video
        if source.get("filename"):
            return "video"
        return "unknown"

    def find_assets(self) -> None:
        """Scan the clip directory for Input, AlphaHint, and mask assets.

        Updates state accordingly. Supports both new format (Frames/, Source/)
        and legacy format (Input/, Input.*) for backward compatibility.
        Also supports externally-referenced image sequences via clip.json.
        """
        # Reset stale state before rescanning
        self.input_asset = None
        self.alpha_asset = None
        self.mask_asset = None
        self.source_type = "unknown"

        # Load per-clip output dir override from clip.json
        from .project import load_custom_output_dir
        self.custom_output_dir = load_custom_output_dir(self.root_path)

        # Input asset — check new names first, fall back to legacy
        frames_dir = os.path.join(self.root_path, "Frames")
        input_dir = os.path.join(self.root_path, "Input")
        source_dir = os.path.join(self.root_path, "Source")

        if os.path.isdir(frames_dir) and os.listdir(frames_dir):
            self.input_asset = ClipAsset(frames_dir, 'sequence')
            # Determine source_type: check clip.json for provenance
            self.source_type = self._resolve_source_type()
        elif os.path.isdir(input_dir) and os.listdir(input_dir):
            self.input_asset = ClipAsset(input_dir, 'sequence')
            self.source_type = self._resolve_source_type()
        elif os.path.isdir(source_dir):
            videos = [f for f in os.listdir(source_dir) if _is_video_file(f)]
            if videos:
                self.input_asset = ClipAsset(
                    os.path.join(source_dir, videos[0]), 'video',
                )
                self.source_type = "video"
            else:
                # Source/ exists but is empty — check project.json for external reference
                original = self._resolve_original_path()
                if original:
                    self.input_asset = ClipAsset(original, 'video')
                    self.source_type = "video"
                else:
                    raise ClipScanError(f"Clip '{self.name}': 'Source' dir has no video.")
        else:
            # No local Frames/Input/Source — check clip.json for external sequence ref
            ext_seq = self._resolve_external_sequence()
            if ext_seq:
                self.input_asset = ClipAsset(ext_seq, 'sequence')
                self.source_type = "sequence"
            elif self.state == ClipState.ERROR:
                # _resolve_external_sequence() set ERROR (missing source folder)
                # Clip remains visible with error state — don't raise
                return
            else:
                candidates = glob_module.glob(os.path.join(self.root_path, "[Ii]nput.*"))
                candidates = [c for c in candidates if _is_video_file(c)]
                if candidates:
                    self.input_asset = ClipAsset(candidates[0], 'video')
                    self.source_type = "video"
                elif os.path.isdir(input_dir):
                    raise ClipScanError(
                        f"Clip '{self.name}': Input dir is empty — no image files."
                    )
                else:
                    raise ClipScanError(f"Clip '{self.name}': no Input found.")

        # Load display name from project.json if available
        from .project import get_display_name
        display = get_display_name(self.root_path)
        if display != os.path.basename(self.root_path):
            self.name = display

        # Alpha hint asset
        alpha_dir = os.path.join(self.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir) and os.listdir(alpha_dir):
            self.alpha_asset = ClipAsset(alpha_dir, 'sequence')
        else:
            alpha_candidates = glob_module.glob(
                os.path.join(self.root_path, "AlphaHint.*")
            )
            alpha_candidates = [c for c in alpha_candidates if _is_video_file(c)]
            if alpha_candidates:
                self.alpha_asset = ClipAsset(alpha_candidates[0], 'video')

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

        # Load in/out range from project.json
        from .project import load_in_out_range
        self.in_out_range = load_in_out_range(self.root_path)

        # Determine initial state
        self._resolve_state()

    def _resolve_state(self) -> None:
        """Set state based on what assets are present on disk.

        Recovers the furthest pipeline stage from disk contents so the
        user never loses completed work after a restart or crash.

        Priority (highest first):
          COMPLETE  — all input frames have matching outputs (manifest-aware)
          READY     — AlphaHint exists (inference-ready)
          MASKED    — VideoMaMa mask hint exists
          EXTRACTING — video source exists but no frame sequence yet
          RAW       — frame sequence exists, no alpha/mask/output
        """
        # Check COMPLETE first: outputs exist and cover all input frames
        if self.alpha_asset is not None and self.input_asset is not None:
            completed = self.completed_stems()
            if completed and len(completed) >= self.input_asset.frame_count:
                self.state = ClipState.COMPLETE
                return

        # READY: AlphaHint must cover the processing range
        # If in/out markers are set, alpha only needs to cover that range.
        # Otherwise, alpha must cover all input frames.
        if self.alpha_asset is not None:
            if self.input_asset is not None:
                required = self.input_asset.frame_count
                if self.in_out_range is not None:
                    required = self.in_out_range.frame_count
                if self.alpha_asset.frame_count < required:
                    logger.info(
                        f"Clip '{self.name}': partial alpha "
                        f"({self.alpha_asset.frame_count}/{required}"
                        f"{' in/out range' if self.in_out_range else ' total'}), "
                        f"staying at lower state"
                    )
                else:
                    self.state = ClipState.READY
                    return
            else:
                self.state = ClipState.READY
                return

        if self.mask_asset is not None:
            self.state = ClipState.MASKED
        elif (self.input_asset is not None
              and self.input_asset.asset_type == "video"):
            # Video input needs extraction to image sequence
            self.state = ClipState.EXTRACTING
        else:
            self.state = ClipState.RAW


# --- Backward-compat re-exports (moved to clip_scanner.py) ---
from .clip_scanner import scan_project_clips, scan_clips_dir  # noqa: F401, E402

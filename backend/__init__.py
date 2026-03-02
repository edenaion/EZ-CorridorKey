"""Backend service layer for ez-CorridorKey."""

from .clip_state import (
    ClipAsset, ClipEntry, ClipState, InOutRange,
    scan_clips_dir, scan_project_clips,
)
from .errors import CorridorKeyError
from .job_queue import GPUJob, GPUJobQueue, JobType, JobStatus
from .project import (
    projects_root, create_project, add_clips_to_project,
    sanitize_stem, get_clip_dirs, is_v2_project,
    write_project_json, read_project_json,
    write_clip_json, read_clip_json,
    get_display_name, set_display_name, is_video_file,
)
from .natural_sort import natural_sort_key, natsorted
from .service import CorridorKeyService, InferenceParams, OutputConfig

__all__ = [
    "CorridorKeyService",
    "InferenceParams",
    "OutputConfig",
    "ClipAsset",
    "ClipEntry",
    "ClipState",
    "InOutRange",
    "scan_clips_dir",
    "scan_project_clips",
    "GPUJob",
    "GPUJobQueue",
    "JobType",
    "JobStatus",
    "CorridorKeyError",
]

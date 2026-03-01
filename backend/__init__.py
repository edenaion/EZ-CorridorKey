"""Backend service layer for ez-CorridorKey."""

from .clip_state import ClipAsset, ClipEntry, ClipState, InOutRange, scan_clips_dir
from .errors import CorridorKeyError
from .job_queue import GPUJob, GPUJobQueue, JobType, JobStatus
from .project import (
    projects_root, create_project, sanitize_stem,
    write_project_json, read_project_json,
    get_display_name, set_display_name, is_video_file,
)
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
    "GPUJob",
    "GPUJobQueue",
    "JobType",
    "JobStatus",
    "CorridorKeyError",
]

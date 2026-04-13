"""Helper functions for GPU job creation and snapshot management."""

from __future__ import annotations

import copy

from backend import (
    ClipEntry,
    GPUJob,
    InferenceParams,
    JobType,
)


def create_job_snapshot(
    clip: ClipEntry,
    params: InferenceParams | None = None,
    job_type: JobType = JobType.INFERENCE,
    resume: bool = False,
    chunk_size: int = 16,
    birefnet_usage: str = "Matting",
) -> GPUJob:
    """Create a frozen job snapshot for the queue.

    The clip is deep-copied so watcher rescans or UI mutations
    cannot desync the running job (Codex critical finding).

    Args:
        clip: The clip to process (will be deep-copied).
        params: Inference parameters (for INFERENCE jobs).
        job_type: Type of GPU job.
        resume: If True, populate skip_stems from existing outputs.
        chunk_size: VideoMaMa chunk size.
        birefnet_usage: BiRefNet model variant name (for BIREFNET_ALPHA jobs).
    """
    # Deep copy clip so the job holds frozen state, not a live reference
    clip_snapshot = copy.deepcopy(clip)

    job_params: dict = {"_clip_snapshot": clip_snapshot}

    if job_type == JobType.INFERENCE:
        if params is None:
            params = InferenceParams()
        job_params["_inference_params"] = params
        if resume:
            job_params["_skip_stems"] = clip.completed_stems()
    elif job_type in (JobType.SAM2_PREVIEW, JobType.SAM2_TRACK):
        if params is not None:
            job_params["_inference_params"] = params
    elif job_type == JobType.VIDEOMAMA_ALPHA:
        job_params["_chunk_size"] = chunk_size
    elif job_type == JobType.BIREFNET_ALPHA:
        job_params["_birefnet_usage"] = birefnet_usage
    elif job_type == JobType.PREVIEW_REPROCESS:
        if params is None:
            params = InferenceParams()
        job_params["_inference_params"] = params

    return GPUJob(
        job_type=job_type,
        clip_name=clip.name,
        params=job_params,
    )

"""Tests for Phase 4 job queue enhancements — PREVIEW_REPROCESS replacement semantics."""
import pytest
from backend.job_queue import GPUJob, GPUJobQueue, JobType, JobStatus


class TestPreviewReprocessQueue:
    def test_preview_reprocess_replaces_existing(self):
        """PREVIEW_REPROCESS jobs should replace previous preview jobs in queue."""
        q = GPUJobQueue()
        job1 = GPUJob(JobType.PREVIEW_REPROCESS, "clip1")
        job2 = GPUJob(JobType.PREVIEW_REPROCESS, "clip1")

        assert q.submit(job1) is True
        assert q.submit(job2) is True  # replaces, not rejected as duplicate

        # Only job2 should be in queue
        snapshot = q.queue_snapshot
        assert len(snapshot) == 1
        assert snapshot[0].id == job2.id
        assert job1.status == JobStatus.CANCELLED

    def test_preview_reprocess_does_not_block_inference(self):
        """PREVIEW_REPROCESS should not block regular inference jobs."""
        q = GPUJobQueue()
        inf_job = GPUJob(JobType.INFERENCE, "clip1")
        preview_job = GPUJob(JobType.PREVIEW_REPROCESS, "clip1")

        assert q.submit(inf_job) is True
        assert q.submit(preview_job) is True
        assert q.pending_count == 2

    def test_inference_dedup_still_works(self):
        """Regular inference dedup is unchanged."""
        q = GPUJobQueue()
        job1 = GPUJob(JobType.INFERENCE, "clip1")
        job2 = GPUJob(JobType.INFERENCE, "clip1")

        assert q.submit(job1) is True
        assert q.submit(job2) is False  # duplicate rejected

    def test_rapid_preview_requests(self):
        """Rapid preview requests should only keep the last one."""
        q = GPUJobQueue()
        jobs = [GPUJob(JobType.PREVIEW_REPROCESS, "clip1") for _ in range(10)]

        for j in jobs:
            q.submit(j)

        assert q.pending_count == 1
        snapshot = q.queue_snapshot
        assert snapshot[0].id == jobs[-1].id

    def test_sam2_preview_replaces_existing(self):
        """SAM2 preview jobs should also use latest-only semantics."""
        q = GPUJobQueue()
        job1 = GPUJob(JobType.SAM2_PREVIEW, "clip1")
        job2 = GPUJob(JobType.SAM2_PREVIEW, "clip1")

        assert q.submit(job1) is True
        assert q.submit(job2) is True

        snapshot = q.queue_snapshot
        assert len(snapshot) == 1
        assert snapshot[0].id == job2.id
        assert job1.status == JobStatus.CANCELLED

"""Comprehensive tests for GPUJobQueue and GPUJob.

Covers: lifecycle, failure, cancellation, mark_cancelled, callbacks,
callback safety, deduplication, find_job_by_id, properties, clear_history,
and GPUJob internals.
"""

import pytest

from backend.job_queue import GPUJob, GPUJobQueue, JobType, JobStatus
from backend.errors import JobCancelledError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_queue() -> GPUJobQueue:
    return GPUJobQueue()


def _inference(clip: str = "clip1") -> GPUJob:
    return GPUJob(JobType.INFERENCE, clip)


def _gvm(clip: str = "clip1") -> GPUJob:
    return GPUJob(JobType.GVM_ALPHA, clip)


# ---------------------------------------------------------------------------
# TestJobLifecycle
# ---------------------------------------------------------------------------


class TestJobLifecycle:
    def test_submit_puts_job_in_queue_with_queued_status(self):
        q = _make_queue()
        job = _inference()
        result = q.submit(job)
        assert result is True
        assert job.status == JobStatus.QUEUED
        assert q.pending_count == 1

    def test_next_job_returns_first_job_without_removing_it(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        peeked = q.next_job()
        assert peeked is job
        assert q.pending_count == 1  # still in queue

    def test_next_job_on_empty_queue_returns_none(self):
        q = _make_queue()
        assert q.next_job() is None

    def test_start_job_removes_from_queue_and_sets_running(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        assert job.status == JobStatus.RUNNING
        assert q.pending_count == 0
        assert q.current_job is job

    def test_complete_job_sets_completed_status(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.complete_job(job)
        assert job.status == JobStatus.COMPLETED

    def test_complete_job_clears_current_job(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.complete_job(job)
        assert q.current_job is None

    def test_complete_job_moves_to_history(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.complete_job(job)
        history = q.history_snapshot
        assert job in history

    def test_full_lifecycle_status_sequence(self):
        """Status transitions: QUEUED → RUNNING → COMPLETED."""
        q = _make_queue()
        job = _inference()
        q.submit(job)
        assert job.status == JobStatus.QUEUED
        q.start_job(job)
        assert job.status == JobStatus.RUNNING
        q.complete_job(job)
        assert job.status == JobStatus.COMPLETED

    def test_multiple_jobs_processed_in_order(self):
        q = _make_queue()
        job_a = _inference("clipA")
        job_b = _inference("clipB")
        q.submit(job_a)
        q.submit(job_b)

        first = q.next_job()
        q.start_job(first)
        q.complete_job(first)

        second = q.next_job()
        q.start_job(second)
        q.complete_job(second)

        assert first is job_a
        assert second is job_b
        assert len(q.history_snapshot) == 2


# ---------------------------------------------------------------------------
# TestJobFailure
# ---------------------------------------------------------------------------


class TestJobFailure:
    def test_fail_job_sets_failed_status(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.fail_job(job, "CUDA out of memory")
        assert job.status == JobStatus.FAILED

    def test_fail_job_stores_error_message(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.fail_job(job, "CUDA out of memory")
        assert job.error_message == "CUDA out of memory"

    def test_fail_job_clears_current_job(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.fail_job(job, "error")
        assert q.current_job is None

    def test_fail_job_moves_to_history(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.fail_job(job, "error")
        assert job in q.history_snapshot


# ---------------------------------------------------------------------------
# TestJobCancellation
# ---------------------------------------------------------------------------


class TestJobCancellation:
    def test_cancel_queued_job_removes_from_queue(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.cancel_job(job)
        assert q.pending_count == 0

    def test_cancel_queued_job_moves_to_history_as_cancelled(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.cancel_job(job)
        assert job.status == JobStatus.CANCELLED
        assert job in q.history_snapshot

    def test_cancel_running_job_only_sets_flag(self):
        """cancel_job on RUNNING job sets cancel flag but leaves status RUNNING."""
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.cancel_job(job)
        assert job.status == JobStatus.RUNNING  # still running
        assert job.is_cancelled is True  # flag set

    def test_cancel_running_job_stays_as_current(self):
        """cancel_job does NOT clear _current_job — worker must call mark_cancelled."""
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.cancel_job(job)
        assert q.current_job is job

    def test_cancel_current_sets_flag_on_running_job(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.cancel_current()
        assert job.is_cancelled is True

    def test_cancel_current_does_nothing_when_no_job_running(self):
        q = _make_queue()
        q.cancel_current()  # must not raise

    def test_cancel_all_flags_current_job(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.cancel_all()
        assert job.is_cancelled is True

    def test_cancel_all_empties_queue_and_moves_to_history(self):
        q = _make_queue()
        current = _inference("clip1")
        queued_a = _inference("clip2")
        queued_b = _gvm("clip3")
        q.submit(current)
        q.submit(queued_a)
        q.submit(queued_b)
        q.start_job(current)
        q.cancel_all()

        assert q.pending_count == 0
        history = q.history_snapshot
        assert queued_a in history
        assert queued_b in history
        assert queued_a.status == JobStatus.CANCELLED
        assert queued_b.status == JobStatus.CANCELLED

    def test_cancel_all_on_idle_queue_does_not_raise(self):
        q = _make_queue()
        q.cancel_all()  # must not raise


# ---------------------------------------------------------------------------
# TestMarkCancelled
# ---------------------------------------------------------------------------


class TestMarkCancelled:
    def test_mark_cancelled_sets_status(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.mark_cancelled(job)
        assert job.status == JobStatus.CANCELLED

    def test_mark_cancelled_clears_current_job(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.mark_cancelled(job)
        assert q.current_job is None

    def test_mark_cancelled_moves_to_history(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.mark_cancelled(job)
        assert job in q.history_snapshot

    def test_mark_cancelled_allows_next_job_to_run(self):
        """Verify _current_job is freed so the next job can be started."""
        q = _make_queue()
        job1 = _inference("clip1")
        job2 = _inference("clip2")
        q.submit(job1)
        q.submit(job2)
        q.start_job(job1)
        q.mark_cancelled(job1)

        job2_peek = q.next_job()
        assert job2_peek is job2
        q.start_job(job2)
        assert q.current_job is job2


# ---------------------------------------------------------------------------
# TestCallbacks
# ---------------------------------------------------------------------------


class TestCallbacks:
    def test_on_completion_called_after_complete_job(self):
        q = _make_queue()
        calls = []
        q.on_completion = lambda clip: calls.append(clip)

        job = _inference("myClip")
        q.submit(job)
        q.start_job(job)
        q.complete_job(job)

        assert calls == ["myClip"]

    def test_on_error_called_after_fail_job(self):
        q = _make_queue()
        calls = []
        q.on_error = lambda clip, err: calls.append((clip, err))

        job = _inference("myClip")
        q.submit(job)
        q.start_job(job)
        q.fail_job(job, "boom")

        assert calls == [("myClip", "boom")]

    def test_on_progress_updates_job_frames_and_calls_callback(self):
        q = _make_queue()
        progress_calls = []
        q.on_progress = lambda clip, cur, tot: progress_calls.append((clip, cur, tot))

        job = _inference("myClip")
        q.submit(job)
        q.start_job(job)
        q.report_progress("myClip", 5, 100)

        assert job.current_frame == 5
        assert job.total_frames == 100
        assert progress_calls == [("myClip", 5, 100)]

    def test_on_warning_called_via_report_warning(self):
        q = _make_queue()
        warnings = []
        q.on_warning = lambda msg: warnings.append(msg)

        q.report_warning("something unusual happened")

        assert warnings == ["something unusual happened"]

    def test_no_callback_set_does_not_raise(self):
        """All callback paths are safe when callbacks are None."""
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.report_progress("clip1", 1, 10)
        q.report_warning("warn")
        q.complete_job(job)  # no on_completion set

    def test_on_error_not_called_when_none(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.fail_job(job, "err")  # must not raise


# ---------------------------------------------------------------------------
# TestCallbackSafety
# ---------------------------------------------------------------------------


class TestCallbackSafety:
    def test_on_completion_exception_does_not_corrupt_history(self):
        """If on_completion raises, the job must still be in history as COMPLETED."""

        def boom(clip):
            raise RuntimeError("cb blew up")

        q = _make_queue()
        q.on_completion = boom

        job = _inference()
        q.submit(job)
        q.start_job(job)

        with pytest.raises(RuntimeError):
            q.complete_job(job)

        # Job state must be intact despite callback failure
        assert job.status == JobStatus.COMPLETED
        assert job in q.history_snapshot
        assert q.current_job is None

    def test_on_error_exception_does_not_corrupt_history(self):
        """If on_error raises, the job must still be in history as FAILED."""

        def boom(clip, err):
            raise RuntimeError("cb blew up")

        q = _make_queue()
        q.on_error = boom

        job = _inference()
        q.submit(job)
        q.start_job(job)

        with pytest.raises(RuntimeError):
            q.fail_job(job, "the real error")

        assert job.status == JobStatus.FAILED
        assert job.error_message == "the real error"
        assert job in q.history_snapshot
        assert q.current_job is None


# ---------------------------------------------------------------------------
# TestDeduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_same_clip_same_type_queued_twice_rejected(self):
        q = _make_queue()
        job1 = _inference()
        job2 = _inference()
        assert q.submit(job1) is True
        assert q.submit(job2) is False
        assert q.pending_count == 1

    def test_same_clip_same_type_while_running_rejected(self):
        q = _make_queue()
        job1 = _inference()
        q.submit(job1)
        q.start_job(job1)

        job2 = _inference()
        assert q.submit(job2) is False

    def test_different_clips_same_type_both_accepted(self):
        q = _make_queue()
        assert q.submit(_inference("clip1")) is True
        assert q.submit(_inference("clip2")) is True
        assert q.pending_count == 2

    def test_same_clip_different_types_both_accepted(self):
        q = _make_queue()
        assert q.submit(_inference("clip1")) is True
        assert q.submit(_gvm("clip1")) is True
        assert q.pending_count == 2

    def test_completed_job_does_not_block_resubmit(self):
        """After a job completes, the same clip+type can be submitted again."""
        q = _make_queue()
        job1 = _inference()
        q.submit(job1)
        q.start_job(job1)
        q.complete_job(job1)

        job2 = _inference()
        assert q.submit(job2) is True


# ---------------------------------------------------------------------------
# TestFindJob
# ---------------------------------------------------------------------------


class TestFindJob:
    def test_find_job_in_queue(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        found = q.find_job_by_id(job.id)
        assert found is job

    def test_find_current_job(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        found = q.find_job_by_id(job.id)
        assert found is job

    def test_find_job_in_history(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.complete_job(job)
        found = q.find_job_by_id(job.id)
        assert found is job

    def test_find_nonexistent_job_returns_none(self):
        q = _make_queue()
        assert q.find_job_by_id("deadbeef") is None


# ---------------------------------------------------------------------------
# TestProperties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_has_pending_true_when_queue_has_jobs(self):
        q = _make_queue()
        q.submit(_inference())
        assert q.has_pending is True

    def test_has_pending_false_when_queue_empty(self):
        q = _make_queue()
        assert q.has_pending is False

    def test_pending_count_reflects_queue_length(self):
        q = _make_queue()
        q.submit(_inference("a"))
        q.submit(_inference("b"))
        assert q.pending_count == 2

    def test_queue_snapshot_returns_copy(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        snap = q.queue_snapshot
        snap.clear()  # mutate the copy
        assert q.pending_count == 1  # original unaffected

    def test_history_snapshot_returns_copy(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.complete_job(job)
        snap = q.history_snapshot
        snap.clear()
        assert len(q.history_snapshot) == 1  # original unaffected

    def test_all_jobs_snapshot_includes_current_queue_and_history(self):
        q = _make_queue()
        hist_job = _inference("clip_hist")
        q.submit(hist_job)
        q.start_job(hist_job)
        q.complete_job(hist_job)

        queued_job = _inference("clip_queued")
        q.submit(queued_job)

        current_job = _gvm("clip_current")
        q.submit(current_job)
        q.start_job(current_job)

        all_jobs = q.all_jobs_snapshot
        assert current_job in all_jobs
        assert queued_job in all_jobs
        assert hist_job in all_jobs

    def test_all_jobs_snapshot_current_first(self):
        q = _make_queue()
        queued = _inference("clip_q")
        running = _gvm("clip_r")
        q.submit(running)
        q.submit(queued)
        q.start_job(running)

        all_jobs = q.all_jobs_snapshot
        assert all_jobs[0] is running

    def test_current_job_none_when_idle(self):
        q = _make_queue()
        assert q.current_job is None

    def test_current_job_set_after_start(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        assert q.current_job is job


# ---------------------------------------------------------------------------
# TestClearHistory
# ---------------------------------------------------------------------------


class TestClearHistory:
    def test_clear_history_empties_history(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.complete_job(job)
        q.clear_history()
        assert q.history_snapshot == []

    def test_clear_history_does_not_affect_queue(self):
        q = _make_queue()
        done = _inference("done")
        q.submit(done)
        q.start_job(done)
        q.complete_job(done)

        pending = _inference("pending")
        q.submit(pending)

        q.clear_history()
        assert q.pending_count == 1

    def test_clear_history_does_not_affect_current_job(self):
        q = _make_queue()
        job = _inference()
        q.submit(job)
        q.start_job(job)
        q.clear_history()
        assert q.current_job is job


# ---------------------------------------------------------------------------
# TestGPUJob
# ---------------------------------------------------------------------------


class TestGPUJob:
    def test_request_cancel_sets_flag(self):
        job = _inference()
        assert job.is_cancelled is False
        job.request_cancel()
        assert job.is_cancelled is True

    def test_is_cancelled_reflects_internal_flag(self):
        job = _inference()
        job._cancel_requested = True
        assert job.is_cancelled is True

    def test_check_cancelled_raises_when_flagged(self):
        job = _inference("myClip")
        job.current_frame = 42
        job.request_cancel()
        with pytest.raises(JobCancelledError) as exc_info:
            job.check_cancelled()
        assert exc_info.value.clip_name == "myClip"
        assert exc_info.value.frame_index == 42

    def test_check_cancelled_does_not_raise_when_not_flagged(self):
        job = _inference()
        job.check_cancelled()  # must not raise

    def test_job_has_unique_id(self):
        job_a = _inference()
        job_b = _inference()
        assert job_a.id != job_b.id

    def test_job_default_status_is_queued(self):
        job = _inference()
        assert job.status == JobStatus.QUEUED

    def test_job_default_frames_are_zero(self):
        job = _inference()
        assert job.current_frame == 0
        assert job.total_frames == 0

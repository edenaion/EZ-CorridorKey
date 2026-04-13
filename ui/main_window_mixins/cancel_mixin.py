from __future__ import annotations

import logging
import os
import subprocess
import sys

from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import Slot

from backend import ClipState, JobType

logger = logging.getLogger(__name__)


class CancelMixin:
    """Cancel/restart methods for MainWindow."""

    def _cancel_extraction(self) -> None:
        """Stop frame extraction and reset the worker."""
        from ui.workers.extract_worker import ExtractWorker

        old = self._extract_worker
        # Disconnect old signals to prevent stale deliveries
        try:
            old.progress.disconnect(self._on_extract_progress)
            old.finished.disconnect(self._on_extract_finished)
            old.error.disconnect(self._on_extract_error)
        except RuntimeError:
            pass  # already disconnected
        old.stop()
        if not old.wait(5000):
            logger.warning("Extract worker did not stop in 5s — terminating")
            old.terminate()
            old.wait(2000)
        # Restart the worker thread (stop kills the thread loop)
        self._extract_worker = ExtractWorker(parent=self)
        self._extract_worker.progress.connect(self._on_extract_progress)
        self._extract_worker.finished.connect(self._on_extract_finished)
        self._extract_worker.error.connect(self._on_extract_error)
        self._extract_progress.clear()
        self._status_bar.reset_progress()
        # Clear extraction overlay on the viewer
        self._dual_viewer.set_extraction_progress(0.0, 0)
        # Reset any EXTRACTING clips back to their original state
        for clip in self._clip_model.clips:
            if clip.state == ClipState.EXTRACTING:
                clip.extraction_progress = 0.0
                clip.extraction_total = 0
                # Check if frames already exist -> RAW, else back to VIDEO
                frames_dir = os.path.join(clip.root_path, "Frames")
                input_dir = os.path.join(clip.root_path, "Input")
                has_frames = (os.path.isdir(frames_dir) and os.listdir(frames_dir)) or (
                    os.path.isdir(input_dir) and os.listdir(input_dir)
                )
                new_state = ClipState.RAW if has_frames else ClipState.ERROR
                self._clip_model.update_clip_state(clip.name, new_state)
        self._io_tray.refresh()
        self._refresh_button_state()
        logger.info("Frame extraction cancelled by user")

    def _cancel_inference(self) -> None:
        """Cancel all inference/GPU jobs."""
        from ui.main_window import _Toast

        queue = self._service.job_queue
        current_job = queue.current_job
        is_videomama = queue.current_job and queue.current_job.job_type == JobType.VIDEOMAMA_ALPHA
        self._cancel_requested_job_id = current_job.id if current_job is not None else None
        queue.cancel_all()
        self._pipeline_steps.clear()
        self._status_bar.stop_job_timer()
        if current_job is not None:
            self._force_stop_armed = True
            self._status_bar.set_running(True)
            self._status_bar.set_stop_button_mode(force=True)
            self._status_bar.set_message(
                "Stop requested — waiting for current GPU step. "
                "Press FORCE STOP to relaunch if it stays stuck."
            )
        else:
            self._force_stop_armed = False
            self._status_bar.set_running(False)
            self._status_bar.set_message("Cancelled queued work.")
        self._queue_panel.refresh()
        logger.info("Processing cancelled by user")
        if is_videomama:
            _Toast(
                self,
                "GPU is finishing the current chunk.\nVideoMaMa will stop after it completes.",
                center=True,
            )

    def _force_restart_app(self) -> None:
        """Hard-stop a blocked GPU phase by relaunching the app process."""
        from PySide6.QtWidgets import QApplication

        try:
            self._auto_save_annotations()
        except Exception:
            logger.exception("Force stop: failed to auto-save annotations")

        try:
            self._auto_save_session()
        except Exception:
            logger.exception("Force stop: failed to auto-save session")

        if getattr(sys, "frozen", False):
            cmd = [sys.executable, *sys.argv[1:]]
            cwd = os.path.dirname(sys.executable)
        else:
            cmd = [sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]]
            cwd = os.path.dirname(os.path.abspath(sys.argv[0]))

        kwargs: dict = {"cwd": cwd}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
                subprocess, "DETACHED_PROCESS", 0
            )
        else:
            kwargs["start_new_session"] = True

        logger.warning("Force stop: relaunching application to break blocked GPU job")
        try:
            subprocess.Popen(cmd, **kwargs)
        except Exception as e:
            logger.exception(f"Force stop relaunch failed: {e}")
            QMessageBox.critical(
                self,
                "Force Stop Failed",
                "Could not relaunch the app automatically.\n\n"
                "Please close and reopen EZ-CorridorKey manually.",
            )
            return

        self._skip_shutdown_cleanup = True
        self._status_bar.set_message("Force restarting...")
        QApplication.instance().quit()

    @Slot()
    def _on_stop_inference(self) -> None:
        """STOP button handler — confirms and cancels inference."""
        if not self._status_bar._stop_btn.isVisible():
            return
        queue = self._service.job_queue
        if not queue.current_job and not queue.has_pending:
            return

        if self._force_stop_armed:
            reply = QMessageBox.question(
                self,
                "Force Stop",
                "The current GPU step has not returned to Python.\n\n"
                "Force Stop will auto-save the session and relaunch the app "
                "to break the stuck job immediately.\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            self._force_restart_app()
            return

        reply = QMessageBox.question(
            self,
            "Cancel",
            "Cancel processing?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        from ui.sounds.audio_manager import UIAudio

        UIAudio.user_cancel()
        self._cancel_inference()

    @Slot(str)
    def _on_cancel_job(self, job_id: str) -> None:
        """Cancel a specific job from the queue panel."""
        job = self._service.job_queue.find_job_by_id(job_id)
        if job:
            self._service.job_queue.cancel_job(job)
            self._queue_panel.refresh()

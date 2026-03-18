from __future__ import annotations

import logging
import os
import shutil
import sys

from PySide6.QtWidgets import QMessageBox, QFileDialog
from PySide6.QtCore import Slot, QThread

from backend import ClipEntry, ClipState
from backend.clip_state import ClipAsset
from backend import InOutRange
from backend.project import save_in_out_range

logger = logging.getLogger(__name__)


class ExportMixin:
    """Extraction, export, timeline I/O, and in/out marker methods for MainWindow."""

    def _auto_extract_clips(self, clips: list[ClipEntry]) -> None:
        """Auto-submit EXTRACTING clips to the extract worker.

        New-format projects (Source/ subdir) already have video in the right
        place — extraction writes to Frames/. Legacy standalone videos
        (root_path == clips_dir) are restructured into a clip subdirectory.
        """
        extracting = [c for c in clips if c.state == ClipState.EXTRACTING]
        if not extracting:
            return

        if not self._extract_worker.isRunning():
            if sys.platform == "win32":
                self._extract_worker.start(QThread.LowPriority)
            else:
                self._extract_worker.start()

        for clip in extracting:
            if not (clip.input_asset and clip.input_asset.asset_type == "video"):
                continue

            video_path = clip.input_asset.path

            # New format: video already lives in Source/ — no restructuring needed.
            # Legacy standalone video: root_path is the parent dir, not a clip dir.
            # Restructure: create clip_name/ dir and copy video as Input.ext
            source_dir = os.path.join(clip.root_path, "Source")
            if not os.path.isdir(source_dir) and clip.root_path == self._clips_dir:
                ext = os.path.splitext(video_path)[1]
                clip_dir = os.path.join(self._clips_dir, clip.name)
                target = os.path.join(clip_dir, f"Input{ext}")
                if not os.path.isfile(target):
                    os.makedirs(clip_dir, exist_ok=True)
                    shutil.copy2(video_path, target)
                    logger.info(f"Restructured standalone video: {video_path} -> {target}")
                clip.root_path = clip_dir
                clip.input_asset.path = target

            self._extract_worker.submit(
                clip.name, clip.input_asset.path, clip.root_path,
            )
        self._status_bar.start_job_timer(label="Extracting")
        logger.info(f"Auto-extraction queued: {len(extracting)} clip(s)")

    def _on_extract_current_clip(self) -> None:
        """Handle RUN EXTRACTION button — extract the currently selected clip."""
        clip = self._current_clip
        if not clip or clip.state != ClipState.EXTRACTING:
            return
        self._on_extract_requested([clip])

    @Slot(list)
    def _on_extract_requested(self, clips: list) -> None:
        """Handle right-click -> Run Extraction on selected clips."""
        if not clips:
            return
        if not self._extract_worker.isRunning():
            if sys.platform == "win32":
                self._extract_worker.start(QThread.LowPriority)
            else:
                self._extract_worker.start()
        count = 0
        for clip in clips:
            if clip.input_asset and clip.input_asset.asset_type == "video":
                # If retrying after error, wipe partial frames so
                # extract_frames() starts fresh instead of resuming
                if clip.state == ClipState.ERROR:
                    for subdir in ("Frames", "Input"):
                        target = os.path.join(clip.root_path, subdir)
                        if os.path.isdir(target):
                            shutil.rmtree(target)
                            os.makedirs(target, exist_ok=True)
                            logger.info(f"Cleared {subdir}/ for retry: {clip.name}")
                    clip.error_message = None
                    self._clip_model.update_clip_state(
                        clip.name, ClipState.EXTRACTING)
                self._extract_worker.submit(
                    clip.name, clip.input_asset.path, clip.root_path,
                )
                count += 1
        if count:
            self._status_bar.start_job_timer(label="Extracting")
            logger.info(f"Manual extraction queued: {count} clip(s)")

    @Slot(str, int, int)
    def _on_extract_progress(self, clip_name: str, current: int, total: int) -> None:
        """Update status bar, clip card, and input viewer progress."""
        # Track per-clip progress for aggregate status bar
        self._extract_progress[clip_name] = (current, total)
        agg_current = sum(c for c, _ in self._extract_progress.values())
        agg_total = sum(t for _, t in self._extract_progress.values())
        self._status_bar.update_progress(agg_current, agg_total)
        progress = current / total if total > 0 else 0.0
        # Update clip for per-card progress bar
        for i, clip in enumerate(self._clip_model.clips):
            if clip.name == clip_name:
                clip.extraction_progress = progress
                clip.extraction_total = total
                idx = self._clip_model.index(i)
                self._clip_model.dataChanged.emit(idx, idx)
                break
        # Update input viewer overlay if this is the selected clip
        if self._current_clip and self._current_clip.name == clip_name:
            self._dual_viewer.set_extraction_progress(progress, total)

    @Slot(str, int)
    def _on_extract_finished(self, clip_name: str, frame_count: int) -> None:
        """Handle extraction complete — update clip to RAW with image sequence."""
        for clip in self._clip_model.clips:
            if clip.name == clip_name:
                # Clear extraction progress
                clip.extraction_progress = 0.0
                clip.extraction_total = 0

                # Clear input viewer overlay
                if self._current_clip and self._current_clip.name == clip_name:
                    self._dual_viewer.set_extraction_progress(0.0, 0)

                # Update input asset to point to extracted sequence
                # Check Frames/ (new format) then Input/ (legacy)
                frames_dir = os.path.join(clip.root_path, "Frames")
                input_dir = os.path.join(clip.root_path, "Input")
                actual_dir = frames_dir if os.path.isdir(frames_dir) else input_dir
                if os.path.isdir(actual_dir):
                    clip.input_asset = ClipAsset(actual_dir, "sequence")

                # Transition EXTRACTING -> RAW
                clip.state = ClipState.RAW
                self._clip_model.update_clip_state(clip_name, ClipState.RAW)

                # Regenerate thumbnail from sequence
                self._refresh_input_thumbnail(clip)

                # If this is the selected clip, fully re-select to update
                # viewer, param panel buttons (GVM/VideoMaMa), and status bar
                if self._current_clip and self._current_clip.name == clip_name:
                    self._on_clip_selected(clip)

                logger.info(f"Extraction complete: {clip_name} ({frame_count} frames)")
                break

        self._io_tray.refresh()
        # Remove from aggregate tracker; reset status bar when all done
        self._extract_progress.pop(clip_name, None)
        if not self._extract_worker.is_busy:
            self._status_bar.reset_progress()
            from ui.sounds.audio_manager import UIAudio
            UIAudio.frame_extract_done()

    @Slot(str, str)
    def _on_extract_error(self, clip_name: str, error_msg: str) -> None:
        """Handle extraction failure."""
        self._clip_model.update_clip_state(clip_name, ClipState.ERROR)
        for clip in self._clip_model.clips:
            if clip.name == clip_name:
                clip.extraction_progress = 0.0
                clip.extraction_total = 0
                clip.error_message = error_msg
                break
        self._extract_progress.pop(clip_name, None)
        # Clear the extraction overlay on the viewer
        if self._current_clip and self._current_clip.name == clip_name:
            self._dual_viewer.set_extraction_progress(0.0, 0)
        if not self._extract_worker.is_busy:
            self._status_bar.reset_progress()
        from ui.sounds.audio_manager import UIAudio
        UIAudio.error()
        logger.error(f"Extraction failed for {clip_name}: {error_msg}")

    # ── Export Video ──

    def _on_export_video(self, clip: ClipEntry | None = None,
                         source_dir: str | None = None) -> None:
        """Export output image sequence as video file.

        Args:
            clip: Specific clip to export. If None, uses current selection.
            source_dir: Specific source directory. If None, auto-detect.
        """
        if clip is None:
            if self._current_clip is None:
                QMessageBox.information(self, "No Clip", "Select a clip first.")
                return
            clip = self._current_clip

        if clip.state != ClipState.COMPLETE:
            QMessageBox.warning(
                self, "Not Complete",
                f"Clip '{clip.name}' must be COMPLETE to export video.",
            )
            return

        # Use provided source_dir or auto-detect
        if source_dir and os.path.isdir(source_dir) and os.listdir(source_dir):
            pass  # use as-is
        else:
            comp_dir = os.path.join(clip.output_dir, "Comp")
            fg_dir = os.path.join(clip.output_dir, "FG")
            if os.path.isdir(comp_dir) and os.listdir(comp_dir):
                source_dir = comp_dir
            elif os.path.isdir(fg_dir) and os.listdir(fg_dir):
                source_dir = fg_dir
            else:
                QMessageBox.warning(self, "No Output", "No output frames found to export.")
                return

        # Read video metadata for fps
        from backend.ffmpeg_tools import read_video_metadata, stitch_video, require_ffmpeg_install
        try:
            require_ffmpeg_install(require_probe=True)
        except RuntimeError as exc:
            QMessageBox.critical(
                self, "FFmpeg Unavailable",
                str(exc),
            )
            return

        metadata = read_video_metadata(clip.root_path)
        fps = metadata.get("fps", 24.0) if metadata else 24.0

        # Default export to _EXPORTS in the clip's project folder
        subdir_name = os.path.basename(source_dir)
        exports_dir = os.path.join(clip.root_path, "_EXPORTS")
        os.makedirs(exports_dir, exist_ok=True)
        default_name = f"{clip.name}_{subdir_name}_export.mp4"
        default_path = os.path.join(exports_dir, default_name)
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", default_path,
            "MP4 Video (*.mp4);;All Files (*)",
        )
        if not out_path:
            return

        # Determine frame pattern from first file
        frames = sorted(os.listdir(source_dir))
        if not frames:
            return

        # Detect pattern (frame_000000.png -> frame_%06d.png)
        first = frames[0]
        ext = os.path.splitext(first)[1]
        pattern = f"frame_%06d{ext}"

        self._status_bar.set_message(f"Exporting {clip.name}...")

        try:
            stitch_video(
                in_dir=source_dir,
                out_path=out_path,
                fps=fps,
                pattern=pattern,
            )
            self._status_bar.set_message("")
            QMessageBox.information(
                self, "Export Complete",
                f"Video exported:\n{out_path}",
            )
        except Exception as e:
            self._status_bar.set_message("")
            from ui.sounds.audio_manager import UIAudio
            UIAudio.error()
            QMessageBox.critical(
                self, "Export Failed",
                f"Failed to export video:\n{e}",
            )

    # ── View Controls ──

    def _on_reset_zoom(self) -> None:
        """Reset preview zoom to fit."""
        self._dual_viewer.reset_zoom()

    @Slot()
    def _sync_io_divider(self, *_args) -> None:
        """Keep the IO tray divider aligned with the dual viewer splitter."""
        viewer_sizes = self._dual_viewer._viewer_splitter.sizes()
        if len(viewer_sizes) < 2:
            return
        self._io_tray.sync_divider(viewer_sizes[0])

    # ── In/Out Markers ──

    def _set_in_point(self) -> None:
        """Set in-point at the current scrubber position."""
        if not self._current_clip:
            return
        idx = self._dual_viewer._scrubber.current_frame()
        _, out = self._dual_viewer.get_in_out()
        # Clamp: in-point cannot exceed out-point
        if out is not None and idx > out:
            idx = out
        # set_in_point emits in_point_changed -> _persist_in_out
        self._dual_viewer._scrubber.set_in_point(idx)

    def _set_out_point(self) -> None:
        """Set out-point at the current scrubber position."""
        if not self._current_clip:
            return
        idx = self._dual_viewer._scrubber.current_frame()
        in_pt, _ = self._dual_viewer.get_in_out()
        # Clamp: out-point cannot precede in-point
        if in_pt is not None and idx < in_pt:
            idx = in_pt
        # set_out_point emits out_point_changed -> _persist_in_out
        self._dual_viewer._scrubber.set_out_point(idx)

    def _clear_in_out(self) -> None:
        """Clear in/out markers (Alt+I)."""
        if not self._current_clip:
            return
        # Use set_in_out instead of clear_in_out to avoid recursion:
        # clear_in_out emits range_cleared which is connected back to this method.
        self._dual_viewer._scrubber.set_in_out(None, None)
        self._current_clip.in_out_range = None
        save_in_out_range(self._current_clip.root_path, None)
        # Re-resolve state: removing in/out may drop READY -> RAW
        # if alpha only partially covers the full clip
        self._current_clip._resolve_state()
        self._clip_model.update_clip_state(
            self._current_clip.name, self._current_clip.state)
        self._io_tray.refresh()
        self._refresh_button_state()

    def _persist_in_out(self) -> None:
        """Save current in/out markers to the clip and project.json."""
        if not self._current_clip:
            return
        in_pt, out_pt = self._dual_viewer.get_in_out()
        if in_pt is not None and out_pt is not None:
            rng = InOutRange(in_point=in_pt, out_point=out_pt)
            self._current_clip.in_out_range = rng
            save_in_out_range(self._current_clip.root_path, rng)
            # Re-resolve state: partial alpha + new in/out range may
            # promote RAW -> READY (v1.2.1 partial alpha logic)
            self._current_clip._resolve_state()
            self._clip_model.update_clip_state(
                self._current_clip.name, self._current_clip.state)
            self._io_tray.refresh()
        self._refresh_button_state()

    def _on_reset_all_in_out(self) -> None:
        """Clear in/out markers on all clips (called from IO tray button)."""
        count = 0
        for clip in self._clip_model.clips:
            if clip.in_out_range is not None:
                clip.in_out_range = None
                save_in_out_range(clip.root_path, None)
                count += 1
        # Clear the scrubber if current clip was affected
        if self._current_clip:
            self._dual_viewer._scrubber.set_in_out(None, None)
        self._refresh_button_state()
        logger.info(f"Reset in/out markers on {count} clips")

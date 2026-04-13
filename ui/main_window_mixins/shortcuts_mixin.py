from __future__ import annotations

from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QKeySequence


class ShortcutsMixin:
    """Keyboard shortcut wiring and view-mode hotkey methods for MainWindow."""

    def _setup_shortcuts(self) -> None:
        """Wire keyboard shortcuts from the centralized registry."""
        self._shortcut_registry.create_shortcuts(self)
        # Sync menu-bar QAction shortcuts (display text + activation)
        reg = self._shortcut_registry
        self._save_action.setShortcut(QKeySequence(reg.get_key("save_session")))
        self._open_action.setShortcut(QKeySequence(reg.get_key("open_project")))
        self._prefs_action.setShortcut(QKeySequence(reg.get_key("preferences")))

    def _toggle_mute(self) -> None:
        """Toggle UI sounds on/off and show a brief overlay indicator."""
        from ui.sounds.audio_manager import UIAudio
        from ui.widgets.preferences_dialog import KEY_UI_SOUNDS
        from PySide6.QtCore import QSettings
        from ui.main_window import _MuteOverlay

        muted = not UIAudio.is_muted()
        UIAudio.set_muted(muted)
        QSettings().setValue(KEY_UI_SOUNDS, not muted)
        # Sync the menu-bar volume control
        if hasattr(self, "_volume_control"):
            self._volume_control.sync_mute_state()
        # Show overlay top-right
        icon = "\U0001f507" if muted else "\U0001f50a"  # muted vs speaker
        text = f"{icon}  Sound {'OFF' if muted else 'ON'}"
        overlay = _MuteOverlay(self, text)
        overlay.show()

    def _toggle_playback(self) -> None:
        """Forward Space key to the scrubber's play/pause toggle."""
        self._dual_viewer.toggle_playback()

    def _toggle_ab_wipe(self) -> None:
        """Toggle A/B wipe comparison mode."""
        self._dual_viewer.toggle_wipe_mode()

    def _view_mode_input(self) -> None:
        self._dual_viewer._output_viewer.set_view_mode("Input")

    def _view_mode_mask(self) -> None:
        self._dual_viewer._output_viewer.set_view_mode("Mask")

    def _view_mode_alpha(self) -> None:
        self._dual_viewer._output_viewer.set_view_mode("Alpha")

    def _view_mode_fg(self) -> None:
        self._dual_viewer._output_viewer.set_view_mode("FG")

    def _view_mode_matte(self) -> None:
        self._dual_viewer._output_viewer.set_view_mode("Matte")

    def _view_mode_comp(self) -> None:
        self._dual_viewer._output_viewer.set_view_mode("Comp")

    def _view_mode_proc(self) -> None:
        self._dual_viewer._output_viewer.set_view_mode("Processed")

    def _on_escape(self) -> None:
        """Escape: cancel the current action — auto-detects what's running."""
        # 1. Exit annotation mode (no confirmation needed)
        iv = self._dual_viewer.input_viewer
        if iv.annotation_mode:
            iv.set_annotation_mode(None)
            return

        # 2. Detect active process and ask to cancel
        process_name = self._detect_active_process()
        if not process_name:
            return

        reply = QMessageBox.question(
            self,
            "Cancel",
            f"Cancel {process_name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        from ui.sounds.audio_manager import UIAudio

        UIAudio.user_cancel()

        if process_name == "frame extraction":
            self._cancel_extraction()
        else:
            self._cancel_inference()

    def _detect_active_process(self) -> str | None:
        """Return a human-readable name for the currently active process, or None."""
        # Check extraction — is_busy means actively extracting or has pending jobs
        if self._extract_worker.is_busy:
            return "frame extraction"
        # Check inference / GPU jobs
        queue = self._service.job_queue
        if queue.current_job or queue.has_pending:
            return "processing"
        return None

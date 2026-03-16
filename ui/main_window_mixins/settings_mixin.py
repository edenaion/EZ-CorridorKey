from __future__ import annotations

import logging
import os
import re

from PySide6.QtWidgets import QWidget, QLabel, QMessageBox
from PySide6.QtCore import Qt, Slot, QSettings

from ui.widgets.preferences_dialog import (
    PreferencesDialog, KEY_SHOW_TOOLTIPS, DEFAULT_SHOW_TOOLTIPS,
    KEY_TRACKER_MODEL, DEFAULT_TRACKER_MODEL,
    KEY_MODEL_RESOLUTION, DEFAULT_MODEL_RESOLUTION,
    get_setting_bool, get_setting_str, get_setting_int,
)

logger = logging.getLogger(__name__)

# TEMPORARY: keep a visible tester build identifier on the user-test
# branch so remote testers can confirm they pulled the right build. Remove this
# before merging the branch back into main.
_SHOW_TESTER_BUILD_ID = True


class SettingsMixin:
    """Layout, preferences, about dialog, and update methods for MainWindow."""

    def _reset_layout(self) -> None:
        self._splitter.setSizes([920, 280])
        self._vsplitter.setSizes([600, 140])

    def _toggle_queue_panel(self) -> None:
        self._queue_panel.toggle_collapsed()

    def _toggle_debug_console(self) -> None:
        """Toggle the in-app debug console (F12)."""
        if self._debug_console.isVisible():
            self._debug_console.hide()
        else:
            self._debug_console.show()

    def _run_startup_diagnostics(self, device: str) -> None:
        """Check environment for known issues and show a diagnostic dialog."""
        from ui.widgets.diagnostic_dialog import run_startup_diagnostics, StartupDiagnosticDialog
        issues = run_startup_diagnostics(device)
        if issues:
            dlg = StartupDiagnosticDialog(issues, parent=self)
            dlg.exec()

    def _show_preferences(self) -> None:
        """Toggle the Preferences dialog (Ctrl+,)."""
        if self._prefs_dialog is not None:
            self._prefs_dialog.reject()
            return
        dlg = PreferencesDialog(self)
        self._prefs_dialog = dlg
        accepted = dlg.exec() == PreferencesDialog.Accepted
        self._prefs_dialog = None
        if accepted:
            self._apply_tooltip_setting()
            self._apply_sound_setting()
            self._apply_tracker_model_setting()
            self._apply_parallel_clips_setting()
            self._apply_model_resolution_setting()

    def _show_hotkeys(self) -> None:
        """Open the Hotkeys configuration dialog and apply changes."""
        from ui.widgets.hotkeys_dialog import HotkeysDialog
        dlg = HotkeysDialog(self._shortcut_registry, self)
        if dlg.exec() == HotkeysDialog.Accepted:
            self._setup_shortcuts()

    def _apply_tooltip_setting(self) -> None:
        """Enable or disable tooltips globally based on saved preference."""
        show = get_setting_bool(KEY_SHOW_TOOLTIPS, DEFAULT_SHOW_TOOLTIPS)
        if not show:
            # Disable: clear all tooltips on the main window tree
            for w in self.findChildren(QWidget):
                w.setToolTip("")
        # If enabled, tooltips stay as set during widget construction.
        # A full re-enable would require rebuilding tooltip strings, which
        # is unnecessary — the setting takes full effect on next app launch.

    def _apply_sound_setting(self) -> None:
        """Apply UI sounds on/off and volume from saved preferences."""
        from ui.widgets.preferences_dialog import KEY_UI_SOUNDS, DEFAULT_UI_SOUNDS
        from ui.sounds.audio_manager import UIAudio
        UIAudio.set_muted(not get_setting_bool(KEY_UI_SOUNDS, DEFAULT_UI_SOUNDS))
        # Restore volume level
        vol = QSettings().value("ui/sounds_volume", 1.0, type=float)
        UIAudio.set_volume(vol)
        if hasattr(self, "_volume_control"):
            self._volume_control.sync_mute_state()

    def _apply_tracker_model_setting(self) -> None:
        """Apply saved SAM2 tracker model preference to the backend service."""
        model_id = get_setting_str(KEY_TRACKER_MODEL, DEFAULT_TRACKER_MODEL)
        self._service.set_sam2_model(model_id)

    def _apply_model_resolution_setting(self) -> None:
        """Apply saved model resolution preference to the backend service."""
        res = get_setting_int(KEY_MODEL_RESOLUTION, DEFAULT_MODEL_RESOLUTION)
        self._service.set_model_resolution(res)

    def _apply_parallel_clips_setting(self) -> None:
        """Apply saved parallel clips preference to the GPU worker."""
        from ui.widgets.preferences_dialog import (
            get_setting_int as _get_int, KEY_PARALLEL_CLIPS, DEFAULT_PARALLEL_CLIPS,
        )
        n = _get_int(KEY_PARALLEL_CLIPS, DEFAULT_PARALLEL_CLIPS)
        self._gpu_worker.set_max_workers(n)

    def _show_report_issue(self) -> None:
        import logging as _logging

        from ui.widgets.report_issue_dialog import ReportIssueDialog

        # Gather GPU info from last monitor update
        gpu_info = {}
        if hasattr(self, "_last_vram_info"):
            gpu_info = self._last_vram_info

        # Gather recent WARNING/ERROR log lines from debug console buffer
        recent_errors: list[str] = []
        if hasattr(self, "_debug_console"):
            for html, levelno in self._debug_console._log_buffer:
                if levelno >= _logging.WARNING:
                    plain = re.sub(r"<[^>]+>", "", html).strip()
                    if plain:
                        recent_errors.append(plain)
                if len(recent_errors) >= 20:
                    break

        dlg = ReportIssueDialog(
            gpu_info=gpu_info,
            recent_errors=recent_errors,
            parent=self,
        )
        dlg.exec()

    def _show_about(self) -> None:
        app_version = self._get_local_version()
        build_id = self._get_visible_build_id()
        tester_note = ""
        if _SHOW_TESTER_BUILD_ID:
            tester_note = (
                "<p><b>Temporary tester build identifier.</b><br>"
                "Remove before merging this branch back to main.</p>"
            )
        box = QMessageBox(self)
        box.setWindowTitle("About EZ-CorridorKey")
        box.setTextFormat(Qt.RichText)
        box.setText(
            f"<h2>EZ-CorridorKey {build_id}</h2>"
            "<p>AI Green Screen Keyer<br>"
            '<a href="https://github.com/nikopueringer/CorridorKey#corridorkey-licensing-and-permissions">'
            "CC BY-NC-SA 4.0 License</a></p>"
            f"<p>Package version: v{app_version}</p>"
            f"{tester_note}"
            "<p><b>Special Thanks</b></p>"
            "<p>"
            '<a href="https://github.com/nikopueringer/">Niko Pueringer</a> — OG CorridorKey Creator<br>'
            '<a href="https://www.edzisk.com">Ed Zisk</a> — GUI, workflow, SFX, QA<br>'
            '<a href="https://www.clade.design/">Sara Ann Stewart</a> — Logo<br>'
            '<a href="https://github.com/Raiden129">Jhe Kim</a> — Hiera optimization<br>'
            '<a href="https://github.com/MarcelLieb">MarcelLieb</a> — Tiling optimization<br>'
            '<a href="https://github.com/cmoyates">Cristopher Yates</a> — MLX Apple Silicon (<a href="https://github.com/cmoyates/corridorkey-mlx">corridorkey-mlx</a>)<br>'
            '<a href="https://github.com/99oblivius">99oblivius</a> — FX graph cache (<a href="https://github.com/99oblivius/CorridorKey-Engine">CorridorKey-Engine</a>)<br>'
            '<a href="https://github.com/Warwlock">Warwlock</a> — BiRefNet integration'
            "</p>"
        )
        # QMessageBox uses an internal QLabel — find it and enable clickable links
        for label in box.findChildren(QLabel):
            label.setOpenExternalLinks(True)
        box.exec()

    # ── Update Check ──────────────────────────────────────────

    def _get_local_version(self) -> str:
        try:
            from importlib.metadata import version
            return version("corridorkey")
        except Exception:
            try:
                import tomllib
                pyproject = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "pyproject.toml"
                )
                with open(pyproject, "rb") as f:
                    return tomllib.load(f)["project"]["version"]
            except Exception:
                return "0.0.0"

    def _get_git_short_hash(self) -> str:
        try:
            import subprocess

            repo_root = os.path.dirname(os.path.dirname(__file__))
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _get_visible_build_id(self) -> str:
        version = self._get_local_version()
        if not _SHOW_TESTER_BUILD_ID:
            return f"v{version}"

        git_hash = self._get_git_short_hash()
        if git_hash:
            return f"v{version} test ({git_hash})"
        return f"v{version} test"

    def _check_for_updates(self) -> None:
        from ui.main_window import _UpdateChecker
        self._update_thread = _UpdateChecker(self._get_local_version())
        self._update_thread.update_available.connect(self._on_update_available)
        self._update_thread.start()

    @Slot(str)
    def _on_update_available(self, remote_version: str) -> None:
        self._update_btn.setText(f"Update Available (v{remote_version})")
        self._update_btn.setToolTip(
            f"A new version (v{remote_version}) is available.\n"
            "Click to save your session and run the updater."
        )
        # Set minimum width from text metrics to prevent Qt corner widget squish
        self._update_btn.setMinimumWidth(self._update_btn.sizeHint().width())
        self._update_btn.setVisible(True)

    def _run_update(self) -> None:
        reply = QMessageBox.question(
            self, "Update EZ-CorridorKey",
            "This will save your session, close the app, and run the updater.\n"
            "The app will relaunch automatically after updating.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return
        # Save session before closing
        self._auto_save_session()
        # Launch update script --relaunch detached, then quit
        import subprocess
        root = os.path.dirname(os.path.dirname(__file__))
        if os.name == "nt":
            bat = os.path.join(root, "3-update.bat")
            subprocess.Popen(
                ["cmd", "/c", "start", "", bat, "--relaunch"],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            sh = os.path.join(root, "3-update.sh")
            subprocess.Popen(
                [sh, "--relaunch"],
                start_new_session=True,
            )
        from PySide6.QtWidgets import QApplication
        QApplication.instance().quit()

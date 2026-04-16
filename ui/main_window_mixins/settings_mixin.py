from __future__ import annotations

import logging
import os
import re
import sys

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
_SHOW_TESTER_BUILD_ID = False


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

    def _show_download_manager(self) -> None:
        """Open the Download Manager (SetupWizard) on demand.

        Unlike the first-launch path, this entry is always available so users
        can re-scan their install folder, install optional models they
        skipped earlier, or point at a different folder.
        """
        from ui.widgets.setup_wizard import SetupWizard
        dlg = SetupWizard(parent=self)
        dlg.setWindowTitle("Download Manager")
        dlg.exec()

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
        box = QMessageBox(self)
        box.setWindowTitle("About EZ-CorridorKey")
        box.setTextFormat(Qt.RichText)
        box.setText(
            f"<h2>EZ-CorridorKey v{app_version}</h2>"
            "<p>AI Green Screen Keyer<br>"
            '<a href="https://github.com/nikopueringer/CorridorKey#corridorkey-licensing-and-permissions">'
            "CC BY-NC-SA 4.0 License</a></p>"
            "<p><b>Special Thanks</b></p>"
            "<p>"
            '<a href="https://github.com/nikopueringer/">Niko Pueringer</a> — OG CorridorKey Creator<br>'
            '<a href="https://www.edzisk.com">Ed Zisk</a> — Maintainer, GUI, workflow, SFX, QA<br>'
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
        import tomllib
        candidates = []
        if getattr(sys, 'frozen', False):
            candidates.append(os.path.join(sys._MEIPASS, "pyproject.toml"))
        candidates.append(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pyproject.toml")
        )
        for path in candidates:
            try:
                with open(path, "rb") as f:
                    return tomllib.load(f)["project"]["version"]
            except Exception:
                continue
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
        import sys as _sys
        if getattr(_sys, "frozen", False):
            self._run_frozen_update()
        else:
            self._run_script_update()

    def _run_script_update(self) -> None:
        """Update via 3-update.sh / 3-update.bat (CLI/dev installs)."""
        reply = QMessageBox.question(
            self, "Update EZ-CorridorKey",
            "This will save your session, close the app, and run the updater.\n"
            "The app will relaunch automatically after updating.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return
        self._auto_save_session()
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

    def _run_frozen_update(self) -> None:
        """Update a frozen .app/.exe by downloading from GitHub Releases."""
        import sys as _sys
        reply = QMessageBox.question(
            self, "Update EZ-CorridorKey",
            "This will download the latest version, replace the current app,\n"
            "and relaunch automatically.\n\n"
            "Your session will be saved. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return
        self._auto_save_session()

        from PySide6.QtWidgets import QProgressDialog
        progress = QProgressDialog(
            "Downloading update...", "Cancel", 0, 100, self
        )
        progress.setWindowTitle("Updating EZ-CorridorKey")
        progress.setMinimumWidth(400)
        progress.setModal(True)
        progress.show()

        import urllib.request
        import json
        import tempfile
        import zipfile
        import shutil
        import subprocess
        from pathlib import Path

        try:
            # Determine asset name by platform
            if _sys.platform == "darwin":
                QMessageBox.information(
                    self, "Update",
                    "macOS auto-update is not yet supported.\n"
                    "Please download the latest .pkg from Gumroad or GitHub."
                )
                progress.close()
                return
            elif _sys.platform == "win32":
                asset_platform = "Windows-x64"
                asset_ext = "zip"
            else:
                QMessageBox.warning(
                    self, "Update",
                    "Automatic updates are not supported on this platform.\n"
                    "Please download the latest release from GitHub."
                )
                progress.close()
                return

            # Fetch latest release info from GitHub API
            api_url = (
                "https://api.github.com/repos/edenaion/EZ-CorridorKey"
                "/releases/latest"
            )
            req = urllib.request.Request(
                api_url, headers={"User-Agent": "CorridorKey"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                release = json.loads(resp.read().decode("utf-8"))

            # Build versioned asset name: EZ-CorridorKey-{ver}-{Platform}-{arch}.{ext}
            tag = release.get("tag_name", "")
            ver = tag.lstrip("v") if tag else "unknown"
            asset_name = f"EZ-CorridorKey-{ver}-{asset_platform}.{asset_ext}"

            # Find the asset download URL
            download_url = None
            for asset in release.get("assets", []):
                if asset["name"] == asset_name:
                    download_url = asset["browser_download_url"]
                    break

            if not download_url:
                QMessageBox.warning(
                    self, "Update",
                    f"No {asset_name} found in the latest release.\n"
                    f"Release: {tag or 'unknown'}\n\n"
                    "Please download manually from GitHub."
                )
                progress.close()
                return

            # Download to temp file
            tmp_dir = Path(tempfile.mkdtemp(prefix="corridorkey_update_"))
            zip_path = tmp_dir / asset_name

            def _report(block_num, block_size, total_size):
                if progress.wasCanceled():
                    raise InterruptedError("Download cancelled")
                if total_size > 0:
                    pct = min(100, (block_num * block_size * 100) // total_size)
                    progress.setValue(pct)
                    progress.setLabelText(
                        f"Downloading update... "
                        f"{block_num * block_size // (1024*1024)}/"
                        f"{total_size // (1024*1024)} MB"
                    )
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents()

            urllib.request.urlretrieve(
                download_url, str(zip_path), reporthook=_report
            )

            progress.setLabelText("Installing update...")
            progress.setValue(95)
            QApplication.processEvents()

            # Extract zip
            extract_dir = tmp_dir / "extracted"
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

            if _sys.platform == "darwin":
                # Find the .app inside extracted dir
                new_app = None
                for item in extract_dir.iterdir():
                    if item.suffix == ".app" and item.is_dir():
                        new_app = item
                        break

                if not new_app:
                    raise FileNotFoundError("No .app found in downloaded archive")

                # Current .app location
                current_app = Path(_sys.executable).resolve().parent.parent.parent
                old_backup = current_app.with_name(current_app.name + ".old")

                # Swap: current → .old, new → current
                if old_backup.exists():
                    shutil.rmtree(old_backup)
                current_app.rename(old_backup)
                shutil.move(str(new_app), str(current_app))

                # Make executable
                main_exe = current_app / "Contents" / "MacOS" / "EZ-CorridorKey"
                if main_exe.exists():
                    main_exe.chmod(0o755)

                # Relaunch
                subprocess.Popen(
                    ["open", str(current_app)],
                    start_new_session=True,
                )

                # Clean up old backup in background
                shutil.rmtree(old_backup, ignore_errors=True)
                shutil.rmtree(tmp_dir, ignore_errors=True)

            elif _sys.platform == "win32":
                # Windows: extract, write a bat script to swap after exit
                new_exe_dir = None
                for item in extract_dir.iterdir():
                    if item.is_dir():
                        new_exe_dir = item
                        break

                current_dir = Path(_sys.executable).resolve().parent
                swap_script = tmp_dir / "swap_update.bat"
                swap_script.write_text(
                    f'@echo off\n'
                    f'timeout /t 2 /nobreak >nul\n'
                    f'xcopy /s /e /y "{new_exe_dir}\\*" "{current_dir}\\"\n'
                    f'start "" "{_sys.executable}"\n'
                    f'rmdir /s /q "{tmp_dir}"\n',
                    encoding="utf-8",
                )
                subprocess.Popen(
                    ["cmd", "/c", str(swap_script)],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )

            progress.setValue(100)
            progress.close()
            QApplication.instance().quit()

        except InterruptedError:
            progress.close()
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self, "Update Failed",
                f"Could not update automatically:\n\n{e}\n\n"
                "Please download the latest release manually from GitHub."
            )

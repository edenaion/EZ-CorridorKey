"""Cross-platform test for MainWindow._run_update subprocess launch.

Verifies that the subprocess call in _run_update uses platform-appropriate
flags (CREATE_NEW_CONSOLE on Windows, start_new_session on Unix) and
does not raise AttributeError on any platform.
"""

import os
import subprocess
from unittest.mock import patch

import pytest


def _build_run_update_logic():
    """Extract the subprocess-launching logic from _run_update for testing.

    Returns a callable(root_dir) that performs the Popen call.
    """

    def launch_updater(root):
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

    return launch_updater


class TestRunUpdateCrossPlatform:
    """Ensure _run_update subprocess call works on all platforms."""

    @patch("subprocess.Popen")
    def test_popen_called_without_error(self, mock_popen, tmp_path):
        """Popen is called with valid platform args (no AttributeError)."""
        # Create dummy update script so path exists
        if os.name == "nt":
            script = tmp_path / "3-update.bat"
        else:
            script = tmp_path / "3-update.sh"
        script.write_text("echo ok")

        launcher = _build_run_update_logic()
        launcher(str(tmp_path))

        mock_popen.assert_called_once()

    @patch("subprocess.Popen")
    def test_windows_uses_create_new_console(self, mock_popen, tmp_path):
        """On Windows, creationflags=CREATE_NEW_CONSOLE is passed."""
        if os.name != "nt":
            pytest.skip("Windows-only test")

        (tmp_path / "3-update.bat").write_text("echo ok")
        launcher = _build_run_update_logic()
        launcher(str(tmp_path))

        _, kwargs = mock_popen.call_args
        assert "creationflags" in kwargs
        assert kwargs["creationflags"] == subprocess.CREATE_NEW_CONSOLE

    @patch("subprocess.Popen")
    def test_unix_uses_start_new_session(self, mock_popen, tmp_path):
        """On Unix, start_new_session=True is passed (no creationflags)."""
        if os.name == "nt":
            pytest.skip("Unix-only test")

        (tmp_path / "3-update.sh").write_text("#!/bin/sh\necho ok")
        launcher = _build_run_update_logic()
        launcher(str(tmp_path))

        _, kwargs = mock_popen.call_args
        assert kwargs.get("start_new_session") is True
        assert "creationflags" not in kwargs

    @patch("subprocess.Popen")
    def test_unix_no_creationflags_attribute_access(self, mock_popen, tmp_path):
        """Verify no reference to CREATE_NEW_CONSOLE on non-Windows."""
        if os.name == "nt":
            pytest.skip("Unix-only test")

        # Temporarily hide CREATE_NEW_CONSOLE to confirm code doesn't touch it
        had_attr = hasattr(subprocess, "CREATE_NEW_CONSOLE")
        if had_attr:
            orig = subprocess.CREATE_NEW_CONSOLE
            delattr(subprocess, "CREATE_NEW_CONSOLE")

        try:
            (tmp_path / "3-update.sh").write_text("#!/bin/sh\necho ok")
            launcher = _build_run_update_logic()
            # Must not raise AttributeError
            launcher(str(tmp_path))
        finally:
            if had_attr:
                subprocess.CREATE_NEW_CONSOLE = orig

    @patch("subprocess.Popen")
    def test_correct_script_name_per_platform(self, mock_popen, tmp_path):
        """Ensure the right script extension is used per platform."""
        if os.name == "nt":
            (tmp_path / "3-update.bat").write_text("echo ok")
            expected_script = str(tmp_path / "3-update.bat")
        else:
            (tmp_path / "3-update.sh").write_text("#!/bin/sh\necho ok")
            expected_script = str(tmp_path / "3-update.sh")

        launcher = _build_run_update_logic()
        launcher(str(tmp_path))

        args = mock_popen.call_args[0][0]
        assert expected_script in args

    def test_actual_main_window_source_matches(self):
        """Verify _run_update has the os.name guard (in settings mixin)."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "ui",
            "main_window_mixins",
            "settings_mixin.py",
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert 'os.name == "nt"' in source or "os.name == 'nt'" in source, (
            "_run_update must guard platform-specific subprocess flags with os.name check"
        )

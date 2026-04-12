"""Tests for install-method detection in the Report Issue dialog.

The detected label shows up in every filed bug report and removes the
guesswork around how the user actually installed the app. These tests
cover the three main branches (frozen installer, frozen .app, dev
clone) by mocking the attributes the function reads.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

from ui.widgets.report_issue_dialog import _detect_install_method


class TestInstallMethodDetection:
    def test_dev_git_clone_in_venv(self):
        """The dev environment running this test suite should report a git
        clone in a venv — this is the ground-truth case."""
        result = _detect_install_method()
        assert "git clone" in result
        assert "venv" in result

    def test_frozen_windows_installer(self, tmp_path):
        """Frozen sys.frozen=True + _internal sibling → Windows Installer."""
        exe_dir = tmp_path / "EZ-CorridorKey"
        exe_dir.mkdir()
        (exe_dir / "_internal").mkdir()
        fake_exe = str(exe_dir / "EZ-CorridorKey.exe")

        with patch.object(sys, "frozen", True, create=True), \
             patch.object(sys, "executable", fake_exe), \
             patch.object(sys, "platform", "win32"):
            assert _detect_install_method() == "Windows Installer (.exe)"

    def test_frozen_portable_wins_over_installer(self, tmp_path):
        """portable.txt marker trumps other frozen-layout heuristics."""
        exe_dir = tmp_path / "EZ-CorridorKey-Portable"
        exe_dir.mkdir()
        (exe_dir / "_internal").mkdir()
        (exe_dir / "portable.txt").write_text("portable mode")
        fake_exe = str(exe_dir / "EZ-CorridorKey.exe")

        with patch.object(sys, "frozen", True, create=True), \
             patch.object(sys, "executable", fake_exe), \
             patch.object(sys, "platform", "win32"):
            assert _detect_install_method() == "Portable (frozen)"

    def test_frozen_macos_app(self, tmp_path):
        """macOS .app bundle detection via path substring."""
        app_inner = tmp_path / "EZ-CorridorKey.app" / "Contents" / "MacOS"
        app_inner.mkdir(parents=True)
        fake_exe = str(app_inner / "EZ-CorridorKey")

        with patch.object(sys, "frozen", True, create=True), \
             patch.object(sys, "executable", fake_exe), \
             patch.object(sys, "platform", "darwin"):
            assert _detect_install_method() == "macOS .app (pkg/dmg)"

    def test_frozen_windows_layout_unknown(self, tmp_path):
        """Frozen Windows with no _internal sibling and no portable
        marker — we return a conservative unknown-layout label rather
        than misreporting it as an installer."""
        exe_dir = tmp_path / "BareFrozen"
        exe_dir.mkdir()
        fake_exe = str(exe_dir / "EZ-CorridorKey.exe")

        with patch.object(sys, "frozen", True, create=True), \
             patch.object(sys, "executable", fake_exe), \
             patch.object(sys, "platform", "win32"):
            assert _detect_install_method() == "Windows frozen (layout unknown)"

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

    def test_github_source_zip_no_git(self, tmp_path, monkeypatch):
        """Source install without .git/ (the GitHub source zip path) should
        be labeled distinctly so triage can recommend git clone. We point
        __file__ at a sandbox that has no .git/ anywhere up the chain and
        verify the function walks out cleanly.
        """
        # Build a fake project tree and place an ``import as`` stand-in
        # for __file__. The detection function walks the real __file__'s
        # ancestors, so monkeypatch the module-global that it reads.
        fake_widget_dir = tmp_path / "fake_proj" / "ui" / "widgets"
        fake_widget_dir.mkdir(parents=True)
        fake_file = fake_widget_dir / "report_issue_dialog.py"
        fake_file.write_text("# stub for detection test")

        # There must be NO .git/ anywhere up the chain. tmp_path is a
        # fresh fixture so that's already true, but assert it.
        probe = fake_widget_dir
        while True:
            assert not (probe / ".git").exists()
            parent = probe.parent
            if parent == probe:
                break
            probe = parent

        import ui.widgets.report_issue_dialog as mod
        monkeypatch.setattr(mod, "__file__", str(fake_file))
        # Ensure sys.frozen is not set so we take the dev branch.
        monkeypatch.delattr(sys, "frozen", raising=False)

        result = mod._detect_install_method()
        assert "GitHub source zip" in result
        assert "no .git" in result

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

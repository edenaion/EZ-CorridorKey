"""Fresh-install smoke checks for CorridorKey.

This script is intentionally narrow:
- verify core imports
- verify settings/model setup helpers import
- create the QApplication
- create the MainWindow headlessly
- open the Preferences dialog
- shut down cleanly

It is meant for CI and post-install smoke tests, not full GPU QA.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path


def _print_step(message: str) -> None:
    print(f"[smoke] {message}", flush=True)


def _prepare_environment() -> Path:
    """Set deterministic env vars so GUI startup works in CI/headless runs."""
    temp_root = Path(tempfile.mkdtemp(prefix="corridorkey_smoke_"))
    config_root = temp_root / "config"
    app_root = temp_root / "app"
    config_root.mkdir(parents=True, exist_ok=True)
    app_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("CORRIDORKEY_SKIP_STARTUP_DIAGNOSTICS", "1")
    os.environ.setdefault("CORRIDORKEY_SKIP_UPDATE_CHECK", "1")
    os.environ.setdefault("QT_QUICK_BACKEND", "software")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if sys.platform != "win32":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    if os.name == "nt":
        os.environ["APPDATA"] = str(config_root)
    else:
        os.environ["XDG_CONFIG_HOME"] = str(config_root)

    return app_root


def _exercise_ffmpeg_repair_button(app, prefs) -> None:
    """Click the Preferences FFmpeg repair button with stubbed backend behavior."""
    _print_step("smoke-testing FFmpeg repair controls")

    import backend.ffmpeg_tools as ffmpeg_tools
    from PySide6.QtWidgets import QMessageBox

    if "download and install a full bundled FFmpeg build" not in prefs._repair_ffmpeg_btn.toolTip():
        raise RuntimeError("Repair FFmpeg tooltip text missing expected explanation")
    if "bundled FFmpeg folder" not in prefs._open_ffmpeg_btn.toolTip():
        raise RuntimeError("Open FFmpeg Folder tooltip text missing expected explanation")

    state = {"ok": False}
    dialogs: list[tuple[str, str]] = []

    def fake_validate(require_probe: bool = True):
        if state["ok"]:
            return ffmpeg_tools.FFmpegValidationResult(
                ok=True,
                message="FFmpeg OK: smoke repair complete",
                ffmpeg_path="ffmpeg",
                ffprobe_path="ffprobe",
            )
        return ffmpeg_tools.FFmpegValidationResult(
            ok=False,
            message="FFmpeg smoke test: invalid install",
        )

    def fake_help() -> str:
        return (
            "Install FFmpeg with your package manager\n\n"
            "Then verify:\n"
            "    ffmpeg -version\n"
            "    ffprobe -version"
        )

    def fake_repair(progress_callback=None):
        if progress_callback:
            progress_callback("Downloading FFmpeg", 64, 128)
            progress_callback("Validating FFmpeg", 128, 128)
        state["ok"] = True
        return ffmpeg_tools.FFmpegValidationResult(
            ok=True,
            message="FFmpeg OK: smoke repair complete",
            ffmpeg_path="ffmpeg",
            ffprobe_path="ffprobe",
        )

    original_validate = ffmpeg_tools.validate_ffmpeg_install
    original_help = ffmpeg_tools.get_ffmpeg_install_help
    original_repair = ffmpeg_tools.repair_ffmpeg_install
    original_info = QMessageBox.information
    original_question = QMessageBox.question
    original_critical = QMessageBox.critical

    try:
        ffmpeg_tools.validate_ffmpeg_install = fake_validate
        ffmpeg_tools.get_ffmpeg_install_help = fake_help
        ffmpeg_tools.repair_ffmpeg_install = fake_repair

        QMessageBox.information = lambda parent, title, text: dialogs.append((title, text)) or QMessageBox.Ok
        QMessageBox.critical = lambda parent, title, text: dialogs.append((title, text)) or QMessageBox.Ok
        QMessageBox.question = lambda parent, title, text: dialogs.append((title, text)) or QMessageBox.Yes

        prefs._refresh_ffmpeg_status()
        prefs._repair_ffmpeg_btn.click()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
            worker = getattr(prefs, "_ffmpeg_repair_worker", None)
            if worker is None or not worker.isRunning():
                break
        app.processEvents()

        if sys.platform in ("win32", "darwin"):
            if not state["ok"]:
                raise RuntimeError("Repair FFmpeg button did not trigger the repair path")
            if "smoke repair complete" not in prefs._ffmpeg_status_label.text():
                raise RuntimeError("Repair FFmpeg button did not refresh the success status")
        else:
            clip_text = app.clipboard().text()
            if "ffprobe -version" not in clip_text:
                raise RuntimeError("Repair FFmpeg button did not copy install guidance")
            if not dialogs or dialogs[-1][0] != "Repair FFmpeg":
                raise RuntimeError("Repair FFmpeg button did not show the guidance dialog")
    finally:
        ffmpeg_tools.validate_ffmpeg_install = original_validate
        ffmpeg_tools.get_ffmpeg_install_help = original_help
        ffmpeg_tools.repair_ffmpeg_install = original_repair
        QMessageBox.information = original_info
        QMessageBox.question = original_question
        QMessageBox.critical = original_critical


def run_smoke() -> None:
    """Run the import/startup smoke sequence."""
    app_root = _prepare_environment()

    _print_step("importing backend and UI modules")
    from backend.project import set_app_dir
    from backend import CorridorKeyService
    from ui.app import create_app
    from ui.main_window import MainWindow
    from ui.recent_sessions import RecentSessionsStore
    from ui.widgets.preferences_dialog import PreferencesDialog

    set_app_dir(str(app_root))

    _print_step("constructing backend service")
    service = CorridorKeyService()
    device = service.detect_device()
    _print_step(f"device detected: {device}")

    _print_step("creating QApplication")
    app = create_app([])

    _print_step("creating recent sessions store")
    store = RecentSessionsStore()
    store.prune_missing()

    _print_step("creating main window")
    window = MainWindow(service, store)
    window.show()

    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    if window.windowTitle() != "CORRIDORKEY":
        raise RuntimeError(f"Unexpected main window title: {window.windowTitle()!r}")

    _print_step("opening preferences dialog")
    prefs = PreferencesDialog(window)
    _exercise_ffmpeg_repair_button(app, prefs)
    app.processEvents()
    prefs.close()

    _print_step("closing main window")
    window.close()
    app.processEvents()

    _print_step("unloading engines")
    service.unload_engines()
    app.processEvents()

    _print_step("smoke test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="CorridorKey fresh-install smoke test")
    parser.parse_args()
    run_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

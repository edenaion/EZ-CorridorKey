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

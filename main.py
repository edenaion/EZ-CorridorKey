"""CorridorKey entry point — GUI (default) or CLI mode.

Usage:
    python main.py              # Launch GUI (default)
    python main.py --gui        # Launch GUI explicitly
    python main.py --cli        # Run CLI wizard (original clip_manager.py)
"""
from __future__ import annotations

import os
# Enable OpenEXR support in OpenCV — must be set before cv2 is imported anywhere
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import logging
import logging.handlers
import sys
from datetime import datetime


class _NullTextStream:
    """Minimal text stream used when GUI launches without a console."""

    encoding = "utf-8"
    errors = "replace"

    def write(self, data) -> int:
        if data is None:
            return 0
        if isinstance(data, bytes):
            return len(data)
        return len(str(data))

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


def ensure_standard_streams() -> None:
    """Provide harmless stdout/stderr sinks for pythonw/PyInstaller GUI launches."""
    if sys.stdout is None:
        sys.stdout = _NullTextStream()
    if sys.stderr is None:
        sys.stderr = _NullTextStream()


def get_base_dir() -> str:
    """Get the project base directory, handling both dev and frozen (PyInstaller) modes.

    In development: returns the directory containing this file.
    In frozen build: returns sys._MEIPASS (PyInstaller temp dir) for bundled
    resources, or the executable's directory for user files.
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def get_app_dir() -> str:
    """Get the application directory (where the .exe lives in frozen mode).

    Use this for user-facing paths (logs, sessions, etc.).
    Use get_base_dir() for bundled resources (checkpoints, QSS, fonts).
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


# Ensure project root is on path
sys.path.insert(0, get_base_dir())


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with dual output: console + per-session file.

    Console respects --log-level flag. File always captures DEBUG.
    All timestamps use the system's local time.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"

    # Console handler — respects --log-level
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    # File handler — session-named log, always DEBUG
    log_dir = os.path.join(get_app_dir(), "logs", "backend")
    os.makedirs(log_dir, exist_ok=True)

    session_ts = datetime.now().strftime("%y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{session_ts}_corridorkey.log")

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=3,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    # Root logger — let handlers filter
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)


def run_gui() -> int:
    """Launch the PySide6 desktop application."""
    ensure_standard_streams()
    # The GUI has its own progress UI; external library progress bars just create
    # stderr/console issues in pythonw and frozen launches.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from ui.app import create_app
    from ui.main_window import MainWindow
    from ui.recent_sessions import RecentSessionsStore
    from backend import CorridorKeyService

    app = create_app()
    service = CorridorKeyService()
    store = RecentSessionsStore()
    store.prune_missing()
    window = MainWindow(service, store)
    window.show()
    rc = app.exec()
    # Force-exit: torch.compile spawns Triton background threads that prevent
    # clean shutdown, leaving a zombie process holding VRAM.
    os._exit(rc)


def run_cli() -> int:
    """Run the original CLI wizard from clip_manager.py as a subprocess.

    Forwards all extra CLI arguments (e.g. --action wizard --win_path ...)
    directly to the upstream script, preserving 100% original behaviour.
    """
    script = os.path.join(get_base_dir(), "clip_manager.py")
    if not os.path.isfile(script):
        logging.error(
            "CLI mode requires clip_manager.py in the project root. "
            "Use --gui (default) for the graphical interface."
        )
        return 1

    import subprocess

    # Forward everything after --cli to clip_manager.py.
    # sys.argv looks like: ['main.py', '--cli', '--action', 'wizard', ...]
    # We strip our own flags (--cli, --gui, --log-level <val>) and pass the rest.
    forwarded: list[str] = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--cli", "--gui"):
            continue
        if arg == "--log-level":
            skip_next = True  # skip the value that follows
            continue
        forwarded.append(arg)

    cmd = [sys.executable, script] + forwarded
    logging.info("CLI passthrough: %s", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    ensure_standard_streams()
    parser = argparse.ArgumentParser(
        description="CorridorKey — AI Green Screen Keyer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run the original CLI wizard instead of the GUI",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="Launch the GUI (default)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    # parse_known_args so CLI-mode flags (--action, --win_path, etc.)
    # pass through to clip_manager.py without error.
    args, _unknown = parser.parse_known_args()
    setup_logging(args.log_level)

    # Configure backend with the application directory
    from backend.project import set_app_dir
    set_app_dir(get_app_dir())

    if args.cli:
        return run_cli()
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())

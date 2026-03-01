"""CorridorKey entry point — GUI (default) or CLI mode.

Usage:
    python main.py              # Launch GUI (default)
    python main.py --gui        # Launch GUI explicitly
    python main.py --cli        # Run CLI wizard (original clip_manager.py)
"""
from __future__ import annotations

import argparse
import logging
import sys
import os


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
    """Configure logging with timestamped format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_gui() -> int:
    """Launch the PySide6 desktop application."""
    from ui.app import create_app
    from ui.main_window import MainWindow
    from backend import CorridorKeyService

    app = create_app()
    service = CorridorKeyService()
    window = MainWindow(service)
    window.show()
    return app.exec()


def run_cli() -> int:
    """Run the original CLI wizard from clip_manager.py.

    Falls back gracefully if the CLI wizard isn't available.
    """
    try:
        import clip_manager
        if hasattr(clip_manager, 'main'):
            clip_manager.main()
        elif hasattr(clip_manager, 'run'):
            clip_manager.run()
        else:
            logging.error("clip_manager module found but has no main() or run() function")
            return 1
        return 0
    except ImportError:
        logging.error(
            "CLI mode requires clip_manager.py in the project root. "
            "Use --gui (default) for the graphical interface."
        )
        return 1
    except Exception as e:
        logging.error(f"CLI error: {e}")
        return 1


def main() -> int:
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

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.cli:
        return run_cli()
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())

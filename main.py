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

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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
    """Run the original CLI wizard from clip_manager.py."""
    try:
        from clip_manager import main as cli_main
        cli_main()
        return 0
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

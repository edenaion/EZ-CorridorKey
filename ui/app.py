"""QApplication setup with Corridor Digital brand theme and bundled fonts.

Fonts:
  - Gagarin: Logo / brand mark text (Corridor Digital identity font)
  - Open Sans: All secondary / body UI text (default app font)
"""
from __future__ import annotations

import sys
import os
import logging

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont, QIcon
from PySide6.QtCore import Qt

from ui.theme import load_stylesheet

logger = logging.getLogger(__name__)


def create_app(argv: list[str] | None = None) -> QApplication:
    """Create and configure the QApplication with brand theming.

    Returns a QApplication instance ready for main window creation.
    """
    if argv is None:
        argv = sys.argv

    # Force software rendering — zero GPU overhead for UI
    os.environ["QT_QUICK_BACKEND"] = "software"

    app = QApplication(argv)
    app.setApplicationName("CorridorKey")
    app.setOrganizationName("Corridor Digital")

    # ── Font loading (frozen-build aware) ──
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
        fonts_dir = os.path.join(base, "ui", "theme", "fonts")
    else:
        base = os.path.dirname(__file__)
        fonts_dir = os.path.join(base, "theme", "fonts")

    # Register all bundled fonts (Gagarin + Open Sans)
    opensans_loaded = False
    gagarin_loaded = False
    if os.path.isdir(fonts_dir):
        for fname in os.listdir(fonts_dir):
            if not fname.lower().endswith((".ttf", ".otf")):
                continue
            font_id = QFontDatabase.addApplicationFont(os.path.join(fonts_dir, fname))
            if font_id >= 0:
                families = QFontDatabase.applicationFontFamilies(font_id)
                if "Open Sans" in families:
                    opensans_loaded = True
                if "Gagarin" in families:
                    gagarin_loaded = True

    # Fallback: search system font dirs for Open Sans
    if not opensans_loaded:
        system_font_dirs = [
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/Library/Fonts"),
            "/System/Library/Fonts",
            "C:/Windows/Fonts",
        ]
        for font_dir in system_font_dirs:
            if not os.path.isdir(font_dir):
                continue
            for fname in os.listdir(font_dir):
                if "opensans" in fname.lower() and fname.lower().endswith((".ttf", ".otf")):
                    font_id = QFontDatabase.addApplicationFont(os.path.join(font_dir, fname))
                    if font_id >= 0:
                        opensans_loaded = True

    # Set default app font (Open Sans for all body text)
    if opensans_loaded:
        app.setFont(QFont("Open Sans", 13))
    else:
        logger.info("Open Sans not found, using system sans-serif")
        fallback = "Segoe UI" if sys.platform == "win32" else "Helvetica"
        app.setFont(QFont(fallback, 13))

    if gagarin_loaded:
        logger.info("Gagarin font loaded for brand mark")
    else:
        logger.warning("Gagarin font not found — brand mark will use fallback")

    # Apply brand stylesheet
    app.setStyleSheet(load_stylesheet())

    # Set app icon (window title bar + taskbar)
    icon_path = os.path.join(base, "theme", "corridorkey.png") if not getattr(sys, 'frozen', False) else os.path.join(base, "ui", "theme", "corridorkey.png")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    return app

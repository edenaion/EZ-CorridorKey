"""QApplication setup with Corridor Digital brand theme and Open Sans font."""
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

    # Load Open Sans font (frozen-build aware)
    font_loaded = False
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(__file__)
    font_search_paths = [
        os.path.join(base, "theme", "fonts") if not getattr(sys, 'frozen', False) else os.path.join(base, "ui", "theme", "fonts"),
        os.path.expanduser("~/.fonts"),                    # Linux
        os.path.expanduser("~/Library/Fonts"),             # macOS
        "/System/Library/Fonts",                           # macOS system
        "C:/Windows/Fonts",                                # Windows
    ]
    for font_dir in font_search_paths:
        if not os.path.isdir(font_dir):
            continue
        for fname in os.listdir(font_dir):
            if "opensans" in fname.lower() and fname.lower().endswith((".ttf", ".otf")):
                font_id = QFontDatabase.addApplicationFont(os.path.join(font_dir, fname))
                if font_id >= 0:
                    font_loaded = True

    if font_loaded:
        app.setFont(QFont("Open Sans", 13))
    else:
        # Fallback — use system sans-serif
        logger.info("Open Sans not found, using system sans-serif")
        fallback = "Segoe UI" if sys.platform == "win32" else "Helvetica"
        app.setFont(QFont(fallback, 13))

    # Apply brand stylesheet
    app.setStyleSheet(load_stylesheet())

    # Set app icon (window title bar + taskbar)
    icon_path = os.path.join(base, "theme", "corridorkey.png") if not getattr(sys, 'frozen', False) else os.path.join(base, "ui", "theme", "corridorkey.png")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    return app

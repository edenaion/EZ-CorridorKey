"""CorridorKey theme — dark-only Corridor Digital brand identity."""

import os
import sys

# Frozen-build aware path resolution
if getattr(sys, "frozen", False):
    THEME_DIR = os.path.join(sys._MEIPASS, "ui", "theme")
else:
    THEME_DIR = os.path.dirname(os.path.abspath(__file__))
QSS_PATH = os.path.join(THEME_DIR, "corridor_theme.qss")


def load_stylesheet() -> str:
    """Load the brand QSS stylesheet.

    Resolves {{THEME_DIR}} placeholders to the actual theme directory
    so QSS url() references work regardless of working directory.
    """
    with open(QSS_PATH, "r", encoding="utf-8") as f:
        qss = f.read()
    # Forward slashes required in Qt QSS url() even on Windows
    theme_path = THEME_DIR.replace("\\", "/")
    return qss.replace("{{THEME_DIR}}", theme_path)

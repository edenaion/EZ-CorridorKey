"""CorridorKey theme — dark-only Corridor Digital brand identity."""

import os

THEME_DIR = os.path.dirname(os.path.abspath(__file__))
QSS_PATH = os.path.join(THEME_DIR, "corridor_theme.qss")


def load_stylesheet() -> str:
    """Load the brand QSS stylesheet."""
    with open(QSS_PATH, "r", encoding="utf-8") as f:
        return f.read()

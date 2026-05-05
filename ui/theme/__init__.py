"""CorridorKey theme — dark-only Corridor Digital brand identity."""

import os
import sys

# Frozen-build aware path resolution
if getattr(sys, 'frozen', False):
    THEME_DIR = os.path.join(sys._MEIPASS, "ui", "theme")
else:
    THEME_DIR = os.path.dirname(os.path.abspath(__file__))
QSS_PATH = os.path.join(THEME_DIR, "corridor_theme.qss")

# Accent color palettes keyed by screen color
ACCENT_COLORS = {
    "green": {
        "accent":         "#2CC350",
        "accent_hover":   "#3DD662",
        "accent_pressed": "#22A040",
    },
    "blue": {
        "accent":         "#0082CF",
        "accent_hover":   "#1A9AE6",
        "accent_pressed": "#006AAB",
    },
}

# Cache the raw QSS template (before accent substitution)
_qss_template: str | None = None


def _load_qss_template() -> str:
    """Load the QSS file and replace hardcoded green accents with placeholders."""
    global _qss_template
    if _qss_template is not None:
        return _qss_template

    with open(QSS_PATH, "r", encoding="utf-8") as f:
        qss = f.read()

    theme_path = THEME_DIR.replace("\\", "/")
    qss = qss.replace("{{THEME_DIR}}", theme_path)

    # Replace hardcoded green accent colors with placeholders
    qss = qss.replace("#2CC350", "{{ACCENT}}")
    qss = qss.replace("#2cc350", "{{ACCENT}}")
    qss = qss.replace("#3DD662", "{{ACCENT_HOVER}}")
    qss = qss.replace("#3dd662", "{{ACCENT_HOVER}}")
    qss = qss.replace("#22A040", "{{ACCENT_PRESSED}}")
    qss = qss.replace("#22a040", "{{ACCENT_PRESSED}}")
    # Also handle the progress bar gradient green start color
    qss = qss.replace("#22C55E", "{{ACCENT}}")
    qss = qss.replace("#22c55e", "{{ACCENT}}")

    _qss_template = qss
    return _qss_template


def load_stylesheet(screen_color: str = "green") -> str:
    """Load the brand QSS stylesheet with accent colors for the given screen color.

    Args:
        screen_color: "green" or "blue". Controls slider handles, buttons,
                      and other accent-colored elements.
    """
    template = _load_qss_template()
    palette = ACCENT_COLORS.get(screen_color, ACCENT_COLORS["green"])
    return (
        template
        .replace("{{ACCENT}}", palette["accent"])
        .replace("{{ACCENT_HOVER}}", palette["accent_hover"])
        .replace("{{ACCENT_PRESSED}}", palette["accent_pressed"])
    )

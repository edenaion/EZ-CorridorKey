import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed")
from PySide6.QtWidgets import QApplication

from ui.preview.frame_index import ViewMode
from ui.widgets.view_mode_bar import ViewModeBar


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_fallback_prefers_alpha_before_input_when_comp_unavailable():
    _app()
    bar = ViewModeBar()

    bar.set_available_modes([ViewMode.INPUT, ViewMode.ALPHA])

    assert bar.current_mode() == ViewMode.ALPHA

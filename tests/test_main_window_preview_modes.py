import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed")
from ui.main_window import MainWindow
from ui.preview.frame_index import ViewMode


class _DummyDualViewer:
    def __init__(self, mode: ViewMode):
        self.current_output_mode = mode
        self.previews: list[object] = []

    def show_reprocess_preview(self, qimage) -> None:
        self.previews.append(qimage)


def test_reprocess_preview_does_not_override_input_mode():
    window = MainWindow.__new__(MainWindow)
    window._dual_viewer = _DummyDualViewer(ViewMode.INPUT)

    MainWindow._on_reprocess_result(
        window,
        "job-1",
        {"comp": np.zeros((4, 4, 3), dtype=np.float32)},
    )

    assert window._dual_viewer.previews == []


def test_reprocess_preview_does_not_override_alpha_mode():
    window = MainWindow.__new__(MainWindow)
    window._dual_viewer = _DummyDualViewer(ViewMode.ALPHA)

    MainWindow._on_reprocess_result(
        window,
        "job-2",
        {"comp": np.zeros((4, 4, 3), dtype=np.float32)},
    )

    assert window._dual_viewer.previews == []

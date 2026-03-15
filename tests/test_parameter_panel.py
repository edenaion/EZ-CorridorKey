import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed")
from PySide6.QtWidgets import QApplication

from ui.widgets.parameter_panel import ParameterPanel


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_color_space_tooltip_explains_preview_and_export_behavior():
    _app()
    panel = ParameterPanel()

    tooltip = panel._color_space.toolTip()

    assert "left INPUT viewer" in tooltip
    assert "future exports" in tooltip
    assert "does not rewrite those files on disk" in tooltip
    assert "rerun inference to save new outputs" in tooltip
    assert panel._color_space_label.toolTip() == tooltip


def test_live_preview_tooltip_mentions_engine_warmup_and_saved_outputs():
    _app()
    panel = ParameterPanel()

    tooltip = panel._live_preview.toolTip()

    assert "first preview change may take a moment" in tooltip
    assert "inference engine loads" in tooltip
    assert "do not rewrite exported files on disk" in tooltip


def test_parallel_frames_tooltip_is_cuda_only():
    _app()
    panel = ParameterPanel()

    tooltip = panel._parallel_spin.toolTip()

    assert "CUDA only right now" in tooltip
    assert "Not currently supported on Apple Silicon" in tooltip
    assert "Apple Silicon with 64GB+ unified RAM" not in tooltip


def test_set_input_is_linear_updates_combo_without_emitting_params_changed():
    _app()
    panel = ParameterPanel()
    fired: list[bool] = []
    panel.params_changed.connect(lambda: fired.append(True))

    panel.set_input_is_linear(True)

    assert panel.get_params().input_is_linear is True
    assert fired == []

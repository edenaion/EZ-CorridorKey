import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed")

from backend.clip_state import ClipEntry, ClipState
from backend.service import InferenceParams
from ui.main_window import MainWindow


class _DummyParamPanel:
    def __init__(self, *, input_is_linear: bool):
        self._input_is_linear = input_is_linear

    def get_params(self) -> InferenceParams:
        return InferenceParams(input_is_linear=self._input_is_linear)


def _clip(name: str) -> ClipEntry:
    clip = ClipEntry(
        name=name,
        root_path=f"/tmp/{name}",
        state=ClipState.RAW,
        input_asset=object(),
    )
    return clip


def test_remembered_input_color_space_override_beats_auto_detect_default():
    window = MainWindow.__new__(MainWindow)
    clip = _clip("shot")
    clip.should_default_input_linear = lambda: False
    window._clip_input_is_linear = {}
    window._current_clip = clip
    window._param_panel = _DummyParamPanel(input_is_linear=True)

    MainWindow._remember_current_clip_input_color_space(window)

    assert MainWindow._input_is_linear_for_clip(window, clip) is True


def test_clip_color_space_defaults_are_cached_per_clip():
    window = MainWindow.__new__(MainWindow)
    clip = _clip("linear_shot")
    clip.should_default_input_linear = lambda: True
    window._clip_input_is_linear = {}

    assert MainWindow._input_is_linear_for_clip(window, clip) is True
    assert window._clip_input_is_linear == {"linear_shot": True}

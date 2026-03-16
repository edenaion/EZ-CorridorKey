import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed")

from backend.clip_state import ClipEntry, ClipState
from ui.preview.frame_index import ViewMode
from ui.widgets.preview_viewport import PreviewViewport


class _DummyAnnotationModel:
    def save(self, _root_path: str) -> None:
        return None

    def load(self, _root_path: str) -> None:
        return None


class _DummyModeBar:
    def __init__(self):
        self._current_mode = ViewMode.INPUT
        self.available = None

    def set_available_modes(self, available) -> None:
        self.available = available

    def current_mode(self):
        return self._current_mode


class _DummySplitView:
    def __init__(self):
        self.placeholder = None
        self._single_image = None
        self._left_image = None
        self._right_image = None

    def set_placeholder(self, text: str) -> None:
        self.placeholder = text


class _DummyFrameIndex:
    frame_count = 0

    def available_modes(self):
        return [ViewMode.INPUT]


def _clip(name: str, *, default_linear: bool) -> ClipEntry:
    clip = ClipEntry(
        name=name,
        root_path=f"/tmp/{name}",
        state=ClipState.RAW,
        input_asset=object(),
    )
    clip.should_default_input_linear = lambda: default_linear
    return clip


def test_set_clip_preserves_existing_input_color_space_override():
    viewport = PreviewViewport.__new__(PreviewViewport)
    viewport._clip = None
    viewport._clip_name = None
    viewport._input_exr_is_linear = True
    viewport._annotation_model = _DummyAnnotationModel()
    viewport._locked_mode = None
    viewport._mode_bar = _DummyModeBar()
    viewport._scrubber = None
    viewport._split_view = _DummySplitView()
    viewport._current_stem_idx = -1
    viewport._current_mode = ViewMode.INPUT
    viewport._update_clip_info = lambda _clip: None
    viewport._build_frame_index = lambda _clip: _DummyFrameIndex()
    viewport._navigate_to = lambda _idx: None

    clip = _clip("shot", default_linear=False)
    PreviewViewport.set_clip(viewport, clip)

    assert viewport._clip is clip
    assert viewport._input_exr_is_linear is True

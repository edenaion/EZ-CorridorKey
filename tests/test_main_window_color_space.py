import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed")

from backend.clip_state import ClipEntry, ClipState
from backend.service import InferenceParams
from ui.main_window import (
    MainWindow,
    _import_alpha_video_as_sequence,
    _remove_alpha_hint_assets,
)


class _DummyParamPanel:
    def __init__(self, *, input_is_linear: bool):
        self._input_is_linear = input_is_linear
        self.last_set_input_is_linear = input_is_linear

    def get_params(self) -> InferenceParams:
        return InferenceParams(input_is_linear=self._input_is_linear)

    def set_input_is_linear(self, input_is_linear: bool) -> None:
        self.last_set_input_is_linear = input_is_linear


class _DummyDualViewer:
    def __init__(self):
        self.clips: list[str] = []
        self.input_linear_flags: list[bool] = []

    def set_clip(self, clip: ClipEntry) -> None:
        self.clips.append(clip.name)

    def set_input_exr_is_linear(self, enabled: bool) -> None:
        self.input_linear_flags.append(enabled)


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


def test_sync_selected_clip_view_reapplies_remembered_input_interpretation():
    window = MainWindow.__new__(MainWindow)
    clip = _clip("shot")
    clip.should_default_input_linear = lambda: False
    clip.input_asset = object()
    window._clip_input_is_linear = {"shot": True}
    window._dual_viewer = _DummyDualViewer()
    window._param_panel = _DummyParamPanel(input_is_linear=False)

    refreshed_input_thumb: list[bool] = []
    refreshed_export_thumb: list[str] = []

    def _refresh_input_thumbnail(_clip: ClipEntry, *, input_is_linear: bool | None = None) -> None:
        refreshed_input_thumb.append(bool(input_is_linear))

    def _refresh_export_thumbnail(_clip: ClipEntry) -> None:
        refreshed_export_thumb.append(_clip.name)

    window._refresh_input_thumbnail = _refresh_input_thumbnail
    window._refresh_export_thumbnail = _refresh_export_thumbnail

    MainWindow._sync_selected_clip_view(window, clip)

    assert window._dual_viewer.clips == ["shot"]
    assert window._dual_viewer.input_linear_flags == [True]
    assert window._param_panel.last_set_input_is_linear is True
    assert refreshed_input_thumb == [True]
    assert refreshed_export_thumb == ["shot"]


class _DummyTray:
    def __init__(self):
        self.selected: list[str] = []

    def selected_count(self) -> int:
        return 0

    def set_selected(self, name: str) -> None:
        self.selected.append(name)


class _DummyGpuWorker:
    def isRunning(self) -> bool:
        return False


class _DummyStatusBar:
    def set_running(self, _running: bool) -> None:
        return None

    def update_button_state(self, **_kwargs) -> None:
        return None


def test_selecting_another_clip_remembers_previous_clip_color_space_override():
    window = MainWindow.__new__(MainWindow)
    previous = _clip("previous")
    previous.state = ClipState.READY
    previous.completed_frame_count = lambda: 0
    selected = _clip("selected")
    selected.state = ClipState.READY
    selected.completed_frame_count = lambda: 0

    window._clip_input_is_linear = {}
    window._current_clip = previous
    window._param_panel = _DummyParamPanel(input_is_linear=True)
    window._io_tray = _DummyTray()
    window._dual_viewer = _DummyDualViewer()
    window._gpu_worker = _DummyGpuWorker()
    window._status_bar = _DummyStatusBar()
    window._sync_selected_clip_view = lambda clip: None
    window._update_annotation_info = lambda: None
    window._clip_has_videomama_ready_mask = lambda _clip: False
    window._refresh_input_thumbnail = lambda *_args, **_kwargs: None
    window._refresh_export_thumbnail = lambda *_args, **_kwargs: None
    window._param_panel.set_gvm_enabled = lambda _enabled: None
    window._param_panel.set_birefnet_enabled = lambda _enabled: None
    window._param_panel.set_videomama_enabled = lambda _enabled: None
    window._param_panel.set_matanyone2_enabled = lambda _enabled: None
    window._param_panel.set_import_alpha_enabled = lambda _enabled: None
    window._last_clip_index = 0

    # Mock _clip_model with a clips list for index tracking
    class _DummyClipModel:
        clips = [previous, selected]

    window._clip_model = _DummyClipModel()

    MainWindow._on_clip_selected(window, selected)

    assert window._clip_input_is_linear["previous"] is True
    assert window._current_clip is selected


def test_remove_alpha_hint_assets_deletes_sequence_dir_and_video_file(tmp_path):
    alpha_dir = tmp_path / "AlphaHint"
    alpha_dir.mkdir()
    (alpha_dir / "frame_000000.png").write_text("dummy", encoding="utf-8")
    alpha_video = tmp_path / "AlphaHint.mov"
    alpha_video.write_text("dummy", encoding="utf-8")

    _remove_alpha_hint_assets(str(tmp_path))

    assert not alpha_dir.exists()
    assert not alpha_video.exists()


def test_import_alpha_video_as_sequence_writes_pngs_named_like_input_frames(tmp_path, monkeypatch):
    alpha_dir = tmp_path / "AlphaHint"
    input_files = ["frame_000000.exr", "frame_000001.exr"]
    frames = [
        np.array([[[0, 0, 255]]], dtype=np.uint8),
        np.array([[[255, 255, 255]]], dtype=np.uint8),
    ]

    class _FakeCapture:
        def __init__(self, _path):
            self._index = 0

        def read(self):
            if self._index >= len(frames):
                return False, None
            frame = frames[self._index]
            self._index += 1
            return True, frame

        def release(self):
            return None

    monkeypatch.setattr("ui.main_window.cv2.VideoCapture", _FakeCapture)

    imported = _import_alpha_video_as_sequence("alpha.mov", str(alpha_dir), input_files)

    assert imported == 2
    assert (alpha_dir / "frame_000000.png").is_file()
    assert (alpha_dir / "frame_000001.png").is_file()
    first = cv2.imread(str(alpha_dir / "frame_000000.png"), cv2.IMREAD_GRAYSCALE)
    second = cv2.imread(str(alpha_dir / "frame_000001.png"), cv2.IMREAD_GRAYSCALE)
    assert int(first[0, 0]) < 255
    assert int(second[0, 0]) == 255

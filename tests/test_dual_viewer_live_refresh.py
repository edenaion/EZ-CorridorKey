import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 not installed")
from PySide6.QtWidgets import QApplication

from backend.clip_state import ClipAsset, ClipEntry, ClipState
from ui.preview.frame_index import ViewMode
from ui.widgets.dual_viewer import DualViewerPanel


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_refresh_generated_assets_preserves_current_frame_and_updates_coverage(tmp_path):
    _app()

    clip_root = os.path.join(tmp_path, "clip")
    input_dir = os.path.join(clip_root, "Input")
    comp_dir = os.path.join(clip_root, "Output", "Comp")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)
    for i in range(3):
        frame = np.full((8, 8, 3), 80 + i, dtype=np.uint8)
        assert cv2.imwrite(os.path.join(input_dir, f"frame_{i:05d}.png"), frame)

    clip = ClipEntry(
        name="clip",
        root_path=clip_root,
        state=ClipState.RAW,
        input_asset=ClipAsset(input_dir, "sequence"),
    )

    viewer = DualViewerPanel()
    viewer.set_clip(clip)
    viewer._scrubber.set_frame(2)
    viewer._on_scrubber_frame(2)

    frame = np.full((8, 8, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(os.path.join(comp_dir, "frame_00002.png"), frame)

    viewer.refresh_generated_assets()

    assert viewer._scrubber.current_frame() == 2
    assert viewer._input_viewer.current_stem_index == 2
    assert viewer._output_viewer.current_stem_index == 2
    assert viewer._output_viewer._frame_index is not None
    assert viewer._output_viewer._frame_index.has_frame(ViewMode.COMP, 2)
    assert viewer._scrubber._coverage_bar._inference == [False, False, True]

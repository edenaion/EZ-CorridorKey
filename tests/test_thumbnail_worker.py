"""Tests for thumbnail generation using the viewer decode path."""

import pytest

pyside6 = pytest.importorskip("PySide6", reason="PySide6 not installed")
from PySide6.QtGui import QImage, QColor

from ui.preview.frame_index import ViewMode
from ui.workers import thumbnail_worker


def _solid_qimage(width: int = 120, height: int = 80, color: str = "#7fb36a") -> QImage:
    img = QImage(width, height, QImage.Format_RGB888)
    img.fill(QColor(color))
    return img


class TestThumbTask:
    def test_sequence_thumbnail_uses_input_decode_transform(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "Frames"
        input_dir.mkdir()
        frame_path = input_dir / "frame_000000.exr"
        frame_path.write_text("dummy", encoding="utf-8")

        calls: list[tuple[str, ViewMode, bool]] = []

        def fake_decode_frame(
            path: str,
            mode: ViewMode,
            *,
            input_exr_is_linear: bool = False,
        ) -> QImage:
            calls.append((path, mode, input_exr_is_linear))
            return _solid_qimage()

        monkeypatch.setattr(thumbnail_worker, "decode_frame", fake_decode_frame)

        task = thumbnail_worker._ThumbTask(
            clip_name="clip",
            clip_root=str(tmp_path),
            kind="input",
            input_path=str(input_dir),
            asset_type="sequence",
            input_exr_is_linear=True,
        )

        qimg = task._generate()

        assert qimg is not None
        assert calls == [(str(frame_path), ViewMode.INPUT, True)]
        assert qimg.width() <= thumbnail_worker.THUMB_WIDTH
        assert qimg.height() <= thumbnail_worker.THUMB_HEIGHT

    def test_video_thumbnail_uses_video_decode_transform(self, tmp_path, monkeypatch):
        video_path = tmp_path / "clip.mp4"
        video_path.write_text("dummy", encoding="utf-8")

        calls: list[tuple[str, int, bool]] = []

        def fake_decode_video_frame(
            path: str,
            frame_index: int,
            *,
            input_exr_is_linear: bool = False,
        ) -> QImage:
            calls.append((path, frame_index, input_exr_is_linear))
            return _solid_qimage()

        monkeypatch.setattr(thumbnail_worker, "decode_video_frame", fake_decode_video_frame)

        task = thumbnail_worker._ThumbTask(
            clip_name="clip",
            clip_root=str(tmp_path),
            kind="input",
            input_path=str(video_path),
            asset_type="video",
            input_exr_is_linear=True,
        )

        qimg = task._generate()

        assert qimg is not None
        assert calls == [(str(video_path), 0, True)]
        assert qimg.width() <= thumbnail_worker.THUMB_WIDTH
        assert qimg.height() <= thumbnail_worker.THUMB_HEIGHT

    def test_export_thumbnail_uses_output_decode_transform(self, tmp_path, monkeypatch):
        comp_dir = tmp_path / "Output" / "Comp"
        comp_dir.mkdir(parents=True)
        frame_path = comp_dir / "frame_000000.png"
        frame_path.write_text("dummy", encoding="utf-8")

        calls: list[tuple[str, ViewMode, bool]] = []

        def fake_decode_frame(
            path: str,
            mode: ViewMode,
            *,
            input_exr_is_linear: bool = False,
        ) -> QImage:
            calls.append((path, mode, input_exr_is_linear))
            return _solid_qimage()

        monkeypatch.setattr(thumbnail_worker, "decode_frame", fake_decode_frame)

        task = thumbnail_worker._ThumbTask(
            clip_name="clip",
            clip_root=str(tmp_path),
            kind="export",
            export_mode=ViewMode.COMP,
        )

        qimg = task._generate()

        assert qimg is not None
        assert calls == [(str(frame_path), ViewMode.COMP, False)]

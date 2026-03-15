"""Compatibility tests for service import fallbacks."""

from __future__ import annotations

import importlib


def test_service_import_survives_missing_decode_video_mask_frame(monkeypatch):
    import backend.frame_io as frame_io
    import backend.service as service

    original = getattr(frame_io, "decode_video_mask_frame", None)
    monkeypatch.delattr(frame_io, "decode_video_mask_frame", raising=False)

    reloaded = importlib.reload(service)
    assert callable(reloaded.decode_video_mask_frame)

    if original is not None:
        monkeypatch.setattr(frame_io, "decode_video_mask_frame", original, raising=False)
    importlib.reload(service)

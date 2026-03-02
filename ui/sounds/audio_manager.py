"""Minimal UI sound manager — hover sounds with amplitude variance.

Uses QSoundEffect for low-latency WAV playback.
Volume varies ±8% on each play for organic feel.
"""
from __future__ import annotations

import os
import random

from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QSoundEffect


_SOUNDS_DIR = os.path.dirname(__file__)

# Base volume (0.0–1.0) and variance (±fraction)
_BASE_VOLUME = 0.35
_VARIANCE = 0.08


def _load_sfx(filename: str) -> QSoundEffect | None:
    path = os.path.join(_SOUNDS_DIR, filename)
    if not os.path.isfile(path):
        return None
    sfx = QSoundEffect()
    sfx.setSource(QUrl.fromLocalFile(path))
    sfx.setVolume(_BASE_VOLUME)
    return sfx


class UIAudio:
    """Singleton-style UI sound player.

    UIAudio.hover()       — hover feedback (single WAV, ±8% volume variance)
    UIAudio.user_cancel() — stop/cancel (random pick from 2 WAV variants)
    UIAudio.error()       — error/failure feedback
    """

    _hover_sfx: QSoundEffect | None = None
    _click_sfx: QSoundEffect | None = None
    _cancel_sfx: list[QSoundEffect] = []
    _error_sfx: QSoundEffect | None = None
    _extract_done_sfx: QSoundEffect | None = None
    _mask_done_sfx: QSoundEffect | None = None
    _inference_done_sfx: QSoundEffect | None = None
    _loaded = False
    _muted = False

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True
        cls._hover_sfx = _load_sfx("CorridorKey_UI_Hover_v1.wav")
        cls._click_sfx = _load_sfx("CorridorKey_UI_Click_v1.wav")
        cls._error_sfx = _load_sfx("CorridorKey_UI_Error_v1.wav")
        cls._extract_done_sfx = _load_sfx("CorridorKey_UI_Frame Extract Done_v1.wav")
        cls._mask_done_sfx = _load_sfx("CorridorKey_UI_Mask Done_v2.wav")
        cls._inference_done_sfx = _load_sfx("CorridorKey_UI_Inference Done_v1.wav")
        for fname in ("CorridorKey_UI_User Cancel_v1.wav",
                       "CorridorKey_UI_User Cancel_v2.wav"):
            sfx = _load_sfx(fname)
            if sfx:
                cls._cancel_sfx.append(sfx)

    @classmethod
    def set_muted(cls, muted: bool) -> None:
        """Global mute toggle for all UI sounds."""
        cls._muted = muted

    @classmethod
    def is_muted(cls) -> bool:
        return cls._muted

    @classmethod
    def _play(cls, sfx: QSoundEffect, variance: float = _VARIANCE,
              db_offset: float = 0.0) -> None:
        if cls._muted:
            return
        vol = _BASE_VOLUME + random.uniform(-variance, variance)
        if db_offset:
            vol *= 10 ** (db_offset / 20.0)
        sfx.setVolume(max(0.0, min(1.0, vol)))
        sfx.play()

    @classmethod
    def click(cls) -> None:
        """Play click sound — for any user click action (−2dB from base)."""
        cls._ensure_loaded()
        if cls._click_sfx:
            cls._play(cls._click_sfx, variance=0.10, db_offset=-2.0)

    @classmethod
    def hover(cls, key: str = "") -> None:
        """Play hover sound with ±8% volume variance."""
        cls._ensure_loaded()
        if cls._hover_sfx:
            cls._play(cls._hover_sfx, db_offset=-1.5)

    @classmethod
    def user_cancel(cls) -> None:
        """Play cancel sound — random pick from 2 variants, ±8% volume."""
        cls._ensure_loaded()
        if cls._cancel_sfx:
            cls._play(random.choice(cls._cancel_sfx))

    @classmethod
    def error(cls) -> None:
        """Play error sound — for failures and critical issues."""
        cls._ensure_loaded()
        if cls._error_sfx:
            cls._play(cls._error_sfx)

    @classmethod
    def frame_extract_done(cls) -> None:
        """Play frame extraction complete sound — ±10% volume variance."""
        cls._ensure_loaded()
        if cls._extract_done_sfx:
            cls._play(cls._extract_done_sfx, variance=0.10)

    @classmethod
    def mask_done(cls) -> None:
        """Play mask/alpha generation complete sound."""
        cls._ensure_loaded()
        if cls._mask_done_sfx:
            cls._play(cls._mask_done_sfx)

    @classmethod
    def inference_done(cls) -> None:
        """Play inference complete sound."""
        cls._ensure_loaded()
        if cls._inference_done_sfx:
            cls._play(cls._inference_done_sfx)

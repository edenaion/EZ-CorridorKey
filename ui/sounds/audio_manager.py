"""Minimal UI sound manager — click sounds with amplitude variance.

Uses QSoundEffect for low-latency WAV playback.
Volume varies ±8% on each play for organic feel.

Unified click sound system: install ButtonClickFilter on QApplication
and every QPushButton automatically gets a click sound on press.
"""

from __future__ import annotations

import array
import os
import random
import tempfile
import time
import wave

from PySide6.QtCore import QUrl, QObject, QEvent
from PySide6.QtMultimedia import QSoundEffect


_SOUNDS_DIR = os.path.dirname(__file__)

# Base volume (0.0–1.0) and variance (±fraction)
_BASE_VOLUME = 0.35
_VARIANCE = 0.08

# Fade duration applied to all UI sounds to prevent clicks/pops
_FADE_MS = 50

# Keep refs to temp files so they aren't garbage-collected while QSoundEffect uses them
_temp_files: list[tempfile.NamedTemporaryFile] = []


def _apply_fade(path: str) -> str:
    """Apply a short linear fade-in/out to a WAV file, return path to faded copy.

    Writes a temp .wav file with the faded audio.  The temp file is kept alive
    for the process lifetime via ``_temp_files``.
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    total_samples = n_frames * n_channels
    fade_samples = int(framerate * _FADE_MS / 1000) * n_channels

    if sampwidth == 2:
        # 16-bit signed
        samples = array.array("h")
        samples.frombytes(raw)
        for i in range(min(fade_samples, total_samples)):
            samples[i] = int(samples[i] * (i / fade_samples))
        for i in range(min(fade_samples, total_samples)):
            idx = total_samples - 1 - i
            samples[idx] = int(samples[idx] * (i / fade_samples))
        out_raw = samples.tobytes()
    elif sampwidth == 3:
        # 24-bit signed (packed as 3 bytes per sample, little-endian)
        buf = bytearray(raw)
        for i in range(min(fade_samples, total_samples)):
            off = i * 3
            val = int.from_bytes(buf[off:off + 3], "little", signed=True)
            val = int(val * (i / fade_samples))
            buf[off:off + 3] = val.to_bytes(3, "little", signed=True)
        for i in range(min(fade_samples, total_samples)):
            idx = total_samples - 1 - i
            off = idx * 3
            val = int.from_bytes(buf[off:off + 3], "little", signed=True)
            val = int(val * (i / fade_samples))
            buf[off:off + 3] = val.to_bytes(3, "little", signed=True)
        out_raw = bytes(buf)
    elif sampwidth == 1:
        # 8-bit unsigned (WAV convention)
        samples = array.array("B")
        samples.frombytes(raw)
        for i in range(min(fade_samples, total_samples)):
            samples[i] = int(128 + (samples[i] - 128) * (i / fade_samples))
        for i in range(min(fade_samples, total_samples)):
            idx = total_samples - 1 - i
            samples[idx] = int(128 + (samples[idx] - 128) * (i / fade_samples))
        out_raw = samples.tobytes()
    else:
        return path

    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    _temp_files.append(tf)
    with wave.open(tf.name, "wb") as out:
        out.setnchannels(n_channels)
        out.setsampwidth(sampwidth)
        out.setframerate(framerate)
        out.writeframes(out_raw)
    tf.close()
    return tf.name


def _load_sfx(filename: str) -> QSoundEffect | None:
    path = os.path.join(_SOUNDS_DIR, filename)
    if not os.path.isfile(path):
        return None
    faded_path = _apply_fade(path)
    sfx = QSoundEffect()
    sfx.setSource(QUrl.fromLocalFile(faded_path))
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
    _toggle_sfx: QSoundEffect | None = None
    _cancel_sfx: list[QSoundEffect] = []
    _error_sfx: QSoundEffect | None = None
    _extract_done_sfx: QSoundEffect | None = None
    _mask_done_sfx: QSoundEffect | None = None
    _inference_done_sfx: QSoundEffect | None = None
    _loaded = False
    _muted = False
    _volume: float = 1.0  # Master volume multiplier (0.0–1.0)
    _last_play_time: float = 0.0
    _DEBOUNCE_MS = 0.20  # 200ms debounce — prevents double-fire on dialog close

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True
        cls._hover_sfx = _load_sfx("CorridorKey_UI_Hover_v1.wav")
        cls._click_sfx = _load_sfx("CorridorKey_UI_Click_v1.wav")
        cls._toggle_sfx = _load_sfx("CorridorKey_UI_Click_v2.wav")
        cls._error_sfx = _load_sfx("CorridorKey_UI_Error_v1.wav")
        cls._extract_done_sfx = _load_sfx("CorridorKey_UI_Frame Extract Done_v1.wav")
        cls._mask_done_sfx = _load_sfx("CorridorKey_UI_Mask Done_v2.wav")
        cls._inference_done_sfx = _load_sfx("CorridorKey_UI_Inference Done_v1.wav")
        for fname in (
            "CorridorKey_UI_User Cancel_v1.wav",
            "CorridorKey_UI_User Cancel_v2.wav",
        ):
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
    def set_volume(cls, volume: float) -> None:
        """Set master volume (0.0–1.0). Persists to QSettings."""
        cls._volume = max(0.0, min(1.0, volume))
        from PySide6.QtCore import QSettings

        QSettings().setValue("ui/sounds_volume", cls._volume)

    @classmethod
    def get_volume(cls) -> float:
        return cls._volume

    @classmethod
    def _play(
        cls, sfx: QSoundEffect, variance: float = _VARIANCE, db_offset: float = 0.0,
        skip_debounce: bool = False,
    ) -> None:
        if cls._muted or cls._volume <= 0.0:
            return
        now = time.monotonic()
        if not skip_debounce and now - cls._last_play_time < cls._DEBOUNCE_MS:
            return
        cls._last_play_time = now
        vol = _BASE_VOLUME + random.uniform(-variance, variance)
        if db_offset:
            vol *= 10 ** (db_offset / 20.0)
        vol *= cls._volume
        sfx.setVolume(max(0.0, min(1.0, vol)))
        sfx.play()

    @classmethod
    def click(cls) -> None:
        """Play click sound — for any user click action (−2dB from base)."""
        cls._ensure_loaded()
        if cls._click_sfx:
            cls._play(cls._click_sfx, variance=0.10, db_offset=-2.0)

    @classmethod
    def toggle(cls) -> None:
        """Play toggle sound — for checkbox state changes (Click_v2)."""
        cls._ensure_loaded()
        if cls._toggle_sfx:
            cls._play(cls._toggle_sfx, variance=0.10, db_offset=-2.0)

    @classmethod
    def hover(cls) -> None:
        """Hover sound — only used on the welcome/home screen."""
        cls._ensure_loaded()
        if cls._hover_sfx:
            cls._play(cls._hover_sfx)

    @classmethod
    def user_cancel(cls) -> None:
        """Play cancel sound — random pick from 2 variants, ±8% volume."""
        cls._ensure_loaded()
        if cls._cancel_sfx:
            cls._play(random.choice(cls._cancel_sfx), skip_debounce=True)

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


class ButtonClickFilter(QObject):
    """App-level event filter — plays sounds on QPushButton press and QCheckBox toggle.

    Install once on QApplication and every button/checkbox gets sound automatically.
    No per-widget wiring needed.
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.MouseButtonPress:
            from PySide6.QtWidgets import QCheckBox, QPushButton

            if isinstance(obj, QPushButton) and obj.isEnabled():
                UIAudio.click()
            elif isinstance(obj, QCheckBox) and obj.isEnabled():
                UIAudio.toggle()
        return False


def install_global_click_sound(app: QObject) -> None:
    """Install the unified click sound filter on the application.

    Call once during app startup:
        install_global_click_sound(app)
    """
    filt = ButtonClickFilter(app)
    app.installEventFilter(filt)

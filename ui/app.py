"""QApplication setup with EZSCAPE brand theme and bundled fonts.

Fonts:
  - Gagarin: Logo / brand mark text (identity font)
  - Open Sans: All secondary / body UI text (default app font)
"""
from __future__ import annotations

import sys
import os
import ctypes
import logging

from PySide6.QtWidgets import QApplication, QMessageBox, QDialogButtonBox
from PySide6.QtGui import QFontDatabase, QFont, QIcon
from PySide6.QtCore import Qt, QObject, QEvent, QTranslator, QLocale

from ui.theme import load_stylesheet

logger = logging.getLogger(__name__)


class _MessageBoxFilter(QObject):
    """Auto-center buttons on every QMessageBox in the app."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Show and isinstance(obj, QMessageBox):
            for bb in obj.findChildren(QDialogButtonBox):
                bb.setCenterButtons(True)
        return False


def _configure_runtime_backends() -> None:
    """Apply lightweight runtime tuning before QApplication is created.

    The UI is QWidget-based, so Qt Quick shouldn't be involved, but leaving the
    software backend enabled avoids accidental GPU work if a Qt Quick control is
    introduced later. On Windows, cap OpenCV's internal thread fan-out so
    background preview/thumbnail decode work does not monopolize the desktop.
    """
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ["QT_QUICK_BACKEND"] = "software"

    # ── CUDA / cuDNN global settings ──
    # These must be set once before any inference.  Previously they lived inside
    # CorridorKeyEngine._load_model() which only runs lazily on first inference,
    # so the first batch of frames ran without TF32 tensor-core acceleration.
    try:
        import torch
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = False
        logger.info("CUDA globals: TF32 precision='high', cuDNN benchmark=off")
    except ImportError:
        pass  # torch not installed – CPU-only environment

    if sys.platform != "win32":
        return

    # Disable Windows Efficiency Mode (EcoQoS) for this process.
    # Without this, Windows 11 aggressively throttles background processes,
    # starving our GPU worker thread of CPU time and stalling CUDA inference.
    try:
        class _POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [
                ("Version", ctypes.c_ulong),
                ("ControlMask", ctypes.c_ulong),
                ("StateMask", ctypes.c_ulong),
            ]

        state = _POWER_THROTTLING_STATE()
        state.Version = 1  # PROCESS_POWER_THROTTLING_CURRENT_VERSION
        state.ControlMask = 0x1  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        state.StateMask = 0  # 0 = disable throttling

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.GetCurrentProcess()
        ok = kernel32.SetProcessInformation(
            handle,
            4,  # ProcessPowerThrottling
            ctypes.byref(state),
            ctypes.sizeof(state),
        )
        if ok:
            logger.info("Disabled Windows power throttling (EcoQoS)")
        else:
            logger.debug("SetProcessInformation failed: %s",
                         ctypes.get_last_error())
    except Exception as exc:
        logger.debug("Power throttling opt-out skipped: %s", exc)

    # Raise process priority so Windows doesn't starve our GPU worker thread
    # when the window loses foreground focus.  ABOVE_NORMAL (0x8000) is a mild
    # bump — enough to keep CPU-side frame I/O feeding the GPU, without
    # impacting system responsiveness the way HIGH/REALTIME would.
    # The EcoQoS opt-out above only works on Win11; this covers Win10.
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
        if kernel32.SetPriorityClass(kernel32.GetCurrentProcess(),
                                     ABOVE_NORMAL_PRIORITY_CLASS):
            logger.info("Process priority set to ABOVE_NORMAL")
        else:
            logger.debug("SetPriorityClass failed: %s", ctypes.get_last_error())
    except Exception as exc:
        logger.debug("Priority class adjustment skipped: %s", exc)

    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception as exc:
        logger.debug(f"OpenCV thread tuning skipped: {exc}")


def _migrate_legacy_settings() -> None:
    """One-time migration from old QSettings path to new one.

    Old: HKCU\\Software\\Corridor Digital\\CorridorKey
    New: HKCU\\Software\\EZSCAPE\\EZ-CorridorKey
    """
    from PySide6.QtCore import QSettings
    new = QSettings()  # Uses current org/app (EZSCAPE / EZ-CorridorKey)
    if new.value("_migrated_from_legacy", False, type=bool):
        return
    old = QSettings("Corridor Digital", "CorridorKey")
    old_keys = old.allKeys()
    if not old_keys:
        return
    for key in old_keys:
        if not new.contains(key):
            new.setValue(key, old.value(key))
    new.setValue("_migrated_from_legacy", True)
    logger.info("Migrated %d settings from Corridor Digital/CorridorKey", len(old_keys))


def apply_language(app: QApplication, lang: str | None = None) -> bool:
    """Install the .qm translator for `lang`, replacing any active one.

    `lang=None` resolves from the saved preference, falling back to the
    system locale. Returns True when the requested language is active
    (English counts: it is the source language and needs no catalogue).
    Safe to call at any point after QApplication creation, so Preferences
    can switch languages at runtime without an app restart.
    """
    from PySide6.QtCore import QSettings

    settings = QSettings()
    auto_resolved = False
    if lang is None:
        lang = settings.value("ui/language", "", type=str)
        if not lang:
            # First run, no saved preference: detect the system language and
            # persist whatever ends up applied, so the Preferences dropdown
            # reflects the active language instead of defaulting to English
            # (issue #168). Only the auto path persists; explicit calls from
            # Preferences store their own value.
            lang = QLocale.system().name().split("_")[0]  # e.g. "fr" from "fr_FR"
            auto_resolved = True

    def _persist(effective: str) -> None:
        if auto_resolved:
            settings.setValue("ui/language", effective)

    # Drop the active translator (no-op on startup)
    old = getattr(app, "_corridorkey_translator", None)
    if old is not None:
        app.removeTranslator(old)
        app._corridorkey_translator = None

    if lang == "en":
        _persist("en")
        return True  # English is the source language, no translation needed

    if getattr(sys, "frozen", False):
        translations_dir = os.path.join(sys._MEIPASS, "ui", "translations")
    else:
        translations_dir = os.path.join(os.path.dirname(__file__), "translations")

    qm_file = os.path.join(translations_dir, f"corridorkey_{lang}.qm")
    if not os.path.isfile(qm_file):
        logger.debug("No translation file for '%s' at %s", lang, qm_file)
        _persist("en")  # unsupported system language; UI stays English
        return False

    translator = QTranslator(app)
    if translator.load(qm_file):
        app.installTranslator(translator)
        # Store ref on app to prevent garbage collection
        app._corridorkey_translator = translator
        logger.info("Loaded translation: %s", lang)
        _persist(lang)
        return True
    logger.warning("Failed to load translation file: %s", qm_file)
    _persist("en")  # load failed; UI is English
    return False


def _prewarm_mlx() -> None:
    """Import mlx.core on the main thread so Metal initializes safely.

    The PyInstaller runtime hook (pyi_rth_mlx.py) does this too, but its
    failure is silent.  This second attempt runs after paths are fully set
    up and logs any failure so we can diagnose crash reports where Metal
    abort()s because mlx.core first loaded on a worker thread.
    """
    if sys.platform != "darwin":
        return
    if "mlx.core" in sys.modules:
        return  # Already initialized (runtime hook succeeded)
    try:
        import mlx.core as mx
        mx.eval(mx.zeros(1))
        logger.info("MLX Metal prewarm succeeded on main thread")
    except Exception as exc:
        logger.warning("MLX prewarm failed (will fall back to MPS): %s", exc)


def create_app(argv: list[str] | None = None) -> QApplication:
    """Create and configure the QApplication with brand theming.

    Returns a QApplication instance ready for main window creation.
    """
    if argv is None:
        argv = sys.argv

    _configure_runtime_backends()
    _prewarm_mlx()

    app = QApplication(argv)
    app.setApplicationName("EZ-CorridorKey")
    app.setApplicationDisplayName("EZ-CorridorKey")
    app.setOrganizationName("EZSCAPE")

    # ── i18n: load translation for user's language ──
    apply_language(app)

    # One-time migration from old registry path
    # (Corridor Digital\CorridorKey → EZSCAPE\EZ-CorridorKey)
    _migrate_legacy_settings()

    # Keep the Windows Apps & Features version in sync with the bundled build
    # after a skinny update. No-op on non-Windows and in dev mode.
    try:
        from backend.version_sync import sync_uninstall_version
        sync_uninstall_version()
    except Exception as exc:
        logger.debug("Version sync skipped: %s", exc)

    # ── Font loading (frozen-build aware) ──
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
        fonts_dir = os.path.join(base, "ui", "theme", "fonts")
    else:
        base = os.path.dirname(__file__)
        fonts_dir = os.path.join(base, "theme", "fonts")

    # Register all bundled fonts (Gagarin + Open Sans)
    opensans_loaded = False
    gagarin_loaded = False
    if os.path.isdir(fonts_dir):
        for fname in os.listdir(fonts_dir):
            if not fname.lower().endswith((".ttf", ".otf")):
                continue
            font_id = QFontDatabase.addApplicationFont(os.path.join(fonts_dir, fname))
            if font_id >= 0:
                families = QFontDatabase.applicationFontFamilies(font_id)
                if "Open Sans" in families:
                    opensans_loaded = True
                if "Gagarin" in families:
                    gagarin_loaded = True

    # Fallback: search system font dirs for Open Sans
    if not opensans_loaded:
        system_font_dirs = [
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/Library/Fonts"),
            "/System/Library/Fonts",
            "C:/Windows/Fonts",
        ]
        for font_dir in system_font_dirs:
            if not os.path.isdir(font_dir):
                continue
            for fname in os.listdir(font_dir):
                if "opensans" in fname.lower() and fname.lower().endswith((".ttf", ".otf")):
                    font_id = QFontDatabase.addApplicationFont(os.path.join(font_dir, fname))
                    if font_id >= 0:
                        opensans_loaded = True

    # Set default app font (Open Sans for all body text)
    if opensans_loaded:
        app.setFont(QFont("Open Sans", 13))
    else:
        logger.info("Open Sans not found, using system sans-serif")
        fallback = "Segoe UI" if sys.platform == "win32" else "Helvetica"
        app.setFont(QFont(fallback, 13))

    if gagarin_loaded:
        logger.info("Gagarin font loaded for brand mark")
    else:
        logger.warning("Gagarin font not found — brand mark will use fallback")

    # Apply brand stylesheet
    app.setStyleSheet(load_stylesheet())

    # Set app icon (window title bar + taskbar) — ICO preferred for window chrome
    # (corridorkey.svg is the brand logo used on the welcome screen, not the app icon)
    theme_dir = os.path.join(base, "ui", "theme") if getattr(sys, 'frozen', False) else os.path.join(base, "theme")
    ico_icon = os.path.join(theme_dir, "corridorkey.ico")
    png_icon = os.path.join(theme_dir, "corridorkey.png")
    if os.path.isfile(ico_icon):
        app.setWindowIcon(QIcon(ico_icon))
    elif os.path.isfile(png_icon):
        app.setWindowIcon(QIcon(png_icon))

    # Install unified click sound — every QPushButton gets click sound automatically
    from ui.sounds.audio_manager import install_global_click_sound
    install_global_click_sound(app)

    # Center buttons on all QMessageBox dialogs globally
    app._msgbox_filter = _MessageBoxFilter(app)
    app.installEventFilter(app._msgbox_filter)

    return app

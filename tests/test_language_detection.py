"""Tests for first-run language detection persisting to preferences (#168).

On first launch with no saved preference, apply_language detects the system
language and applies it. It must also persist the effective language so the
Preferences dropdown reflects what is actually active, instead of defaulting
to English while the UI is in another language.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed")

from PySide6.QtCore import QCoreApplication, QLocale, QSettings
from PySide6.QtWidgets import QApplication

from ui.app import apply_language


class _FakeLocale:
    """Stand-in for QLocale.system() with a controlled name()."""

    def __init__(self, name: str):
        self._name = name

    def name(self) -> str:
        return self._name


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    # Remove any translator these tests installed. The QApplication is a
    # process-wide singleton, so a leaked non-English translator would make
    # later tests that assert on English strings (tooltips, labels) fail.
    tr = getattr(app, "_corridorkey_translator", None)
    if tr is not None:
        app.removeTranslator(tr)
        app._corridorkey_translator = None


@pytest.fixture
def isolated_settings():
    """Route QSettings to a throwaway org/app store, restored afterwards."""
    prev_org = QCoreApplication.organizationName()
    prev_app = QCoreApplication.applicationName()
    QCoreApplication.setOrganizationName("EZSCAPE-test")
    QCoreApplication.setApplicationName("CK-langtest")
    s = QSettings()
    s.clear()
    s.sync()
    yield s
    s.clear()
    s.sync()
    QCoreApplication.setOrganizationName(prev_org)
    QCoreApplication.setApplicationName(prev_app)


def _fake_system(monkeypatch, name: str):
    monkeypatch.setattr(QLocale, "system", staticmethod(lambda: _FakeLocale(name)))


def test_first_run_persists_detected_system_language(qapp, isolated_settings, monkeypatch):
    _fake_system(monkeypatch, "ru_RU")
    assert isolated_settings.value("ui/language", "", type=str) == ""

    assert apply_language(qapp) is True

    assert isolated_settings.value("ui/language", "", type=str) == "ru"


def test_first_run_unsupported_language_persists_english(qapp, isolated_settings, monkeypatch):
    _fake_system(monkeypatch, "zz_ZZ")  # no catalogue ships for this

    apply_language(qapp)

    assert isolated_settings.value("ui/language", "", type=str) == "en"


def test_first_run_english_system_persists_english(qapp, isolated_settings, monkeypatch):
    _fake_system(monkeypatch, "en_US")

    assert apply_language(qapp) is True

    assert isolated_settings.value("ui/language", "", type=str) == "en"


def test_saved_preference_is_not_overwritten_by_system(qapp, isolated_settings, monkeypatch):
    isolated_settings.setValue("ui/language", "en")
    isolated_settings.sync()
    _fake_system(monkeypatch, "ru_RU")

    apply_language(qapp)

    # Saved English preference must win over a Russian system locale.
    assert isolated_settings.value("ui/language", "", type=str) == "en"


def test_explicit_language_argument_does_not_persist(qapp, isolated_settings, monkeypatch):
    _fake_system(monkeypatch, "ru_RU")

    # Preferences passes an explicit code; apply_language must not write it
    # (the Preferences dialog stores its own value).
    apply_language(qapp, "fr")

    assert isolated_settings.value("ui/language", "", type=str) == ""

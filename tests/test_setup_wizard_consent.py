"""First-run crash reporting ask in the setup wizard."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def isolated_settings(qapp):
    QCoreApplication.setOrganizationName("EZSCAPE-Test")
    QCoreApplication.setApplicationName("EZ-CorridorKey-Test")
    s = QSettings()
    s.clear()
    s.sync()
    yield
    s = QSettings()
    s.clear()
    s.sync()
    QCoreApplication.setOrganizationName("EZSCAPE")
    QCoreApplication.setApplicationName("EZ-CorridorKey")


def _make_wizard():
    from ui.widgets.setup_wizard import SetupWizard
    return SetupWizard()


class TestFirstRunConsent:
    def test_shown_when_never_prompted(self):
        w = _make_wizard()
        assert w._crash_check is not None
        assert w._crash_check.isChecked() is False, "must default off"

    def test_persist_yes(self):
        w = _make_wizard()
        w._crash_check.setChecked(True)
        w._persist_crash_consent()
        s = QSettings()
        assert s.value("privacy/crash_reports_enabled", False, type=bool) is True
        assert s.value("privacy/crash_reports_prompted", False, type=bool) is True

    def test_persist_no(self):
        w = _make_wizard()
        w._persist_crash_consent()
        s = QSettings()
        assert s.value("privacy/crash_reports_enabled", True, type=bool) is False
        assert s.value("privacy/crash_reports_prompted", False, type=bool) is True

    def test_hidden_once_answered(self):
        w = _make_wizard()
        w._persist_crash_consent()
        w2 = _make_wizard()
        assert w2._crash_check is None

    def test_persist_noop_when_not_shown(self):
        w = _make_wizard()
        w._persist_crash_consent()
        w2 = _make_wizard()
        # Answered already: persisting again must not touch the choice
        s = QSettings()
        s.setValue("privacy/crash_reports_enabled", True)
        w2._persist_crash_consent()
        assert s.value("privacy/crash_reports_enabled", False, type=bool) is True

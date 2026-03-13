import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from ui.widgets.status_bar import StatusBar


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_warning_tooltip_wraps_full_message_without_truncation():
    _app()
    status_bar = StatusBar()
    message = (
        "Some saved annotations fall outside the image bounds and could not be reused on "
        "frame(s) 80. Redo the annotations on those frames if tracking looks wrong."
    )

    status_bar.add_warning(message)
    tooltip = status_bar._warn_btn.toolTip()

    assert "..." not in tooltip
    assert "Click for all warnings" in tooltip
    assert message in tooltip.replace("\n", " ")


def test_warnings_dialog_is_parented_to_main_window():
    _app()
    window = QMainWindow()
    container = QWidget()
    layout = QVBoxLayout(container)
    status_bar = StatusBar()
    layout.addWidget(status_bar)
    window.setCentralWidget(container)
    status_bar.add_warning("Redo the annotations on frame 80.")

    dlg = status_bar._build_warnings_dialog()
    try:
        assert dlg.parentWidget() is window
        assert dlg.windowModality() == Qt.WindowModal
    finally:
        dlg.close()

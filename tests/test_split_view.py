import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from ui.widgets.split_view import SplitViewWidget


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_display_to_image_rejects_points_outside_visible_left_view():
    _app()
    widget = SplitViewWidget()
    widget.resize(800, 600)

    image = QImage(400, 200, QImage.Format_RGB32)
    widget.set_left_image(image)
    widget.set_right_image(image)
    widget.set_split_enabled(True)

    full_rect = widget._image_rect(image)
    paintable_rect = widget._annotation_paint_rect(image)

    assert paintable_rect is not None
    assert paintable_rect.width() < full_rect.width()

    inside = QPointF(
        paintable_rect.left() + 10.0,
        paintable_rect.center().y(),
    )
    outside_right = QPointF(
        min(full_rect.right() - 1.0, paintable_rect.right() + 10.0),
        paintable_rect.center().y(),
    )
    outside_image = QPointF(full_rect.left() - 5.0, full_rect.top() - 5.0)

    assert widget._display_to_image(inside) is not None
    assert widget._display_to_image(outside_right) is None
    assert widget._display_to_image(outside_image) is None

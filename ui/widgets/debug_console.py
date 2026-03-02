"""In-app debug console — live log viewer.

Shows real-time log output inside the app. Toggle with F12 or Help > Console.
Captures all Python logging output via a custom handler attached to the root logger.
"""
from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from PySide6.QtCore import QObject, QSettings, Qt, Signal, Slot
from PySide6.QtGui import QCursor, QFont, QKeySequence, QMouseEvent, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_EASTERN = ZoneInfo("America/New_York")
_MAX_LINES = 5000

# Color map per log level
_LEVEL_COLORS = {
    "DEBUG": "#808070",
    "INFO": "#E0E0E0",
    "WARNING": "#FFF203",
    "ERROR": "#D10000",
    "CRITICAL": "#D10000",
}


# ---------------------------------------------------------------------------
# Qt-safe logging handler
# ---------------------------------------------------------------------------

class _QtLogHandler(logging.Handler, QObject):
    """Logging handler that emits a Qt signal for each log record.

    The signal carries pre-formatted HTML so the widget just appends it.
    Thread-safe: Python logging serializes emit() calls, and the signal
    is queued across threads by Qt automatically.
    """

    log_received = Signal(str, int)  # (html_line, levelno)

    def __init__(self) -> None:
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setLevel(logging.DEBUG)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            dt = datetime.fromtimestamp(record.created, tz=_EASTERN)
            ts = dt.strftime("%H:%M:%S")
            level = record.levelname
            color = _LEVEL_COLORS.get(level, "#E0E0E0")
            name = record.name
            msg = record.getMessage()
            # Escape HTML entities in the message
            msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            html = (
                f'<span style="color:#FFF203">{ts}</span> '
                f'<span style="color:{color}">[{level:<7s}]</span> '
                f'<span style="color:#808070">{name}:</span> '
                f'<span style="color:#E0E0E0">{msg}</span>'
            )
            self.log_received.emit(html, record.levelno)
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Debug Console Widget
# ---------------------------------------------------------------------------

class DebugConsoleWidget(QWidget):
    """Floating debug console window with live log output."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window | Qt.FramelessWindowHint)
        self.setWindowTitle("Console")
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._min_level = logging.DEBUG
        self._paused = False
        self._drag_pos = None
        self._resize_edge = None
        self._log_buffer: list[tuple[str, int]] = []  # (html, levelno)

        # Handler (installed on root logger when shown)
        self._handler = _QtLogHandler()
        self._handler.log_received.connect(self._on_log)
        self._handler_installed = False

        self._build_ui()
        self._restore_geometry()

        # Local shortcut so Escape closes console when it has focus
        # (F12 toggle is handled globally via ShortcutRegistry with ApplicationShortcut)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.hide)

    # --- UI Construction ------------------------------------------------

    def _build_ui(self) -> None:
        self.setMinimumSize(400, 200)
        self.setStyleSheet(
            "DebugConsoleWidget { background: #0E0D00; border: 1px solid #2A2910; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        # Title bar
        title_bar = QWidget()
        title_bar.setFixedHeight(28)
        title_bar.setStyleSheet("background: #0E0D00; border-bottom: 1px solid #2A2910;")
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(10, 0, 6, 0)
        tb_layout.setSpacing(8)

        title_label = QLabel("CONSOLE")
        title_label.setStyleSheet(
            "color: #FFF203; font-size: 11px; font-weight: 700;"
            "letter-spacing: 3px; border: none;"
        )
        tb_layout.addWidget(title_label)
        tb_layout.addStretch()

        close_btn = QPushButton("\u2715")  # ✕
        close_btn.setFixedSize(28, 28)
        close_btn.setStyleSheet(
            "QPushButton { color: #E0E0E0; background: transparent;"
            "font-size: 14px; border: none; padding: 0; }"
            "QPushButton:hover { color: #FFF203; }"
        )
        close_btn.clicked.connect(self.hide)
        tb_layout.addWidget(close_btn)

        layout.addWidget(title_bar)
        self._title_bar = title_bar

        # Toolbar
        toolbar = QWidget()
        toolbar.setFixedHeight(30)
        toolbar.setStyleSheet("background: #0E0D00; border-bottom: 1px solid #2A2910;")
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(8, 2, 8, 2)
        tl.setSpacing(6)

        # Level filter
        self._level_combo = QComboBox()
        self._level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self._level_combo.setCurrentText("DEBUG")
        self._level_combo.setFixedWidth(100)
        self._level_combo.setStyleSheet(
            "QComboBox { background: #1A1900; color: #E0E0E0; border: 1px solid #2A2910;"
            "padding: 2px 6px; font-size: 11px; }"
            "QComboBox:hover { border-color: #FFF203; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #1A1900; color: #E0E0E0;"
            "selection-background-color: #2A2910; border: 1px solid #2A2910; }"
        )
        self._level_combo.currentTextChanged.connect(self._on_level_changed)
        tl.addWidget(QLabel("Level:"))
        tl.addWidget(self._level_combo)

        tl.addStretch()

        # Pause button
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setFixedWidth(60)
        self._pause_btn.setStyleSheet(self._toolbar_btn_style())
        self._pause_btn.clicked.connect(self._toggle_pause)
        tl.addWidget(self._pause_btn)

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.setStyleSheet(self._toolbar_btn_style())
        clear_btn.clicked.connect(self._clear)
        tl.addWidget(clear_btn)

        # Style the level label
        for child in toolbar.findChildren(QLabel):
            child.setStyleSheet("color: #808070; font-size: 11px; border: none;")

        layout.addWidget(toolbar)

        # Log output area
        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(QFont("Consolas", 12))
        self._output.setStyleSheet(
            "QTextEdit { background: #000000; color: #E0E0E0;"
            "border: none; padding: 4px 8px; }"
            "QScrollBar:vertical { background: #0A0A00; width: 10px; }"
            "QScrollBar::handle:vertical { background: #454430; min-height: 30px; }"
            "QScrollBar::handle:vertical:hover { background: #5A5940; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )
        layout.addWidget(self._output, 1)

        # Resize grip visual (bottom-right corner)
        grip = QLabel("\u25e2")  # ◢
        grip.setFixedSize(14, 14)
        grip.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        grip.setStyleSheet("color: #454430; font-size: 12px; border: none; background: transparent;")
        layout.addWidget(grip, 0, Qt.AlignRight)

    @staticmethod
    def _toolbar_btn_style() -> str:
        return (
            "QPushButton { background: #1A1900; color: #CCCCAA; border: 1px solid #2A2910;"
            "font-size: 11px; padding: 2px 8px; }"
            "QPushButton:hover { border-color: #FFF203; color: #FFF203; }"
            "QPushButton:pressed { background: #0E0D00; }"
        )

    # --- Log Handler Management -----------------------------------------

    def show(self) -> None:
        if not self._handler_installed:
            logging.getLogger().addHandler(self._handler)
            self._handler_installed = True
        super().show()
        self.raise_()
        self.activateWindow()

    def hide(self) -> None:
        self._save_geometry()
        super().hide()

    def close_permanently(self) -> None:
        """Remove the log handler entirely (call on app shutdown)."""
        if self._handler_installed:
            logging.getLogger().removeHandler(self._handler)
            self._handler_installed = False
        self._save_geometry()

    # --- Log Display ----------------------------------------------------

    @Slot(str, int)
    def _on_log(self, html: str, levelno: int) -> None:
        if self._paused:
            return

        # Always buffer (trim oldest if over limit)
        self._log_buffer.append((html, levelno))
        if len(self._log_buffer) > _MAX_LINES:
            self._log_buffer = self._log_buffer[-(_MAX_LINES - 500):]

        # Only display if it passes the current filter
        if levelno >= self._min_level:
            self._output.append(html)

    def _on_level_changed(self, text: str) -> None:
        self._min_level = getattr(logging, text, logging.DEBUG)
        self._refilter()

    def _refilter(self) -> None:
        """Re-render the log output from the buffer using the current level filter."""
        self._output.clear()
        for html, levelno in self._log_buffer:
            if levelno >= self._min_level:
                self._output.append(html)

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self._pause_btn.setText("Resume" if self._paused else "Pause")

    def _clear(self) -> None:
        self._output.clear()
        self._log_buffer.clear()

    # --- Geometry Persistence -------------------------------------------

    def _save_geometry(self) -> None:
        s = QSettings()
        s.setValue("debug_console/geometry", self.saveGeometry())

    def _restore_geometry(self) -> None:
        s = QSettings()
        geo = s.value("debug_console/geometry")
        if geo:
            self.restoreGeometry(geo)
        else:
            self.resize(700, 400)

    # --- Dragging & Resizing --------------------------------------------

    _EDGE_MARGIN = 8

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            pos = event.position().toPoint()
            # Check if near bottom-right corner for resize
            if (self.width() - pos.x() < self._EDGE_MARGIN
                    and self.height() - pos.y() < self._EDGE_MARGIN):
                self._resize_edge = True
                self._drag_pos = event.globalPosition().toPoint()
            elif pos.y() < 28:  # Title bar drag
                self._drag_pos = event.globalPosition().toPoint() - self.pos()
                self._resize_edge = False
            else:
                self._drag_pos = None
                self._resize_edge = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos is None:
            # Update cursor for resize hint
            pos = event.position().toPoint()
            if (self.width() - pos.x() < self._EDGE_MARGIN
                    and self.height() - pos.y() < self._EDGE_MARGIN):
                self.setCursor(Qt.SizeFDiagCursor)
            elif pos.y() < 28:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return

        if self._resize_edge:
            delta = event.globalPosition().toPoint() - self._drag_pos
            self._drag_pos = event.globalPosition().toPoint()
            new_w = max(self.minimumWidth(), self.width() + delta.x())
            new_h = max(self.minimumHeight(), self.height() + delta.y())
            self.resize(new_w, new_h)
        else:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_pos = None
        self._resize_edge = None
        self.setCursor(Qt.ArrowCursor)

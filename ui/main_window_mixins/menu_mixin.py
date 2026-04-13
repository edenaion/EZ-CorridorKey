from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget
from PySide6.QtCore import Qt


class MenuMixin:
    """Menu bar construction methods for MainWindow."""

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")
        import_menu = file_menu.addMenu("Import Clips")
        import_menu.addAction("Import Folder...", self._on_import_folder)
        import_menu.addAction("Import Video(s)...", self._on_import_videos)
        import_menu.addAction("Import Image Sequence...", self._on_import_image_sequence)
        file_menu.addSeparator()

        # Session save/load (shortcuts managed by registry)
        self._save_action = file_menu.addAction("Save Session", self._on_save_session)
        self._open_action = file_menu.addAction("Open Project...", self._on_open_project)

        file_menu.addSeparator()
        file_menu.addAction("Export Video...", self._on_export_video)
        file_menu.addAction("Export All Videos", self._on_export_all_videos)
        file_menu.addSeparator()
        file_menu.addAction("Return to Home", self._return_to_welcome)
        file_menu.addAction("Exit", self.close)

        # Edit menu
        edit_menu = menu_bar.addMenu("Edit")
        self._prefs_action = edit_menu.addAction("Preferences...", self._show_preferences)
        edit_menu.addAction("Hotkeys...", self._show_hotkeys)
        edit_menu.addSeparator()
        edit_menu.addAction("Track Paint Masks", self._on_track_masks)
        edit_menu.addAction("Clear Paint Strokes", self._on_clear_annotations)

        # View menu
        view_menu = menu_bar.addMenu("View")
        view_menu.addAction("Reset Layout", self._reset_layout)
        view_menu.addAction("Toggle Queue Panel", self._toggle_queue_panel)

        view_menu.addAction("Reset Zoom", self._on_reset_zoom)

        # Help menu
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("Console", self._toggle_debug_console)
        help_menu.addSeparator()
        help_menu.addAction("Report Issue...", self._show_report_issue)
        help_menu.addSeparator()
        help_menu.addAction("About", self._show_about)

        # Click sound on any menu action
        menu_bar.triggered.connect(lambda _: self._menu_click_sound())

        # Right corner: update button (hidden) + volume control
        self._corner_widget = QWidget()
        corner_layout = QHBoxLayout(self._corner_widget)
        corner_layout.setContentsMargins(0, 0, 4, 0)
        corner_layout.setSpacing(8)

        self._update_btn = QPushButton("Update Available")
        self._update_btn.setVisible(False)
        self._update_btn.setCursor(Qt.PointingHandCursor)
        self._update_btn.setStyleSheet(
            "QPushButton {"
            "  background: #FFF203; color: #141300; border: none;"
            "  border-radius: 3px; padding: 2px 10px;"
            "  font-family: 'Open Sans'; font-size: 11px; font-weight: 700;"
            "}"
            "QPushButton:hover { background: #E0D600; }"
        )
        self._update_btn.clicked.connect(self._run_update)
        corner_layout.addWidget(self._update_btn)

        from ui.widgets.volume_control import VolumeControl

        self._volume_control = VolumeControl(self._corner_widget)
        corner_layout.addWidget(self._volume_control)

        menu_bar.setCornerWidget(self._corner_widget)

    def _menu_click_sound(self) -> None:
        from ui.sounds.audio_manager import UIAudio

        UIAudio.click()

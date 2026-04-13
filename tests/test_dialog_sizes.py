"""Show every QMessageBox variant our app produces so we can check sizes."""

import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QDialogButtonBox,
    QScrollArea,
)
from PySide6.QtCore import QObject, QEvent

# Load the theme
THEME_PATH = "ui/theme/corridor_theme.qss"


class _MessageBoxFilter(QObject):
    """Auto-center buttons on every QMessageBox — same as real app."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Show and isinstance(obj, QMessageBox):
            for bb in obj.findChildren(QDialogButtonBox):
                bb.setCenterButtons(True)
        return False


def load_theme(app):
    import pathlib

    theme_dir = str(pathlib.Path(THEME_PATH).parent.resolve()).replace("\\", "/")
    raw = pathlib.Path(THEME_PATH).read_text(encoding="utf-8")
    app.setStyleSheet(raw.replace("{{THEME_DIR}}", theme_dir))


# Every QMessageBox in the app — exhaustive list from grep
# (type, title, message, buttons)
DIALOGS = [
    # main_window.py:496 — cancel extraction/GVM
    ("question", "Cancel", "Cancel extraction?", QMessageBox.Yes | QMessageBox.No),
    # main_window.py:630
    (
        "critical",
        "Force Stop Failed",
        "Could not restart inference engine:\nPermission denied",
        None,
    ),
    # main_window.py:687
    ("information", "No Annotations", "Paint green (1) and red (2) strokes on frames first.", None),
    # main_window.py:728
    (
        "question",
        "Replace Existing Alpha?",
        "This clip already has an AlphaHint (from GVM or a previous run).\n\n"
        "To use your annotations with VideoMaMa, the existing AlphaHint must\n"
        "be removed so it can be regenerated.\n\n"
        "Remove existing AlphaHint and proceed?",
        QMessageBox.Yes | QMessageBox.No,
    ),
    # main_window.py:773
    ("information", "Masks Exported", "Exported 120 mask frames to AlphaHint/", None),
    # main_window.py:790 — custom 3-button
    ("custom_clear", "Clear Annotations", "What would you like to clear?", None),
    # main_window.py:1104
    ("critical", "Scan Error", "Failed to scan clips directory:\nPermission denied", None),
    # main_window.py:1128
    ("warning", "Missing", "Workspace no longer exists:\nC:\\Users\\Johan\\clips", None),
    # main_window.py:1289
    ("information", "No Media", "No video files or image sequences found in that folder.", None),
    # main_window.py:1340
    (
        "information",
        "Already Imported",
        'All selected videos are already in the project ("clip1", "clip2").',
        None,
    ),
    # main_window.py:1374
    (
        "information",
        "No Images",
        "No image files found in that folder.\n\nSupported formats: PNG, JPG, TIFF, EXR, BMP",
        None,
    ),
    # main_window.py:1385
    (
        "information",
        "Already Imported",
        'This sequence is already in the project as "Woman_Jumps".',
        None,
    ),
    # main_window.py:1405
    (
        "warning",
        "Duplicate Filenames",
        "Found files with the same name but different extensions:\n"
        "frame_001.png, frame_001.jpg\n\n"
        "Only the first match per stem will be used.",
        None,
    ),
    # main_window.py:1464 — custom 3-button
    (
        "custom_import",
        "Import Image Frames",
        "You selected 24 images from a folder containing 120.\n\n"
        "Import just the selected files, or the full sequence?",
        None,
    ),
    # main_window.py:1562
    ("information", "No Media", "No video files or image sequences found in that folder.", None),
    # main_window.py:1741
    (
        "warning",
        "Not Ready",
        "Clip 'Woman_Jumps' is in EXTRACTING state.\n"
        "Wait for extraction to finish, or use a READY clip.",
        None,
    ),
    # main_window.py:1753 — custom 3-button
    (
        "custom_incomplete",
        "Incomplete Alpha",
        "Only 24 of 120 alpha hint frames found.\n\nProcess with available frames, or re-run GVM?",
        None,
    ),
    # main_window.py:1799
    ("information", "Duplicate", "'Woman_Jumps_For_Joy' is already queued.", None),
    # main_window.py:1826
    ("information", "Duplicate", "'Woman_Jumps_For_Joy' is already queued.", None),
    # main_window.py:1853
    ("information", "No Clips", "No READY clips to process.", None),
    # main_window.py:1901
    ("information", "Nothing to Process", "No selected clips are in a processable state.", None),
    # main_window.py:2011
    (
        "question",
        "Replace Alpha Hints?",
        "Clip 'Woman_Jumps' already has alpha hint images.\n\nReplace them with new ones?",
        QMessageBox.Yes | QMessageBox.No,
    ),
    # main_window.py:2045
    (
        "warning",
        "No Images",
        "No image files found in the selected folder.\nSupported: PNG, JPG, TIFF, EXR, BMP",
        None,
    ),
    # main_window.py:2058
    (
        "warning",
        "Frame Count Mismatch",
        "Clip 'Woman_Jumps' has 120 input frames but you "
        "selected 96 alpha images.\n\n"
        "Continue anyway?",
        QMessageBox.Ok | QMessageBox.Cancel,
    ),
    # main_window.py:2074
    (
        "question",
        "Import Alpha",
        "Import 120 alpha hint images?\n(24 frames will have no alpha hint)",
        QMessageBox.Yes | QMessageBox.No,
    ),
    # main_window.py:2132 — custom 3-button
    (
        "custom_partial",
        "Partial Alpha Found",
        "Found 24 of 120 alpha frames from a previous run.\n\n"
        "Resume from where it left off, or regenerate all?",
        None,
    ),
    # main_window.py:2209
    (
        "question",
        "Force Stop",
        "The current GPU step has not returned to Python.\n\n"
        "Force-kill the inference engine process?\n"
        "This will lose any in-progress frame.",
        QMessageBox.Yes | QMessageBox.No,
    ),
    # main_window.py:2223
    ("question", "Cancel", "Cancel processing?", QMessageBox.Yes | QMessageBox.No),
    # main_window.py:2437
    (
        "critical",
        "Processing Error",
        "Clip: Woman_Jumps_For_Joy\n\n"
        "FileNotFoundError: [Errno 2] No such file or directory: triton kernel cache",
        None,
    ),
    # main_window.py:2633
    ("information", "No Clip", "Select a clip first.", None),
    # main_window.py:2638
    ("warning", "Not Complete", "Clip 'Woman_Jumps' must be COMPLETE to export video.", None),
    # main_window.py:2655
    ("warning", "No Output", "No output frames found to export.", None),
    # main_window.py:2661
    (
        "critical",
        "FFmpeg Not Found",
        "FFmpeg is required for video export.\nInstall it and make sure it's on your PATH.",
        None,
    ),
    # main_window.py:2704
    ("information", "Export Complete", "Video exported:\nC:\\Users\\Johan\\output\\clip.mp4", None),
    # main_window.py:2712
    ("critical", "Export Failed", "Failed to export video:\nFFmpeg returned code 1", None),
    # main_window.py:2840
    ("information", "No Folder", "Open a clips folder first.", None),
    # main_window.py:3061
    (
        "question",
        "Update EZ-CorridorKey",
        "This will save your session, close the app, and run the updater.\nContinue?",
        QMessageBox.Yes | QMessageBox.No,
    ),
    # io_tray_panel.py:493
    ("information", "No Markers", "No clips have in/out markers set.", None),
    # io_tray_panel.py:501
    (
        "question",
        "Reset In/Out Markers",
        "This will clear in/out markers on 5 clips.\n\nContinue?",
        QMessageBox.Yes | QMessageBox.No,
    ),
    # io_tray_panel.py:512
    (
        "warning",
        "Confirm Reset",
        "Are you sure? This cannot be undone.\n\n5 clips will have their in/out markers removed.",
        QMessageBox.Ok | QMessageBox.Cancel,
    ),
    # io_tray_panel.py:773
    (
        "question",
        "Clear Outputs",
        "Remove all output files for 3 clip(s)?\nClip1, Clip2, Clip3\n\nThis cannot be undone.",
        QMessageBox.Yes | QMessageBox.No,
    ),
    # io_tray_panel.py:815 — custom with detailed text
    ("custom_remove", "Remove Clips", "Remove 3 clips from the project?", None),
    # hotkeys_dialog.py:93
    (
        "warning",
        "Shortcut Conflict",
        '"Ctrl+S" is already assigned to:\nSave Session\n\nReassign it?',
        QMessageBox.Yes | QMessageBox.No,
    ),
    # hotkeys_dialog.py:309
    (
        "question",
        "Reset All Shortcuts",
        "Reset all shortcuts to their default values?",
        QMessageBox.Yes | QMessageBox.No,
    ),
    # recent_projects_panel.py:237 — custom 3-button
    ("custom_remove_project", "Remove Project", 'Remove "My Project" from recent projects?', None),
    # recent_projects_panel.py:259
    (
        "warning",
        "Confirm Delete",
        "Permanently delete this project folder?\n\nC:\\Users\\Johan\\clips",
        QMessageBox.Yes | QMessageBox.No,
    ),
    # recent_projects_panel.py:282
    ("warning", "Delete Failed", "Could not delete project:\nPermission denied", None),
]


class DialogTester(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dialog Size Tester")
        self.setMinimumWidth(500)
        central = QWidget()
        self.setCentralWidget(central)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)

        for dlg in DIALOGS:
            label = f'{dlg[0].upper()}: {dlg[1]} - "{dlg[2][:60]}..."'
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, d=dlg: self._show(d))
            layout.addWidget(btn)

        show_all = QPushButton(">>> SHOW ALL ONE BY ONE <<<")
        show_all.setStyleSheet(
            "background: #FFF203; color: #000; font-weight: bold; padding: 10px;"
        )
        show_all.clicked.connect(self._show_all)
        layout.addWidget(show_all)

        scroll.setWidget(inner)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _show(self, dlg):
        kind, title, text, buttons = dlg

        if kind == "question":
            box = QMessageBox(
                QMessageBox.Question,
                title,
                text,
                buttons or (QMessageBox.Yes | QMessageBox.No),
                self,
            )
            box.setDefaultButton(QMessageBox.No)
        elif kind == "critical":
            box = QMessageBox(QMessageBox.Critical, title, text, QMessageBox.Ok, self)
        elif kind == "information":
            box = QMessageBox(QMessageBox.Information, title, text, QMessageBox.Ok, self)
        elif kind == "warning":
            box = QMessageBox(QMessageBox.Warning, title, text, buttons or QMessageBox.Ok, self)
        elif kind == "custom_clear":
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle(title)
            box.setText(text)
            box.addButton("This Frame", QMessageBox.AcceptRole)
            box.addButton("Entire Clip", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
        elif kind == "custom_import":
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle(title)
            box.setText(text)
            box.addButton("Copy Just These 24", QMessageBox.AcceptRole)
            box.addButton("Import Full Sequence", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Cancel)
        elif kind == "custom_incomplete":
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle(title)
            box.setText(text)
            box.addButton("Process Available", QMessageBox.AcceptRole)
            box.addButton("Re-run GVM", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Cancel)
        elif kind == "custom_partial":
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle(title)
            box.setText(text)
            box.addButton("Resume", QMessageBox.AcceptRole)
            box.addButton("Regenerate", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
        elif kind == "custom_remove":
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle(title)
            box.setText(text)
            box.addButton("Remove from Project", QMessageBox.DestructiveRole)
            box.addButton("Delete from Disk", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
        elif kind == "custom_remove_project":
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle(title)
            box.setText(text)
            box.addButton("Remove from List", QMessageBox.AcceptRole)
            box.addButton("Delete from Disk", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
        else:
            return

        box.exec()
        size = box.size()
        print(f"[{kind.upper():12s}] {title:20s} -> {size.width()}x{size.height()}px")

    def _show_all(self):
        for dlg in DIALOGS:
            self._show(dlg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    load_theme(app)

    # Install the same event filter as the real app
    filt = _MessageBoxFilter(app)
    app.installEventFilter(filt)

    win = DialogTester()
    win.show()
    sys.exit(app.exec())

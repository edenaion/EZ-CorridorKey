"""First-launch setup wizard — download model checkpoints with progress.

Shown automatically when the required CorridorKey checkpoint is missing.
Provides the same model selection as 1-install.sh / 1-install.bat but in
a PySide6 GUI with checkboxes and progress bars.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import platform
import sys
import traceback
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QProgressBar, QWidget, QScrollArea,
    QFrame,
)
from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtGui import QFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent.parent


def _data_root() -> Path:
    if not getattr(sys, "frozen", False):
        return _project_root()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "CorridorKey"
    elif sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home())) / "CorridorKey"
    else:
        return Path.home() / ".local" / "share" / "CorridorKey"


def _checkpoint_dir() -> Path:
    return _data_root() / "CorridorKeyModule" / "checkpoints"


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

_IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

MODELS: list[dict] = [
    {
        "key": "corridorkey",
        "label": "CorridorKey Model",
        "size": "383 MB",
        "required": True,
        "description": "Core green-screen keying model (required)",
        "default_checked": True,
    },
]

if _IS_APPLE_SILICON:
    MODELS.append({
        "key": "corridorkey-mlx",
        "label": "CorridorKey MLX (Apple Silicon)",
        "size": "380 MB",
        "required": False,
        "description": "1.5-2x faster inference on Apple Silicon",
        "default_checked": True,
    })

MODELS += [
    {
        "key": "sam2",
        "label": "SAM2 Tracker (Base+)",
        "size": "324 MB",
        "required": False,
        "description": "Object tracking for multi-frame consistency",
        "default_checked": True,
    },
    {
        "key": "gvm",
        "label": "GVM Alpha Generator",
        "size": "~6 GB",
        "required": False,
        "description": "AI-generated alpha mattes",
        "default_checked": False,
    },
    {
        "key": "videomama",
        "label": "VideoMaMa Alpha Generator",
        "size": "~37 GB",
        "required": False,
        "description": "Video-based AI alpha (large download)",
        "default_checked": False,
    },
]


def needs_setup() -> bool:
    import glob
    ckpt_dir = _checkpoint_dir()
    return len(glob.glob(str(ckpt_dir / "*.pth"))) == 0


# ---------------------------------------------------------------------------
# Import setup_models
# ---------------------------------------------------------------------------

def _load_setup_models():
    script_path = _project_root() / "scripts" / "setup_models.py"
    if not script_path.is_file():
        raise ImportError(f"setup_models.py not found at {script_path}")
    spec = importlib.util.spec_from_file_location("setup_models", str(script_path))
    if not spec or not spec.loader:
        raise ImportError(f"Cannot create import spec for {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Download worker
# ---------------------------------------------------------------------------

class _DownloadWorker(QThread):
    """Download selected models sequentially in a background thread.

    IMPORTANT: This thread must be properly waited on before the parent
    dialog is destroyed, otherwise Qt will abort() on ~QThread.
    """
    model_started = Signal(str, str)          # key, label
    model_progress = Signal(str, int, int)    # key, downloaded_mb, total_mb
    model_finished = Signal(str, bool, str)   # key, success, message
    all_finished = Signal(bool)               # overall success

    def __init__(self, selected_keys: list[str], parent=None):
        super().__init__(parent)
        self._selected = selected_keys
        self._cancelled = False
        self._current_key = ""

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            setup_models = _load_setup_models()
        except Exception as e:
            logger.exception("Failed to load setup_models")
            self.all_finished.emit(False)
            return

        self._patch_mlx_progress(setup_models)

        all_ok = True
        for key in self._selected:
            if self._cancelled:
                break

            self._current_key = key
            label = next((m["label"] for m in MODELS if m["key"] == key), key)
            self.model_started.emit(key, label)

            try:
                ok = self._download_one(setup_models, key)
            except Exception as e:
                logger.exception("Download failed for %s", key)
                ok = False

            if self._cancelled:
                break

            self.model_finished.emit(key, ok, "OK" if ok else "Failed")
            if not ok:
                all_ok = False

        self.all_finished.emit(all_ok)

    def _download_one(self, setup_models, key: str) -> bool:
        if key == "corridorkey":
            return setup_models.download_model("corridorkey")
        elif key == "corridorkey-mlx":
            return setup_models.download_corridorkey_mlx()
        elif key == "sam2":
            return setup_models.download_sam2_model("base-plus")
        elif key == "gvm":
            return setup_models.download_model("gvm")
        elif key == "videomama":
            return setup_models.download_model("videomama")
        return False

    def _patch_mlx_progress(self, setup_models):
        worker = self
        original = setup_models.download_corridorkey_mlx

        def patched():
            import urllib.request
            cfg = setup_models.MLX_CHECKPOINT
            local_dir = cfg["local_dir"]
            local_dir.mkdir(parents=True, exist_ok=True)
            dest = local_dir / cfg["filename"]
            if dest.is_file():
                return True
            if not setup_models.check_disk_space(cfg["size_bytes"], local_dir):
                return False

            tmp = dest.with_suffix(".safetensors.tmp")
            try:
                def hook(bn, bs, total):
                    if total > 0 and not worker._cancelled:
                        worker.model_progress.emit(
                            worker._current_key,
                            (bn * bs) // (1024 * 1024),
                            total // (1024 * 1024),
                        )
                urllib.request.urlretrieve(cfg["url"], str(tmp), reporthook=hook)
                try:
                    import hashlib
                    resp = urllib.request.urlopen(cfg["sha256_url"])
                    expected = resp.read().decode().strip().split()[0]
                    actual = hashlib.sha256(tmp.read_bytes()).hexdigest()
                    if actual != expected:
                        tmp.unlink(missing_ok=True)
                        return False
                except Exception:
                    pass
                tmp.rename(dest)
                return True
            except Exception:
                tmp.unlink(missing_ok=True)
                return False

        setup_models.download_corridorkey_mlx = patched

        try:
            import huggingface_hub
            orig_hf = huggingface_hub.hf_hub_download
            def patched_hf(*a, **kw):
                r = orig_hf(*a, **kw)
                worker.model_progress.emit(worker._current_key, 100, 100)
                return r
            huggingface_hub.hf_hub_download = patched_hf
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Model row widget
# ---------------------------------------------------------------------------

class _ModelRow(QWidget):
    def __init__(self, model: dict, parent=None):
        super().__init__(parent)
        self.key = model["key"]

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(12)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(model["default_checked"])
        if model["required"]:
            self.checkbox.setChecked(True)
            self.checkbox.setEnabled(False)
        layout.addWidget(self.checkbox)

        info = QVBoxLayout()
        info.setSpacing(2)
        label = QLabel(f"<b>{model['label']}</b>")
        label.setStyleSheet("color: #FFFFFF;")
        info.addWidget(label)
        desc = QLabel(f"{model['description']}  ({model['size']})")
        desc.setStyleSheet("color: #999999; font-size: 11px;")
        info.addWidget(desc)
        layout.addLayout(info, 1)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setFixedHeight(18)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.progress.setFormat("%v / %m MB")
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444; border-radius: 4px;
                background: #1a1a18; text-align: center;
                color: #ccc; font-size: 11px;
            }
            QProgressBar::chunk { background: #FFF203; border-radius: 3px; }
        """)
        layout.addWidget(self.progress)

        self.status_icon = QLabel("")
        self.status_icon.setFixedWidth(32)
        self.status_icon.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        layout.addWidget(self.status_icon)

    def is_selected(self) -> bool:
        return self.checkbox.isChecked()

    def set_downloading(self):
        self.checkbox.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Downloading...")
        self.status_icon.setText("")

    def set_progress(self, downloaded_mb: int, total_mb: int):
        if total_mb > 0:
            self.progress.setRange(0, total_mb)
            self.progress.setValue(min(downloaded_mb, total_mb))
            self.progress.setFormat(f"{downloaded_mb} / {total_mb} MB")

    def set_finished(self, ok: bool):
        self.progress.setVisible(False)
        if ok:
            self.status_icon.setText("\u2714")
            self.status_icon.setStyleSheet("color: #4CAF50; font-size: 20px;")
        else:
            self.status_icon.setText("\u2718")
            self.status_icon.setStyleSheet("color: #F44336; font-size: 20px;")


# ---------------------------------------------------------------------------
# Setup Wizard Dialog
# ---------------------------------------------------------------------------

class SetupWizard(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EZ-CorridorKey Setup")
        self.setMinimumSize(700, 500)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._worker: _DownloadWorker | None = None
        self._downloading = False
        self._rows: dict[str, _ModelRow] = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(28, 28, 28, 28)

        title = QLabel("Welcome to EZ-CorridorKey")
        title.setFont(QFont("Open Sans", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #FFF203;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Select which models to download. The core CorridorKey model is required.\n"
            "Optional models can be downloaded later from Preferences."
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #CCCCCC;")
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        model_container = QWidget()
        model_container.setStyleSheet("background: transparent;")
        model_layout = QVBoxLayout(model_container)
        model_layout.setSpacing(4)
        model_layout.setContentsMargins(0, 0, 0, 0)

        for model in MODELS:
            row = _ModelRow(model)
            self._rows[model["key"]] = row
            model_layout.addWidget(row)

        model_layout.addStretch()
        scroll.setWidget(model_container)
        layout.addWidget(scroll, 1)

        self._overall_label = QLabel("")
        self._overall_label.setAlignment(Qt.AlignCenter)
        self._overall_label.setStyleSheet("color: #999;")
        layout.addWidget(self._overall_label)

        # Buttons — centered
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._cancel_btn = QPushButton("Cancel && Exit")
        self._cancel_btn.setStyleSheet(
            "QPushButton { color: #F44336; border: 1px solid #F44336; padding: 8px 16px; "
            "border-radius: 4px; }"
            "QPushButton:hover { background: #F44336; color: #fff; }"
        )
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._cancel_btn)

        btn_layout.addSpacing(12)

        self._install_btn = QPushButton("Download && Install")
        self._install_btn.setStyleSheet(
            "QPushButton { background: #FFF203; color: #000; font-weight: bold; "
            "padding: 8px 24px; border-radius: 4px; }"
            "QPushButton:hover { background: #E6D900; }"
            "QPushButton:disabled { background: #555; color: #999; }"
        )
        self._install_btn.clicked.connect(self._on_install)
        btn_layout.addWidget(self._install_btn)

        btn_layout.addStretch()

        layout.addLayout(btn_layout)

    def _on_cancel(self):
        if self._downloading:
            # Signal cancel but let thread finish gracefully
            self._cancel_btn.setEnabled(False)
            self._cancel_btn.setText("Cancelling...")
            if self._worker:
                self._worker.cancel()
            # Don't reject yet — wait for all_finished signal
        else:
            self.reject()

    def _on_install(self):
        selected = [key for key, row in self._rows.items() if row.is_selected()]
        if not selected:
            self.accept()
            return

        self._downloading = True
        self._install_btn.setEnabled(False)
        for row in self._rows.values():
            row.checkbox.setEnabled(False)

        self._completed_count = 0
        self._total_count = len(selected)
        self._overall_label.setText(f"Preparing downloads (0/{self._total_count})...")
        self._overall_label.setStyleSheet("color: #FFF203;")

        # Create worker as child of this dialog so it stays alive
        self._worker = _DownloadWorker(selected, parent=self)
        self._worker.model_started.connect(self._on_model_started)
        self._worker.model_progress.connect(self._on_model_progress)
        self._worker.model_finished.connect(self._on_model_finished)
        self._worker.all_finished.connect(self._on_all_finished)
        self._worker.start()

    def _on_model_started(self, key: str, label: str):
        if key in self._rows:
            self._rows[key].set_downloading()
        self._overall_label.setText(
            f"Downloading {self._completed_count + 1}/{self._total_count}: {label}..."
        )

    def _on_model_progress(self, key: str, downloaded_mb: int, total_mb: int):
        if key in self._rows:
            self._rows[key].set_progress(downloaded_mb, total_mb)

    def _on_model_finished(self, key: str, ok: bool, msg: str):
        if key in self._rows:
            self._rows[key].set_finished(ok)
        self._completed_count += 1

    def _on_all_finished(self, success: bool):
        # Wait for thread to fully exit before allowing dialog destruction
        if self._worker:
            self._worker.wait()
            self._worker = None

        self._downloading = False

        # If user pressed cancel, reject now that thread is safely done
        if not self._cancel_btn.isEnabled() and self._cancel_btn.text() == "Cancelling...":
            self.reject()
            return

        if success:
            self._overall_label.setText(f"All {self._total_count} downloads complete!")
            self._overall_label.setStyleSheet("color: #4CAF50;")
        else:
            self._overall_label.setText(
                "Some downloads failed. You can retry from Preferences later."
            )
            self._overall_label.setStyleSheet("color: #F44336;")

        self._install_btn.setText("Continue")
        self._install_btn.setEnabled(True)
        try:
            self._install_btn.clicked.disconnect()
        except RuntimeError:
            pass
        self._install_btn.clicked.connect(self.accept)

        self._cancel_btn.setVisible(False)

    def closeEvent(self, event):
        """Block close while downloading — user must use Cancel button."""
        if self._downloading:
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self):
        """Block reject (Escape key / close) while downloading."""
        if self._downloading:
            return
        super().reject()

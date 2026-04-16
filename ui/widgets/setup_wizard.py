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
    QPushButton, QProgressBar, QWidget,
    QFrame, QLineEdit, QFileDialog,
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


def _default_install_dir() -> Path:
    """Platform-appropriate default location for model data."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "EZ-CorridorKey"
    elif sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home())) / "EZ-CorridorKey"
    else:
        return Path.home() / ".local" / "share" / "EZ-CorridorKey"


def _data_root() -> Path:
    """Writable data root where models are stored.

    Dev mode: project root. Frozen: delegates to backend.project.get_data_dir().
    """
    if not getattr(sys, "frozen", False):
        return _project_root()
    try:
        from backend.project import get_data_dir
        return Path(get_data_dir())
    except ImportError:
        return _default_install_dir()


def _checkpoint_dir() -> Path:
    return _data_root() / "CorridorKeyModule" / "checkpoints"


def _hf_hub_cache_dir() -> Path:
    """Hugging Face hub cache location, respecting HF_HOME / HF_HUB_CACHE env vars.

    SAM2 checkpoints are downloaded into this shared cache regardless of the
    user's chosen install path, so the wizard detects them here — same location
    across fresh installs, reinstalls, and folder changes.
    """
    override = os.environ.get("HF_HUB_CACHE")
    if override:
        return Path(override)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _saved_install_path() -> Path:
    """Return the user's last-saved install path, falling back to default."""
    try:
        from PySide6.QtCore import QSettings
        saved = QSettings().value("app/install_path", "", type=str)
        if saved:
            return Path(saved)
    except Exception:
        pass
    return _default_install_dir()


def _current_version() -> str:
    """Read version from pyproject.toml (frozen or dev). 0.0.0 on failure."""
    try:
        import tomllib
    except Exception:
        return "0.0.0"
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys._MEIPASS) / "pyproject.toml")
    candidates.append(_project_root() / "pyproject.toml")
    for path in candidates:
        try:
            with open(path, "rb") as fh:
                return tomllib.load(fh)["project"]["version"]
        except Exception:
            continue
    return "0.0.0"


def detect_installed_models(install_path: Path | None = None) -> dict[str, bool]:
    """Return {model_key: present} for every model in the wizard catalogue.

    Models that live under the install path (CorridorKey, MLX, GVM, VideoMaMa,
    MatAnyone2) are resolved against ``install_path``. Models that live in the
    Hugging Face hub cache (SAM2) are resolved against the HF cache regardless
    of the install path — that's how hf_hub_download stores them without a
    local_dir override.

    Detection is intentionally decoupled from scripts/setup_models.py because
    that module bakes ``PROJECT_ROOT`` at import time, so re-scanning after a
    path change would see stale paths.
    """
    root = Path(install_path) if install_path else _saved_install_path()
    hf = _hf_hub_cache_dir()

    results: dict[str, bool] = {}

    # CorridorKey core .pth — any .pth in the checkpoints dir counts
    ck_dir = root / "CorridorKeyModule" / "checkpoints"
    try:
        results["corridorkey"] = ck_dir.is_dir() and any(ck_dir.glob("*.pth"))
    except OSError:
        results["corridorkey"] = False

    # CorridorKey MLX — specific .safetensors filename
    results["corridorkey-mlx"] = (
        ck_dir / "corridorkey_mlx.safetensors"
    ).is_file()

    # SAM2 Base+ — HF hub cache (shared across installs)
    results["sam2"] = _hf_cache_has("facebook/sam2.1-hiera-base-plus", hf)

    # GVM — unet safetensors
    results["gvm"] = (
        root / "gvm_core" / "weights" / "unet" / "diffusion_pytorch_model.safetensors"
    ).is_file()

    # MatAnyone2 — single .pth
    results["matanyone2"] = (
        root / "modules" / "MatAnyone2Module" / "checkpoints" / "matanyone2.pth"
    ).is_file()

    # BiRefNet default Matting variant — any .safetensors in the snapshot dir
    brn_dir = root / "modules" / "BiRefNetModule" / "checkpoints" / "BiRefNet-matting"
    try:
        results["birefnet"] = brn_dir.is_dir() and any(
            p.suffix == ".safetensors" for p in brn_dir.iterdir() if p.is_file()
        )
    except OSError:
        results["birefnet"] = False

    # VideoMaMa — needs BOTH the VideoMaMa weights and the SVD base model
    vm_root = root / "VideoMaMaInferenceModule" / "checkpoints"
    vm_main = vm_root / "VideoMaMa" / "diffusion_pytorch_model.safetensors"
    vm_base = vm_root / "stable-video-diffusion-img2vid-xt" / "model_index.json"
    results["videomama"] = vm_main.is_file() and vm_base.is_file()

    return results


def _hf_cache_has(repo_id: str, cache_dir: Path) -> bool:
    """True when any snapshot for ``repo_id`` exists under the HF hub cache.

    huggingface_hub stores repos as ``models--<org>--<name>/snapshots/<rev>/``.
    We don't probe for a specific filename because users may have downloaded
    different SAM2 variants (small / base-plus / large) and we just want to
    know whether the shared cache already has a usable checkpoint.
    """
    try:
        if not cache_dir.is_dir():
            return False
        repo_folder = "models--" + repo_id.replace("/", "--")
        snapshots = cache_dir / repo_folder / "snapshots"
        if not snapshots.is_dir():
            return False
        for snap in snapshots.iterdir():
            if snap.is_dir() and any(snap.iterdir()):
                return True
    except OSError:
        pass
    return False


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
        "key": "matanyone2",
        "label": "MatAnyone2 Alpha Generator",
        "size": "141 MB",
        "required": False,
        "description": "Matting-based AI alpha from brush prompts",
        "default_checked": True,
    },
    {
        "key": "birefnet",
        "label": "BiRefNet Matting",
        "size": "~940 MB",
        "required": False,
        "description": "Automatic alpha generator (subject matting). "
                       "Optional: skipped downloads auto-fetch on first use.",
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


def _required_model_keys() -> list[str]:
    return [m["key"] for m in MODELS if m.get("required")]


def needs_setup() -> bool:
    """True when the wizard should auto-open at startup.

    Two triggers:
    1. A required model is missing at the current install path. The user
       cannot use the app without this, so we force-show the wizard.
    2. The installed version has changed since the last successful launch
       (fresh install or upgrade — including the 1.9.1 -> 1.9.2 jump). This
       gives new/upgrading users a look at the catalogue so they can pick up
       any optional models they want, and is dismissable via Cancel/X.
    """
    try:
        install_path = _saved_install_path()
        present = detect_installed_models(install_path)
        for key in _required_model_keys():
            if not present.get(key, False):
                return True
    except Exception:
        logger.exception("detect_installed_models failed during needs_setup()")
        # Safe fallback: mirror the old behaviour so a broken detect doesn't
        # leave users stuck with no CorridorKey weights and no wizard.
        import glob
        ckpt_dir = _checkpoint_dir()
        if len(glob.glob(str(ckpt_dir / "*.pth"))) == 0:
            return True

    try:
        from PySide6.QtCore import QSettings
        last_seen = QSettings().value("app/version_last_seen", "", type=str)
        if last_seen != _current_version():
            return True
    except Exception:
        pass

    return False


def has_required_models() -> bool:
    """True when every required model is present at the current install path.

    Used by main.py to decide whether to exit after the wizard closes: if the
    user cancelled without installing a required model, we still can't launch.
    A version-bump-only trigger with all requireds present lets the user X out
    and keep going.
    """
    try:
        present = detect_installed_models(_saved_install_path())
    except Exception:
        logger.exception("detect_installed_models failed during has_required_models()")
        return False
    return all(present.get(k, False) for k in _required_model_keys())


def mark_setup_seen() -> None:
    """Persist the current version so the version-bump trigger doesn't fire again.

    Called from main.py after the wizard runs (or is skipped) so that the next
    launch on the same version does not re-show the wizard. A later version
    upgrade bumps the key and re-triggers it once.
    """
    try:
        from PySide6.QtCore import QSettings
        QSettings().setValue("app/version_last_seen", _current_version())
    except Exception:
        logger.exception("Failed to persist app/version_last_seen")


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
        elif key == "matanyone2":
            return setup_models.download_matanyone2()
        elif key == "birefnet":
            return setup_models.download_birefnet()
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
        self._required = bool(model["required"])
        self._default_checked = bool(model["default_checked"])
        self._base_desc = f"{model['description']}  ({model['size']})"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(12)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(self._default_checked)
        if self._required:
            self.checkbox.setChecked(True)
            self.checkbox.setEnabled(False)
        layout.addWidget(self.checkbox)

        info = QVBoxLayout()
        info.setSpacing(2)
        label = QLabel(model["label"])
        label_font = QFont()
        label_font.setBold(True)
        label_font.setPointSize(13)
        label.setFont(label_font)
        label.setStyleSheet("color: #FFFFFF; background: transparent;")
        info.addWidget(label)
        self._desc_label = QLabel(self._base_desc)
        desc_font = QFont()
        desc_font.setPointSize(11)
        self._desc_label.setFont(desc_font)
        self._desc_label.setStyleSheet("color: #999999; background: transparent;")
        info.addWidget(self._desc_label)
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

    def set_installed_state(self, installed: bool) -> None:
        """Reflect whether this model is already present at the current install path.

        Installed: green check, checkbox disabled+unchecked (nothing to do), label
        shows "Installed". Missing: restore default checked state and the model
        description. Required models stay checked+disabled in both states so the
        user can't accidentally deselect them.
        """
        self.progress.setVisible(False)
        if installed:
            self.checkbox.blockSignals(True)
            self.checkbox.setChecked(False)
            self.checkbox.setEnabled(False)
            self.checkbox.blockSignals(False)
            self._desc_label.setText(self._base_desc + "  — Installed")
            self._desc_label.setStyleSheet(
                "color: #4CAF50; background: transparent;"
            )
            self.status_icon.setText("\u2714")
            self.status_icon.setStyleSheet("color: #4CAF50; font-size: 20px;")
        else:
            self.checkbox.blockSignals(True)
            self.checkbox.setChecked(self._default_checked or self._required)
            self.checkbox.setEnabled(not self._required)
            self.checkbox.blockSignals(False)
            self._desc_label.setText(self._base_desc)
            self._desc_label.setStyleSheet(
                "color: #999999; background: transparent;"
            )
            self.status_icon.setText("")
            self.status_icon.setStyleSheet("")

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

        title = QLabel('Welcome to <span style="color:#FFF203;">EZ</span><span style="color:#4CAF50;">-</span><span style="color:#FFF203;">Corridor</span><span style="color:#4CAF50;">Key</span>')
        title.setFont(QFont("Open Sans", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #FFFFFF;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Select which models to download. The core CorridorKey model is required.\n"
            "Optional models can be downloaded later from Edit \u2192 Download Manager."
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #CCCCCC;")
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        # Install path picker
        loc_label = QLabel("Install path:")
        loc_label.setStyleSheet("color: #CCCCCC;")
        layout.addWidget(loc_label)

        loc_row = QHBoxLayout()
        loc_row.setSpacing(8)
        # Pre-fill with the user's previously saved install path so a re-open
        # of the wizard scans the folder the user is actually installed to —
        # not a fresh default that would report everything as missing.
        self._loc_edit = QLineEdit(str(_saved_install_path()))
        self._loc_edit.setStyleSheet(
            "QLineEdit { background: #1a1a18; color: #fff; border: 1px solid #444; "
            "border-radius: 4px; padding: 6px; }"
        )
        self._loc_edit.setReadOnly(True)
        self._loc_edit.textChanged.connect(self._refresh_installed_state)
        loc_row.addWidget(self._loc_edit, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet(
            "QPushButton { color: #ccc; border: 1px solid #555; padding: 6px 14px; "
            "border-radius: 4px; }"
            "QPushButton:hover { color: #fff; border-color: #888; }"
        )
        browse_btn.clicked.connect(self._on_browse_location)
        loc_row.addWidget(browse_btn)

        default_btn = QPushButton("Default Location")
        default_btn.setToolTip(
            "Reset the install path to the platform default "
            "(in case you changed it and want to return)."
        )
        default_btn.setStyleSheet(
            "QPushButton { color: #ccc; border: 1px solid #555; padding: 6px 14px; "
            "border-radius: 4px; }"
            "QPushButton:hover { color: #fff; border-color: #888; }"
        )
        default_btn.clicked.connect(self._on_default_location)
        loc_row.addWidget(default_btn)
        layout.addLayout(loc_row)

        layout.addSpacing(8)

        # Scrollable model list — labels always display full width
        from PySide6.QtWidgets import QScrollArea
        model_container = QWidget()
        model_layout = QVBoxLayout(model_container)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(4)
        for model in MODELS:
            row = _ModelRow(model)
            self._rows[model["key"]] = row
            model_layout.addWidget(row)
        model_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(model_container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }"
                             "QWidget { background: transparent; }")
        layout.addWidget(scroll, 1)

        # Divider + desktop shortcut
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("color: #333;")
        layout.addWidget(divider)

        self._shortcut_check = QCheckBox("Create Desktop shortcut")
        self._shortcut_check.setChecked(True)
        self._shortcut_check.setStyleSheet(
            "QCheckBox { color: #4CAF50; }"
            "QCheckBox::indicator:checked { background: #4CAF50; border: 1px solid #4CAF50; border-radius: 2px; }"
            "QCheckBox::indicator { border: 1px solid #555; border-radius: 2px; width: 14px; height: 14px; }"
        )
        layout.addWidget(self._shortcut_check)

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

        # Initial scan of the pre-filled install path so present/missing
        # state is visible the moment the wizard opens — not only after the
        # user interacts with the path picker.
        self._refresh_installed_state()

    def _refresh_installed_state(self) -> None:
        """Scan the current install path and update each _ModelRow accordingly.

        Called on wizard open and again every time ``_loc_edit`` changes
        (Browse / Default Location). Downloads in progress must not be
        clobbered — we skip the refresh if the worker is running.
        """
        if self._downloading:
            return
        install_path = Path(self._loc_edit.text()) if self._loc_edit.text() else None
        try:
            present = detect_installed_models(install_path)
        except Exception:
            logger.exception("Failed to detect installed models")
            return
        for key, row in self._rows.items():
            row.set_installed_state(present.get(key, False))

    def _on_browse_location(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose Install Location", self._loc_edit.text()
        )
        if path:
            self._loc_edit.setText(path)

    def _on_default_location(self):
        """Reset the install path field to the platform default and re-scan."""
        self._loc_edit.setText(str(_default_install_dir()))

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

        # Save chosen install directory
        install_dir = self._loc_edit.text()
        from PySide6.QtCore import QSettings
        QSettings().setValue("app/install_path", install_dir)

        self._downloading = True
        self._install_btn.setEnabled(False)
        self._loc_edit.setEnabled(False)
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
                "Some downloads failed. You can retry from Edit \u2192 Download Manager."
            )
            self._overall_label.setStyleSheet("color: #F44336;")

        # Create desktop shortcut if requested
        if self._shortcut_check.isChecked():
            self._create_desktop_shortcut()

        self._install_btn.setText("Continue")
        self._install_btn.setEnabled(True)
        try:
            self._install_btn.clicked.disconnect()
        except RuntimeError:
            pass
        self._install_btn.clicked.connect(self.accept)

        self._cancel_btn.setVisible(False)

    def _create_desktop_shortcut(self):
        """Create a desktop shortcut/alias to the .app bundle."""
        try:
            if not getattr(sys, "frozen", False):
                return
            desktop = Path.home() / "Desktop"
            if not desktop.is_dir():
                return

            if sys.platform == "darwin":
                # macOS: create a symlink to the .app on the Desktop
                app_path = Path(sys.executable).resolve().parent.parent.parent
                link = desktop / app_path.name
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(app_path)
                logger.info("Desktop shortcut created: %s -> %s", link, app_path)
            elif sys.platform == "win32":
                # Windows: create a .lnk shortcut via PowerShell
                import subprocess
                app_exe = sys.executable
                icon = Path(sys._MEIPASS) / "ui" / "theme" / "corridorkey.ico"
                lnk = desktop / "CorridorKey.lnk"
                ps_cmd = (
                    f'$ws = New-Object -ComObject WScript.Shell; '
                    f'$s = $ws.CreateShortcut("{lnk}"); '
                    f'$s.TargetPath = "{app_exe}"; '
                    f'$s.IconLocation = "{icon},0"; '
                    f'$s.Description = "CorridorKey - AI Green Screen"; '
                    f'$s.Save()'
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    capture_output=True, timeout=10,
                )
                logger.info("Desktop shortcut created: %s", lnk)
        except Exception as e:
            logger.warning("Failed to create desktop shortcut: %s", e)

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

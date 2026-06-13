"""CorridorKey entry point — GUI (default) or CLI mode.

Usage:
    python main.py              # Launch GUI (default)
    python main.py --gui        # Launch GUI explicitly
    python main.py --cli        # Run CLI wizard (original clip_manager.py)
"""
from __future__ import annotations

import os
# Enable OpenEXR support in OpenCV — must be set before cv2 is imported anywhere
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import json
import logging
import logging.handlers
import platform
import sys
from datetime import datetime
from pathlib import Path


class _NullTextStream:
    """Minimal text stream used when GUI launches without a console."""

    encoding = "utf-8"
    errors = "replace"

    def write(self, data) -> int:
        if data is None:
            return 0
        if isinstance(data, bytes):
            return len(data)
        return len(str(data))

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


def ensure_standard_streams() -> None:
    """Provide harmless stdout/stderr sinks for pythonw/PyInstaller GUI launches."""
    if sys.stdout is None:
        sys.stdout = _NullTextStream()
    if sys.stderr is None:
        sys.stderr = _NullTextStream()


def get_base_dir() -> str:
    """Get the project base directory, handling both dev and frozen (PyInstaller) modes.

    In development: returns the directory containing this file.
    In frozen build: returns sys._MEIPASS (PyInstaller temp dir) for bundled
    resources, or the executable's directory for user files.
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def is_portable() -> bool:
    """True when a 'portable.txt' marker exists next to the executable."""
    if getattr(sys, 'frozen', False):
        return os.path.isfile(os.path.join(os.path.dirname(sys.executable), 'portable.txt'))
    return False


def get_app_dir() -> str:
    """Get the application directory for user-facing paths (logs, sessions, etc.).

    Portable mode: all data stays next to the .exe (USB-stick friendly).
    macOS frozen: /Applications is read-only → ~/Library/Application Support/EZ-CorridorKey.
    Windows frozen: returns the .exe directory (user-writable).
    Use get_base_dir() for bundled resources (checkpoints, QSS, fonts).
    """
    if getattr(sys, 'frozen', False):
        if is_portable():
            return os.path.dirname(sys.executable)
        if sys.platform == 'darwin':
            support = os.path.join(
                os.path.expanduser('~'), 'Library', 'Application Support', 'EZ-CorridorKey'
            )
            os.makedirs(support, exist_ok=True)
            return support
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


# Ensure project root is on path
sys.path.insert(0, get_base_dir())


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with dual output: console + per-session file.

    Console respects --log-level flag. File always captures DEBUG.
    All timestamps use the system's local time.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"

    # Console handler — respects --log-level
    # Force UTF-8 so non-ASCII paths (e.g. CJK characters) don't crash
    # on Windows cp1252 consoles.
    import io
    utf8_stream = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True,
    ) if hasattr(sys.stderr, "buffer") else sys.stderr
    console = logging.StreamHandler(utf8_stream)
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    # File handler — session-named log, always DEBUG
    log_dir = os.path.join(get_app_dir(), "logs", "backend")
    os.makedirs(log_dir, exist_ok=True)

    session_ts = datetime.now().strftime("%y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{session_ts}_corridorkey.log")

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    # Root logger — let handlers filter
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)


def run_gui() -> int:
    """Launch the PySide6 desktop application."""
    ensure_standard_streams()
    # The GUI has its own progress UI; external library progress bars just create
    # stderr/console issues in pythonw and frozen launches.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from ui.app import create_app
    from ui.main_window import MainWindow
    from ui.recent_sessions import RecentSessionsStore
    from backend import CorridorKeyService

    app = create_app()

    # First-launch / version-upgrade setup: show the Download Manager when a
    # required model is missing OR the installed version changed since the
    # last successful launch. The user can dismiss the version-upgrade case
    # and continue; missing-required still blocks launch.
    from ui.widgets.setup_wizard import (
        SetupWizard,
        needs_setup,
        has_required_models,
        mark_setup_seen,
    )
    if needs_setup():
        wizard = SetupWizard()
        wizard.exec()
        if not has_required_models():
            return 0
    mark_setup_seen()

    # Frozen builds: point projects at the user-chosen install directory
    # (same location as model checkpoints) instead of the exe directory.
    if getattr(sys, 'frozen', False):
        from backend.project import get_data_dir, set_app_dir
        set_app_dir(get_data_dir())

    service = CorridorKeyService()
    store = RecentSessionsStore()
    store.prune_missing()
    window = MainWindow(service, store)
    window.show()
    rc = app.exec()
    # Force-exit: torch.compile spawns Triton background threads that prevent
    # clean shutdown, leaving a zombie process holding VRAM.
    os._exit(rc)


def run_cli() -> int:
    """Run the original CLI wizard from clip_manager.py as a subprocess.

    Forwards all extra CLI arguments (e.g. --action wizard --win_path ...)
    directly to the upstream script, preserving 100% original behaviour.
    """
    script = os.path.join(get_base_dir(), "clip_manager.py")
    if not os.path.isfile(script):
        logging.error(
            "CLI mode requires clip_manager.py in the project root. "
            "Use --gui (default) for the graphical interface."
        )
        return 1

    import subprocess

    # Forward everything after --cli to clip_manager.py.
    # sys.argv looks like: ['main.py', '--cli', '--action', 'wizard', ...]
    # We strip our own flags (--cli, --gui, --log-level <val>) and pass the rest.
    forwarded: list[str] = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--cli", "--gui"):
            continue
        if arg == "--log-level":
            skip_next = True  # skip the value that follows
            continue
        forwarded.append(arg)

    cmd = [sys.executable, script] + forwarded
    logging.info("CLI passthrough: %s", " ".join(cmd))
    return subprocess.call(cmd)


def _checkpoint_info(path: str | os.PathLike[str]) -> dict[str, object]:
    p = Path(path)
    return {
        "path": str(p),
        "size": p.stat().st_size,
    }


def _verify_torch_checkpoint_load(path: str | os.PathLike[str]) -> dict[str, object]:
    import gc
    import torch

    p = Path(path)
    checkpoint = torch.load(str(p), map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    key_count = len(state_dict) if hasattr(state_dict, "keys") else None
    info = _checkpoint_info(p)
    info["state_dict_keys"] = key_count
    del state_dict
    del checkpoint
    gc.collect()
    return info


def _verify_safetensors_checkpoint_load(path: str | os.PathLike[str]) -> dict[str, object]:
    from safetensors import safe_open

    p = Path(path)
    with safe_open(str(p), framework="np") as handle:
        keys = list(handle.keys())
    info = _checkpoint_info(p)
    info["tensor_keys"] = len(keys)
    info["first_keys"] = keys[:5]
    return info


def _verify_birefnet_model_load(model_dir: str | os.PathLike[str]) -> dict[str, object]:
    import gc
    import torch
    from transformers import AutoConfig, AutoModelForImageSegmentation

    model_path = Path(model_dir)
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForImageSegmentation.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    model.to("cpu")
    model.eval()
    info = {
        "path": str(model_path),
        "model_type": getattr(config, "model_type", None),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    del model
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    return info


def _verify_matanyone2_model_load(ckpt_path: str | os.PathLike[str]) -> dict[str, object]:
    import gc
    import torch
    from modules.MatAnyone2Module.wrapper import MatAnyone2Processor

    p = Path(ckpt_path)
    processor = MatAnyone2Processor(device="cpu", ckpt_path=str(p), n_warmup=0)
    processor._ensure_loaded()
    model = processor._model
    info = _checkpoint_info(p)
    info["parameter_count"] = (
        sum(param.numel() for param in model.parameters())
        if model is not None
        else None
    )
    processor.clear()
    del processor
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    return info


def run_model_path_check(
    data_dir: str | None = None,
    *,
    verify_model_loads: bool = False,
    verify_optional_model_downloads: bool = False,
) -> int:
    """Hidden frozen-app smoke check for DMG install path testing."""
    from PySide6.QtCore import QCoreApplication

    QCoreApplication.setOrganizationName("EZSCAPE")
    QCoreApplication.setApplicationName("EZ-CorridorKey")

    settings_patch = None
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        from unittest.mock import patch

        settings_patch = patch("PySide6.QtCore.QSettings.value", return_value=data_dir)
        settings_patch.start()

    try:
        from backend.project import get_data_dir, set_app_dir

        resolved_data_dir = os.path.abspath(get_data_dir())
        set_app_dir(resolved_data_dir)

        import CorridorKeyModule.backend as ck_backend

        result: dict[str, object] = {
            "frozen": bool(getattr(sys, "frozen", False)),
            "executable": sys.executable,
            "base_dir": get_base_dir(),
            "data_dir": resolved_data_dir,
            "verify_model_loads": verify_model_loads,
            "verify_optional_model_downloads": verify_optional_model_downloads,
            "checkpoint_search_dirs": [
                os.path.abspath(path)
                for path in ck_backend._checkpoint_search_dirs()
            ],
        }
        errors: list[str] = []
        model_loads: dict[str, object] = {}

        if data_dir and os.path.abspath(data_dir) != resolved_data_dir:
            errors.append(
                "Data dir mismatch: expected "
                f"{os.path.abspath(data_dir)}, got {resolved_data_dir}"
            )

        try:
            torch_green = ck_backend._discover_checkpoint(
                ck_backend.TORCH_EXT, "green"
            )
            result["torch_green"] = str(torch_green)
            if verify_model_loads:
                model_loads["torch_green"] = _verify_torch_checkpoint_load(torch_green)
        except Exception as exc:
            errors.append(f"Torch green checkpoint missing: {exc}")

        try:
            torch_blue = ck_backend._discover_checkpoint(
                ck_backend.TORCH_EXT, "blue"
            )
            result["torch_blue"] = str(torch_blue)
            if verify_model_loads:
                model_loads["torch_blue"] = _verify_torch_checkpoint_load(torch_blue)
        except Exception as exc:
            errors.append(f"Torch blue checkpoint missing: {exc}")

        if sys.platform == "darwin" and platform.machine() == "arm64":
            try:
                mlx_green = ck_backend._discover_checkpoint(
                    ck_backend.MLX_EXT, "green"
                )
                result["mlx_green"] = str(mlx_green)
                if verify_model_loads:
                    model_loads["mlx_green"] = _verify_safetensors_checkpoint_load(mlx_green)
            except Exception as exc:
                errors.append(f"MLX green checkpoint missing: {exc}")

            try:
                mlx_blue = ck_backend._discover_checkpoint(
                    ck_backend.MLX_EXT, "blue"
                )
                result["mlx_blue"] = str(mlx_blue)
                if verify_model_loads:
                    model_loads["mlx_blue"] = _verify_safetensors_checkpoint_load(mlx_blue)
            except Exception as exc:
                errors.append(f"MLX blue checkpoint missing: {exc}")

        from modules.BiRefNetModule import wrapper as birefnet_wrapper
        from modules.MatAnyone2Module import wrapper as matanyone2_wrapper

        _existing_birefnet, birefnet_download_dir = (
            birefnet_wrapper._resolve_model_dir("BiRefNet-matting")
        )
        matanyone2_candidates = matanyone2_wrapper._candidate_checkpoint_dirs()
        result["birefnet_download_dir"] = birefnet_download_dir
        result["matanyone2_download_dir"] = matanyone2_candidates[0]

        expected_birefnet = os.path.join(
            resolved_data_dir,
            "modules",
            "BiRefNetModule",
            "checkpoints",
            "BiRefNet-matting",
        )
        expected_matanyone2 = os.path.join(
            resolved_data_dir,
            "modules",
            "MatAnyone2Module",
            "checkpoints",
        )
        if os.path.abspath(birefnet_download_dir) != os.path.abspath(expected_birefnet):
            errors.append(f"BiRefNet download target mismatch: {birefnet_download_dir}")
        if os.path.abspath(matanyone2_candidates[0]) != os.path.abspath(expected_matanyone2):
            errors.append(f"MatAnyone2 download target mismatch: {matanyone2_candidates[0]}")

        setup_path = Path(get_base_dir()) / "scripts" / "setup_models.py"
        result["setup_models_path"] = str(setup_path)
        if not setup_path.is_file():
            errors.append(f"setup_models.py missing: {setup_path}")
        else:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "setup_models_verify", setup_path
            )
            if not spec or not spec.loader:
                errors.append(f"Cannot load setup_models.py: {setup_path}")
            else:
                setup_models = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(setup_models)
                result["setup_models_project_root"] = str(setup_models.PROJECT_ROOT)
                if os.path.abspath(str(setup_models.PROJECT_ROOT)) != resolved_data_dir:
                    errors.append(
                        "setup_models PROJECT_ROOT mismatch: "
                        f"{setup_models.PROJECT_ROOT}"
                    )
                setup_targets = {
                    "corridorkey": setup_models.MODELS["corridorkey"]["local_dir"],
                    "corridorkey_blue": setup_models.MODELS["corridorkey-blue"]["local_dir"],
                    "mlx": setup_models.MLX_CHECKPOINTS[0]["local_dir"],
                    "matanyone2": setup_models.MATANYONE2_CHECKPOINT["local_dir"],
                    "birefnet": setup_models.BIREFNET_DEFAULT_CHECKPOINT["local_dir"],
                    "gvm": setup_models.MODELS["gvm"]["local_dir"],
                    "videomama": setup_models.MODELS["videomama"]["local_dir"],
                }
                result["setup_model_dirs"] = {
                    key: str(value) for key, value in setup_targets.items()
                }
                data_prefix = resolved_data_dir + os.sep
                for key, target in setup_targets.items():
                    target_abs = os.path.abspath(str(target))
                    if (
                        target_abs != resolved_data_dir
                        and not target_abs.startswith(data_prefix)
                    ):
                        errors.append(f"{key} target outside data dir: {target_abs}")

                if verify_optional_model_downloads:
                    if not setup_models.download_matanyone2():
                        errors.append("MatAnyone2 download failed")
                    if not setup_models.download_birefnet():
                        errors.append("BiRefNet download failed")

                if verify_model_loads:
                    matanyone2_ckpt = Path(
                        resolved_data_dir,
                        "modules",
                        "MatAnyone2Module",
                        "checkpoints",
                        "matanyone2.pth",
                    )
                    if matanyone2_ckpt.is_file():
                        try:
                            model_loads["matanyone2"] = _verify_matanyone2_model_load(
                                matanyone2_ckpt
                            )
                        except Exception as exc:
                            errors.append(f"MatAnyone2 model load failed: {exc}")
                    else:
                        model_loads["matanyone2"] = {
                            "status": "not_installed",
                            "expected_path": str(matanyone2_ckpt),
                        }

                    existing_birefnet, _download_dir = (
                        birefnet_wrapper._resolve_model_dir("BiRefNet-matting")
                    )
                    if existing_birefnet:
                        try:
                            model_loads["birefnet"] = _verify_birefnet_model_load(
                                existing_birefnet
                            )
                        except Exception as exc:
                            errors.append(f"BiRefNet model load failed: {exc}")
                    else:
                        model_loads["birefnet"] = {
                            "status": "not_installed",
                            "expected_path": expected_birefnet,
                        }

        if model_loads:
            result["model_loads"] = model_loads

        ok = not errors
        result["ok"] = ok
        if errors:
            result["errors"] = errors
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if ok else 1
    finally:
        if settings_patch is not None:
            settings_patch.stop()


def main() -> int:
    ensure_standard_streams()
    parser = argparse.ArgumentParser(
        description="CorridorKey — AI Green Screen Keyer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run the original CLI wizard instead of the GUI",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="Launch the GUI (default)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--opt-mode",
        default=None,
        choices=["auto", "speed", "lowvram"],
        help="GPU optimization mode: auto (detect VRAM), speed (torch.compile), "
             "lowvram (tiled refiner for 8GB cards). Overrides CORRIDORKEY_OPT_MODE env var.",
    )
    parser.add_argument(
        "--verify-model-paths",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verify-model-data-dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verify-model-loads",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verify-optional-model-downloads",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    # parse_known_args so CLI-mode flags (--action, --win_path, etc.)
    # pass through to clip_manager.py without error.
    args, _unknown = parser.parse_known_args()

    # CLI flag takes priority over env var for optimization mode
    if args.opt_mode:
        os.environ['CORRIDORKEY_OPT_MODE'] = args.opt_mode

    if args.verify_model_paths:
        return run_model_path_check(
            args.verify_model_data_dir,
            verify_model_loads=args.verify_model_loads,
            verify_optional_model_downloads=args.verify_optional_model_downloads,
        )

    setup_logging(args.log_level)

    # Configure backend with the application directory
    from backend.project import set_app_dir
    set_app_dir(get_app_dir())

    if args.cli:
        return run_cli()
    return run_gui()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())

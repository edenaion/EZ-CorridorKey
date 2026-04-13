"""Download model weights for EZ-CorridorKey.

Uses huggingface_hub for robust downloading with resume support,
progress bars, and idempotent behavior (skips existing files).

Usage:
    python scripts/setup_models.py --corridorkey       # Required (383MB)
    python scripts/setup_models.py --corridorkey-mlx   # Apple Silicon MLX weights (380MB)
    python scripts/setup_models.py --sam2             # SAM2 Base+ (324MB)
    python scripts/setup_models.py --sam2 large       # SAM2 Large (898MB)
    python scripts/setup_models.py --gvm               # Optional (~6GB)
    python scripts/setup_models.py --videomama          # Optional (~37GB)
    python scripts/setup_models.py --matanyone2         # Optional (~141MB)
    python scripts/setup_models.py --all                # Everything
    python scripts/setup_models.py --check              # Status report
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import platform
import shutil
import sys
import urllib.request
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parent.parent


def _data_root() -> Path:
    """Writable root for model downloads.

    In dev mode: project root (checkpoints live in CorridorKeyModule/checkpoints/).
    In frozen PyInstaller builds: delegates to backend.project.get_data_dir().
    Falls back to standalone logic when backend is not importable (script run directly).
    """
    if not getattr(sys, "frozen", False):
        return _SCRIPT_ROOT
    try:
        from backend.project import get_data_dir

        return Path(get_data_dir())
    except ImportError:
        pass
    # Standalone fallback (script run outside frozen app)
    try:
        from PySide6.QtCore import QSettings

        saved = QSettings().value("app/install_path", "", type=str)
        if saved and os.path.isdir(saved):
            return Path(saved)
    except Exception:
        pass
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "EZ-CorridorKey"
    elif sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home())) / "EZ-CorridorKey"
    else:
        return Path.home() / ".local" / "share" / "EZ-CorridorKey"


PROJECT_ROOT = _data_root()
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

# MLX checkpoint served from GitHub Releases (not HuggingFace)
MLX_CHECKPOINT = {
    "url": "https://github.com/nikopueringer/corridorkey-mlx/releases/download/v1.0.0/corridorkey_mlx.safetensors",
    "sha256_url": "https://github.com/nikopueringer/corridorkey-mlx/releases/download/v1.0.0/corridorkey_mlx.safetensors.sha256",
    "filename": "corridorkey_mlx.safetensors",
    "local_dir": PROJECT_ROOT / "CorridorKeyModule" / "checkpoints",
    "size_human": "380 MB",
    "size_bytes": 398_849_072,
}

MODELS = {
    "corridorkey": {
        "repo_id": "nikopueringer/CorridorKey_v1.0",
        "filename": "CorridorKey_v1.0.pth",
        "local_dir": PROJECT_ROOT / "CorridorKeyModule" / "checkpoints",
        "check_glob": "*.pth",
        "size_human": "383 MB",
        "size_bytes": 400_000_000,
        "required": True,
    },
    "gvm": {
        "repo_id": "geyongtao/gvm",
        "local_dir": PROJECT_ROOT / "gvm_core" / "weights",
        "check_file": "unet/diffusion_pytorch_model.safetensors",
        "size_human": "~6 GB",
        "size_bytes": 6_500_000_000,
        "required": False,
    },
    "videomama": {
        "repo_id": "SammyLim/VideoMaMa",
        "local_dir": PROJECT_ROOT / "VideoMaMaInferenceModule" / "checkpoints",
        "check_file": "VideoMaMa/diffusion_pytorch_model.safetensors",
        "size_human": "~37 GB",
        "size_bytes": 40_000_000_000,
        "required": False,
        "base_model": {
            "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt",
            "subfolder": "stable-video-diffusion-img2vid-xt",
            "check_file": "stable-video-diffusion-img2vid-xt/model_index.json",
        },
    },
}

MATANYONE2_CHECKPOINT = {
    "url": "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth",
    "filename": "matanyone2.pth",
    "local_dir": PROJECT_ROOT / "modules" / "MatAnyone2Module" / "checkpoints",
    "size_human": "141 MB",
    "size_bytes": 141_429_115,
}

SAM2_MODELS = {
    "small": {
        "repo_id": "facebook/sam2.1-hiera-small",
        "filename": "sam2.1_hiera_small.pt",
        "size_human": "184 MB",
        "size_bytes": 184_000_000,
        "default": False,
    },
    "base-plus": {
        "repo_id": "facebook/sam2.1-hiera-base-plus",
        "filename": "sam2.1_hiera_base_plus.pt",
        "size_human": "324 MB",
        "size_bytes": 324_000_000,
        "default": True,
    },
    "large": {
        "repo_id": "facebook/sam2.1-hiera-large",
        "filename": "sam2.1_hiera_large.pt",
        "size_human": "898 MB",
        "size_bytes": 898_000_000,
        "default": False,
    },
}


def tracker_dependency_installed() -> bool:
    """Check whether the optional SAM2 Python package is installed."""
    try:
        import sam2  # noqa: F401
    except Exception:
        return False
    return True


def is_installed(name: str) -> bool:
    """Check if a model's weights are already downloaded."""
    cfg = MODELS[name]
    local_dir = cfg["local_dir"]
    if "check_glob" in cfg:
        if len(glob.glob(str(local_dir / cfg["check_glob"]))) == 0:
            return False
    if "check_file" in cfg:
        if not (local_dir / cfg["check_file"]).is_file():
            return False
    # Also check base_model dependency if defined
    if "base_model" in cfg:
        base_check = cfg["base_model"].get("check_file")
        if base_check and not (local_dir / base_check).is_file():
            return False
    if "check_glob" not in cfg and "check_file" not in cfg:
        return False
    return True


def is_sam2_installed(name: str) -> bool:
    """Check whether a SAM2 checkpoint is present in the HF cache."""
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.file_download import _CACHED_NO_EXIST

    cfg = SAM2_MODELS[name]
    cached = try_to_load_from_cache(
        repo_id=cfg["repo_id"],
        filename=cfg["filename"],
        cache_dir=HF_CACHE_DIR,
    )
    return isinstance(cached, str) and cached != _CACHED_NO_EXIST and os.path.isfile(cached)


def check_disk_space(needed_bytes: int, path: Path) -> bool:
    """Check if there's enough disk space for a download."""
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    # Require 10% headroom beyond the download size
    return usage.free > needed_bytes * 1.1


def download_corridorkey() -> bool:
    """Download the CorridorKey checkpoint (single file)."""
    from huggingface_hub import hf_hub_download

    cfg = MODELS["corridorkey"]
    local_dir = cfg["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading CorridorKey checkpoint ({cfg['size_human']})...")
    try:
        downloaded = hf_hub_download(
            repo_id=cfg["repo_id"],
            filename=cfg["filename"],
            local_dir=str(local_dir),
        )
        # huggingface_hub downloads to local_dir/filename
        # The backend globs for *.pth, so the exact name doesn't matter
        print(f"  Saved to: {downloaded}")
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        print(f"  Manual download: https://huggingface.co/{cfg['repo_id']}")
        return False


def is_mlx_installed() -> bool:
    """Check if the MLX .safetensors checkpoint is already present."""
    return (MLX_CHECKPOINT["local_dir"] / MLX_CHECKPOINT["filename"]).is_file()


def download_corridorkey_mlx() -> bool:
    """Download the MLX .safetensors checkpoint from GitHub Releases.

    Verifies SHA256 after download. Does not require huggingface_hub.
    """
    cfg = MLX_CHECKPOINT
    local_dir = cfg["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / cfg["filename"]

    if dest.is_file():
        print("  [OK] MLX checkpoint already installed")
        return True

    if not check_disk_space(cfg["size_bytes"], local_dir):
        usage = shutil.disk_usage(local_dir)
        free_gb = usage.free / (1024**3)
        print(f"  [ERROR] Not enough disk space for MLX checkpoint ({cfg['size_human']})")
        print(f"  Available: {free_gb:.1f} GB")
        return False

    print(f"  Downloading MLX checkpoint ({cfg['size_human']})...")
    tmp_dest = dest.with_suffix(".safetensors.tmp")
    try:

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)

        urllib.request.urlretrieve(cfg["url"], str(tmp_dest), reporthook=_progress)
        print()  # newline after progress

        # Verify SHA256
        try:
            sha256_resp = urllib.request.urlopen(cfg["sha256_url"])
            expected_hash = sha256_resp.read().decode().strip().split()[0]
            actual_hash = hashlib.sha256(tmp_dest.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                print("  [ERROR] SHA256 mismatch!")
                print(f"  Expected: {expected_hash}")
                print(f"  Got:      {actual_hash}")
                tmp_dest.unlink(missing_ok=True)
                return False
            print("  [OK] SHA256 verified")
        except Exception as e:
            print(f"  [WARN] Could not verify SHA256: {e}")
            print("  Proceeding anyway (file downloaded successfully)")

        tmp_dest.rename(dest)
        print(f"  Saved to: {dest}")
        return True
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        print(f"  Manual download: {cfg['url']}")
        print(f"  Place in: {local_dir}/")
        tmp_dest.unlink(missing_ok=True)
        return False


def download_sam2(name: str) -> bool:
    """Download a SAM2 checkpoint into the shared Hugging Face cache."""
    from huggingface_hub import hf_hub_download

    cfg = SAM2_MODELS[name]
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading SAM2 {name} checkpoint ({cfg['size_human']})...")
    try:
        downloaded = hf_hub_download(
            repo_id=cfg["repo_id"],
            filename=cfg["filename"],
            cache_dir=HF_CACHE_DIR,
        )
        print(f"  Saved to cache: {downloaded}")
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        print(f"  Manual download: https://huggingface.co/{cfg['repo_id']}")
        return False


def download_repo(name: str) -> bool:
    """Download a full HuggingFace repo (GVM or VideoMaMa)."""
    from huggingface_hub import snapshot_download

    cfg = MODELS[name]
    local_dir = cfg["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download base model dependency first (e.g. SVD for VideoMaMa)
    if "base_model" in cfg:
        base = cfg["base_model"]
        base_dir = local_dir / base["subfolder"]
        base_check = base.get("check_file")
        if base_check and (local_dir / base_check).is_file():
            print("  [OK] Base model already downloaded")
        else:
            print(f"  Downloading base model ({base['repo_id']})...")
            print("  This may take a while. Downloads resume if interrupted.")
            try:
                snapshot_download(
                    repo_id=base["repo_id"],
                    local_dir=str(base_dir),
                )
                print(f"  [OK] Base model saved to: {base_dir}")
            except Exception as e:
                print(f"  [ERROR] Base model download failed: {e}")
                print(f"  Manual download: https://huggingface.co/{base['repo_id']}")
                return False

    print(f"  Downloading {name} weights ({cfg['size_human']})...")
    print("  This may take a while. Downloads resume if interrupted.")
    try:
        snapshot_download(
            repo_id=cfg["repo_id"],
            local_dir=str(local_dir),
        )
        # SammyLim/VideoMaMa repo has unet/ but code expects VideoMaMa/
        if name == "videomama":
            unet_dir = local_dir / "unet"
            videomama_dir = local_dir / "VideoMaMa"
            if unet_dir.is_dir() and not videomama_dir.is_dir():
                unet_dir.rename(videomama_dir)
                print("  Renamed unet/ -> VideoMaMa/")
        print(f"  Saved to: {local_dir}")
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        print(f"  Manual download: https://huggingface.co/{cfg['repo_id']}")
        return False


def download_model(name: str) -> bool:
    """Download a model's weights, skipping if already present."""
    cfg = MODELS[name]

    if is_installed(name):
        print(f"  [OK] {name} weights already installed")
        return True

    # Disk space check
    if not check_disk_space(cfg["size_bytes"], cfg["local_dir"]):
        usage = shutil.disk_usage(cfg["local_dir"])
        free_gb = usage.free / (1024**3)
        print(f"  [ERROR] Not enough disk space for {name} ({cfg['size_human']})")
        print(f"  Available: {free_gb:.1f} GB")
        return False

    if name == "corridorkey":
        return download_corridorkey()
    else:
        return download_repo(name)


def download_sam2_model(name: str) -> bool:
    """Download one SAM2 checkpoint, skipping if already cached."""
    cfg = SAM2_MODELS[name]

    if is_sam2_installed(name):
        print(f"  [OK] sam2-{name} checkpoint already cached")
        return True

    if not check_disk_space(cfg["size_bytes"], HF_CACHE_DIR):
        usage = shutil.disk_usage(HF_CACHE_DIR)
        free_gb = usage.free / (1024**3)
        print(f"  [ERROR] Not enough disk space for SAM2 {name} ({cfg['size_human']})")
        print(f"  Available: {free_gb:.1f} GB")
        return False

    return download_sam2(name)


def is_matanyone2_installed() -> bool:
    """Check if the MatAnyone2 checkpoint is already present."""
    return (MATANYONE2_CHECKPOINT["local_dir"] / MATANYONE2_CHECKPOINT["filename"]).is_file()


def download_matanyone2() -> bool:
    """Download the MatAnyone2 checkpoint from GitHub Releases."""
    cfg = MATANYONE2_CHECKPOINT
    local_dir = cfg["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / cfg["filename"]

    if dest.is_file():
        print("  [OK] MatAnyone2 checkpoint already installed")
        return True

    if not check_disk_space(cfg["size_bytes"], local_dir):
        usage = shutil.disk_usage(local_dir)
        free_gb = usage.free / (1024**3)
        print(f"  [ERROR] Not enough disk space for MatAnyone2 ({cfg['size_human']})")
        print(f"  Available: {free_gb:.1f} GB")
        return False

    print(f"  Downloading MatAnyone2 checkpoint ({cfg['size_human']})...")
    tmp_dest = dest.with_suffix(".pth.tmp")
    try:

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)

        urllib.request.urlretrieve(cfg["url"], str(tmp_dest), reporthook=_progress)
        print()  # newline after progress

        tmp_dest.rename(dest)
        print(f"  Saved to: {dest}")
        return True
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        print(f"  Manual download: {cfg['url']}")
        print(f"  Place in: {local_dir}/")
        tmp_dest.unlink(missing_ok=True)
        return False


def check_all():
    """Print status of all models."""
    print("\nModel Status:")
    print("-" * 50)
    for name, cfg in MODELS.items():
        installed = is_installed(name)
        status = "INSTALLED" if installed else "NOT INSTALLED"
        required = " (required)" if cfg["required"] else " (optional)"
        mark = "[OK]" if installed else "[--]"
        print(f"  {mark} {name:12s} {cfg['size_human']:>8s}  {status}{required}")

        if installed and "check_glob" in cfg:
            files = glob.glob(str(cfg["local_dir"] / cfg["check_glob"]))
            for f in files:
                size_mb = os.path.getsize(f) / (1024**2)
                print(f"       -> {os.path.basename(f)} ({size_mb:.0f} MB)")

    # MatAnyone2 checkpoint
    ma2_installed = is_matanyone2_installed()
    ma2_mark = "[OK]" if ma2_installed else "[--]"
    ma2_status = "INSTALLED" if ma2_installed else "NOT INSTALLED"
    print(
        f"  {ma2_mark} matanyone2   {MATANYONE2_CHECKPOINT['size_human']:>8s}  {ma2_status} (optional)"
    )

    # MLX checkpoint (Apple Silicon only, but show status on all platforms)
    mlx_installed = is_mlx_installed()
    mlx_mark = "[OK]" if mlx_installed else "[--]"
    mlx_status = "INSTALLED" if mlx_installed else "NOT INSTALLED"
    mlx_note = (
        " (Apple Silicon)"
        if sys.platform == "darwin" and platform.machine() == "arm64"
        else " (Apple Silicon only)"
    )
    print(
        f"  {mlx_mark} corridorkey-mlx {MLX_CHECKPOINT['size_human']:>8s}  {mlx_status}{mlx_note}"
    )

    tracker_installed = tracker_dependency_installed()
    tracker_mark = "[OK]" if tracker_installed else "[--]"
    tracker_status = "INSTALLED" if tracker_installed else "NOT INSTALLED"
    print(f"  {tracker_mark} sam2-tracker  python pkg   {tracker_status} (optional)")
    print(f"       -> cache: {HF_CACHE_DIR}")
    for name, cfg in SAM2_MODELS.items():
        installed = is_sam2_installed(name)
        status = "CACHED" if installed else "NOT CACHED"
        mark = "[OK]" if installed else "[--]"
        label = f"sam2-{name}"
        if cfg.get("default"):
            label += " (default)"
        print(f"  {mark} {label:18s} {cfg['size_human']:>8s}  {status}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Download model weights for EZ-CorridorKey")
    parser.add_argument(
        "--corridorkey",
        action="store_true",
        help="Download CorridorKey checkpoint (383MB, required)",
    )
    parser.add_argument(
        "--corridorkey-mlx",
        action="store_true",
        help="Download MLX checkpoint for Apple Silicon (380MB)",
    )
    parser.add_argument(
        "--sam2",
        nargs="?",
        const="base-plus",
        choices=["small", "base-plus", "large", "all"],
        help="Download SAM2 checkpoint(s): small, base-plus, large, or all (default: base-plus)",
    )
    parser.add_argument("--gvm", action="store_true", help="Download GVM weights (~6GB, optional)")
    parser.add_argument(
        "--videomama", action="store_true", help="Download VideoMaMa weights (~37GB, optional)"
    )
    parser.add_argument(
        "--matanyone2", action="store_true", help="Download MatAnyone2 weights (~141MB, optional)"
    )
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--check", action="store_true", help="Check installation status")
    args = parser.parse_args()

    mlx_flag = getattr(args, "corridorkey_mlx", False)

    # Default to --check if no flags
    if not any(
        [
            args.corridorkey,
            mlx_flag,
            args.sam2,
            args.gvm,
            args.videomama,
            args.matanyone2,
            args.all,
            args.check,
        ]
    ):
        args.check = True

    if args.check:
        check_all()
        if not any(
            [
                args.corridorkey,
                mlx_flag,
                args.sam2,
                args.gvm,
                args.videomama,
                args.matanyone2,
                args.all,
            ]
        ):
            return

    targets = []
    sam2_targets: list[str] = []
    download_mlx = False
    download_ma2 = False
    if args.all:
        targets = list(MODELS.keys())
        sam2_targets = list(SAM2_MODELS.keys())
        download_ma2 = True
        # --all on Apple Silicon auto-includes MLX weights
        if sys.platform == "darwin" and platform.machine() == "arm64":
            download_mlx = True
    else:
        if args.corridorkey:
            targets.append("corridorkey")
        if mlx_flag:
            download_mlx = True
        if args.matanyone2:
            download_ma2 = True
        if args.sam2:
            if args.sam2 == "all":
                sam2_targets = list(SAM2_MODELS.keys())
            else:
                sam2_targets = [args.sam2]
        if args.gvm:
            targets.append("gvm")
        if args.videomama:
            targets.append("videomama")

    if not targets and not sam2_targets and not download_mlx and not download_ma2:
        return

    total_targets = (
        len(targets) + len(sam2_targets) + (1 if download_mlx else 0) + (1 if download_ma2 else 0)
    )
    print(f"\nDownloading {total_targets} model(s)...\n")
    results = {}
    for name in targets:
        print(f"[{name}]")
        results[name] = download_model(name)
        print()
    if download_mlx:
        print("[corridorkey-mlx]")
        results["corridorkey-mlx"] = download_corridorkey_mlx()
        print()
    if download_ma2:
        print("[matanyone2]")
        results["matanyone2"] = download_matanyone2()
        print()
    for name in sam2_targets:
        result_key = f"sam2-{name}"
        print(f"[{result_key}]")
        results[result_key] = download_sam2_model(name)
        print()

    # Summary
    print("Summary:")
    for name, ok in results.items():
        print(f"  {'[OK]' if ok else '[FAIL]'} {name}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

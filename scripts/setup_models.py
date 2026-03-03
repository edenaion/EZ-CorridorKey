"""Download model weights for EZ-CorridorKey.

Uses huggingface_hub for robust downloading with resume support,
progress bars, and idempotent behavior (skips existing files).

Usage:
    python scripts/setup_models.py --corridorkey       # Required (383MB)
    python scripts/setup_models.py --gvm               # Optional (~6GB)
    python scripts/setup_models.py --videomama          # Optional (~37GB)
    python scripts/setup_models.py --all                # Everything
    python scripts/setup_models.py --check              # Status report
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
            print(f"  [OK] Base model already downloaded")
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
        unet_dir = local_dir / "unet"
        videomama_dir = local_dir / "VideoMaMa"
        if unet_dir.is_dir() and not videomama_dir.is_dir():
            unet_dir.rename(videomama_dir)
            print(f"  Renamed unet/ -> VideoMaMa/")
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
    print()


def main():
    parser = argparse.ArgumentParser(description="Download model weights for EZ-CorridorKey")
    parser.add_argument("--corridorkey", action="store_true", help="Download CorridorKey checkpoint (383MB, required)")
    parser.add_argument("--gvm", action="store_true", help="Download GVM weights (~6GB, optional)")
    parser.add_argument("--videomama", action="store_true", help="Download VideoMaMa weights (~37GB, optional)")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--check", action="store_true", help="Check installation status")
    args = parser.parse_args()

    # Default to --check if no flags
    if not any([args.corridorkey, args.gvm, args.videomama, args.all, args.check]):
        args.check = True

    if args.check:
        check_all()
        if not any([args.corridorkey, args.gvm, args.videomama, args.all]):
            return

    targets = []
    if args.all:
        targets = list(MODELS.keys())
    else:
        if args.corridorkey:
            targets.append("corridorkey")
        if args.gvm:
            targets.append("gvm")
        if args.videomama:
            targets.append("videomama")

    if not targets:
        return

    print(f"\nDownloading {len(targets)} model(s)...\n")
    results = {}
    for name in targets:
        print(f"[{name}]")
        results[name] = download_model(name)
        print()

    # Summary
    print("Summary:")
    for name, ok in results.items():
        print(f"  {'[OK]' if ok else '[FAIL]'} {name}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

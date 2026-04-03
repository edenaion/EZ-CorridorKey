"""Build a lightweight update zip for frozen EZ-CorridorKey (Windows).

Packages only the app code from dist/CorridorKey/, excluding heavy
runtime dependencies (torch, CUDA, triton, etc.) that don't change
between versions. The result is small enough for GitHub Releases
(~50-100 MB vs 2-3 GB for the full installer).

Usage:
    python scripts/windows/build_update_zip.py

Output:
    dist/CorridorKey-windows-x64.zip

The in-app updater downloads this zip and xcopy's it over the existing
install. Since xcopy only overwrites files present in the zip, the
heavy dependencies stay untouched.
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

# Directories to EXCLUDE from the update zip (heavy, version-stable deps)
EXCLUDE_DIRS = {
    # PyTorch + CUDA (4.2 GB) — only changes on major torch version bumps
    "torch",
    "torch.libs",
    # Triton (123 MB) — tied to torch version
    "triton",
    # NVIDIA CUDA runtime DLLs
    "nvidia",
    # torchvision (tied to torch version)
    "torchvision",
    "torchvision.libs",
    # scipy (large, rarely changes)
    "scipy",
    "scipy.libs",
    # numpy internals (tied to numpy version)
    "numpy.libs",
    # OpenCV (99 MB, rarely changes)
    "cv2",
    # PyAV libs (74 MB, rarely changes)
    "av.libs",
    # PySide6 (114 MB, rarely changes)
    "PySide6",
    "shiboken6",
    # transformers/diffusers (47 MB + deps, rarely changes)
    "transformers",
    "diffusers",
    "tokenizers",
    "hf_xet",
    "huggingface_hub",
    "safetensors",
    # PIL/Pillow (13 MB)
    "PIL",
    # Large ML framework internals
    "sympy",
    "networkx",
    "mpmath",
    # certifi (CA certs, stable)
    "certifi",
    # Pythonwin / win32 (stable)
    "Pythonwin",
    "win32",
    "win32com",
    "win32comext",
    # kornia (10 MB, tied to version)
    "kornia",
    "kornia_rs",
    # timm (9 MB, tied to version)
    "timm",
}

# Directories to exclude by path prefix (deeper than top-level _internal/)
EXCLUDE_PATH_PREFIXES = {
    # BiRefNet model weights (1.27 GB) — downloaded by setup wizard, not app code
    os.path.join("_internal", "modules", "BiRefNetModule", "checkpoints"),
    # MatAnyone2 model code (135 MB) — tied to version, rarely changes
    os.path.join("_internal", "MatAnyone2Module"),
}

# File patterns to exclude
EXCLUDE_PATTERNS = {
    ".pyc",  # bytecode regenerated on launch
    ".pyo",
}


def should_exclude(rel_path: Path) -> bool:
    """Check if a relative path should be excluded from the update zip."""
    parts = rel_path.parts
    rel_str = str(rel_path)

    # Check if any path component matches an excluded directory
    # Files are under _internal/ in the PyInstaller output
    if len(parts) >= 2 and parts[0] == "_internal":
        if parts[1] in EXCLUDE_DIRS:
            return True

    # Check deeper path prefixes
    for prefix in EXCLUDE_PATH_PREFIXES:
        if rel_str.startswith(prefix):
            return True

    # Check file extension
    if rel_path.suffix in EXCLUDE_PATTERNS:
        return True

    return False


def build_update_zip(dist_dir: Path, output_path: Path) -> None:
    """Build the update zip from the PyInstaller dist directory."""
    source = dist_dir / "EZ-CorridorKey"
    if not source.is_dir():
        print(f"ERROR: {source} does not exist. Run PyInstaller first.")
        sys.exit(1)

    included = 0
    excluded = 0
    included_size = 0
    excluded_size = 0

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        for root, dirs, files in os.walk(source):
            for fname in files:
                full_path = Path(root) / fname
                rel_path = full_path.relative_to(source)

                if should_exclude(rel_path):
                    excluded += 1
                    excluded_size += full_path.stat().st_size
                    continue

                # Write into zip under EZ-CorridorKey/ prefix so extraction
                # produces a EZ-CorridorKey/ folder (matches updater expectation)
                arc_name = Path("EZ-CorridorKey") / rel_path
                zf.write(full_path, arc_name)
                included += 1
                included_size += full_path.stat().st_size

    zip_size = output_path.stat().st_size
    print(f"Update zip built: {output_path}")
    print(f"  Included: {included} files ({included_size / 1024 / 1024:.1f} MB uncompressed)")
    print(f"  Excluded: {excluded} files ({excluded_size / 1024 / 1024:.1f} MB)")
    print(f"  Zip size: {zip_size / 1024 / 1024:.1f} MB")


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    dist_dir = project_root / "dist"
    output = dist_dir / "CorridorKey-windows-x64.zip"

    print(f"Building update zip from: {dist_dir / 'CorridorKey'}")
    print(f"Output: {output}")
    print()
    build_update_zip(dist_dir, output)


if __name__ == "__main__":
    main()

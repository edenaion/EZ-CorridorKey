"""Validate the local FFmpeg install without importing the full backend package."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_ffmpeg_tools():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "backend" / "ffmpeg_tools.py"
    spec = importlib.util.spec_from_file_location("corridorkey_ffmpeg_tools", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate FFmpeg/FFprobe for CorridorKey.")
    parser.add_argument("--quiet", action="store_true", help="Suppress human-readable output.")
    parser.add_argument(
        "--no-require-probe",
        action="store_true",
        help="Validate ffmpeg only (default requires both ffmpeg and ffprobe).",
    )
    args = parser.parse_args()

    ffmpeg_tools = _load_ffmpeg_tools()
    result = ffmpeg_tools.validate_ffmpeg_install(require_probe=not args.no_require_probe)

    if not args.quiet:
        print(result.message)

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

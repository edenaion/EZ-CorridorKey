"""Validate the local FFmpeg install."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.ffmpeg_tools import validate_ffmpeg_install


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate FFmpeg/FFprobe for CorridorKey.")
    parser.add_argument("--quiet", action="store_true", help="Suppress human-readable output.")
    parser.add_argument(
        "--no-require-probe",
        action="store_true",
        help="Validate ffmpeg only (default requires both ffmpeg and ffprobe).",
    )
    args = parser.parse_args()

    result = validate_ffmpeg_install(require_probe=not args.no_require_probe)

    if not args.quiet:
        print(result.message)

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

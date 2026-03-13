"""Open a pre-filled GitHub issue for installer failures.

This mirrors the in-app "Report Issue" experience closely enough for installer
errors, but keeps the implementation stdlib-only so it can run before the full
project environment is healthy.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import webbrowser
from pathlib import Path
from urllib.parse import quote

_GITHUB_ISSUES_URL = "https://github.com/edenaion/EZ-CorridorKey/issues/new"
_MAX_URL_LENGTH = 7500


def _read_json(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _copy_to_clipboard(text: str) -> bool:
    if sys.platform == "win32":
        try:
            subprocess.run(["clip"], input=text, text=True, check=True)
            return True
        except Exception:
            return False
    if sys.platform == "darwin":
        try:
            subprocess.run(["pbcopy"], input=text, text=True, check=True)
            return True
        except Exception:
            return False

    for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
        try:
            subprocess.run(cmd, input=text, text=True, check=True)
            return True
        except Exception:
            continue
    return False


def _load_version() -> str:
    try:
        import tomllib

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with pyproject.open("rb") as handle:
            return tomllib.load(handle)["project"]["version"]
    except Exception:
        return "unknown"


def _build_body(data: dict[str, object], failure_stage: str, log_path: str) -> str:
    message = str(data.get("message", "Installer verification failed."))
    lines = [
        "## Installer Failure",
        message,
        "",
        "## What Happened",
        "The one-click installer failed while validating the installed runtime.",
        "",
        "## Auto-Collected Diagnostics",
        f"- **CorridorKey Version:** {_load_version()}",
        f"- **Failure Stage:** {failure_stage}",
        f"- **OS:** {platform.platform()}",
        f"- **Python:** {data.get('python_version', platform.python_version())}",
        f"- **Torch:** {data.get('torch_version', 'unknown')}",
        f"- **TorchVision:** {data.get('torchvision_version', 'unknown')}",
        f"- **Torch CUDA Version:** {data.get('torch_cuda_version', 'unknown')}",
        f"- **CUDA Available:** {data.get('cuda_available', False)}",
        f"- **CUDA Device Count:** {data.get('cuda_device_count', 0)}",
        f"- **MPS Available:** {data.get('mps_available', False)}",
        f"- **nvidia-smi Path:** {data.get('nvidia_smi_path', '') or '(not found)'}",
        f"- **nvidia-smi Summary:** {data.get('nvidia_smi_summary', '') or '(none)'}",
        "",
        "## Raw Diagnostic JSON",
        "```json",
        json.dumps(data, indent=2, sort_keys=True),
        "```",
        "",
        "## Local Files",
        f"- Installer diagnostic JSON: `{log_path}`",
    ]
    return "\n".join(lines)


def _build_url(title: str, body: str) -> str:
    url = f"{_GITHUB_ISSUES_URL}?title={quote(title)}&body={quote(body)}"
    if len(url) <= _MAX_URL_LENGTH:
        return url
    short = body.split("## Raw Diagnostic JSON")[0].rstrip()
    short += "\n\n_Full report was copied to the clipboard or written locally because the URL would be too long._"
    return f"{_GITHUB_ISSUES_URL}?title={quote(title)}&body={quote(short)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Open a pre-filled GitHub issue for an installer failure")
    parser.add_argument("--json", required=True, help="Path to installer diagnostic JSON")
    parser.add_argument("--stage", default="installer-runtime-verification", help="Short failure stage label")
    parser.add_argument("--title", default="Installer failed to validate GPU/PyTorch runtime", help="Issue title")
    parser.add_argument(
        "--body-out",
        default="logs/diagnostics/install-support-report.md",
        help="Where to save the markdown report",
    )
    args = parser.parse_args()

    data = _read_json(args.json)
    body = _build_body(data, args.stage, args.json)
    body_out = Path(args.body_out)
    body_out.parent.mkdir(parents=True, exist_ok=True)
    body_out.write_text(body, encoding="utf-8")

    copied = _copy_to_clipboard(body)
    url = _build_url(args.title, body)
    webbrowser.open(url)

    if copied:
        print(f"Installer support report copied to clipboard and saved to {body_out}.")
    else:
        print(f"Installer support report saved to {body_out}.")
    print("GitHub issue page opened in your browser. Log in and submit the issue if prompted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

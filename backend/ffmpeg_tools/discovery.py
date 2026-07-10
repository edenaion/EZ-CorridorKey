"""FFmpeg binary discovery, validation, and repair.

Locates ffmpeg/ffprobe binaries, validates version requirements,
and provides repair/install functionality per platform.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_METADATA_FILENAME = ".video_metadata.json"
_MIN_FFMPEG_MAJOR = 7
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOCAL_FFMPEG_BIN = os.path.join(_REPO_ROOT, "tools", "ffmpeg", "bin")

# Repair pulls a Windows build from BtbN/FFmpeg-Builds. We deliberately avoid the
# "master" nightly: nightlies remove deprecated options (the -vsync removal broke
# imports in issue #175).
#
# PINNED to an exact dated autobuild, never a rolling tag. The rolling
# "latest" stable asset burned us in issue #184: FFmpeg release/8.1 picked up
# a vf_scale_cuda/hwcontext_cuda rework on 2026-06-29 (post n8.1.2 tag) with a
# race that corrupts random frames when decoding with -hwaccel cuda (invalid
# zlib chunks inside ZIP16 EXR output). Every BtbN rolling build from
# 2026-06-30 onward ships it. The dated autobuild release below is immutable
# and contains the tag-exact n8.1.2 build, verified clean against the issue
# #184 reproduction clip.
#
# When bumping this pin: extract a known clip with -hwaccel cuda and verify
# every EXR frame reads back (see issue #184), then update both constants AND
# _KNOWN_BAD_BUILD_RE below.
_BTBN_PINNED_ASSET = "ffmpeg-n8.1.2-win64-gpl-8.1.zip"
_BTBN_PINNED_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/"
    "autobuild-2026-06-27-13-21/" + _BTBN_PINNED_ASSET
)

# Installed builds known to corrupt EXR frames (issue #184): any post-tag
# n8.1.2 rolling build (suffix "-N-g<hash>") carries the 2026-06-29 CUDA
# commits. Tag-exact n8.1.2 builds do not match. Deliberately conservative:
# even if upstream later fixes the race on release/8.1, a post-tag rolling
# build remains unverifiable by us — users get pointed at Repair, which
# installs the pinned verified build. Revisit alongside the pin above.
_KNOWN_BAD_BUILD_RE = re.compile(r"\bn8\.1\.2-\d+-g[0-9a-f]+", re.IGNORECASE)


def _resolve_windows_ffmpeg_asset(
    ssl_ctx: ssl.SSLContext | None,
) -> tuple[str, str]:
    """Return (asset_name, download_url) for the pinned verified BtbN build.

    Always the immutable dated autobuild pinned above — never a rolling
    "latest" tag, whose contents change under us (issue #184 frame
    corruption). The ssl_ctx parameter is kept for signature compatibility
    with the download step.
    """
    logger.info("Repair FFmpeg: using pinned verified build %s", _BTBN_PINNED_ASSET)
    return _BTBN_PINNED_ASSET, _BTBN_PINNED_URL

# QSettings key for user-configured FFmpeg directory
_QSETTINGS_FFMPEG_DIR = "tools/ffmpeg_custom_dir"


def _get_custom_ffmpeg_dir() -> str | None:
    """Read the user-configured custom FFmpeg directory from QSettings."""
    try:
        from PySide6.QtCore import QSettings
        val = QSettings().value(_QSETTINGS_FFMPEG_DIR, "", type=str)
        return val if val and os.path.isdir(val) else None
    except Exception:
        return None


def set_custom_ffmpeg_dir(path: str) -> None:
    """Persist a user-chosen FFmpeg directory in QSettings."""
    from PySide6.QtCore import QSettings
    QSettings().setValue(_QSETTINGS_FFMPEG_DIR, path)


def _build_windows_search_paths() -> list[str]:
    """Build an extended list of Windows directories to search for FFmpeg."""
    paths = [_LOCAL_FFMPEG_BIN]

    # User-configured custom path (highest priority after bundled)
    custom = _get_custom_ffmpeg_dir()
    if custom:
        paths.append(custom)

    # Standard install locations
    paths.extend([
        r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files (x86)\ffmpeg\bin",
        r"C:\ffmpeg\bin",
    ])

    # Common user-level locations
    local_app = os.environ.get("LOCALAPPDATA", "")
    if local_app:
        paths.append(os.path.join(local_app, "ffmpeg", "bin"))
        paths.append(os.path.join(local_app, "Programs", "ffmpeg", "bin"))

    # Scoop / Chocolatey / winget
    userprofile = os.environ.get("USERPROFILE", "")
    if userprofile:
        paths.append(os.path.join(userprofile, "scoop", "shims"))
        paths.append(os.path.join(userprofile, "scoop", "apps", "ffmpeg", "current", "bin"))

    # Chocolatey
    choco = os.environ.get("ChocolateyInstall", r"C:\ProgramData\chocolatey")
    paths.append(os.path.join(choco, "bin"))

    # Glob for extracted ffmpeg-* folders in Downloads and Desktop
    if userprofile:
        for parent in ("Downloads", "Desktop"):
            parent_dir = os.path.join(userprofile, parent)
            if os.path.isdir(parent_dir):
                try:
                    for name in os.listdir(parent_dir):
                        if name.lower().startswith("ffmpeg") and os.path.isdir(
                            os.path.join(parent_dir, name)
                        ):
                            candidate = os.path.join(parent_dir, name, "bin")
                            if os.path.isdir(candidate):
                                paths.append(candidate)
                            # Some users extract without the inner folder
                            paths.append(os.path.join(parent_dir, name))
                except OSError:
                    pass

    return paths


# Common install locations per platform
_FFMPEG_SEARCH_PATHS_WINDOWS = _build_windows_search_paths()

_FFMPEG_SEARCH_PATHS_UNIX = [
    _LOCAL_FFMPEG_BIN,
    "/opt/homebrew/bin",        # macOS Homebrew (Apple Silicon)
    "/usr/local/bin",           # macOS Homebrew (Intel) / Linux manual install
    "/usr/bin",                 # Linux system package
    "/snap/bin",                # Linux snap
    os.path.expanduser("~/bin"),
]

_FFMPEG_SEARCH_PATHS = (
    _FFMPEG_SEARCH_PATHS_WINDOWS if sys.platform == "win32"
    else _FFMPEG_SEARCH_PATHS_UNIX
)
_FFMPEG_RELEASE_RE = re.compile(
    r"\b(?:ffmpeg|ffprobe)(?:\.exe)?\s+version\s+(?:n)?(?P<major>\d+)(?:\.\d+)*",
    re.IGNORECASE,
)
_FFMPEG_DEV_BUILD_RE = re.compile(
    r"\b(?:ffmpeg|ffprobe)\s+version\s+(?:n-|git-|master-)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FFmpegVersionInfo:
    """Parsed `ffmpeg -version` or `ffprobe -version` first-line summary."""

    first_line: str
    major: int | None
    is_dev_build: bool = False


@dataclass(frozen=True)
class FFmpegValidationResult:
    """Validation result for the current FFmpeg installation."""

    ok: bool
    message: str
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None
    ffmpeg_version: FFmpegVersionInfo | None = None
    ffprobe_version: FFmpegVersionInfo | None = None
    # True when the installed build is on the known frame-corruption list
    # (issue #184) — the UI shows a dedicated "run Repair FFmpeg" warning.
    known_bad: bool = False


def _local_ffmpeg_binary(name: str) -> str | None:
    """Return the bundled repo-local FFmpeg binary if present."""
    ext = ".exe" if sys.platform == "win32" else ""
    candidate = os.path.join(_LOCAL_FFMPEG_BIN, f"{name}{ext}")
    return candidate if os.path.isfile(candidate) else None


def _find_binary(name: str) -> str | None:
    """Locate an FFmpeg-family binary by name (e.g. 'ffmpeg', 'ffprobe')."""
    # 1. Bundled local build
    local = _local_ffmpeg_binary(name)
    if local:
        return local

    # 2. User-configured custom directory
    custom = _get_custom_ffmpeg_dir()
    if custom:
        ext = ".exe" if sys.platform == "win32" else ""
        candidate = os.path.join(custom, f"{name}{ext}")
        if os.path.isfile(candidate):
            return candidate

    # 3. System PATH
    found = shutil.which(name)
    if found:
        return found

    # 4. Platform-specific search paths
    ext = ".exe" if sys.platform == "win32" else ""
    # Rebuild Windows paths each call so a newly-set custom dir is picked up
    search_paths = (
        _build_windows_search_paths() if sys.platform == "win32"
        else _FFMPEG_SEARCH_PATHS_UNIX
    )
    for d in search_paths:
        candidate = os.path.join(d, f"{name}{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None


def find_ffmpeg() -> str | None:
    """Locate ffmpeg binary. Prefer the bundled local build when present."""
    return _find_binary("ffmpeg")


def find_ffprobe() -> str | None:
    """Locate ffprobe binary. Prefer the bundled local build when present."""
    return _find_binary("ffprobe")


def _read_program_version(binary_path: str, program_name: str) -> FFmpegVersionInfo:
    """Run `<program> -version` and parse the first output line."""
    result = subprocess.run(
        [binary_path, "-version"],
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"{program_name} failed to report its version: {stderr[:300]}"
        )

    output = result.stdout or result.stderr or ""
    first_line = next((line.strip() for line in output.splitlines() if line.strip()), "")
    if not first_line:
        raise RuntimeError(f"{program_name} did not report a version string")

    match = _FFMPEG_RELEASE_RE.search(first_line)
    if match:
        return FFmpegVersionInfo(first_line=first_line, major=int(match.group("major")))
    if _FFMPEG_DEV_BUILD_RE.search(first_line):
        return FFmpegVersionInfo(first_line=first_line, major=None, is_dev_build=True)

    raise RuntimeError(
        f"Could not determine {program_name} version from: {first_line}"
    )


def validate_ffmpeg_install(require_probe: bool = True) -> FFmpegValidationResult:
    """Validate FFmpeg/FFprobe availability, age, and Windows build type."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return FFmpegValidationResult(
            ok=False,
            message=(
                "FFmpeg not found. CorridorKey requires FFmpeg 7.0+ and FFprobe. "
                "Install a current FFmpeg build or re-run the installer."
            ),
        )

    ffprobe = find_ffprobe()
    if require_probe and not ffprobe:
        return FFmpegValidationResult(
            ok=False,
            message=(
                "FFprobe not found. CorridorKey requires both FFmpeg and FFprobe. "
                "Install a full FFmpeg build or re-run the installer."
            ),
            ffmpeg_path=ffmpeg,
        )

    try:
        ffmpeg_version = _read_program_version(ffmpeg, "ffmpeg")
    except RuntimeError as exc:
        return FFmpegValidationResult(ok=False, message=str(exc), ffmpeg_path=ffmpeg)

    if ffmpeg_version.major is not None and ffmpeg_version.major < _MIN_FFMPEG_MAJOR:
        return FFmpegValidationResult(
            ok=False,
            message=(
                f"FFmpeg 7.0 or newer is required. Detected {ffmpeg_version.first_line}."
            ),
            ffmpeg_path=ffmpeg,
            ffprobe_path=ffprobe,
            ffmpeg_version=ffmpeg_version,
        )

    if sys.platform == "win32" and "essentials_build" in ffmpeg_version.first_line.lower():
        return FFmpegValidationResult(
            ok=False,
            message=(
                "CorridorKey requires a full FFmpeg build on Windows. "
                "Detected a Gyan essentials build."
            ),
            ffmpeg_path=ffmpeg,
            ffprobe_path=ffprobe,
            ffmpeg_version=ffmpeg_version,
        )

    if _KNOWN_BAD_BUILD_RE.search(ffmpeg_version.first_line):
        return FFmpegValidationResult(
            ok=False,
            message=(
                "This FFmpeg build has a known frame-corruption bug with "
                "NVIDIA hardware decoding (random frames written as unreadable "
                f"EXR files). Detected {ffmpeg_version.first_line}. "
                "Run Repair FFmpeg to install a verified build."
            ),
            ffmpeg_path=ffmpeg,
            ffprobe_path=ffprobe,
            ffmpeg_version=ffmpeg_version,
            known_bad=True,
        )

    ffprobe_version: FFmpegVersionInfo | None = None
    if require_probe and ffprobe:
        try:
            ffprobe_version = _read_program_version(ffprobe, "ffprobe")
        except RuntimeError as exc:
            return FFmpegValidationResult(
                ok=False,
                message=str(exc),
                ffmpeg_path=ffmpeg,
                ffprobe_path=ffprobe,
                ffmpeg_version=ffmpeg_version,
            )

        if ffprobe_version.major is not None and ffprobe_version.major < _MIN_FFMPEG_MAJOR:
            return FFmpegValidationResult(
                ok=False,
                message=(
                    f"FFprobe 7.0 or newer is required. Detected {ffprobe_version.first_line}."
                ),
                ffmpeg_path=ffmpeg,
                ffprobe_path=ffprobe,
                ffmpeg_version=ffmpeg_version,
                ffprobe_version=ffprobe_version,
            )

        if (
            ffmpeg_version.major is not None
            and ffprobe_version.major is not None
            and ffmpeg_version.major != ffprobe_version.major
        ):
            return FFmpegValidationResult(
                ok=False,
                message=(
                    "FFmpeg and FFprobe major versions do not match. "
                    f"Detected ffmpeg {ffmpeg_version.major} and ffprobe {ffprobe_version.major}."
                ),
                ffmpeg_path=ffmpeg,
                ffprobe_path=ffprobe,
                ffmpeg_version=ffmpeg_version,
                ffprobe_version=ffprobe_version,
            )

        if (
            sys.platform == "win32"
            and "essentials_build" in ffprobe_version.first_line.lower()
        ):
            return FFmpegValidationResult(
                ok=False,
                message=(
                    "CorridorKey requires a full FFmpeg build on Windows. "
                    "Detected a Gyan essentials build."
                ),
                ffmpeg_path=ffmpeg,
                ffprobe_path=ffprobe,
                ffmpeg_version=ffmpeg_version,
                ffprobe_version=ffprobe_version,
            )

    if ffprobe_version is not None:
        summary = (
            f"FFmpeg OK: {ffmpeg_version.first_line} | {ffprobe_version.first_line}"
        )
    else:
        summary = f"FFmpeg OK: {ffmpeg_version.first_line}"

    return FFmpegValidationResult(
        ok=True,
        message=summary,
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
        ffmpeg_version=ffmpeg_version,
        ffprobe_version=ffprobe_version,
    )


def require_ffmpeg_install(require_probe: bool = True) -> FFmpegValidationResult:
    """Return the validated FFmpeg install or raise RuntimeError with detail."""
    result = validate_ffmpeg_install(require_probe=require_probe)
    if not result.ok:
        raise RuntimeError(result.message)
    return result


def get_ffmpeg_install_help() -> str:
    """Return concise install guidance for the current platform."""
    if sys.platform == "win32":
        return (
            "Use the CorridorKey Repair FFmpeg action or re-run 1-install.bat.\n"
            "CorridorKey will install a full bundled FFmpeg build into tools\\ffmpeg."
        )
    if sys.platform == "darwin":
        return (
            "Install a current FFmpeg build with Homebrew:\n"
            "    brew install ffmpeg\n\n"
            "Then verify:\n"
            "    ffmpeg -version\n"
            "    ffprobe -version"
        )
    if os.path.isfile("/etc/debian_version"):
        install_cmd = "sudo apt install ffmpeg"
    elif os.path.isfile("/etc/fedora-release"):
        install_cmd = "sudo dnf install ffmpeg"
    elif os.path.isfile("/etc/arch-release"):
        install_cmd = "sudo pacman -S ffmpeg"
    else:
        install_cmd = "Install ffmpeg with your package manager"
    return (
        f"{install_cmd}\n\n"
        "Then verify:\n"
        "    ffmpeg -version\n"
        "    ffprobe -version"
    )


def _safe_extract_zip(zip_path: str, dest_dir: str) -> None:
    """Extract a zip, rejecting any member that resolves outside dest_dir.

    Plain extractall is vulnerable to zip-slip (absolute paths or ../ escapes).
    """
    dest_root = os.path.abspath(dest_dir)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            target = os.path.abspath(os.path.join(dest_root, member))
            if target != dest_root and not target.startswith(dest_root + os.sep):
                raise RuntimeError(
                    f"Refusing to extract unsafe path from FFmpeg archive: {member}"
                )
        archive.extractall(dest_root)


def _find_staged_ffmpeg_dir(extract_dir: str) -> str | None:
    """Find the extracted folder that actually contains ffmpeg.exe + ffprobe.exe.

    Selects by CONTENTS, not folder name, so it survives BtbN renaming the inner
    directory across builds (master vs n8.1 etc.).
    """
    for root, _dirs, files in os.walk(extract_dir):
        lowered = {f.lower() for f in files}
        if "ffmpeg.exe" in lowered and "ffprobe.exe" in lowered:
            # root is the bin/ dir; the install root is its parent
            return os.path.dirname(root)
    return None


def _validate_staged_ffmpeg(install_root: str) -> None:
    """Validate staged ffmpeg/ffprobe binaries before swapping them in.

    Raises RuntimeError if either binary is missing, fails to report a version,
    or reports a major version below the minimum.
    """
    bin_dir = os.path.join(install_root, "bin")
    for name in ("ffmpeg", "ffprobe"):
        binary = os.path.join(bin_dir, f"{name}.exe")
        if not os.path.isfile(binary):
            raise RuntimeError(f"Downloaded FFmpeg build is missing {name}.exe")
        info = _read_program_version(binary, name)
        if info.major is not None and info.major < _MIN_FFMPEG_MAJOR:
            raise RuntimeError(
                f"Downloaded {name} is version {info.major}, but CorridorKey "
                f"requires {_MIN_FFMPEG_MAJOR}.0 or newer."
            )


def _robust_copytree(src: str, dst: str, attempts: int = 8, delay: float = 0.6) -> None:
    """Copy a directory tree to ``dst``, tolerating transient Windows file locks.

    A freshly extracted ``.exe`` is often briefly locked by an antivirus scan or
    a lingering execute handle from validation. Copying only needs read access to
    the source, so it succeeds where a move (which must delete the locked source)
    fails with ``WinError 32``. Retries with backoff cover the transient scan window.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            return
        except (PermissionError, OSError) as exc:
            last_exc = exc
            shutil.rmtree(dst, ignore_errors=True)
            time.sleep(delay * (attempt + 1))
    raise RuntimeError(
        f"Could not copy FFmpeg into place after {attempts} attempts "
        f"(file locked by antivirus or another process): {last_exc}"
    )


def _robust_move(src: str, dst: str, attempts: int = 8, delay: float = 0.6) -> None:
    """Move a directory, retrying through transient Windows file locks."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            shutil.move(src, dst)
            return
        except (PermissionError, OSError) as exc:
            last_exc = exc
            time.sleep(delay * (attempt + 1))
    raise RuntimeError(
        f"Could not move {src} to {dst} after {attempts} attempts: {last_exc}"
    )


def repair_ffmpeg_install(
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> FFmpegValidationResult:
    """Repair FFmpeg for the current platform.

    On Windows, downloads and installs a bundled full build into tools/ffmpeg.
    On macOS, installs via Homebrew (no sudo needed).
    On Linux, raises with install instructions (sudo requires a terminal).
    """
    if sys.platform == "darwin":
        def _emit(phase: str, current: int = 0, total: int = 0) -> None:
            if progress_callback:
                progress_callback(phase, current, total)

        if not shutil.which("brew"):
            raise RuntimeError(
                "Homebrew is not installed.\n\n"
                "Install it first:\n"
                '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"\n\n'
                "Then retry Repair FFmpeg."
            )

        _emit("Installing FFmpeg via Homebrew", 0, 0)
        try:
            subprocess.run(
                ["brew", "install", "ffmpeg"],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"FFmpeg install failed:\n{exc.stderr or exc.stdout or str(exc)}"
            ) from exc

        _emit("Validating FFmpeg", 0, 0)
        result = validate_ffmpeg_install(require_probe=True)
        if not result.ok:
            raise RuntimeError(result.message)
        return result

    if sys.platform != "win32":
        # Linux: needs sudo, can't run from GUI — show instructions instead
        raise RuntimeError(get_ffmpeg_install_help())

    def _emit(phase: str, current: int = 0, total: int = 0) -> None:
        if progress_callback:
            progress_callback(phase, current, total)

    tools_dir = os.path.join(_REPO_ROOT, "tools")
    dest_dir = os.path.join(tools_dir, "ffmpeg")
    temp_dir = os.path.join(_REPO_ROOT, ".tmp", "ffmpeg-repair")
    extract_dir = os.path.join(temp_dir, "extract")
    backup_dir = os.path.join(temp_dir, "ffmpeg-backup")

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(tools_dir, exist_ok=True)

    # Build an SSL context that works in frozen (PyInstaller) builds where
    # Python's bundled cert store may be incomplete. Try certifi first,
    # then the system default, then fall back to unverified as last resort.
    ssl_ctx: ssl.SSLContext | None = None
    try:
        import certifi
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        try:
            ssl_ctx = ssl.create_default_context()
        except Exception:
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            logger.warning(
                "SSL certificate verification disabled for FFmpeg download — "
                "certifi is missing and the system cert store failed."
            )

    # Resolve the newest STABLE BtbN n-build (never the master nightly).
    asset_name, download_url = _resolve_windows_ffmpeg_asset(ssl_ctx)
    zip_path = os.path.join(temp_dir, asset_name)

    _emit("Downloading FFmpeg", 0, 0)
    with urllib.request.urlopen(
        download_url, timeout=60, context=ssl_ctx,
    ) as response:
        total_header = response.headers.get("Content-Length", "")
        total_bytes = int(total_header) if total_header.isdigit() else 0
        downloaded = 0
        with open(zip_path, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                _emit("Downloading FFmpeg", downloaded, total_bytes)

    # Extract to a staging dir and validate BEFORE touching the working install,
    # so a bad download never strands the user without a usable FFmpeg.
    _emit("Extracting FFmpeg", 0, 0)
    if os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir, ignore_errors=True)
    os.makedirs(extract_dir, exist_ok=True)
    _safe_extract_zip(zip_path, extract_dir)

    staged_root = _find_staged_ffmpeg_dir(extract_dir)
    if staged_root is None:
        raise RuntimeError(
            "Downloaded FFmpeg archive did not contain ffmpeg.exe and ffprobe.exe."
        )

    _emit("Validating FFmpeg", 0, 0)
    _validate_staged_ffmpeg(staged_root)

    # Swap into place with backup + rollback. Move the old install aside first;
    # only delete it once the new one is validated in its final location.
    if os.path.isdir(backup_dir):
        shutil.rmtree(backup_dir, ignore_errors=True)
    had_previous = os.path.isdir(dest_dir)
    if had_previous:
        _robust_move(dest_dir, backup_dir)
    try:
        # COPY rather than move: the just-validated ffprobe.exe is briefly
        # locked (antivirus scan / execute handle), and a move would have to
        # delete that locked source file (WinError 32). Copying only reads the
        # staged tree, so the lock does not block it. Leftover staging is
        # cleaned best-effort in the finally below.
        _robust_copytree(staged_root, dest_dir)
        result = validate_ffmpeg_install(require_probe=True)
        if not result.ok:
            raise RuntimeError(result.message)
    except Exception:
        # Roll back to the previous working install.
        shutil.rmtree(dest_dir, ignore_errors=True)
        if had_previous and os.path.isdir(backup_dir):
            _robust_move(backup_dir, dest_dir)
            logger.warning("Repair FFmpeg failed; restored previous FFmpeg install.")
        raise
    finally:
        # Best-effort staging cleanup. A locked staged .exe here must never
        # fail the repair: the validated install already lives in dest_dir.
        shutil.rmtree(extract_dir, ignore_errors=True)

    # Success: drop the backup.
    shutil.rmtree(backup_dir, ignore_errors=True)
    return result

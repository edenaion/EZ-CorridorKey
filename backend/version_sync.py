"""Keep the Windows Apps & Features DisplayVersion in sync with the bundled version.

Inno Setup writes the uninstall metadata once at install time. Our skinny updater
replaces the on-disk binaries but never touches the registry, so the Installed
apps panel keeps showing the old number after an update. This module fixes that
at app startup.

Safety constraints (the whole point of doing this in Python rather than a shell
script):

* Windows only, frozen builds only.
* Only reads/writes the one value (DisplayVersion) under our own AppId uninstall
  key. Never creates keys. If the key is absent, we do nothing.
* Never raises. All exceptions are swallowed so app startup cannot be blocked.
* Self-healing: safe to call on every launch. Becomes a no-op once the registry
  and the bundled version match.
* Disable by setting the EZCK_DISABLE_VERSION_SYNC env var.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Inno Setup AppId for EZ-CorridorKey, suffixed with _is1 as Inno does for its
# uninstall entry. If the AppId in the .iss is ever regenerated, update this.
_INNO_APP_ID = "{E7A3F1B2-4D5C-4B8A-9E6F-1C2D3E4F5A6B}_is1"
_UNINSTALL_SUBKEY = (
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
    + "\\" + _INNO_APP_ID
)


def _read_bundled_version() -> str | None:
    """Return the version string from the bundled pyproject.toml, or None."""
    import tomllib

    candidates: list[str] = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(os.path.join(meipass, "pyproject.toml"))
    here = Path(__file__).resolve().parent
    candidates.append(str(here.parent / "pyproject.toml"))

    for path in candidates:
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)["project"]["version"]
        except Exception:
            continue
    return None


# Matches a trailing " v1.2.3" (with optional .4 / -rc1 / +build style suffix).
# Anchored at end-of-string so we only touch the last version token.
_NAME_VERSION_SUFFIX = re.compile(r" v\d+\.\d+(?:\.\d+)?[\w\.\-+]*$")


def _rewrite_display_name(current: str, new_version: str) -> str | None:
    """Return an updated DisplayName, or None if the pattern does not match.

    Inno writes DisplayName from `AppVerName = {#MyAppName} v{#MyAppVersion}`,
    so the expected shape is "EZ-CorridorKey v1.9.1". We only replace the
    trailing version token; the prefix is kept exactly as-is. If the input
    shape does not match we return None and skip the write.
    """
    if not current:
        return None
    new = _NAME_VERSION_SUFFIX.sub(f" v{new_version}", current)
    if new == current:
        return None
    return new


def _write_if_changed(key, name: str, new_value: str) -> tuple[bool, str | None]:
    """Write name=new_value to key as REG_SZ if it would change the stored value.

    Returns (wrote, previous_value). If the value is already correct or the
    value is absent (we only update, never create), returns (False, prev_or_None).
    """
    import winreg

    try:
        current, typ = winreg.QueryValueEx(key, name)
    except FileNotFoundError:
        return False, None
    if current == new_value:
        return False, current
    try:
        winreg.SetValueEx(key, name, 0, winreg.REG_SZ, new_value)
    except (OSError, PermissionError) as exc:
        logger.debug("%s write failed: %s", name, exc)
        return False, current
    return True, current


def _sync_hive(hive, new_version: str) -> bool:
    """Update DisplayVersion (and DisplayName when the pattern matches) under
    one hive if the key already exists.

    Returns True if any write actually happened. False means the key was
    missing, permission denied, or both values already matched.
    """
    import winreg

    try:
        key = winreg.OpenKey(
            hive,
            _UNINSTALL_SUBKEY,
            0,
            winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE,
        )
    except (FileNotFoundError, OSError, PermissionError):
        return False

    try:
        wrote_any = False

        wrote, prev = _write_if_changed(key, "DisplayVersion", new_version)
        if wrote:
            logger.info(
                "Apps & Features DisplayVersion: %s -> %s",
                prev, new_version,
            )
            wrote_any = True

        try:
            current_name, _ = winreg.QueryValueEx(key, "DisplayName")
        except FileNotFoundError:
            current_name = None
        if current_name:
            new_name = _rewrite_display_name(current_name, new_version)
            if new_name and new_name != current_name:
                try:
                    winreg.SetValueEx(
                        key, "DisplayName", 0, winreg.REG_SZ, new_name
                    )
                    logger.info(
                        "Apps & Features DisplayName: %s -> %s",
                        current_name, new_name,
                    )
                    wrote_any = True
                except (OSError, PermissionError) as exc:
                    logger.debug("DisplayName write failed: %s", exc)

        return wrote_any
    finally:
        try:
            winreg.CloseKey(key)
        except Exception:
            pass


def sync_uninstall_version() -> None:
    """Bring the uninstall registry entry's DisplayVersion in line with the app.

    Called from ui.app.create_app() once per launch. Short-circuits on non-
    Windows, in dev builds, or when disabled by env var.
    """
    if sys.platform != "win32":
        return
    if not getattr(sys, "frozen", False):
        return
    if os.environ.get("EZCK_DISABLE_VERSION_SYNC"):
        return

    try:
        import winreg  # noqa: F401 - availability check on Windows
    except ImportError:
        return

    new_version = _read_bundled_version()
    if not new_version:
        return

    try:
        import winreg

        # Try HKCU first because lowest-privilege installs land there.
        # Also try HKLM because users who elevated during install land there.
        # Silent failure on either is fine; at least one should match.
        hkcu_wrote = _sync_hive(winreg.HKEY_CURRENT_USER, new_version)
        hklm_wrote = _sync_hive(winreg.HKEY_LOCAL_MACHINE, new_version)
        if not (hkcu_wrote or hklm_wrote):
            logger.debug(
                "Uninstall version sync: no matching key found or already "
                "up to date (version=%s)", new_version,
            )
    except Exception as exc:
        logger.debug("Uninstall version sync skipped: %s", exc)

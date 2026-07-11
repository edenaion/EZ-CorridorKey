"""Optional error reporting.

Nothing is sent unless the user submits a report themselves or enables
crash reporting in Preferences (off by default). Paths are anonymized
and media file names are removed before anything leaves the machine.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import sys
import uuid

logger = logging.getLogger(__name__)

PRODUCT = "ez-corridorkey"

_DSN = "".join([
    "https://", "c6c8d171-fdd6-4649-8eb4-2ea4de28943e",
    "@", "ingest.", "ezscape.space", "/9",
])

# Stage values: installer, updater, runtime, inference.
# Extraction/import failures map to runtime; GPU job failures to inference.
VALID_STAGES = frozenset({"installer", "updater", "runtime", "inference"})

SETTINGS_ORG = "EZSCAPE"
SETTINGS_APP = "EZ-CorridorKey"
KEY_CRASH_REPORTS = "privacy/crash_reports_enabled"

_install_id_cache: str | None = None
_crash_reporting_active = False


def get_app_version() -> str:
    """App version from the bundled pyproject.toml."""
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(os.path.join(meipass, "pyproject.toml"))
    candidates.append(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "pyproject.toml",
    ))
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    m = re.match(r'\s*version\s*=\s*"([^"]+)"', line)
                    if m:
                        return m.group(1)
        except OSError:
            continue
    return "unknown"


def _config_dir() -> str:
    """Per-user local config dir (not the relocatable data dir)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(
            "~\\AppData\\Local")
        return os.path.join(base, "EZ-CorridorKey")
    if sys.platform == "darwin":
        return os.path.expanduser(
            "~/Library/Application Support/EZ-CorridorKey")
    return os.path.join(
        os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
        "EZ-CorridorKey",
    )


def get_install_id() -> str:
    """Stable random id for this install. Contains nothing identifying."""
    global _install_id_cache
    if _install_id_cache:
        return _install_id_cache

    path = os.path.join(_config_dir(), "install_id.json")
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh).get("install_id", "")
        uuid.UUID(value)
        _install_id_cache = value
        return value
    except (OSError, ValueError, AttributeError):
        pass

    value = str(uuid.uuid4())
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"install_id": value}, fh)
        os.replace(tmp, path)
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh).get("install_id", value)
    except (OSError, ValueError):
        pass  # in-memory id for this session only
    _install_id_cache = value
    return value


# ── Scrubbing ────────────────────────────────────────────────────────

_USER_PATH_RE = re.compile(
    r"(?i)((?:[A-Z]:)?[\\/](?:Users|home)[\\/])([^\\/\s\"']+)"
)
_PROJECT_SEG_RE = re.compile(
    r"(?i)([\\/](?:Projects|clips)[\\/])([^\\/\s\"']+)"
)
# App-generated file names that are safe to keep
_SAFE_BASENAME_RE = re.compile(
    r"(?i)^(frame_\d+\.exr|alpha[\w-]*\.(?:png|exr)|\.dwab_done|clip\.json|"
    r"\.?video_metadata\.json|[\w.-]+\.log)$"
)
_MEDIA_EXTS = (
    ".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v", ".gif",
    ".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp", ".dpx",
)
_MEDIA_EXT_GROUP = "|".join(ext.lstrip(".") for ext in _MEDIA_EXTS)
# Path segment (spaces allowed) after a separator, ending in a media ext.
# Angle brackets excluded so already-scrubbed placeholders never rematch.
_MEDIA_SEGMENT_RE = re.compile(
    r"(?i)([\\/])([^\\/\"'<>\r\n]{1,200}?)\.(" + _MEDIA_EXT_GROUP + r")\b"
)
# Bare token with no separator (single word file names in prose)
_MEDIA_TOKEN_RE = re.compile(
    r"(?i)(?<![\\/])\b([^\\/\s\"',;=<>]+)\.(" + _MEDIA_EXT_GROUP + r")\b"
)


def scrub_text(text: str) -> str:
    """Remove user names, project names, and media file names from text."""
    if not text:
        return text

    text = _USER_PATH_RE.sub(lambda m: m.group(1) + "<user>", text)
    text = _PROJECT_SEG_RE.sub(lambda m: m.group(1) + "<project>", text)

    def _segment(m: re.Match) -> str:
        base = m.group(2) + "." + m.group(3)
        if _SAFE_BASENAME_RE.match(base):
            return m.group(0)
        return m.group(1) + "<media>." + m.group(3)

    def _token(m: re.Match) -> str:
        if _SAFE_BASENAME_RE.match(m.group(0)):
            return m.group(0)
        return "<media>." + m.group(2)

    text = _MEDIA_SEGMENT_RE.sub(_segment, text)
    return _MEDIA_TOKEN_RE.sub(_token, text)


def scrub_event(event, hint=None):
    """Apply scrub_text to every string in an event, recursively."""
    def walk(node):
        if isinstance(node, str):
            return scrub_text(node)
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return [walk(v) for v in node]
        return node

    try:
        event = walk(event)
        event["server_name"] = ""
    except Exception:
        return None  # never send anything we failed to clean
    return event


# ── Environment tags ─────────────────────────────────────────────────

def collect_tags(gpu_info: dict | None = None, stage: str = "runtime") -> dict:
    """Machine-environment tags. Every value passes through scrub_text."""
    tags: dict[str, str] = {
        "product": PRODUCT,
        "app_version": get_app_version(),
        "os": platform.system().lower() or "unknown",
        "os_version": platform.release() or "unknown",
        "python_version": platform.python_version(),
        "stage": stage if stage in VALID_STAGES else "runtime",
    }

    try:
        import torch
        tags["torch_version"] = str(torch.__version__)
        cuda_ok = bool(torch.cuda.is_available())
        tags["cuda_available"] = "true" if cuda_ok else "false"
        tags["cuda_version"] = str(torch.version.cuda) if cuda_ok else "none"
        if cuda_ok:
            major, minor = torch.cuda.get_device_capability(0)
            tags["gpu_arch"] = f"sm_{major}{minor}"
            tags["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            tags["gpu_arch"] = "none"
    except Exception:
        tags.setdefault("torch_version", "missing")
        tags.setdefault("cuda_available", "false")
        tags.setdefault("gpu_arch", "none")

    if gpu_info:
        try:
            name = gpu_info.get("name")
            if name:
                tags["gpu_name"] = str(name)
            total = gpu_info.get("total_gb")
            if total:
                tags["vram_gb"] = str(int(round(float(total))))
        except Exception:
            pass

    try:
        from backend.ffmpeg_tools.discovery import validate_ffmpeg_install
        result = validate_ffmpeg_install(require_probe=False)
        if result.ffmpeg_version is not None:
            parts = result.ffmpeg_version.first_line.split()
            tags["ffmpeg_version"] = parts[2] if len(parts) > 2 else "unknown"
        else:
            tags["ffmpeg_version"] = "missing"
    except Exception:
        tags["ffmpeg_version"] = "missing"

    return {k: scrub_text(str(v)) for k, v in tags.items()}


# ── Sending ──────────────────────────────────────────────────────────

def _base_init_kwargs() -> dict:
    frozen = bool(getattr(sys, "frozen", False))
    return {
        "dsn": _DSN,
        "release": f"{PRODUCT}@{get_app_version()}",
        "environment": "production" if frozen else "dev",
        "auto_session_tracking": False,
        "traces_sample_rate": None,
        "before_send_transaction": lambda event, hint: None,
        "send_default_pii": False,
        "send_client_reports": False,
        "include_local_variables": False,
        "before_send": scrub_event,
        "shutdown_timeout": 2,
    }


def send_report(
    bundle_text: str,
    stage: str = "runtime",
    gpu_info: dict | None = None,
    exc_info=None,
    timeout: float = 5.0,
) -> bool:
    """Send one user-initiated report. Returns False on any failure.

    Uses its own client so nothing stays active afterwards. Must never
    raise: the caller's clipboard/GitHub flow proceeds regardless.
    """
    try:
        import sentry_sdk

        client = sentry_sdk.Client(
            **_base_init_kwargs(),
            default_integrations=False,
            max_breadcrumbs=0,
        )
        try:
            scope = sentry_sdk.Scope(client=client)
            scope.set_user({"id": get_install_id()})
            for key, value in collect_tags(gpu_info, stage).items():
                scope.set_tag(key, value)
            scope.set_extra("bundle", scrub_text(bundle_text or ""))

            if exc_info is not None:
                scope.capture_exception(exc_info)
            else:
                headline = "User report"
                for line in (bundle_text or "").splitlines():
                    if "ERROR" in line:
                        headline = scrub_text(line.strip())[:200]
                        break
                scope.capture_message(headline, level="error")

            client.flush(timeout=timeout)
        finally:
            client.close(timeout=0)
        return True
    except Exception as exc:
        logger.debug(f"Report send skipped: {exc}")
        return False


def is_crash_reporting_enabled() -> bool:
    """The Preferences toggle. Off means nothing is ever initialized."""
    try:
        from PySide6.QtCore import QSettings
        s = QSettings(SETTINGS_ORG, SETTINGS_APP)
        return s.value(KEY_CRASH_REPORTS, False, type=bool)
    except Exception:
        return False


def init_crash_reporting() -> bool:
    """Start automatic crash capture. Only called when the user opted in.

    No-op in non-frozen builds and whenever the toggle is off.
    """
    global _crash_reporting_active
    if _crash_reporting_active:
        return True
    if not getattr(sys, "frozen", False):
        return False
    if not is_crash_reporting_enabled():
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.atexit import AtexitIntegration
        from sentry_sdk.integrations.dedupe import DedupeIntegration
        from sentry_sdk.integrations.excepthook import ExcepthookIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
        from sentry_sdk.integrations.threading import ThreadingIntegration

        sentry_sdk.init(
            **_base_init_kwargs(),
            default_integrations=False,
            auto_enabling_integrations=False,
            max_breadcrumbs=20,
            integrations=[
                ExcepthookIntegration(always_run=True),
                ThreadingIntegration(propagate_hub=True),
                AtexitIntegration(callback=lambda p, t: None),
                DedupeIntegration(),
                LoggingIntegration(level=None, event_level=None),
            ],
        )
        sentry_sdk.set_user({"id": get_install_id()})
        for key, value in collect_tags(None, "runtime").items():
            if key != "stage":
                sentry_sdk.set_tag(key, value)
        _crash_reporting_active = True
        return True
    except Exception as exc:
        logger.debug(f"Crash reporting not started: {exc}")
        return False


def capture_stage_exception(stage: str, exc: BaseException) -> None:
    """Record an exception from a known subsystem. Honors the toggle."""
    if not _crash_reporting_active or not is_crash_reporting_enabled():
        return
    try:
        import sentry_sdk

        with sentry_sdk.new_scope() as scope:
            scope.set_tag("stage",
                          stage if stage in VALID_STAGES else "runtime")
            sentry_sdk.capture_exception(exc)
    except Exception:
        pass

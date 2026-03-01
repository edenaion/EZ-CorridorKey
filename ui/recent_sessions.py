"""App-level recent sessions store.

Tracks which workspaces the user has opened, persisted as JSON at
%APPDATA%/CorridorKey/recent_sessions.json (Windows) or
~/.config/corridorkey/recent_sessions.json (Linux/macOS).

Independent of per-workspace session sidecars (.corridorkey_session.json).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

_FILENAME = "recent_sessions.json"


def _config_dir() -> str:
    """Platform-appropriate config directory for app-level settings."""
    if os.name == "nt":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        return os.path.join(base, "CorridorKey")
    # Linux/macOS
    xdg = os.environ.get("XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config"))
    return os.path.join(xdg, "corridorkey")


@dataclass
class RecentSession:
    """A recently-opened workspace."""
    workspace_path: str
    display_name: str
    last_opened: float
    clip_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RecentSession:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


class RecentSessionsStore:
    """Manages the list of recently-opened workspaces.

    Persisted as a JSON file in the app config directory.
    Thread-safe for single-threaded Qt main thread usage.
    """
    MAX_RECENT = 20

    def __init__(self, config_dir: str | None = None):
        self._dir = config_dir or _config_dir()
        os.makedirs(self._dir, exist_ok=True)
        self._path = os.path.join(self._dir, _FILENAME)
        self._sessions: list[RecentSession] = self._load()

    def _norm(self, path: str) -> str:
        """Normalize path for comparison (case-insensitive on Windows)."""
        return os.path.normcase(os.path.normpath(path))

    def _load(self) -> list[RecentSession]:
        """Load sessions from JSON file."""
        if not os.path.isfile(self._path):
            return []
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            sessions = []
            for entry in data:
                try:
                    sessions.append(RecentSession.from_dict(entry))
                except (TypeError, KeyError):
                    continue
            return sessions
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load recent sessions: {e}")
            return []

    def _save(self) -> None:
        """Atomic write to JSON file."""
        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump([s.to_dict() for s in self._sessions], f, indent=2)
            if os.path.exists(self._path):
                os.remove(self._path)
            os.rename(tmp, self._path)
        except OSError as e:
            logger.warning(f"Failed to save recent sessions: {e}")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def add_or_update(self, workspace_path: str, display_name: str, clip_count: int) -> None:
        """Add or update a workspace in the recents list."""
        norm = self._norm(workspace_path)
        # Remove existing entry for this path
        self._sessions = [s for s in self._sessions if self._norm(s.workspace_path) != norm]
        # Add at front
        self._sessions.insert(0, RecentSession(
            workspace_path=workspace_path,
            display_name=display_name,
            last_opened=time.time(),
            clip_count=clip_count,
        ))
        # Trim
        self._sessions = self._sessions[:self.MAX_RECENT]
        self._save()

    def remove(self, workspace_path: str) -> None:
        """Remove a workspace from the recents list."""
        norm = self._norm(workspace_path)
        self._sessions = [s for s in self._sessions if self._norm(s.workspace_path) != norm]
        self._save()

    def get_all(self) -> list[RecentSession]:
        """Return all recent sessions, sorted by last_opened descending."""
        return sorted(self._sessions, key=lambda s: s.last_opened, reverse=True)

    def prune_missing(self) -> int:
        """Remove entries whose workspace directory no longer exists."""
        before = len(self._sessions)
        self._sessions = [s for s in self._sessions if os.path.isdir(s.workspace_path)]
        pruned = before - len(self._sessions)
        if pruned > 0:
            self._save()
            logger.info(f"Pruned {pruned} missing workspace(s) from recent sessions")
        return pruned

    def has_session_file(self, workspace_path: str) -> bool:
        """Check if a workspace has a .corridorkey_session.json sidecar."""
        return os.path.isfile(os.path.join(workspace_path, ".corridorkey_session.json"))

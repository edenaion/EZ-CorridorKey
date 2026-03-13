"""Shortcut registry — single source of truth for keyboard bindings.

Stores default shortcuts, loads/saves user overrides via QSettings,
creates QShortcut objects, and detects conflicts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)

_QSETTINGS_GROUP = "shortcuts"


@dataclass
class ShortcutDef:
    """One shortcut definition."""
    action_id: str          # Unique internal ID, e.g. "run_inference"
    display_name: str       # Human-readable label shown in the dialog
    category: str           # Grouping: "Global", "Timeline", etc.
    default_key: str        # Default key sequence string, e.g. "Ctrl+R"
    callback_name: str      # Name of the method on MainWindow to call
    menu_action: bool = False  # True for shortcuts managed by QAction (menu bar)
    app_shortcut: bool = False  # True to use ApplicationShortcut context (fires globally)


# Authoritative list — order here = display order in dialog
SHORTCUT_DEFAULTS: list[ShortcutDef] = [
    # Global
    ShortcutDef("escape",             "Stop / Cancel",                "Global",     "Esc",          "_on_escape"),
    ShortcutDef("run_inference",      "Run Inference",                "Global",     "Ctrl+R",       "_on_run_inference"),
    ShortcutDef("run_all",            "Run All Clips",                "Global",     "Ctrl+Shift+R", "_on_run_all_ready"),
    ShortcutDef("save_session",       "Save Session",                 "Global",     "Ctrl+S",       "_on_save_session",    menu_action=True),
    ShortcutDef("open_project",       "Open Project",                 "Global",     "Ctrl+O",       "_on_open_project",    menu_action=True),
    ShortcutDef("toggle_mute",        "Toggle Mute",                  "Global",     "Ctrl+M",       "_toggle_mute"),
    ShortcutDef("welcome_screen",     "Return to Home",               "Global",     "Home",         "_return_to_welcome"),
    ShortcutDef("delete_clips",       "Remove Selected Clips",        "Global",     "Del",          "_on_delete_selected_clips"),
    ShortcutDef("toggle_queue",       "Toggle Queue",                 "Global",     "Q",            "_toggle_queue_panel"),
    ShortcutDef("debug_console",     "Debug Console",                "Global",     "F12",          "_toggle_debug_console", app_shortcut=True),
    # Timeline
    ShortcutDef("set_in",             "Set In-Point",                 "Timeline",   "I",            "_set_in_point"),
    ShortcutDef("set_out",            "Set Out-Point",                "Timeline",   "O",            "_set_out_point"),
    ShortcutDef("clear_in_out",       "Clear In/Out",                 "Timeline",   "Alt+I",        "_clear_in_out"),
    # Playback
    ShortcutDef("play_pause",         "Play / Pause",                 "Playback",   "Space",        "_toggle_playback"),
    # Paint
    ShortcutDef("annotation_fg",      "Foreground Paint",             "Paint",      "1",            "_toggle_annotation_fg"),
    ShortcutDef("annotation_bg",      "Background Paint (Red)",       "Paint",      "2",            "_toggle_annotation_bg"),
    ShortcutDef("cycle_fg_color",     "Cycle Foreground Color",       "Paint",      "C",            "_cycle_fg_color"),
    ShortcutDef("undo_annotation",    "Undo Paint Stroke",            "Paint",      "Ctrl+Z",       "_undo_annotation"),
    ShortcutDef("clear_annotations",  "Clear Paint Strokes",          "Paint",      "Ctrl+C",       "_confirm_clear_annotations"),
]

# Category display order
CATEGORY_ORDER = ["Global", "Timeline", "Playback", "Paint"]


class ShortcutRegistry:
    """Manages shortcut definitions, user overrides, QShortcut objects, and conflicts."""

    def __init__(self) -> None:
        self._defs: dict[str, ShortcutDef] = {d.action_id: d for d in SHORTCUT_DEFAULTS}
        self._overrides: dict[str, str] = {}
        self._shortcuts: dict[str, QShortcut] = {}
        self._load_overrides()

    def _load_overrides(self) -> None:
        """Load user-customized key bindings from QSettings."""
        s = QSettings()
        s.beginGroup(_QSETTINGS_GROUP)
        for key in s.childKeys():
            if key in self._defs:
                self._overrides[key] = s.value(key, type=str)
        s.endGroup()
        if self._overrides:
            logger.info(f"Loaded {len(self._overrides)} custom shortcut(s)")

    def save_overrides(self) -> None:
        """Persist current overrides to QSettings. Only stores non-default keys."""
        s = QSettings()
        s.beginGroup(_QSETTINGS_GROUP)
        for key in s.childKeys():
            s.remove(key)
        for action_id, key_str in self._overrides.items():
            if key_str != self._defs[action_id].default_key:
                s.setValue(action_id, key_str)
        s.endGroup()

    def get_key(self, action_id: str) -> str:
        """Return the effective key sequence string for an action."""
        return self._overrides.get(action_id, self._defs[action_id].default_key)

    def set_key(self, action_id: str, key_str: str) -> None:
        """Set a custom key binding (in-memory only until save_overrides)."""
        self._overrides[action_id] = key_str

    def reset_key(self, action_id: str) -> None:
        """Reset a single action to its default key."""
        self._overrides.pop(action_id, None)

    def reset_all(self) -> None:
        """Reset all actions to defaults."""
        self._overrides.clear()

    def is_default(self, action_id: str) -> bool:
        """True if the action uses its default key (no custom override)."""
        return action_id not in self._overrides or \
            self._overrides[action_id] == self._defs[action_id].default_key

    def get_def(self, action_id: str) -> ShortcutDef | None:
        """Return the definition for an action."""
        return self._defs.get(action_id)

    def definitions(self) -> list[ShortcutDef]:
        """Return all definitions in their original order."""
        return list(SHORTCUT_DEFAULTS)

    def snapshot_overrides(self) -> dict[str, str]:
        """Return a copy of current overrides (for cancel/revert in dialogs)."""
        return dict(self._overrides)

    def restore_overrides(self, snapshot: dict[str, str]) -> None:
        """Restore overrides from a previous snapshot."""
        self._overrides = dict(snapshot)

    def find_conflicts(self, action_id: str, key_str: str) -> list[str]:
        """Return action_ids that already use key_str (excluding action_id itself)."""
        if not key_str:
            return []
        target = QKeySequence(key_str)
        conflicts = []
        for aid in self._defs:
            if aid == action_id:
                continue
            effective = self.get_key(aid)
            if effective and QKeySequence(effective) == target:
                conflicts.append(aid)
        return conflicts

    # ── Live QShortcut management ──

    def create_shortcuts(self, owner: QWidget) -> None:
        """Create QShortcut objects for all non-menu-action shortcuts."""
        self.destroy_shortcuts()
        for defn in SHORTCUT_DEFAULTS:
            if defn.menu_action:
                continue  # handled by QAction in menu bar
            key_str = self.get_key(defn.action_id)
            if not key_str:
                continue  # unbound
            callback = getattr(owner, defn.callback_name, None)
            if callback is None:
                logger.warning(f"Shortcut callback not found: {defn.callback_name}")
                continue
            sc = QShortcut(QKeySequence(key_str), owner)
            if defn.app_shortcut:
                sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(callback)
            self._shortcuts[defn.action_id] = sc

    def destroy_shortcuts(self) -> None:
        """Delete all existing QShortcut objects."""
        for sc in self._shortcuts.values():
            sc.deleteLater()
        self._shortcuts.clear()

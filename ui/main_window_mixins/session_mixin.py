from __future__ import annotations

import json
import logging
import os

from PySide6.QtWidgets import QMessageBox, QFileDialog
from PySide6.QtCore import Slot, QTimer

from backend import ClipEntry, InferenceParams, OutputConfig

logger = logging.getLogger(__name__)

# Session file stored in clips dir (Codex: JSON sidecar)
_SESSION_FILENAME = ".corridorkey_session.json"
_SESSION_VERSION = 1


class SessionMixin:
    """Session save/load and color-space memory for MainWindow."""

    def _session_path(self) -> str | None:
        """Return session file path, or None if no clips dir or Projects root.

        The Projects root should never have a session file — sessions are
        scoped to individual project folders to prevent cross-contamination.
        """
        if not self._clips_dir:
            return None
        from backend.project import projects_root as _projects_root

        try:
            if os.path.normcase(os.path.abspath(self._clips_dir)) == os.path.normcase(
                os.path.abspath(_projects_root())
            ):
                return None
        except Exception:
            pass
        return os.path.join(self._clips_dir, _SESSION_FILENAME)

    def _build_session_data(self) -> dict:
        """Build session data dict from current UI state."""
        self._remember_current_clip_input_color_space()
        data: dict = {
            "version": _SESSION_VERSION,
            "params": self._param_panel.get_params().to_dict(),
            "output_config": self._param_panel.get_output_config().to_dict(),
            "live_preview": self._param_panel.live_preview_enabled,
        }

        # Window geometry
        geo = self.geometry()
        data["geometry"] = {
            "x": geo.x(),
            "y": geo.y(),
            "width": geo.width(),
            "height": geo.height(),
        }

        # Splitter sizes
        data["splitter_sizes"] = self._splitter.sizes()
        data["vsplitter_sizes"] = self._vsplitter.sizes()

        # Workspace path (for absolute reference)
        if self._clips_dir:
            data["workspace_path"] = self._clips_dir

        # Selected clip
        if self._current_clip:
            data["selected_clip"] = self._current_clip.name
        if self._clip_input_is_linear:
            data["clip_input_is_linear"] = dict(sorted(self._clip_input_is_linear.items()))

        return data

    def _apply_session_data(self, data: dict) -> None:
        """Apply session data to UI widgets.

        Codex: block widget signals during restore to prevent event storms.
        Ignores unknown keys for forward compatibility.
        """
        version = data.get("version", 0)
        if version > _SESSION_VERSION:
            logger.warning(f"Session version {version} is newer than supported {_SESSION_VERSION}")

        # Restore params
        if "params" in data:
            try:
                params = InferenceParams.from_dict(data["params"])
                self._param_panel.set_params(params)
            except Exception as e:
                logger.warning(f"Failed to restore params: {e}")

        # Restore output config
        if "output_config" in data:
            try:
                config = OutputConfig.from_dict(data["output_config"])
                self._param_panel.set_output_config(config)
            except Exception as e:
                logger.warning(f"Failed to restore output config: {e}")

        # Restore live preview toggle
        if "live_preview" in data:
            self._param_panel._live_preview.setChecked(bool(data["live_preview"]))

        if "clip_input_is_linear" in data:
            try:
                loaded = data["clip_input_is_linear"]
                if isinstance(loaded, dict):
                    self._clip_input_is_linear = {
                        str(name): bool(value) for name, value in loaded.items()
                    }
            except Exception as e:
                logger.warning(f"Failed to restore clip color-space overrides: {e}")

        # Restore splitter sizes (validate: must have 2 panels, none at 0)
        if "splitter_sizes" in data:
            try:
                sizes = [int(s) for s in data["splitter_sizes"]]
                if len(sizes) == 2 and all(s > 0 for s in sizes):
                    self._splitter.setSizes(sizes)
                else:
                    logger.info("Ignoring invalid splitter_sizes, using defaults")
            except Exception:
                pass

        # Restore vertical splitter sizes (validate: must have 2 panels)
        if "vsplitter_sizes" in data:
            try:
                sizes = [int(s) for s in data["vsplitter_sizes"]]
                if len(sizes) == 2 and all(s > 0 for s in sizes):
                    self._vsplitter.setSizes(sizes)
                else:
                    logger.info("Ignoring invalid vsplitter_sizes, using defaults")
            except Exception:
                pass

        # Restore window geometry (clamped to current screen)
        if "geometry" in data:
            try:
                if not getattr(self, "_container_mode", False):
                    g = data["geometry"]
                    self.setGeometry(g["x"], g["y"], g["width"], g["height"])
            except Exception:
                pass

        QTimer.singleShot(0, self._ensure_window_mode)

        # Restore selected clip
        if "selected_clip" in data:
            clip_name = data["selected_clip"]
            for clip in self._clip_model.clips:
                if clip.name == clip_name:
                    self._on_clip_selected(clip)
                    break

    def _remember_current_clip_input_color_space(self) -> None:
        """Persist the current clip's chosen input interpretation in memory."""
        if self._current_clip is None:
            return
        self._clip_input_is_linear[self._current_clip.name] = (
            self._param_panel.get_params().input_is_linear
        )

    def _input_is_linear_for_clip(self, clip: ClipEntry) -> bool:
        """Return the remembered input interpretation for a clip, or its default."""
        if clip.name in self._clip_input_is_linear:
            return self._clip_input_is_linear[clip.name]

        input_is_linear = bool(clip.input_asset is not None and clip.should_default_input_linear())
        self._clip_input_is_linear[clip.name] = input_is_linear
        return input_is_linear

    @Slot()
    def _on_save_session(self) -> None:
        """Save session to JSON sidecar in clips directory."""
        path = self._session_path()
        if not path:
            QMessageBox.information(self, "No Folder", "Open a clips folder first.")
            return

        data = self._build_session_data()
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            # Atomic rename (Windows: need to remove target first)
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp_path, path)
            logger.info(f"Session saved: {path}")
        except OSError as e:
            logger.warning(f"Failed to save session: {e}")
            # Clean up tmp if it exists
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _auto_save_session(self) -> None:
        """Periodic auto-save for crash recovery (called by timer)."""
        if self._clips_dir and self._stack.currentIndex() == 1:
            path = self._session_path()
            if not path:
                return
            data = self._build_session_data()
            tmp_path = path + ".tmp"
            try:
                with open(tmp_path, "w") as f:
                    json.dump(data, f, indent=2)
                if os.path.exists(path):
                    os.remove(path)
                os.rename(tmp_path, path)
            except OSError:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    @Slot()
    def _on_open_project(self) -> None:
        """Open a project folder via directory picker (Ctrl+O)."""
        from backend.project import projects_root

        start_dir = projects_root()
        folder = QFileDialog.getExistingDirectory(
            self,
            "Open Project",
            start_dir,
        )
        if not folder:
            return
        self._switch_to_workspace()
        self._on_clips_dir_changed(folder, skip_session_restore=False)

    def _try_auto_load_session(self, clips_dir: str) -> None:
        """Auto-load session if .corridorkey_session.json exists in clips dir."""
        path = os.path.join(clips_dir, _SESSION_FILENAME)
        if os.path.isfile(path):
            self._load_session_from(path)

    def _load_session_from(self, path: str) -> None:
        """Load session data from a file path."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._apply_session_data(data)
            logger.info(f"Session loaded: {path}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load session: {e}")

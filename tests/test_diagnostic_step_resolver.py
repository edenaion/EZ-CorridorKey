"""Tests for context-aware Startup Diagnostic step resolution.

Covers the regression from GitHub issue #89: a user on the Windows
installer build with no NVIDIA GPU was shown dev-mode guidance
(``.venv\\Scripts\\activate`` / ``pip install ...``) that cannot work
on a frozen build and cannot help a machine without an NVIDIA card.
``resolve_steps`` must produce contextually appropriate guidance for
each combination of ``sys.frozen`` and NVIDIA presence.
"""
from __future__ import annotations

from unittest.mock import patch

from ui.widgets.diagnostic_checks import (
    _DIAGNOSTICS,
    Diagnostic,
    resolve_steps,
)


def _by_id(diag_id: str) -> Diagnostic:
    return next(d for d in _DIAGNOSTICS if d.id == diag_id)


def _text(steps: list[str]) -> str:
    return " ".join(steps).lower()


class TestGpuRequiredResolution:
    """Issue #89 is specifically this diagnostic's dev-mode defaults."""

    DIAG_ID = "gpu-required"

    def test_frozen_no_nvidia_replaces_venv_guidance(self):
        """The exact bug in #89: frozen installer + no NVIDIA must NOT
        tell the user to activate a venv or pip-install torch."""
        ctx = {"is_frozen": True, "has_nvidia": False, "platform": "win32"}
        with patch(
            "ui.widgets.diagnostic_checks._runtime_context", return_value=ctx
        ):
            steps = resolve_steps(_by_id(self.DIAG_ID))
        text = _text(steps)
        assert "venv" not in text
        assert "pip install" not in text
        assert "nvidia" in text
        assert "nvidia-smi" in text

    def test_frozen_with_nvidia_points_to_reinstall(self):
        """Frozen build + NVIDIA present but CUDA broken → reinstall
        the official build, not venv/pip."""
        ctx = {"is_frozen": True, "has_nvidia": True, "platform": "win32"}
        with patch(
            "ui.widgets.diagnostic_checks._runtime_context", return_value=ctx
        ):
            steps = resolve_steps(_by_id(self.DIAG_ID))
        text = _text(steps)
        assert "venv" not in text
        assert "installer" in text
        assert "releases" in text

    def test_dev_with_nvidia_keeps_venv_guidance(self):
        """Developers running from a git clone should still see the
        original ``.venv`` activation + pip reinstall steps — that's
        who those instructions were written for."""
        ctx = {"is_frozen": False, "has_nvidia": True, "platform": "win32"}
        with patch(
            "ui.widgets.diagnostic_checks._runtime_context", return_value=ctx
        ):
            steps = resolve_steps(_by_id(self.DIAG_ID))
        text = _text(steps)
        assert "venv" in text
        assert "pip install" in text

    def test_dev_no_nvidia_still_says_need_nvidia(self):
        """Dev clone without an NVIDIA card is the same end-user
        situation: no amount of pip install fixes it."""
        ctx = {"is_frozen": False, "has_nvidia": False, "platform": "linux"}
        with patch(
            "ui.widgets.diagnostic_checks._runtime_context", return_value=ctx
        ):
            steps = resolve_steps(_by_id(self.DIAG_ID))
        text = _text(steps)
        assert "nvidia" in text
        assert "pip install" not in text


class TestPytorchCpuWheelResolution:
    DIAG_ID = "pytorch-cpu-wheel"

    def test_frozen_points_to_reinstall(self):
        ctx = {"is_frozen": True, "has_nvidia": True, "platform": "win32"}
        with patch(
            "ui.widgets.diagnostic_checks._runtime_context", return_value=ctx
        ):
            steps = resolve_steps(_by_id(self.DIAG_ID))
        text = _text(steps)
        assert "venv" not in text
        assert "installer" in text

    def test_no_nvidia_redirects_to_hardware_guidance(self):
        ctx = {"is_frozen": False, "has_nvidia": False, "platform": "win32"}
        with patch(
            "ui.widgets.diagnostic_checks._runtime_context", return_value=ctx
        ):
            steps = resolve_steps(_by_id(self.DIAG_ID))
        text = _text(steps)
        assert "nvidia" in text
        assert "pip uninstall" not in text


class TestTritonMissingResolution:
    DIAG_ID = "triton-missing"

    def test_frozen_downgrades_to_safe_to_ignore(self):
        ctx = {"is_frozen": True, "has_nvidia": True, "platform": "win32"}
        with patch(
            "ui.widgets.diagnostic_checks._runtime_context", return_value=ctx
        ):
            steps = resolve_steps(_by_id(self.DIAG_ID))
        text = _text(steps)
        assert "safe to ignore" in text or "safe to\nignore" in text
        assert "venv" not in text
        assert "pip install" not in text

    def test_dev_keeps_install_instructions(self):
        ctx = {"is_frozen": False, "has_nvidia": True, "platform": "win32"}
        with patch(
            "ui.widgets.diagnostic_checks._runtime_context", return_value=ctx
        ):
            steps = resolve_steps(_by_id(self.DIAG_ID))
        text = _text(steps)
        assert "triton" in text
        assert "venv" in text


class TestMissingCheckpointResolution:
    DIAG_ID = "missing-checkpoint"

    def test_frozen_routes_to_setup_wizard(self):
        ctx = {"is_frozen": True, "has_nvidia": True, "platform": "win32"}
        with patch(
            "ui.widgets.diagnostic_checks._runtime_context", return_value=ctx
        ):
            steps = resolve_steps(_by_id(self.DIAG_ID))
        text = _text(steps)
        assert "setup wizard" in text
        # Must not leak the dev-only file-path instruction.
        assert "corridorkeymodule/checkpoints" not in text

    def test_dev_keeps_original_instructions(self):
        ctx = {"is_frozen": False, "has_nvidia": True, "platform": "win32"}
        with patch(
            "ui.widgets.diagnostic_checks._runtime_context", return_value=ctx
        ):
            steps = resolve_steps(_by_id(self.DIAG_ID))
        text = _text(steps)
        assert "corridorkeymodule/checkpoints" in text or "checkpoints" in text


class TestUnregisteredDiagnosticPassthrough:
    def test_diagnostic_without_resolver_returns_original_steps(self):
        """Diagnostics that aren't in _STEP_RESOLVERS must fall through
        to their static ``steps`` list unchanged — the resolver is
        opt-in per-diagnostic."""
        diag = _by_id("cuda-oom")  # not in _STEP_RESOLVERS
        ctx = {"is_frozen": True, "has_nvidia": False, "platform": "win32"}
        with patch(
            "ui.widgets.diagnostic_checks._runtime_context", return_value=ctx
        ):
            steps = resolve_steps(diag)
        assert steps == diag.steps

    def test_resolver_exception_falls_back_to_static_steps(self):
        """If the resolver explodes we must degrade gracefully to the
        original step list, not break the dialog."""
        diag = _by_id("gpu-required")
        with patch(
            "ui.widgets.diagnostic_checks._runtime_context",
            side_effect=RuntimeError("pynvml is on fire"),
        ):
            steps = resolve_steps(diag)
        assert steps == diag.steps

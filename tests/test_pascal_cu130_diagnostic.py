"""Tests for the Pascal-on-cu130 startup diagnostic.

Background
----------
EZ-CorridorKey 1.9.1 shipped with PyTorch cu130 wheels, which omit
sm_60/61 (Pascal) GPU kernels. Pascal users (GTX 10-series) crash
with "CUDA error: no kernel image is available for execution on the
device" the first time inference runs.

1.9.2 fixes this by shipping cu128 wheels in the full installer.
**However**, the in-app updater intentionally only ships a code-only
update zip, leaving the existing torch runtime in place. So a 1.9.1
Pascal user who clicks "Check for Updates" still has cu130 wheels and
will still crash.

The ``_pascal_cu130_mismatch`` helper detects exactly this state at
startup so we can show them a clear "download the full installer"
dialog before they hit the cryptic CUDA error.

Safety contract
---------------
This helper runs at every app startup. It MUST never raise — any
torch / nvidia / pynvml weirdness has to degrade silently to "not a
mismatch". These tests pin that contract by exercising every failure
mode I can think of.

False-positive contract
-----------------------
This helper MUST NOT flag:
  * RTX 20-series and newer (cc 7.5+) on any wheel
  * Pascal cards on cu128 wheels (the post-1.9.2 happy path)
  * machines without an NVIDIA GPU
  * cards on the cpu wheel
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

from ui.widgets.diagnostic_checks import (
    _DIAGNOSTICS,
    _pascal_cu130_mismatch,
    run_startup_diagnostics,
)


def _fake_torch(
    *,
    cuda_version: str | None,
    cuda_available: bool = True,
    device_count: int = 1,
    capability: tuple = (6, 1),
    device_name: str = "NVIDIA GeForce GTX 1080 Ti",
    raise_on: str | None = None,
):
    """Build a stand-in torch module exposing exactly what the helper reads.

    ``raise_on`` accepts the name of an attribute access that should
    raise instead of returning — used to verify each step's
    try/except guards individually.
    """

    fake = types.SimpleNamespace()
    fake.__version__ = "2.9.1+cu130" if cuda_version == "13.0" else "2.9.1"

    version = types.SimpleNamespace()
    if raise_on == "version.cuda":

        def _raise():
            raise RuntimeError("torch.version.cuda is on fire")

        version.__getattr__ = lambda self, name: _raise()  # noqa: ARG005
    else:
        version.cuda = cuda_version
    fake.version = version

    cuda = types.SimpleNamespace()

    def _is_available():
        if raise_on == "is_available":
            raise RuntimeError("cuda.is_available is on fire")
        return cuda_available

    def _device_count():
        if raise_on == "device_count":
            raise RuntimeError("cuda.device_count is on fire")
        return device_count

    def _get_device_capability(idx):
        if raise_on == "get_device_capability":
            raise RuntimeError("get_device_capability is on fire")
        return capability

    def _get_device_name(idx):
        if raise_on == "get_device_name":
            raise RuntimeError("get_device_name is on fire")
        return device_name

    cuda.is_available = _is_available
    cuda.device_count = _device_count
    cuda.get_device_capability = _get_device_capability
    cuda.get_device_name = _get_device_name
    fake.cuda = cuda
    return fake


def _patch_torch(monkeypatch, fake):
    """Inject a fake torch module so the helper imports our stub."""
    monkeypatch.setitem(sys.modules, "torch", fake)


# ── Positive cases (must flag) ──────────────────────────────────────


class TestMustFlagPascalOnCu130:

    def test_gtx_1080_ti_on_cu130_is_flagged(self, monkeypatch):
        """The exact #87 user scenario: 1080 Ti on cu130."""
        _patch_torch(
            monkeypatch,
            _fake_torch(
                cuda_version="13.0",
                capability=(6, 1),
                device_name="NVIDIA GeForce GTX 1080 Ti",
            ),
        )
        is_mismatch, detail = _pascal_cu130_mismatch()
        assert is_mismatch is True
        assert "1080 Ti" in detail
        assert "6.1" in detail
        assert "13.0" in detail

    def test_titan_x_pascal_on_cu130_is_flagged(self, monkeypatch):
        _patch_torch(
            monkeypatch,
            _fake_torch(
                cuda_version="13.0",
                capability=(6, 1),
                device_name="NVIDIA TITAN X (Pascal)",
            ),
        )
        is_mismatch, _ = _pascal_cu130_mismatch()
        assert is_mismatch is True

    def test_maxwell_gtx_980_on_cu130_is_flagged(self, monkeypatch):
        """Maxwell (cc 5.x) is even older than Pascal — same outcome."""
        _patch_torch(
            monkeypatch,
            _fake_torch(
                cuda_version="13.0",
                capability=(5, 2),
                device_name="NVIDIA GeForce GTX 980",
            ),
        )
        is_mismatch, _ = _pascal_cu130_mismatch()
        assert is_mismatch is True

    def test_cu131_future_wheel_also_flags_pascal(self, monkeypatch):
        """If a future cu131 wheel ever ships, the prefix check still
        catches it for Pascal users."""
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version="13.1", capability=(6, 1)),
        )
        is_mismatch, _ = _pascal_cu130_mismatch()
        assert is_mismatch is True


# ── False-positive guards (must NOT flag) ──────────────────────────


class TestMustNotFlag:

    def test_rtx_4090_on_cu130_is_not_flagged(self, monkeypatch):
        """RTX 4090 (cc 8.9) is fully supported by cu130 wheels."""
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version="13.0", capability=(8, 9)),
        )
        is_mismatch, _ = _pascal_cu130_mismatch()
        assert is_mismatch is False

    def test_rtx_5090_on_cu130_is_not_flagged(self, monkeypatch):
        """RTX 5090 (Blackwell, cc 10.x) — explicitly the dev box.
        Must never false-positive here or I just bricked my own machine."""
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version="13.0", capability=(10, 0)),
        )
        is_mismatch, _ = _pascal_cu130_mismatch()
        assert is_mismatch is False

    def test_rtx_2060_on_cu130_is_not_flagged(self, monkeypatch):
        """RTX 2060 (Turing, cc 7.5) is the lowest supported card."""
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version="13.0", capability=(7, 5)),
        )
        is_mismatch, _ = _pascal_cu130_mismatch()
        assert is_mismatch is False

    def test_pascal_on_cu128_is_not_flagged(self, monkeypatch):
        """Post-1.9.2 happy path: Pascal user did the full reinstall
        and is now on cu128. Should never re-flag them."""
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version="12.8", capability=(6, 1)),
        )
        is_mismatch, _ = _pascal_cu130_mismatch()
        assert is_mismatch is False

    def test_pascal_on_cu126_is_not_flagged(self, monkeypatch):
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version="12.6", capability=(6, 1)),
        )
        is_mismatch, _ = _pascal_cu130_mismatch()
        assert is_mismatch is False

    def test_cpu_only_torch_is_not_flagged(self, monkeypatch):
        """A CPU-only build (cuda_version=None) is a different
        diagnostic — this helper must not double-flag it."""
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version=None, cuda_available=False),
        )
        is_mismatch, _ = _pascal_cu130_mismatch()
        assert is_mismatch is False

    def test_no_nvidia_gpu_present_is_not_flagged(self, monkeypatch):
        """Driver/torch installed but no card visible — different
        diagnostic territory, not Pascal mismatch."""
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version="13.0", cuda_available=False),
        )
        is_mismatch, _ = _pascal_cu130_mismatch()
        assert is_mismatch is False

    def test_zero_devices_is_not_flagged(self, monkeypatch):
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version="13.0", device_count=0),
        )
        is_mismatch, _ = _pascal_cu130_mismatch()
        assert is_mismatch is False


# ── Defensive failure modes (must NOT raise) ───────────────────────


class TestNeverRaises:
    """Every layer of the helper is wrapped in try/except. Verify each
    failure mode degrades silently to ``(False, '')`` instead of
    propagating an exception. If any of these tests fail, ``startup``
    will start crashing for users in the corresponding state."""

    def test_torch_import_failure_returns_false(self, monkeypatch):
        # Setting sys.modules["torch"] = None makes ``import torch``
        # raise ``ImportError: import of torch halted; None in sys.modules``
        # which is the cleanest way to simulate a broken/absent torch
        # without touching the real package on disk.
        monkeypatch.setitem(sys.modules, "torch", None)
        is_mismatch, detail = _pascal_cu130_mismatch()
        assert is_mismatch is False
        assert detail == ""

    def test_version_cuda_raise_returns_false(self, monkeypatch):
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version="13.0", raise_on="version.cuda"),
        )
        assert _pascal_cu130_mismatch() == (False, "")

    def test_is_available_raise_returns_false(self, monkeypatch):
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version="13.0", raise_on="is_available"),
        )
        assert _pascal_cu130_mismatch() == (False, "")

    def test_device_count_raise_returns_false(self, monkeypatch):
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version="13.0", raise_on="device_count"),
        )
        assert _pascal_cu130_mismatch() == (False, "")

    def test_get_device_capability_raise_returns_false(self, monkeypatch):
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version="13.0", raise_on="get_device_capability"),
        )
        assert _pascal_cu130_mismatch() == (False, "")

    def test_get_device_name_raise_returns_false(self, monkeypatch):
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version="13.0", raise_on="get_device_name"),
        )
        assert _pascal_cu130_mismatch() == (False, "")

    def test_capability_returns_unexpected_shape(self, monkeypatch):
        """Some torch builds historically returned strings or weird
        objects. Helper should refuse to interpret them."""
        fake = _fake_torch(cuda_version="13.0")
        fake.cuda.get_device_capability = lambda _idx: "(6, 1)"  # str, not tuple
        _patch_torch(monkeypatch, fake)
        assert _pascal_cu130_mismatch() == (False, "")


# ── Diagnostic registration ────────────────────────────────────────


class TestDiagnosticRegistered:

    def test_pascal_diagnostic_exists_in_registry(self):
        diag = next(
            (d for d in _DIAGNOSTICS if d.id == "pascal-cu130-mismatch"),
            None,
        )
        assert diag is not None
        # Pattern is intentionally a no-match — the diagnostic must
        # never fire from error-text matching, only from the
        # startup-time helper.
        assert diag.pattern.search("CUDA error: no kernel image is available") is None
        assert diag.pattern.search("anything at all") is None

    def test_steps_mention_full_installer_url(self):
        diag = next(d for d in _DIAGNOSTICS if d.id == "pascal-cu130-mismatch")
        joined = " ".join(diag.steps).lower()
        assert "github.com" in joined
        assert "releases" in joined
        assert "installer" in joined


# ── End-to-end through run_startup_diagnostics ─────────────────────


class TestRunStartupDiagnosticsIntegration:

    def test_pascal_user_gets_diagnostic_at_startup(self, monkeypatch):
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version="13.0", capability=(6, 1)),
        )
        # Stub validate_ffmpeg_install so we don't drag the ffmpeg
        # subsystem into this test.
        import backend.ffmpeg_tools as ff

        def _ok():
            class R:
                ok = True
                ffmpeg_path = "stub"
                message = ""

            return R()

        monkeypatch.setattr(ff, "validate_ffmpeg_install", _ok)

        issues = run_startup_diagnostics("cuda")
        ids = [i.diagnostic.id for i in issues]
        assert "pascal-cu130-mismatch" in ids

    def test_rtx_user_does_not_get_pascal_diagnostic(self, monkeypatch):
        _patch_torch(
            monkeypatch,
            _fake_torch(cuda_version="13.0", capability=(8, 9)),
        )
        import backend.ffmpeg_tools as ff

        def _ok():
            class R:
                ok = True
                ffmpeg_path = "stub"
                message = ""

            return R()

        monkeypatch.setattr(ff, "validate_ffmpeg_install", _ok)

        issues = run_startup_diagnostics("cuda")
        ids = [i.diagnostic.id for i in issues]
        assert "pascal-cu130-mismatch" not in ids

    def test_helper_exception_does_not_break_startup(self, monkeypatch):
        """If _pascal_cu130_mismatch ever raises despite its own
        guards, the call site swallows the exception so startup keeps
        running. This is the last line of defense."""
        with patch(
            "ui.widgets.diagnostic_checks._pascal_cu130_mismatch",
            side_effect=RuntimeError("everything is on fire"),
        ):
            import backend.ffmpeg_tools as ff

            def _ok():
                class R:
                    ok = True
                    ffmpeg_path = "stub"
                    message = ""

                return R()

            monkeypatch.setattr(ff, "validate_ffmpeg_install", _ok)
            # Must not raise.
            issues = run_startup_diagnostics("cuda")
            ids = [i.diagnostic.id for i in issues]
            assert "pascal-cu130-mismatch" not in ids

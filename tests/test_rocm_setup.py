"""Tests for ROCm helper and startup path (mocked; no AMD GPU required)."""

from __future__ import annotations

import builtins
import logging
import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_device_utils_and_main_loggers() -> None:
    """Avoid handler leakage across tests (stderr handlers, etc.)."""
    import device_utils as du
    import main

    du.logger.handlers.clear()
    du.logger.setLevel(logging.NOTSET)
    du.logger.propagate = True
    logging.getLogger(main.__name__).handlers.clear()
    yield
    du.logger.handlers.clear()
    logging.getLogger(main.__name__).handlers.clear()


@pytest.mark.rocm
def test_setup_rocm_env_no_hip_no_env_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    import torch

    with patch.object(torch.version, "hip", None):
        from device_utils import setup_rocm_env

        setup_rocm_env()
    assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ


@pytest.mark.rocm
def test_setup_rocm_env_with_hip_sets_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    import torch

    with patch.object(torch.version, "hip", "6.0"):
        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(torch.cuda, "get_device_name", return_value="AMD test"):
                from device_utils import setup_rocm_env

                setup_rocm_env()
    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0"


@pytest.mark.rocm
def test_setup_rocm_env_hip_preserves_existing_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    import torch

    with patch.object(torch.version, "hip", "6.0"):
        with patch.object(torch.cuda, "is_available", return_value=False):
            from device_utils import setup_rocm_env

            setup_rocm_env()
    assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"


@pytest.mark.rocm
def test_try_setup_rocm_env_import_error_is_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "device_utils":
            raise ImportError("simulated missing module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import main

    main._try_setup_rocm_env()


@pytest.mark.rocm
def test_try_setup_rocm_env_inner_failure_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import main

    with caplog.at_level(logging.DEBUG, logger="main"):
        with patch("device_utils.setup_rocm_env", side_effect=RuntimeError("boom")):
            main._try_setup_rocm_env()
    assert any("ROCm environment setup skipped" in r.getMessage() for r in caplog.records)


@pytest.mark.rocm
def test_setup_rocm_env_hip_writes_to_stderr(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    import torch

    import device_utils as du

    du.logger.handlers.clear()
    with patch.object(torch.version, "hip", "6.0"):
        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(torch.cuda, "get_device_name", return_value="AMD test"):
                du.setup_rocm_env()
    err = capsys.readouterr().err
    assert "ROCm:" in err

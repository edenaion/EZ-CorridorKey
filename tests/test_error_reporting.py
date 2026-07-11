"""Tests for backend.error_reporting."""
import json
import sys
import types
import uuid

import pytest

from backend import error_reporting as er


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch, tmp_path):
    monkeypatch.setattr(er, "_install_id_cache", None)
    monkeypatch.setattr(er, "_crash_reporting_active", False)
    monkeypatch.setattr(er, "_config_dir", lambda: str(tmp_path / "cfg"))
    yield


class TestScrubText:
    def test_windows_username(self):
        s = r"C:\Users\FlavioZ\AppData\Local\app\file.log"
        out = er.scrub_text(s)
        assert "FlavioZ" not in out
        assert r"C:\Users\<user>" in out

    def test_linux_and_mac_usernames(self):
        assert "bob" not in er.scrub_text("/home/bob/x.log")
        assert "carol" not in er.scrub_text("/Users/carol/x.log")

    def test_media_basenames_redacted(self):
        s = r"Failed to read C:\proj\My Wedding Video.mp4 frame"
        out = er.scrub_text(s)
        assert "Wedding" not in out
        assert "<media>.mp4" in out

    def test_app_generated_names_kept(self):
        for name in ("frame_000123.exr", "clip.json", "session_260101.log"):
            assert name in er.scrub_text(f"path/{name} ok"), name

    def test_project_and_clip_segments(self):
        s = r"N:\data\Projects\260710_wedding\clips\bride_close\Frames\frame_000001.exr"
        out = er.scrub_text(s)
        assert "wedding" not in out
        assert "bride" not in out
        assert "frame_000001.exr" in out

    def test_idempotent(self):
        cases = [
            r"C:\Users\Someone\Projects\secret\clips\shot1\a.mp4",
            r"C:/Users/user/Downloads/PF0044 - Copia.mp4 -> dest",
            "standalone My Take 07.mov in prose",
        ]
        for s in cases:
            once = er.scrub_text(s)
            assert er.scrub_text(once) == once, s
            assert "<<" not in once, s

    def test_empty(self):
        assert er.scrub_text("") == ""


class TestScrubEvent:
    def test_recursive_walk(self):
        event = {
            "message": r"C:\Users\Someone\clip.mp4",
            "extra": {"bundle": ["/home/bob/take01.mov", {"k": "/Users/x/y.png"}]},
            "server_name": "MY-PC",
            "number": 5,
        }
        out = er.scrub_event(event)
        blob = json.dumps(out)
        assert "Someone" not in blob
        assert "bob" not in blob
        assert "take01" not in blob
        assert out["server_name"] == ""
        assert out["number"] == 5

    def test_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(er, "scrub_text", lambda s: 1 / 0)
        assert er.scrub_event({"message": "x"}) is None


class TestInstallId:
    def test_minted_once_and_stable(self):
        a = er.get_install_id()
        b = er.get_install_id()
        assert a == b
        uuid.UUID(a)

    def test_survives_reread_from_disk(self, monkeypatch):
        a = er.get_install_id()
        monkeypatch.setattr(er, "_install_id_cache", None)
        assert er.get_install_id() == a

    def test_disk_failure_falls_back_in_memory(self, monkeypatch):
        monkeypatch.setattr(er, "_config_dir",
                            lambda: "\\\\?\\invalid\0dir")
        value = er.get_install_id()
        uuid.UUID(value)


class TestCollectTags:
    def _fake_torch(self, cuda: bool, version="2.9.1+cu128"):
        torch = types.SimpleNamespace()
        torch.__version__ = version
        torch.version = types.SimpleNamespace(cuda="12.8" if cuda else None)
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: cuda,
            get_device_capability=lambda i: (12, 0),
            get_device_name=lambda i: "NVIDIA GeForce RTX 5090",
        )
        return torch

    def test_with_cuda_wheel_suffix_preserved(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", self._fake_torch(True))
        tags = er.collect_tags({"name": "RTX 5090", "total_gb": 31.8}, "inference")
        assert tags["torch_version"] == "2.9.1+cu128"
        assert tags["cuda_available"] == "true"
        assert tags["gpu_arch"] == "sm_120"
        assert tags["vram_gb"] == "32"
        assert tags["stage"] == "inference"
        assert tags["product"] == er.PRODUCT

    def test_without_cuda(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "torch", self._fake_torch(False, "2.9.1+xpu"))
        tags = er.collect_tags(None, "runtime")
        assert tags["torch_version"] == "2.9.1+xpu"
        assert tags["cuda_available"] == "false"
        assert tags["gpu_arch"] == "none"

    def test_invalid_stage_maps_to_runtime(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", self._fake_torch(False))
        assert er.collect_tags(None, "banana")["stage"] == "runtime"

    def test_ffmpeg_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", self._fake_torch(False))
        import backend.ffmpeg_tools.discovery as disc
        monkeypatch.setattr(
            disc, "validate_ffmpeg_install",
            lambda require_probe=False: types.SimpleNamespace(
                ffmpeg_version=None),
        )
        assert er.collect_tags(None, "runtime")["ffmpeg_version"] == "missing"


class _FakeScope:
    def __init__(self, client=None):
        self.client = client
        self.user = None
        self.tags = {}
        self.extras = {}

    def set_user(self, u):
        self.user = u

    def set_tag(self, k, v):
        self.tags[k] = v

    def set_extra(self, k, v):
        self.extras[k] = v


class _FakeClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.options = kwargs
        self.flushed = None
        self.closed = False
        self.captured = []
        _FakeClient.instances.append(self)

    def capture_event(self, event, hint=None, scope=None):
        # Mirrors the real contract: events send ONLY through the client
        # they were built for. Scope.capture_* resolves the global client
        # and must not be used by send_report.
        self.captured.append((event, hint, scope))
        return "event-id"

    def flush(self, timeout=None):
        self.flushed = timeout

    def close(self, timeout=None):
        self.closed = True


class TestSendReport:
    def _install_fake_sdk(self, monkeypatch):
        _FakeClient.instances = []
        scopes = []

        def make_scope(client=None):
            s = _FakeScope(client)
            scopes.append(s)
            return s

        utils = types.SimpleNamespace(
            event_from_exception=lambda exc, client_options=None: (
                {"exception": {"values": [{"value": str(exc)}]},
                 "level": "error"},
                {"exc_info": exc},
            ),
        )
        fake = types.SimpleNamespace(
            Client=_FakeClient, Scope=make_scope, utils=utils)
        monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
        monkeypatch.setitem(sys.modules, "sentry_sdk.utils", utils)
        return scopes

    def test_send_uses_isolated_client_and_hygiene_kwargs(self, monkeypatch):
        scopes = self._install_fake_sdk(monkeypatch)
        ok = er.send_report("line one\nERROR bad thing happened", "runtime")
        assert ok
        client = _FakeClient.instances[0]
        kw = client.kwargs
        assert kw["default_integrations"] is False
        assert kw["traces_sample_rate"] is None
        assert kw["auto_session_tracking"] is False
        assert kw["send_default_pii"] is False
        assert kw["send_client_reports"] is False
        assert kw["include_local_variables"] is False
        assert kw["max_breadcrumbs"] == 0
        assert client.flushed == 5.0
        assert client.closed
        # Event must go through THIS client, carrying the prepared scope
        assert len(client.captured) == 1
        event, hint, scope = client.captured[0]
        assert "bad thing" in event["message"]
        assert event["level"] == "error"
        uuid.UUID(scope.user["id"])
        assert scope.tags["product"] == er.PRODUCT
        assert scopes[0] is scope

    def test_send_with_exception(self, monkeypatch):
        self._install_fake_sdk(monkeypatch)
        try:
            raise ValueError("boom")
        except ValueError as e:
            ok = er.send_report("bundle", "updater", exc_info=e)
        assert ok
        event, hint, scope = _FakeClient.instances[0].captured[0]
        assert "boom" in str(event["exception"])
        assert hint is not None
        assert scope.tags["stage"] == "updater"

    def test_missing_sdk_returns_false(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentry_sdk", None)
        assert er.send_report("bundle", "runtime") is False

    def test_internal_failure_never_raises(self, monkeypatch):
        self._install_fake_sdk(monkeypatch)
        monkeypatch.setattr(er, "collect_tags", lambda *a: 1 / 0)
        assert er.send_report("bundle", "runtime") is False


class TestCrashReportingGates:
    def test_toggle_off_no_init(self, monkeypatch):
        monkeypatch.setattr(er, "is_crash_reporting_enabled", lambda: False)
        monkeypatch.setattr(er.sys, "frozen", True, raising=False)
        assert er.init_crash_reporting() is False

    def test_dev_build_no_init_even_when_on(self, monkeypatch):
        monkeypatch.setattr(er, "is_crash_reporting_enabled", lambda: True)
        monkeypatch.delattr(er.sys, "frozen", raising=False)
        assert er.init_crash_reporting() is False

    def test_capture_stage_noop_when_inactive(self, monkeypatch):
        called = []
        fake = types.SimpleNamespace(
            capture_exception=lambda e: called.append(e))
        monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
        er.capture_stage_exception("updater", RuntimeError("x"))
        assert called == []

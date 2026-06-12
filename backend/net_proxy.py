"""Proxy sanitation for model downloads.

huggingface_hub 1.x uses httpx, which rejects socks4 proxy URLs outright
("Unknown scheme for proxy URL") and needs the socksio package for socks5.
Local proxy clients in the V2Ray/Xray family serve socks4 and socks5 on the
same inbound port, so a socks4 proxy can be upgraded to socks5 in place for
the duration of a download.

Used as a context manager around the model download call sites:

    with sanitized_proxy_env():
        hf_hub_download(...)

Behavior by user class:
☼ No proxy configured: nothing matches, exact no-op.
☼ http/https proxy: untouched.
☼ socks5 proxy: untouched (socksio ships with the app so it works).
☼ socks4/socks4a proxy: scheme upgraded to socks5 when socksio is
  available, otherwise the variable is removed for the duration of the
  call. Originals are always restored.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_PROXY_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)

_SOCKS4_SCHEMES = ("socks4://", "socks4a://")

# Substrings that identify a proxy-configuration failure raised by httpx
# or httpcore somewhere below huggingface_hub.
_PROXY_ERROR_SIGNATURES = (
    "unknown scheme for proxy url",
    "socksio",
    "socks proxy",
)


def _socks5_available() -> bool:
    try:
        return importlib.util.find_spec("socksio") is not None
    except Exception:
        return False


@contextmanager
def sanitized_proxy_env():
    """Temporarily rewrite socks4 proxy env vars so httpx can use them.

    Never raises: on any inspection error the environment is left as-is.
    Restores every touched variable on exit, including after exceptions.
    """
    original: dict[str, str] = {}
    try:
        upgrade = _socks5_available()
        for var in _PROXY_VARS:
            value = os.environ.get(var)
            if not value:
                continue
            lowered = value.strip().lower()
            if not lowered.startswith(_SOCKS4_SCHEMES):
                continue
            original[var] = value
            if upgrade:
                scheme_len = len(lowered.split("://", 1)[0])
                rewritten = "socks5://" + value.strip()[scheme_len + 3:]
                os.environ[var] = rewritten
                logger.warning(
                    "Proxy %s=%s uses socks4, which model downloads do not "
                    "support. Using %s for this download (V2Ray-family "
                    "clients accept both on the same port).",
                    var, value, rewritten,
                )
            else:
                del os.environ[var]
                logger.warning(
                    "Proxy %s=%s uses socks4, which model downloads do not "
                    "support, and socks5 support is unavailable. Ignoring "
                    "this proxy for the download.",
                    var, value,
                )
    except Exception:
        logger.exception("Proxy sanitation failed; leaving environment untouched")
    try:
        yield
    finally:
        for var, value in original.items():
            os.environ[var] = value


def friendly_proxy_error(exc: BaseException) -> str | None:
    """Return an actionable message if *exc* is a proxy configuration error.

    Returns None for anything else so callers re-raise unchanged.
    """
    text = str(exc).lower()
    if not any(sig in text for sig in _PROXY_ERROR_SIGNATURES):
        return None
    return (
        "Model download failed because of your system proxy settings "
        f"({exc}).\n\n"
        "Your proxy app is configured with a SOCKS proxy that Python "
        "cannot use. In your proxy client, switch the system proxy to "
        "socks5 or http, or disable the system proxy, then retry."
    )

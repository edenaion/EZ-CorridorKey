"""PyInstaller runtime hook — fix Triton path discovery for frozen builds.

Triton uses sysconfig.get_paths()["platlib"] to locate its bundled TCC
compiler, CUDA headers, and ptxas binary.  In frozen (PyInstaller) builds
sysconfig returns paths that don't exist, so Triton can't compile the
small C wrappers it needs (driver.c → cuda_utils.pyd, kernel launchers).

This hook sets three environment variables that Triton checks BEFORE
falling back to sysconfig, plus patches the one sysconfig path that has
no env-var override (Python include dir for Python.h).

Result: Triton compiles everything on first launch using the bundled TCC
compiler, then caches in ~/.triton/cache/ for instant subsequent starts.
Works on any NVIDIA GPU — no CUDA Toolkit required on the end-user machine.
"""

import logging
import os
import sys

_log = logging.getLogger("pyi_rth_triton")

if getattr(sys, "frozen", False):
    _meipass = sys._MEIPASS
    _log.info("Triton frozen-build hook: _MEIPASS = %s", _meipass)

    # ── 1. C Compiler ──────────────────────────────────────────────────
    # triton.runtime.build.get_cc() checks CC env var first.
    # TCC is bundled with triton-windows at triton/runtime/tcc/tcc.exe
    # and ships with cuda.def + python311.def for linking.
    _tcc = os.path.join(_meipass, "triton", "runtime", "tcc", "tcc.exe")
    if os.path.exists(_tcc):
        os.environ.setdefault("CC", _tcc)
        _log.info("CC -> %s", _tcc)
    else:
        _log.warning("TCC not found at %s", _tcc)

    # ── 2. CUDA headers + libs ─────────────────────────────────────────
    # triton.windows_utils.find_cuda() checks CUDA_PATH env var first.
    # check_and_find_cuda() expects: bin/ptxas.exe, include/cuda.h,
    # lib/x64/cuda.lib — all present in the bundled nvidia backend.
    _cuda = os.path.join(_meipass, "triton", "backends", "nvidia")
    if os.path.isdir(_cuda):
        os.environ.setdefault("CUDA_PATH", _cuda)
        _log.info("CUDA_PATH -> %s", _cuda)
        # Verify expected files exist
        for _check in ["bin/ptxas.exe", "include/cuda.h", "lib/x64/cuda.lib"]:
            _p = os.path.join(_cuda, _check)
            _log.info("  %s: %s", _check, "OK" if os.path.exists(_p) else "MISSING")
    else:
        _log.warning("CUDA dir not found at %s", _cuda)

    # ── 3. Writable cache directory ────────────────────────────────────
    # _MEIPASS is read-only (installer) or cluttered (portable).
    # Compiled modules go to the user's home — persists across updates,
    # shared between installer and portable builds on the same machine.
    _cache = os.path.join(os.path.expanduser("~"), ".triton", "cache")
    os.makedirs(_cache, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", _cache)
    _log.info("TRITON_CACHE_DIR -> %s", _cache)

    # ── 4. Python include headers (Python.h) ───────────────────────────
    # triton.runtime.build._build() uses sysconfig.get_paths()["include"]
    # to find Python.h.  No env-var override exists, so we patch sysconfig.
    # Only the "include" key is changed; all other paths are untouched.
    _py_inc = os.path.join(_meipass, "python_include")
    if os.path.isdir(_py_inc):
        import sysconfig as _sysconfig

        _orig_get_paths = _sysconfig.get_paths

        def _frozen_get_paths(scheme=None, vars=None, expand=True):
            paths = _orig_get_paths(scheme, vars, expand)
            paths["include"] = _py_inc
            return paths

        _sysconfig.get_paths = _frozen_get_paths
        _log.info("sysconfig.get_paths['include'] -> %s", _py_inc)
    else:
        _log.warning("Python include dir not found at %s", _py_inc)

    # ── 5. Backend entry points ────────────────────────────────────────
    # Triton discovers backends via importlib.metadata entry_points.
    # If triton_windows .dist-info is missing, it finds 0 backends →
    # "0 active drivers" error.  Register nvidia backend as fallback.
    try:
        if sys.version_info >= (3, 10):
            from importlib.metadata import entry_points as _ep
        else:
            from importlib_metadata import entry_points as _ep
        _triton_eps = _ep().select(group="triton.backends")
        _names = [e.name for e in _triton_eps]
        if "nvidia" not in _names:
            _log.warning(
                "triton.backends entry points missing nvidia (found: %s), patching", _names
            )
            # Monkey-patch _discover_backends after triton is imported
            os.environ["_TRITON_PATCH_BACKENDS"] = "1"
        else:
            _log.info("triton.backends entry points OK: %s", _names)
    except Exception as _e:
        _log.warning("Entry points check failed: %s, will patch backends", _e)
        os.environ["_TRITON_PATCH_BACKENDS"] = "1"

    _log.info("Triton frozen-build hook complete")

"""Firewall regression tests for the in-app update zip builder.

Background
----------
The Windows in-app updater (see ``ui/main_window_mixins/settings_mixin.py``
``_run_frozen_update``) downloads ``EZ-CorridorKey-{ver}-Windows-x64.zip``
from GitHub Releases and xcopy's the contents over an existing install.
``xcopy`` only touches files that are inside the zip, so any directory
**omitted** from the zip is left untouched on the user's disk.

That exclusion is the entire reason existing 1.9.1 users keep their
heavy runtime (torch + CUDA + triton + torchvision + nvidia DLLs) when
they hit "Check for Updates". If one of those directories ever slips
OUT of ``build_update_zip.EXCLUDE_DIRS``, the small update zip would
start overwriting the user's working torch runtime with nothing (or
worse, a half-set of incompatible files), and every existing user
would be broken by the next release.

These tests codify that invariant. **Do not relax them casually.** If
you ever need to intentionally ship a torch upgrade, do it via a
separate "fat" update asset and a deliberate version-gated fetch —
never by removing entries from this list.
"""
from __future__ import annotations

import importlib.util
import os
import sys


def _load_build_update_zip():
    """Load the build_update_zip module without importing scripts.windows
    as a package (which may not exist). Uses importlib.util.spec so the
    test file works whether or not ``scripts`` is a package.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.normpath(
        os.path.join(here, "..", "scripts", "windows", "build_update_zip.py")
    )
    spec = importlib.util.spec_from_file_location(
        "build_update_zip_for_test", script_path
    )
    assert spec is not None and spec.loader is not None, script_path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestUpdateZipPreservesExistingTorchRuntime:
    """The small update zip must NOT carry torch/CUDA/triton/torchvision
    over existing installs. These are the load-bearing exclusions that
    keep 1.9.1 users' cu130 runtime intact when they update to a later
    code-only release."""

    def test_torch_directory_is_excluded(self):
        mod = _load_build_update_zip()
        assert "torch" in mod.EXCLUDE_DIRS, (
            "torch/ must stay in EXCLUDE_DIRS — removing it would "
            "make the small update zip overwrite existing users' "
            "working torch runtime with whatever is in the current "
            "build env. See test module docstring."
        )

    def test_torch_libs_directory_is_excluded(self):
        mod = _load_build_update_zip()
        assert "torch.libs" in mod.EXCLUDE_DIRS

    def test_torchvision_is_excluded(self):
        mod = _load_build_update_zip()
        assert "torchvision" in mod.EXCLUDE_DIRS
        assert "torchvision.libs" in mod.EXCLUDE_DIRS

    def test_triton_is_excluded(self):
        mod = _load_build_update_zip()
        assert "triton" in mod.EXCLUDE_DIRS

    def test_nvidia_cuda_dlls_excluded(self):
        mod = _load_build_update_zip()
        assert "nvidia" in mod.EXCLUDE_DIRS, (
            "nvidia/ holds the bundled CUDA runtime DLLs. If removed, "
            "the zip would carry the build machine's cuXXX dlls over "
            "the user's install and potentially mismatch their driver."
        )

    def test_entire_torch_stack_excluded_in_one_assertion(self):
        """Belt-and-suspenders: a single guard that fails loudly if
        *any* of the torch-runtime directories is missing. Easier to
        spot in CI output than five separate failures."""
        mod = _load_build_update_zip()
        required = {
            "torch",
            "torch.libs",
            "torchvision",
            "torchvision.libs",
            "triton",
            "nvidia",
        }
        missing = required - set(mod.EXCLUDE_DIRS)
        assert not missing, (
            f"EXCLUDE_DIRS is missing: {sorted(missing)}.\n"
            "These entries protect existing users' torch runtime from "
            "being overwritten by code-only updates. Re-read the test "
            "module docstring before removing any of them."
        )


class TestShouldExcludeFunction:
    """Exercise the ``should_exclude`` helper against paths we know
    *must* be excluded, so a future refactor can't silently break the
    predicate while leaving EXCLUDE_DIRS intact."""

    def test_torch_dll_is_excluded(self):
        mod = _load_build_update_zip()
        from pathlib import Path
        assert mod.should_exclude(
            Path("_internal") / "torch" / "lib" / "torch_cpu.dll"
        )

    def test_nvidia_cuda_dll_is_excluded(self):
        mod = _load_build_update_zip()
        from pathlib import Path
        assert mod.should_exclude(
            Path("_internal") / "nvidia" / "cuda_runtime" / "bin" / "cudart64_12.dll"
        )

    def test_triton_so_is_excluded(self):
        mod = _load_build_update_zip()
        from pathlib import Path
        assert mod.should_exclude(
            Path("_internal") / "triton" / "runtime" / "libtriton.so"
        )

    def test_app_python_file_is_NOT_excluded(self):
        """Regression guard in the other direction: ordinary app code
        must still be shipped in the update zip."""
        mod = _load_build_update_zip()
        from pathlib import Path
        assert not mod.should_exclude(
            Path("_internal") / "ui" / "widgets" / "preview_viewport.py"
        )
        assert not mod.should_exclude(
            Path("_internal") / "backend" / "service" / "core.py"
        )

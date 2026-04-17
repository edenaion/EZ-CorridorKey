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

import pytest

# The in-app update zip is Windows-only. The build script lives under
# scripts/windows/ which is gitignored and only present on Windows dev
# checkouts. Skip the whole module on non-Windows so Mac / Linux test runs
# stay green without masking real regressions on Windows.
pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or not os.path.exists(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "scripts", "windows", "build_update_zip.py",
        )
    ),
    reason="build_update_zip.py is Windows-only and not present on this checkout",
)


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


class TestDistInfoFoldersExcluded:
    """Regression guard for v1.9.2 rc1: the skinny zip leaked
    ``pillow-12.1.1.dist-info`` alongside an otherwise-excluded PIL
    folder. Modern Pillow reads ``__version__`` from
    ``importlib.metadata`` at import time, so dropping a newer
    dist-info onto a user's disk while leaving the old ``_imaging.pyd``
    in place blows up with "Core version X / Pillow version Y".

    Every package in ``EXCLUDE_DIRS`` that has a PyPI dist-info MUST
    also be filtered at the dist-info level. These tests codify that.
    """

    def test_pillow_distinfo_is_excluded(self):
        """Direct regression for the 1.9.2 rc1 launch crash."""
        mod = _load_build_update_zip()
        from pathlib import Path
        assert mod.should_exclude(
            Path("_internal") / "pillow-12.1.1.dist-info" / "METADATA"
        )
        assert mod.should_exclude(
            Path("_internal") / "pillow-12.1.1.dist-info" / "RECORD"
        )

    def test_torch_distinfo_is_excluded(self):
        mod = _load_build_update_zip()
        from pathlib import Path
        assert mod.should_exclude(
            Path("_internal") / "torch-2.9.1+cu128.dist-info" / "METADATA"
        )

    def test_torchvision_distinfo_is_excluded(self):
        mod = _load_build_update_zip()
        from pathlib import Path
        assert mod.should_exclude(
            Path("_internal") / "torchvision-0.24.1+cu128.dist-info" / "METADATA"
        )

    def test_triton_windows_distinfo_is_excluded(self):
        """triton-windows is the Windows dist name for the triton import."""
        mod = _load_build_update_zip()
        from pathlib import Path
        assert mod.should_exclude(
            Path("_internal") / "triton_windows-3.5.1.post24.dist-info" / "METADATA"
        )

    def test_transformers_distinfo_is_excluded(self):
        mod = _load_build_update_zip()
        from pathlib import Path
        assert mod.should_exclude(
            Path("_internal") / "transformers-5.5.3.dist-info" / "METADATA"
        )

    def test_timm_distinfo_is_excluded(self):
        mod = _load_build_update_zip()
        from pathlib import Path
        assert mod.should_exclude(
            Path("_internal") / "timm-1.0.24.dist-info" / "METADATA"
        )

    def test_huggingface_hub_distinfo_is_excluded(self):
        mod = _load_build_update_zip()
        from pathlib import Path
        assert mod.should_exclude(
            Path("_internal") / "huggingface_hub-1.10.1.dist-info" / "METADATA"
        )

    def test_safetensors_distinfo_is_excluded(self):
        mod = _load_build_update_zip()
        from pathlib import Path
        assert mod.should_exclude(
            Path("_internal") / "safetensors-0.7.0.dist-info" / "METADATA"
        )

    def test_tokenizers_distinfo_is_excluded(self):
        mod = _load_build_update_zip()
        from pathlib import Path
        assert mod.should_exclude(
            Path("_internal") / "tokenizers-0.22.2.dist-info" / "METADATA"
        )

    def test_non_excluded_distinfo_is_NOT_excluded(self):
        """Regression guard the other way: dist-info folders for packages
        that ARE shipped in the skinny zip (numpy app-level code is not in
        EXCLUDE_DIRS — only numpy.libs is — but we still exclude the numpy
        dist-info to keep it in sync with numpy.libs on disk) should still
        be filtered. Meanwhile, dist-infos for genuinely code-only packages
        that ship in the zip must pass through.
        """
        mod = _load_build_update_zip()
        from pathlib import Path
        # numpy dist-info IS in EXCLUDE_DISTRIBUTIONS — version must track numpy.libs
        assert mod.should_exclude(
            Path("_internal") / "numpy-2.4.3.dist-info" / "METADATA"
        )
        # packaging ships in the zip and must NOT be filtered
        assert not mod.should_exclude(
            Path("_internal") / "packaging-26.0.dist-info" / "METADATA"
        )
        # click ships in the zip and must NOT be filtered
        assert not mod.should_exclude(
            Path("_internal") / "click-8.3.2.dist-info" / "METADATA"
        )

    def test_canonical_dist_name_handles_local_versions(self):
        """torch ships with a local version tag like ``2.9.1+cu128``.
        The dist-info folder is ``torch-2.9.1+cu128.dist-info``. The
        parser must strip the version (including the ``+local``) and
        return ``torch``."""
        mod = _load_build_update_zip()
        assert mod._canonical_dist_name("torch-2.9.1+cu128.dist-info") == "torch"
        assert mod._canonical_dist_name("pillow-12.1.1.dist-info") == "pillow"
        assert mod._canonical_dist_name(
            "triton_windows-3.5.1.post24.dist-info"
        ) == "triton_windows"
        assert mod._canonical_dist_name("not-a-dist-info") is None
        assert mod._canonical_dist_name("noversion.dist-info") is None

    def test_entire_distribution_set_is_codified(self):
        """Belt-and-suspenders: every package in EXCLUDE_DISTRIBUTIONS must
        exist in the set so a refactor can't silently drop one. This test
        is what you look at in CI when the set shrinks unexpectedly."""
        mod = _load_build_update_zip()
        required = {
            "torch",
            "torchvision",
            "triton_windows",
            "pillow",
            "transformers",
            "timm",
            "huggingface_hub",
            "safetensors",
            "tokenizers",
        }
        missing = required - set(mod.EXCLUDE_DISTRIBUTIONS)
        assert not missing, (
            f"EXCLUDE_DISTRIBUTIONS is missing: {sorted(missing)}.\n"
            "These dist-info folders must stay filtered to prevent "
            "ABI mismatches against stale .pyd files on user disks. "
            "See the 1.9.2 rc1 Pillow regression for context."
        )


class TestBundledModelCheckpointsExcluded:
    """Model checkpoints that ship in the full installer must stay OUT of
    the small update zip. Otherwise a code-only update would clobber
    existing users' weights with whatever happened to be on the build
    machine, or worse, overwrite a newer .pth the user pulled manually.

    This mirrors the torch-runtime firewall philosophy: the update zip is
    code-only, and anything heavy (torch, CUDA, model weights) is handled
    exclusively by the full installer.
    """

    def test_corridorkey_pth_is_excluded(self):
        """CorridorKey.pth is bundled into the installer and must never
        be re-shipped via the update zip. If this ever fails, a code-only
        update will overwrite user weights on extraction."""
        mod = _load_build_update_zip()
        from pathlib import Path
        assert mod.should_exclude(
            Path("_internal") / "CorridorKeyModule" / "checkpoints" / "CorridorKey_v1.0.pth"
        )

    def test_corridorkey_checkpoints_prefix_in_exclude_set(self):
        """Belt-and-suspenders: the prefix itself must be in EXCLUDE_PATH_PREFIXES,
        not just filtered by some other rule, so a future refactor of
        ``should_exclude`` cannot silently let checkpoints through."""
        mod = _load_build_update_zip()
        import os
        assert (
            os.path.join("_internal", "CorridorKeyModule", "checkpoints")
            in mod.EXCLUDE_PATH_PREFIXES
        )

    def test_birefnet_weights_still_excluded(self):
        """BiRefNet checkpoints are also installer-bundled. Regression
        guard for the long-standing exclusion that never had a test."""
        mod = _load_build_update_zip()
        from pathlib import Path
        assert mod.should_exclude(
            Path("_internal") / "modules" / "BiRefNetModule"
            / "checkpoints" / "BiRefNet-matting" / "model.safetensors"
        )

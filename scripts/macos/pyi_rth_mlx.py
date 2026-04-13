"""PyInstaller runtime hook — MLX Metal kernel path fixup for frozen builds.

MLX's native library (libmlx.dylib) looks for compiled Metal shaders
(.metallib files) at a path relative to its package directory.  In a frozen
PyInstaller bundle the directory layout differs from a normal pip install,
so we help MLX find the kernels by:

1. Setting MLX_METAL_PATH to point at the bundled metallib location.
2. Ensuring the mlx package directory is on sys.path so that
   ``import mlx.core`` can find the native extension and its resources.
"""

import os
import sys
import glob


def _fixup_mlx_paths():
    """Locate bundled .metallib files and set MLX_METAL_PATH."""
    # In a PyInstaller bundle, _MEIPASS is the temp extraction directory
    bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))

    # Search common locations where PyInstaller places MLX files
    search_dirs = [
        os.path.join(bundle_dir, "mlx", "lib"),
        os.path.join(bundle_dir, "mlx"),
        os.path.join(bundle_dir, "mlx_metal"),
        bundle_dir,
    ]

    # Find .metallib files
    metallib_path = None
    for search_dir in search_dirs:
        pattern = os.path.join(search_dir, "*.metallib")
        matches = glob.glob(pattern)
        if matches:
            metallib_path = matches[0]
            break

    # Also do a recursive search if not found in expected locations
    if metallib_path is None:
        for match in glob.glob(os.path.join(bundle_dir, "**", "*.metallib"), recursive=True):
            metallib_path = match
            break

    if metallib_path:
        metallib_dir = os.path.dirname(metallib_path)
        # MLX checks MLX_METAL_PATH for the directory containing metallib files
        os.environ["MLX_METAL_PATH"] = metallib_dir
        # Also set the individual file path variant some versions use
        os.environ["MLX_METALLIB_PATH"] = metallib_path

    # Ensure mlx package directory is findable
    mlx_dir = os.path.join(bundle_dir, "mlx")
    if os.path.isdir(mlx_dir) and bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)


def _prewarm_mlx():
    """Import mlx.core on the main thread to initialize Metal device.

    MLX's Metal backend must be initialized on the main thread.
    If the first import happens on a worker thread (e.g. GPUJobWorker),
    Metal device creation deadlocks.  By importing here — in a runtime
    hook that runs on the main thread before any app code — we ensure
    the Metal context is ready for worker threads.
    """
    try:
        import mlx.core as mx

        # Force Metal device initialization by running a trivial op
        mx.eval(mx.zeros(1))
    except Exception:
        pass  # MLX not available — app will fall back to torch


_fixup_mlx_paths()
_prewarm_mlx()

"""PyInstaller runtime hook — cv2 fixes for frozen builds.

1. Set OPENCV_IO_ENABLE_OPENEXR before cv2 initializes (must be in env
   before the native module loads, or EXR codec stays disabled forever).

2. Pre-import cv2.pyd to avoid recursive import: opencv-python-headless
   ships a bootstrap __init__.py that calls importlib.import_module("cv2")
   which in a frozen build finds itself again → infinite recursion.
"""

import importlib.util
import os
import sys

# Must be set before cv2 native module initializes
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def _preload_cv2_native():
    for p in sys.path:
        cv2_dir = os.path.join(p, "cv2")
        if not os.path.isdir(cv2_dir):
            continue
        # Look for cv2.pyd (Windows) or cv2.abi3.pyd
        for f in os.listdir(cv2_dir):
            if f.startswith("cv2") and f.endswith(".pyd"):
                pyd = os.path.join(cv2_dir, f)
                break
        else:
            continue
        spec = importlib.util.spec_from_file_location("cv2", pyd, submodule_search_locations=[])
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["cv2"] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                del sys.modules["cv2"]
            return


_preload_cv2_native()

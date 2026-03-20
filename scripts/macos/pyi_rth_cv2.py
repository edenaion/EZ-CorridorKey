"""PyInstaller runtime hook — fix cv2 recursive import in macOS .app bundle.

opencv-python-headless ships a bootstrap __init__.py that calls
importlib.import_module("cv2") to load the native extension. In a PyInstaller
BUNDLE this finds the bootstrap __init__.py again, causing infinite recursion.

Fix: pre-import the native cv2.abi3.so directly into sys.modules["cv2"] before
the bootstrap __init__.py gets a chance to run.
"""
import importlib.util
import os
import sys


def _preload_cv2_native():
    for p in sys.path:
        so = os.path.join(p, "cv2", "cv2.abi3.so")
        if not os.path.isfile(so):
            # Also check for cv2.cpython-*.so naming
            cv2_dir = os.path.join(p, "cv2")
            if os.path.isdir(cv2_dir):
                for f in os.listdir(cv2_dir):
                    if f.startswith("cv2") and f.endswith(".so"):
                        so = os.path.join(cv2_dir, f)
                        break
                else:
                    continue
            else:
                continue
        if os.path.isfile(so):
            spec = importlib.util.spec_from_file_location("cv2", so,
                                                           submodule_search_locations=[])
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["cv2"] = mod
                try:
                    spec.loader.exec_module(mod)
                except Exception:
                    del sys.modules["cv2"]
                return


_preload_cv2_native()

"""Module-level helpers and standalone functions for the service package."""

from __future__ import annotations

import importlib
import logging
import os
import sys
import warnings


logger = logging.getLogger(__name__)


def _configure_runtime_warnings() -> None:
    """Hide non-actionable NVML deprecation chatter during startup/runtime checks."""
    warnings.filterwarnings(
        "ignore",
        message=r"The pynvml package is deprecated\..*",
        category=FutureWarning,
    )


_configure_runtime_warnings()

# Project paths — frozen-build aware
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _import_matanyone2_processor_class():
    """Import MatAnyone2Processor from either the new or legacy module layout.

    Supported layouts:
    - modules/MatAnyone2Module/wrapper.py  -> modules.MatAnyone2Module.wrapper
    - MatAnyone2Module/wrapper.py          -> MatAnyone2Module.wrapper
    """
    candidates = (
        ("modules.MatAnyone2Module.wrapper", BASE_DIR),
        ("MatAnyone2Module.wrapper", os.path.join(BASE_DIR, "modules")),
        ("MatAnyone2Module.wrapper", BASE_DIR),
    )
    missing_roots = (
        os.path.join(BASE_DIR, "modules", "MatAnyone2Module"),
        os.path.join(BASE_DIR, "MatAnyone2Module"),
    )
    last_error: ModuleNotFoundError | None = None

    for module_name, path_entry in candidates:
        if path_entry and path_entry not in sys.path:
            sys.path.insert(0, path_entry)
        try:
            module = importlib.import_module(module_name)
            return module.MatAnyone2Processor
        except ModuleNotFoundError as exc:
            # Only fall through on the layout module itself missing. If an inner
            # dependency is missing, bubble that error up unchanged.
            if exc.name not in {
                "modules",
                "modules.MatAnyone2Module",
                "modules.MatAnyone2Module.wrapper",
                "MatAnyone2Module",
                "MatAnyone2Module.wrapper",
            }:
                raise
            last_error = exc

    expected = " or ".join(missing_roots)
    raise ModuleNotFoundError(f"MatAnyone2 module not found. Expected {expected}") from last_error


def export_masks_headless(clip) -> str | None:
    """Export annotation masks for a clip without requiring the UI viewer.

    Loads annotations from disk, determines frame dimensions from the first
    input frame, and calls AnnotationModel.export_masks().

    Returns:
        Path to VideoMamaMaskHint directory, or None if no annotations found.
    """
    from ui.widgets.annotation_overlay import AnnotationModel
    from ..frame_io import read_image_frame

    model = AnnotationModel()
    model.load(clip.root_path)

    if not model.has_annotations():
        return None

    if clip.input_asset is None or clip.input_asset.asset_type != "sequence":
        return None

    frame_files = clip.input_asset.get_frame_files()
    if not frame_files:
        return None

    stems = [os.path.splitext(f)[0] for f in frame_files]

    # Get dimensions from first input frame
    first_path = os.path.join(clip.input_asset.path, frame_files[0])
    sample = read_image_frame(first_path)
    if sample is None:
        return None
    h, w = sample.shape[:2]

    # Respect in/out range
    start_idx = 0
    if clip.in_out_range:
        lo = clip.in_out_range.in_point
        hi = clip.in_out_range.out_point
        stems = stems[lo : hi + 1]
        start_idx = lo

    return model.export_masks(clip.root_path, stems, w, h, start_index=start_idx)

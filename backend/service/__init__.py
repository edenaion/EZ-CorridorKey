"""Backend service package — re-exports the public API."""

import importlib
import cv2
from .core import CorridorKeyService, InferenceParams, OutputConfig, FrameResult
from .model_manager import _ActiveModel
from .helpers import export_masks_headless, _import_matanyone2_processor_class
from backend.frame_io import read_image_frame, read_mask_frame, write_exr
from backend.annotation_prompts import load_annotation_prompt_frames

__all__ = [
    "CorridorKeyService",
    "InferenceParams",
    "OutputConfig",
    "FrameResult",
    "_ActiveModel",
    "export_masks_headless",
    "_import_matanyone2_processor_class",
    "read_image_frame",
    "read_mask_frame",
    "write_exr",
    "load_annotation_prompt_frames",
    "cv2",
    "importlib",
]

"""Specialized inference pipelines — GVM, SAM2, VideoMaMa, MatAnyone2, BiRefNet.

Split into two sub-modules for maintainability:
- pipelines_auto: GVM, BiRefNet (auto-alpha, no annotations)
- pipelines_guided: SAM2, VideoMaMa, MatAnyone2 (guided, need masks/annotations)

This module provides the combined PipelinesMixin for backward compatibility.
"""

from __future__ import annotations

from .pipelines_auto import AutoPipelinesMixin
from .pipelines_guided import GuidedPipelinesMixin


class PipelinesMixin(AutoPipelinesMixin, GuidedPipelinesMixin):
    """Combined mixin providing all specialized inference pipelines."""

    pass

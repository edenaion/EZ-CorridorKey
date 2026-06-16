"""Translated display labels for backend enum values.

Backend enums keep stable English .value strings (state machine, logs,
session files). Anything user-facing goes through these helpers so the
labels translate. Written as literal QCoreApplication.translate() calls
because lupdate only extracts literal context + source pairs.
"""
from __future__ import annotations

from PySide6.QtCore import QCoreApplication

from backend.clip_state import ClipState


def state_display_name(state: ClipState) -> str:
    """Return the translated UI label for a clip state badge or info line."""
    labels = {
        ClipState.EXTRACTING: QCoreApplication.translate("ClipState", "EXTRACTING"),
        ClipState.RAW: QCoreApplication.translate("ClipState", "RAW"),
        ClipState.MASKED: QCoreApplication.translate("ClipState", "MASKED"),
        ClipState.READY: QCoreApplication.translate("ClipState", "READY"),
        ClipState.COMPLETE: QCoreApplication.translate("ClipState", "COMPLETE"),
        ClipState.ERROR: QCoreApplication.translate("ClipState", "ERROR"),
    }
    return labels.get(state, state.value)


def job_type_display_name(job_type) -> str:
    """Return the translated UI label for a GPU job type.

    Single source of truth — the status bar, queue panel, and worker
    mixin all render job types through this. Model names (GVM, BiRefNet,
    VideoMaMa, MatAnyone2) stay untranslated by policy.
    """
    from backend.job_queue import JobType
    labels = {
        JobType.INFERENCE: QCoreApplication.translate("JobType", "Inference"),
        JobType.GVM_ALPHA: "GVM Auto",
        JobType.BIREFNET_ALPHA: "BiRefNet",
        JobType.SAM2_PREVIEW: QCoreApplication.translate("JobType", "Track Preview"),
        JobType.SAM2_TRACK: QCoreApplication.translate("JobType", "Track Mask"),
        JobType.VIDEOMAMA_ALPHA: "VideoMaMa",
        JobType.MATANYONE2_ALPHA: "MatAnyone2",
        JobType.PREVIEW_REPROCESS: QCoreApplication.translate("JobType", "Preview"),
    }
    return labels.get(job_type, QCoreApplication.translate("JobType", "Pipeline"))


# Backend services emit progress phase strings in English (the backend is
# Qt-free and must stay that way). The UI translates known phrases at the
# display boundary; unknown strings pass through untranslated so new
# backend phases degrade gracefully instead of breaking.
def backend_status_text(message: str) -> str:
    """Translate a backend status/phase message for the status bar."""
    if not message:
        return message
    known = {
        "Loading model...": QCoreApplication.translate("BackendStatus", "Loading model..."),
        "Loading frames...": QCoreApplication.translate("BackendStatus", "Loading frames..."),
        "Loading masks...": QCoreApplication.translate("BackendStatus", "Loading masks..."),
        "Loading preview frame...": QCoreApplication.translate("BackendStatus", "Loading preview frame..."),
        "Loading first-frame mask...": QCoreApplication.translate("BackendStatus", "Loading first-frame mask..."),
        "Loading state dict...": QCoreApplication.translate("BackendStatus", "Loading state dict..."),
        "Loading checkpoint weights...": QCoreApplication.translate("BackendStatus", "Loading checkpoint weights..."),
        "Loading MatAnyone2 checkpoint...": QCoreApplication.translate("BackendStatus", "Loading MatAnyone2 checkpoint..."),
        "Loading MatAnyone2 model...": QCoreApplication.translate("BackendStatus", "Loading MatAnyone2 model..."),
        "Initializing model backbone...": QCoreApplication.translate("BackendStatus", "Initializing model backbone..."),
        "Moving model to GPU...": QCoreApplication.translate("BackendStatus", "Moving model to GPU..."),
        "Patching attention blocks...": QCoreApplication.translate("BackendStatus", "Patching attention blocks..."),
        "Compiling model (first run may take a minute)...": QCoreApplication.translate("BackendStatus", "Compiling model (first run may take a minute)..."),
        "Compiling (first frame may take a minute)...": QCoreApplication.translate("BackendStatus", "Compiling (first frame may take a minute)..."),
        "Model ready": QCoreApplication.translate("BackendStatus", "Model ready"),
        "BiRefNet model ready": QCoreApplication.translate("BackendStatus", "BiRefNet model ready"),
        "MatAnyone2 model ready": QCoreApplication.translate("BackendStatus", "MatAnyone2 model ready"),
        "Running SAM2 tracker...": QCoreApplication.translate("BackendStatus", "Running SAM2 tracker..."),
        "Running BiRefNet inference...": QCoreApplication.translate("BackendStatus", "Running BiRefNet inference..."),
        "Running MatAnyone2 inference...": QCoreApplication.translate("BackendStatus", "Running MatAnyone2 inference..."),
        "Previewing SAM2 on annotated frame...": QCoreApplication.translate("BackendStatus", "Previewing SAM2 on annotated frame..."),
        "Finalizing alpha hints...": QCoreApplication.translate("BackendStatus", "Finalizing alpha hints..."),
        "Releasing Python references...": QCoreApplication.translate("BackendStatus", "Releasing Python references..."),
        "Waiting for CUDA to finish...": QCoreApplication.translate("BackendStatus", "Waiting for CUDA to finish..."),
        "Clearing CUDA cache...": QCoreApplication.translate("BackendStatus", "Clearing CUDA cache..."),
        "UNet forward pass": QCoreApplication.translate("BackendStatus", "UNet forward pass"),
        "VAE encode": QCoreApplication.translate("BackendStatus", "VAE encode"),
        "CLIP encode": QCoreApplication.translate("BackendStatus", "CLIP encode"),
    }
    return known.get(message, message)

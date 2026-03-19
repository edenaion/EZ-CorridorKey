"""
inference_engine_amd.py — Universal AMD/DirectML engine for CorridorKey.

Drop-in replacement for CorridorKeyEngine on AMD hardware.
Interface is identical so backend.py / main.py need no changes.
Note from toowyred: This is not meant to be run directly, but just in case. :3

Architecture (AMD path):
  - Backbone (Hiera encoder + decoders): CPU via ORT
    Hiera's windowed attention exceeds DirectML's 8-dimension tensor limit.
    Running on CPU is the only stable path on Windows without ROCm.
  - Refiner (CNN): AMD GPU via DirectML ORT
    Pure 2D convolutions — compatible with any RDNA GPU (RX 6000+).

Requires pre-exported ONNX files — run export_to_onnx.py once first.
"""

import logging
import os
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from .core import color_utils as cu
from .inference_engine import INFERENCE_DEFAULTS


def _get_amd_vram_gb() -> float:
    """Best-effort VRAM size probe for AMD GPUs on Windows."""
    try:
        import subprocess
        result = subprocess.run(
            ["powershell", "-Command",
             "(Get-WmiObject Win32_VideoController | Select-Object -First 1).AdapterRAM"],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        return 0.0


class CorridorKeyEngineAMD:
    """
    AMD/DirectML inference engine.
    Identical interface to CorridorKeyEngine (CUDA).
    """

    def __init__(self, backbone_onnx, refiner_onnx, img_size=2048, on_status=None):
        import onnxruntime as ort

        self._on_status = on_status
        self.img_size = img_size

        self._status("Initializing AMD ORT engine...")
        vram = _get_amd_vram_gb()
        if vram > 0:
            logger.info(f"AMD GPU detected: ~{vram:.1f} GB VRAM")

        # Backbone: CPU only (Hiera attention exceeds DML tensor rank limit)
        self._status("Loading backbone (CPU)...")
        if not os.path.isfile(backbone_onnx):
            raise FileNotFoundError(
                f"Backbone ONNX not found: {backbone_onnx}\n"
                "Run export_to_onnx.py first."
            )
        cpu_opts = ort.SessionOptions()
        cpu_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
        self._backbone = ort.InferenceSession(
            backbone_onnx,
            sess_options=cpu_opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info(f"Backbone: CPU ORT <- {backbone_onnx}")

        # Refiner: DirectML (pure 2D convolutions, no rank issues)
        self._status("Loading refiner (DirectML)...")
        if not os.path.isfile(refiner_onnx):
            raise FileNotFoundError(
                f"Refiner ONNX not found: {refiner_onnx}\n"
                "Run export_to_onnx.py first."
            )
        dml_opts = ort.SessionOptions()
        dml_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._refiner = ort.InferenceSession(
            refiner_onnx,
            sess_options=dml_opts,
            providers=[
                ("DmlExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ],
        )
        active = self._refiner.get_providers()
        if "DmlExecutionProvider" in active:
            self._status("AMD GPU active (DirectML)")
            logger.info("Refiner: DmlExecutionProvider (GPU)")
        else:
            self._status("Warning: DirectML unavailable, refiner on CPU")
            logger.warning("DmlExecutionProvider not available — refiner on CPU")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self._status("AMD engine ready.")

    def _status(self, msg: str) -> None:
        logger.info(msg)
        if self._on_status:
            self._on_status(msg)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    _D = INFERENCE_DEFAULTS

    def process_frame(
        self,
        image,
        mask_linear,
        refiner_scale=_D["refiner_scale"],
        input_is_linear=False,
        fg_is_straight=True,
        despill_strength=_D["despill_strength"],
        auto_despeckle=_D["auto_despeckle"],
        despeckle_size=_D["despeckle_size"],
        despeckle_dilation=_D["despeckle_dilation"],
        despeckle_blur=_D["despeckle_blur"],
        source_passthrough=_D["source_passthrough"],
        edge_erode_px=_D["edge_erode_px"],
        edge_blur_px=_D["edge_blur_px"],
    ):
        """
        Identical interface to CorridorKeyEngine.process_frame.
        Returns: dict with keys 'alpha', 'fg', 'comp', 'processed'
        """
        t0 = time.monotonic()

        # 1. Normalise inputs
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0
        h, w = image.shape[:2]
        if mask_linear.ndim == 2:
            mask_linear = mask_linear[:, :, np.newaxis]

        # 2. Resize to model resolution
        if input_is_linear:
            img_lin = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img_resized = cu.linear_to_srgb(img_lin)
        else:
            img_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        mask_resized = cv2.resize(mask_linear, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

        # 3. ImageNet normalisation → [1, 4, H, W] float32
        img_norm = (img_resized - self.mean) / self.std
        inp_np   = np.concatenate([img_norm, mask_resized], axis=-1)
        inp_ort  = inp_np.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        # 4. Backbone on CPU → coarse preds + upsampled logits
        alpha_coarse, fg_coarse, alpha_logits_up, fg_logits_up = self._backbone.run(
            ["alpha_coarse", "fg_coarse", "alpha_logits_up", "fg_logits_up"],
            {"input": inp_ort},
        )

        # 5. Refiner on AMD GPU → delta logits
        coarse_ort   = np.concatenate([alpha_coarse, fg_coarse], axis=1)
        delta_logits, = self._refiner.run(
            ["delta_logits"],
            {"rgb": inp_ort[:, :3], "coarse_pred": coarse_ort},
        )
        delta_logits *= float(refiner_scale)

        # 6. Residual addition in logit space (matches GreenFormer.forward)
        alpha_final = self._sigmoid(alpha_logits_up + delta_logits[:, 0:1])
        fg_final    = self._sigmoid(fg_logits_up    + delta_logits[:, 1:4])

        # 7. To HWC numpy, resize back to original resolution
        res_alpha = cv2.resize(alpha_final[0].transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LANCZOS4)
        res_fg    = cv2.resize(fg_final[0].transpose(1, 2, 0),    (w, h), interpolation=cv2.INTER_LANCZOS4)
        if res_alpha.ndim == 2:
            res_alpha = res_alpha[:, :, np.newaxis]

        # 8. Post-processing (identical to CorridorKeyEngine)
        if source_passthrough:
            original_srgb = cu.linear_to_srgb(image) if input_is_linear else image
            res_fg = cu.source_passthrough(original_srgb, res_fg, res_alpha,
                                            erode_px=edge_erode_px, blur_px=edge_blur_px)

        processed_alpha = (cu.clean_matte(res_alpha, area_threshold=despeckle_size,
                                           dilation=despeckle_dilation, blur_size=despeckle_blur)
                           if auto_despeckle else res_alpha)

        fg_despilled     = cu.despill(res_fg, green_limit_mode='average', strength=despill_strength)
        fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
        fg_premul_lin    = cu.premultiply(fg_despilled_lin, processed_alpha)
        processed_rgba   = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)

        bg_srgb  = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
        bg_lin   = cu.srgb_to_linear(bg_srgb)
        comp_lin = (cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
                    if fg_is_straight
                    else cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha))
        comp_srgb = cu.linear_to_srgb(comp_lin)

        logger.debug(f"process_frame (AMD): {h}x{w} in {time.monotonic() - t0:.3f}s")

        return {
            'alpha':     res_alpha,
            'fg':        res_fg,
            'comp':      comp_srgb,
            'processed': processed_rgba,
        }

"""Right panel — alpha generation, inference parameters, and output config.

Sections follow the pipeline order:
1. Alpha Generation — GVM Auto (automatic) or VideoMaMa (manual masks)
2. Inference — Color Space, Despill, Despeckle, Refiner, Live Preview
3. Output — FG/Matte/Comp/Processed format selectors
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QCheckBox, QSpinBox, QPushButton, QGroupBox,
    QScrollArea,
)
from PySide6.QtCore import Qt, Signal, QEvent

from backend import InferenceParams, OutputConfig


class ParameterPanel(QWidget):
    """Right panel with all inference parameter controls."""

    params_changed = Signal()  # emitted when any parameter changes
    gvm_requested = Signal()      # GVM AUTO button clicked
    videomama_requested = Signal() # VIDEOMAMA button clicked
    export_masks_requested = Signal()  # Export annotation masks clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("paramPanel")
        self.setMinimumWidth(240)

        # Signal suppression flag (Codex: block signals during session restore)
        self._suppress_signals = False

        # Wrap all controls in a scroll area so they never squish below
        # their natural size — panel scrolls instead of compressing.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("paramPanelScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QScrollArea.NoFrame)

        inner = QWidget()
        inner.setObjectName("paramPanelInner")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # ── ALPHA GENERATION section (Step 1) ──
        alpha_group = QGroupBox("ALPHA GENERATION")
        alpha_layout = QVBoxLayout(alpha_group)
        alpha_layout.setSpacing(8)

        self._gvm_btn = QPushButton("GVM AUTO")
        self._gvm_btn.setEnabled(False)
        self._gvm_btn.setToolTip(
            "Auto-generate alpha hint for the entire clip.\n"
            "Uses GVM to predict foreground/background separation.\n"
            "Available when clip is in RAW state (frames extracted)."
        )
        self._gvm_btn.clicked.connect(self.gvm_requested.emit)
        alpha_layout.addWidget(self._gvm_btn)

        or_label = QLabel("— or —")
        or_label.setAlignment(Qt.AlignCenter)
        or_label.setStyleSheet("color: #808070; font-size: 11px;")
        alpha_layout.addWidget(or_label)

        self._videomama_btn = QPushButton("VIDEOMAMA")
        self._videomama_btn.setEnabled(False)
        self._videomama_btn.setToolTip(
            "Generate alpha from painted annotations via VideoMaMa.\n\n"
            "1. Paint masks on frames with hotkey 1 (green/foreground)\n"
            "   or 2 (red/background)\n"
            "2. Click Export Masks to save annotations\n"
            "3. Click VIDEOMAMA to generate alpha\n\n"
            "Tip: Annotate keyframes where the subject changes shape\n"
            "or overlaps complex backgrounds — more frames = better results."
        )
        self._videomama_btn.clicked.connect(self.videomama_requested.emit)
        alpha_layout.addWidget(self._videomama_btn)

        self._export_masks_btn = QPushButton("EXPORT MASKS")
        self._export_masks_btn.setEnabled(False)
        self._export_masks_btn.setToolTip(
            "Export painted annotations as VideoMamaMaskHint.\n"
            "Then click VIDEOMAMA to generate alpha from your masks."
        )
        self._export_masks_btn.clicked.connect(self.export_masks_requested.emit)
        alpha_layout.addWidget(self._export_masks_btn)

        self._annotation_info = QLabel("")
        self._annotation_info.setStyleSheet("color: #808070; font-size: 10px;")
        alpha_layout.addWidget(self._annotation_info)

        layout.addWidget(alpha_group)

        # ── INFERENCE section (Step 2) ──
        inf_group = QGroupBox("INFERENCE")
        inf_layout = QVBoxLayout(inf_group)
        inf_layout.setSpacing(8)

        # Color Space
        cs_row = QHBoxLayout()
        cs_label = QLabel("Color Space")
        cs_label.setFixedWidth(80)
        cs_row.addWidget(cs_label)
        self._color_space = QComboBox()
        self._color_space.addItems(["sRGB", "Linear"])
        self._color_space.setToolTip(
            "Input color space.\n"
            "sRGB: standard gamma-corrected footage (most cameras).\n"
            "Linear: raw linear-light footage (EXR sequences, CG renders)."
        )
        self._color_space.currentIndexChanged.connect(self._emit_changed)
        cs_row.addWidget(self._color_space, 1)
        inf_layout.addLayout(cs_row)

        # Despill Strength (slider 0-10 → 0.0-1.0)
        self._despill_label = QLabel("Despill: 1.0")
        inf_layout.addWidget(self._despill_label)
        self._despill_slider = QSlider(Qt.Horizontal)
        self._despill_slider.setRange(0, 10)
        self._despill_slider.setValue(10)
        self._despill_slider.setToolTip(
            "Green spill removal strength (0.0–1.0).\n"
            "Removes green color bleed from hair, skin, and edges.\n"
            "1.0 = full despill, 0.0 = no despill (keep original colors)."
        )
        self._despill_slider.valueChanged.connect(self._on_despill_changed)
        inf_layout.addWidget(self._despill_slider)

        # Despeckle toggle + size
        despeckle_row = QHBoxLayout()
        self._despeckle_check = QCheckBox("Despeckle")
        self._despeckle_check.setChecked(True)
        self._despeckle_check.setToolTip(
            "Automatic garbage matte — removes small floating noise\n"
            "and speckles from the alpha by discarding isolated regions\n"
            "smaller than the size threshold."
        )
        self._despeckle_check.stateChanged.connect(self._on_despeckle_toggled)
        despeckle_row.addWidget(self._despeckle_check)
        self._despeckle_size = QSpinBox()
        self._despeckle_size.setRange(50, 2000)
        self._despeckle_size.setValue(400)
        self._despeckle_size.setSuffix("px")
        self._despeckle_size.setToolTip(
            "Minimum area (in pixels) for a region to survive.\n"
            "Isolated alpha blobs smaller than this are removed.\n"
            "Lower = keep more detail, higher = cleaner matte."
        )
        self._despeckle_size.valueChanged.connect(self._emit_changed)
        despeckle_row.addWidget(self._despeckle_size, 1)
        inf_layout.addLayout(despeckle_row)

        # Refiner Scale (slider 0-30 → 0.0-3.0)
        self._refiner_label = QLabel("Refiner: 1.0")
        inf_layout.addWidget(self._refiner_label)
        self._refiner_slider = QSlider(Qt.Horizontal)
        self._refiner_slider.setRange(0, 30)
        self._refiner_slider.setValue(10)
        self._refiner_slider.setToolTip(
            "Edge refinement strength (0.0–3.0).\n"
            "Scales the CNN refiner's edge corrections.\n"
            "1.0 = default, 0.0 = backbone only (no refinement),\n"
            "higher = sharper edges but may introduce artifacts."
        )
        self._refiner_slider.valueChanged.connect(self._on_refiner_changed)
        inf_layout.addWidget(self._refiner_slider)

        # Live Preview toggle
        self._live_preview = QCheckBox("Live Preview")
        self._live_preview.setToolTip("Reprocess current frame on parameter change (requires loaded engine)")
        inf_layout.addWidget(self._live_preview)

        layout.addWidget(inf_group)

        # ── OUTPUT FORMAT section (Step 3) ──
        out_group = QGroupBox("OUTPUT")
        out_layout = QVBoxLayout(out_group)
        out_layout.setSpacing(6)

        # FG
        fg_row = QHBoxLayout()
        self._fg_check = QCheckBox("FG")
        self._fg_check.setChecked(True)
        self._fg_check.setToolTip(
            "Foreground — despilled subject on black background.\n"
            "Green spill removed from hair and edges.\n"
            "Straight alpha (not premultiplied)."
        )
        fg_row.addWidget(self._fg_check, 1)
        self._fg_format = QComboBox()
        self._fg_format.addItems(["exr", "png"])
        self._fg_format.setFixedWidth(70)
        self._fg_format.setToolTip("EXR = 32-bit float (post-production).\nPNG = 8-bit (general use).")
        fg_row.addWidget(self._fg_format)
        out_layout.addLayout(fg_row)

        # Matte
        matte_row = QHBoxLayout()
        self._matte_check = QCheckBox("Matte")
        self._matte_check.setChecked(True)
        self._matte_check.setToolTip(
            "Alpha matte — grayscale transparency map.\n"
            "White = fully opaque, black = fully transparent.\n"
            "Use in compositing software for manual keying control."
        )
        matte_row.addWidget(self._matte_check, 1)
        self._matte_format = QComboBox()
        self._matte_format.addItems(["exr", "png"])
        self._matte_format.setFixedWidth(70)
        self._matte_format.setToolTip("EXR = 32-bit float (post-production).\nPNG = 8-bit (general use).")
        matte_row.addWidget(self._matte_format)
        out_layout.addLayout(matte_row)

        # Comp
        comp_row = QHBoxLayout()
        self._comp_check = QCheckBox("Comp")
        self._comp_check.setChecked(True)
        self._comp_check.setToolTip(
            "Composite — final keyed result over checkerboard.\n"
            "Best representation of the key quality.\n"
            "Colors match the original input faithfully."
        )
        comp_row.addWidget(self._comp_check, 1)
        self._comp_format = QComboBox()
        self._comp_format.addItems(["png", "exr"])
        self._comp_format.setFixedWidth(70)
        self._comp_format.setToolTip("PNG = 8-bit with transparency.\nEXR = 32-bit float (post-production).")
        comp_row.addWidget(self._comp_format)
        out_layout.addLayout(comp_row)

        # Processed
        proc_row = QHBoxLayout()
        self._proc_check = QCheckBox("Processed")
        self._proc_check.setChecked(True)
        self._proc_check.setToolTip(
            "Processed — production-ready RGBA (premultiplied, linear).\n"
            "Designed for import into compositing tools (Nuke, After Effects).\n"
            "Includes despill + garbage matte cleanup applied."
        )
        proc_row.addWidget(self._proc_check, 1)
        self._proc_format = QComboBox()
        self._proc_format.addItems(["exr", "png"])
        self._proc_format.setFixedWidth(70)
        self._proc_format.setToolTip("EXR = 32-bit float (recommended for Processed).\nPNG = 8-bit (lossy for premultiplied data).")
        proc_row.addWidget(self._proc_format)
        out_layout.addLayout(proc_row)

        layout.addWidget(out_group)

        layout.addStretch(1)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # Middle-click reset: map widget → (setter_callable, default_value)
        self._middle_click_defaults: dict[QWidget, tuple] = {
            self._despill_slider: (self._despill_slider.setValue, 10),      # 1.0
            self._refiner_slider: (self._refiner_slider.setValue, 10),      # 1.0
            self._despeckle_size: (self._despeckle_size.setValue, 400),      # 400px
        }
        for widget in self._middle_click_defaults:
            widget.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        """Middle-click resets a control to its default value."""
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.MiddleButton:
            if obj in self._middle_click_defaults:
                setter, default = self._middle_click_defaults[obj]
                setter(default)
                return True
        return super().eventFilter(obj, event)

    def _emit_changed(self) -> None:
        """Emit params_changed unless signals are suppressed."""
        if not self._suppress_signals:
            self.params_changed.emit()

    def _on_despeckle_toggled(self, state: int) -> None:
        self._emit_changed()

    def _on_despill_changed(self, value: int) -> None:
        display = value / 10.0
        self._despill_label.setText(f"Despill: {display:.1f}")
        self._emit_changed()

    def _on_refiner_changed(self, value: int) -> None:
        display = value / 10.0
        self._refiner_label.setText(f"Refiner: {display:.1f}")
        self._emit_changed()

    @property
    def live_preview_enabled(self) -> bool:
        return self._live_preview.isChecked()

    def get_params(self) -> InferenceParams:
        """Snapshot current parameter values into a frozen InferenceParams."""
        return InferenceParams(
            input_is_linear=self._color_space.currentIndex() == 1,
            despill_strength=self._despill_slider.value() / 10.0,
            auto_despeckle=self._despeckle_check.isChecked(),
            despeckle_size=self._despeckle_size.value(),
            despeckle_dilation=25,  # fixed default
            despeckle_blur=5,       # fixed default
            refiner_scale=self._refiner_slider.value() / 10.0,
        )

    def get_output_config(self) -> OutputConfig:
        """Snapshot current output format configuration."""
        return OutputConfig(
            fg_enabled=self._fg_check.isChecked(),
            fg_format=self._fg_format.currentText(),
            matte_enabled=self._matte_check.isChecked(),
            matte_format=self._matte_format.currentText(),
            comp_enabled=self._comp_check.isChecked(),
            comp_format=self._comp_format.currentText(),
            processed_enabled=self._proc_check.isChecked(),
            processed_format=self._proc_format.currentText(),
        )

    def set_params(self, params: InferenceParams) -> None:
        """Load parameter values (e.g. from a saved session).

        Suppresses signals during restore to prevent event storms (Codex).
        """
        self._suppress_signals = True
        try:
            self._color_space.setCurrentIndex(1 if params.input_is_linear else 0)
            self._despill_slider.setValue(int(params.despill_strength * 10))
            self._despeckle_check.setChecked(params.auto_despeckle)
            self._despeckle_size.setValue(params.despeckle_size)
            # despeckle_dilation / despeckle_blur: no longer exposed in UI (fixed defaults)
            self._refiner_slider.setValue(int(params.refiner_scale * 10))
        finally:
            self._suppress_signals = False

    def set_output_config(self, config: OutputConfig) -> None:
        """Load output config values (e.g. from a saved session)."""
        self._suppress_signals = True
        try:
            self._fg_check.setChecked(config.fg_enabled)
            self._fg_format.setCurrentText(config.fg_format)
            self._matte_check.setChecked(config.matte_enabled)
            self._matte_format.setCurrentText(config.matte_format)
            self._comp_check.setChecked(config.comp_enabled)
            self._comp_format.setCurrentText(config.comp_format)
            self._proc_check.setChecked(config.processed_enabled)
            self._proc_format.setCurrentText(config.processed_format)
        finally:
            self._suppress_signals = False

    def set_gvm_enabled(self, enabled: bool) -> None:
        """Enable/disable GVM button based on clip state."""
        self._gvm_btn.setEnabled(enabled)

    def set_videomama_enabled(self, enabled: bool) -> None:
        """Enable/disable VideoMaMa button based on clip state."""
        self._videomama_btn.setEnabled(enabled)

    def set_annotation_info(self, annotated: int, total: int) -> None:
        """Update annotation frame counter."""
        if annotated > 0 and total > 0:
            self._annotation_info.setText(f"Annotated: {annotated} / {total} frames")
            self._export_masks_btn.setEnabled(True)
        else:
            self._annotation_info.setText("")
            self._export_masks_btn.setEnabled(False)

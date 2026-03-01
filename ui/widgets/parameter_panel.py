"""Right panel — inference parameters, output config, and alpha generation controls.

Provides sliders/controls for:
- Color Space (sRGB / Linear)
- Despill strength (0-10, maps to 0.0-1.0 internally)
- Despeckle toggle + size
- Refiner scale (0-30, maps to 0.0-3.0)
- Live Preview toggle (debounced single-frame reprocess)
- Output format options (FG/Matte/Comp/Processed, format selectors)
- Alpha generation buttons (GVM Auto, VideoMaMa)
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QCheckBox, QSpinBox, QPushButton, QGroupBox,
)
from PySide6.QtCore import Qt, Signal

from backend import InferenceParams, OutputConfig


class ParameterPanel(QWidget):
    """Right panel with all inference parameter controls."""

    params_changed = Signal()  # emitted when any parameter changes
    gvm_requested = Signal()      # GVM AUTO button clicked
    videomama_requested = Signal() # VIDEOMAMA button clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("paramPanel")
        self.setMinimumWidth(240)

        # Signal suppression flag (Codex: block signals during session restore)
        self._suppress_signals = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # ── INFERENCE section ──
        inf_group = QGroupBox("INFERENCE")
        inf_layout = QVBoxLayout(inf_group)
        inf_layout.setSpacing(8)

        # Color Space
        cs_row = QHBoxLayout()
        cs_row.addWidget(QLabel("Color Space"))
        cs_row.addStretch()
        self._color_space = QComboBox()
        self._color_space.addItems(["sRGB", "Linear"])
        self._color_space.setMinimumWidth(90)
        self._color_space.currentIndexChanged.connect(self._emit_changed)
        cs_row.addWidget(self._color_space)
        inf_layout.addLayout(cs_row)

        # Despill Strength (slider 0-10 → 0.0-1.0)
        self._despill_label = QLabel("Despill: 1.0")
        inf_layout.addWidget(self._despill_label)
        self._despill_slider = QSlider(Qt.Horizontal)
        self._despill_slider.setRange(0, 10)
        self._despill_slider.setValue(10)
        self._despill_slider.valueChanged.connect(self._on_despill_changed)
        inf_layout.addWidget(self._despill_slider)

        # Despeckle toggle + size
        despeckle_row = QHBoxLayout()
        self._despeckle_check = QCheckBox("Despeckle")
        self._despeckle_check.setChecked(True)
        self._despeckle_check.stateChanged.connect(self._emit_changed)
        despeckle_row.addWidget(self._despeckle_check)

        despeckle_row.addStretch()
        self._despeckle_size = QSpinBox()
        self._despeckle_size.setRange(50, 2000)
        self._despeckle_size.setValue(400)
        self._despeckle_size.setSuffix("px")
        self._despeckle_size.setMinimumWidth(90)
        self._despeckle_size.valueChanged.connect(self._emit_changed)
        despeckle_row.addWidget(self._despeckle_size)
        inf_layout.addLayout(despeckle_row)

        # Refiner Scale (slider 0-30 → 0.0-3.0)
        self._refiner_label = QLabel("Refiner: 1.0")
        inf_layout.addWidget(self._refiner_label)
        self._refiner_slider = QSlider(Qt.Horizontal)
        self._refiner_slider.setRange(0, 30)
        self._refiner_slider.setValue(10)
        self._refiner_slider.valueChanged.connect(self._on_refiner_changed)
        inf_layout.addWidget(self._refiner_slider)

        # Live Preview toggle
        self._live_preview = QCheckBox("Live Preview")
        self._live_preview.setToolTip("Reprocess current frame on parameter change (requires loaded engine)")
        inf_layout.addWidget(self._live_preview)

        layout.addWidget(inf_group)

        # ── OUTPUT FORMAT section ──
        out_group = QGroupBox("OUTPUT")
        out_layout = QVBoxLayout(out_group)
        out_layout.setSpacing(6)

        # FG
        fg_row = QHBoxLayout()
        self._fg_check = QCheckBox("FG")
        self._fg_check.setChecked(True)
        fg_row.addWidget(self._fg_check)
        fg_row.addStretch()
        self._fg_format = QComboBox()
        self._fg_format.addItems(["exr", "png"])
        self._fg_format.setMinimumWidth(70)
        fg_row.addWidget(self._fg_format)
        out_layout.addLayout(fg_row)

        # Matte
        matte_row = QHBoxLayout()
        self._matte_check = QCheckBox("Matte")
        self._matte_check.setChecked(True)
        matte_row.addWidget(self._matte_check)
        matte_row.addStretch()
        self._matte_format = QComboBox()
        self._matte_format.addItems(["exr", "png"])
        self._matte_format.setMinimumWidth(70)
        matte_row.addWidget(self._matte_format)
        out_layout.addLayout(matte_row)

        # Comp
        comp_row = QHBoxLayout()
        self._comp_check = QCheckBox("Comp")
        self._comp_check.setChecked(True)
        comp_row.addWidget(self._comp_check)
        comp_row.addStretch()
        self._comp_format = QComboBox()
        self._comp_format.addItems(["png", "exr"])
        self._comp_format.setMinimumWidth(70)
        comp_row.addWidget(self._comp_format)
        out_layout.addLayout(comp_row)

        # Processed
        proc_row = QHBoxLayout()
        self._proc_check = QCheckBox("Processed")
        self._proc_check.setChecked(True)
        proc_row.addWidget(self._proc_check)
        proc_row.addStretch()
        self._proc_format = QComboBox()
        self._proc_format.addItems(["exr", "png"])
        self._proc_format.setMinimumWidth(70)
        proc_row.addWidget(self._proc_format)
        out_layout.addLayout(proc_row)

        layout.addWidget(out_group)

        # ── ALPHA GENERATION section ──
        alpha_group = QGroupBox("ALPHA GENERATION")
        alpha_layout = QVBoxLayout(alpha_group)
        alpha_layout.setSpacing(8)

        self._gvm_btn = QPushButton("GVM AUTO")
        self._gvm_btn.setEnabled(False)
        self._gvm_btn.setToolTip("Auto-generate alpha hint via GVM (RAW clips only)")
        self._gvm_btn.clicked.connect(self.gvm_requested.emit)
        alpha_layout.addWidget(self._gvm_btn)

        self._videomama_btn = QPushButton("VIDEOMAMA")
        self._videomama_btn.setEnabled(False)
        self._videomama_btn.setToolTip("Generate alpha from user mask via VideoMaMa (MASKED clips only)")
        self._videomama_btn.clicked.connect(self.videomama_requested.emit)
        alpha_layout.addWidget(self._videomama_btn)

        layout.addWidget(alpha_group)

        layout.addStretch(1)

    def _emit_changed(self) -> None:
        """Emit params_changed unless signals are suppressed."""
        if not self._suppress_signals:
            self.params_changed.emit()

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

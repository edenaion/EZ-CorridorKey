"""Right panel — inference parameters and alpha generation controls.

Provides sliders/controls for:
- Color Space (sRGB / Linear)
- Despill strength (0-10, maps to 0.0-1.0 internally)
- Despeckle toggle + size
- Refiner scale (0-30, maps to 0.0-3.0)
- Alpha generation buttons (GVM Auto, VideoMaMa) — Phase 2 wiring
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QCheckBox, QSpinBox, QPushButton, QGroupBox,
)
from PySide6.QtCore import Qt, Signal

from backend import InferenceParams


class ParameterPanel(QWidget):
    """Right panel with all inference parameter controls."""

    params_changed = Signal()  # emitted when any parameter changes
    gvm_requested = Signal()      # GVM AUTO button clicked
    videomama_requested = Signal() # VIDEOMAMA button clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("paramPanel")
        self.setMinimumWidth(240)

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
        self._color_space = QComboBox()
        self._color_space.addItems(["sRGB", "Linear"])
        self._color_space.currentIndexChanged.connect(lambda: self.params_changed.emit())
        cs_row.addWidget(self._color_space)
        inf_layout.addLayout(cs_row)

        # Despill Strength (slider 0-10 → 0.0-1.0)
        self._despill_label = QLabel("Despill: 10")
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
        self._despeckle_check.stateChanged.connect(lambda: self.params_changed.emit())
        despeckle_row.addWidget(self._despeckle_check)

        self._despeckle_size = QSpinBox()
        self._despeckle_size.setRange(50, 2000)
        self._despeckle_size.setValue(400)
        self._despeckle_size.setSuffix("px")
        self._despeckle_size.valueChanged.connect(lambda: self.params_changed.emit())
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

        layout.addWidget(inf_group)

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

    def _on_despill_changed(self, value: int) -> None:
        self._despill_label.setText(f"Despill: {value}")
        self.params_changed.emit()

    def _on_refiner_changed(self, value: int) -> None:
        display = value / 10.0
        self._refiner_label.setText(f"Refiner: {display:.1f}")
        self.params_changed.emit()

    def get_params(self) -> InferenceParams:
        """Snapshot current parameter values into a frozen InferenceParams."""
        return InferenceParams(
            input_is_linear=self._color_space.currentIndex() == 1,
            despill_strength=self._despill_slider.value() / 10.0,
            auto_despeckle=self._despeckle_check.isChecked(),
            despeckle_size=self._despeckle_size.value(),
            refiner_scale=self._refiner_slider.value() / 10.0,
        )

    def set_params(self, params: InferenceParams) -> None:
        """Load parameter values (e.g. from a saved session)."""
        self._color_space.setCurrentIndex(1 if params.input_is_linear else 0)
        self._despill_slider.setValue(int(params.despill_strength * 10))
        self._despeckle_check.setChecked(params.auto_despeckle)
        self._despeckle_size.setValue(params.despeckle_size)
        self._refiner_slider.setValue(int(params.refiner_scale * 10))

    def set_gvm_enabled(self, enabled: bool) -> None:
        """Enable/disable GVM button based on clip state."""
        self._gvm_btn.setEnabled(enabled)

    def set_videomama_enabled(self, enabled: bool) -> None:
        """Enable/disable VideoMaMa button based on clip state."""
        self._videomama_btn.setEnabled(enabled)

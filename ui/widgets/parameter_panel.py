"""Right panel — alpha generation, tracking, and output config."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QSlider,
    QComboBox, QCheckBox, QSpinBox, QPushButton, QGroupBox,
    QScrollArea,
)
from PySide6.QtCore import Qt, Signal, QEvent


def _no_scroll_wheel(widget: QWidget) -> QWidget:
    """Prevent mouse-wheel from changing value unless widget is focused.

    Users scrolling the panel accidentally change sliders/spinboxes.
    StrongFocus means the widget only accepts wheel events after a click.
    """
    widget.setFocusPolicy(Qt.StrongFocus)
    widget.installEventFilter(_WheelGuard.instance())
    return widget




class _WheelGuard(QWidget):
    """Event filter: ignores wheel on unfocused widgets, enforces single-step on focused sliders."""

    _singleton = None

    @classmethod
    def instance(cls):
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            if not obj.hasFocus():
                event.ignore()
                return True
            # Force exactly 1 singleStep per scroll notch on sliders
            if isinstance(obj, QSlider):
                delta = event.angleDelta().y()
                if delta > 0:
                    obj.setValue(obj.value() + obj.singleStep())
                elif delta < 0:
                    obj.setValue(obj.value() - obj.singleStep())
                return True
        return False

from backend import InferenceParams, OutputConfig


_COLOR_SPACE_TOOLTIP = (
    "How CorridorKey interprets the source before inference.\n"
    "The left INPUT viewer always shows this interpretation, so it should match "
    "what CorridorKey thinks your footage is.\n\n"
    "sRGB: standard gamma-corrected footage (most cameras, phone video, PNG/JPG).\n"
    "Linear: linear-light footage (true linear EXRs, CG renders).\n\n"
    "Changing this before Run Inference affects live preview and any future "
    "exports you generate.\n"
    "Changing it after files are already exported does not rewrite those files on "
    "disk; rerun inference to save new outputs.\n"
    "Auto-detected from format/metadata when possible, but you can override it if "
    "the INPUT viewer looks wrong."
)

_LIVE_PREVIEW_TOOLTIP = (
    "Instantly reprocess the current frame when you adjust Color Space, Despill, "
    "Refiner, or Despeckle.\n"
    "Requires a READY or COMPLETE clip with alpha hints.\n"
    "On a fresh launch, the first preview change may take a moment while the "
    "inference engine loads.\n"
    "Preview updates do not rewrite exported files on disk; rerun inference to "
    "save them."
)


class ParameterPanel(QWidget):
    """Right panel with all inference parameter controls."""

    params_changed = Signal()  # emitted when any parameter changes
    parallel_frames_changed = Signal(int)  # parallel engine count changed
    screen_color_changed = Signal(str)  # "green" or "blue" — UI accent swap
    gvm_requested = Signal()      # GVM AUTO button clicked
    birefnet_requested = Signal(str)  # BiRefNet button clicked, emits model variant name
    videomama_requested = Signal() # VIDEOMAMA button clicked
    matanyone2_requested = Signal()  # MatAnyone2 button clicked
    track_masks_requested = Signal()  # Track annotation prompts into dense masks
    import_alpha_requested = Signal()  # Import own AlphaHint folder
    import_vmama_mask_requested = Signal()  # Import mask for VideoMaMa bypass
    chroma_key_requested = Signal(dict)  # GENERATE chroma key, emits params dict
    chroma_key_preview = Signal()         # any chroma key param changed (live preview)
    eyedropper_requested = Signal(bool)   # toggle eyedropper mode on viewer

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
        alpha_group = QGroupBox(self.tr("ALPHA GENERATION"))
        alpha_layout = QVBoxLayout(alpha_group)
        alpha_layout.setSpacing(8)

        # -- Manual sub-section (Chroma Key) --
        manual_label = QLabel(self.tr("Manual"))
        manual_label.setAlignment(Qt.AlignCenter)
        manual_label.setStyleSheet("color: #A0A090; font-size: 10px; margin: 0px 0 2px 0;")
        alpha_layout.addWidget(manual_label)

        self._chroma_key_btn = QPushButton(self.tr("CHROMA KEY"))
        self._chroma_key_btn.setEnabled(False)
        self._chroma_key_btn.setCheckable(True)
        self._chroma_key_btn.setToolTip(
            self.tr(
                "Generate alpha hints using a traditional chroma keyer.\n"
                "Best for clean green/blue screen shots.\n"
                "No GPU or AI model required \u2014 instant processing.\n\n"
                "Click to expand parameters, then click GENERATE."
            )
        )
        self._chroma_key_btn.toggled.connect(self._on_chroma_key_toggled)
        alpha_layout.addWidget(self._chroma_key_btn)

        # Chroma key parameter container (hidden until button is toggled on)
        self._chroma_params_widget = QWidget()
        ck_layout = QVBoxLayout(self._chroma_params_widget)
        ck_layout.setContentsMargins(0, 4, 0, 4)
        ck_layout.setSpacing(4)

        # Eyedropper row: button + color swatch
        eyedropper_row = QHBoxLayout()
        eyedropper_row.setSpacing(4)
        self._eyedropper_btn = QPushButton(self.tr("\U0001F4A7 Pick Screen Color (E)"))
        self._eyedropper_btn.setToolTip(
            self.tr(
                "Click on the viewer to sample the screen color.\n"
                "Works on either the input or output viewport.\n"
                "Hotkey: E"
            )
        )
        self._eyedropper_btn.setCheckable(True)
        self._eyedropper_btn.toggled.connect(self._on_eyedropper_toggled)
        eyedropper_row.addWidget(self._eyedropper_btn, 1)

        self._color_swatch = QLabel("")
        self._color_swatch.setFixedSize(30, 24)
        self._color_swatch.setStyleSheet("background: #00B140; border: 1px solid #5A5940;")
        self._color_swatch.setToolTip(self.tr("Sampled screen color"))
        eyedropper_row.addWidget(self._color_swatch)
        ck_layout.addLayout(eyedropper_row)

        # Key Strength slider (0.1 - 3.0)
        self._ck_strength_label = QLabel(self.tr("Key Strength: 1.0"))
        ck_layout.addWidget(self._ck_strength_label)
        self._ck_strength = _no_scroll_wheel(QSlider(Qt.Horizontal))
        self._ck_strength.setRange(1, 100)  # 0.1 to 10.0 in steps of 0.1
        self._ck_strength.setValue(10)
        self._ck_strength.setToolTip(self.tr("How aggressively to key the screen color. Higher = more separation."))
        self._ck_strength.valueChanged.connect(
            lambda v: self._ck_strength_label.setText(self.tr("Key Strength: %s") % f"{v / 10:.1f}")
        )
        self._ck_strength.valueChanged.connect(lambda _: self.chroma_key_preview.emit())
        ck_layout.addWidget(self._ck_strength)

        # Clip Black slider (0.0 - 1.0)
        self._ck_clip_black_label = QLabel(self.tr("Clip Black: 0.0"))
        ck_layout.addWidget(self._ck_clip_black_label)
        self._ck_clip_black = _no_scroll_wheel(QSlider(Qt.Horizontal))
        self._ck_clip_black.setRange(0, 100)
        self._ck_clip_black.setValue(0)
        self._ck_clip_black.setToolTip(self.tr("Push near-transparent values to fully transparent.\nCleans up noise in background areas."))
        self._ck_clip_black.valueChanged.connect(
            lambda v: self._ck_clip_black_label.setText(self.tr("Clip Black: %s") % f"{v / 100:.2f}")
        )
        self._ck_clip_black.valueChanged.connect(lambda _: self.chroma_key_preview.emit())
        ck_layout.addWidget(self._ck_clip_black)

        # Clip White slider (0.0 - 1.0)
        self._ck_clip_white_label = QLabel(self.tr("Clip White: 1.0"))
        ck_layout.addWidget(self._ck_clip_white_label)
        self._ck_clip_white = _no_scroll_wheel(QSlider(Qt.Horizontal))
        self._ck_clip_white.setRange(0, 100)
        self._ck_clip_white.setValue(100)
        self._ck_clip_white.setToolTip(self.tr("Push near-opaque values to fully opaque.\nSolidifies the foreground core."))
        self._ck_clip_white.valueChanged.connect(
            lambda v: self._ck_clip_white_label.setText(self.tr("Clip White: %s") % f"{v / 100:.2f}")
        )
        self._ck_clip_white.valueChanged.connect(lambda _: self.chroma_key_preview.emit())
        ck_layout.addWidget(self._ck_clip_white)

        # Shrink/Grow row (stretch 1:1 to match BIREFNET / General HR split)
        sg_row = QHBoxLayout()
        sg_row.setSpacing(4)
        sg_row.addWidget(QLabel(self.tr("Shrink/Grow")), 1)
        self._ck_shrink_grow = _no_scroll_wheel(QSpinBox())
        self._ck_shrink_grow.setRange(-250, 250)
        self._ck_shrink_grow.setValue(0)
        self._ck_shrink_grow.setSuffix("px")
        self._ck_shrink_grow.setToolTip(self.tr("Erode (negative) or dilate (positive) the matte edge.\n0 = no change."))
        self._ck_shrink_grow.valueChanged.connect(lambda _: self.chroma_key_preview.emit())
        sg_row.addWidget(self._ck_shrink_grow, 1)
        ck_layout.addLayout(sg_row)

        # Edge Blur row (stretch 1:1 to match BIREFNET / General HR split)
        eb_row = QHBoxLayout()
        eb_row.setSpacing(4)
        eb_row.addWidget(QLabel(self.tr("Edge Blur")), 1)
        self._ck_edge_blur = _no_scroll_wheel(QSpinBox())
        self._ck_edge_blur.setRange(0, 50)
        self._ck_edge_blur.setValue(0)
        self._ck_edge_blur.setSuffix("px")
        self._ck_edge_blur.setToolTip(self.tr("Gaussian blur radius for softening matte edges.\n0 = no blur."))
        self._ck_edge_blur.valueChanged.connect(lambda _: self.chroma_key_preview.emit())
        eb_row.addWidget(self._ck_edge_blur, 1)
        ck_layout.addLayout(eb_row)

        # Generate button
        self._ck_generate_btn = QPushButton(self.tr("GENERATE"))
        self._ck_generate_btn.setToolTip(self.tr("Generate alpha hint frames for the entire clip using these chroma key settings."))
        self._ck_generate_btn.clicked.connect(self._on_chroma_key_generate)
        ck_layout.addWidget(self._ck_generate_btn)

        self._chroma_params_widget.hide()
        alpha_layout.addWidget(self._chroma_params_widget)

        # Sampled screen color for chroma key (RGB tuple or None)
        self._ck_screen_color: tuple[int, int, int] | None = None
        # Full list of eyedropper samples for multi-reference keying
        self._ck_screen_samples: list[tuple[int, int, int]] = []

        ck_or_label = QLabel("— or —")
        ck_or_label.setAlignment(Qt.AlignCenter)
        ck_or_label.setStyleSheet("color: #808070; font-size: 11px;")
        alpha_layout.addWidget(ck_or_label)

        # -- Automatic sub-section --
        auto_label = QLabel(self.tr("Automatic"))
        auto_label.setAlignment(Qt.AlignCenter)
        auto_label.setStyleSheet("color: #A0A090; font-size: 10px; margin: 0px 0 2px 0;")
        alpha_layout.addWidget(auto_label)

        self._gvm_btn = QPushButton(self.tr("GVM AUTO"))
        self._gvm_btn.setEnabled(False)
        self._gvm_btn.setToolTip(
            self.tr(
                "Auto-generate alpha hint for the entire clip.\n"
                "Uses GVM to predict foreground/background separation.\n"
                "Available when clip is in RAW state (frames extracted)."
            )
        )
        self._gvm_btn.clicked.connect(self.gvm_requested.emit)
        alpha_layout.addWidget(self._gvm_btn)

        # BiRefNet: button + model variant dropdown in a single row
        birefnet_row = QHBoxLayout()
        birefnet_row.setSpacing(4)
        self._birefnet_btn = QPushButton(self.tr("BIREFNET"))
        self._birefnet_btn.setEnabled(False)
        self._birefnet_btn.setToolTip(
            self.tr(
                "Auto-generate alpha hint using BiRefNet.\n"
                "Fully automatic \u2014 no painting or annotation needed.\n"
                "Downloads the selected model variant on first use.\n\n"
                "Matting: Best for hair/transparency detail (recommended).\n"
                "Portrait: Optimized for human close-ups.\n"
                "General: Balanced foreground/background separation.\n"
                "HR variants: For 2K/4K footage (uses more VRAM)."
            )
        )
        self._birefnet_btn.clicked.connect(self._on_birefnet_clicked)
        birefnet_row.addWidget(self._birefnet_btn, 1)

        self._birefnet_model = QComboBox()
        self._birefnet_model.setMinimumWidth(120)
        self._birefnet_model.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._birefnet_model.setToolTip(self.tr("BiRefNet model variant \u2014 changes take effect on next run."))
        # Populate from the wrapper's model registry
        from modules.BiRefNetModule.wrapper import BIREFNET_MODELS, DEFAULT_MODEL
        for display_name in BIREFNET_MODELS:
            self._birefnet_model.addItem(display_name)
        # Restore last-used model from QSettings
        from PySide6.QtCore import QSettings
        saved_model = QSettings().value("alpha/birefnet_model", DEFAULT_MODEL)
        idx = self._birefnet_model.findText(saved_model)
        if idx >= 0:
            self._birefnet_model.setCurrentIndex(idx)
        self._birefnet_model.currentTextChanged.connect(self._on_birefnet_model_changed)
        birefnet_row.addWidget(self._birefnet_model, 1)
        alpha_layout.addLayout(birefnet_row)

        or_label = QLabel("— or —")
        or_label.setAlignment(Qt.AlignCenter)
        or_label.setStyleSheet("color: #808070; font-size: 11px;")
        alpha_layout.addWidget(or_label)

        # -- Guided sub-section --
        guided_label = QLabel(self.tr("Requires brushstrokes"))
        guided_label.setAlignment(Qt.AlignCenter)
        guided_label.setStyleSheet("color: #A0A090; font-size: 10px; margin: 0px 0 2px 0;")
        alpha_layout.addWidget(guided_label)

        annotate_hint = QLabel(self.tr("Paint subject with 1, background with 2"))
        annotate_hint.setAlignment(Qt.AlignCenter)
        annotate_hint.setWordWrap(True)
        annotate_hint.setStyleSheet("color: #A0A090; font-size: 10px; margin: 2px 0;")
        alpha_layout.addWidget(annotate_hint)

        self._track_masks_btn = QPushButton(self.tr("TRACK MASK"))
        self._track_masks_btn.setEnabled(False)
        self._track_masks_btn.setToolTip(
            self.tr(
                "Use SAM2 to turn painted prompts into a dense mask track.\n"
                "Required before running MatAnyone2 or VideoMaMa.\n\n"
                "HOW TO USE:\n"
                "1. Press 1 to select the GREEN brush (foreground \u2014 subject to keep)\n"
                "2. Press 2 to select the RED brush (background \u2014 area to remove)\n"
                "3. Paint strokes on the left viewer over your footage\n"
                "4. Click TRACK MASK to preview SAM2 on the painted frame\n"
                "5. If the preview looks right, confirm to propagate across all frames"
            )
        )
        self._track_masks_btn.clicked.connect(self.track_masks_requested.emit)
        alpha_layout.addWidget(self._track_masks_btn)

        self._annotation_info = QLabel("")
        self._annotation_info.setStyleSheet("color: #808070; font-size: 10px;")
        alpha_layout.addWidget(self._annotation_info)

        matanyone2_hint = QLabel(self.tr("Requires paint strokes on frame 1"))
        matanyone2_hint.setAlignment(Qt.AlignCenter)
        matanyone2_hint.setWordWrap(True)
        matanyone2_hint.setStyleSheet("color: #A0A090; font-size: 10px; margin: 2px 0;")
        alpha_layout.addWidget(matanyone2_hint)

        self._matanyone2_btn = QPushButton(self.tr("MATANYONE2"))
        self._matanyone2_btn.setEnabled(False)
        self._matanyone2_btn.setToolTip(
            self.tr(
                "Generate alpha hints using MatAnyone2 video matting.\n"
                "Requires paint strokes on the FIRST FRAME (frame 1).\n\n"
                "1. Navigate to frame 1 (the very first frame)\n"
                "2. Paint foreground (hotkey 1) and background (hotkey 2)\n"
                "3. Click Track Mask to generate dense masks with SAM2\n"
                "4. Click MATANYONE2 to generate temporally coherent AlphaHint"
            )
        )
        self._matanyone2_btn.clicked.connect(self.matanyone2_requested.emit)
        alpha_layout.addWidget(self._matanyone2_btn)

        self._videomama_btn = QPushButton(self.tr("VIDEOMAMA"))
        self._videomama_btn.setEnabled(False)
        self._videomama_btn.setToolTip(
            self.tr(
                "Generate alpha hints from a dense VideoMaMa mask track.\n\n"
                "1. Paint sparse foreground/background prompts\n"
                "2. Click Track Mask to generate dense masks with SAM2\n"
                "3. Click VIDEOMAMA to generate AlphaHint"
            )
        )
        self._videomama_btn.clicked.connect(self.videomama_requested.emit)
        alpha_layout.addWidget(self._videomama_btn)

        # + overlay: child of VIDEOMAMA button for automatic positioning.
        # We override setEnabled on the + button to ignore parent disable.
        self._vmama_import_btn = QPushButton("+", self._videomama_btn)
        self._vmama_import_btn.setFixedSize(26, 26)
        self._vmama_import_btn.setToolTip(
            self.tr(
                "Import your own mask for VideoMaMa.\n\n"
                "Bypasses the Track Mask step. Select a folder or\n"
                "video of grayscale masks and they will be used as\n"
                "VideoMaMa's guidance input directly."
            )
        )
        self._vmama_import_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 14px; padding: 0px;"
            "  background: #454430; border: 1px solid #5A5940; }"
            "QPushButton:hover { background: #5A5940; }"
        )
        self._vmama_import_btn.clicked.connect(self.import_vmama_mask_requested.emit)
        self._vmama_import_btn.raise_()
        # Position: left edge, vertically centered inside the button
        orig_resize = self._videomama_btn.resizeEvent
        def _on_vmama_resize(e):
            if orig_resize:
                orig_resize(e)
            h = self._videomama_btn.height()
            self._vmama_import_btn.move(3, (h - 26) // 2)
        self._videomama_btn.resizeEvent = _on_vmama_resize

        or_label2 = QLabel("— or —")
        or_label2.setAlignment(Qt.AlignCenter)
        or_label2.setStyleSheet("color: #808070; font-size: 11px;")
        alpha_layout.addWidget(or_label2)

        self._import_alpha_btn = QPushButton(self.tr("IMPORT ALPHA"))
        self._import_alpha_btn.setEnabled(False)
        self._import_alpha_btn.setToolTip(
            self.tr(
                "Import alpha hints from an image folder or video file.\n"
                "Supports: PNG/JPG/TIF/EXR sequences, or MOV/MP4/ProRes video.\n"
                "White = foreground, black = background.\n"
                "Files are copied into the clip's AlphaHint/ folder\n"
                "and the clip advances to READY state for inference."
            )
        )
        self._import_alpha_btn.clicked.connect(self.import_alpha_requested.emit)
        alpha_layout.addWidget(self._import_alpha_btn)

        layout.addWidget(alpha_group)

        # ── INFERENCE section (Step 2) ──
        inf_group = QGroupBox(self.tr("INFERENCE"))
        inf_layout = QVBoxLayout(inf_group)
        inf_layout.setSpacing(8)

        # BG Color (screen color: auto-detect, green, or blue)
        bg_row = QHBoxLayout()
        self._bg_color_label = QLabel(self.tr("BG Color"))
        self._bg_color_label.setFixedWidth(80)
        self._bg_color_label.setToolTip(
            self.tr(
                "Background screen color for this clip.\n\n"
                "Auto: detected from the middle frame of the clip.\n"
                "Green: force green screen processing.\n"
                "Blue: force blue screen processing.\n\n"
                "Controls which checkpoint, despill math, and spill\n"
                "detection are used. Also changes the UI accent color."
            )
        )
        bg_row.addWidget(self._bg_color_label)
        self._bg_color = QComboBox()
        self._bg_color.addItems([self.tr("Auto"), self.tr("Green"), self.tr("Blue")])
        self._bg_color.setToolTip(self._bg_color_label.toolTip())
        self._bg_color.currentIndexChanged.connect(self._on_bg_color_changed)
        bg_row.addWidget(self._bg_color, 1)
        inf_layout.addLayout(bg_row)

        # Color Space
        cs_row = QHBoxLayout()
        self._color_space_label = QLabel(self.tr("Color Space"))
        self._color_space_label.setFixedWidth(80)
        self._color_space_label.setToolTip(self.tr(_COLOR_SPACE_TOOLTIP))
        cs_row.addWidget(self._color_space_label)
        self._color_space = QComboBox()
        self._color_space.addItems([self.tr("sRGB"), self.tr("Linear")])
        self._color_space.setToolTip(self.tr(_COLOR_SPACE_TOOLTIP))
        self._color_space.currentIndexChanged.connect(self._emit_changed)
        cs_row.addWidget(self._color_space, 1)
        inf_layout.addLayout(cs_row)

        # Despill Strength (slider 0-10 → 0.0-1.0)
        self._despill_label = QLabel(self.tr("Despill: 0.5"))
        inf_layout.addWidget(self._despill_label)
        self._despill_slider = _no_scroll_wheel(QSlider(Qt.Horizontal))
        self._despill_slider.setRange(0, 10)
        self._despill_slider.setValue(5)
        self._despill_slider.setToolTip(
            self.tr(
                "Screen spill removal strength (0.0-1.0).\n"
                "Removes background color bleed from hair, skin, and edges.\n"
                "1.0 = full despill, 0.0 = no despill (keep original colors)."
            )
        )
        self._despill_slider.valueChanged.connect(self._on_despill_changed)
        inf_layout.addWidget(self._despill_slider)

        # Despeckle toggle + size
        despeckle_row = QHBoxLayout()
        self._despeckle_check = QCheckBox(self.tr("Despeckle"))
        self._despeckle_check.setChecked(True)
        self._despeckle_check.setToolTip(
            self.tr(
                "Automatic garbage matte \u2014 removes small floating noise\n"
                "and speckles from the alpha by discarding isolated regions\n"
                "smaller than the size threshold."
            )
        )
        self._despeckle_check.stateChanged.connect(self._on_despeckle_toggled)
        despeckle_row.addWidget(self._despeckle_check)
        self._despeckle_size = _no_scroll_wheel(QSpinBox())
        self._despeckle_size.setRange(0, 999999)
        self._despeckle_size.setValue(400)
        self._despeckle_size.setSuffix("px")
        self._despeckle_size.setToolTip(
            self.tr(
                "Minimum area (in pixels) for a region to survive.\n"
                "Isolated alpha blobs smaller than this are removed.\n"
                "Lower = keep more detail, higher = cleaner matte."
            )
        )
        self._despeckle_size.valueChanged.connect(self._emit_changed)
        despeckle_row.addWidget(self._despeckle_size, 1)
        inf_layout.addLayout(despeckle_row)

        # Refiner Scale (slider 0-30 → 0.0-3.0)
        self._refiner_label = QLabel(self.tr("Refiner: 1.0"))
        inf_layout.addWidget(self._refiner_label)
        self._refiner_slider = _no_scroll_wheel(QSlider(Qt.Horizontal))
        self._refiner_slider.setRange(0, 30)
        self._refiner_slider.setValue(10)
        self._refiner_slider.setToolTip(
            self.tr(
                "Edge refinement strength (0.0\u20133.0).\n"
                "Scales the CNN refiner's edge corrections.\n"
                "1.0 = default, 0.0 = backbone only (no refinement),\n"
                "higher = sharper edges but may introduce artifacts."
            )
        )
        self._refiner_slider.valueChanged.connect(self._on_refiner_changed)
        inf_layout.addWidget(self._refiner_slider)

        # Live Preview toggle
        self._live_preview = QCheckBox(self.tr("Live Preview"))
        self._live_preview.setChecked(True)
        self._live_preview.setToolTip(self.tr(_LIVE_PREVIEW_TOOLTIP))
        inf_layout.addWidget(self._live_preview)

        layout.addWidget(inf_group)

        # ── OUTPUT FORMAT section (Step 3) ──
        out_group = QGroupBox(self.tr("OUTPUT"))
        out_layout = QVBoxLayout(out_group)
        out_layout.setSpacing(6)

        # FG
        fg_row = QHBoxLayout()
        self._fg_check = QCheckBox(self.tr("FG"))
        self._fg_check.setChecked(True)
        self._fg_check.setToolTip(
            self.tr(
                "Foreground \u2014 despilled subject on black background.\n"
                "Screen spill removed from hair and edges.\n"
                "Straight alpha (not premultiplied)."
            )
        )
        fg_row.addWidget(self._fg_check, 1)
        self._fg_format = QComboBox()
        self._fg_format.addItems(["exr", "png"])
        self._fg_format.setFixedWidth(70)
        self._fg_format.setToolTip(self.tr("EXR = 32-bit float (post-production).\nPNG = 8-bit (general use)."))
        fg_row.addWidget(self._fg_format)
        out_layout.addLayout(fg_row)

        # Matte
        matte_row = QHBoxLayout()
        self._matte_check = QCheckBox(self.tr("Matte"))
        self._matte_check.setChecked(True)
        self._matte_check.setToolTip(
            self.tr(
                "Alpha matte \u2014 grayscale transparency map.\n"
                "White = fully opaque, black = fully transparent.\n"
                "Use in compositing software for manual keying control."
            )
        )
        matte_row.addWidget(self._matte_check, 1)
        self._matte_format = QComboBox()
        self._matte_format.addItems(["exr", "png"])
        self._matte_format.setFixedWidth(70)
        self._matte_format.setToolTip(self.tr("EXR = 32-bit float (post-production).\nPNG = 8-bit (general use)."))
        matte_row.addWidget(self._matte_format)
        out_layout.addLayout(matte_row)

        # Comp
        comp_row = QHBoxLayout()
        self._comp_check = QCheckBox(self.tr("Comp"))
        self._comp_check.setChecked(True)
        self._comp_check.setToolTip(
            self.tr(
                "Composite \u2014 final keyed result over checkerboard.\n"
                "Best representation of the key quality.\n"
                "Colors match the original input faithfully."
            )
        )
        comp_row.addWidget(self._comp_check, 1)
        self._comp_format = QComboBox()
        self._comp_format.addItems(["png", "exr"])
        self._comp_format.setFixedWidth(70)
        self._comp_format.setToolTip(self.tr("PNG = 8-bit with transparency.\nEXR = 32-bit float (post-production)."))
        comp_row.addWidget(self._comp_format)
        out_layout.addLayout(comp_row)

        # Processed
        proc_row = QHBoxLayout()
        self._proc_check = QCheckBox(self.tr("Processed"))
        self._proc_check.setChecked(True)
        self._proc_check.setToolTip(
            self.tr(
                "Processed \u2014 production-ready RGBA (straight, linear).\n"
                "Designed for import into Resolve, Premiere, and compositing tools.\n"
                "Includes despill + garbage matte cleanup applied."
            )
        )
        proc_row.addWidget(self._proc_check, 1)
        self._proc_format = QComboBox()
        self._proc_format.addItems(["exr", "png"])
        self._proc_format.setFixedWidth(70)
        self._proc_format.setToolTip(self.tr("EXR = 32-bit float (recommended for Processed).\nPNG = 8-bit (lossy for straight linear RGBA)."))
        proc_row.addWidget(self._proc_format)
        out_layout.addLayout(proc_row)

        layout.addWidget(out_group)

        # ── PERFORMANCE section ──
        perf_group = QGroupBox(self.tr("PERFORMANCE"))
        perf_layout = QVBoxLayout(perf_group)
        perf_layout.setSpacing(6)

        parallel_row = QHBoxLayout()
        parallel_label = QLabel(self.tr("Parallel frames"))
        parallel_row.addWidget(parallel_label, 1)
        self._parallel_spin = QSpinBox()
        self._parallel_spin.setRange(1, 64)
        self._parallel_spin.setToolTip(
            self.tr(
                "Process multiple frames simultaneously using parallel engines.\n\n"
                "Each extra engine loads a full copy of the model.\n"
                "CUDA: ~6-8 GB VRAM per engine.\n"
                "\n"
                "Default: 1 (safest). Try 2 first, then increase if stable.\n\n"
                "EXPERIMENTAL: Values above 8 are for high-memory CUDA systems\n"
                "(e.g. RTX 6000).\n"
                "If you run out of memory, the app will automatically scale\n"
                "back to however many engines fit.\n\n"
                "CUDA only right now. Not currently supported on Apple Silicon."
            )
        )
        self._parallel_spin.setFixedWidth(60)
        from ui.widgets.preferences_dialog import get_setting_int, KEY_PARALLEL_CLIPS, DEFAULT_PARALLEL_CLIPS
        self._parallel_spin.setValue(get_setting_int(KEY_PARALLEL_CLIPS, DEFAULT_PARALLEL_CLIPS))
        self._parallel_spin.valueChanged.connect(self._on_parallel_changed)
        parallel_row.addWidget(self._parallel_spin)
        perf_layout.addLayout(parallel_row)

        layout.addWidget(perf_group)

        layout.addStretch(1)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # Middle-click reset: map widget → (setter_callable, default_value)
        self._middle_click_defaults: dict[QWidget, tuple] = {
            self._despill_slider: (self._despill_slider.setValue, 5),       # 0.5
            self._refiner_slider: (self._refiner_slider.setValue, 10),      # 1.0
            self._despeckle_size: (self._despeckle_size.setValue, 400),      # 400px
            self._ck_strength: (self._ck_strength.setValue, 10),            # 1.0
            self._ck_clip_black: (self._ck_clip_black.setValue, 0),         # 0.0
            self._ck_clip_white: (self._ck_clip_white.setValue, 100),       # 1.0
            self._ck_shrink_grow: (self._ck_shrink_grow.setValue, 0),       # 0px
            self._ck_edge_blur: (self._ck_edge_blur.setValue, 0),           # 0px
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
        self._despill_label.setText(self.tr("Despill: %s") % f"{display:.1f}")
        self._emit_changed()

    def _on_refiner_changed(self, value: int) -> None:
        display = value / 10.0
        self._refiner_label.setText(self.tr("Refiner: %s") % f"{display:.1f}")
        self._emit_changed()

    def _on_bg_color_changed(self, index: int) -> None:
        """BG Color dropdown changed. Emit signal for accent swap."""
        color = ["auto", "green", "blue"][index]
        self.screen_color_changed.emit(color)
        self._emit_changed()

    def _on_birefnet_clicked(self) -> None:
        """Emit birefnet_requested with the currently selected model variant."""
        self.birefnet_requested.emit(self._birefnet_model.currentText())

    def _on_birefnet_model_changed(self, text: str) -> None:
        """Persist the selected BiRefNet model variant to QSettings."""
        from PySide6.QtCore import QSettings
        QSettings().setValue("alpha/birefnet_model", text)

    def _on_parallel_changed(self, value: int) -> None:
        from PySide6.QtCore import QSettings
        from ui.widgets.preferences_dialog import KEY_PARALLEL_CLIPS
        QSettings().setValue(KEY_PARALLEL_CLIPS, value)
        self.parallel_frames_changed.emit(value)

    @property
    def live_preview_enabled(self) -> bool:
        return self._live_preview.isChecked()

    def get_params(self) -> InferenceParams:
        """Snapshot current parameter values into a frozen InferenceParams."""
        bg_idx = self._bg_color.currentIndex()
        screen_color = ["auto", "green", "blue"][bg_idx]
        return InferenceParams(
            input_is_linear=self._color_space.currentIndex() == 1,
            despill_strength=self._despill_slider.value() / 10.0,
            auto_despeckle=self._despeckle_check.isChecked(),
            despeckle_size=self._despeckle_size.value(),
            despeckle_dilation=25,  # fixed default
            despeckle_blur=5,       # fixed default
            refiner_scale=self._refiner_slider.value() / 10.0,
            screen_color=screen_color,
        )

    def get_output_config(self) -> OutputConfig:
        """Snapshot current output format configuration."""
        from ui.widgets.preferences_dialog import (
            KEY_EXR_COMPRESSION, DEFAULT_EXR_COMPRESSION, get_setting_str,
        )
        return OutputConfig(
            fg_enabled=self._fg_check.isChecked(),
            fg_format=self._fg_format.currentText(),
            matte_enabled=self._matte_check.isChecked(),
            matte_format=self._matte_format.currentText(),
            comp_enabled=self._comp_check.isChecked(),
            comp_format=self._comp_format.currentText(),
            processed_enabled=self._proc_check.isChecked(),
            processed_format=self._proc_format.currentText(),
            exr_compression=get_setting_str(KEY_EXR_COMPRESSION, DEFAULT_EXR_COMPRESSION),
        )

    def auto_detect_color_space(self, prefer_linear: bool) -> None:
        """Auto-set color space based on input format.

        Standalone linear EXR sequences → Linear, video-derived footage → sRGB.
        """
        target = 1 if prefer_linear else 0  # 1=Linear, 0=sRGB
        if self._color_space.currentIndex() != target:
            self._color_space.setCurrentIndex(target)

    def set_input_is_linear(self, input_is_linear: bool) -> None:
        """Programmatically set Color Space without emitting params_changed."""
        self._suppress_signals = True
        try:
            self._color_space.setCurrentIndex(1 if input_is_linear else 0)
        finally:
            self._suppress_signals = False

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

    def set_birefnet_enabled(self, enabled: bool) -> None:
        """Enable/disable BiRefNet button based on clip state."""
        self._birefnet_btn.setEnabled(enabled)

    def set_videomama_enabled(self, enabled: bool) -> None:
        """Enable/disable VideoMaMa button based on clip state."""
        self._videomama_btn.setEnabled(enabled)
        # Qt disables children when parent is disabled. Force + back on
        # after a 0ms timer so it runs after Qt propagates the disable.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: self._vmama_import_btn.setEnabled(True))

    def set_matanyone2_enabled(self, enabled: bool) -> None:
        """Enable/disable MatAnyone2 button based on clip state."""
        self._matanyone2_btn.setEnabled(enabled)

    def set_import_alpha_enabled(self, enabled: bool) -> None:
        """Enable/disable Import Alpha button based on clip state."""
        self._import_alpha_btn.setEnabled(enabled)

    def set_annotation_info(self, annotated: int, total: int) -> None:
        """Update annotation frame counter."""
        if annotated > 0 and total > 0:
            self._annotation_info.setText(self.tr("Painted: %d / %d frames") % (annotated, total))
            self._track_masks_btn.setEnabled(True)
        else:
            self._annotation_info.setText("")
            self._track_masks_btn.setEnabled(False)

    # ── Chroma Key ──

    def set_chroma_key_enabled(self, enabled: bool) -> None:
        """Enable/disable Chroma Key button based on clip state."""
        self._chroma_key_btn.setEnabled(enabled)
        if not enabled:
            self._chroma_key_btn.setChecked(False)

    def _on_chroma_key_toggled(self, checked: bool) -> None:
        """Show/hide chroma key parameter panel."""
        self._chroma_params_widget.setVisible(checked)
        if not checked:
            # Deactivate eyedropper when collapsing
            self._eyedropper_btn.setChecked(False)

    def _on_eyedropper_toggled(self, checked: bool) -> None:
        """Toggle eyedropper mode on the viewer."""
        self.eyedropper_requested.emit(checked)

    def preview_screen_color(self, r: int, g: int, b: int) -> None:
        """Live swatch update during eyedropper drag (visual only, no commit)."""
        self._color_swatch.setStyleSheet(
            f"background: rgb({r},{g},{b}); border: 2px solid #FFF203;"
        )

    def set_screen_samples(self, samples: list[tuple[int, int, int]]) -> None:
        """Store eyedropper samples for multi-reference keying (capped at 200)."""
        if len(samples) > 200:
            # Evenly subsample to avoid bloating clip.json
            step = len(samples) / 200
            samples = [samples[int(i * step)] for i in range(200)]
        self._ck_screen_samples = samples

    def set_sampled_screen_color(self, r: int, g: int, b: int) -> None:
        """Commit the sampled screen color (on release)."""
        self._ck_screen_color = (r, g, b)
        self._color_swatch.setStyleSheet(
            f"background: rgb({r},{g},{b}); border: 1px solid #5A5940;"
        )
        # Auto-exit eyedropper after sampling
        self._eyedropper_btn.setChecked(False)

    def get_chroma_params(self) -> dict:
        """Snapshot current chroma key parameters."""
        bg_idx = self._bg_color.currentIndex()
        screen_type = ["auto", "green", "blue"][bg_idx]
        # "auto" passes through — resolved by the caller using clip's detected color
        return {
            "screen_color": self._ck_screen_color,
            "screen_samples": self._ck_screen_samples or None,
            "screen_type": screen_type,
            "strength": self._ck_strength.value() / 10.0,
            "clip_black": self._ck_clip_black.value() / 100.0,
            "clip_white": self._ck_clip_white.value() / 100.0,
            "shrink_grow": self._ck_shrink_grow.value(),
            "edge_blur": self._ck_edge_blur.value(),
        }

    def set_chroma_params(self, params: dict) -> None:
        """Restore chroma key parameters from a saved dict (per-clip persistence)."""
        self._suppress_signals = True
        try:
            sc = params.get("screen_color")
            if sc is not None:
                self.set_sampled_screen_color(*sc)
            else:
                self._ck_screen_color = None
                self._color_swatch.setStyleSheet(
                    "background: #333; border: 1px solid #5A5940;"
                )
            # Restore multi-reference samples
            samples = params.get("screen_samples")
            self._ck_screen_samples = [tuple(s) for s in samples] if samples else []
            self._ck_strength.setValue(int(params.get("strength", 1.0) * 10))
            self._ck_clip_black.setValue(int(params.get("clip_black", 0.0) * 100))
            self._ck_clip_white.setValue(int(params.get("clip_white", 1.0) * 100))
            self._ck_shrink_grow.setValue(params.get("shrink_grow", 0))
            self._ck_edge_blur.setValue(params.get("edge_blur", 0))
        finally:
            self._suppress_signals = False

    def reset_chroma_params(self) -> None:
        """Reset chroma key parameters to defaults (new clip with no saved state)."""
        self._suppress_signals = True
        try:
            self._ck_screen_color = None
            self._ck_screen_samples = []
            self._color_swatch.setStyleSheet(
                "background: #333; border: 1px solid #5A5940;"
            )
            self._ck_strength.setValue(10)
            self._ck_clip_black.setValue(0)
            self._ck_clip_white.setValue(100)
            self._ck_shrink_grow.setValue(0)
            self._ck_edge_blur.setValue(0)
        finally:
            self._suppress_signals = False

    def _on_chroma_key_generate(self) -> None:
        """Emit chroma_key_requested with current params."""
        self.chroma_key_requested.emit(self.get_chroma_params())

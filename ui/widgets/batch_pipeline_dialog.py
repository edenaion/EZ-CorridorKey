"""Batch Pipeline dialog — scan a folder, configure per-clip pipelines, run."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from backend.batch_scanner import BatchClipInfo, scan_batch_folder
from backend.job_queue import JobType

logger = logging.getLogger(__name__)

# Friendly labels for the "no hint" alpha method
_NO_HINT_OPTIONS = [
    ("GVM", JobType.GVM_ALPHA),
    ("BiRefNet", JobType.BIREFNET_ALPHA),
]

# Friendly labels for the "maskhint" refinement method
_MASKHINT_OPTIONS = [
    ("VideoMaMa", JobType.VIDEOMAMA_ALPHA),
    ("MatAnyone2", JobType.MATANYONE2_ALPHA),
]

# Table column indices
_COL_NAME = 0
_COL_DETECTED = 1
_COL_PIPELINE = 2
_COL_STATUS = 3


@dataclass
class BatchClipConfig:
    """Per-clip pipeline configuration returned by the dialog."""

    clip_info: BatchClipInfo
    alpha_job_type: JobType  # GVM_ALPHA, BIREFNET_ALPHA, VIDEOMAMA_ALPHA, MATANYONE2_ALPHA, or INFERENCE
    birefnet_usage: str = "Matting"


class BatchPipelineDialog(QDialog):
    """Dialog for configuring and running a batch folder pipeline."""

    run_requested = Signal()    # User clicked Run Batch
    clear_requested = Signal()  # User clicked Clear Pipeline

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Batch Pipeline"))
        self.setMinimumWidth(750)
        self.setMinimumHeight(480)

        self._clips: list[BatchClipInfo] = []
        self._per_clip_combos: list[QComboBox | None] = []
        self._per_clip_birefnet: list[QComboBox | None] = []
        self._status_widgets: list[QLabel] = []
        self._batch_active = False  # True while processing
        self._progress_bars: list[QProgressBar] = []

        self._build_ui()

    def closeEvent(self, event) -> None:
        """Hide instead of close while batch is active so progress is preserved."""
        if self._batch_active:
            event.ignore()
            self.hide()
        else:
            super().closeEvent(event)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # Title
        title = QLabel(self.tr("Batch Pipeline"))
        title.setStyleSheet("color: #FFF203; font-size: 16px; font-weight: bold;")
        root.addWidget(title)

        desc = QLabel(self.tr(
            "Select a folder containing video clips. Files with "
            "\"alphahint\" or \"maskhint\" in the name are automatically "
            "paired as hints."
        ))
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #A0A090; font-size: 12px;")
        root.addWidget(desc)

        # Folder selection
        folder_row = QHBoxLayout()
        self._folder_btn = QPushButton(self.tr("Select Folder..."))
        self._folder_btn.setFixedWidth(130)
        self._folder_btn.clicked.connect(self._on_select_folder)
        folder_row.addWidget(self._folder_btn)
        self._folder_label = QLabel(self.tr("No folder selected"))
        self._folder_label.setStyleSheet("color: #808070;")
        folder_row.addWidget(self._folder_label, 1)
        root.addLayout(folder_row)

        # Global settings group
        _COMBO_WIDTH = 150
        self._global_group = QGroupBox(self.tr("Global Settings"))
        global_layout = QHBoxLayout(self._global_group)
        global_layout.setSpacing(16)

        # No-hint column
        no_hint_lbl = QLabel(self.tr("No-hint clips:"))
        no_hint_lbl.setStyleSheet("color: #A0A090;")
        no_hint_lbl.setToolTip(self.tr(
            "Alpha generation method for clips with no companion hint file.\n"
            "GVM: fast automatic alpha.\n"
            "BiRefNet: higher quality, select a model variant."
        ))
        global_layout.addWidget(no_hint_lbl)
        self._global_no_hint = QComboBox()
        self._global_no_hint.setFixedWidth(_COMBO_WIDTH)
        for label, _ in _NO_HINT_OPTIONS:
            self._global_no_hint.addItem(label)
        self._global_no_hint.currentIndexChanged.connect(self._on_global_changed)
        global_layout.addWidget(self._global_no_hint)

        # BiRefNet variant (inline, shows/hides)
        self._global_birefnet_variant = QComboBox()
        self._global_birefnet_variant.setFixedWidth(_COMBO_WIDTH)
        self._populate_birefnet_models(self._global_birefnet_variant)
        self._global_birefnet_variant.currentIndexChanged.connect(self._on_global_changed)
        global_layout.addWidget(self._global_birefnet_variant)
        self._update_birefnet_visibility()
        self._global_no_hint.currentIndexChanged.connect(
            lambda: self._update_birefnet_visibility()
        )

        # Separator
        global_layout.addSpacing(24)

        # MaskHint column
        mask_lbl = QLabel(self.tr("MaskHint clips:"))
        mask_lbl.setStyleSheet("color: #A0A090;")
        mask_lbl.setToolTip(self.tr(
            "Mask refinement method for clips with a companion MaskHint file.\n"
            "VideoMaMa: temporal consistency, best for video.\n"
            "MatAnyone2: single-frame matting with mask guidance."
        ))
        global_layout.addWidget(mask_lbl)
        self._global_maskhint = QComboBox()
        self._global_maskhint.setFixedWidth(_COMBO_WIDTH)
        for label, _ in _MASKHINT_OPTIONS:
            self._global_maskhint.addItem(label)
        self._global_maskhint.currentIndexChanged.connect(self._on_global_changed)
        global_layout.addWidget(self._global_maskhint)

        global_layout.addStretch()
        root.addWidget(self._global_group)

        # Per-clip toggle
        self._per_clip_cb = QCheckBox(self.tr("Per-clip overrides"))
        self._per_clip_cb.toggled.connect(self._on_per_clip_toggled)
        root.addWidget(self._per_clip_cb)

        # Clip table
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels([
            self.tr("Clip"), self.tr("Detected"), self.tr("Pipeline"), self.tr("Status"),
        ])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(_COL_NAME, QHeaderView.Stretch)
        header.setSectionResizeMode(_COL_DETECTED, QHeaderView.Fixed)
        header.resizeSection(_COL_DETECTED, 90)
        header.setSectionResizeMode(_COL_PIPELINE, QHeaderView.Stretch)
        header.setSectionResizeMode(_COL_STATUS, QHeaderView.Fixed)
        header.resizeSection(_COL_STATUS, 80)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        root.addWidget(self._table, 1)

        # Summary label
        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("color: #A0A090; font-size: 11px;")
        root.addWidget(self._summary_label)

        # Buttons
        btn_row = QHBoxLayout()
        self._clear_btn = QPushButton(self.tr("Clear Pipeline"))
        self._clear_btn.setToolTip(self.tr("Cancel all pending batch jobs and reset."))
        self._clear_btn.setVisible(False)
        self._clear_btn.clicked.connect(self.clear_requested.emit)
        btn_row.addWidget(self._clear_btn)
        btn_row.addStretch()
        self._cancel_btn = QPushButton(self.tr("Cancel"))
        self._cancel_btn.clicked.connect(self.close)
        btn_row.addWidget(self._cancel_btn)
        self._run_btn = QPushButton(self.tr("Run Batch"))
        self._run_btn.setEnabled(False)
        self._run_btn.setToolTip(self.tr(
            "Inference settings (despill, refiner, edge, color space, etc.) "
            "are inherited from the right panel. Adjust them there before running."
        ))
        self._run_btn.setStyleSheet(
            "QPushButton { background: #FFF203; color: #141300; font-weight: bold; padding: 6px 20px; }"
            "QPushButton:disabled { background: #555; color: #888; }"
        )
        self._run_btn.clicked.connect(self.run_requested.emit)
        btn_row.addWidget(self._run_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Folder selection
    # ------------------------------------------------------------------

    @Slot()
    def _on_select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Select Batch Folder"), "",
        )
        if not folder:
            return
        self._folder_label.setText(folder)
        self._folder_label.setStyleSheet("color: #E0E0E0;")
        self._clips = scan_batch_folder(folder)
        self._populate_table()
        self._clear_btn.setVisible(len(self._clips) > 0)

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def _populate_table(self) -> None:
        self._per_clip_combos.clear()
        self._per_clip_birefnet.clear()
        self._status_widgets.clear()
        self._progress_bars.clear()
        self._table.setRowCount(len(self._clips))

        counts = {"none": 0, "alphahint": 0, "maskhint": 0}

        for row, clip in enumerate(self._clips):
            counts[clip.hint_type] += 1

            # Clip name
            name_item = QTableWidgetItem(os.path.basename(clip.source_path))
            name_item.setToolTip(clip.source_path)
            self._table.setItem(row, _COL_NAME, name_item)

            # Detected hint
            detect_labels = {
                "none": self.tr("No hint"),
                "alphahint": self.tr("AlphaHint"),
                "maskhint": self.tr("MaskHint"),
            }
            det_item = QTableWidgetItem(detect_labels[clip.hint_type])
            det_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, _COL_DETECTED, det_item)

            # Pipeline column (combo or label)
            _ROW_COMBO_W = 120
            per_clip_on = self._per_clip_cb.isChecked()

            if clip.hint_type == "alphahint":
                # Fixed: straight to CK
                lbl = QLabel(self.tr("CK Inference"))
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("color: #808070;")
                self._table.setCellWidget(row, _COL_PIPELINE, lbl)
                self._per_clip_combos.append(None)
                self._per_clip_birefnet.append(None)
            elif clip.hint_type == "maskhint":
                container = QWidget()
                h = QHBoxLayout(container)
                h.setContentsMargins(4, 0, 4, 0)
                h.setSpacing(6)
                combo = QComboBox()
                combo.setFixedWidth(_ROW_COMBO_W)
                for label, _ in _MASKHINT_OPTIONS:
                    combo.addItem(label)
                combo.setCurrentIndex(self._global_maskhint.currentIndex())
                combo.setEnabled(per_clip_on)
                h.addWidget(combo)
                arrow = QLabel(self.tr("\u2192 CK"))
                arrow.setStyleSheet("color: #808070; font-size: 11px;")
                h.addWidget(arrow)
                h.addStretch()
                self._table.setCellWidget(row, _COL_PIPELINE, container)
                self._per_clip_combos.append(combo)
                self._per_clip_birefnet.append(None)
            else:
                # No hint: GVM or BiRefNet
                container = QWidget()
                h = QHBoxLayout(container)
                h.setContentsMargins(4, 0, 4, 0)
                h.setSpacing(6)
                combo = QComboBox()
                combo.setFixedWidth(_ROW_COMBO_W)
                for label, _ in _NO_HINT_OPTIONS:
                    combo.addItem(label)
                combo.setCurrentIndex(self._global_no_hint.currentIndex())
                combo.setEnabled(per_clip_on)
                # BiRefNet variant per-clip
                br_combo = QComboBox()
                br_combo.setFixedWidth(_ROW_COMBO_W)
                self._populate_birefnet_models(br_combo)
                br_combo.setCurrentIndex(self._global_birefnet_variant.currentIndex())
                br_combo.setEnabled(per_clip_on)
                br_combo.setVisible(combo.currentIndex() == 1)
                combo.currentIndexChanged.connect(
                    lambda idx, b=br_combo: b.setVisible(idx == 1)
                )
                h.addWidget(combo)
                h.addWidget(br_combo)
                arrow = QLabel(self.tr("\u2192 CK"))
                arrow.setStyleSheet("color: #808070; font-size: 11px;")
                h.addWidget(arrow)
                h.addStretch()
                self._table.setCellWidget(row, _COL_PIPELINE, container)
                self._per_clip_combos.append(combo)
                self._per_clip_birefnet.append(br_combo)

            # Status column: label (replaced with progress bar during run)
            status_lbl = QLabel("")
            status_lbl.setAlignment(Qt.AlignCenter)
            self._table.setCellWidget(row, _COL_STATUS, status_lbl)
            self._status_widgets.append(status_lbl)
            self._progress_bars.append(None)  # created on demand

        # Update summary
        parts = []
        if counts["none"]:
            parts.append(f"{counts['none']} no hint")
        if counts["alphahint"]:
            parts.append(f"{counts['alphahint']} AlphaHint")
        if counts["maskhint"]:
            parts.append(f"{counts['maskhint']} MaskHint")
        total = len(self._clips)
        self._summary_label.setText(
            self.tr("Found %d clip(s): %s") % (total, ", ".join(parts)) if total
            else self.tr("No video clips found in this folder.")
        )

        # Enable/disable global controls based on what was found
        self._global_no_hint.setEnabled(counts["none"] > 0)
        self._global_maskhint.setEnabled(counts["maskhint"] > 0)
        self._run_btn.setEnabled(total > 0)

    # ------------------------------------------------------------------
    # Global / per-clip toggle
    # ------------------------------------------------------------------

    @Slot()
    def _on_global_changed(self) -> None:
        """Propagate global settings to all per-clip combos (when in global mode)."""
        if self._per_clip_cb.isChecked():
            return
        for row, clip in enumerate(self._clips):
            combo = self._per_clip_combos[row]
            if combo is None:
                continue
            if clip.hint_type == "maskhint":
                combo.setCurrentIndex(self._global_maskhint.currentIndex())
            elif clip.hint_type == "none":
                combo.setCurrentIndex(self._global_no_hint.currentIndex())
                br = self._per_clip_birefnet[row]
                if br:
                    br.setCurrentIndex(self._global_birefnet_variant.currentIndex())

    @Slot(bool)
    def _on_per_clip_toggled(self, checked: bool) -> None:
        """Toggle between global and per-clip mode."""
        # Gray out globals when per-clip is enabled
        self._global_group.setEnabled(not checked)

        if checked:
            # Propagate current global values to all rows first
            self._on_global_changed()

        # Enable/disable per-clip combos
        for row, clip in enumerate(self._clips):
            combo = self._per_clip_combos[row]
            if combo is not None:
                combo.setEnabled(checked)
            br = self._per_clip_birefnet[row]
            if br is not None:
                br.setEnabled(checked)

    def _update_birefnet_visibility(self) -> None:
        """Show/hide global BiRefNet variant dropdown based on selected method."""
        self._global_birefnet_variant.setVisible(
            self._global_no_hint.currentIndex() == 1  # BiRefNet
        )

    def _populate_birefnet_models(self, combo: QComboBox) -> None:
        """Fill a combo with available BiRefNet model variants."""
        try:
            from modules.BiRefNetModule.wrapper import BIREFNET_MODELS, DEFAULT_MODEL
            for name in BIREFNET_MODELS:
                combo.addItem(name)
            idx = combo.findText(DEFAULT_MODEL)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        except ImportError:
            combo.addItem("Matting")

    # ------------------------------------------------------------------
    # Config output
    # ------------------------------------------------------------------

    def get_batch_config(self) -> list[BatchClipConfig]:
        """Return per-clip pipeline configuration."""
        configs: list[BatchClipConfig] = []
        for row, clip in enumerate(self._clips):
            if clip.hint_type == "alphahint":
                configs.append(BatchClipConfig(
                    clip_info=clip,
                    alpha_job_type=JobType.INFERENCE,
                ))
            elif clip.hint_type == "maskhint":
                combo = self._per_clip_combos[row]
                idx = combo.currentIndex() if combo else 0
                _, job_type = _MASKHINT_OPTIONS[idx]
                configs.append(BatchClipConfig(
                    clip_info=clip,
                    alpha_job_type=job_type,
                ))
            else:
                combo = self._per_clip_combos[row]
                idx = combo.currentIndex() if combo else 0
                _, job_type = _NO_HINT_OPTIONS[idx]
                birefnet_usage = "Matting"
                if job_type == JobType.BIREFNET_ALPHA:
                    br = self._per_clip_birefnet[row]
                    if br:
                        birefnet_usage = br.currentText()
                configs.append(BatchClipConfig(
                    clip_info=clip,
                    alpha_job_type=job_type,
                    birefnet_usage=birefnet_usage,
                ))
        return configs

    def get_folder_path(self) -> str:
        """Return the selected folder path."""
        text = self._folder_label.text()
        if text == self.tr("No folder selected"):
            return ""
        return text

    # ------------------------------------------------------------------
    # Processing mode
    # ------------------------------------------------------------------

    def enter_processing_mode(self, clips: list[BatchClipInfo] | None = None) -> None:
        """Switch dialog to processing mode. Disables config, shows progress."""
        self._batch_active = True
        if clips:
            self._clips = clips
            self._populate_table()
        self.setWindowTitle(self.tr("Batch Pipeline - Processing"))
        self._run_btn.setEnabled(False)
        self._run_btn.setText(self.tr("Running..."))
        self._folder_btn.setEnabled(False)
        self._per_clip_cb.setEnabled(False)
        self._global_group.setEnabled(False)
        self._clear_btn.setVisible(True)
        # Disable all per-clip combos
        for combo in self._per_clip_combos:
            if combo is not None:
                combo.setEnabled(False)
        for br in self._per_clip_birefnet:
            if br is not None:
                br.setEnabled(False)

    def reset_to_initial(self) -> None:
        """Reset dialog back to folder-selection state."""
        self._batch_active = False
        self._clips.clear()
        self._table.setRowCount(0)
        self._per_clip_combos.clear()
        self._per_clip_birefnet.clear()
        self._status_widgets.clear()
        self._progress_bars.clear()
        self.setWindowTitle(self.tr("Batch Pipeline"))
        self._folder_label.setText(self.tr("No folder selected"))
        self._folder_label.setStyleSheet("color: #808070;")
        self._folder_btn.setEnabled(True)
        self._per_clip_cb.setEnabled(True)
        self._per_clip_cb.setChecked(False)
        self._global_group.setEnabled(True)
        self._run_btn.setEnabled(False)
        self._run_btn.setText(self.tr("Run Batch"))
        self._cancel_btn.setText(self.tr("Cancel"))
        self._clear_btn.setVisible(False)
        self._summary_label.setText("")

    # ------------------------------------------------------------------
    # Status updates (called by mixin during batch processing)
    # ------------------------------------------------------------------

    def set_clip_running(self, clip_name: str, current: int = 0, total: int = 0) -> None:
        """Mark a clip row as currently processing."""
        row = self._find_clip_row(clip_name)
        if row < 0:
            return
        # Replace status label with progress bar if not already
        if self._progress_bars[row] is None:
            pb = QProgressBar()
            pb.setMaximumHeight(18)
            pb.setTextVisible(True)
            pb.setStyleSheet(
                "QProgressBar { background: #2A2910; border: 1px solid #3A3920; }"
                "QProgressBar::chunk { background: #FFF203; }"
            )
            self._table.setCellWidget(row, _COL_STATUS, pb)
            self._progress_bars[row] = pb
        pb = self._progress_bars[row]
        if total > 0:
            pb.setMaximum(total)
            pb.setValue(current)
        else:
            pb.setMaximum(0)  # indeterminate

    def set_clip_progress(self, clip_name: str, current: int, total: int) -> None:
        """Update progress for a running clip."""
        row = self._find_clip_row(clip_name)
        if row < 0:
            return
        pb = self._progress_bars[row]
        if pb:
            pb.setMaximum(total)
            pb.setValue(current)

    def set_clip_done(self, clip_name: str) -> None:
        """Mark a clip as completed (green checkmark)."""
        row = self._find_clip_row(clip_name)
        if row < 0:
            return
        lbl = QLabel("\u2714")  # checkmark
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #4CAF50; font-size: 16px; font-weight: bold;")
        self._table.setCellWidget(row, _COL_STATUS, lbl)
        self._progress_bars[row] = None

    def set_clip_error(self, clip_name: str, error_msg: str = "") -> None:
        """Mark a clip as failed (red X)."""
        row = self._find_clip_row(clip_name)
        if row < 0:
            return
        lbl = QLabel("\u2718")  # X mark
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #F44336; font-size: 16px; font-weight: bold;")
        lbl.setToolTip(error_msg or self.tr("Processing failed"))
        self._table.setCellWidget(row, _COL_STATUS, lbl)
        self._progress_bars[row] = None

    def set_batch_complete(self) -> None:
        """Called when the entire batch finishes."""
        self._batch_active = False
        self.setWindowTitle(self.tr("Batch Pipeline - Complete"))
        self._run_btn.setText(self.tr("Done"))
        self._run_btn.setEnabled(False)
        self._cancel_btn.setText(self.tr("Close"))
        self._clear_btn.setVisible(False)

    def _find_clip_row(self, clip_name: str) -> int:
        """Find table row index for a clip by name."""
        for row, clip in enumerate(self._clips):
            if clip.name == clip_name:
                return row
        return -1

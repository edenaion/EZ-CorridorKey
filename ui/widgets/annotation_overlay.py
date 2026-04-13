"""Annotation overlay — green/red brush strokes for prompt authoring.

Provides a stroke-based annotation model and rendering/input mixin for
SplitViewWidget. Users paint foreground (green, hotkey 1) and background
(red, hotkey 2) strokes directly on frames. Strokes are stored per-frame
in image-pixel coordinates and are primarily consumed as prompts for
downstream mask tracking.

Strokes are persisted to {clip_root}/annotations.json so they survive
app restarts.

Brush size: Shift+left-drag up/down.
Straight line: Alt+left-drag draws a straight line at current brush size.
Undo: Ctrl+Z pops last stroke on the current frame.
"""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QColor

logger = logging.getLogger(__name__)


# ── Data Model ──────────────────────────────────────────────────────────────


@dataclass
class AnnotationStroke:
    """A single brush stroke in image-pixel coordinates."""

    points: list[tuple[float, float]] = field(default_factory=list)
    brush_type: str = "fg"  # "fg" (foreground/keep) or "bg" (background/remove)
    radius: float = 15.0  # brush radius in image pixels


class AnnotationModel:
    """Per-frame stroke storage with optional raster export."""

    def __init__(self) -> None:
        # stem_index -> list of completed strokes
        self._strokes: dict[int, list[AnnotationStroke]] = {}
        # In-progress stroke (not yet finished)
        self._current_stroke: AnnotationStroke | None = None
        self._current_frame: int = -1

    def start_stroke(
        self, stem_idx: int, x: float, y: float, brush_type: str, radius: float
    ) -> None:
        """Begin a new stroke on the given frame."""
        self._current_stroke = AnnotationStroke(
            points=[(x, y)],
            brush_type=brush_type,
            radius=radius,
        )
        self._current_frame = stem_idx

    def add_point(self, x: float, y: float) -> None:
        """Extend the current stroke."""
        if self._current_stroke is not None:
            self._current_stroke.points.append((x, y))

    def finish_stroke(self) -> None:
        """Finalize the current stroke and store it."""
        if self._current_stroke is not None and self._current_frame >= 0:
            if len(self._current_stroke.points) >= 1:
                strokes = self._strokes.setdefault(self._current_frame, [])
                strokes.append(self._current_stroke)
        self._current_stroke = None

    @property
    def current_stroke(self) -> AnnotationStroke | None:
        return self._current_stroke

    def undo(self, stem_idx: int) -> bool:
        """Pop the last stroke on a frame. Returns True if a stroke was removed."""
        strokes = self._strokes.get(stem_idx)
        if strokes:
            strokes.pop()
            if not strokes:
                del self._strokes[stem_idx]
            return True
        return False

    def clear(self, stem_idx: int | None = None) -> None:
        """Clear annotations for one frame or all frames."""
        if stem_idx is not None:
            self._strokes.pop(stem_idx, None)
        else:
            self._strokes.clear()
        self._current_stroke = None

    def get_strokes(self, stem_idx: int) -> list[AnnotationStroke]:
        """Return completed strokes for a frame."""
        return self._strokes.get(stem_idx, [])

    def has_annotations(self, stem_idx: int | None = None) -> bool:
        """Check if any annotations exist (optionally for a specific frame)."""
        if stem_idx is not None:
            return bool(self._strokes.get(stem_idx))
        return bool(self._strokes)

    def annotated_frame_count(self) -> int:
        return len(self._strokes)

    # ── Persistence ────────────────────────────────────────────────────────

    _FILENAME = "annotations.json"

    def save(self, clip_root: str) -> None:
        """Persist all strokes to {clip_root}/annotations.json."""
        if not clip_root:
            return
        data: dict[str, list[dict]] = {}
        for idx, strokes in self._strokes.items():
            data[str(idx)] = [
                {
                    "points": s.points,
                    "brush_type": s.brush_type,
                    "radius": s.radius,
                }
                for s in strokes
            ]
        path = os.path.join(clip_root, self._FILENAME)
        try:
            if data:
                with open(path, "w") as f:
                    json.dump(data, f)
            elif os.path.isfile(path):
                os.remove(path)
        except OSError as e:
            logger.warning(f"Failed to save annotations: {e}")

    def load(self, clip_root: str) -> None:
        """Load strokes from {clip_root}/annotations.json if it exists."""
        self._strokes.clear()
        self._current_stroke = None
        if not clip_root:
            return
        path = os.path.join(clip_root, self._FILENAME)
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            for idx_str, stroke_list in data.items():
                idx = int(idx_str)
                self._strokes[idx] = [
                    AnnotationStroke(
                        points=[tuple(p) for p in s["points"]],
                        brush_type=s.get("brush_type", "fg"),
                        radius=s.get("radius", 15.0),
                    )
                    for s in stroke_list
                ]
            logger.info(f"Loaded annotations for {len(self._strokes)} frames from {path}")
        except (json.JSONDecodeError, OSError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load annotations: {e}")

    def export_masks(
        self, clip_root: str, frame_stems: list[str], width: int, height: int, start_index: int = 0
    ) -> str:
        """Rasterize annotations to binary mask PNGs for VideoMamaMaskHint.

        Args:
            clip_root: Path to clip directory.
            frame_stems: Ordered list of frame stem names to export.
            width: Input frame width (pixels).
            height: Input frame height (pixels).
            start_index: Absolute frame index of the first stem (for stroke lookup
                         when exporting a sub-range via in/out points).

        Returns:
            Path to the created VideoMamaMaskHint directory.
        """
        mask_dir = os.path.join(clip_root, "VideoMamaMaskHint")
        os.makedirs(mask_dir, exist_ok=True)

        for i, stem in enumerate(frame_stems):
            abs_idx = start_index + i
            strokes = self._strokes.get(abs_idx, [])
            if strokes:
                mask = self._rasterize_strokes(strokes, width, height)
            else:
                # Unannotated frame: all-black (user chose "everything is background")
                mask = np.zeros((height, width), dtype=np.uint8)

            out_path = os.path.join(mask_dir, f"{stem}.png")
            cv2.imwrite(out_path, mask)

        logger.info(
            f"Exported {len(frame_stems)} mask frames to {mask_dir} "
            f"({self.annotated_frame_count()} annotated)"
        )
        return mask_dir

    @staticmethod
    def _rasterize_strokes(strokes: list[AnnotationStroke], width: int, height: int) -> np.ndarray:
        """Render strokes into a binary mask (0=bg, 255=fg).

        Foreground strokes draw white, background strokes draw black.
        Strokes are applied in order, so later strokes override earlier ones.
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        for stroke in strokes:
            color = 255 if stroke.brush_type == "fg" else 0
            radius = max(1, int(round(stroke.radius)))

            # Draw circles along the stroke path
            for px, py in stroke.points:
                cx, cy = int(round(px)), int(round(py))
                cv2.circle(mask, (cx, cy), radius, color, -1)

            # Connect consecutive points with thick lines for smooth strokes
            if len(stroke.points) >= 2:
                for i in range(len(stroke.points) - 1):
                    p1 = (int(round(stroke.points[i][0])), int(round(stroke.points[i][1])))
                    p2 = (int(round(stroke.points[i + 1][0])), int(round(stroke.points[i + 1][1])))
                    cv2.line(mask, p1, p2, color, radius * 2)

        return mask


# ── Colors ──────────────────────────────────────────────────────────────────

# Foreground color palette — cycle with C key
FG_PALETTE = [
    {"name": "Green", "fill": QColor(44, 195, 80, 128), "cursor": QColor(44, 195, 80, 200)},
    {"name": "Blue", "fill": QColor(50, 140, 255, 128), "cursor": QColor(50, 140, 255, 200)},
]
_fg_palette_idx = 0


def get_fg_color() -> QColor:
    return FG_PALETTE[_fg_palette_idx]["fill"]


def get_fg_cursor() -> QColor:
    return FG_PALETTE[_fg_palette_idx]["cursor"]


def get_fg_name() -> str:
    return FG_PALETTE[_fg_palette_idx]["name"]


def cycle_fg_color() -> str:
    """Advance to next foreground color. Returns the new color name."""
    global _fg_palette_idx
    _fg_palette_idx = (_fg_palette_idx + 1) % len(FG_PALETTE)
    return FG_PALETTE[_fg_palette_idx]["name"]


_BG_COLOR = QColor(209, 0, 0, 128)  # #D10000 at 50% opacity
_BG_CURSOR = QColor(209, 0, 0, 200)
_RESIZE_TEXT = QColor(255, 242, 3, 220)  # brand yellow for size text


# ── Rendering ───────────────────────────────────────────────────────────────


def paint_annotations(
    painter: QPainter,
    strokes: list[AnnotationStroke],
    current_stroke: AnnotationStroke | None,
    image_rect: QRectF,
    img_w: int,
    img_h: int,
) -> None:
    """Paint annotation strokes onto the viewport.

    Converts image-pixel coords to display coords via image_rect.
    """
    all_strokes = list(strokes)
    if current_stroke is not None:
        all_strokes.append(current_stroke)

    if not all_strokes:
        return

    painter.save()
    painter.setRenderHint(QPainter.Antialiasing)

    for stroke in all_strokes:
        color = get_fg_color() if stroke.brush_type == "fg" else _BG_COLOR
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)

        # Scale radius from image pixels to display pixels
        display_radius = stroke.radius * image_rect.width() / img_w

        display_pts = []
        for px, py in stroke.points:
            dx = image_rect.x() + (px / img_w) * image_rect.width()
            dy = image_rect.y() + (py / img_h) * image_rect.height()
            display_pts.append(QPointF(dx, dy))
            painter.drawEllipse(QPointF(dx, dy), display_radius, display_radius)

        # Connect consecutive points with thick lines for smooth coverage
        if len(display_pts) >= 2:
            pen = QPen(color, display_radius * 2)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            for i in range(len(display_pts) - 1):
                painter.drawLine(display_pts[i], display_pts[i + 1])
            painter.setPen(Qt.NoPen)

    painter.restore()


def paint_brush_cursor(
    painter: QPainter, pos: QPointF, radius_display: float, brush_type: str
) -> None:
    """Draw a circle outline at the cursor position showing brush size."""
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing)

    color = get_fg_cursor() if brush_type == "fg" else _BG_CURSOR
    pen = QPen(color, 1.5)
    pen.setStyle(Qt.SolidLine)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(pos, radius_display, radius_display)

    painter.restore()


def paint_resize_indicator(
    painter: QPainter, pos: QPointF, radius_display: float, radius_image: float, brush_type: str
) -> None:
    """Draw brush resize feedback: circle + size text."""
    paint_brush_cursor(painter, pos, radius_display, brush_type)

    # Size text below cursor
    painter.save()
    painter.setPen(_RESIZE_TEXT)
    font = painter.font()
    font.setPointSize(10)
    painter.setFont(font)
    text = f"{int(round(radius_image))}px"
    text_rect = QRectF(pos.x() - 30, pos.y() + radius_display + 4, 60, 18)
    painter.drawText(text_rect, Qt.AlignCenter, text)
    painter.restore()


def paint_annotation_hud(
    painter: QPainter,
    *,
    image_rect: QRectF,
    brush_type: str,
    radius_image: float,
) -> None:
    """Draw a persistent HUD describing the exact SAM prompt semantics."""
    painter.save()
    font = painter.font()
    font.setPointSize(10)
    painter.setFont(font)

    brush_label = "FG" if brush_type == "fg" else "BG"
    text = f"{brush_label} brush {int(round(radius_image))}px | SAM prompt: sparse points + box"
    hud_rect = QRectF(image_rect.x() + 12, image_rect.y() + 12, 380, 24)
    painter.fillRect(hud_rect, QColor(0, 0, 0, 150))
    painter.setPen(_RESIZE_TEXT)
    painter.drawText(hud_rect, Qt.AlignCenter, text)
    painter.restore()

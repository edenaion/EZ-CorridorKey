"""Convert persisted annotation strokes into tracking prompts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class AnnotationPromptFrame:
    """Prompt bundle extracted from one annotated frame."""

    frame_index: int
    positive_points: list[tuple[float, float]]
    negative_points: list[tuple[float, float]]
    box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class AnnotationMaskFrame:
    """Exact rasterized prompt mask extracted from one annotated frame."""

    frame_index: int
    mask: np.ndarray


@dataclass(frozen=True)
class AnnotationStroke:
    """One saved annotation stroke in image-pixel coordinates."""

    points: list[tuple[float, float]]
    brush_type: str = "fg"
    radius: float = 15.0


def load_annotation_mask_frames(
    clip_root: str,
    *,
    width: int,
    height: int,
    allowed_indices: Sequence[int] | None = None,
) -> list[AnnotationMaskFrame]:
    """Load exact rasterized mask prompts from ``annotations.json``."""
    raw = _load_annotations_json(clip_root)
    if raw is None:
        return []

    allowed = set(allowed_indices) if allowed_indices is not None else None
    prompt_frames: list[AnnotationMaskFrame] = []

    for frame_key, strokes_data in raw.items():
        frame_index = int(frame_key)
        if allowed is not None and frame_index not in allowed:
            continue

        strokes = [_coerce_stroke(stroke) for stroke in strokes_data]
        if not strokes:
            continue

        prompt_frames.append(
            AnnotationMaskFrame(
                frame_index=frame_index,
                mask=rasterize_annotation_strokes(strokes, width=width, height=height),
            )
        )

    prompt_frames.sort(key=lambda item: item.frame_index)
    return prompt_frames


def load_annotation_prompt_frames(
    clip_root: str,
    *,
    allowed_indices: Sequence[int] | None = None,
    max_points_per_stroke: int | None = None,
    max_points_per_frame: int | None = 1024,
    max_positive_points_per_frame: int | None = 24,
    max_negative_points_per_stroke: int | None = 12,
    max_negative_points_per_frame: int | None = 8,
) -> list[AnnotationPromptFrame]:
    """Load annotation prompts from ``annotations.json`` without importing UI code."""
    raw = _load_annotations_json(clip_root)
    if raw is None:
        return []

    allowed = set(allowed_indices) if allowed_indices is not None else None
    prompt_frames: list[AnnotationPromptFrame] = []

    for frame_key, strokes in raw.items():
        frame_index = int(frame_key)
        if allowed is not None and frame_index not in allowed:
            continue

        positives: list[tuple[float, float]] = []
        negatives: list[tuple[float, float]] = []
        positive_extents: list[tuple[float, float, float, float]] = []
        for stroke in strokes:
            brush_type = stroke.get("brush_type", "fg")
            radius = float(stroke.get("radius", 15.0))
            if brush_type == "bg":
                bg_points = _sample_points(
                    stroke.get("points", []),
                    _effective_cap(max_points_per_stroke, max_negative_points_per_stroke),
                )
                negatives.extend(bg_points)
            else:
                points = _sample_points(stroke.get("points", []), max_points_per_stroke)
                expanded_points = _expand_points_for_brush(points, radius)
                positives.extend(expanded_points)
                positive_extents.extend(_points_to_extents(points, radius))

        if not positives and not negatives:
            continue

        prompt_frames.append(
            AnnotationPromptFrame(
                frame_index=frame_index,
                positive_points=_cap_points(
                    _dedupe_points(positives),
                    _effective_cap(max_points_per_frame, max_positive_points_per_frame),
                ),
                negative_points=_cap_points(
                    _dedupe_points(negatives),
                    _effective_cap(max_points_per_frame, max_negative_points_per_frame),
                ),
                box=_bounding_box_from_extents(positive_extents),
            )
        )

    prompt_frames.sort(key=lambda item: item.frame_index)
    return prompt_frames


def rasterize_annotation_strokes(
    strokes: Sequence[AnnotationStroke],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Render strokes into a binary mask exactly like the UI export path."""
    mask = np.zeros((height, width), dtype=np.uint8)

    for stroke in strokes:
        color = 255 if stroke.brush_type == "fg" else 0
        radius = max(1, int(round(stroke.radius)))

        for px, py in stroke.points:
            cx, cy = int(round(px)), int(round(py))
            cv2.circle(mask, (cx, cy), radius, color, -1)

        if len(stroke.points) >= 2:
            for i in range(len(stroke.points) - 1):
                p1 = (
                    int(round(stroke.points[i][0])),
                    int(round(stroke.points[i][1])),
                )
                p2 = (
                    int(round(stroke.points[i + 1][0])),
                    int(round(stroke.points[i + 1][1])),
                )
                cv2.line(mask, p1, p2, color, radius * 2)

    return mask


def _load_annotations_json(clip_root: str) -> dict[str, list[dict]] | None:
    path = os.path.join(clip_root, "annotations.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_stroke(raw: dict) -> AnnotationStroke:
    points = []
    for point in raw.get("points", []):
        if not isinstance(point, Sequence) or len(point) != 2:
            continue
        points.append((float(point[0]), float(point[1])))
    return AnnotationStroke(
        points=points,
        brush_type=str(raw.get("brush_type", "fg")),
        radius=float(raw.get("radius", 15.0)),
    )


def _sample_points(
    points: Iterable[Sequence[float]],
    limit: int | None,
) -> list[tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in points]
    if limit is None:
        return pts
    if limit <= 0:
        return []
    if len(pts) <= limit:
        return pts
    indices = np.linspace(0, len(pts) - 1, num=limit, dtype=int)
    return [pts[i] for i in indices.tolist()]


def _expand_points_for_brush(
    points: Sequence[tuple[float, float]],
    radius: float,
    max_prompt_points: int = 96,
) -> list[tuple[float, float]]:
    """Approximate brush radius with bounded prompt density."""
    if not points:
        return []

    r = max(0.0, float(radius))
    if r < 1.0:
        return list(points)

    # Large brushes do not need point-level density at every stroke sample.
    stride = max(1, int(round(r / 10.0)))
    anchors = list(points[::stride])
    if points[-1] != anchors[-1]:
        anchors.append(points[-1])

    half = r * 0.5
    diag = r * 0.35
    offsets = [
        (0.0, 0.0),
        (half, 0.0),
        (-half, 0.0),
        (0.0, half),
        (0.0, -half),
        (diag, diag),
        (diag, -diag),
        (-diag, diag),
        (-diag, -diag),
    ]

    expanded: list[tuple[float, float]] = []
    for x, y in anchors:
        for ox, oy in offsets:
            expanded.append((x + ox, y + oy))
    # Keep sparse centerline detail so thin structures are not lost.
    center_stride = max(1, stride // 2)
    expanded.extend(points[::center_stride])
    if points[-1] != expanded[-1]:
        expanded.append(points[-1])

    if max_prompt_points > 0 and len(expanded) > max_prompt_points:
        idx = np.linspace(0, len(expanded) - 1, num=max_prompt_points, dtype=int)
        expanded = [expanded[i] for i in idx.tolist()]
    return expanded


def _dedupe_points(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    seen: set[tuple[int, int]] = set()
    result: list[tuple[float, float]] = []
    for x, y in points:
        key = (int(round(x)), int(round(y)))
        if key in seen:
            continue
        seen.add(key)
        result.append((float(key[0]), float(key[1])))
    return result


def _bounding_box(
    points: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def _points_to_extents(
    points: Sequence[tuple[float, float]],
    radius: float,
) -> list[tuple[float, float, float, float]]:
    r = max(0.0, float(radius))
    return [(x - r, y - r, x + r, y + r) for x, y in points]


def _bounding_box_from_extents(
    extents: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not extents:
        return None
    return (
        float(min(e[0] for e in extents)),
        float(min(e[1] for e in extents)),
        float(max(e[2] for e in extents)),
        float(max(e[3] for e in extents)),
    )


def _cap_points(
    points: Sequence[tuple[float, float]],
    max_points: int | None,
) -> list[tuple[float, float]]:
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return list(points)
    idx = np.linspace(0, len(points) - 1, num=max_points, dtype=int)
    return [points[i] for i in idx.tolist()]


def _effective_cap(*caps: int | None) -> int | None:
    valid = [cap for cap in caps if cap is not None and cap > 0]
    if not valid:
        return None
    return min(valid)

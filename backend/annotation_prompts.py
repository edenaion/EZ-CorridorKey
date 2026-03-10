"""Convert persisted annotation strokes into sparse tracking prompts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class AnnotationPromptFrame:
    """Prompt bundle extracted from one annotated frame."""

    frame_index: int
    positive_points: list[tuple[float, float]]
    negative_points: list[tuple[float, float]]
    box: tuple[float, float, float, float] | None = None


def load_annotation_prompt_frames(
    clip_root: str,
    *,
    allowed_indices: Sequence[int] | None = None,
    max_points_per_stroke: int = 8,
) -> list[AnnotationPromptFrame]:
    """Load annotation prompts from ``annotations.json`` without importing UI code."""
    path = os.path.join(clip_root, "annotations.json")
    if not os.path.isfile(path):
        return []

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    allowed = set(allowed_indices) if allowed_indices is not None else None
    prompt_frames: list[AnnotationPromptFrame] = []

    for frame_key, strokes in raw.items():
        frame_index = int(frame_key)
        if allowed is not None and frame_index not in allowed:
            continue

        positives: list[tuple[float, float]] = []
        negatives: list[tuple[float, float]] = []
        for stroke in strokes:
            points = _sample_points(stroke.get("points", []), max_points_per_stroke)
            brush_type = stroke.get("brush_type", "fg")
            if brush_type == "bg":
                negatives.extend(points)
            else:
                positives.extend(points)

        if not positives and not negatives:
            continue

        prompt_frames.append(
            AnnotationPromptFrame(
                frame_index=frame_index,
                positive_points=_dedupe_points(positives),
                negative_points=_dedupe_points(negatives),
                box=_bounding_box(positives),
            )
        )

    prompt_frames.sort(key=lambda item: item.frame_index)
    return prompt_frames


def _sample_points(points: Iterable[Sequence[float]], limit: int) -> list[tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) <= limit:
        return pts
    indices = np.linspace(0, len(pts) - 1, num=limit, dtype=int)
    return [pts[i] for i in indices.tolist()]


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


def _bounding_box(points: Sequence[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

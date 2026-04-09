"""Wipe (A/B comparison) geometry and rendering helper.

Encapsulates wipe-line geometry calculations, hit-testing, drag logic,
and painting. Used by SplitViewWidget to keep wipe concerns separate.

All methods are pure functions of explicit parameters — no widget state owned here.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QPolygonF, QPainterPath


# ── Geometry helpers ──

def wipe_line_endpoints(
    width: float, height: float,
    wipe_angle: float, wipe_offset: float,
) -> tuple[QPointF, QPointF, QPointF]:
    """Compute the wipe line endpoints from angle + offset.

    Returns (start, end, center) in widget coords.
    """
    cx, cy = width / 2.0, height / 2.0

    angle_rad = math.radians(wipe_angle)
    perp_x = -math.sin(angle_rad)
    perp_y = math.cos(angle_rad)
    diag = math.sqrt(width * width + height * height)
    offset_px = wipe_offset * diag

    # Center of line shifted by offset
    lx = cx + perp_x * offset_px
    ly = cy + perp_y * offset_px

    # Line direction (along the angle)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    # Extend line well beyond viewport
    ext = diag
    p1 = QPointF(lx - dx * ext, ly - dy * ext)
    p2 = QPointF(lx + dx * ext, ly + dy * ext)
    center = QPointF(lx, ly)
    return p1, p2, center


def wipe_handle_rect(center: QPointF, hit: bool = False) -> QRectF:
    """Return the center square handle rect. hit=True returns 2x hitbox."""
    s = 12.0 if hit else 6.0
    return QRectF(center.x() - s, center.y() - s, s * 2, s * 2)


def wipe_distance_to_line(
    pos: QPointF,
    width: float, height: float,
    wipe_angle: float, wipe_offset: float,
) -> float:
    """Signed perpendicular distance from pos to the wipe line (pixels)."""
    angle_rad = math.radians(wipe_angle)
    _, _, center = wipe_line_endpoints(width, height, wipe_angle, wipe_offset)
    nx = -math.sin(angle_rad)
    ny = math.cos(angle_rad)
    return (pos.x() - center.x()) * nx + (pos.y() - center.y()) * ny


# ── Painting ──

def paint_wipe(
    painter: QPainter,
    width: float, height: float,
    wipe_angle: float, wipe_offset: float,
    left_image, right_image,
    image_rect_fn,
    divider_width: int = 2,
) -> None:
    """Draw A/B wipe comparison with diagonal divider.

    Parameters
    ----------
    painter : QPainter
        Active painter on the widget.
    width, height : float
        Widget dimensions.
    wipe_angle, wipe_offset : float
        Current wipe line angle and offset.
    left_image, right_image : QImage
        The A (input) and B (output) images.
    image_rect_fn : callable(QImage) -> QRectF
        Function to compute destination rect for an image (handles zoom/pan).
    divider_width : int
        Width of the wipe divider line.
    """
    p1, p2, center = wipe_line_endpoints(width, height, wipe_angle, wipe_offset)

    # Build clip path for side A (above/left of line)
    angle_rad = math.radians(wipe_angle)
    perp_x = -math.sin(angle_rad)
    perp_y = math.cos(angle_rad)

    far = max(width, height) * 2
    a1 = QPointF(p1.x() - perp_x * far, p1.y() - perp_y * far)
    a2 = QPointF(p2.x() - perp_x * far, p2.y() - perp_y * far)

    side_a_poly = QPolygonF([p1, p2, a2, a1])
    side_a_path = QPainterPath()
    side_a_path.addPolygon(side_a_poly)

    # Side A: INPUT image (above/left of line)
    if left_image:
        dest = image_rect_fn(left_image)
        painter.setClipPath(side_a_path)
        painter.drawImage(dest, left_image)

    # Side B: OUTPUT image (below/right of line)
    if right_image:
        dest = image_rect_fn(right_image)
        full = QPainterPath()
        full.addRect(QRectF(0, 0, width, height))
        side_b_path = full - side_a_path
        painter.setClipPath(side_b_path)
        painter.drawImage(dest, right_image)

    # Remove clip for divider drawing
    painter.setClipping(False)

    # Draw wipe line
    pen = QPen(QColor("#FFF203"), divider_width)
    painter.setPen(pen)
    painter.drawLine(p1, p2)

    # Draw center handle (filled square)
    handle = wipe_handle_rect(center)
    painter.setBrush(QColor("#FFF203"))
    painter.setPen(Qt.NoPen)
    painter.drawRect(handle)

    # Draw A/B labels near the line
    painter.setPen(QColor("#FFF203"))
    font = painter.font()
    font.setPointSize(10)
    font.setBold(True)
    painter.setFont(font)

    label_offset = 16
    painter.drawText(
        QPointF(center.x() - perp_x * label_offset - 4,
                center.y() - perp_y * label_offset + 4), "A")
    painter.drawText(
        QPointF(center.x() + perp_x * label_offset - 4,
                center.y() + perp_y * label_offset + 4), "B")


# ── Drag handling ──

def handle_wipe_press(
    pos: QPointF,
    width: float, height: float,
    wipe_angle: float, wipe_offset: float,
    hit_zone: int,
) -> tuple[str | None, QPointF, float, float]:
    """Check if a left-click hits the wipe handle or line.

    Returns (drag_type, drag_start, start_offset, start_angle) where
    drag_type is "handle", "line", or None.
    """
    _, _, center = wipe_line_endpoints(width, height, wipe_angle, wipe_offset)
    if wipe_handle_rect(center, hit=True).contains(pos):
        return "handle", pos, wipe_offset, wipe_angle
    dist = abs(wipe_distance_to_line(pos, width, height, wipe_angle, wipe_offset))
    if dist < hit_zone:
        return "line", pos, wipe_offset, wipe_angle
    return None, pos, wipe_offset, wipe_angle


def handle_wipe_drag(
    drag_type: str,
    event_pos: QPointF,
    drag_start: QPointF,
    start_offset: float,
    start_angle: float,
    width: float, height: float,
    wipe_angle: float,
) -> tuple[float, float]:
    """Process a wipe drag move event.

    Returns (new_offset, new_angle).
    """
    if drag_type == "handle":
        angle_rad = math.radians(wipe_angle)
        nx = -math.sin(angle_rad)
        ny = math.cos(angle_rad)
        dx = event_pos.x() - drag_start.x()
        dy = event_pos.y() - drag_start.y()
        diag = math.sqrt(width ** 2 + height ** 2)
        delta_offset = (dx * nx + dy * ny) / diag
        new_offset = max(-0.5, min(0.5, start_offset + delta_offset))
        return new_offset, wipe_angle

    if drag_type == "line":
        cx, cy = width / 2.0, height / 2.0
        mx = event_pos.x() - cx
        my = event_pos.y() - cy
        angle = math.degrees(math.atan2(my, mx))
        if angle > 90.0:
            angle = 90.0
        elif angle < -90.0:
            angle = -90.0
        return wipe_angle, angle  # offset unchanged, angle changed

    return start_offset, wipe_angle  # no-op fallback


def wipe_cursor_for_pos(
    pos: QPointF,
    width: float, height: float,
    wipe_angle: float, wipe_offset: float,
    hit_zone: int,
) -> Qt.CursorShape | None:
    """Return cursor shape for wipe hover, or None if not near wipe elements."""
    _, _, center = wipe_line_endpoints(width, height, wipe_angle, wipe_offset)
    if wipe_handle_rect(center, hit=True).contains(pos):
        return Qt.SizeAllCursor
    if abs(wipe_distance_to_line(pos, width, height, wipe_angle, wipe_offset)) < hit_zone:
        return Qt.OpenHandCursor
    return None


def handle_wipe_scroll(
    delta: int,
    shift: bool,
    wipe_offset: float,
) -> float:
    """Handle scroll wheel in wipe mode. Returns new offset."""
    if shift:
        return max(-0.5, min(0.5, wipe_offset - delta / 15000.0))
    else:
        step = -0.03 if delta > 0 else 0.03
        return max(-0.5, min(0.5, wipe_offset + step))

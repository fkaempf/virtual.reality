"""Arena geometry utilities for circular boundary enforcement.

Functions for clamping positions to a circular arena, enforcing
minimum distances between entities, and computing 3D separations.
"""

from __future__ import annotations

import math


def clamp_to_arena(
    x: float,
    y: float,
    radius: float,
) -> tuple[float, float]:
    """Clamp a 2D position to stay inside a circular arena.

    If the point lies outside the arena, it is projected back onto
    the boundary circle.

    Args:
        x: X coordinate.
        y: Y coordinate.
        radius: Arena radius in the same units as *x* and *y*.

    Returns:
        The clamped ``(x, y)`` position.
    """
    dist = math.hypot(x, y)
    if dist > radius:
        scale = radius / max(dist, 1e-9)
        return x * scale, y * scale
    return x, y


def enforce_min_distance(
    pos: tuple[float, float],
    other: tuple[float, float],
    min_dist: float,
) -> tuple[float, float]:
    """Push *pos* away from *other* so they are at least *min_dist* apart.

    If the two positions overlap exactly, *pos* is pushed along +X.

    Args:
        pos: The position to adjust as ``(x, y)``.
        other: The reference position as ``(x, y)``.
        min_dist: Minimum allowed distance.

    Returns:
        The adjusted ``(x, y)`` for *pos*.
    """
    px, py = pos
    ox, oy = other
    dx = px - ox
    dy = py - oy
    dist = math.hypot(dx, dy)
    if dist < min_dist:
        if dist < 1e-6:
            px = ox + min_dist
            py = oy
        else:
            scale = min_dist / dist
            px = ox + dx * scale
            py = oy + dy * scale
    return px, py


def compute_camera_fly_distance_mm(
    fly_pos: tuple[float, float],
    cam_pos: tuple[float, float],
    cam_height_mm: float,
) -> float:
    """Compute the 3D distance between camera and fly.

    The fly is assumed to be on the arena plane (height = 0).

    Args:
        fly_pos: Fly position as ``(x, y)`` in mm.
        cam_pos: Camera position as ``(x, y)`` in mm.
        cam_height_mm: Camera height above the arena plane in mm.

    Returns:
        The 3D Euclidean distance in mm.
    """
    fx, fy = fly_pos
    cx, cy = cam_pos
    dx = fx - cx
    dy = fy - cy
    return math.sqrt(dx * dx + dy * dy + cam_height_mm * cam_height_mm)

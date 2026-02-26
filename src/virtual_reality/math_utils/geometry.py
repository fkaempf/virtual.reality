"""Angle and direction helpers for fly/camera heading math."""

from __future__ import annotations

import math

import numpy as np


def compute_light_dirs(elevation_deg: float) -> np.ndarray:
    """Compute four directional light directions from compass headings.

    Lights come from N, E, S, W at the given elevation above the
    horizon. Each direction points *toward* the light source (the
    shader uses ``dot(normal, light_dir)`` directly).

    Args:
        elevation_deg: Angle above the horizon in degrees. 0 = horizon,
            90 = straight down.

    Returns:
        A ``(4, 3)`` float32 array of normalized light directions.
    """
    elev = math.radians(elevation_deg)
    horiz = math.cos(elev)
    up = math.sin(elev)
    dirs = [
        np.array([0.0, up, -horiz], dtype=np.float32),
        np.array([-horiz, up, 0.0], dtype=np.float32),
        np.array([0.0, up, horiz], dtype=np.float32),
        np.array([horiz, up, 0.0], dtype=np.float32),
    ]
    dirs = [d / max(np.linalg.norm(d), 1e-6) for d in dirs]
    return np.stack(dirs, axis=0)


def heading_to_direction(
    heading_deg: float,
) -> tuple[float, float]:
    """Convert a heading angle to a 2D unit direction vector.

    Convention: 0 degrees = +Y (forward), 90 degrees = +X (right).

    Args:
        heading_deg: Heading in degrees.

    Returns:
        A ``(dx, dy)`` unit direction tuple.
    """
    rad = math.radians(heading_deg)
    return math.sin(rad), math.cos(rad)


def wrap_angle_deg(angle: float) -> float:
    """Wrap an angle to the range ``[0, 360)``.

    Args:
        angle: Angle in degrees.

    Returns:
        The wrapped angle.
    """
    return angle % 360.0

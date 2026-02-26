"""Projection matrix utilities.

Provides ``perspective_matrix`` and ``look_at_matrix`` for building
view and projection transforms independent of the rendering backend.
These duplicate the math_utils versions but operate as standalone
helpers for the rendering module.
"""

from __future__ import annotations

import math

import numpy as np


# Projection mode constants matching the GLSL u_projMode uniform.
PROJ_PERSPECTIVE = 0
PROJ_EQUIDISTANT = 1
PROJ_EQUIRECTANGULAR = 2

PROJ_MODE_MAP: dict[str, int] = {
    "perspective": PROJ_PERSPECTIVE,
    "equidistant": PROJ_EQUIDISTANT,
    "equirect": PROJ_EQUIRECTANGULAR,
}


def projection_mode_to_int(mode: str) -> int:
    """Convert a projection mode name to its GLSL integer constant.

    Args:
        mode: One of ``"perspective"``, ``"equidistant"``,
            ``"equirect"``.

    Returns:
        Integer constant for the ``u_projMode`` uniform.

    Raises:
        ValueError: If *mode* is not recognised.
    """
    key = mode.lower().strip()
    if key not in PROJ_MODE_MAP:
        raise ValueError(
            f"Unknown projection mode {mode!r}; "
            f"choose from {list(PROJ_MODE_MAP)}",
        )
    return PROJ_MODE_MAP[key]

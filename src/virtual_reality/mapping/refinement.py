"""Warp map refinement using a dot-grid visibility probe.

The refinement step projects individual dots at grid positions and
captures the camera response to determine which projector pixels are
actually visible.  The result is a tighter validity mask that clips
the warp maps to the physically visible region.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def expand_mask(
    mask: np.ndarray,
    buffer_px: int,
) -> np.ndarray:
    """Dilate a boolean mask by *buffer_px* pixels.

    Args:
        mask: Boolean 2-D array.
        buffer_px: Number of pixels to expand in each direction.

    Returns:
        Dilated boolean mask.
    """
    if buffer_px <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * buffer_px + 1, 2 * buffer_px + 1),
    )
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    return dilated.astype(bool)


def refine_with_visibility(
    mapx: np.ndarray,
    mapy: np.ndarray,
    valid_mask: np.ndarray,
    visibility_map: np.ndarray,
    buffer_px: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine warp maps by clipping to a visibility mask.

    Args:
        mapx: Projector-to-camera X map ``(proj_h, proj_w)``.
        mapy: Projector-to-camera Y map ``(proj_h, proj_w)``.
        valid_mask: Current validity mask.
        visibility_map: Boolean map of projector pixels confirmed
            visible by a dot-grid probe.
        buffer_px: Dilation buffer to account for edge effects.

    Returns:
        Refined ``(mapx, mapy, new_valid)`` with invisible pixels
        set to ``NaN``.
    """
    expanded = expand_mask(visibility_map, buffer_px)
    new_valid = valid_mask & expanded

    mx = mapx.copy()
    my = mapy.copy()
    mx[~new_valid] = np.nan
    my[~new_valid] = np.nan

    n_before = valid_mask.sum()
    n_after = new_valid.sum()
    logger.info(
        "Refinement: %d → %d valid pixels (removed %d)",
        n_before, n_after, n_before - n_after,
    )
    return mx, my, new_valid

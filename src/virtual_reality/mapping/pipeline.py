"""Mapping pipeline: post-processing for projector-camera maps.

Provides despeckling, inpainting, and the high-level
:func:`process_raw_maps` function that chains all post-processing
steps on raw ``proj_x``/``proj_y`` measurements.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from virtual_reality.mapping.warp import build_proj_to_cam_map

logger = logging.getLogger(__name__)


def despeckle_maps(
    mapx: np.ndarray,
    mapy: np.ndarray,
    valid_mask: np.ndarray,
    k_med: int = 3,
    k_avg: int = 5,
    tolerance: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove speckle artefacts from warp maps.

    Replaces outlier pixels (those that deviate from the local median
    by more than *tolerance*) with a local box-filtered average.

    Args:
        mapx: Float32 camera-X map ``(proj_h, proj_w)``.
        mapy: Float32 camera-Y map ``(proj_h, proj_w)``.
        valid_mask: Boolean validity mask.
        k_med: Median filter kernel size (odd).
        k_avg: Box filter kernel size (odd).
        tolerance: Maximum deviation from median in pixels.

    Returns:
        Despeckled ``(mapx, mapy)`` copies.
    """
    mx = mapx.copy()
    my = mapy.copy()

    for arr in (mx, my):
        work = arr.copy()
        work[~valid_mask] = np.nan

        med = cv2.medianBlur(
            np.nan_to_num(work, nan=0.0).astype(np.float32), k_med,
        )
        diff = np.abs(work - med)
        bad = diff > tolerance

        avg = cv2.blur(
            np.nan_to_num(work, nan=0.0).astype(np.float32),
            (k_avg, k_avg),
        )
        arr[bad & valid_mask] = avg[bad & valid_mask]

    return mx, my


def inpaint_invalid(
    mapx: np.ndarray,
    mapy: np.ndarray,
    valid_mask: np.ndarray,
    inpaint_radius: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill invalid regions using Navier-Stokes inpainting.

    Args:
        mapx: Float32 camera-X map.
        mapy: Float32 camera-Y map.
        valid_mask: Boolean validity mask.
        inpaint_radius: Inpainting neighbourhood radius.

    Returns:
        Inpainted ``(mapx, mapy)`` copies.
    """
    mask_u8 = (~valid_mask).astype(np.uint8) * 255

    mx = cv2.inpaint(
        np.nan_to_num(mapx, nan=0.0).astype(np.float32),
        mask_u8, inpaint_radius, cv2.INPAINT_NS,
    )
    my = cv2.inpaint(
        np.nan_to_num(mapy, nan=0.0).astype(np.float32),
        mask_u8, inpaint_radius, cv2.INPAINT_NS,
    )
    return mx, my


def process_raw_maps(
    proj_x: np.ndarray,
    proj_y: np.ndarray,
    proj_w: int,
    proj_h: int,
    valid_mask: np.ndarray | None = None,
    despeckle: bool = True,
    inpaint: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process raw structured-light measurements into final warp maps.

    Steps:
    1. Invert camera-to-projector measurements into projector-to-camera
       maps using :func:`build_proj_to_cam_map`.
    2. Optionally despeckle outlier pixels.
    3. Optionally inpaint invalid regions.

    Args:
        proj_x: Camera-pixel array of projector X coordinates.
        proj_y: Camera-pixel array of projector Y coordinates.
        proj_w: Output projector width.
        proj_h: Output projector height.
        valid_mask: Optional validity mask for measurements.
        despeckle: Whether to apply despeckle filtering.
        inpaint: Whether to inpaint invalid regions.

    Returns:
        ``(mapx, mapy, valid)`` in projector space.
    """
    mapx, mapy = build_proj_to_cam_map(
        proj_x, proj_y, proj_w, proj_h, valid_mask,
    )
    valid = np.isfinite(mapx) & np.isfinite(mapy)
    logger.info(
        "Raw map: %d / %d valid pixels (%.1f%%)",
        valid.sum(), valid.size, 100.0 * valid.sum() / valid.size,
    )

    if despeckle:
        mapx, mapy = despeckle_maps(mapx, mapy, valid)

    if inpaint:
        mapx, mapy = inpaint_invalid(mapx, mapy, valid)

    return mapx, mapy, valid

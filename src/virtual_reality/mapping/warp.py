"""Projector-camera warp map loading and conversion.

Loads ``mapx.npy``/``mapy.npy`` arrays that define the pixel mapping
from projector space to camera space, and prepares them for use as
OpenGL textures or OpenCV remap arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WarpMap:
    """A projector-to-camera warp mapping.

    Attributes:
        mapx: Float32 array ``(proj_h, proj_w)`` of camera X coords.
        mapy: Float32 array ``(proj_h, proj_w)`` of camera Y coords.
        cam_w: Inferred camera image width.
        cam_h: Inferred camera image height.
        proj_w: Projector width.
        proj_h: Projector height.
        valid_mask: Boolean array marking valid (finite, non-negative)
            entries.
    """

    mapx: np.ndarray
    mapy: np.ndarray
    cam_w: int
    cam_h: int
    proj_w: int
    proj_h: int
    valid_mask: np.ndarray


def load_warp_map(
    mapx_path: str | Path,
    mapy_path: str | Path,
    factor: float = 1.0,
) -> WarpMap:
    """Load a projector-camera warp map from numpy files.

    The maps are flipped vertically to match OpenGL conventions, and
    camera resolution is inferred from the maximum valid coordinates.

    Args:
        mapx_path: Path to the ``mapx.npy`` file.
        mapy_path: Path to the ``mapy.npy`` file.
        factor: Divisor applied to raw coordinates (for subsampling).

    Returns:
        A ``WarpMap`` with all fields populated.

    Raises:
        RuntimeError: If the maps contain no valid entries.
        FileNotFoundError: If either file does not exist.
    """
    mapx = np.load(str(mapx_path)).astype(np.float32) / factor
    mapy = np.load(str(mapy_path)).astype(np.float32) / factor

    mapx = np.flipud(mapx)
    mapy = np.flipud(mapy)

    proj_h, proj_w = mapx.shape
    logger.info(
        "Loaded warp map: projector size = %d x %d", proj_w, proj_h,
    )

    valid = (
        np.isfinite(mapx)
        & np.isfinite(mapy)
        & (mapx >= 0)
        & (mapy >= 0)
    )
    if not np.any(valid):
        raise RuntimeError("mapx/mapy contain no valid entries")

    cam_w = int(np.ceil(mapx[valid].max())) + 1
    cam_h = int(np.ceil(mapy[valid].max())) + 1
    logger.info(
        "Inferred camera size = %d x %d", cam_w, cam_h,
    )

    return WarpMap(
        mapx=mapx,
        mapy=mapy,
        cam_w=cam_w,
        cam_h=cam_h,
        proj_w=proj_w,
        proj_h=proj_h,
        valid_mask=valid,
    )


def warp_to_gl_texture(warp: WarpMap) -> np.ndarray:
    """Convert a WarpMap to a normalized UV texture for OpenGL.

    Produces a ``(proj_h, proj_w, 2)`` float32 array where channel 0
    is the normalized U coordinate and channel 1 is the normalized V
    coordinate. Invalid pixels are set to ``-1.0``.

    Args:
        warp: A loaded ``WarpMap``.

    Returns:
        A float32 array of shape ``(proj_h, proj_w, 2)`` with values
        in ``[0, 1]`` for valid pixels.
    """
    tex = np.zeros(
        (warp.proj_h, warp.proj_w, 2), dtype=np.float32,
    )
    tex[..., 0] = warp.mapx / float(warp.cam_w)
    tex[..., 1] = warp.mapy / float(warp.cam_h)
    tex[~warp.valid_mask, 0] = -1.0
    tex[~warp.valid_mask, 1] = -1.0
    return tex


def build_proj_to_cam_map(
    proj_x: np.ndarray,
    proj_y: np.ndarray,
    proj_w: int,
    proj_h: int,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a camera-to-projector measurement into remap arrays.

    Given per-camera-pixel projector coordinates, produces per-projector-
    pixel camera coordinates suitable for ``cv2.remap()``.

    Args:
        proj_x: Camera-pixel array of projector X coordinates.
        proj_y: Camera-pixel array of projector Y coordinates.
        proj_w: Output projector width.
        proj_h: Output projector height.
        valid_mask: Boolean mask for valid measurements. If ``None``,
            all finite pixels are used.

    Returns:
        A tuple ``(mapx, mapy)`` of float32 arrays with shape
        ``(proj_h, proj_w)``.
    """
    cam_h, cam_w = proj_x.shape[:2]
    yy, xx = np.mgrid[0:cam_h, 0:cam_w]

    if valid_mask is None:
        valid_mask = np.isfinite(proj_x) & np.isfinite(proj_y)

    ok = valid_mask.ravel()
    pu = np.round(proj_x.ravel()[ok]).astype(int)
    pv = np.round(proj_y.ravel()[ok]).astype(int)
    cx = xx.ravel()[ok].astype(np.float32)
    cy = yy.ravel()[ok].astype(np.float32)

    in_bounds = (pu >= 0) & (pu < proj_w) & (pv >= 0) & (pv < proj_h)
    pu, pv, cx, cy = pu[in_bounds], pv[in_bounds], cx[in_bounds], cy[in_bounds]

    mapx = np.full((proj_h, proj_w), np.nan, dtype=np.float32)
    mapy = np.full((proj_h, proj_w), np.nan, dtype=np.float32)
    mapx[pv, pu] = cx
    mapy[pv, pu] = cy

    return mapx, mapy

"""Intrinsics I/O: load and save camera calibration parameters.

Provides a unified :class:`Intrinsics` container and convenience
functions to persist calibration data as ``.npy`` files compatible
with the legacy ``configs/`` directory layout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Intrinsics:
    """Camera intrinsic parameters.

    Supports both pinhole and fisheye (omnidir/MEI) models.

    Attributes:
        K: 3x3 camera intrinsic matrix.
        D: Distortion coefficient array.
        xi: MEI model parameter (``None`` for pinhole cameras).
        model: ``"pinhole"`` or ``"fisheye"``.
    """

    K: np.ndarray
    D: np.ndarray
    xi: float | None = None
    model: str = "pinhole"

    @property
    def is_fisheye(self) -> bool:
        """Whether this represents a fisheye/omnidir model."""
        return self.xi is not None


def load_intrinsics(directory: str | Path) -> Intrinsics:
    """Load intrinsics from a directory of ``.npy`` files.

    Looks for either ``fisheye.K.npy`` / ``fisheye.D.npy`` /
    ``fisheye.xi.npy`` (fisheye model) or ``pinhole.K.npy`` /
    ``pinhole.D.npy`` (pinhole model).

    Args:
        directory: Path to the config directory.

    Returns:
        An :class:`Intrinsics` instance.

    Raises:
        FileNotFoundError: If no valid calibration files are found.
    """
    d = Path(directory)

    # Try fisheye first
    fisheye_K = d / "fisheye.K.npy"
    if fisheye_K.exists():
        K = np.load(str(fisheye_K))
        D = np.load(str(d / "fisheye.D.npy"))
        xi_arr = np.load(str(d / "fisheye.xi.npy"))
        xi = float(xi_arr.ravel()[0])
        logger.info("Loaded fisheye intrinsics from %s", d)
        return Intrinsics(K=K, D=D, xi=xi, model="fisheye")

    # Try pinhole
    pinhole_K = d / "pinhole.K.npy"
    if pinhole_K.exists():
        K = np.load(str(pinhole_K))
        D = np.load(str(d / "pinhole.D.npy"))
        logger.info("Loaded pinhole intrinsics from %s", d)
        return Intrinsics(K=K, D=D, model="pinhole")

    raise FileNotFoundError(
        f"No calibration files (fisheye.K.npy or pinhole.K.npy) in {d}",
    )


def save_intrinsics(
    intrinsics: Intrinsics,
    directory: str | Path,
) -> None:
    """Save intrinsics as ``.npy`` files.

    Args:
        intrinsics: The intrinsics to save.
        directory: Target directory (created if it does not exist).
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)

    prefix = "fisheye" if intrinsics.is_fisheye else "pinhole"
    np.save(str(d / f"{prefix}.K.npy"), intrinsics.K)
    np.save(str(d / f"{prefix}.D.npy"), intrinsics.D)
    if intrinsics.is_fisheye:
        np.save(
            str(d / f"{prefix}.xi.npy"),
            np.array([intrinsics.xi], dtype=np.float64),
        )
    logger.info("Saved %s intrinsics to %s", prefix, d)


def build_rectify_maps(
    K: np.ndarray,
    D: np.ndarray,
    xi: np.ndarray | float,
    width: int,
    height: int,
    zoom: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build undistortion rectification maps for a fisheye camera.

    Uses ``cv2.omnidir.initUndistortRectifyMap`` to generate remap
    arrays suitable for ``cv2.remap()``.

    Args:
        K: 3x3 intrinsic matrix.
        D: 1x4 distortion coefficients.
        xi: MEI model parameter.
        width: Output image width.
        height: Output image height.
        zoom: Zoom factor applied to focal lengths.

    Returns:
        ``(map1, map2)`` arrays for ``cv2.remap()``.
    """
    import cv2

    K = np.ascontiguousarray(np.asarray(K, np.float64).reshape(3, 3))
    D = np.ascontiguousarray(np.asarray(D, np.float64).reshape(1, 4))
    xi_arr = np.array(
        [float(np.asarray(xi).ravel()[0])], dtype=np.float64,
    )

    Knew = K.copy()
    Knew[0, 0] *= zoom
    Knew[1, 1] *= zoom
    Knew[0, 2] = width / 2.0
    Knew[1, 2] = height / 2.0

    map1, map2 = cv2.omnidir.initUndistortRectifyMap(
        K, D, xi_arr,
        np.eye(3, dtype=np.float64),
        Knew, (int(width), int(height)),
        5, 1,
    )
    return map1, map2

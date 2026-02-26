"""Pinhole camera calibration using ChArUco boards.

Extracts and refactors the calibration logic from ``pinhole_KD.py``
and ``detect_pose.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CharucoBoardConfig:
    """Configuration for a ChArUco calibration board.

    Attributes:
        squares_x: Number of squares along X (horizontal).
        squares_y: Number of squares along Y (vertical).
        square_length: Physical length of one chessboard square (m).
        marker_length: Physical length of one ArUco marker (m).
        aruco_dict_id: OpenCV ArUco dictionary identifier.
    """

    squares_x: int = 7
    squares_y: int = 5
    square_length: float = 0.03
    marker_length: float = 0.015
    aruco_dict_id: int = cv2.aruco.DICT_6X6_250


@dataclass
class PinholeResult:
    """Result of a pinhole camera calibration.

    Attributes:
        K: 3x3 camera intrinsic matrix.
        D: Distortion coefficients.
        rms: Overall RMS reprojection error.
        rvecs: Per-view rotation vectors.
        tvecs: Per-view translation vectors.
        image_shape: ``(height, width)`` of calibration images.
    """

    K: np.ndarray
    D: np.ndarray
    rms: float
    rvecs: list[np.ndarray] = field(default_factory=list)
    tvecs: list[np.ndarray] = field(default_factory=list)
    image_shape: tuple[int, int] = (0, 0)


def _make_board(cfg: CharucoBoardConfig) -> tuple:
    """Create ArUco dictionary, board, and detector parameters."""
    dictionary = cv2.aruco.getPredefinedDictionary(cfg.aruco_dict_id)
    board = cv2.aruco.CharucoBoard(
        (cfg.squares_x, cfg.squares_y),
        cfg.square_length,
        cfg.marker_length,
        dictionary,
    )
    params = cv2.aruco.DetectorParameters()
    return dictionary, board, params


def detect_charuco_corners(
    image: np.ndarray,
    board_config: CharucoBoardConfig,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Detect ChArUco corners in a single image.

    Args:
        image: BGR image.
        board_config: Board geometry configuration.

    Returns:
        ``(charuco_corners, charuco_ids)`` or ``(None, None)``
        if detection failed.
    """
    dictionary, board, params = _make_board(board_config)
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
        image, dictionary, parameters=params,
    )
    if marker_ids is None or len(marker_ids) == 0:
        return None, None

    retval, corners, ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, image, board,
    )
    if not retval:
        return None, None
    return corners, ids


def calibrate_pinhole(
    image_paths: list[str | Path],
    board_config: CharucoBoardConfig | None = None,
) -> PinholeResult:
    """Run pinhole camera calibration from ChArUco board images.

    Args:
        image_paths: Paths to calibration images.
        board_config: Board configuration.  Uses defaults if ``None``.

    Returns:
        A :class:`PinholeResult` with intrinsics and pose data.

    Raises:
        RuntimeError: If fewer than 3 valid detections are found.
    """
    if board_config is None:
        board_config = CharucoBoardConfig()

    dictionary, board, params = _make_board(board_config)

    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    img_shape: tuple[int, int] | None = None

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        if img_shape is None:
            img_shape = image.shape[:2]

        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            image, dictionary, parameters=params,
        )
        if marker_ids is None or len(marker_ids) == 0:
            continue

        retval, corners, ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, image, board,
        )
        if retval:
            all_corners.append(corners)
            all_ids.append(ids)

    if len(all_corners) < 3:
        raise RuntimeError(
            f"Only {len(all_corners)} valid detections (need >= 3)",
        )

    assert img_shape is not None
    retval, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, img_shape, None, None,
    )

    logger.info("Pinhole calibration RMS: %.3f px", retval)
    return PinholeResult(
        K=K,
        D=D,
        rms=float(retval),
        rvecs=list(rvecs),
        tvecs=list(tvecs),
        image_shape=img_shape,
    )


def detect_pose(
    image: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    board_config: CharucoBoardConfig | None = None,
    *,
    undistort: bool = True,
    draw_axes: bool = True,
    axis_length: float = 0.1,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Detect a ChArUco board and estimate its 6-DoF pose.

    Args:
        image: BGR input image.
        K: Camera intrinsic matrix.
        D: Distortion coefficients.
        board_config: Board geometry.  Uses defaults if ``None``.
        undistort: Whether to undistort the image before detection.
        draw_axes: Whether to draw 3-D axes on the returned image.
        axis_length: Length of the drawn axes in board units.

    Returns:
        ``(annotated_image, rvec, tvec)`` where *rvec* and *tvec*
        are ``None`` if pose estimation failed.
    """
    if board_config is None:
        board_config = CharucoBoardConfig()

    if undistort:
        image = cv2.undistort(image, K, D)

    dictionary, board, params = _make_board(board_config)
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
        image, dictionary, parameters=params,
    )

    rvec = tvec = None
    if marker_ids is not None and len(marker_ids) > 0:
        retval, corners, ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, image, board,
        )
        if retval:
            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                corners, ids, board, K, D, None, None,
            )
            if ok and draw_axes:
                cv2.drawFrameAxes(
                    image, K, D, rvec, tvec,
                    length=axis_length, thickness=15,
                )

    return image, rvec, tvec

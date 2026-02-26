"""Omnidirectional (fisheye) camera calibration using the MEI model.

Extracts and refactors the calibration pipeline from
``fisheye_KDxi.py``.  The public entry point is
:func:`calibrate_fisheye`, which runs a two-pass robust calibration
(initial fit → MAD outlier rejection → optional refit).

Requires ``cv2.omnidir`` (OpenCV contrib).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FisheyeResult:
    """Result of a fisheye (omnidir) calibration.

    Attributes:
        K: 3x3 camera intrinsic matrix.
        D: 1x4 distortion coefficients.
        xi: Scalar MEI model parameter.
        rms: Overall RMS reprojection error in pixels.
        rvecs: Per-view rotation vectors.
        tvecs: Per-view translation vectors.
        per_view_errors: Per-view mean reprojection errors.
        image_shape: ``(height, width)`` of the calibration images.
        board_shape: ``(cols, rows)`` of the chessboard pattern used.
        kept: Number of views kept after filtering.
        total: Total number of views detected.
    """

    K: np.ndarray
    D: np.ndarray
    xi: float
    rms: float
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    per_view_errors: list[tuple[str, float]] = field(default_factory=list)
    image_shape: tuple[int, int] = (0, 0)
    board_shape: tuple[int, int] = (0, 0)
    kept: int = 0
    total: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_object_grid(
    cols: int,
    rows: int,
    square_size: float,
) -> np.ndarray:
    """Build a 3-D object point grid for a chessboard pattern.

    Args:
        cols: Number of inner corners along the width.
        rows: Number of inner corners along the height.
        square_size: Physical size of one square (arbitrary units).

    Returns:
        Float64 array of shape ``(rows * cols, 1, 3)``.
    """
    obj = np.zeros((rows * cols, 3), np.float64)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return obj.reshape(-1, 1, 3)


def collect_chessboard_points(
    image_paths: list[str | Path],
    board_shape: tuple[int, int],
    square_size: float,
    subpix_criteria: tuple[int, int, float] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], tuple[int, int] | None]:
    """Detect chessboard corners across multiple images.

    Args:
        image_paths: Paths to calibration images.
        board_shape: ``(cols, rows)`` inner corner counts.
        square_size: Physical square size.
        subpix_criteria: OpenCV termination criteria for corner
            sub-pixel refinement.  Defaults to ``(EPS+MAX_ITER, 200, 1e-5)``.

    Returns:
        A tuple ``(objpoints, imgpoints, kept_paths, image_shape)``
        where *image_shape* is ``(h, w)`` or ``None`` if no images
        were loaded.
    """
    if subpix_criteria is None:
        subpix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            200,
            1e-5,
        )

    cols, rows = board_shape
    obj_grid = build_object_grid(cols, rows, square_size)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    kept: list[str] = []
    img_shape: tuple[int, int] | None = None

    cb_flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )

    for fname in image_paths:
        img = cv2.imread(str(fname), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        if img_shape is None:
            img_shape = (h, w)
        elif img_shape != (h, w):
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCornersSB(
            gray, (cols, rows), flags=cb_flags,
        )
        if not ok:
            continue

        corners = cv2.cornerSubPix(
            gray, corners, (15, 15), (-1, -1), subpix_criteria,
        )
        objpoints.append(obj_grid.copy())
        imgpoints.append(corners.reshape(-1, 1, 2).astype(np.float64))
        kept.append(str(fname))

    logger.info(
        "Chessboard detection: %d / %d images accepted",
        len(kept), len(image_paths),
    )
    return objpoints, imgpoints, kept, img_shape


def omnidir_calibrate(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    image_shape: tuple[int, int],
    calib_criteria: tuple[int, int, float] | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, list, list, np.ndarray]:
    """Run ``cv2.omnidir.calibrate`` with sensible defaults.

    Args:
        objpoints: Per-view object point arrays.
        imgpoints: Per-view image point arrays.
        image_shape: ``(h, w)`` of images.
        calib_criteria: Termination criteria for the calibration
            solver.  Defaults to ``(EPS+MAX_ITER, 300, 1e-10)``.

    Returns:
        ``(rms, K, xi, D, rvecs, tvecs, used_idx)``.
    """
    if calib_criteria is None:
        calib_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            300,
            1e-10,
        )

    h, w = image_shape
    K = np.eye(3, dtype=np.float64)
    D = np.zeros((1, 4), dtype=np.float64)
    xi = np.array([1.2], dtype=np.float64)

    rms, K, xi, D, rvecs, tvecs, used_idx = cv2.omnidir.calibrate(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        size=(int(w), int(h)),
        K=K, xi=xi, D=D,
        rvecs=None, tvecs=None,
        flags=cv2.omnidir.CALIB_FIX_SKEW,
        criteria=calib_criteria,
    )
    return rms, K, xi, D, rvecs, tvecs, used_idx


def project_points(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    xi: np.ndarray | float,
    D: np.ndarray,
    *,
    perspective: bool = True,
) -> np.ndarray:
    """Project 3-D points through the omnidir model.

    Args:
        object_points: ``(N, 3)`` or ``(N, 1, 3)`` object points.
        rvec: Rotation vector ``(3,)`` or ``(3, 1)``.
        tvec: Translation vector ``(3,)`` or ``(3, 1)``.
        K: 3x3 intrinsic matrix.
        xi: MEI model scalar.
        D: 1x4 distortion vector.
        perspective: If True use perspective projection flag.

    Returns:
        Projected 2-D points ``(N, 1, 2)``.
    """
    op = np.asarray(object_points, np.float64).reshape(-1, 1, 3)
    rv = np.asarray(rvec, np.float64).reshape(3, 1)
    tv = np.asarray(tvec, np.float64).reshape(3, 1)
    K_ = np.asarray(K, np.float64)
    D_ = np.asarray(D, np.float64)
    xi_val = float(np.asarray(xi, np.float64).ravel()[0])
    flag = 1 if perspective else 0
    out = cv2.omnidir.projectPoints(op, rv, tv, K_, xi_val, D_, flag)
    return out[0] if isinstance(out, tuple) else out


def per_view_errors(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    K: np.ndarray,
    xi: np.ndarray,
    D: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, list[tuple[str, float]]]:
    """Compute per-view mean reprojection errors.

    Returns:
        ``(errors_array, [(name, error), ...])``.
    """
    errs: list[float] = []
    per_file: list[tuple[str, float]] = []
    for op, ip, rv, tv, name in zip(
        objpoints, imgpoints, rvecs, tvecs, names,
    ):
        proj = project_points(op, rv, tv, K, xi, D)
        e = float(np.linalg.norm(
            proj.reshape(-1, 2) - ip.reshape(-1, 2), axis=1,
        ).mean())
        errs.append(e)
        per_file.append((name, e))
    return np.asarray(errs, dtype=np.float64), per_file


def robust_filter(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: list,
    tvecs: list,
    names: list[str],
    errors: np.ndarray,
    z_threshold: float = 2.5,
) -> tuple[list, list, list, list, list, np.ndarray, float]:
    """Reject outlier views using MAD-based robust statistics.

    Args:
        z_threshold: Number of scaled MADs above the median to keep.

    Returns:
        Filtered ``(objpoints, imgpoints, rvecs, tvecs, names, mask, threshold)``.
    """
    med = np.median(errors)
    mad = np.median(np.abs(errors - med)) + 1e-9
    thr = med + z_threshold * 1.4826 * mad
    mask = errors <= thr

    def _select(xs: list) -> list:
        return [x for x, m in zip(xs, mask) if m]

    return (
        _select(objpoints),
        _select(imgpoints),
        _select(rvecs),
        _select(tvecs),
        _select(names),
        mask,
        float(thr),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calibrate_fisheye(
    image_paths: list[str | Path],
    board_shape: tuple[int, int],
    square_size: float = 1.0,
    alt_shapes: list[tuple[int, int]] | None = None,
    z_threshold: float = 2.5,
    subpix_criteria: tuple[int, int, float] | None = None,
    calib_criteria: tuple[int, int, float] | None = None,
) -> FisheyeResult:
    """Run full omnidirectional (MEI model) calibration.

    Tries each board orientation in *alt_shapes* (defaults to
    ``[board_shape, (board_shape[1], board_shape[0])]``), picks the
    best by RMS, then does a robust outlier pass and optional refit.

    Args:
        image_paths: Glob-expanded list of calibration image paths.
        board_shape: ``(cols, rows)`` of inner chessboard corners.
        square_size: Physical square size (arbitrary units).
        alt_shapes: Alternative board orientations to try.
        z_threshold: MAD z-score for outlier rejection.
        subpix_criteria: Corner sub-pixel criteria.
        calib_criteria: Solver termination criteria.

    Returns:
        A :class:`FisheyeResult` with the best calibration.

    Raises:
        RuntimeError: If no valid detections are found.
    """
    if alt_shapes is None:
        alt_shapes = [board_shape, (board_shape[1], board_shape[0])]

    results: list[dict] = []
    for shape in alt_shapes:
        res = _run_for_shape(
            image_paths, shape, square_size,
            subpix_criteria, calib_criteria, z_threshold,
        )
        if res is not None:
            results.append(res)

    if not results:
        raise RuntimeError(
            "No valid chessboard detections across any board orientation",
        )

    best = min(results, key=lambda r: r["rms"])
    logger.info(
        "Best calibration: shape=%s, RMS=%.3f px, kept=%d/%d",
        best["shape"], best["rms"], best["kept"], best["total"],
    )

    return FisheyeResult(
        K=best["K"],
        D=best["D"],
        xi=float(np.asarray(best["xi"]).ravel()[0]),
        rms=float(best["rms"]),
        rvecs=best["rvecs"],
        tvecs=best["tvecs"],
        per_view_errors=best["per_file"],
        image_shape=best["img_shape"],
        board_shape=best["shape"],
        kept=best["kept"],
        total=best["total"],
    )


def _run_for_shape(
    image_paths: list[str | Path],
    shape: tuple[int, int],
    square_size: float,
    subpix_criteria: tuple | None,
    calib_criteria: tuple | None,
    z_threshold: float,
) -> dict | None:
    """Single-shape calibration with optional robust refit."""
    paths = [str(p) for p in image_paths]
    objpoints, imgpoints, kept, img_shape = collect_chessboard_points(
        paths, shape, square_size, subpix_criteria,
    )
    if len(objpoints) < 3 or img_shape is None:
        return None

    rms, K, xi, D, rvecs, tvecs, used_idx = omnidir_calibrate(
        objpoints, imgpoints, img_shape, calib_criteria,
    )

    used_idx = np.asarray(used_idx, dtype=int).ravel()
    obj_used = [objpoints[i] for i in used_idx]
    img_used = [imgpoints[i] for i in used_idx]
    names_used = [kept[i] for i in used_idx]

    errs, per_file = per_view_errors(
        obj_used, img_used, rvecs, tvecs, K, xi, D, names_used,
    )

    op_f, ip_f, rv_f, tv_f, names_f, mask, thr = robust_filter(
        obj_used, img_used, rvecs, tvecs, names_used, errs, z_threshold,
    )

    if len(op_f) >= 3 and len(op_f) < len(obj_used):
        rms2, K2, xi2, D2, rvecs2, tvecs2, used_idx2 = omnidir_calibrate(
            op_f, ip_f, img_shape, calib_criteria,
        )
        used_idx2 = np.asarray(used_idx2, dtype=int).ravel()
        op2 = [op_f[i] for i in used_idx2]
        ip2 = [ip_f[i] for i in used_idx2]
        names2 = [names_f[i] for i in used_idx2]
        errs2, per_file2 = per_view_errors(
            op2, ip2, rvecs2, tvecs2, K2, xi2, D2, names2,
        )
        return {
            "shape": shape, "rms": rms2, "K": K2, "xi": xi2, "D": D2,
            "rvecs": rvecs2, "tvecs": tvecs2, "errs": errs2,
            "kept": len(errs2), "total": len(errs),
            "img_shape": img_shape, "per_file": per_file2,
        }

    return {
        "shape": shape, "rms": rms, "K": K, "xi": xi, "D": D,
        "rvecs": rvecs, "tvecs": tvecs, "errs": errs,
        "kept": len(errs), "total": len(errs),
        "img_shape": img_shape, "per_file": per_file,
    }

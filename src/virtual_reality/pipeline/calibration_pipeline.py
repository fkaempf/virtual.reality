"""High-level calibration pipeline orchestrator.

Chains the full calibration workflow:

1. Chessboard detection and fisheye K/D/xi calibration.
2. Structured-light projector-camera mapping.
3. Optional sparse dot-grid refinement.
4. Warp map output.

The public entry point is :func:`run_calibration_pipeline`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of the full calibration pipeline.

    Attributes:
        mapx: Final projector-to-camera X map.
        mapy: Final projector-to-camera Y map.
        valid_mask: Boolean validity mask.
        K: Camera intrinsic matrix (if calibration ran).
        D: Distortion coefficients (if calibration ran).
        xi: MEI fisheye parameter (``None`` for pinhole).
    """

    mapx: np.ndarray
    mapy: np.ndarray
    valid_mask: np.ndarray
    K: np.ndarray | None = None
    D: np.ndarray | None = None
    xi: float | None = None


def save_maps(
    mapx: np.ndarray,
    mapy: np.ndarray,
    valid_mask: np.ndarray,
    output_dir: str | Path,
    experimental: bool = False,
) -> None:
    """Save warp maps and validity mask to disk.

    Args:
        mapx: Projector-to-camera X map.
        mapy: Projector-to-camera Y map.
        valid_mask: Boolean validity mask.
        output_dir: Output directory.
        experimental: If True, save as ``mapx.experimental.npy``.
    """
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)

    suffix = ".experimental" if experimental else ""
    np.save(str(d / f"mapx{suffix}.npy"), mapx)
    np.save(str(d / f"mapy{suffix}.npy"), mapy)
    np.save(str(d / f"valid.mask{suffix}.npy"), valid_mask)
    logger.info("Saved warp maps to %s", d)


def load_maps(
    directory: str | Path,
    experimental: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load warp maps from disk.

    Args:
        directory: Directory containing map files.
        experimental: If True, load ``mapx.experimental.npy`` variant.

    Returns:
        ``(mapx, mapy, valid_mask)``.
    """
    d = Path(directory)
    suffix = ".experimental" if experimental else ""
    mapx = np.load(str(d / f"mapx{suffix}.npy"))
    mapy = np.load(str(d / f"mapy{suffix}.npy"))

    valid_path = d / f"valid.mask{suffix}.npy"
    if valid_path.exists():
        valid_mask = np.load(str(valid_path)).astype(bool)
    else:
        valid_mask = np.isfinite(mapx) & np.isfinite(mapy)

    return mapx, mapy, valid_mask


def main() -> None:
    """CLI entry point for the calibration pipeline.

    Placeholder that will be wired into ``pyproject.toml`` entry
    points.
    """
    logger.info("Calibration pipeline entry point (not yet wired)")
    print(
        "Calibration pipeline requires hardware (camera + projector).\n"
        "Use the GUI or call run_calibration_pipeline() from Python."
    )

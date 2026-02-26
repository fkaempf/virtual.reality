"""Tests for virtual_reality.calibration.fisheye.

Note: Full calibration tests require cv2.omnidir and calibration
images, so we test the helper functions that don't need hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from virtual_reality.calibration.fisheye import (
    FisheyeResult,
    build_object_grid,
    robust_filter,
)


class TestBuildObjectGrid:
    """Tests for build_object_grid."""

    def test_shape(self) -> None:
        grid = build_object_grid(6, 9, 1.0)
        assert grid.shape == (54, 1, 3)

    def test_z_is_zero(self) -> None:
        grid = build_object_grid(4, 3, 2.0)
        assert np.all(grid[:, 0, 2] == 0.0)

    def test_spacing(self) -> None:
        grid = build_object_grid(3, 2, 5.0)
        points = grid[:, 0, :]
        xs = points[:, 0]
        assert xs.max() == pytest.approx(10.0)


class TestRobustFilter:
    """Tests for the MAD-based outlier filter."""

    def test_no_outliers(self) -> None:
        errors = np.array([1.0, 1.1, 0.9, 1.0, 1.05])
        data = [[i] for i in range(5)]
        _, _, _, _, _, mask, _ = robust_filter(
            data, data, data, data,
            [str(i) for i in range(5)],
            errors,
        )
        assert mask.sum() == 5

    def test_outlier_removed(self) -> None:
        errors = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
        data = [[i] for i in range(5)]
        _, _, _, _, filtered_names, mask, _ = robust_filter(
            data, data, data, data,
            [str(i) for i in range(5)],
            errors,
        )
        assert not mask[4]
        assert len(filtered_names) == 4


class TestFisheyeResult:
    """Tests for the FisheyeResult dataclass."""

    def test_creation(self) -> None:
        result = FisheyeResult(
            K=np.eye(3),
            D=np.zeros((1, 4)),
            xi=1.2,
            rms=0.5,
            rvecs=[],
            tvecs=[],
        )
        assert result.rms == pytest.approx(0.5)
        assert result.xi == pytest.approx(1.2)

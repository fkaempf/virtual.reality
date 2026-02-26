"""Tests for virtual_reality.math_utils.geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from virtual_reality.math_utils.geometry import (
    compute_light_dirs,
    heading_to_direction,
    wrap_angle_deg,
)


class TestComputeLightDirs:
    """Tests for compute_light_dirs."""

    def test_returns_4x3(self) -> None:
        dirs = compute_light_dirs(65.0)
        assert dirs.shape == (4, 3)

    def test_all_normalized(self) -> None:
        dirs = compute_light_dirs(45.0)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_array_almost_equal(norms, 1.0)

    def test_dtype_is_float32(self) -> None:
        dirs = compute_light_dirs(65.0)
        assert dirs.dtype == np.float32

    def test_horizontal_lights_at_zero_elevation(self) -> None:
        dirs = compute_light_dirs(0.0)
        for d in dirs:
            assert d[1] == pytest.approx(0.0, abs=1e-6)


class TestHeadingToDirection:
    """Tests for heading_to_direction."""

    def test_zero_is_forward(self) -> None:
        dx, dy = heading_to_direction(0.0)
        assert dx == pytest.approx(0.0, abs=1e-6)
        assert dy == pytest.approx(1.0)

    def test_90_is_right(self) -> None:
        dx, dy = heading_to_direction(90.0)
        assert dx == pytest.approx(1.0)
        assert dy == pytest.approx(0.0, abs=1e-6)

    def test_180_is_backward(self) -> None:
        dx, dy = heading_to_direction(180.0)
        assert dx == pytest.approx(0.0, abs=1e-6)
        assert dy == pytest.approx(-1.0)


class TestWrapAngleDeg:
    """Tests for wrap_angle_deg."""

    def test_in_range_unchanged(self) -> None:
        assert wrap_angle_deg(45.0) == pytest.approx(45.0)

    def test_negative_wraps(self) -> None:
        assert wrap_angle_deg(-90.0) == pytest.approx(270.0)

    def test_360_wraps_to_zero(self) -> None:
        assert wrap_angle_deg(360.0) == pytest.approx(0.0)

    def test_large_positive(self) -> None:
        assert wrap_angle_deg(720.0) == pytest.approx(0.0)

    def test_large_negative(self) -> None:
        assert wrap_angle_deg(-450.0) == pytest.approx(270.0)

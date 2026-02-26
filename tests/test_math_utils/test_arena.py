"""Tests for virtual_reality.math_utils.arena."""

from __future__ import annotations

import math

import pytest

from virtual_reality.math_utils.arena import (
    clamp_to_arena,
    compute_camera_fly_distance_mm,
    enforce_min_distance,
)


class TestClampToArena:
    """Tests for clamp_to_arena."""

    def test_inside_unchanged(self) -> None:
        x, y = clamp_to_arena(5.0, 5.0, 40.0)
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(5.0)

    def test_on_boundary_unchanged(self) -> None:
        x, y = clamp_to_arena(40.0, 0.0, 40.0)
        assert math.hypot(x, y) == pytest.approx(40.0)

    def test_outside_clamped(self) -> None:
        x, y = clamp_to_arena(60.0, 0.0, 40.0)
        assert math.hypot(x, y) == pytest.approx(40.0)
        assert x == pytest.approx(40.0)

    def test_origin_stays(self) -> None:
        x, y = clamp_to_arena(0.0, 0.0, 40.0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_diagonal_outside(self) -> None:
        x, y = clamp_to_arena(100.0, 100.0, 40.0)
        assert math.hypot(x, y) == pytest.approx(40.0, abs=1e-4)


class TestEnforceMinDistance:
    """Tests for enforce_min_distance."""

    def test_already_far_enough(self) -> None:
        pos = enforce_min_distance((10.0, 0.0), (0.0, 0.0), 5.0)
        assert pos == pytest.approx((10.0, 0.0))

    def test_too_close_pushed_away(self) -> None:
        pos = enforce_min_distance((1.0, 0.0), (0.0, 0.0), 5.0)
        assert math.hypot(pos[0], pos[1]) == pytest.approx(5.0)

    def test_overlapping_pushed_along_x(self) -> None:
        pos = enforce_min_distance((0.0, 0.0), (0.0, 0.0), 3.0)
        assert pos == pytest.approx((3.0, 0.0))

    def test_at_exact_boundary(self) -> None:
        pos = enforce_min_distance((5.0, 0.0), (0.0, 0.0), 5.0)
        assert pos == pytest.approx((5.0, 0.0))


class TestComputeCameraFlyDistance:
    """Tests for compute_camera_fly_distance_mm."""

    def test_same_position_returns_height(self) -> None:
        dist = compute_camera_fly_distance_mm(
            (0.0, 0.0), (0.0, 0.0), 5.0,
        )
        assert dist == pytest.approx(5.0)

    def test_horizontal_distance(self) -> None:
        dist = compute_camera_fly_distance_mm(
            (3.0, 0.0), (0.0, 4.0), 0.0,
        )
        assert dist == pytest.approx(5.0)

    def test_3d_distance(self) -> None:
        dist = compute_camera_fly_distance_mm(
            (3.0, 0.0), (0.0, 4.0), 12.0,
        )
        assert dist == pytest.approx(13.0)

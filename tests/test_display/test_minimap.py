"""Tests for virtual_reality.display.minimap."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from virtual_reality.display.minimap import (
    build_minimap_base,
    draw_arrow,
    draw_minimap_dynamic,
    world_to_minimap,
)


class TestWorldToMinimap:
    """Tests for coordinate conversion."""

    def test_origin(self) -> None:
        u, v = world_to_minimap(0, 0, 100, 100, 2.0)
        assert u == 100
        assert v == 100

    def test_positive_x(self) -> None:
        u, v = world_to_minimap(10, 0, 100, 100, 2.0)
        assert u == 120
        assert v == 100

    def test_positive_y_goes_up(self) -> None:
        u, v = world_to_minimap(0, 10, 100, 100, 2.0)
        assert u == 100
        assert v == 80  # Y is flipped


class TestDrawArrow:
    """Tests for draw_arrow."""

    def test_modifies_image(self) -> None:
        img = np.full((200, 200, 3), 255, np.uint8)
        original = img.copy()
        draw_arrow(img, 100, 100, 0.0)
        assert not np.array_equal(img, original)


class TestBuildMinimapBase:
    """Tests for build_minimap_base."""

    def test_shape(self) -> None:
        base = build_minimap_base(
            arena_radius_mm=40.0,
            map_w=200, map_h=200,
            center_u=100, center_v=100,
            scale=2.0,
        )
        assert base.shape == (200, 200, 3)
        assert base.dtype == np.uint8

    def test_has_circle(self) -> None:
        base = build_minimap_base(
            arena_radius_mm=40.0,
            map_w=200, map_h=200,
            center_u=100, center_v=100,
            scale=2.0,
        )
        # The circle should have drawn some non-white pixels.
        assert not np.all(base == 255)


class TestDrawMinimapDynamic:
    """Tests for draw_minimap_dynamic."""

    def test_returns_image(self) -> None:
        base = build_minimap_base(
            arena_radius_mm=40.0,
            map_w=200, map_h=200,
            center_u=100, center_v=100,
            scale=2.0,
        )
        result = draw_minimap_dynamic(
            base_img=base,
            fly_x=0.0, fly_y=0.0, fly_heading=0.0,
            trail_pts_uv=[(100, 100), (110, 100)],
            trail_color=(0, 200, 0),
            trail_thick=2,
            center_u=100, center_v=100,
            scale=2.0,
            fps=60.0,
            cam_x=0.0, cam_y=-20.0, cam_heading=0.0,
            camera_fov_x_deg=200.0,
            arena_radius_mm=40.0,
        )
        assert result.shape == base.shape
        assert not np.array_equal(result, base)

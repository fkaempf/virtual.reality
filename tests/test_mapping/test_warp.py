"""Tests for virtual_reality.mapping.warp."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from virtual_reality.mapping.warp import (
    WarpMap,
    build_proj_to_cam_map,
    load_warp_map,
    warp_to_gl_texture,
)


class TestWarpMap:
    """Tests for the WarpMap dataclass."""

    def test_basic_fields(self) -> None:
        mapx = np.zeros((10, 20), dtype=np.float32)
        mapy = np.zeros((10, 20), dtype=np.float32)
        wm = WarpMap(
            mapx=mapx, mapy=mapy,
            cam_w=100, cam_h=80,
            proj_w=20, proj_h=10,
            valid_mask=np.ones((10, 20), dtype=bool),
        )
        assert wm.proj_w == 20
        assert wm.cam_h == 80


class TestLoadWarpMap:
    """Tests for load_warp_map."""

    def test_identity_map(self, tmp_path: Path) -> None:
        """An identity map where projector pixel (x, y) maps to
        camera pixel (x, y)."""
        proj_h, proj_w = 8, 10
        mapy, mapx = np.mgrid[0:proj_h, 0:proj_w].astype(np.float32)
        # Flip because load_warp_map applies flipud.
        np.save(tmp_path / "mapx.npy", np.flipud(mapx))
        np.save(tmp_path / "mapy.npy", np.flipud(mapy))

        wm = load_warp_map(
            tmp_path / "mapx.npy",
            tmp_path / "mapy.npy",
        )
        assert wm.proj_w == proj_w
        assert wm.proj_h == proj_h
        assert wm.cam_w == proj_w
        assert wm.cam_h == proj_h
        assert np.all(wm.valid_mask)

    def test_invalid_entries_detected(self, tmp_path: Path) -> None:
        mapx = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
        mapy = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        np.save(tmp_path / "mapx.npy", np.flipud(mapx))
        np.save(tmp_path / "mapy.npy", np.flipud(mapy))

        wm = load_warp_map(
            tmp_path / "mapx.npy",
            tmp_path / "mapy.npy",
        )
        # One NaN pixel should be invalid.
        assert not np.all(wm.valid_mask)

    def test_all_invalid_raises(self, tmp_path: Path) -> None:
        mapx = np.full((4, 4), np.nan, dtype=np.float32)
        mapy = np.full((4, 4), np.nan, dtype=np.float32)
        np.save(tmp_path / "mapx.npy", mapx)
        np.save(tmp_path / "mapy.npy", mapy)

        with pytest.raises(RuntimeError, match="no valid entries"):
            load_warp_map(
                tmp_path / "mapx.npy",
                tmp_path / "mapy.npy",
            )

    def test_factor_scaling(self, tmp_path: Path) -> None:
        mapx = np.array([[0, 4], [0, 4]], dtype=np.float32)
        mapy = np.array([[0, 0], [4, 4]], dtype=np.float32)
        np.save(tmp_path / "mapx.npy", np.flipud(mapx))
        np.save(tmp_path / "mapy.npy", np.flipud(mapy))

        wm = load_warp_map(
            tmp_path / "mapx.npy",
            tmp_path / "mapy.npy",
            factor=2.0,
        )
        assert wm.mapx.max() == pytest.approx(2.0)


class TestWarpToGlTexture:
    """Tests for warp_to_gl_texture."""

    def test_output_shape(self) -> None:
        wm = WarpMap(
            mapx=np.ones((4, 6), dtype=np.float32) * 50,
            mapy=np.ones((4, 6), dtype=np.float32) * 40,
            cam_w=100, cam_h=80,
            proj_w=6, proj_h=4,
            valid_mask=np.ones((4, 6), dtype=bool),
        )
        tex = warp_to_gl_texture(wm)
        assert tex.shape == (4, 6, 2)
        assert tex.dtype == np.float32

    def test_normalized_range(self) -> None:
        wm = WarpMap(
            mapx=np.array([[0, 50], [0, 50]], dtype=np.float32),
            mapy=np.array([[0, 0], [40, 40]], dtype=np.float32),
            cam_w=100, cam_h=80,
            proj_w=2, proj_h=2,
            valid_mask=np.ones((2, 2), dtype=bool),
        )
        tex = warp_to_gl_texture(wm)
        assert tex[0, 1, 0] == pytest.approx(0.5)
        assert tex[1, 1, 1] == pytest.approx(0.5)

    def test_invalid_pixels_are_negative(self) -> None:
        mask = np.array([[True, False], [True, True]], dtype=bool)
        wm = WarpMap(
            mapx=np.ones((2, 2), dtype=np.float32),
            mapy=np.ones((2, 2), dtype=np.float32),
            cam_w=10, cam_h=10,
            proj_w=2, proj_h=2,
            valid_mask=mask,
        )
        tex = warp_to_gl_texture(wm)
        assert tex[0, 1, 0] == -1.0
        assert tex[0, 1, 1] == -1.0


class TestBuildProjToCamMap:
    """Tests for build_proj_to_cam_map."""

    def test_identity_inversion(self) -> None:
        """A simple case where camera pixel (y,x) maps to projector
        pixel (x,y)."""
        proj_w, proj_h = 4, 3
        # Camera has 3 rows, 4 cols. proj_x[y,x] = x, proj_y[y,x] = y.
        cam_h, cam_w = proj_h, proj_w
        proj_x = np.zeros((cam_h, cam_w), dtype=np.float32)
        proj_y = np.zeros((cam_h, cam_w), dtype=np.float32)
        for y in range(cam_h):
            for x in range(cam_w):
                proj_x[y, x] = x
                proj_y[y, x] = y

        mapx, mapy = build_proj_to_cam_map(
            proj_x, proj_y, proj_w, proj_h,
        )
        # mapx[py, px] should equal px (camera x == projector x).
        for py in range(proj_h):
            for px in range(proj_w):
                assert mapx[py, px] == pytest.approx(px)
                assert mapy[py, px] == pytest.approx(py)

    def test_out_of_bounds_ignored(self) -> None:
        proj_x = np.array([[100.0]], dtype=np.float32)
        proj_y = np.array([[100.0]], dtype=np.float32)
        mapx, mapy = build_proj_to_cam_map(proj_x, proj_y, 4, 4)
        assert np.all(np.isnan(mapx))

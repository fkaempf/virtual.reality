"""Tests for virtual_reality.mapping.pipeline."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from virtual_reality.mapping.pipeline import (
    despeckle_maps,
    process_raw_maps,
)


class TestDespeckleMaps:
    """Tests for despeckle_maps."""

    def test_no_change_smooth(self) -> None:
        """A smooth map should be mostly unchanged."""
        h, w = 20, 20
        mapx = np.arange(w, dtype=np.float32)[np.newaxis, :].repeat(h, axis=0)
        mapy = np.arange(h, dtype=np.float32)[:, np.newaxis].repeat(w, axis=1)
        valid = np.ones((h, w), dtype=bool)

        dx, dy = despeckle_maps(mapx, mapy, valid)
        np.testing.assert_array_almost_equal(dx, mapx, decimal=0)

    def test_speckle_removed(self) -> None:
        """A single outlier pixel should be corrected."""
        h, w = 20, 20
        mapx = np.ones((h, w), dtype=np.float32) * 10.0
        mapx[10, 10] = 100.0  # outlier
        mapy = np.ones((h, w), dtype=np.float32) * 10.0
        valid = np.ones((h, w), dtype=bool)

        dx, _ = despeckle_maps(mapx, mapy, valid)
        # Outlier should be brought closer to 10.
        assert abs(dx[10, 10] - 10.0) < 50.0


class TestProcessRawMaps:
    """Tests for the full processing pipeline."""

    def test_identity_map(self) -> None:
        """Simple identity where each camera pixel maps to itself."""
        h, w = 8, 10
        proj_x = np.zeros((h, w), dtype=np.float32)
        proj_y = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                proj_x[y, x] = x
                proj_y[y, x] = y

        mapx, mapy, valid = process_raw_maps(
            proj_x, proj_y, w, h,
            despeckle=False, inpaint=False,
        )
        assert mapx.shape == (h, w)
        assert valid.any()

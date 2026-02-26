"""Tests for virtual_reality.mapping.refinement."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from virtual_reality.mapping.refinement import (
    expand_mask,
    refine_with_visibility,
)


class TestExpandMask:
    """Tests for mask dilation."""

    def test_no_expansion(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        result = expand_mask(mask, 0)
        assert result.sum() == 1

    def test_expands(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 10] = True
        result = expand_mask(mask, 3)
        assert result.sum() > 1
        assert result[10, 10]


class TestRefineWithVisibility:
    """Tests for refine_with_visibility."""

    def test_invisible_pixels_become_nan(self) -> None:
        h, w = 10, 10
        mapx = np.ones((h, w), dtype=np.float32)
        mapy = np.ones((h, w), dtype=np.float32)
        valid = np.ones((h, w), dtype=bool)
        visibility = np.zeros((h, w), dtype=bool)
        visibility[5, 5] = True

        mx, my, new_valid = refine_with_visibility(
            mapx, mapy, valid, visibility, buffer_px=1,
        )
        assert np.isnan(mx[0, 0])
        assert new_valid.sum() < valid.sum()

    def test_visible_pixels_preserved(self) -> None:
        h, w = 10, 10
        mapx = np.ones((h, w), dtype=np.float32) * 5.0
        mapy = np.ones((h, w), dtype=np.float32) * 3.0
        valid = np.ones((h, w), dtype=bool)
        visibility = np.ones((h, w), dtype=bool)

        mx, my, new_valid = refine_with_visibility(
            mapx, mapy, valid, visibility, buffer_px=0,
        )
        assert mx[5, 5] == pytest.approx(5.0)
        assert new_valid.all()

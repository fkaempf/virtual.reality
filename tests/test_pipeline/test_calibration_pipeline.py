"""Tests for virtual_reality.pipeline.calibration_pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from virtual_reality.pipeline.calibration_pipeline import (
    CalibrationResult,
    load_maps,
    save_maps,
)


class TestSaveLoadMaps:
    """Tests for save/load round-trip."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        h, w = 4, 6
        mapx = np.random.rand(h, w).astype(np.float32)
        mapy = np.random.rand(h, w).astype(np.float32)
        valid = np.ones((h, w), dtype=bool)
        valid[0, 0] = False

        save_maps(mapx, mapy, valid, tmp_path)
        mx, my, vm = load_maps(tmp_path)

        np.testing.assert_array_almost_equal(mx, mapx)
        np.testing.assert_array_almost_equal(my, mapy)
        assert not vm[0, 0]

    def test_experimental_suffix(self, tmp_path: Path) -> None:
        h, w = 3, 3
        mapx = np.ones((h, w), dtype=np.float32)
        mapy = np.ones((h, w), dtype=np.float32)
        valid = np.ones((h, w), dtype=bool)

        save_maps(mapx, mapy, valid, tmp_path, experimental=True)
        assert (tmp_path / "mapx.experimental.npy").exists()

        mx, my, vm = load_maps(tmp_path, experimental=True)
        np.testing.assert_array_almost_equal(mx, mapx)

    def test_creates_directory(self, tmp_path: Path) -> None:
        sub = tmp_path / "nested" / "dir"
        mapx = np.ones((2, 2), dtype=np.float32)
        mapy = np.ones((2, 2), dtype=np.float32)
        valid = np.ones((2, 2), dtype=bool)
        save_maps(mapx, mapy, valid, sub)
        assert sub.exists()


class TestCalibrationResult:
    """Tests for the CalibrationResult dataclass."""

    def test_creation(self) -> None:
        result = CalibrationResult(
            mapx=np.zeros((4, 4)),
            mapy=np.zeros((4, 4)),
            valid_mask=np.ones((4, 4), dtype=bool),
        )
        assert result.K is None
        assert result.xi is None

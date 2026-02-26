"""Tests for virtual_reality.calibration.intrinsics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from virtual_reality.calibration.intrinsics import (
    Intrinsics,
    load_intrinsics,
    save_intrinsics,
)


class TestIntrinsics:
    """Tests for the Intrinsics dataclass."""

    def test_pinhole_defaults(self) -> None:
        intr = Intrinsics(
            K=np.eye(3), D=np.zeros(5),
        )
        assert intr.model == "pinhole"
        assert not intr.is_fisheye

    def test_fisheye(self) -> None:
        intr = Intrinsics(
            K=np.eye(3), D=np.zeros(4),
            xi=1.2, model="fisheye",
        )
        assert intr.is_fisheye


class TestSaveLoad:
    """Tests for save_intrinsics / load_intrinsics round-trip."""

    def test_pinhole_roundtrip(self, tmp_path: Path) -> None:
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        D = np.array([0.1, -0.2, 0, 0, 0.05], dtype=np.float64)
        original = Intrinsics(K=K, D=D, model="pinhole")

        save_intrinsics(original, tmp_path)
        loaded = load_intrinsics(tmp_path)

        assert loaded.model == "pinhole"
        assert not loaded.is_fisheye
        np.testing.assert_array_almost_equal(loaded.K, K)
        np.testing.assert_array_almost_equal(loaded.D, D)

    def test_fisheye_roundtrip(self, tmp_path: Path) -> None:
        K = np.eye(3, dtype=np.float64) * 300
        K[2, 2] = 1.0
        D = np.array([[0.01, -0.02, 0.001, 0.0]], dtype=np.float64)
        original = Intrinsics(K=K, D=D, xi=1.5, model="fisheye")

        save_intrinsics(original, tmp_path)
        loaded = load_intrinsics(tmp_path)

        assert loaded.model == "fisheye"
        assert loaded.is_fisheye
        assert loaded.xi == pytest.approx(1.5)
        np.testing.assert_array_almost_equal(loaded.K, K)

    def test_missing_files_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_intrinsics(tmp_path)

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        sub = tmp_path / "nested" / "dir"
        intr = Intrinsics(K=np.eye(3), D=np.zeros(5))
        save_intrinsics(intr, sub)
        assert sub.exists()

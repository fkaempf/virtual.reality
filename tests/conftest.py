"""Shared test fixtures for the virtual_reality test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from virtual_reality.config.schema import VirtualRealityConfig


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-skip tests marked ``gpu`` or ``hardware`` by default."""
    skip_gpu = pytest.mark.skip(reason="requires OpenGL context")
    skip_hw = pytest.mark.skip(reason="requires physical hardware")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
        if "hardware" in item.keywords:
            item.add_marker(skip_hw)


@pytest.fixture
def default_config() -> VirtualRealityConfig:
    """Return a VirtualRealityConfig with default values."""
    return VirtualRealityConfig()


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def configs_dir(project_root: Path) -> Path:
    """Return the configs/ directory path."""
    return project_root / "configs"


@pytest.fixture
def synthetic_warp_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return small synthetic mapx/mapy arrays for testing.

    Creates a 16x20 identity warp map where each projector pixel maps
    to the same camera pixel coordinate.
    """
    proj_h, proj_w = 16, 20
    mapy, mapx = np.mgrid[0:proj_h, 0:proj_w].astype(np.float32)
    return mapx, mapy

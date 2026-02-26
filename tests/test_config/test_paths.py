"""Tests for virtual_reality.config.paths."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from virtual_reality.config.paths import (
    find_assets_dir,
    find_config_dir,
    find_project_root,
    resolve_platform_path,
)


class TestResolvePlatformPath:
    """Tests for resolve_platform_path."""

    def test_linux_on_linux(self) -> None:
        with mock.patch("virtual_reality.config.paths.sys") as m:
            m.platform = "linux"
            result = resolve_platform_path(
                linux_path="/linux/path",
                win_path="D:\\win",
                mac_path="/mac/path",
            )
            assert result == Path("/linux/path")

    def test_darwin_on_mac(self) -> None:
        with mock.patch("virtual_reality.config.paths.sys") as m:
            m.platform = "darwin"
            result = resolve_platform_path(
                mac_path="/Users/test",
                win_path="C:\\test",
            )
            assert result == Path("/Users/test")

    def test_fallback_when_platform_empty(self) -> None:
        with mock.patch("virtual_reality.config.paths.sys") as m:
            m.platform = "linux"
            result = resolve_platform_path(
                linux_path="",
                win_path="D:\\fallback",
            )
            assert result == Path("D:\\fallback")

    def test_no_path_raises(self) -> None:
        with mock.patch("virtual_reality.config.paths.sys") as m:
            m.platform = "linux"
            with pytest.raises(ValueError, match="No path"):
                resolve_platform_path()


class TestFindProjectRoot:
    """Tests for find_project_root."""

    def test_finds_root(self) -> None:
        root = find_project_root()
        assert (root / "pyproject.toml").exists()

    def test_root_contains_src(self) -> None:
        root = find_project_root()
        assert (root / "src").is_dir()


class TestFindConfigDir:
    """Tests for find_config_dir."""

    def test_returns_configs_path(self) -> None:
        result = find_config_dir()
        assert result.name == "configs"


class TestFindAssetsDir:
    """Tests for find_assets_dir."""

    def test_returns_assets_path(self) -> None:
        result = find_assets_dir()
        assert result.name == "assets"

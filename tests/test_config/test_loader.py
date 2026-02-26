"""Tests for virtual_reality.config.loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from virtual_reality.config.loader import (
    _apply_dict_to_dataclass,
    _dataclass_to_dict,
    load_config,
    save_config,
)
from virtual_reality.config.schema import (
    ArenaConfig,
    VirtualRealityConfig,
)


class TestDataclassToDict:
    """Tests for _dataclass_to_dict."""

    def test_flat_dataclass(self) -> None:
        cfg = ArenaConfig(radius_mm=50.0)
        result = _dataclass_to_dict(cfg)
        assert result == {"radius_mm": 50.0}

    def test_nested_dataclass(self) -> None:
        cfg = VirtualRealityConfig()
        result = _dataclass_to_dict(cfg)
        assert isinstance(result, dict)
        assert "arena" in result
        assert result["arena"]["radius_mm"] == 40.0

    def test_tuple_fields_become_lists(self) -> None:
        cfg = VirtualRealityConfig()
        result = _dataclass_to_dict(cfg)
        assert isinstance(result["lighting"]["intensities"], list)


class TestApplyDictToDataclass:
    """Tests for _apply_dict_to_dataclass."""

    def test_flat_update(self) -> None:
        cfg = ArenaConfig()
        _apply_dict_to_dataclass(cfg, {"radius_mm": 99.0})
        assert cfg.radius_mm == 99.0

    def test_nested_update(self) -> None:
        cfg = VirtualRealityConfig()
        _apply_dict_to_dataclass(cfg, {
            "arena": {"radius_mm": 77.0},
            "display": {"target_fps": 30},
        })
        assert cfg.arena.radius_mm == 77.0
        assert cfg.display.target_fps == 30

    def test_unknown_keys_ignored(self) -> None:
        cfg = ArenaConfig()
        _apply_dict_to_dataclass(cfg, {"nonexistent": 42})
        assert cfg.radius_mm == 40.0

    def test_list_to_tuple_conversion(self) -> None:
        cfg = VirtualRealityConfig()
        _apply_dict_to_dataclass(cfg, {
            "lighting": {"intensities": [1.0, 1.0, 1.0, 1.0]},
        })
        assert cfg.lighting.intensities == (1.0, 1.0, 1.0, 1.0)


class TestLoadConfig:
    """Tests for load_config."""

    def test_none_returns_defaults(self) -> None:
        cfg = load_config(None)
        assert cfg.arena.radius_mm == 40.0

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_from_file(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump({
            "arena": {"radius_mm": 55.5},
            "display": {"target_fps": 120},
        }))
        cfg = load_config(yaml_path)
        assert cfg.arena.radius_mm == 55.5
        assert cfg.display.target_fps == 120
        # Unspecified fields keep defaults.
        assert cfg.fly_model.phys_length_mm == 3.0

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        cfg = load_config(yaml_path)
        assert cfg.arena.radius_mm == 40.0


class TestSaveConfig:
    """Tests for save_config."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        original = VirtualRealityConfig()
        original.arena.radius_mm = 123.0
        original.camera.projection = "perspective"

        path = tmp_path / "out.yaml"
        save_config(original, path)
        loaded = load_config(path)

        assert loaded.arena.radius_mm == 123.0
        assert loaded.camera.projection == "perspective"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "subdir" / "deep" / "config.yaml"
        save_config(VirtualRealityConfig(), path)
        assert path.exists()

    def test_output_is_valid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "out.yaml"
        save_config(VirtualRealityConfig(), path)
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict)
        assert "arena" in data

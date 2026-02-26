"""YAML-based configuration loading and saving.

Provides functions to serialize ``VirtualRealityConfig`` to YAML and
deserialize it back, supporting layered configuration with defaults.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from virtual_reality.config.schema import VirtualRealityConfig


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass instance to a plain dict."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _dataclass_to_dict(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def _apply_dict_to_dataclass(obj: Any, data: dict[str, Any]) -> None:
    """Recursively apply a dict of values onto a dataclass instance.

    Only keys that match existing field names are applied. Nested
    dataclass fields are updated recursively rather than replaced.

    Args:
        obj: The dataclass instance to update.
        data: A dict whose keys correspond to field names.
    """
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if (
            dataclasses.is_dataclass(current)
            and isinstance(value, dict)
        ):
            _apply_dict_to_dataclass(current, value)
        else:
            field_info = {
                f.name: f for f in dataclasses.fields(obj)
            }
            if key in field_info:
                field_type = field_info[key].type
                if isinstance(value, list) and "tuple" in str(field_type):
                    value = tuple(value)
                setattr(obj, key, value)


def load_config(
    path: str | Path | None = None,
) -> VirtualRealityConfig:
    """Load a configuration from a YAML file.

    If *path* is ``None``, returns the default configuration. If a path
    is given, it is loaded and merged on top of the defaults.

    Args:
        path: Optional path to a YAML configuration file.

    Returns:
        A fully populated ``VirtualRealityConfig`` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    config = VirtualRealityConfig()
    if path is None:
        return config

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data and isinstance(data, dict):
        _apply_dict_to_dataclass(config, data)

    return config


def save_config(
    config: VirtualRealityConfig,
    path: str | Path,
) -> None:
    """Save a configuration to a YAML file.

    Args:
        config: The configuration to serialize.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = _dataclass_to_dict(config)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

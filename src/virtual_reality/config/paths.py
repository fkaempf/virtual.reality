"""Platform-aware path resolution for configuration and asset files.

Centralizes path logic that was previously scattered across scripts
with ``sys.platform == "darwin"`` checks.
"""

from __future__ import annotations

import sys
from pathlib import Path


def resolve_platform_path(
    mac_path: str = "",
    win_path: str = "",
    linux_path: str = "",
) -> Path:
    """Select a file path based on the current platform.

    Falls back through platforms: if the current platform's path is
    empty, tries the others. Raises ``ValueError`` if no path is
    provided for any platform.

    Args:
        mac_path: Path to use on macOS.
        win_path: Path to use on Windows.
        linux_path: Path to use on Linux.

    Returns:
        A ``Path`` object for the current platform.

    Raises:
        ValueError: If no path is available for the current platform.
    """
    platform_map = {
        "darwin": mac_path,
        "win32": win_path,
        "linux": linux_path,
    }
    path = platform_map.get(sys.platform, "")
    if not path:
        for fallback in (linux_path, win_path, mac_path):
            if fallback:
                path = fallback
                break
    if not path:
        raise ValueError(
            "No path provided for any platform."
        )
    return Path(path)


def find_project_root() -> Path:
    """Locate the project root directory.

    Walks up from this file's location looking for ``pyproject.toml``.

    Returns:
        The directory containing ``pyproject.toml``.

    Raises:
        FileNotFoundError: If no ``pyproject.toml`` is found.
    """
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Could not find pyproject.toml in any parent directory."
    )


def find_config_dir() -> Path:
    """Return the ``configs/`` directory under the project root."""
    return find_project_root() / "configs"


def find_assets_dir() -> Path:
    """Return the ``assets/`` directory under the project root."""
    return find_project_root() / "assets"

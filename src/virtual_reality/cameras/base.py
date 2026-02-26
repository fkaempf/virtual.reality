"""Abstract camera interface for hardware acquisition.

Defines the ``Camera`` protocol that all camera drivers must implement.
Use ``create_camera()`` from :mod:`virtual_reality.cameras.factory` to
instantiate the appropriate driver.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Camera(Protocol):
    """Protocol for camera hardware drivers.

    All drivers must support ``start``, ``grab``, and ``stop`` methods,
    plus context manager usage for safe cleanup.
    """

    def start(self) -> None:
        """Initialize hardware and begin acquisition."""
        ...

    def grab(self, timeout_s: float = 1.0) -> np.ndarray:
        """Capture and return a single frame.

        Args:
            timeout_s: Maximum wait time for the frame in seconds.

        Returns:
            A uint8 numpy array of shape ``(H, W)`` for mono or
            ``(H, W, 3)`` for color.
        """
        ...

    def stop(self) -> None:
        """Stop acquisition and release hardware resources."""
        ...

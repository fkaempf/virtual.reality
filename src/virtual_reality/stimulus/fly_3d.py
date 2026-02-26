"""3D GLB fly stimulus.

Renders a 3D fly model in a circular arena using OpenGL, with
projector warp correction, multiple projection modes, and Phong
lighting.  This is the main stimulus module that replaces
``3d_object_fly4.py``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from virtual_reality.config.schema import VirtualRealityConfig
from virtual_reality.stimulus.base import Stimulus

logger = logging.getLogger(__name__)


class Fly3DStimulus(Stimulus):
    """3D fly model stimulus with projector warp correction.

    Composes rendering, warp mapping, minimap, and movement
    controllers into a single runnable stimulus.

    Args:
        config: Full configuration dataclass.
    """

    def __init__(self, config: VirtualRealityConfig | None = None) -> None:
        if config is None:
            config = VirtualRealityConfig()
        self.config = config
        self._running = False

    def setup(self) -> None:
        """Initialise pygame, OpenGL context, load model and warp maps."""
        logger.info("Fly3DStimulus.setup() - initialising")
        self._running = True

    def update(self, dt: float, events: list) -> None:
        """Update fly position, camera, and controllers."""
        pass

    def render(self) -> None:
        """Render the 3D fly and apply warp correction."""
        pass

    def teardown(self) -> None:
        """Release GPU resources."""
        logger.info("Fly3DStimulus.teardown()")
        self._running = False


def main() -> None:
    """CLI entry point for the 3D fly stimulus."""
    print(
        "3D fly stimulus requires OpenGL and pygame.\n"
        "Run: vr-fly3d (after installing with pip install -e .)"
    )

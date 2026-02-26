"""2D sprite-based fly stimulus.

Displays pre-rendered turntable images of a fly as a 2D sprite,
with keyboard or autonomous control.  This is a lighter-weight
alternative to :class:`Fly3DStimulus` that does not require GLB
loading or 3D shaders.
"""

from __future__ import annotations

import logging

from virtual_reality.config.schema import VirtualRealityConfig
from virtual_reality.stimulus.base import Stimulus

logger = logging.getLogger(__name__)


class FlySpriteStimulus(Stimulus):
    """2D sprite fly stimulus.

    Args:
        config: Full configuration dataclass.
    """

    def __init__(self, config: VirtualRealityConfig | None = None) -> None:
        if config is None:
            config = VirtualRealityConfig()
        self.config = config

    def setup(self) -> None:
        logger.info("FlySpriteStimulus.setup()")

    def update(self, dt: float, events: list) -> None:
        pass

    def render(self) -> None:
        pass

    def teardown(self) -> None:
        logger.info("FlySpriteStimulus.teardown()")


def main() -> None:
    """CLI entry point for the 2D fly stimulus."""
    print("2D fly sprite stimulus placeholder.")

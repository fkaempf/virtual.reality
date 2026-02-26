"""Warp-circle calibration test stimulus.

Displays a moving circle through the projector warp map to verify
calibration quality.  Consolidates the various ``warp_circle*.py``
variants from the legacy codebase.
"""

from __future__ import annotations

import logging

from virtual_reality.config.schema import VirtualRealityConfig
from virtual_reality.stimulus.base import Stimulus

logger = logging.getLogger(__name__)


class WarpCircleStimulus(Stimulus):
    """Moving circle stimulus for warp verification.

    Args:
        config: Full configuration dataclass.
    """

    def __init__(self, config: VirtualRealityConfig | None = None) -> None:
        if config is None:
            config = VirtualRealityConfig()
        self.config = config

    def setup(self) -> None:
        logger.info("WarpCircleStimulus.setup()")

    def update(self, dt: float, events: list) -> None:
        pass

    def render(self) -> None:
        pass

    def teardown(self) -> None:
        logger.info("WarpCircleStimulus.teardown()")


def main() -> None:
    """CLI entry point for the warp circle test."""
    print("Warp circle test stimulus placeholder.")

"""Tests for concrete stimulus classes."""

from __future__ import annotations

from virtual_reality.config.schema import VirtualRealityConfig
from virtual_reality.stimulus.fly_3d import Fly3DStimulus
from virtual_reality.stimulus.fly_sprite import FlySpriteStimulus
from virtual_reality.stimulus.warp_circle import WarpCircleStimulus


class TestFly3DStimulus:
    """Tests for Fly3DStimulus."""

    def test_default_config(self) -> None:
        s = Fly3DStimulus()
        assert s.config is not None

    def test_custom_config(self) -> None:
        cfg = VirtualRealityConfig()
        s = Fly3DStimulus(config=cfg)
        assert s.config is cfg

    def test_lifecycle(self) -> None:
        s = Fly3DStimulus()
        s.setup()
        s.update(0.016, [])
        s.render()
        s.teardown()


class TestFlySpriteStimulus:
    """Tests for FlySpriteStimulus."""

    def test_lifecycle(self) -> None:
        s = FlySpriteStimulus()
        s.setup()
        s.update(0.016, [])
        s.render()
        s.teardown()


class TestWarpCircleStimulus:
    """Tests for WarpCircleStimulus."""

    def test_lifecycle(self) -> None:
        s = WarpCircleStimulus()
        s.setup()
        s.update(0.016, [])
        s.render()
        s.teardown()

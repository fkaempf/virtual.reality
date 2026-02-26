"""Tests for concrete stimulus classes.

The actual rendering lifecycle (setup/update/render/teardown) requires
an OpenGL context and hardware; those tests are marked ``@pytest.mark.gpu``.
Instantiation and parameter tests run without GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from virtual_reality.config.schema import VirtualRealityConfig
from virtual_reality.stimulus.fly_3d import Fly3DStimulus
from virtual_reality.stimulus.fly_sprite import (
    FlySpriteStimulus,
    _angle_to_index,
    _render_sprite_masked,
)
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

    def test_not_running_initially(self) -> None:
        s = Fly3DStimulus()
        assert s._running is False

    @pytest.mark.gpu
    def test_lifecycle(self) -> None:
        s = Fly3DStimulus()
        s.setup()
        s.update(0.016, [])
        s.render()
        s.teardown()


class TestFlySpriteStimulus:
    """Tests for FlySpriteStimulus."""

    def test_default_config(self) -> None:
        s = FlySpriteStimulus()
        assert s.config is not None

    def test_custom_config(self) -> None:
        cfg = VirtualRealityConfig()
        s = FlySpriteStimulus(config=cfg)
        assert s.config is cfg

    def test_ref_params(self) -> None:
        s = FlySpriteStimulus(ref_distance_mm=100.0, ref_height_px=200.0)
        assert s._ref_dist_mm == 100.0
        assert s._ref_height_px == 200.0

    @pytest.mark.gpu
    def test_lifecycle(self) -> None:
        s = FlySpriteStimulus()
        s.setup()
        s.update(0.016, [])
        s.render()
        s.teardown()


class TestWarpCircleStimulus:
    """Tests for WarpCircleStimulus."""

    def test_default_config(self) -> None:
        s = WarpCircleStimulus()
        assert s.config is not None

    def test_custom_freq(self) -> None:
        s = WarpCircleStimulus(freq_hz=0.5)
        assert s._freq_hz == 0.5

    def test_paused_initially_false(self) -> None:
        s = WarpCircleStimulus()
        assert s._paused is False

    @pytest.mark.gpu
    def test_lifecycle(self) -> None:
        s = WarpCircleStimulus()
        s.setup()
        s.update(0.016, [])
        s.render()
        s.teardown()


class TestAngleToIndex:
    """Tests for the sprite angle-to-index helper."""

    def test_zero(self) -> None:
        assert _angle_to_index(0.0, 361) == 0

    def test_180(self) -> None:
        assert _angle_to_index(180.0, 361) == 180

    def test_360_wraps_to_0(self) -> None:
        assert _angle_to_index(360.0, 361) == 0

    def test_negative_wraps(self) -> None:
        idx = _angle_to_index(-90.0, 361)
        assert idx == _angle_to_index(270.0, 361)

    def test_clamps_to_range(self) -> None:
        for angle in [0.0, 90.0, 180.0, 270.0, 359.9]:
            idx = _angle_to_index(angle, 100)
            assert 0 <= idx < 100


class TestRenderSpriteMasked:
    """Tests for the masked sprite rendering helper."""

    def test_basic_blit(self) -> None:
        cv2 = pytest.importorskip("cv2")
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        sprite = np.full((10, 10, 3), 128, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=bool)

        _render_sprite_masked(canvas, sprite, mask, 50, 50, 1.0)
        assert canvas[50, 50, 0] == 128

    def test_mask_blocks_pixels(self) -> None:
        cv2 = pytest.importorskip("cv2")
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        sprite = np.full((10, 10, 3), 200, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=bool)

        _render_sprite_masked(canvas, sprite, mask, 50, 50, 1.0)
        assert canvas.max() == 0

    def test_scaled_blit(self) -> None:
        cv2 = pytest.importorskip("cv2")
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        sprite = np.full((20, 20, 3), 100, dtype=np.uint8)
        mask = np.ones((20, 20), dtype=bool)

        _render_sprite_masked(canvas, sprite, mask, 100, 100, 2.0)
        assert canvas[100, 100, 0] == 100

    def test_off_canvas_no_crash(self) -> None:
        cv2 = pytest.importorskip("cv2")
        canvas = np.zeros((50, 50, 3), dtype=np.uint8)
        sprite = np.full((10, 10, 3), 255, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=bool)

        _render_sprite_masked(canvas, sprite, mask, -100, -100, 1.0)
        assert canvas.max() == 0

"""Tests for stimulus controllers (autonomous + keyboard)."""

from __future__ import annotations

import math

import pytest

from virtual_reality.stimulus.autonomous import AutonomousFlyController
from virtual_reality.stimulus.keyboard_control import KeyboardFlyController


class TestAutonomousFlyController:
    """Tests for the autonomous state machine."""

    def test_initial_state(self) -> None:
        ctrl = AutonomousFlyController()
        assert ctrl.state in ("run", "pause")
        assert ctrl.x == 0.0
        assert ctrl.y == 0.0

    def test_movement_changes_position(self) -> None:
        ctrl = AutonomousFlyController(speed=100.0)
        ctrl._state = "run"
        ctrl._timer = 10.0
        ctrl.heading_deg = 0.0
        ctrl.update(1.0)
        # Heading 0 = +Y direction, so y should increase.
        assert ctrl.y > 0.0

    def test_pause_no_movement(self) -> None:
        ctrl = AutonomousFlyController()
        ctrl._state = "pause"
        ctrl._timer = 10.0
        old_x, old_y = ctrl.x, ctrl.y
        ctrl.update(0.1)
        assert ctrl.x == old_x
        assert ctrl.y == old_y

    def test_arena_clamping(self) -> None:
        ctrl = AutonomousFlyController(
            arena_radius=10.0, speed=1000.0,
        )
        ctrl._state = "run"
        ctrl._timer = 100.0
        ctrl.heading_deg = 0.0
        # Move far enough to exceed arena.
        for _ in range(100):
            ctrl.update(0.1)
        dist = math.hypot(ctrl.x, ctrl.y)
        assert dist <= 10.0 + 0.01

    def test_state_transition(self) -> None:
        ctrl = AutonomousFlyController()
        initial_state = ctrl.state
        ctrl._timer = 0.01
        ctrl.update(0.1)
        # After timer expires, state should flip.
        assert ctrl.state != initial_state

    def test_heading_rad(self) -> None:
        ctrl = AutonomousFlyController()
        ctrl.heading_deg = 90.0
        assert ctrl.heading_rad == pytest.approx(math.pi / 2)


class TestKeyboardFlyController:
    """Tests for the keyboard controller."""

    def test_initial_state(self) -> None:
        ctrl = KeyboardFlyController()
        assert ctrl.x == 0.0
        assert ctrl.y == 0.0
        assert ctrl.heading_deg == 0.0

    def test_forward_movement(self) -> None:
        ctrl = KeyboardFlyController(speed=100.0)
        ctrl.forward = True
        ctrl.heading_deg = 0.0
        ctrl.update(1.0)
        assert ctrl.y > 0.0

    def test_backward_movement(self) -> None:
        ctrl = KeyboardFlyController(speed=100.0)
        ctrl.backward = True
        ctrl.heading_deg = 0.0
        ctrl.update(1.0)
        assert ctrl.y < 0.0

    def test_turn_left(self) -> None:
        ctrl = KeyboardFlyController(turn_rate=90.0)
        ctrl.turn_left = True
        ctrl.update(1.0)
        # Heading should have decreased by 90.
        assert ctrl.heading_deg == pytest.approx(270.0)

    def test_turn_right(self) -> None:
        ctrl = KeyboardFlyController(turn_rate=90.0)
        ctrl.turn_right = True
        ctrl.update(1.0)
        assert ctrl.heading_deg == pytest.approx(90.0)

    def test_arena_clamping(self) -> None:
        ctrl = KeyboardFlyController(
            arena_radius=10.0, speed=1000.0,
        )
        ctrl.forward = True
        ctrl.heading_deg = 0.0
        for _ in range(100):
            ctrl.update(0.1)
        dist = math.hypot(ctrl.x, ctrl.y)
        assert dist <= 10.0 + 0.01

    def test_no_input_no_movement(self) -> None:
        ctrl = KeyboardFlyController()
        ctrl.update(1.0)
        assert ctrl.x == 0.0
        assert ctrl.y == 0.0

    def test_heading_wraps(self) -> None:
        ctrl = KeyboardFlyController(turn_rate=400.0)
        ctrl.turn_right = True
        ctrl.update(1.0)
        assert 0 <= ctrl.heading_deg < 360

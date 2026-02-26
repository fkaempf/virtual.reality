"""Autonomous fly movement controller.

Implements a simple run/pause state machine with edge avoidance,
extracted from the ``3d_object_fly4.py`` main loop.
"""

from __future__ import annotations

import math
import random

from virtual_reality.math_utils.arena import clamp_to_arena


class AutonomousFlyController:
    """State-machine controller for automatic fly movement.

    The fly alternates between *running* (moving forward) and
    *pausing* (stationary).  When near the arena edge it turns
    inward.

    Args:
        arena_radius: Arena radius in mm.
        speed: Movement speed in mm/s.
        run_duration: Mean running duration in seconds.
        pause_duration: Mean pause duration in seconds.
        turn_rate: Maximum turn rate in degrees/s.
        edge_margin: Distance from the arena edge (mm) at which
            edge-avoidance steering kicks in.
    """

    def __init__(
        self,
        arena_radius: float = 40.0,
        speed: float = 20.0,
        run_duration: float = 2.0,
        pause_duration: float = 1.0,
        turn_rate: float = 180.0,
        edge_margin: float = 5.0,
    ) -> None:
        self.arena_radius = arena_radius
        self.speed = speed
        self.run_duration = run_duration
        self.pause_duration = pause_duration
        self.turn_rate = turn_rate
        self.edge_margin = edge_margin

        self.x: float = 0.0
        self.y: float = 0.0
        self.heading_deg: float = random.uniform(0, 360)

        self._state: str = "run"
        self._timer: float = self._random_duration("run")

    def _random_duration(self, state: str) -> float:
        mean = (
            self.run_duration if state == "run" else self.pause_duration
        )
        return max(0.1, random.gauss(mean, mean * 0.3))

    def update(self, dt: float) -> None:
        """Advance the controller by *dt* seconds.

        Args:
            dt: Time step in seconds.
        """
        self._timer -= dt
        if self._timer <= 0:
            self._state = "pause" if self._state == "run" else "run"
            self._timer = self._random_duration(self._state)

        if self._state == "run":
            self._steer(dt)
            rad = math.radians(self.heading_deg)
            dx = math.sin(rad) * self.speed * dt
            dy = math.cos(rad) * self.speed * dt
            self.x += dx
            self.y += dy

        self.x, self.y = clamp_to_arena(
            self.x, self.y, self.arena_radius,
        )

    def _steer(self, dt: float) -> None:
        """Apply random walk + edge-avoidance steering."""
        dist = math.hypot(self.x, self.y)
        threshold = self.arena_radius - self.edge_margin

        if dist > threshold and dist > 0:
            # Steer towards centre.
            to_center_deg = math.degrees(math.atan2(-self.x, -self.y))
            diff = (to_center_deg - self.heading_deg + 180) % 360 - 180
            max_turn = self.turn_rate * dt
            self.heading_deg += max(
                -max_turn, min(max_turn, diff),
            )
        else:
            # Random wandering.
            self.heading_deg += random.gauss(0, self.turn_rate * 0.3 * dt)

        self.heading_deg %= 360

    @property
    def heading_rad(self) -> float:
        """Current heading in radians."""
        return math.radians(self.heading_deg)

    @property
    def state(self) -> str:
        """Current state: ``"run"`` or ``"pause"``."""
        return self._state

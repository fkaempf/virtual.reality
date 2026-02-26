"""Keyboard-driven fly movement controller.

Maps WASD / arrow keys to fly translation and rotation,
extracted from the ``3d_object_fly4.py`` event loop.
"""

from __future__ import annotations

import math

from virtual_reality.math_utils.arena import clamp_to_arena


class KeyboardFlyController:
    """WASD / arrow-key fly movement controller.

    Args:
        arena_radius: Arena radius in mm.
        speed: Movement speed in mm/s.
        turn_rate: Rotation speed in degrees/s.
    """

    def __init__(
        self,
        arena_radius: float = 40.0,
        speed: float = 20.0,
        turn_rate: float = 180.0,
    ) -> None:
        self.arena_radius = arena_radius
        self.speed = speed
        self.turn_rate = turn_rate

        self.x: float = 0.0
        self.y: float = 0.0
        self.heading_deg: float = 0.0

        # Input state (set each frame from key state).
        self.forward: bool = False
        self.backward: bool = False
        self.turn_left: bool = False
        self.turn_right: bool = False

    def update(self, dt: float) -> None:
        """Advance the controller by *dt* seconds.

        Args:
            dt: Time step in seconds.
        """
        if self.turn_left:
            self.heading_deg -= self.turn_rate * dt
        if self.turn_right:
            self.heading_deg += self.turn_rate * dt
        self.heading_deg %= 360

        move = 0.0
        if self.forward:
            move += self.speed * dt
        if self.backward:
            move -= self.speed * dt

        if move != 0.0:
            rad = math.radians(self.heading_deg)
            self.x += math.sin(rad) * move
            self.y += math.cos(rad) * move

        self.x, self.y = clamp_to_arena(
            self.x, self.y, self.arena_radius,
        )

    @property
    def heading_rad(self) -> float:
        """Current heading in radians."""
        return math.radians(self.heading_deg)

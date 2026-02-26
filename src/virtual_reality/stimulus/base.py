"""Abstract base class for stimulus presentations.

Every stimulus follows the same lifecycle::

    stimulus.setup()
    while running:
        stimulus.update(dt, events)
        stimulus.render()
    stimulus.teardown()

The :meth:`run` convenience method implements this loop with
pygame event handling and FPS tracking.
"""

from __future__ import annotations

import abc
import logging
import time

logger = logging.getLogger(__name__)


class Stimulus(abc.ABC):
    """Abstract stimulus base class.

    Subclasses must implement :meth:`setup`, :meth:`update`,
    :meth:`render`, and :meth:`teardown`.
    """

    @abc.abstractmethod
    def setup(self) -> None:
        """Initialise GPU resources, load models, etc."""

    @abc.abstractmethod
    def update(self, dt: float, events: list) -> None:
        """Advance simulation state.

        Args:
            dt: Wall-clock seconds since the last frame.
            events: List of ``pygame.event.Event`` objects.
        """

    @abc.abstractmethod
    def render(self) -> None:
        """Draw the current frame."""

    @abc.abstractmethod
    def teardown(self) -> None:
        """Release GPU resources and close windows."""

    def run(self, target_fps: int = 60) -> None:
        """Run the stimulus main loop.

        Args:
            target_fps: Target frame rate (used for clock tick).
        """
        import pygame

        self.setup()
        clock = pygame.time.Clock()
        running = True
        last_time = time.perf_counter()

        try:
            while running:
                now = time.perf_counter()
                dt = now - last_time
                last_time = now

                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False

                if not running:
                    break

                self.update(dt, events)
                self.render()
                pygame.display.flip()
                clock.tick(target_fps)
        finally:
            self.teardown()

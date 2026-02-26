"""Pygame/OpenGL window creation helpers."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def setup_pygame_window(
    width: int,
    height: int,
    monitor_x: int = 0,
    monitor_y: int = 0,
    borderless: bool = True,
    opengl: bool = True,
) -> object:
    """Create a pygame display window with optional OpenGL context.

    Args:
        width: Window width in pixels.
        height: Window height in pixels.
        monitor_x: Desktop X offset for window placement.
        monitor_y: Desktop Y offset for window placement.
        borderless: If True, create a borderless window.
        opengl: If True, request an OpenGL-capable surface.

    Returns:
        The ``pygame.Surface`` returned by ``pygame.display.set_mode``.
    """
    import pygame
    from pygame.locals import DOUBLEBUF, NOFRAME, OPENGL

    os.environ["SDL_VIDEO_WINDOW_POS"] = (
        f"{monitor_x},{monitor_y}"
    )

    flags = 0
    if opengl:
        flags |= DOUBLEBUF | OPENGL
    if borderless:
        flags |= NOFRAME

    if not pygame.get_init():
        pygame.init()

    screen = pygame.display.set_mode((width, height), flags)
    logger.info(
        "Created %dx%d window at (%d, %d) [gl=%s, borderless=%s]",
        width, height, monitor_x, monitor_y, opengl, borderless,
    )
    return screen

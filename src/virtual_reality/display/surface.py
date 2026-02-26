"""Pygame surface conversion utilities.

Replaces 8+ copies of ``frame_to_surface`` and similar helpers
across the legacy codebase.
"""

from __future__ import annotations

import numpy as np


def frame_to_surface(
    img: np.ndarray,
    target_size: tuple[int, int] | None = None,
) -> object:
    """Convert a numpy image array to a pygame Surface.

    Handles both grayscale ``(H, W)`` and color ``(H, W, 3)`` inputs.
    Color images are assumed BGR (OpenCV convention) and are converted
    to RGB for pygame.

    Args:
        img: Input image as a uint8 numpy array.
        target_size: Optional ``(width, height)`` to scale the surface.
            If ``None``, the surface matches the image dimensions.

    Returns:
        A ``pygame.Surface`` ready for blitting.
    """
    import cv2
    import pygame

    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    surface = pygame.image.frombuffer(
        rgb.tobytes(), (w, h), "RGB",
    )
    if target_size and (target_size[0] != w or target_size[1] != h):
        surface = pygame.transform.scale(surface, target_size)
    return surface


def bgr_to_surface(bgr: np.ndarray) -> object:
    """Convert a BGR numpy image to a pygame Surface.

    Convenience wrapper around :func:`frame_to_surface` that makes
    the BGR assumption explicit.

    Args:
        bgr: A ``(H, W, 3)`` uint8 BGR image.

    Returns:
        A ``pygame.Surface``.
    """
    return frame_to_surface(bgr)

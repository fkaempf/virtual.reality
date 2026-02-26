"""Monitor detection and selection utilities.

Replaces 20+ copies of ``pick_monitor`` scattered across the legacy
codebase with a single implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MonitorInfo:
    """Describes a connected display.

    Attributes:
        x: Horizontal offset of the monitor in desktop coordinates.
        y: Vertical offset of the monitor.
        width: Width in pixels.
        height: Height in pixels.
        name: Human-readable monitor name (may be empty).
    """

    x: int = 0
    y: int = 0
    width: int = 800
    height: int = 600
    name: str = ""


def pick_monitor(
    target_w: int = 0,
    target_h: int = 0,
    which: str = "right",
    fallback_w: int = 800,
    fallback_h: int = 600,
) -> MonitorInfo:
    """Select a monitor from the connected displays.

    Tries to use ``screeninfo`` for auto-detection. Falls back to a
    synthetic monitor when the library is unavailable or no displays
    are found.

    Args:
        target_w: Preferred width (used for logging only).
        target_h: Preferred height (used for logging only).
        which: Selection strategy: ``"right"`` for the rightmost
            monitor, ``"left"`` for the leftmost.
        fallback_w: Width to use if detection fails.
        fallback_h: Height to use if detection fails.

    Returns:
        A ``MonitorInfo`` describing the chosen monitor.
    """
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
    except Exception:
        monitors = []

    if not monitors:
        logger.info(
            "No monitors detected, using fallback %dx%d",
            fallback_w, fallback_h,
        )
        return MonitorInfo(
            x=0, y=0, width=fallback_w, height=fallback_h,
        )

    if which == "right":
        chosen = max(monitors, key=lambda m: m.x)
    else:
        chosen = min(monitors, key=lambda m: m.x)

    logger.info(
        "Selected monitor: %s (%dx%d at %d,%d)",
        chosen.name, chosen.width, chosen.height,
        chosen.x, chosen.y,
    )

    return MonitorInfo(
        x=chosen.x,
        y=chosen.y,
        width=chosen.width,
        height=chosen.height,
        name=getattr(chosen, "name", ""),
    )

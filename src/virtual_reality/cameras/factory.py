"""Camera factory for creating driver instances by name."""

from __future__ import annotations

from virtual_reality.cameras.base import Camera


def create_camera(
    camera_type: str,
    exposure_ms: float = 10.0,
    gain_db: float = 0.0,
    **kwargs: object,
) -> Camera:
    """Create a camera driver instance by type name.

    Args:
        camera_type: Driver name, one of ``"alvium"`` or ``"rotpy"``.
        exposure_ms: Exposure time in milliseconds.
        gain_db: Analog gain in decibels.
        **kwargs: Additional driver-specific arguments.

    Returns:
        A ``Camera``-compatible driver instance.

    Raises:
        ValueError: If *camera_type* is not recognized.
        ImportError: If the required driver package is not installed.
    """
    camera_type = camera_type.lower().strip()

    if camera_type == "alvium":
        from virtual_reality.cameras.alvium import CamAlvium
        return CamAlvium(
            exposure_ms=exposure_ms,
            gain_db=gain_db,
            **kwargs,
        )

    if camera_type in ("rotpy", "flir", "spinnaker"):
        from virtual_reality.cameras.rotpy_driver import CamRotPy
        return CamRotPy(
            exposure_ms=exposure_ms,
            gain_db=gain_db,
            **kwargs,
        )

    raise ValueError(
        f"Unknown camera type: {camera_type!r}. "
        f"Supported: 'alvium', 'rotpy'."
    )

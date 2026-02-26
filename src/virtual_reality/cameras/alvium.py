"""Allied Vision Alvium camera driver via Harvesters/GenTL.

Requires the ``harvesters`` package and an installed GenTL producer
(e.g., Allied Vision Vimba X).
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class CamAlvium:
    """Driver for Allied Vision Alvium cameras.

    Attributes:
        exposure_ms: Exposure time in milliseconds.
        gain_db: Analog gain in decibels.
    """

    def __init__(
        self,
        exposure_ms: float = 10.0,
        gain_db: float = 6.0,
        cti_path: str | None = None,
    ) -> None:
        self.exposure_ms = exposure_ms
        self.gain_db = gain_db
        self._cti_path = cti_path or os.environ.get(
            "CAM_CTI_PATH",
            r"C:\Program Files\Allied Vision"
            r"\Vimba X\cti\VimbaUSBTL.cti",
        )
        self._timeout_s = 2.0
        self._h = None
        self._ia = None
        self._nm = None

    def start(self) -> None:
        """Initialize Harvester, configure camera, begin acquisition."""
        from harvesters.core import Harvester

        self._h = Harvester()
        self._h.add_file(self._cti_path)
        self._h.update()

        self._ia = self._h.create()
        self._nm = self._ia.remote_device.node_map

        nm = self._nm
        nm.PixelFormat.value = "Mono8"
        nm.ExposureAuto.value = "Off"
        nm.GainAuto.value = "Off"
        nm.TriggerMode.value = "Off"
        nm.TriggerSelector.value = "FrameStart"
        nm.TriggerSource.value = "Software"
        nm.TriggerMode.value = "On"
        nm.ExposureTime.value = float(self.exposure_ms) * 1000.0
        nm.Gain.value = float(self.gain_db)

        self._ia.start()

    def grab(self, timeout_s: float = 1.0) -> np.ndarray:
        """Software-trigger and fetch one frame.

        Args:
            timeout_s: Maximum wait time in seconds.

        Returns:
            A contiguous uint8 array of shape ``(H, W)``.
        """
        timeout_ms = int(max(1, (timeout_s or self._timeout_s) * 1000))
        self._nm.TriggerSoftware.execute()
        buf = self._ia.fetch(timeout=timeout_ms)
        comp = buf.payload.components[0]
        h, w = int(comp.height), int(comp.width)
        mv = comp.data
        stride = len(mv) // h
        frame = (
            np.frombuffer(mv, np.uint8, count=h * stride)
            .reshape(h, stride)[:, :w]
            .copy()
        )
        buf.queue()
        return np.ascontiguousarray(frame)

    def stop(self) -> None:
        """Stop acquisition and release all resources."""
        if self._ia is not None:
            try:
                self._ia.stop()
            except Exception:
                logger.debug("Failed to stop acquisition", exc_info=True)
            try:
                self._ia.destroy()
            except Exception:
                logger.debug("Failed to destroy acquirer", exc_info=True)
        if self._h is not None:
            try:
                self._h.reset()
            except Exception:
                logger.debug("Failed to reset Harvester", exc_info=True)

    def __enter__(self) -> CamAlvium:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

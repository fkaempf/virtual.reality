"""FLIR/Spinnaker camera driver via RotPy.

Requires the ``rotpy`` package and an installed Spinnaker SDK.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class CamRotPy:
    """Driver for FLIR cameras via the RotPy/Spinnaker wrapper.

    Attributes:
        exposure_ms: Exposure time in milliseconds.
        gain_db: Analog gain in decibels.
    """

    def __init__(
        self,
        exposure_ms: float = 10.0,
        gain_db: float = 0.0,
    ) -> None:
        self.exposure_ms = exposure_ms
        self.gain_db = gain_db
        self._cam = None
        self._system = None

    def start(self) -> None:
        """Initialize Spinnaker, configure camera, begin acquisition."""
        from rotpy.camera import CameraList
        from rotpy.system import SpinSystem

        self._system = SpinSystem()
        cam_list = CameraList.create_from_system(
            self._system, True, True,
        )
        if cam_list.get_size() < 1:
            raise RuntimeError("No FLIR camera found")
        self._cam = cam_list.create_camera_by_index(0)
        c = self._cam
        c.init_cam()

        nodes = c.camera_nodes
        self._try_set(nodes.PixelFormat, "Mono8")
        self._try_set(nodes.ExposureAuto, "Off")
        self._try_set_value(
            nodes.ExposureTime,
            max(500.0, min(self.exposure_ms * 1000.0, 3e7)),
        )
        self._try_set(nodes.GainAuto, "Off")
        self._try_set_value(nodes.Gain, self.gain_db)

        self._try_set(nodes.TriggerMode, "Off")
        self._try_set(nodes.TriggerSelector, "FrameStart")
        self._try_set(nodes.TriggerSource, "Software")
        self._try_set(nodes.TriggerMode, "On")

        c.begin_acquisition()

        # Flush initial stale frames.
        for _ in range(3):
            try:
                im = c.get_next_image(timeout=0.2)
                im.release()
            except Exception:
                break

    def grab(self, timeout_s: float = 1.0) -> np.ndarray:
        """Software-trigger and fetch one frame.

        Args:
            timeout_s: Maximum wait time in seconds.

        Returns:
            A contiguous uint8 array of shape ``(H, W)``.
        """
        c = self._cam
        try:
            c.camera_nodes.TriggerSoftware.execute_node()
        except Exception:
            logger.debug("Software trigger failed", exc_info=True)

        im = c.get_next_image(timeout=timeout_s)
        try:
            try:
                im = im.convert_fmt("Mono8")
            except Exception:
                pass
            h = im.get_height()
            w = im.get_width()
            stride = im.get_stride()
            try:
                b = im.get_image_data_bytes()
            except Exception:
                b = bytes(im.get_image_data_memoryview())
            frame = (
                np.frombuffer(b, np.uint8)[:h * stride]
                .reshape(h, stride)[:, :w]
            )
            return np.ascontiguousarray(frame)
        finally:
            im.release()

    def stop(self) -> None:
        """Stop acquisition and release all resources."""
        if self._cam is not None:
            try:
                self._cam.end_acquisition()
            except Exception:
                logger.debug(
                    "Failed to end acquisition", exc_info=True,
                )
            try:
                self._cam.deinit_cam()
            except Exception:
                logger.debug(
                    "Failed to deinit camera", exc_info=True,
                )
            try:
                self._cam.release()
            except Exception:
                logger.debug(
                    "Failed to release camera", exc_info=True,
                )

    def __enter__(self) -> CamRotPy:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    @staticmethod
    def _try_set(node: object, value: str) -> None:
        try:
            node.set_node_value_from_str(value)
        except Exception:
            logger.debug(
                "Could not set %s to %s", node, value,
            )

    @staticmethod
    def _try_set_value(node: object, value: float) -> None:
        try:
            node.set_node_value(value)
        except Exception:
            logger.debug(
                "Could not set %s to %s", node, value,
            )

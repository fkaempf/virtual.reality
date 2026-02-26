"""Tests for virtual_reality.cameras (protocol and factory)."""

from __future__ import annotations

from unittest import mock

import pytest

from virtual_reality.cameras.base import Camera
from virtual_reality.cameras.factory import create_camera


class TestCameraProtocol:
    """Tests for the Camera protocol."""

    def test_protocol_has_required_methods(self) -> None:
        assert hasattr(Camera, "start")
        assert hasattr(Camera, "grab")
        assert hasattr(Camera, "stop")


class TestCreateCamera:
    """Tests for create_camera factory."""

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown camera type"):
            create_camera("nonexistent")

    def test_alvium_import_error(self) -> None:
        with mock.patch.dict(
            "sys.modules", {"harvesters": None, "harvesters.core": None},
        ):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                cam = create_camera("alvium")
                cam.start()

    def test_rotpy_import_error(self) -> None:
        with mock.patch.dict(
            "sys.modules",
            {"rotpy": None, "rotpy.system": None, "rotpy.camera": None},
        ):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                cam = create_camera("rotpy")
                cam.start()

    def test_case_insensitive(self) -> None:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            cam = create_camera("ALVIUM")
            cam.start()

    def test_flir_alias(self) -> None:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            cam = create_camera("flir")
            cam.start()

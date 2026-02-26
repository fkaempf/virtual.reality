"""Tests for virtual_reality.config.schema."""

from __future__ import annotations

import dataclasses

from virtual_reality.config.schema import (
    ArenaConfig,
    AutonomousConfig,
    CameraConfig,
    CalibrationConfig,
    DisplayConfig,
    FlyModelConfig,
    LightingConfig,
    MinimapConfig,
    MovementConfig,
    ScalingConfig,
    VirtualRealityConfig,
    WarpConfig,
)


class TestArenaConfig:
    """Tests for ArenaConfig."""

    def test_default_radius(self) -> None:
        cfg = ArenaConfig()
        assert cfg.radius_mm == 40.0

    def test_custom_radius(self) -> None:
        cfg = ArenaConfig(radius_mm=100.0)
        assert cfg.radius_mm == 100.0


class TestFlyModelConfig:
    """Tests for FlyModelConfig."""

    def test_defaults(self) -> None:
        cfg = FlyModelConfig()
        assert cfg.model_path == ""
        assert cfg.phys_length_mm == 3.0
        assert cfg.base_scale == 1.0
        assert cfg.yaw_offset_deg == 0.0


class TestCameraConfig:
    """Tests for CameraConfig."""

    def test_defaults(self) -> None:
        cfg = CameraConfig()
        assert cfg.x_mm == 0.0
        assert cfg.y_mm == -40.0
        assert cfg.height_mm == 0.89
        assert cfg.projection == "equirect"
        assert cfg.fov_x_deg == 200.0

    def test_custom_projection(self) -> None:
        cfg = CameraConfig(projection="perspective")
        assert cfg.projection == "perspective"


class TestMovementConfig:
    """Tests for MovementConfig."""

    def test_defaults(self) -> None:
        cfg = MovementConfig()
        assert cfg.speed_mm_s == 20.0
        assert cfg.back_mm_s == 12.8
        assert cfg.turn_deg_s == 200.0


class TestAutonomousConfig:
    """Tests for AutonomousConfig."""

    def test_defaults(self) -> None:
        cfg = AutonomousConfig()
        assert cfg.enabled is True
        assert cfg.edge_thresh_frac == 0.8

    def test_disabled(self) -> None:
        cfg = AutonomousConfig(enabled=False)
        assert cfg.enabled is False


class TestLightingConfig:
    """Tests for LightingConfig."""

    def test_defaults(self) -> None:
        cfg = LightingConfig()
        assert cfg.ambient == 0.6
        assert len(cfg.intensities) == 4
        assert cfg.elevation_deg == 65.0


class TestMinimapConfig:
    """Tests for MinimapConfig."""

    def test_defaults(self) -> None:
        cfg = MinimapConfig()
        assert cfg.width == 420
        assert cfg.trail_color == (255, 200, 0)


class TestDisplayConfig:
    """Tests for DisplayConfig."""

    def test_defaults(self) -> None:
        cfg = DisplayConfig()
        assert cfg.target_fps == 60
        assert cfg.bg_color == (255, 255, 255)


class TestScalingConfig:
    """Tests for ScalingConfig."""

    def test_defaults(self) -> None:
        cfg = ScalingConfig()
        assert cfg.screen_distance_mm == 60.0
        assert cfg.apparent_distance_mm is None


class TestCalibrationConfig:
    """Tests for CalibrationConfig."""

    def test_defaults(self) -> None:
        cfg = CalibrationConfig()
        assert cfg.proj_w == 1280
        assert cfg.proj_h == 800
        assert cfg.mode == "sine_hybrid"


class TestVirtualRealityConfig:
    """Tests for VirtualRealityConfig."""

    def test_default_construction(self) -> None:
        cfg = VirtualRealityConfig()
        assert cfg.arena.radius_mm == 40.0
        assert cfg.fly_model.phys_length_mm == 3.0
        assert cfg.camera.projection == "equirect"

    def test_nested_modification(self) -> None:
        cfg = VirtualRealityConfig()
        cfg.arena.radius_mm = 80.0
        assert cfg.arena.radius_mm == 80.0

    def test_all_fields_are_dataclasses(self) -> None:
        cfg = VirtualRealityConfig()
        for f in dataclasses.fields(cfg):
            value = getattr(cfg, f.name)
            assert dataclasses.is_dataclass(value), (
                f"Field {f.name} is not a dataclass"
            )

    def test_independent_instances(self) -> None:
        cfg1 = VirtualRealityConfig()
        cfg2 = VirtualRealityConfig()
        cfg1.arena.radius_mm = 999.0
        assert cfg2.arena.radius_mm == 40.0

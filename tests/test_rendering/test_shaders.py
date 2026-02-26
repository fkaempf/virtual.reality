"""Tests for virtual_reality.rendering.shaders."""

from __future__ import annotations

import pytest

from virtual_reality.rendering.shaders import (
    FLY_FRAG_SRC,
    FLY_VERT_SRC,
    WARP_FRAG_SRC,
    WARP_VERT_SRC,
)


class TestShaderSources:
    """Validate shader source strings contain expected elements."""

    def test_warp_vert_has_version(self) -> None:
        assert "#version 330 core" in WARP_VERT_SRC

    def test_warp_frag_has_uniforms(self) -> None:
        assert "u_cam" in WARP_FRAG_SRC
        assert "u_warp" in WARP_FRAG_SRC
        assert "u_useWarp" in WARP_FRAG_SRC

    def test_fly_vert_has_projection_modes(self) -> None:
        assert "u_projMode" in FLY_VERT_SRC
        assert "equidistant" in FLY_VERT_SRC.lower() or "fisheye" in FLY_VERT_SRC.lower()
        assert "equirectangular" in FLY_VERT_SRC.lower()

    def test_fly_frag_has_lighting(self) -> None:
        assert "u_ambient" in FLY_FRAG_SRC
        assert "u_lightDirs" in FLY_FRAG_SRC
        assert "u_lightIntensities" in FLY_FRAG_SRC
        assert "u_lightMaxGain" in FLY_FRAG_SRC

    def test_fly_vert_has_inputs(self) -> None:
        assert "in_pos" in FLY_VERT_SRC
        assert "in_normal" in FLY_VERT_SRC
        assert "in_color" in FLY_VERT_SRC
        assert "in_uv" in FLY_VERT_SRC

    def test_fly_frag_has_texture_support(self) -> None:
        assert "u_hasTexture" in FLY_FRAG_SRC
        assert "u_baseColorTex" in FLY_FRAG_SRC

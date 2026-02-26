"""Tests for virtual_reality.rendering.projections."""

from __future__ import annotations

import pytest

from virtual_reality.rendering.projections import (
    PROJ_EQUIDISTANT,
    PROJ_EQUIRECTANGULAR,
    PROJ_PERSPECTIVE,
    projection_mode_to_int,
)


class TestProjectionModeToInt:
    """Tests for projection_mode_to_int."""

    def test_perspective(self) -> None:
        assert projection_mode_to_int("perspective") == PROJ_PERSPECTIVE

    def test_equidistant(self) -> None:
        assert projection_mode_to_int("equidistant") == PROJ_EQUIDISTANT

    def test_equirect(self) -> None:
        assert projection_mode_to_int("equirect") == PROJ_EQUIRECTANGULAR

    def test_case_insensitive(self) -> None:
        assert projection_mode_to_int("PERSPECTIVE") == PROJ_PERSPECTIVE
        assert projection_mode_to_int("Equirect") == PROJ_EQUIRECTANGULAR

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown projection"):
            projection_mode_to_int("ortho")

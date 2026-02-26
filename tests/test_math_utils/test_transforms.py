"""Tests for virtual_reality.math_utils.transforms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from virtual_reality.math_utils.transforms import (
    look_at,
    mat4_rotate_x,
    mat4_rotate_y,
    mat4_scale,
    mat4_translate,
    perspective,
    quat_to_mat4,
)


class TestMat4Translate:
    """Tests for mat4_translate."""

    def test_identity_at_zero(self) -> None:
        m = mat4_translate(0, 0, 0)
        np.testing.assert_array_almost_equal(m, np.eye(4))

    def test_translation_values(self) -> None:
        m = mat4_translate(1.0, 2.0, 3.0)
        assert m[0, 3] == pytest.approx(1.0)
        assert m[1, 3] == pytest.approx(2.0)
        assert m[2, 3] == pytest.approx(3.0)

    def test_dtype_is_float32(self) -> None:
        m = mat4_translate(1, 2, 3)
        assert m.dtype == np.float32

    def test_transforms_point(self) -> None:
        m = mat4_translate(10, 20, 30)
        point = np.array([1, 2, 3, 1], dtype=np.float32)
        result = m @ point
        np.testing.assert_array_almost_equal(
            result, [11, 22, 33, 1]
        )


class TestMat4RotateY:
    """Tests for mat4_rotate_y."""

    def test_zero_is_identity(self) -> None:
        m = mat4_rotate_y(0.0)
        np.testing.assert_array_almost_equal(m, np.eye(4))

    def test_90_degrees(self) -> None:
        m = mat4_rotate_y(math.pi / 2)
        point = np.array([1, 0, 0, 1], dtype=np.float32)
        result = m @ point
        np.testing.assert_array_almost_equal(
            result, [0, 0, -1, 1], decimal=5
        )

    def test_dtype_is_float32(self) -> None:
        m = mat4_rotate_y(1.0)
        assert m.dtype == np.float32


class TestMat4RotateX:
    """Tests for mat4_rotate_x."""

    def test_zero_is_identity(self) -> None:
        m = mat4_rotate_x(0.0)
        np.testing.assert_array_almost_equal(m, np.eye(4))

    def test_90_degrees(self) -> None:
        m = mat4_rotate_x(math.pi / 2)
        point = np.array([0, 1, 0, 1], dtype=np.float32)
        result = m @ point
        np.testing.assert_array_almost_equal(
            result, [0, 0, 1, 1], decimal=5
        )


class TestMat4Scale:
    """Tests for mat4_scale."""

    def test_scale_one_is_identity(self) -> None:
        m = mat4_scale(1.0)
        np.testing.assert_array_almost_equal(m, np.eye(4))

    def test_scale_two(self) -> None:
        m = mat4_scale(2.0)
        point = np.array([1, 2, 3, 1], dtype=np.float32)
        result = m @ point
        np.testing.assert_array_almost_equal(
            result, [2, 4, 6, 1]
        )


class TestQuatToMat4:
    """Tests for quat_to_mat4."""

    def test_identity_quaternion(self) -> None:
        q = np.array([0, 0, 0, 1], dtype=np.float32)
        m = quat_to_mat4(q)
        np.testing.assert_array_almost_equal(m, np.eye(4))

    def test_180_about_y(self) -> None:
        q = np.array([0, 1, 0, 0], dtype=np.float32)
        m = quat_to_mat4(q)
        point = np.array([1, 0, 0, 1], dtype=np.float32)
        result = m @ point
        np.testing.assert_array_almost_equal(
            result, [-1, 0, 0, 1], decimal=5
        )

    def test_dtype_is_float32(self) -> None:
        q = np.array([0, 0, 0, 1])
        m = quat_to_mat4(q)
        assert m.dtype == np.float32


class TestPerspective:
    """Tests for perspective."""

    def test_basic_shape(self) -> None:
        m = perspective(math.radians(60), 1.0, 0.1, 100.0)
        assert m.shape == (4, 4)
        assert m.dtype == np.float32

    def test_ultrawide_clamping(self) -> None:
        m_wide = perspective(
            math.radians(200), 1.0, 0.1, 100.0,
            allow_ultrawide=True,
        )
        assert np.all(np.isfinite(m_wide))

    def test_narrow_clamping(self) -> None:
        m = perspective(
            math.radians(200), 1.0, 0.1, 100.0,
            allow_ultrawide=False,
        )
        assert np.all(np.isfinite(m))

    def test_aspect_ratio_affects_x(self) -> None:
        m1 = perspective(math.radians(60), 1.0, 0.1, 100.0)
        m2 = perspective(math.radians(60), 2.0, 0.1, 100.0)
        assert m1[0, 0] != pytest.approx(m2[0, 0])


class TestLookAt:
    """Tests for look_at."""

    def test_basic_shape(self) -> None:
        m = look_at(
            np.array([0, 0, 5]),
            np.array([0, 0, 0]),
            np.array([0, 1, 0]),
        )
        assert m.shape == (4, 4)
        assert m.dtype == np.float32

    def test_looking_down_z(self) -> None:
        m = look_at(
            np.array([0, 0, 1]),
            np.array([0, 0, 0]),
            np.array([0, 1, 0]),
        )
        # Origin should map to (0, 0, -1) in view space.
        origin = np.array([0, 0, 0, 1], dtype=np.float32)
        result = m @ origin
        assert result[2] == pytest.approx(-1.0, abs=1e-5)

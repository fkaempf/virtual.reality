"""Tests for virtual_reality.rendering.glb_loader."""

from __future__ import annotations

import numpy as np
import pytest

# GLB loader needs both cv2 and pygltflib at import time.
cv2 = pytest.importorskip("cv2")
pytest.importorskip("pygltflib")

from virtual_reality.rendering.glb_loader import (
    DrawCall,
    Mesh,
    VERTEX_STRIDE,
    _quat_to_mat4,
    _node_local_matrix,
)


class TestDrawCall:
    """Tests for the DrawCall dataclass."""

    def test_defaults(self) -> None:
        dc = DrawCall()
        assert dc.base_index == 0
        assert dc.count == 0
        np.testing.assert_array_equal(
            dc.base_color_factor, np.ones(4, dtype=np.float32),
        )
        assert dc.base_color_image is None


class TestMesh:
    """Tests for the Mesh dataclass."""

    def test_vertex_stride(self) -> None:
        assert VERTEX_STRIDE == 12

    def test_empty_mesh(self) -> None:
        m = Mesh(
            vertices=np.zeros((0, 12), dtype=np.float32),
            indices=np.zeros(0, dtype=np.uint32),
        )
        assert m.vertices.shape[1] == 12
        assert len(m.draw_calls) == 0


class TestQuatToMat4:
    """Tests for quaternion to matrix conversion."""

    def test_identity_quaternion(self) -> None:
        mat = _quat_to_mat4([0, 0, 0, 1])
        np.testing.assert_array_almost_equal(mat, np.eye(4, dtype=np.float32))

    def test_90deg_rotation_z(self) -> None:
        """Quaternion for 90-degree rotation about Z."""
        import math
        angle = math.pi / 2
        q = [0, 0, math.sin(angle / 2), math.cos(angle / 2)]
        mat = _quat_to_mat4(q)
        # After 90-deg Z rotation: x-axis → y-axis
        x_axis = mat[:3, 0]
        assert x_axis[1] == pytest.approx(1.0, abs=1e-6)


class TestNodeLocalMatrix:
    """Tests for _node_local_matrix."""

    def test_identity_node(self) -> None:
        class MockNode:
            matrix = None
            translation = None
            rotation = None
            scale = None

        mat = _node_local_matrix(MockNode())
        np.testing.assert_array_almost_equal(mat, np.eye(4, dtype=np.float32))

    def test_translation_only(self) -> None:
        class MockNode:
            matrix = None
            translation = [1.0, 2.0, 3.0]
            rotation = None
            scale = None

        mat = _node_local_matrix(MockNode())
        assert mat[0, 3] == pytest.approx(1.0)
        assert mat[1, 3] == pytest.approx(2.0)
        assert mat[2, 3] == pytest.approx(3.0)

    def test_scale_only(self) -> None:
        class MockNode:
            matrix = None
            translation = None
            rotation = None
            scale = [2.0, 3.0, 4.0]

        mat = _node_local_matrix(MockNode())
        assert mat[0, 0] == pytest.approx(2.0)
        assert mat[1, 1] == pytest.approx(3.0)
        assert mat[2, 2] == pytest.approx(4.0)

"""4x4 matrix transform utilities for 3D rendering.

All functions return ``numpy.float32`` 4x4 matrices suitable for
direct upload to OpenGL uniforms.
"""

from __future__ import annotations

import math

import numpy as np


def mat4_translate(x: float, y: float, z: float) -> np.ndarray:
    """Build a 4x4 translation matrix.

    Args:
        x: Translation along the X axis.
        y: Translation along the Y axis.
        z: Translation along the Z axis.

    Returns:
        A 4x4 float32 translation matrix.
    """
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def mat4_rotate_y(angle_rad: float) -> np.ndarray:
    """Build a 4x4 rotation matrix about the Y axis.

    Args:
        angle_rad: Rotation angle in radians.

    Returns:
        A 4x4 float32 rotation matrix.
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def mat4_rotate_x(angle_rad: float) -> np.ndarray:
    """Build a 4x4 rotation matrix about the X axis.

    Args:
        angle_rad: Rotation angle in radians.

    Returns:
        A 4x4 float32 rotation matrix.
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def mat4_scale(s: float) -> np.ndarray:
    """Build a 4x4 uniform scale matrix.

    Args:
        s: Scale factor applied to all three axes.

    Returns:
        A 4x4 float32 scale matrix.
    """
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = s
    m[1, 1] = s
    m[2, 2] = s
    return m


def quat_to_mat4(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion ``(x, y, z, w)`` to a 4x4 rotation matrix.

    Args:
        q: A length-4 array or sequence ``[x, y, z, w]``.

    Returns:
        A 4x4 float32 rotation matrix.
    """
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),
         2 * (xz + wy), 0.0],
        [2 * (xy + wz), 1 - 2 * (xx + zz),
         2 * (yz - wx), 0.0],
        [2 * (xz - wy), 2 * (yz + wx),
         1 - 2 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def perspective(
    fov_y_rad: float,
    aspect: float,
    z_near: float,
    z_far: float,
    allow_ultrawide: bool = True,
) -> np.ndarray:
    """Build an OpenGL perspective projection matrix.

    Guards against pathological FOV values that would cause ``tan()``
    to blow up.

    Args:
        fov_y_rad: Vertical field of view in radians.
        aspect: Width / height aspect ratio.
        z_near: Near clipping plane distance.
        z_far: Far clipping plane distance.
        allow_ultrawide: If True, clamp FOV to 179 degrees. Otherwise
            clamp to 120 degrees.

    Returns:
        A 4x4 float32 perspective matrix.
    """
    max_fov = math.radians(179.0 if allow_ultrawide else 120.0)
    fov_y_rad = min(fov_y_rad, max_fov)

    f = 1.0 / math.tan(0.5 * fov_y_rad)
    nf = 1.0 / (z_near - z_far)
    return np.array([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (z_far + z_near) * nf,
         2 * z_far * z_near * nf],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype=np.float32)


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Build a view matrix looking from *eye* toward *target*.

    Args:
        eye: Camera position as a 3-element array.
        target: Look-at target position as a 3-element array.
        up: World up direction as a 3-element array.

    Returns:
        A 4x4 float32 view matrix.
    """
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    f = target - eye
    f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m

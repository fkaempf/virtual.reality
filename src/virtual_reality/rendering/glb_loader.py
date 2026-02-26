"""GLB/glTF 2.0 model loader.

Loads a ``.glb`` file via ``pygltflib``, flattens all meshes and
primitives into a single interleaved vertex buffer, and returns
draw-call metadata (material colours and base-colour textures).

The vertex layout is::

    [pos.xyz (3), normal.xyz (3), color.rgba (4), uv.xy (2)]

totalling 12 floats per vertex.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from pygltflib import GLTF2

logger = logging.getLogger(__name__)

# Floats per vertex in the interleaved buffer.
VERTEX_STRIDE = 12


@dataclass
class DrawCall:
    """Describes one draw call within the flattened vertex buffer.

    Attributes:
        base_index: Offset into the index array.
        count: Number of indices to draw.
        base_color_factor: RGBA multiplier from the PBR material.
        base_color_image: Optional RGBA texture (uint8 numpy array).
    """

    base_index: int = 0
    count: int = 0
    base_color_factor: np.ndarray = field(
        default_factory=lambda: np.ones(4, dtype=np.float32),
    )
    base_color_image: np.ndarray | None = None


@dataclass
class Mesh:
    """A loaded and flattened GLB mesh.

    Attributes:
        vertices: Float32 array ``(N, 12)`` of interleaved vertex
            data (position, normal, colour, UV).
        indices: Uint32 index array.
        draw_calls: Per-primitive draw call metadata.
        center: Geometric center that was subtracted from positions.
    """

    vertices: np.ndarray
    indices: np.ndarray
    draw_calls: list[DrawCall] = field(default_factory=list)
    center: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _access_data(
    gltf: GLTF2,
    accessor_index: int,
    blob: bytes,
) -> np.ndarray:
    """Extract typed data from a glTF accessor."""
    acc = gltf.accessors[accessor_index]
    bv = gltf.bufferViews[acc.bufferView]
    offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    count = acc.count

    _DTYPES = {
        5126: np.float32,  # FLOAT
        5123: np.uint16,   # UNSIGNED_SHORT
        5125: np.uint32,   # UNSIGNED_INT
    }
    dt = _DTYPES.get(acc.componentType)
    if dt is None:
        raise RuntimeError(
            f"Unsupported componentType {acc.componentType}",
        )

    ncomp = {
        "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16,
    }[acc.type]

    arr = np.frombuffer(blob, dtype=dt, count=count * ncomp, offset=offset)

    if acc.normalized:
        arr = arr.astype(np.float32)
        if acc.componentType in (5121, 5123):
            max_val = {5121: 255.0, 5123: 65535.0}[acc.componentType]
            arr /= max_val
        elif acc.componentType in (5120, 5122):
            max_val = {5120: 127.0, 5122: 32767.0}[acc.componentType]
            arr = np.clip(arr / max_val, -1.0, 1.0)

    return arr.reshape(count, ncomp)


def _extract_image_bytes(
    gltf: GLTF2,
    image,
    blob: bytes,
    path: Path,
) -> bytes | None:
    """Extract raw image bytes from a glTF image reference."""
    if image.uri:
        if image.uri.startswith("data:"):
            try:
                b64_data = image.uri.split(",", 1)[1]
                return base64.b64decode(b64_data)
            except Exception:
                return None
        img_path = (path.parent / image.uri).expanduser()
        if img_path.exists():
            return img_path.read_bytes()
        return None

    if image.bufferView is not None:
        bv = gltf.bufferViews[image.bufferView]
        offset = bv.byteOffset or 0
        length = bv.byteLength or 0
        return blob[offset:offset + length]
    return None


def _decode_image_to_rgba(data: bytes | None) -> np.ndarray | None:
    """Decode image bytes to an RGBA uint8 numpy array."""
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        return None
    return np.ascontiguousarray(img)


def _quat_to_mat4(q) -> np.ndarray:
    """Convert quaternion ``[x, y, z, w]`` to a 4x4 rotation matrix."""
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0.0],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0.0],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def _node_local_matrix(node) -> np.ndarray:
    """Compute the local transformation matrix for a glTF node."""
    if node.matrix:
        return np.array(node.matrix, dtype=np.float32).reshape(4, 4).T
    t = np.array(node.translation or [0, 0, 0], dtype=np.float32)
    s = np.array(node.scale or [1, 1, 1], dtype=np.float32)
    q = np.array(node.rotation or [0, 0, 0, 1], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[0:3, 3] = t
    R = _quat_to_mat4(q)
    S = np.diag(np.concatenate([s, [1.0]])).astype(np.float32)
    return T @ R @ S


def _compute_world_matrices(gltf: GLTF2) -> dict[int, np.ndarray]:
    """Recursively compute world-space matrices for all scene nodes."""
    world: dict[int, np.ndarray] = {}

    def _dfs(node_idx: int, parent_mat: np.ndarray) -> None:
        node = gltf.nodes[node_idx]
        local = _node_local_matrix(node)
        wm = parent_mat @ local
        world[node_idx] = wm
        for child in getattr(node, "children", []) or []:
            _dfs(child, wm)

    roots: list[int] = []
    if gltf.scene is not None and gltf.scenes:
        roots = list(gltf.scenes[gltf.scene].nodes or [])
    elif gltf.scenes:
        for sc in gltf.scenes:
            roots.extend(sc.nodes or [])
    else:
        roots = list(range(len(gltf.nodes)))

    identity = np.eye(4, dtype=np.float32)
    for n in roots:
        _dfs(n, identity)
    return world


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_glb(path: str | Path) -> Mesh:
    """Load a GLB file and flatten into a single vertex/index buffer.

    All mesh primitives are collected, world-transformed, and
    concatenated.  The geometry is recentered around its bounding-box
    midpoint.

    Args:
        path: Path to the ``.glb`` file.

    Returns:
        A :class:`Mesh` with interleaved vertices, indices, and
        draw-call metadata.

    Raises:
        RuntimeError: If the file has no meshes or primitives.
    """
    path = Path(path)
    gltf = GLTF2().load(str(path))
    blob = gltf.binary_blob()

    if not gltf.meshes:
        raise RuntimeError(f"GLB '{path}' has no meshes")

    world_mats = _compute_world_matrices(gltf)

    all_positions: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_uvs: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    draw_calls: list[DrawCall] = []
    idx_offset = 0

    for node_idx, node in enumerate(gltf.nodes):
        if node.mesh is None:
            continue
        mesh = gltf.meshes[node.mesh]
        if not mesh.primitives:
            continue

        world = world_mats.get(node_idx, np.eye(4, dtype=np.float32))
        normal_mat = np.linalg.inv(world[:3, :3]).T

        for prim in mesh.primitives:
            attrs = prim.attributes
            pos_acc = attrs.POSITION
            if pos_acc is None:
                continue

            positions = _access_data(gltf, pos_acc, blob).astype(np.float32)
            pos_h = np.concatenate(
                [positions, np.ones((positions.shape[0], 1), np.float32)],
                axis=1,
            )
            positions = (pos_h @ world.T)[:, :3]

            norm_acc = getattr(attrs, "NORMAL", None)
            if norm_acc is not None:
                normals = _access_data(gltf, norm_acc, blob).astype(np.float32)
            else:
                normals = np.zeros_like(positions)
            normals = normals @ normal_mat
            nlen = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
            normals = normals / nlen

            color_acc = getattr(attrs, "COLOR_0", None)
            colors = None
            if color_acc is not None:
                raw = _access_data(gltf, color_acc, blob)
                raw = raw.reshape(raw.shape[0], -1)
                if raw.shape[1] == 3:
                    raw = np.concatenate(
                        [raw, np.ones((raw.shape[0], 1), dtype=raw.dtype)],
                        axis=1,
                    )
                colors = raw.astype(np.float32)

            uv_acc = getattr(attrs, "TEXCOORD_0", None)
            texcoords = None
            if uv_acc is not None:
                texcoords = _access_data(gltf, uv_acc, blob).astype(np.float32)

            n_verts = positions.shape[0]
            if normals.shape[0] != n_verts:
                normals = np.zeros_like(positions)
            if colors is None or colors.shape[0] != n_verts:
                colors = np.ones((n_verts, 4), dtype=np.float32)
            if texcoords is None or texcoords.shape[0] != n_verts:
                texcoords = np.zeros((n_verts, 2), dtype=np.float32)

            all_positions.append(positions)
            all_normals.append(normals)
            all_colors.append(colors)
            all_uvs.append(texcoords)

            if prim.indices is not None:
                idx_local = _access_data(
                    gltf, prim.indices, blob,
                ).astype(np.uint32).ravel()
            else:
                idx_local = np.arange(n_verts, dtype=np.uint32)
            all_indices.append(idx_local + idx_offset)

            # Material
            base_color_factor = np.ones(4, dtype=np.float32)
            base_color_image = None
            if prim.material is not None and gltf.materials:
                mat = gltf.materials[prim.material]
                pbr = mat.pbrMetallicRoughness
                if pbr is not None and pbr.baseColorFactor:
                    base_color_factor = np.array(
                        pbr.baseColorFactor, dtype=np.float32,
                    )
                tex_info = pbr.baseColorTexture if pbr else None
                if (
                    tex_info
                    and tex_info.index is not None
                    and gltf.textures
                ):
                    tex = gltf.textures[tex_info.index]
                    if (
                        tex.source is not None
                        and 0 <= tex.source < len(gltf.images)
                    ):
                        img = gltf.images[tex.source]
                        data = _extract_image_bytes(gltf, img, blob, path)
                        base_color_image = _decode_image_to_rgba(data)

            draw_calls.append(DrawCall(
                base_index=0,  # filled below
                count=idx_local.size,
                base_color_factor=base_color_factor,
                base_color_image=base_color_image,
            ))
            idx_offset += n_verts

    if not all_positions:
        raise RuntimeError(
            f"GLB '{path}' contains no mesh primitives with POSITION",
        )

    positions = np.concatenate(all_positions, axis=0)
    normals = np.concatenate(all_normals, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    texcoords = np.concatenate(all_uvs, axis=0)
    indices = np.concatenate(all_indices, axis=0)

    # Recenter geometry.
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    positions -= center

    # Fill draw-call base indices.
    base = 0
    for dc in draw_calls:
        dc.base_index = base
        base += dc.count

    vertices = np.concatenate(
        [positions, normals, colors, texcoords], axis=1,
    ).astype(np.float32)

    logger.info(
        "Loaded GLB '%s': %d vertices, %d indices, %d draw calls",
        path.name, vertices.shape[0], indices.size, len(draw_calls),
    )
    return Mesh(
        vertices=vertices,
        indices=indices,
        draw_calls=draw_calls,
        center=center,
    )

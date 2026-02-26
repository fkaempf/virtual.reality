"""Top-down arena minimap renderer.

Draws a 2-D bird's-eye view showing fly position, heading, camera
FOV cone, and movement trail.  Used as a debugging overlay during
stimulus presentation.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def world_to_minimap(
    x_mm: float,
    y_mm: float,
    center_u: int,
    center_v: int,
    scale: float,
) -> tuple[int, int]:
    """Convert world coordinates (mm) to minimap pixel coordinates.

    Args:
        x_mm: World X position in mm.
        y_mm: World Y position in mm.
        center_u: Minimap center pixel column.
        center_v: Minimap center pixel row.
        scale: Pixels per mm.

    Returns:
        ``(u, v)`` pixel coordinates.
    """
    u = int(round(center_u + x_mm * scale))
    v = int(round(center_v - y_mm * scale))
    return u, v


def draw_arrow(
    img: np.ndarray,
    center_u: int,
    center_v: int,
    heading_rad: float,
    size_px: int = 18,
    color: tuple[int, int, int] = (0, 120, 255),
) -> None:
    """Draw a filled triangular arrow on *img*.

    Args:
        img: BGR image (modified in place).
        center_u: Arrow center column.
        center_v: Arrow center row.
        heading_rad: Arrow heading in radians (0 = up).
        size_px: Arrow size in pixels.
        color: BGR colour tuple.
    """
    tip = np.array([0, -size_px], dtype=np.float32)
    left = np.array([-size_px * 0.6, size_px * 0.7], dtype=np.float32)
    right = np.array([size_px * 0.6, size_px * 0.7], dtype=np.float32)
    R = np.array([
        [math.cos(heading_rad), -math.sin(heading_rad)],
        [math.sin(heading_rad), math.cos(heading_rad)],
    ], dtype=np.float32)
    pts = np.stack([tip, left, right], axis=0) @ R.T
    pts[:, 0] += center_u
    pts[:, 1] += center_v
    pts_i = pts.astype(np.int32)
    cv2.fillConvexPoly(img, pts_i, color)
    cv2.polylines(img, [pts_i], True, (0, 0, 0), 1, cv2.LINE_AA)


def build_minimap_base(
    arena_radius_mm: float,
    map_w: int,
    map_h: int,
    center_u: int,
    center_v: int,
    scale: float,
) -> np.ndarray:
    """Create the static minimap background with the arena circle.

    Args:
        arena_radius_mm: Arena radius in mm.
        map_w: Minimap image width.
        map_h: Minimap image height.
        center_u: Arena center column in the minimap.
        center_v: Arena center row in the minimap.
        scale: Pixels per mm.

    Returns:
        BGR uint8 image.
    """
    img = np.full((map_h, map_w, 3), 255, np.uint8)
    radius_px = int(round(arena_radius_mm * scale))
    cv2.circle(img, (center_u, center_v), radius_px, (0, 0, 0), 2)
    return img


def draw_minimap_dynamic(
    base_img: np.ndarray,
    fly_x: float,
    fly_y: float,
    fly_heading: float,
    trail_pts_uv: list[tuple[int, int]],
    trail_color: tuple[int, int, int],
    trail_thick: int,
    center_u: int,
    center_v: int,
    scale: float,
    fps: float,
    cam_x: float,
    cam_y: float,
    cam_heading: float,
    camera_fov_x_deg: float,
    arena_radius_mm: float,
) -> np.ndarray:
    """Render dynamic minimap elements on top of a base image.

    Draws the movement trail, camera FOV cone, fly and camera
    arrows, and text overlays.

    Args:
        base_img: Static background from :func:`build_minimap_base`.
        fly_x: Fly world X in mm.
        fly_y: Fly world Y in mm.
        fly_heading: Fly heading in radians.
        trail_pts_uv: Trail history as minimap pixel coords.
        trail_color: BGR colour for the trail.
        trail_thick: Trail line thickness.
        center_u: Minimap center column.
        center_v: Minimap center row.
        scale: Pixels per mm.
        fps: Current frames per second.
        cam_x: Camera world X in mm.
        cam_y: Camera world Y in mm.
        cam_heading: Camera heading in radians.
        camera_fov_x_deg: Camera horizontal FOV in degrees.
        arena_radius_mm: Arena radius in mm.

    Returns:
        BGR uint8 image with dynamic overlays.
    """
    img = base_img.copy()

    # Trail
    if len(trail_pts_uv) >= 2:
        now_color = np.array(trail_color, dtype=np.float32)
        nseg = len(trail_pts_uv) - 1
        for i in range(nseg):
            p0 = trail_pts_uv[i]
            p1 = trail_pts_uv[i + 1]
            alpha = (i + 1) / nseg
            col = tuple(
                (now_color * (0.3 + 0.7 * alpha)).astype(np.int32).tolist(),
            )
            cv2.line(img, p0, p1, col, trail_thick, cv2.LINE_AA)

    # Camera FOV cone
    cam_u, cam_v = world_to_minimap(cam_x, cam_y, center_u, center_v, scale)
    fov_rad = math.radians(camera_fov_x_deg)
    half_fov = 0.5 * fov_rad
    cone_range_mm = arena_radius_mm
    n_samples = 24

    cone_pts: list[tuple[int, int]] = [(cam_u, cam_v)]
    for i in range(n_samples + 1):
        a = cam_heading - half_fov + 2.0 * half_fov * (i / n_samples)
        wx = cam_x + cone_range_mm * math.sin(a)
        wy = cam_y + cone_range_mm * math.cos(a)
        u, v = world_to_minimap(wx, wy, center_u, center_v, scale)
        cone_pts.append((u, v))

    cone_np = np.array(cone_pts, dtype=np.int32)
    overlay = img.copy()
    cv2.fillConvexPoly(overlay, cone_np, (230, 230, 255))
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    cv2.polylines(img, [cone_np], True, (150, 150, 255), 1, cv2.LINE_AA)

    # Fly arrow
    fu, fv = world_to_minimap(fly_x, fly_y, center_u, center_v, scale)
    draw_arrow(img, fu, fv, fly_heading, size_px=18, color=(0, 120, 255))

    # Camera arrow
    draw_arrow(img, cam_u, cam_v, cam_heading, size_px=16, color=(0, 0, 255))
    cv2.putText(
        img, "cam", (cam_u + 10, cam_v - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
    )

    # Text overlays
    cv2.putText(
        img, f"FPS: {fps:.1f}",
        (10, img.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
    )
    dist_mm = math.hypot(fly_x - cam_x, fly_y - cam_y)
    cv2.putText(
        img, f"cam-fly: {dist_mm:.1f} mm",
        (10, img.shape[0] - 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
    )
    return img

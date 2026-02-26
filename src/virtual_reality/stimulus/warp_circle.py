"""Warp-circle calibration test stimulus.

Displays a moving circle through the projector warp map to verify
calibration quality.  The circle oscillates horizontally in camera
space and is warped to projector space via a fullscreen quad shader.

Consolidates the various ``warp_circle*.py`` variants from the
legacy codebase.
"""

from __future__ import annotations

import ctypes
import logging
import math
import time

import numpy as np

from virtual_reality.config.schema import VirtualRealityConfig
from virtual_reality.rendering.shaders import WARP_FRAG_SRC, WARP_VERT_SRC
from virtual_reality.stimulus.base import Stimulus

logger = logging.getLogger(__name__)

# Simple fragment shader that draws a white circle on black background
_CIRCLE_FRAG_SRC: str = r"""
#version 330 core

out vec4 fragColor;
uniform vec2 u_center;    // circle center in pixels
uniform float u_radius;   // circle radius in pixels
uniform vec2 u_resolution; // render target size

void main() {
    vec2 frag = gl_FragCoord.xy;
    float dist = length(frag - u_center);
    if (dist <= u_radius) {
        fragColor = vec4(1.0, 1.0, 1.0, 1.0);
    } else {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
"""

_CIRCLE_VERT_SRC: str = r"""
#version 330 core

in vec2 in_pos;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


class WarpCircleStimulus(Stimulus):
    """Moving circle stimulus for warp verification.

    A white circle oscillates left-right in camera space.  The
    rendered image is warped through the projector-camera map, so
    the circle should appear undistorted when viewed through the
    camera.

    Args:
        config: Full configuration dataclass.
        freq_hz: Oscillation frequency in Hz.
    """

    def __init__(
        self,
        config: VirtualRealityConfig | None = None,
        freq_hz: float = 0.25,
    ) -> None:
        if config is None:
            config = VirtualRealityConfig()
        self.config = config
        self._freq_hz = freq_hz
        self._paused = False
        self._use_warp = True

    def setup(self) -> None:
        """Create OpenGL context, load warp maps, compile shaders."""
        import pygame
        from OpenGL import GL

        from virtual_reality.display.monitor import pick_monitor
        from virtual_reality.display.window import setup_pygame_window
        from virtual_reality.mapping.warp import (
            load_warp_map,
            warp_to_gl_texture,
        )
        from virtual_reality.rendering.gl_utils import (
            create_offscreen_fbo,
            create_program,
            create_texture_2d,
        )

        cfg = self.config
        logger.info("WarpCircleStimulus.setup()")

        # Load warp map
        warp = load_warp_map(cfg.warp.mapx_path, cfg.warp.mapy_path)
        warp_uv = warp_to_gl_texture(warp)
        self._cam_w = warp.cam_w
        self._cam_h = warp.cam_h
        self._proj_w = warp.proj_w
        self._proj_h = warp.proj_h

        # Circle parameters
        self._radius = max(4, min(self._cam_w, self._cam_h) // 10)
        self._y_center = int(self._cam_h * 0.65)
        x_min = self._radius / 2.0
        x_max = self._cam_w - self._radius / 2.0
        self._amplitude = (x_max - x_min) / 2.0
        self._x_offset = x_min + self._amplitude

        # Window
        mon = pick_monitor(
            self._proj_w, self._proj_h, which=cfg.display.monitor,
        )
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_MAJOR_VERSION, 3,
        )
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_MINOR_VERSION, 3,
        )
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK,
            pygame.GL_CONTEXT_PROFILE_CORE,
        )
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, 1,
        )
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

        setup_pygame_window(
            self._proj_w, self._proj_h,
            monitor_x=mon.x, monitor_y=mon.y,
            borderless=cfg.display.borderless,
            opengl=True,
        )
        pygame.display.set_caption("Virtual Reality - Warp Circle Test")

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glDisable(GL.GL_DEPTH_TEST)

        # -- Circle shader (renders to offscreen FBO) --
        self._circle_prog = create_program(
            _CIRCLE_VERT_SRC, _CIRCLE_FRAG_SRC,
        )
        self._u_circle_center = GL.glGetUniformLocation(
            self._circle_prog, "u_center",
        )
        self._u_circle_radius = GL.glGetUniformLocation(
            self._circle_prog, "u_radius",
        )
        self._u_circle_res = GL.glGetUniformLocation(
            self._circle_prog, "u_resolution",
        )

        # Fullscreen quad VAO for circle pass
        quad_verts = np.array([
            -1.0, -1.0,  1.0, -1.0,  -1.0, 1.0,  1.0, 1.0,
        ], dtype=np.float32)
        self._circle_vao = GL.glGenVertexArrays(1)
        self._circle_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._circle_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._circle_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, quad_verts.nbytes,
            quad_verts, GL.GL_STATIC_DRAW,
        )
        loc = GL.glGetAttribLocation(self._circle_prog, "in_pos")
        GL.glEnableVertexAttribArray(loc)
        GL.glVertexAttribPointer(
            loc, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0),
        )
        GL.glBindVertexArray(0)

        # -- Offscreen FBO --
        self._fbo, self._cam_tex, self._depth_rb = (
            create_offscreen_fbo(self._cam_w, self._cam_h)
        )

        # -- Warp quad program --
        self._warp_prog = create_program(WARP_VERT_SRC, WARP_FRAG_SRC)
        GL.glUseProgram(self._warp_prog)

        warp_quad = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype=np.float32)

        self._warp_vao = GL.glGenVertexArrays(1)
        self._warp_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._warp_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._warp_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, warp_quad.nbytes,
            warp_quad, GL.GL_STATIC_DRAW,
        )
        stride = 4 * warp_quad.itemsize
        loc_pos = GL.glGetAttribLocation(self._warp_prog, "in_pos")
        GL.glEnableVertexAttribArray(loc_pos)
        GL.glVertexAttribPointer(
            loc_pos, 2, GL.GL_FLOAT, GL.GL_FALSE,
            stride, ctypes.c_void_p(0),
        )
        loc_uv = GL.glGetAttribLocation(self._warp_prog, "in_uv")
        GL.glEnableVertexAttribArray(loc_uv)
        GL.glVertexAttribPointer(
            loc_uv, 2, GL.GL_FLOAT, GL.GL_FALSE,
            stride, ctypes.c_void_p(2 * warp_quad.itemsize),
        )
        GL.glBindVertexArray(0)

        # Warp texture
        self._warp_tex = create_texture_2d(
            self._proj_w, self._proj_h,
            GL.GL_RG32F, GL.GL_RG, GL.GL_FLOAT,
        )
        self._u_cam = GL.glGetUniformLocation(self._warp_prog, "u_cam")
        self._u_warp = GL.glGetUniformLocation(
            self._warp_prog, "u_warp",
        )
        self._u_useWarp = GL.glGetUniformLocation(
            self._warp_prog, "u_useWarp",
        )
        GL.glUniform1i(self._u_cam, 0)
        GL.glUniform1i(self._u_warp, 1)

        # Upload warp data
        warp_c = np.ascontiguousarray(warp_uv.astype(np.float32))
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._warp_tex)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0,
            self._proj_w, self._proj_h,
            GL.GL_RG, GL.GL_FLOAT,
            warp_c.ctypes.data_as(ctypes.c_void_p),
        )

        self._start_time = time.perf_counter()
        logger.info(
            "WarpCircle ready: cam=%dx%d proj=%dx%d radius=%d freq=%.2fHz",
            self._cam_w, self._cam_h,
            self._proj_w, self._proj_h,
            self._radius, self._freq_hz,
        )

    def update(self, dt: float, events: list) -> None:
        """Handle input events."""
        import pygame

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_SPACE, pygame.K_p):
                    self._paused = not self._paused
                elif event.key == pygame.K_u:
                    self._use_warp = not self._use_warp
                    logger.info("Warp: %s", self._use_warp)

    def render(self) -> None:
        """Draw circle to FBO, then warp to screen."""
        from OpenGL import GL

        # Compute circle position
        if self._paused:
            t = 0.0
        else:
            t = time.perf_counter() - self._start_time
        x_center = self._x_offset + self._amplitude * math.sin(
            2.0 * math.pi * self._freq_hz * t,
        )

        # Pass 1: Draw circle to offscreen FBO
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glViewport(0, 0, self._cam_w, self._cam_h)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(self._circle_prog)
        GL.glUniform2f(
            self._u_circle_center,
            float(x_center), float(self._y_center),
        )
        GL.glUniform1f(self._u_circle_radius, float(self._radius))
        GL.glUniform2f(
            self._u_circle_res,
            float(self._cam_w), float(self._cam_h),
        )
        GL.glBindVertexArray(self._circle_vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # Pass 2: Warp to screen
        GL.glViewport(0, 0, self._proj_w, self._proj_h)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(self._warp_prog)
        GL.glUniform1i(
            self._u_useWarp, 1 if self._use_warp else 0,
        )
        GL.glBindVertexArray(self._warp_vao)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._cam_tex)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._warp_tex)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)

    def teardown(self) -> None:
        """Release GPU resources."""
        from OpenGL import GL

        logger.info("WarpCircleStimulus.teardown()")

        GL.glDeleteTextures([self._warp_tex, self._cam_tex])
        GL.glDeleteRenderbuffers(1, [self._depth_rb])
        GL.glDeleteFramebuffers(1, [self._fbo])
        GL.glDeleteBuffers(
            2, [self._circle_vbo, self._warp_vbo],
        )
        GL.glDeleteVertexArrays(2, [self._circle_vao, self._warp_vao])
        GL.glDeleteProgram(self._circle_prog)
        GL.glDeleteProgram(self._warp_prog)

        import pygame
        pygame.quit()


def main() -> None:
    """CLI entry point for the warp circle test."""
    from virtual_reality.config.loader import load_config
    from virtual_reality.config.schema import _resolve_default_paths

    import sys

    if len(sys.argv) > 1:
        config = load_config(sys.argv[1])
    else:
        config = _resolve_default_paths()

    stimulus = WarpCircleStimulus(config=config)
    stimulus.run()

"""Low-level OpenGL helper functions.

Wraps common PyOpenGL boilerplate for shader compilation, program
linking, texture creation, and offscreen FBO setup.  All functions
require an active OpenGL context.
"""

from __future__ import annotations

import logging

import numpy as np
from OpenGL import GL

logger = logging.getLogger(__name__)


def compile_shader(source: str, shader_type: int) -> int:
    """Compile a single GLSL shader.

    Args:
        source: GLSL source code string.
        shader_type: ``GL_VERTEX_SHADER`` or ``GL_FRAGMENT_SHADER``.

    Returns:
        OpenGL shader handle.

    Raises:
        RuntimeError: If compilation fails.
    """
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if not status:
        log = GL.glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile error:\n{log}")
    return shader


def create_program(vs_src: str, fs_src: str) -> int:
    """Create a GPU program from vertex and fragment shader sources.

    Args:
        vs_src: Vertex shader GLSL source.
        fs_src: Fragment shader GLSL source.

    Returns:
        OpenGL program handle.

    Raises:
        RuntimeError: If linking fails.
    """
    vs = compile_shader(vs_src, GL.GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL.GL_FRAGMENT_SHADER)
    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, vs)
    GL.glAttachShader(prog, fs)
    GL.glLinkProgram(prog)
    status = GL.glGetProgramiv(prog, GL.GL_LINK_STATUS)
    if not status:
        log = GL.glGetProgramInfoLog(prog).decode()
        raise RuntimeError(f"Program link error:\n{log}")
    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    return prog


def create_texture_2d(
    width: int,
    height: int,
    internal_format: int,
    fmt: int,
    dtype: int,
    min_filter: int = GL.GL_LINEAR,
    mag_filter: int = GL.GL_LINEAR,
) -> int:
    """Create an empty 2-D texture.

    Args:
        width: Texture width.
        height: Texture height.
        internal_format: GPU internal format (e.g. ``GL_RGBA8``).
        fmt: Pixel format (e.g. ``GL_RGBA``).
        dtype: Data type (e.g. ``GL_UNSIGNED_BYTE``).
        min_filter: Minification filter.
        mag_filter: Magnification filter.

    Returns:
        OpenGL texture handle.
    """
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, min_filter,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, mag_filter,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE,
    )
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, internal_format,
        width, height, 0, fmt, dtype, None,
    )
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tex


def create_texture_from_image(img_rgba: np.ndarray) -> int:
    """Upload an RGBA image to the GPU as a mipmapped texture.

    Args:
        img_rgba: uint8 array of shape ``(height, width, 4)``.

    Returns:
        OpenGL texture handle.
    """
    h, w, c = img_rgba.shape
    assert c == 4, f"Expected 4 channels, got {c}"
    internal_fmt = getattr(GL, "GL_SRGB8_ALPHA8", GL.GL_RGBA8)

    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
        GL.GL_LINEAR_MIPMAP_LINEAR,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT,
    )
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, internal_fmt, w, h, 0,
        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_rgba,
    )
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tex


def create_offscreen_fbo(
    width: int,
    height: int,
) -> tuple[int, int, int]:
    """Create an offscreen framebuffer with color and depth attachments.

    Args:
        width: FBO width.
        height: FBO height.

    Returns:
        ``(fbo, color_tex, depth_rb)`` handles.

    Raises:
        RuntimeError: If the framebuffer is incomplete.
    """
    color_tex = create_texture_2d(
        width, height,
        GL.GL_RGBA8, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
    )

    depth_rb = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_rb)
    GL.glRenderbufferStorage(
        GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, width, height,
    )
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)

    fbo = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
        GL.GL_TEXTURE_2D, color_tex, 0,
    )
    GL.glFramebufferRenderbuffer(
        GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
        GL.GL_RENDERBUFFER, depth_rb,
    )

    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if status != GL.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(
            f"Framebuffer incomplete: status=0x{status:04X}",
        )
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    logger.info("Created %dx%d offscreen FBO", width, height)
    return fbo, color_tex, depth_rb

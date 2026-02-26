"""GLSL shader source strings for the rendering pipeline.

Contains vertex and fragment shaders for:

- **Warp quad**: Full-screen pass that remaps the offscreen camera
  render through the projector-camera warp map.
- **Fly 3D**: Phong-lit 3D model rendering with perspective,
  equidistant fisheye, and equirectangular projection modes.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Warp quad (full-screen post-process pass)
# ---------------------------------------------------------------------------

WARP_VERT_SRC: str = r"""
#version 330 core

in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

WARP_FRAG_SRC: str = r"""
#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_cam;   // offscreen camera image (3D fly)
uniform sampler2D u_warp;  // warp map (RG = camera UV)
uniform int u_useWarp;     // 1 = use warp, 0 = bypass

void main() {
    vec2 cam_uv;

    if (u_useWarp == 0) {
        cam_uv = v_uv;
    } else {
        cam_uv = texture(u_warp, v_uv).rg;
        cam_uv.y = 1.0 - cam_uv.y;
    }

    if (cam_uv.x < 0.0 || cam_uv.y < 0.0 || cam_uv.x > 1.0 || cam_uv.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 col = texture(u_cam, cam_uv).rgb;
    fragColor = vec4(col, 1.0);
}
"""

# ---------------------------------------------------------------------------
# 3D fly model shaders
# ---------------------------------------------------------------------------

FLY_VERT_SRC: str = r"""
#version 330 core

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec2 in_uv;

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat4 u_view;
uniform float u_far;
uniform float u_fovY;
uniform float u_fovX;
uniform int u_projMode; // 0 = perspective, 1 = equidistant fisheye, 2 = equirectangular

out vec3 v_normal;
out vec4 v_color;
out vec2 v_uv;

void main() {
    vec4 world_pos = u_model * vec4(in_pos, 1.0);
    vec4 view_pos  = u_view * world_pos;
    v_normal = mat3(u_model) * in_normal;
    v_color = in_color;
    v_uv = in_uv;

    if (u_projMode == 1) {
        // Equidistant fisheye (theta-proportional)
        vec3 dir = normalize(-view_pos.xyz);
        if (dir.z <= 0.0) {
            gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
            return;
        }
        float theta = acos(clamp(dir.z, -1.0, 1.0));
        float max_theta = 0.5 * max(u_fovX, u_fovY);
        if (theta > max_theta) {
            gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
            return;
        }
        float r = theta / max_theta;

        float len_xy = length(dir.xy);
        vec2 dir_xy_norm = (len_xy > 1e-6) ? dir.xy / len_xy : vec2(0.0, 0.0);
        vec2 proj = r * vec2(dir_xy_norm.x, -dir_xy_norm.y);

        float depth = -view_pos.z / u_far;
        gl_Position = vec4(proj.x, proj.y, depth, 1.0);
    } else if (u_projMode == 2) {
        // Equirectangular
        vec3 dir = normalize(-view_pos.xyz);
        if (dir.z <= 0.0) {
            gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
            return;
        }
        float az = atan(dir.x, dir.z);
        float el = asin(clamp(dir.y, -1.0, 1.0));

        float half_fov_x = max(u_fovX * 0.5, 1e-6);
        float half_fov_y = max(u_fovY * 0.5, 1e-6);
        if (abs(az) > half_fov_x || abs(el) > half_fov_y) {
            gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
            return;
        }
        vec2 ndc;
        ndc.x = az / half_fov_x;
        ndc.y = -el / half_fov_y;

        float depth = -view_pos.z / u_far;
        gl_Position = vec4(ndc.x, ndc.y, depth, 1.0);
    } else {
        // Standard perspective
        gl_Position = u_mvp * vec4(in_pos, 1.0);
    }
}
"""

FLY_FRAG_SRC: str = r"""
#version 330 core

in vec3 v_normal;
in vec4 v_color;
in vec2 v_uv;
out vec4 fragColor;

uniform vec4 u_baseColor;
uniform int  u_hasTexture;
uniform sampler2D u_baseColorTex;
uniform float u_ambient;
uniform vec4 u_lightIntensities;
uniform vec3 u_lightDirs[4];
uniform float u_lightMaxGain;

void main() {
    vec3 N = normalize(v_normal);

    vec3 base = (v_color.rgb * u_baseColor.rgb);
    if (u_hasTexture == 1) {
        base *= texture(u_baseColorTex, v_uv).rgb;
    }

    float gain = u_ambient;
    for (int i = 0; i < 4; ++i) {
        gain += u_lightIntensities[i] * max(dot(N, u_lightDirs[i]), 0.0);
    }
    gain = min(gain, u_lightMaxGain);

    vec3 col = base * gain;
    fragColor = vec4(col, 1.0);
}
"""

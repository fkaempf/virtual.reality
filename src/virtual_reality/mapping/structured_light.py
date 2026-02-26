"""Structured-light pattern generation and decoding.

Implements Gray code and phase-shifted sinusoidal fringe patterns
for projector-camera calibration.  The pure-math functions
(generation and decoding) are separated from hardware I/O so they
can be tested without a projector or camera.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gray code patterns
# ---------------------------------------------------------------------------

def generate_graycode_patterns(
    width: int,
    height: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Generate binary Gray code patterns for structured-light capture.

    Produces positive and negative (inverted) pattern pairs for each
    bit plane along both X and Y axes.

    Args:
        width: Projector width in pixels.
        height: Projector height in pixels.

    Returns:
        ``(patterns, black, white)`` where *patterns* is a list of
        uint8 images and *black*/*white* are uniform reference frames.
    """
    nx = int(math.ceil(math.log2(width)))
    ny = int(math.ceil(math.log2(height)))
    xs = np.arange(width, dtype=np.uint32)
    ys = np.arange(height, dtype=np.uint32)
    gx = xs ^ (xs >> 1)
    gy = ys ^ (ys >> 1)

    pats: list[np.ndarray] = []

    # X-axis bit planes (MSB first)
    for k in range(nx - 1, -1, -1):
        col = ((gx >> k) & 1).astype(np.uint8) * 255
        img = np.repeat(col[np.newaxis, :], height, axis=0)
        pats.append(img)
        pats.append(255 - img)

    # Y-axis bit planes (MSB first)
    for k in range(ny - 1, -1, -1):
        row = ((gy >> k) & 1).astype(np.uint8) * 255
        img = np.repeat(row[:, np.newaxis], width, axis=1)
        pats.append(img)
        pats.append(255 - img)

    black = np.zeros((height, width), np.uint8)
    white = np.full((height, width), 255, np.uint8)
    return pats, black, white


def gray_to_binary(bits: np.ndarray) -> np.ndarray:
    """Convert a Gray-code bit array to standard binary.

    Args:
        bits: Integer array where the last axis holds the bit planes
            (MSB first).

    Returns:
        Binary-coded array of the same shape.
    """
    out = bits.copy()
    for i in range(1, out.shape[-1]):
        out[..., i] ^= out[..., i - 1]
    return out


def decode_gray(
    captured: list[np.ndarray],
    black_cap: np.ndarray,
    white_cap: np.ndarray,
    proj_w: int,
    proj_h: int,
    contrast_threshold: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode captured Gray code images to projector coordinates.

    Args:
        captured: List of captured camera images (positive/negative
            pairs interleaved), one per projected pattern.
        black_cap: Camera image captured with all-black projection.
        white_cap: Camera image captured with all-white projection.
        proj_w: Projector width.
        proj_h: Projector height.
        contrast_threshold: Minimum white-minus-black difference for
            a pixel to be considered valid.

    Returns:
        ``(proj_x, proj_y, valid)`` integer arrays in camera
        resolution.  Invalid pixels are set to ``-1``.
    """
    C = np.stack(captured, 0).astype(np.int16, copy=False)
    B = black_cap.astype(np.int16, copy=False)
    W = white_cap.astype(np.int16, copy=False)

    nx = int(math.ceil(math.log2(proj_w)))
    ny = int(math.ceil(math.log2(proj_h)))

    x_pos = C[0:nx * 2:2]
    x_neg = C[1:nx * 2:2]
    y_pos = C[nx * 2:nx * 2 + ny * 2:2]
    y_neg = C[nx * 2 + 1:nx * 2 + ny * 2:2]

    gx_bits = (x_pos > x_neg).astype(np.uint8)
    gy_bits = (y_pos > y_neg).astype(np.uint8)

    bx_bits = gray_to_binary(np.moveaxis(gx_bits, 0, -1))
    by_bits = gray_to_binary(np.moveaxis(gy_bits, 0, -1))

    def _bits_to_int(bits: np.ndarray) -> np.ndarray:
        v = np.zeros(bits.shape[:2], np.int32)
        for i in range(bits.shape[-1]):
            v = (v << 1) | bits[..., i].astype(np.int32)
        return v

    proj_x = _bits_to_int(bx_bits)
    proj_y = _bits_to_int(by_bits)

    valid = (W - B) > contrast_threshold
    proj_x[(proj_x < 0) | (proj_x >= proj_w) | (~valid)] = -1
    proj_y[(proj_y < 0) | (proj_y >= proj_h) | (~valid)] = -1
    return proj_x, proj_y, valid.astype(np.uint8)


# ---------------------------------------------------------------------------
# Sinusoidal fringe patterns
# ---------------------------------------------------------------------------

def generate_sine_patterns(
    width: int,
    height: int,
    periods: float,
    n_phases: int = 4,
    axis: str = "x",
    gamma: float | None = None,
) -> list[np.ndarray]:
    """Generate phase-shifted cosine fringe patterns.

    Args:
        width: Pattern width in pixels.
        height: Pattern height in pixels.
        periods: Number of fringe periods across the axis.
        n_phases: Number of phase steps (typically 4).
        axis: ``"x"`` or ``"y"`` for fringe direction.
        gamma: If not ``None``, apply inverse-gamma LUT to
            linearise projector response.

    Returns:
        List of *n_phases* uint8 pattern images.
    """
    if axis == "x":
        u = np.linspace(
            0, 2 * np.pi * periods, width, endpoint=False,
        )[np.newaxis, :]
        u = np.repeat(u, height, axis=0)
    else:
        u = np.linspace(
            0, 2 * np.pi * periods, height, endpoint=False,
        )[:, np.newaxis]
        u = np.repeat(u, width, axis=1)

    pats: list[np.ndarray] = []
    for k in range(n_phases):
        phi = 2 * np.pi * (k / n_phases)
        img = 0.5 + 0.5 * np.cos(u + phi)
        u8 = (img * 255.0 + 0.5).astype(np.uint8)
        if gamma is not None:
            import cv2
            x = np.arange(256, dtype=np.float32) / 255.0
            lut = np.clip(
                (x ** (1.0 / gamma)) * 255.0 + 0.5, 0, 255,
            ).astype(np.uint8)
            u8 = cv2.LUT(u8, lut)
        pats.append(u8)

    return pats


def decode_phase(
    frames: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """General N-step phase-shifting decoder.

    Uses first-harmonic Fourier coefficient extraction to recover
    the wrapped phase and modulation from *N* equally-spaced
    phase-shifted captures.

    Args:
        frames: List of *N* captured images (uint8 or float32).

    Returns:
        ``(phase, modulation)`` where *phase* is in ``[-pi, pi)``
        and *modulation* is in ``[0, 1]``.

    Raises:
        ValueError: If fewer than 3 frames are provided.
    """
    I = np.stack(
        [f.astype(np.float32) for f in frames], axis=-1,
    )
    N = I.shape[-1]
    if N < 3:
        raise ValueError("Need at least 3 phase-shifted images")

    phi = np.linspace(
        0, 2 * np.pi, N, endpoint=False,
    ).astype(np.float32)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    a_real = np.tensordot(I, cos_phi, axes=([-1], [0]))
    a_imag = np.tensordot(I, sin_phi, axes=([-1], [0]))

    phase = np.arctan2(-a_imag, a_real)
    A = np.sqrt(a_real ** 2 + a_imag ** 2) * (2.0 / N)
    I_avg = np.mean(I, axis=-1)
    modulation = np.clip(A / (I_avg + 1e-6), 0, 1)

    return phase, modulation


def unwrap_with_gray(
    phase_wrapped: np.ndarray,
    coarse_int: np.ndarray,
) -> np.ndarray:
    """Combine Gray code integer index with sine phase for subpixel precision.

    Args:
        phase_wrapped: Wrapped phase in ``[-pi, pi)``.
        coarse_int: Integer projector coordinate from Gray code.

    Returns:
        Float32 projector coordinates with subpixel accuracy.
    """
    frac = (phase_wrapped + np.pi) / (2 * np.pi)
    return coarse_int.astype(np.float32) + (frac - 0.5)

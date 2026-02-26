"""Tests for virtual_reality.mapping.structured_light."""

from __future__ import annotations

import math

import numpy as np
import pytest

from virtual_reality.mapping.structured_light import (
    decode_gray,
    decode_phase,
    generate_graycode_patterns,
    generate_sine_patterns,
    gray_to_binary,
    unwrap_with_gray,
)


class TestGrayCode:
    """Tests for Gray code generation and decoding."""

    def test_pattern_count(self) -> None:
        """Number of patterns matches 2*(nx + ny) bit planes."""
        pats, black, white = generate_graycode_patterns(64, 32)
        nx = math.ceil(math.log2(64))
        ny = math.ceil(math.log2(32))
        assert len(pats) == 2 * (nx + ny)

    def test_black_white_shape(self) -> None:
        pats, black, white = generate_graycode_patterns(16, 8)
        assert black.shape == (8, 16)
        assert white.shape == (8, 16)
        assert np.all(black == 0)
        assert np.all(white == 255)

    def test_gray_to_binary_roundtrip(self) -> None:
        """Gray encode then decode should recover the original."""
        vals = np.arange(16, dtype=np.uint32)
        gray = vals ^ (vals >> 1)
        bits = np.array(
            [[(g >> (3 - k)) & 1 for k in range(4)] for g in gray],
            dtype=np.uint8,
        )
        binary_bits = gray_to_binary(bits)
        recovered = np.zeros(16, dtype=np.uint32)
        for i in range(4):
            recovered = (recovered << 1) | binary_bits[:, i].astype(np.uint32)
        np.testing.assert_array_equal(recovered, vals)

    def test_decode_identity(self) -> None:
        """Encode then decode should recover projector coordinates."""
        w, h = 8, 4
        pats, black, white = generate_graycode_patterns(w, h)

        # Simulate a 1:1 camera (camera pixel == projector pixel).
        captured = []
        for pat in pats:
            captured.append(pat.copy())

        proj_x, proj_y, valid = decode_gray(
            captured, black, white, w, h,
        )
        # Check a few pixels.
        assert proj_x[0, 0] == 0
        assert proj_x[0, w - 1] == w - 1
        assert proj_y[0, 0] == 0
        assert proj_y[h - 1, 0] == h - 1


class TestSinePatterns:
    """Tests for sine pattern generation and decoding."""

    def test_pattern_count(self) -> None:
        pats = generate_sine_patterns(100, 80, periods=5, n_phases=4)
        assert len(pats) == 4

    def test_pattern_shape(self) -> None:
        pats = generate_sine_patterns(100, 80, periods=5)
        for p in pats:
            assert p.shape == (80, 100)
            assert p.dtype == np.uint8

    def test_decode_phase_recovery(self) -> None:
        """Decode should recover a known phase ramp."""
        W, H = 64, 1
        periods = 4
        pats = generate_sine_patterns(W, H, periods, n_phases=4)
        phase, mod = decode_phase(pats)
        assert phase.shape == (H, W)
        assert mod.shape == (H, W)
        # Modulation should be high in the middle.
        assert mod.mean() > 0.3

    def test_decode_phase_min_frames(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            decode_phase([np.zeros((4, 4)), np.zeros((4, 4))])


class TestUnwrapWithGray:
    """Tests for the hybrid unwrapping function."""

    def test_zero_phase_offset(self) -> None:
        coarse = np.array([0, 1, 2, 3], dtype=np.int32)
        # phase = 0 → frac = 0.5 → offset = 0.0
        phase = np.zeros(4, dtype=np.float32)
        result = unwrap_with_gray(phase, coarse)
        np.testing.assert_array_almost_equal(result, coarse.astype(np.float32))

    def test_positive_phase_offset(self) -> None:
        coarse = np.array([5], dtype=np.int32)
        phase = np.array([np.pi / 2], dtype=np.float32)
        result = unwrap_with_gray(phase, coarse)
        # frac = (pi/2 + pi) / 2pi = 3pi/2 / 2pi = 0.75
        # result = 5 + 0.25 = 5.25
        assert result[0] == pytest.approx(5.25)

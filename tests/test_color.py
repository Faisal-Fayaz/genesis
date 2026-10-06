"""Color conversion and palette sampling."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from genesis.color import (
    Palette,
    hex_to_oklab,
    hex_to_rgb,
    oklab_to_hex,
    oklab_to_srgb,
    rgb_to_hex,
    srgb_to_oklab,
)


@pytest.mark.parametrize(
    "value", ["#000000", "#ffffff", "#1a2b3c", "#ff0066", "#00ccff", "#4a4a4a"]
)
def test_hex_round_trips(value: str) -> None:
    assert oklab_to_hex(hex_to_oklab(value)) == value


def test_srgb_short_hand_expands() -> None:
    assert np.allclose(hex_to_rgb("#fff"), 1.0)


def test_oklab_of_white_is_neutral_bright() -> None:
    lab = hex_to_oklab("#ffffff")
    assert lab[0] == pytest.approx(1.0, abs=0.01)
    assert lab[1] == pytest.approx(0.0, abs=0.01)
    assert lab[2] == pytest.approx(0.0, abs=0.01)


def test_oklab_of_black_is_zero() -> None:
    assert np.allclose(hex_to_oklab("#000000"), 0.0, atol=1e-6)


def test_lightness_is_monotonic() -> None:
    shades = [f"#{value:02x}{value:02x}{value:02x}" for value in (0, 64, 128, 192, 255)]
    lightness = [hex_to_oklab(color)[0] for color in shades]
    assert all(a < b for a, b in itertools.pairwise(lightness))


def test_oklab_srgb_inverse() -> None:
    original = np.array([0.2, 0.5, 0.8])
    assert np.allclose(oklab_to_srgb(srgb_to_oklab(original)), original, atol=1e-9)


def test_rgb_to_hex_rounds_and_clamps() -> None:
    assert rgb_to_hex(np.array([1.4, -0.2, 0.5])) == "#ff0080"


def test_bad_hex_is_rejected() -> None:
    with pytest.raises(ValueError, match="hex color"):
        hex_to_rgb("not-a-color")


def test_palette_endpoints() -> None:
    palette = Palette(["#ff0066", "#00ccff", "#ffee00"])
    assert palette.to_hex(0.0) == "#ff0066"
    assert palette.to_hex(1.0) == "#ffee00"


def test_palette_clamps_out_of_range() -> None:
    palette = Palette(["#000000", "#ffffff"])
    assert palette.to_hex(-3.0) == "#000000"
    assert palette.to_hex(9.0) == "#ffffff"


def test_palette_midpoint_differs_from_srgb_mix() -> None:
    """The reason palettes interpolate in Oklab rather than sRGB."""
    palette = Palette(["#ff0000", "#0000ff"])
    oklab_mid = palette.to_hex(0.5)
    naive_srgb = rgb_to_hex(np.array([0.5, 0.0, 0.5]))
    assert oklab_mid != naive_srgb


def test_palette_accepts_array_of_positions() -> None:
    palette = Palette(["#000000", "#ffffff"])
    samples = palette.to_linear(np.array([0.0, 0.5, 1.0]))
    assert samples.shape == (3, 3)


def test_palette_rotation_wraps() -> None:
    palette = Palette(["#111111", "#222222", "#333333", "#444444"])
    # A half-turn of a 4-stop ramp is its own inverse, so -0.5 matches +0.5.
    assert palette.rotated(0.5).colors == ["#333333", "#444444", "#111111", "#222222"]
    assert palette.rotated(-0.5).colors == palette.rotated(0.5).colors
    assert palette.rotated(0.25).colors == ["#222222", "#333333", "#444444", "#111111"]
    assert palette.rotated(-0.25).colors == ["#444444", "#111111", "#222222", "#333333"]


def test_palette_rotation_is_periodic() -> None:
    """Turning the ramp by a whole number of cycles is the identity."""
    palette = Palette(["#111111", "#222222", "#333333", "#444444"])
    assert palette.rotated(1.0).colors == palette.colors
    assert palette.rotated(2.25).colors == palette.rotated(0.25).colors


def test_palette_requires_colors() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Palette([])


def test_palette_of_one_color_is_constant() -> None:
    palette = Palette(["#abcdef"])
    assert palette.to_hex(0.5) == "#abcdef"

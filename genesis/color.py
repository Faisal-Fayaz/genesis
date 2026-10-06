"""Color handling in Oklab.

Interpolating in sRGB is what makes naive gradients look muddy: the midpoint of
two saturated hues passes through a dull third color. Oklab is perceptually
uniform enough that a straight lerp looks like the eye expects, so palettes are
converted to it before any mixing happens.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)

_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)

_M1_INV = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ],
    dtype=np.float64,
)

_M2_INV = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ],
    dtype=np.float64,
)

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    c = np.asarray(linear, dtype=np.float64)
    return np.where(
        c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0.0, None) ** (1 / 2.4) - 0.055
    )


def linear_to_oklab(linear: np.ndarray) -> np.ndarray:
    lms = np.asarray(linear, dtype=np.float64) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lms = np.asarray(lab, dtype=np.float64) @ _M2_INV.T
    return lms**3 @ _M1_INV.T


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    return linear_to_oklab(srgb_to_linear(rgb))


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    return linear_to_srgb(oklab_to_linear(lab))


def hex_to_rgb(value: str) -> np.ndarray:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    if len(text) != 6:
        raise ValueError(f"expected a hex color like #1a2b3c, got {value!r}")
    return np.array(
        [int(text[index : index + 2], 16) / 255.0 for index in (0, 2, 4)],
        dtype=np.float64,
    )


def rgb_to_hex(rgb: np.ndarray) -> str:
    scaled = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in scaled)


def hex_to_oklab(value: str) -> np.ndarray:
    return srgb_to_oklab(hex_to_rgb(value))


def oklab_to_hex(lab: np.ndarray) -> str:
    return rgb_to_hex(oklab_to_srgb(lab))


def lerp(a: np.ndarray, b: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Linear interpolation; a scalar `t` keeps the operands' shape."""
    weight = float(t) if np.ndim(t) == 0 else np.asarray(t, dtype=np.float64)[..., None]
    return np.asarray(a) + (np.asarray(b) - np.asarray(a)) * weight


class Palette:
    """An ordered set of stops plus the sampling helpers generators need."""

    def __init__(self, colors: Sequence[str], name: str = "custom") -> None:
        if not colors:
            raise ValueError("a palette needs at least one color")
        self.name = name
        self.colors = list(colors)
        self._oklab = np.stack([hex_to_oklab(color) for color in self.colors])

    def __len__(self) -> int:
        return len(self.colors)

    def __repr__(self) -> str:
        return f"Palette(name={self.name!r}, colors={self.colors!r})"

    def to_linear(self, t: float | np.ndarray) -> np.ndarray:
        """Sample the ramp in Oklab and return linear-light RGB.

        `t` outside [0, 1] clamps to the end stops, so callers can overshoot
        without bounds-checking every draw.
        """
        scalar = np.ndim(t) == 0
        values = np.atleast_1d(np.asarray(t, dtype=np.float64))
        positions = np.clip(values, 0.0, 1.0) * (len(self.colors) - 1)
        low = np.floor(positions).astype(np.int64)
        high = np.minimum(low + 1, len(self.colors) - 1)
        frac = (positions - low)[..., None]
        lab = self._oklab[low] * (1.0 - frac) + self._oklab[high] * frac
        rgb = linear_to_srgb(oklab_to_linear(lab))
        return rgb[0] if scalar else rgb

    def to_hex(self, t: float) -> str:
        return rgb_to_hex(self.to_linear(t))

    def rotated(self, amount: float) -> Palette:
        """Return the same stops with the ramp's starting point shifted."""
        n = len(self.colors)
        offset = round(amount * n) % n
        return Palette(self.colors[offset:] + self.colors[:offset], self.name)


def as_palette(colors: Iterable[str] | None, default: Sequence[str]) -> Palette:
    return Palette(list(colors), "recipe") if colors else Palette(list(default), "default")

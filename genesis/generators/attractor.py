"""Strange attractors: iterated chaotic maps plotted as density.

The Clifford, de Jong, and Hopalong maps each take a point to a new point in a
bounded region. Plotting millions of them yields a density field whose structure
comes entirely from the map's parameters. Point colors follow the map's
velocity, which makes the interior structure visible.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..color import Palette
from ..params import ParamSpec, build_schema
from ..registry import register
from ..rng import stream
from ._common import downsample, scatter_add

SCHEMA = build_schema(
    [
        ParamSpec(
            "points",
            "int",
            900000,
            minimum=10000,
            maximum=8000000,
            step=100000,
            label="Points",
            help="Iterations to plot; more points means smoother density",
        ),
        ParamSpec(
            "form",
            "enum",
            "clifford",
            choices=("clifford", "dejong", "hopalong", "svensson"),
            label="Form",
            help="Which chaotic map to iterate",
        ),
        ParamSpec(
            "a",
            "float",
            -1.4,
            minimum=-3.0,
            maximum=3.0,
            step=0.01,
            label="a",
            help="Map coefficient; the interesting values are near this range",
        ),
        ParamSpec(
            "b",
            "float",
            1.6,
            minimum=-3.0,
            maximum=3.0,
            step=0.01,
            label="b",
        ),
        ParamSpec(
            "c",
            "float",
            -1.2,
            minimum=-3.0,
            maximum=3.0,
            step=0.01,
            label="c",
        ),
        ParamSpec(
            "d",
            "float",
            1.6,
            minimum=-3.0,
            maximum=3.0,
            step=0.01,
            label="d",
        ),
        ParamSpec(
            "warmup",
            "int",
            2000,
            minimum=0,
            maximum=200000,
            step=100,
            label="Warmup",
            help="Iterations to discard so the initial transient does not show",
        ),
        ParamSpec(
            "color_count",
            "int",
            5,
            minimum=2,
            maximum=12,
            label="Palette stops",
        ),
        ParamSpec(
            "hue_bias",
            "float",
            0.0,
            minimum=-1.0,
            maximum=1.0,
            step=0.05,
            label="Hue bias",
        ),
        ParamSpec(
            "exposure",
            "float",
            2.4,
            minimum=0.1,
            maximum=20.0,
            step=0.1,
            label="Exposure",
            help="How quickly dense regions saturate",
        ),
        ParamSpec(
            "gamma",
            "float",
            0.75,
            minimum=0.2,
            maximum=2.5,
            step=0.05,
            label="Gamma",
            help="<1 lifts faint structure, >1 emphasizes the dense core",
        ),
        ParamSpec(
            "supersample",
            "int",
            1,
            minimum=1,
            maximum=3,
            label="Supersampling",
        ),
    ]
)


def _iterate(
    form: str, px: np.ndarray, py: np.ndarray, a: float, b: float, c: float, d: float
) -> tuple[np.ndarray, np.ndarray]:
    if form == "clifford":
        nx = np.sin(a * py) + c * np.cos(a * px)
        ny = np.sin(b * px) + d * np.cos(b * py)
    elif form == "dejong":
        nx = np.sin(a * py) - np.cos(b * px)
        ny = np.sin(c * px) - np.cos(d * py)
    elif form == "hopalong":
        nx = py - d * np.sign(px)
        ny = b * px + c * np.cos(a * px)
    elif form == "svensson":
        d2 = px * px - py * py
        nx = a * px * np.sin(b * py) - d2
        ny = c * py * np.cos(a * px) + d2
    else:
        raise ValueError(f"unknown form {form!r}")
    return nx, ny


def _render(
    params: dict[str, Any], seed: int, width: int, height: int, palette: Palette
) -> np.ndarray:
    factor = int(params["supersample"])
    work_w, work_h = width * factor, height * factor

    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])
    d = float(params["d"])

    rng = stream(seed, "points")
    px = rng.uniform(-0.1, 0.1, 2048)
    py = rng.uniform(-0.1, 0.1, 2048)

    for _ in range(int(params["warmup"])):
        px, py = _iterate(str(params["form"]), px, py, a, b, c, d)

    points = int(params["points"])
    batch = min(262144, max(points // 16, 8192))
    color_sum = np.zeros((work_h, work_w, 3), dtype=np.float64)
    weight_sum = np.zeros((work_h, work_w), dtype=np.float64)

    stops = max(int(params["color_count"]), 2)
    bias = float(params["hue_bias"])
    ramp_t = np.linspace(0.0, 1.0, stops)
    ramp_rgb = palette.to_linear(np.clip(ramp_t + bias, 0.0, 1.0))

    remaining = points
    while remaining > 0:
        count = min(batch, remaining)
        remaining -= count

        xs = np.empty(count, dtype=np.float64)
        ys = np.empty(count, dtype=np.float64)
        speed = np.empty(count, dtype=np.float64)
        for index in range(count):
            slot = index % px.size
            old_x, old_y = px[slot], py[slot]
            nx, ny = _iterate(str(params["form"]), old_x, old_y, a, b, c, d)
            px[slot], py[slot] = nx, ny
            xs[index] = nx
            ys[index] = ny
            speed[index] = np.hypot(nx - old_x, ny - old_y)

        # Attractor output sits in roughly [-2, 2]; scale to the canvas.
        col = np.clip(((xs + 2.0) / 4.0 * work_w).astype(np.int64), 0, work_w - 1)
        row = np.clip(((2.0 - ys) / 4.0 * work_h).astype(np.int64), 0, work_h - 1)
        flat = (row * work_w + col).astype(np.int64)

        speed_t = np.clip(speed / max(float(np.percentile(speed, 98)), 1e-9), 0.0, 1.0)
        weights = 0.25 + 0.75 * speed_t
        scatter_add(weight_sum, flat, weights)
        scatter_add(
            color_sum,
            flat,
            np.stack(
                [np.interp(speed_t, ramp_t, ramp_rgb[:, channel]) for channel in range(3)],
                axis=-1,
            )
            * weights[:, None],
        )

    weight_sum = downsample(weight_sum, factor)
    color_sum = downsample(color_sum, factor)

    lit = weight_sum > 1e-9
    averaged = np.zeros_like(color_sum)
    averaged[lit] = color_sum[lit] / weight_sum[lit][:, None]

    density = 1.0 - np.exp(-weight_sum * float(params["exposure"]))
    density = density ** float(params["gamma"])
    background = palette.to_linear(0.0)
    result = background * (1.0 - density[..., None]) + averaged * density[..., None]
    return np.clip(result, 0.0, 1.0)


def render(
    params: dict[str, Any], seed: int, width: int, height: int, palette: Palette
) -> np.ndarray:
    return _render(params, seed, width, height, palette)


register(
    "attractor",
    "Chaotic maps (Clifford, de Jong, Hopalong, Svensson) plotted as density",
    SCHEMA,
    render,
    family="vector",
)

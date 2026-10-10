"""Splash: billowing ink clouds blooming on a dark field.

Each blob is a soft radial splat of ink, displaced by a turbulence field so
clouds cluster and curl instead of scattering uniformly, then composited
back-to-front with plain alpha-over. Soft falloff plus turbulence is what
reads as ink in water rather than polka dots: hard edges or grid layouts
would break the illusion immediately.

Costs scale with blob count times blob area, so the count follows the longest
edge like everywhere else in this codebase, and each blob only touches its
own bounding box.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import noise
from ..color import Palette
from ..params import ParamSpec, build_schema
from ..registry import register
from ..rng import stream
from ._common import downsample

SCHEMA = build_schema(
    [
        ParamSpec(
            "blobs",
            "int",
            45,
            minimum=8,
            maximum=2000,
            step=5,
            label="Blob count",
            help="Ink blobs to bloom, scaled up with canvas size",
        ),
        ParamSpec(
            "spread",
            "float",
            0.11,
            minimum=0.01,
            maximum=0.4,
            step=0.01,
            label="Spread",
            help="Mean blob radius as a fraction of the longest edge",
        ),
        ParamSpec(
            "turbulence",
            "float",
            0.35,
            minimum=0.0,
            maximum=1.5,
            step=0.05,
            label="Turbulence",
            help="How far the noise field drags blobs from their grid; 0 is uniform scatter",
        ),
        ParamSpec(
            "fray",
            "float",
            0.7,
            minimum=0.0,
            maximum=1.5,
            step=0.05,
            label="Fray",
            help="How much noise frays blob edges into tendrils; 0 is smooth bokeh",
        ),
        ParamSpec(
            "detail",
            "int",
            4,
            minimum=1,
            maximum=8,
            label="Detail",
            help="fBm octaves driving the turbulence field",
        ),
        ParamSpec(
            "color_count",
            "int",
            3,
            minimum=2,
            maximum=8,
            label="Palette stops",
            help="How many colors to sample from the palette",
        ),
        ParamSpec(
            "hue_bias",
            "float",
            0.0,
            minimum=-1.0,
            maximum=1.0,
            step=0.05,
            label="Hue bias",
            help="Shifts which part of the palette blobs land on",
        ),
        ParamSpec(
            "exposure",
            "float",
            1.2,
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            label="Exposure",
            help="Overall gain around the palette floor",
        ),
        ParamSpec(
            "supersample",
            "int",
            1,
            minimum=1,
            maximum=3,
            label="Supersampling",
            help=(
                "Render at N times size and downsample for softer blob edges. "
                "Off by default: blobs are already soft, and it multiplies cost."
            ),
        ),
    ]
)

# Cap on blobs after validation. Coverage saturates well before this, and the
# cap keeps a careless recipe from spending minutes on clouds past the point
# of visible difference.
MAX_BLOBS = 2000

# Falloff width as a fraction of blob radius: exp(-d^2 / (2 sigma^2)) with
# sigma = radius / 2, cut off where it no longer matters.
SIGMA_FRACTION = 0.5
CUTOFF_SIGMAS = 3.0


def _render(
    params: dict[str, Any], seed: int, width: int, height: int, palette: Palette
) -> np.ndarray:
    factor = int(params["supersample"])
    work_w, work_h = width * factor, height * factor
    edge = max(work_w, work_h)

    # The count stays fixed across canvas sizes: blob radii already scale with
    # the longest edge, so a constant count holds the coverage fraction (and
    # therefore the composition) steady. Scaling the count too would grow
    # covered area with the cube of the edge — the preset suite caught exactly
    # that as unstable quiet area across resolutions.
    count = min(MAX_BLOBS, int(params["blobs"]))
    mean_radius = float(params["spread"]) * edge
    turbulence = float(params["turbulence"]) * edge

    centers = stream(seed, "centers")
    cx = centers.random(count) * work_w
    cy = centers.random(count) * work_h

    if turbulence > 0.0:
        # Drift each blob along a noise vector field so clouds gather and
        # curl. Sampled only at blob centers, so this costs almost nothing
        # next to the compositing below.
        perm = noise.permutation_table(stream(seed, "turbulence"))
        detail = int(params["detail"])
        raw_x = noise.fbm(perm, cx / edge, cy / edge, octaves=detail, frequency=1.0)
        raw_y = noise.fbm(perm, cy / edge, cx / edge, octaves=detail, frequency=2.0)
        cx = cx + (raw_x - raw_x.mean()) * turbulence
        cy = cy + (raw_y - raw_y.mean()) * turbulence

    radii = stream(seed, "radii")
    # Lognormal sizes: mostly modest puffs with a few large billows, which is
    # what keeps two same-seed renders from looking like fixed-size confetti.
    radius = np.clip(
        mean_radius * np.exp(0.7 * radii.standard_normal(count)),
        2.0,
        edge,
    )

    stops = max(int(params["color_count"]), 2)
    bias = float(params["hue_bias"])
    ramp_rgb = palette.to_linear(np.linspace(0.0, 1.0, stops))
    colors = stream(seed, "colors")
    # Picks cluster around the middle of the ramp, shifted by hue_bias, with
    # a triangular distribution. Uniform picks would land on the near-white
    # top stop as often as on saturated mids, and big soft blobs of near-white
    # wash the whole field out to beige soup. Saturated cores on a dark field
    # are the entire look. The bias moves only the peak, so unlike a sliding
    # ramp it cannot pin every pick to one end stop.
    peak = np.clip((stops - 1) * 0.5 * (1.0 + bias), 0.0, stops - 1)
    draws = np.clip(
        np.floor(colors.triangular(-0.5, peak, stops - 0.5, count)).astype(np.int64),
        0,
        stops - 1,
    )
    ink = ramp_rgb[draws]
    alpha = colors.uniform(0.35, 0.85, count).astype(np.float32)

    # One shared fray texture for the whole render: sampling it per blob is
    # nearly free, while evaluating noise per blob would multiply the field
    # cost by the blob count.
    fray_strength = float(params["fray"])
    fray_tex = None
    if fray_strength > 0.0:
        grid = stream(seed, "fray")
        fxs, fys = noise.coordinate_grid(work_w, work_h, work_h / work_w)
        raw = noise.fbm(
            noise.permutation_table(grid), fxs, fys,
            octaves=int(params["detail"]), frequency=3.0,
        )
        span = max(float(raw.max() - raw.min()), 1e-12)
        fray_tex = ((raw - raw.min()) / span).astype(np.float32)

    # Back-to-front in a shuffled order: nearer clouds occlude farther ones,
    # and the shuffle (not sorted order) is what stops banding between
    # overlapping blobs of similar color.
    order = stream(seed, "order").permutation(count)

    # Start at the palette floor, not black: the exposure gain below pivots
    # around the floor, so untouched pixels map to themselves at any exposure
    # instead of crushing to black. Ink composites over the floor the way
    # light would actually layer.
    background = np.asarray(palette.to_linear(0.0), dtype=np.float32)
    canvas = np.zeros((work_h, work_w, 3), dtype=np.float32) + background
    for index in order:
        x, y, r = float(cx[index]), float(cy[index]), float(radius[index])
        sigma = max(r * SIGMA_FRACTION, 1.0)
        half = int(np.ceil(CUTOFF_SIGMAS * sigma))
        x0, x1 = max(0, int(x - half)), min(work_w, int(x + half) + 1)
        y0, y1 = max(0, int(y - half)), min(work_h, int(y + half) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        falloff = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2.0 * sigma * sigma))
        if fray_tex is not None:
            frayed = 1.0 + fray_strength * (2.0 * fray_tex[y0:y1, x0:x1] - 1.0)
            falloff = falloff * np.clip(frayed, 0.0, 1.5)
        a = (float(alpha[index]) * falloff)[..., None]
        region = canvas[y0:y1, x0:x1]
        region[:] = region * (1.0 - a) + ink[index] * a

    canvas = downsample(canvas, factor)

    # Gain around the palette floor: the background stays put at any exposure,
    # and overlapping clouds saturate toward their ink rather than clipping.
    # This only holds because the canvas starts *at* the background (above):
    # gain pivots around the floor, so untouched pixels map to themselves.
    background = palette.to_linear(0.0)
    result = background + (canvas - background) * float(params["exposure"])
    return np.clip(result, 0.0, 1.0)


def render(
    params: dict[str, Any], seed: int, width: int, height: int, palette: Palette
) -> np.ndarray:
    return _render(params, seed, width, height, palette)


register(
    "splash",
    "Billowing ink clouds blooming on a dark field",
    SCHEMA,
    render,
    family="field",
)

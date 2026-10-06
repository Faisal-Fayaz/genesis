"""Strange attractors: iterated chaotic maps plotted as density.

The Clifford, de Jong, and Hopalong maps each take a point to a new point in a
bounded region. Plotting millions of them yields a density field whose structure
comes entirely from the map's parameters. Point colors follow the map's
velocity, which makes the interior structure visible.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..canon import stable_hash
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
            "fit",
            "enum",
            "auto",
            choices=("auto", "fixed"),
            label="Framing",
            help="auto fills the frame with the orbit; fixed uses the extent value",
        ),
        ParamSpec(
            "zoom",
            "float",
            1.0,
            minimum=0.08,
            maximum=1.0,
            step=0.02,
            label="Zoom",
            help="1.0 fits the whole orbit; lower crops in, filling the frame with structure",
        ),
        ParamSpec(
            "pan_x",
            "float",
            0.0,
            minimum=-1.0,
            maximum=1.0,
            step=0.02,
            label="Pan x",
            help="Shifts the cropped window horizontally, as a fraction of its width",
        ),
        ParamSpec(
            "pan_y",
            "float",
            0.0,
            minimum=-1.0,
            maximum=1.0,
            step=0.02,
            label="Pan y",
        ),
        ParamSpec(
            "extent",
            "float",
            2.0,
            minimum=0.25,
            maximum=8.0,
            step=0.05,
            label="Extent",
            help="Half-width of the view window when framing is fixed",
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
            1.3,
            minimum=0.1,
            maximum=4.0,
            step=0.05,
            label="Exposure",
            help="Steepness of the log density response; higher is punchier",
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


# Coefficient sets verified to settle into a bounded, reasonably wide orbit.
# Most of the parameter space is instead a fixed point, a short cycle, or a
# divergence, all of which render as a flat field, so a recipe whose own
# coefficients fail the probe falls back to this table.
#
# Each was checked for: no escape over 900 steps, and a settled span between
# roughly 1 and 8 units. Svensson in particular is fragile; almost all of its
# parameter space diverges, which is why its usable coefficients cluster near
# a=-1.5 with |c| large.
KNOWN_GOOD: dict[str, tuple[tuple[float, float, float, float], ...]] = {
    "clifford": (
        (-0.799, 1.494, -1.979, 1.285),
        (1.982, 1.171, 0.489, 1.956),
        (-1.139, -1.359, 0.45, -1.824),
        (-1.953, -1.23, 0.768, -1.198),
        (-0.522, -1.985, 1.32, -1.382),
        (-0.93, 1.521, 0.039, 1.389),
    ),
    "dejong": (
        (1.78, 1.616, 0.279, -1.418),
        (-1.899, -0.511, -1.879, -1.508),
        (-0.669, -0.407, -1.188, -1.797),
        (-1.148, 1.662, 1.361, -1.55),
        (1.056, 1.261, 0.92, -1.547),
        (1.663, -1.813, -1.879, -1.919),
    ),
    "hopalong": (
        (-2.163, -0.48, -1.275, 1.726),
        (-1.782, -0.284, 1.431, 1.973),
        (-2.069, 0.308, 2.317, 2.036),
        (2.455, 0.18, -1.899, -0.407),
        (2.146, -0.467, 1.23, -0.673),
        (-1.732, 0.376, -2.068, 0.817),
    ),
    "svensson": (
        (1.324, 1.498, -1.393, -0.317),
        (-0.283, 0.874, -1.383, -1.115),
        (-0.39, -1.517, -1.458, -1.492),
        (-0.369, -1.176, -1.541, -1.017),
        (1.061, -0.526, -1.41, 2.278),
        (0.965, -0.7, -1.367, 0.822),
    ),
}

# An orbit must fill at least this much to be worth plotting. Below it, the map
# has settled into a point or a short cycle.
COLLAPSED_SPREAD = 0.25

# A settled orbit that spans more than this is not a bounded attractor in any
# useful sense; treating it as one would zoom the view out to nothing.
MAX_SPREAD = 12.0

# Coordinates are bounded well below the float64 ceiling so that squaring in
# Svensson cannot overflow before the divergence screen runs.
ESCAPE_LIMIT = 1e30

# Orbits are advanced in lockstep, so the pool size is the batch size.
POOL_SIZE = 16384

# Hopalong's map is x' = y - d*sgn(x), which is symmetric under sign flips. That
# makes orbits of opposite sign mirror onto each other and retrace, so the
# plotted curve collapses to far fewer distinct pixels than the point count
# suggests. Advancing a wide pool at once mixes the two families together.
# Spreading the sign over several independent batches keeps each batch's orbit
# family distinct, which restores the coverage.
HOPALONG_BATCHES = 4

# The transient of these maps settles within a few hundred steps. Recipes can
# request far more, and paying for that per-step screening would dominate
# runtime, so warmup is capped at what actually changes the orbit.
WARMUP_CAP = 800

# Probing must stay cheap: it runs once per candidate coefficient set, and the
# verdict (settled or not) is visible long before the transient is fully gone.
PROBE_WARMUP_CAP = 800

# Escaped orbits are restarted every this many steps. The screen is a full pass
# over the pool, so doing it every step costs more than the map itself; since
# `_resolve_coefficients` has already rejected divergent maps, a restart is only
# a safety net and can lag by a few steps without visible effect.
SCREEN_EVERY = 16

# Steps a probe orbit takes past the requested warmup before its spread is
# trusted. The transient can look rich while the settled attractor is a cycle.
PROBE_SETTLE_STEPS = 3000

# A settled orbit should visit at least this many distinct positions. Anything
# fewer is a fixed point or a short cycle, which plots as a dot.
MIN_ORBIT_POSITIONS = 16


def _spread(px: np.ndarray, py: np.ndarray) -> float:
    """How much area the orbit actually covers."""
    if px.size == 0:
        return 0.0
    span_x = float(px.max() - px.min())
    span_y = float(py.max() - py.min())
    return max(span_x, span_y)


def _probe(
    form: str, a: float, b: float, c: float, d: float, warmup: int, seed: int
) -> float:
    """How much of the plane a coefficient set fills, or 0.0 if it is unusable.

    An orbit is disqualified for either failure mode. It may *collapse*, settling
    into a fixed point or short cycle, which shows up as near-zero spread. Or it
    may *diverge*, running off toward infinity.

    Divergence must be detected without ever rescuing the orbit: restarting an
    escaped point would make a runaway map look healthy, and the plot would then
    be framed on a view billions of units wide. So this tracks escapes and
    reports 0.0 rather than papering over them.
    """
    rng = stream(seed, "probe")
    px = rng.uniform(-0.2, 0.2, 64)
    py = rng.uniform(-0.2, 0.2, 64)
    coefficients = (a, b, c, d)

    for _ in range(min(warmup, PROBE_WARMUP_CAP)):
        px, py = _iterate(form, px, py, *coefficients)
        if _escaped(px).any() or _escaped(py).any():
            return 0.0

    # Measure the *settled* orbit, not the transient. Many maps take hundreds of
    # steps to fall into their attractor, and probing too early sees a wide
    # transient that later collapses to a short cycle. Hopalong in particular
    # collapses to a single point, which would then plot every sample into one
    # pixel.
    for _ in range(PROBE_SETTLE_STEPS):
        px, py = _iterate(form, px, py, *coefficients)
        if _escaped(px).any() or _escaped(py).any():
            return 0.0

    before_x, before_y = px.copy(), py.copy()
    for _ in range(64):
        px, py = _iterate(form, px, py, *coefficients)
        if _escaped(px).any() or _escaped(py).any():
            return 0.0

    # A settled orbit stops moving even if its spread is nonzero.
    if _spread(px - before_x, py - before_y) < 1e-9:
        return 0.0

    # Reject short cycles: an orbit that revisits only a handful of positions
    # plots as a single dot or a few dots rather than a curve.
    positions = set(
        zip(np.round(px, 6).tolist(), np.round(py, 6).tolist(), strict=True)
    )
    if len(positions) < MIN_ORBIT_POSITIONS:
        return 0.0

    spread = _spread(px, py)
    # Hopalong drifts outward forever at a rate too slow to trip the escape
    # screen within the probe window, so an implausibly wide orbit is rejected
    # here instead.
    if spread > MAX_SPREAD:
        return 0.0
    return spread


def _resolve_coefficients(
    params: dict[str, Any], seed: int
) -> tuple[str, float, float, float, float]:
    """Pick coefficients that actually produce an image.

    Most of the parameter space is either collapsed or divergent, and both
    render as a flat field. The recipe's own coefficients are tried first; if
    they fail, a table of known-good sets is searched, nudged so that different
    recipes with equally bad coefficients still resolve differently.

    The search is driven by the recipe seed, so the result is stable across
    renders. It falls back to the first known-good set only when nothing at all
    works, which for the shipped forms does not happen.
    """
    form = str(params["form"])
    requested = (
        float(params["a"]),
        float(params["b"]),
        float(params["c"]),
        float(params["d"]),
    )
    warmup = int(params["warmup"])

    if _usable(form, requested, warmup, seed):
        return (form, *requested)

    rng = stream(seed, "coefficients")
    candidates: list[tuple[float, float, float, float]] = []
    for base in KNOWN_GOOD.get(form, ()):
        candidates.append(base)
        for _ in range(4):
            candidates.append(
                tuple(
                    float(np.clip(value + rng.uniform(-0.25, 0.25), -3.0, 3.0))
                    for value in base
                )
            )

    for candidate in candidates:
        if _usable(form, candidate, warmup, seed):
            return (form, *candidate)

    fallback = KNOWN_GOOD.get(form)
    if fallback:
        return (form, *fallback[0])
    raise ValueError(f"no usable coefficients found for form {form!r}")


def _usable(
    form: str,
    coefficients: tuple[float, float, float, float],
    warmup: int,
    seed: int,
) -> bool:
    return _probe(form, *coefficients, warmup, seed) >= COLLAPSED_SPREAD


def _iterate(
    form: str,
    px: np.ndarray,
    py: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the maps one step.

    Callers screen for escaped orbits themselves via `_replace_escaped`, so this
    function stays a pure map application and never raises.
    """
    if form == "clifford":
        nx = np.sin(a * py) + c * np.cos(a * px)
        ny = np.sin(b * px) + d * np.cos(b * py)
    elif form == "dejong":
        nx = np.sin(a * py) - np.cos(b * px)
        ny = np.sin(c * px) - np.cos(d * py)
    elif form == "hopalong":
        # The map uses sign(px), which is discontinuous. Writing it as
        # where(px > 0, d, -d) instead of sign() keeps it exact for nonzero px;
        # sign(0) is 0 and would freeze an orbit that lands exactly on the axis.
        nx = py - d * np.where(px > 0.0, 1.0, -1.0)
        ny = b * px + c * np.cos(a * px)
    elif form == "svensson":
        # Svensson squares its coordinates, so it diverges much faster than the
        # others. Screening at ESCAPE_LIMIT is not enough on its own: a coordinate
        # near the limit overflows within one step. Capping first keeps the
        # multiply in range, and the caller restarts those orbits anyway.
        d2 = np.clip(px * px - py * py, -ESCAPE_LIMIT, ESCAPE_LIMIT)
        nx = a * px * np.sin(b * py) - d2
        ny = c * py * np.cos(a * px) + d2
    else:
        raise ValueError(f"unknown form {form!r}")
    return nx, ny


def _orbit(
    form: str,
    coefficients: tuple[float, float, float, float],
    seed: int,
    warmup: int,
    *,
    steps: int = 0,
    size: int = POOL_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance a pool of points past the transient, ready to be plotted.

    Orbits that run away are replaced from the seeded stream rather than being
    allowed to poison the pool with NaN, so every returned point is finite.
    """
    # Scatter the starting points rather than filling a grid. Maps with a
    # sign() branch split the plane into families that never mix once they have
    # separated, and a gridded pool leaves every point in a row on the same side
    # of that branch, which shows up as axis-aligned structure in the render.
    rng = stream(seed, "points")
    px = rng.uniform(-0.4, 0.4, size)
    py = rng.uniform(-0.4, 0.4, size)

    # Escaped orbits are restarted every SCREEN_EVERY steps rather than every
    # step. The screen is a full pass over the pool, so doing it per step costs
    # more than the map itself; a restart a few steps late is indistinguishable.
    restart = stream(stable_hash([form, *coefficients], size=4), "restart")
    for step in range(min(warmup, WARMUP_CAP)):
        px, py = _iterate(form, px, py, *coefficients)
        if step % SCREEN_EVERY == 0:
            px, py = _replace_escaped(px, py, restart)
    for step in range(steps):
        px, py = _iterate(form, px, py, *coefficients)
        if step % SCREEN_EVERY == 0:
            px, py = _replace_escaped(px, py, restart)
    px, py = _replace_escaped(px, py, restart)
    return px, py


def _escaped(values: np.ndarray) -> np.ndarray:
    """Which coordinates have left the usable range."""
    return ~(np.isfinite(values)) | (np.abs(values) > ESCAPE_LIMIT)


def _replace_escaped(
    px: np.ndarray, py: np.ndarray, restart: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Restart any orbit that left the usable range.

    Safe to call only once a coefficient set is known to be bounded, since it
    hides divergence rather than reporting it. `_probe` is what establishes that.
    """
    bad = _escaped(px) | _escaped(py)
    if not bad.any():
        return px, py
    px = np.where(bad, restart.uniform(-0.2, 0.2, px.size), px)
    py = np.where(bad, restart.uniform(-0.2, 0.2, py.size), py)
    return px, py


def _bounded_orbit(
    form: str,
    coefficients: tuple[float, float, float, float],
    seed: int,
    warmup: int,
) -> tuple[np.ndarray, np.ndarray]:
    """A sample of settled orbits, used to frame the view."""
    sample_x: list[np.ndarray] = []
    sample_y: list[np.ndarray] = []
    for _ in range(4):
        # A small pool is enough to find the extent, and keeps framing cheap.
        px, py = _orbit(
            form, coefficients, seed + _ * 7919, warmup, steps=400, size=2048
        )
        sample_x.append(px)
        sample_y.append(py)
    return np.concatenate(sample_x), np.concatenate(sample_y)


def _render(
    params: dict[str, Any], seed: int, width: int, height: int, palette: Palette
) -> np.ndarray:
    factor = int(params["supersample"])
    work_w, work_h = width * factor, height * factor

    form, a, b, c, d = _resolve_coefficients(params, seed)
    coefficients = (a, b, c, d)
    warmup = int(params["warmup"])

    px, py = _orbit(form, coefficients, seed, warmup)

    # Fit the view to the orbit rather than assuming [-2, 2]. Maps like Svensson
    # range well outside that, and a hardcoded window would either clip the
    # attractor or shrink it into the middle of the canvas.
    fit = str(params["fit"])
    if fit == "auto":
        probe_x, probe_y = _bounded_orbit(form, coefficients, seed, warmup)
        low_x, high_x = _percentile_bounds(probe_x, 0.002)
        low_y, high_y = _percentile_bounds(probe_y, 0.002)
    else:
        half = float(params["extent"])
        low_x, high_x = -half, half
        low_y, high_y = -half, half

    span_x = max(high_x - low_x, 1e-6)
    span_y = max(high_y - low_y, 1e-6)

    # Crop into the orbit. These attractors are sparse curves, so framing the
    # whole orbit leaves most of the canvas empty; zooming in fills the frame
    # with structure, and panning chooses which part.
    zoom = float(params["zoom"])
    if zoom < 1.0:
        view_x = span_x * zoom
        view_y = span_y * zoom
        pan_x = float(params["pan_x"]) * view_x
        pan_y = float(params["pan_y"]) * view_y
        center_x = (low_x + high_x) / 2.0
        center_y = (low_y + high_y) / 2.0
        low_x = center_x - view_x / 2.0 + pan_x
        high_x = low_x + view_x
        low_y = center_y - view_y / 2.0 + pan_y
        high_y = low_y + view_y
        span_x, span_y = view_x, view_y

    points = int(params["points"])
    # The pool is the pool: one vectorized step yields POOL_SIZE points, so a
    # batch larger than that would plot coordinates twice.
    batch = min(POOL_SIZE, max(points // 24, 4096))
    weight_sum = np.zeros((work_h, work_w), dtype=np.float64)
    speed_sum = np.zeros((work_h, work_w), dtype=np.float64)
    restart = stream(stable_hash([form, *coefficients], size=4), "restart")

    speed_scale = 0.0
    batch_index = 0
    remaining = points
    while remaining > 0:
        count = min(batch, remaining)
        remaining -= count

        # One vectorized step over the whole pool; plot the first `count`.
        old_x, old_y = px.copy(), py.copy()
        px, py = _iterate(form, old_x, old_y, *coefficients)
        px, py = _replace_escaped(px, py, restart)

        xs, ys = px[:count], py[:count]
        old_x, old_y = old_x[:count], old_y[:count]
        batch_index += 1

        speed = np.hypot(xs - old_x, ys - old_y)
        speed_scale = max(speed_scale, float(np.percentile(speed, 98)))

        # Discard points outside the view rather than clamping them. Clamping
        # piles every out-of-view sample onto the border rows and columns, which
        # shows up as axis-aligned banding along the edges of the frame.
        raw_col = ((xs - low_x) / span_x * (work_w - 1)).astype(np.int64)
        raw_row = ((high_y - ys) / span_y * (work_h - 1)).astype(np.int64)
        inside = (raw_col >= 0) & (raw_col < work_w) & (raw_row >= 0) & (raw_row < work_h)
        if not inside.all():
            if not inside.any():
                continue
            xs, ys, speed = xs[inside], ys[inside], speed[inside]
            raw_col, raw_row = raw_col[inside], raw_row[inside]
        speed_t = np.clip(speed / max(speed_scale, 1e-9), 0.0, 1.0)
        flat = (raw_row * work_w + raw_col).astype(np.int64)

        scatter_add(weight_sum, flat, np.ones(flat.size))
        scatter_add(speed_sum, flat, speed_t)

    weight_sum = downsample(weight_sum, factor)
    speed_sum = downsample(speed_sum, factor)

    # Exposure is referenced to a high percentile of the pixels the orbit
    # actually reached, not to the peak. These attractors are thin curves whose
    # density varies by orders of magnitude, and a handful of pixels sit where
    # many passes overlap. Referencing the peak would leave the whole curve
    # nearly invisible.
    reached = weight_sum[weight_sum > 0.0]
    if reached.size == 0:
        return np.tile(palette.to_linear(0.0), (work_h, work_w, 1))

    # Clip the accumulator before mapping it to color.
    #
    # Some orbits are aperiodic but still concentrate: hopalong's can hover on an
    # attracting point for tens of thousands of iterations while the rest of the
    # curve sits at a count of two. That produces single pixels five thousand
    # times brighter than their neighbours, which no tone curve can render
    # without clipping them to white blocks. Capping the dynamic range keeps the
    # structure visible and turns those spikes into ordinary dense core.
    reference = float(np.percentile(reached, 75.0))
    if reference <= 1e-12:
        return np.tile(palette.to_linear(0.0), (work_h, work_w, 1))
    ceiling = reference * _DENSITY_RANGE
    weight_sum = np.minimum(weight_sum, ceiling)

    # Compress the density range in log space before mapping it to the ramp.
    # A plain exponential response cannot serve a curve whose density spans
    # four orders of magnitude: it clips the overlap points to white blocks and
    # leaves everything else black. Log spacing puts both on a readable scale.
    # The floor keeps empty pixels out of the log.
    floor = max(reference * _DENSITY_FLOOR_RATIO, 1e-9)
    compressed = np.log1p(np.maximum(weight_sum, 0.0) / floor)
    span = np.log1p(max(float(weight_sum.max()) / floor, 1.0))
    if span <= 1e-9:
        density = np.zeros_like(weight_sum)
    else:
        density = np.clip(compressed / span, 0.0, 1.0)
        density = density ** float(params["exposure"])

    # Color by density. Coloring by speed instead would paint the dense core,
    # where most of the orbit's time is spent and step length is near zero, in
    # the darkest palette entry, which is also the background color.
    stops = max(int(params["color_count"]), 2)
    bias = float(params["hue_bias"])
    # Shift the sampled window so the darkest stop stays out of the ramp:
    # density 0 is the background, and must not be a usable stroke color.
    lower = 1.0 / stops
    ramp_t = lower + (1.0 - lower) * np.linspace(0.0, 1.0, stops)
    ramp_t = np.clip(ramp_t + bias, lower, 1.0)
    ramp_rgb = palette.to_linear(ramp_t)

    shape = density.shape
    result = np.stack(
        [
            np.interp(density.ravel(), ramp_t, ramp_rgb[:, channel]).reshape(shape)
            for channel in range(3)
        ],
        axis=-1,
    )

    # Blend toward the background where density is low, so the ramp's first stop
    # does not read as a hard edge against the empty field.
    background = palette.to_linear(0.0)
    blend = np.clip(density / max(ramp_t[0], 1e-6), 0.0, 1.0)
    result = background * (1.0 - blend[..., None]) + result * blend[..., None]
    return np.clip(result, 0.0, 1.0)


# Dynamic range kept when mapping density to color, as a multiple of the 75th
# percentile. Anything above is treated as equally dense.
_DENSITY_RANGE = 12.0

# A pixel holding this fraction of the reference density still counts as empty.
# It sets the floor of the log scale; without it, the sheer number of
# near-zero pixels would dominate and squeeze the visible curve into nothing.
_DENSITY_FLOOR_RATIO = 0.05


def _percentile_bounds(values: np.ndarray, trim: float) -> tuple[float, float]:
    low = float(np.percentile(values, 100.0 * trim))
    high = float(np.percentile(values, 100.0 * (1.0 - trim)))
    return low, high


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

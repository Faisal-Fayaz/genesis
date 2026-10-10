"""Dense dye advection: the whole screen animates, not just trails.

A live session keeps a full-frame dye field — initialized pixel-identical to
the generated still — and advects it through a velocity field every frame.
The velocity is a static ambient flow plus a decaying stir layer the cursor
paints into. Because every pixel moves, the image reads as fluid rather than
as sparse lines drawn over a picture.

Two properties keep this affordable on a CPU:

- The simulation runs at half output resolution and upscales for display.
  Ink clouds are low-frequency by nature; a 2x bilinear upscale loses nothing
  the eye can see, and quarters the per-frame traffic.
- All scratch buffers are preallocated once. A frame is a fixed sequence of
  vectorized passes with no per-frame allocation beyond the upload bytes.

Determinism holds for identical input sequences: same base, same cursor path,
same dye out. The cursor is external input, so live output is interactive
rather than byte-identical across runs — but nothing here rolls dice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .field import build_field

# Simulation resolution as a fraction of output. Ink clouds are smooth, so a
# 2x bilinear upscale is visually lossless and quarters the traffic.
SIM_SCALE = 0.5
# Fidelity ladder for weak machines: the governor steps down through these
# (rebuilding the grid) before it steps the frame rate down. Each step
# roughly halves the gather traffic, which is what the frame budget is made
# of; each step also softens motion detail, which is why fps notches stay
# the last resort rather than the first.
SIM_LEVELS = (0.5, 0.4, 0.35)


@dataclass
class CursorTracker:
    """Exponentially smoothed pointer position with idle detection."""

    x: float = 0.0
    y: float = 0.0
    last_move_frame: int = 0
    initialized: bool = False
    smoothing: float = 0.35

    def observe(self, x: float, y: float, frame: int) -> None:
        if not self.initialized:
            self.x, self.y = float(x), float(y)
            self.initialized = True
        else:
            self.x += (float(x) - self.x) * self.smoothing
            self.y += (float(y) - self.y) * self.smoothing
        self.last_move_frame = frame

    def idle_seconds(self, frame: int, fps: float) -> float:
        return max(0, (frame - self.last_move_frame)) / max(fps, 1.0)

    def position(self) -> tuple[float, float] | None:
        return (self.x, self.y) if self.initialized else None


@dataclass
class FluidState:
    """Everything one advection step reads and writes, preallocated.

    A frame is a fixed sequence of vectorized passes with no per-frame
    allocation beyond the upload bytes: even index arithmetic and the
    relax blend run into these buffers. This matters because the frame
    budget is ~17 ms and a 1080p session touches tens of megabytes per
    frame; allocator traffic alone would eat it.
    """

    width: int
    height: int
    out_width: int
    out_height: int
    dye: np.ndarray
    vel_base: np.ndarray
    stir: np.ndarray
    base: np.ndarray
    xx: np.ndarray
    yy: np.ndarray
    sx: np.ndarray = field(repr=False)
    sy: np.ndarray = field(repr=False)
    fx: np.ndarray = field(repr=False)
    fy: np.ndarray = field(repr=False)
    out: np.ndarray = field(repr=False)
    tmp: np.ndarray = field(repr=False)
    vel: np.ndarray = field(repr=False)
    ix0: np.ndarray = field(repr=False)
    iy0: np.ndarray = field(repr=False)
    ix1: np.ndarray = field(repr=False)
    iy1: np.ndarray = field(repr=False)
    small_u8: np.ndarray = field(repr=False)
    mix: np.ndarray = field(repr=False)
    sharp: np.ndarray = field(repr=False)


def _downscale(base_hi: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit a still to simulation resolution, once per session.

    Larger stills are block-averaged (exact); smaller ones — a 96 px test
    preview, for example — are bilinear-upscaled, since averaging a smaller
    image into a larger grid is empty. Either way the dye starts from the
    still's own pixels, never from blank.
    """
    base = np.asarray(base_hi, dtype=np.float32)
    h, w = base.shape[:2]
    if h >= height and w >= width:
        th, tw = (h // height) * height, (w // width) * width
        blocks = base[:th, :tw].reshape(height, th // height, width, tw // width, 3)
        return blocks.mean(axis=(1, 3)).astype(np.float32)
    from PIL import Image

    picture = Image.fromarray(
        np.clip(base * 255.0, 0, 255).astype(np.uint8)
    ).resize((width, height), Image.BILINEAR)
    return np.asarray(picture, dtype=np.float32) / 255.0


def build_fluid(
    base_hi: np.ndarray,
    params: dict,
    seed: int,
    out_width: int,
    out_height: int,
    ambient_speed: float = 1.5,
    sim_scale: float = SIM_SCALE,
) -> FluidState:
    """Assemble simulation state from the still every session starts from.

    The ambient flow reuses the flowfield angle field at simulation size, so
    live motion rhymes with still structure for flowfield recipes; other
    recipes get the same smooth noise flow regardless of generator.
    `sim_scale` selects a rung of the fidelity ladder (see `SIM_LEVELS`).
    """
    sim_w = max(160, int(out_width * sim_scale))
    sim_h = max(90, int(out_height * sim_scale))
    base = _downscale(np.asarray(base_hi, dtype=np.float32), sim_w, sim_h)
    h, w = base.shape[:2]
    field = build_field(params, seed, w, h)
    vel_base = np.stack([field.cos_a, field.sin_a], axis=-1).astype(np.float32)
    vel_base *= ambient_speed
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    shape3 = (h, w, 3)
    return FluidState(
        width=w,
        height=h,
        out_width=out_width,
        out_height=out_height,
        dye=base.copy(),
        vel_base=vel_base,
        stir=np.zeros((h, w, 2), dtype=np.float32),
        base=base,
        xx=xx,
        yy=yy,
        sx=np.empty((h, w), np.float32),
        sy=np.empty((h, w), np.float32),
        fx=np.empty((h, w, 1), np.float32),
        fy=np.empty((h, w, 1), np.float32),
        out=np.empty(shape3, np.float32),
        tmp=np.empty(shape3, np.float32),
        vel=np.empty((h, w, 2), np.float32),
        ix0=np.empty((h, w), np.int64),
        iy0=np.empty((h, w), np.int64),
        ix1=np.empty((h, w), np.int64),
        iy1=np.empty((h, w), np.int64),
        small_u8=np.empty((h, w, 3), np.uint8),
        mix=np.empty(shape3, np.float32),
        sharp=np.empty(shape3, np.float32),
    )


def stir(
    state: FluidState,
    cx: float,
    cy: float,
    vx: float,
    vy: float,
    radius: float,
    strength: float,
    mode: str,
) -> None:
    """Paint cursor motion into the stir layer: a jet plus a radial push.

    The jet follows the pointer's own velocity, so fast flicks throw fluid
    aside and leave a wake; the radial term pushes out (`repel`) or pulls in
    (`attract`) around the pointer itself. `off` adds nothing. Only the
    bounding box is touched, so an idle cursor costs nothing.
    """
    if mode == "off" or strength <= 0.0 or radius <= 0.0:
        return
    x0 = max(0, int(cx - radius))
    x1 = min(state.width, int(cx + radius) + 1)
    y0 = max(0, int(cy - radius))
    y1 = min(state.height, int(cy + radius) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xs - cx
    dy = ys - cy
    dist = np.sqrt(dx * dx + dy * dy)
    fall = np.clip(1.0 - dist / radius, 0.0, 1.0).astype(np.float32)
    speed = float(np.sqrt(vx * vx + vy * vy))
    if speed > 1e-6:
        jet_x, jet_y = vx / speed, vy / speed
    else:
        jet_x, jet_y = 0.0, 0.0
    push = 1.0 if mode == "repel" else -1.0
    region = state.stir[y0:y1, x0:x1]
    region[..., 0] += (strength * fall * (jet_x + push * dx / np.maximum(dist, 1e-6))).astype(np.float32)
    region[..., 1] += (strength * fall * (jet_y + push * dy / np.maximum(dist, 1e-6))).astype(np.float32)
    np.clip(state.stir, -60.0, 60.0, out=state.stir)


def advance(state: FluidState, relax: float, dt: float = 1.0) -> None:
    """One frame: advect dye through ambient plus stir, relax toward base.

    `dt` is frame duration in sixtieths (1.0 at 60 fps): ambient velocity,
    stir decay, and the relax pull all scale with it, so wall-clock motion
    is identical at any frame rate. At `dt == 1.0` this matches the unscaled
    loop up to floating-point rounding.

    Every temporary lands in a preallocated buffer (see `FluidState`): the
    index floors go to `ix0/iy0` by truncating assignment (values are clipped
    non-negative first, so truncation is flooring - identical to the old
    `astype`), the +1 neighbors to `ix1/iy1`, and the relax blend into
    `tmp`. The four bilinear gathers are the only large reads and are
    inherent to backward advection.
    """
    # Ambient scales with dt; cursor-painted stir does not (pointer deltas
    # already grow with slower frames, which keeps paint per second steady).
    np.multiply(state.vel_base, dt, out=state.vel)
    np.add(state.vel, state.stir, out=state.vel)
    vel = state.vel
    np.subtract(state.xx, vel[..., 0], out=state.sx)
    np.clip(state.sx, 0, state.width - 1.001, out=state.sx)
    np.subtract(state.yy, vel[..., 1], out=state.sy)
    np.clip(state.sy, 0, state.height - 1.001, out=state.sy)
    x0, y0, x1, y1 = state.ix0, state.iy0, state.ix1, state.iy1
    x0[:] = state.sx
    y0[:] = state.sy
    np.subtract(state.sx, x0, out=state.fx[..., 0])
    np.subtract(state.sy, y0, out=state.fy[..., 0])
    np.add(x0, 1, out=x1)
    np.minimum(x1, state.width - 1, out=x1)
    np.add(y0, 1, out=y1)
    np.minimum(y1, state.height - 1, out=y1)
    dye, fx, fy, out, tmp = state.dye, state.fx, state.fy, state.out, state.tmp
    a = dye[y0, x0]
    b = dye[y0, x1]
    c = dye[y1, x0]
    d = dye[y1, x1]
    np.subtract(b, a, out=out)
    np.multiply(out, fx, out=out)
    np.add(out, a, out=out)
    np.subtract(c, a, out=tmp)
    np.multiply(tmp, fy, out=tmp)
    np.add(out, tmp, out=out)
    np.subtract(a, b, out=tmp)
    np.subtract(tmp, c, out=tmp)
    np.add(tmp, d, out=tmp)
    np.multiply(tmp, fx, out=tmp)
    np.multiply(tmp, fy, out=tmp)
    np.add(out, tmp, out=out)
    if relax > 0.0:
        # Exact per-second form: at dt == 1 this is the old `relax` exactly.
        pull = 1.0 - (1.0 - relax) ** dt
        np.subtract(state.base, out, out=tmp)
        np.multiply(tmp, pull, out=tmp)
        np.add(out, tmp, out=out)
    state.dye, state.out = out, dye
    np.multiply(state.stir, 0.97**dt, out=state.stir)


def _pack(
    state: FluidState,
    src: np.ndarray,
    glow: tuple[float, float, float, float] | None = None,
    sharpen: float = 0.0,
) -> bytes:
    """Quantize a sim-res float frame and bilinear-upscale for display.

    `src` is the dye (or a crossfade blend of it). `glow` is an optional
    `(cx, cy, radius, strength)` lift in sim pixels: ink near the pointer
    brightens up to `1 + strength`, far ink is untouched. `sharpen` is an
    optional 0..1 unsharp amount countering upscale softness. Quantization
    reuses the `tmp` scratch and the `small_u8` buffer; glow reuses the
    `sx`/`sy` index scratch and sharpen the `sharp` buffer, all free outside
    `advance` — so packing still allocates nothing per frame.
    """
    from PIL import Image

    np.multiply(src, 255.0, out=state.tmp)
    if sharpen > 0.0:
        _apply_sharpen(state, sharpen)
    if glow is not None:
        _apply_glow(state, glow)
    np.clip(state.tmp, 0, 255, out=state.tmp)
    np.copyto(state.small_u8, state.tmp, casting="unsafe")
    big = Image.fromarray(state.small_u8).resize(
        (state.out_width, state.out_height), Image.BILINEAR
    )
    return big.tobytes()


def _apply_sharpen(state: FluidState, amount: float) -> None:
    """Unsharp mask on the quantized scratch, in sim space, in place.

    A 4-neighbor box blur into `sharp`, then detail back into `tmp`. Runs
    where the pixels are (sim res, ~1 ms), not where they display: full-res
    sharpening would cost more than the upscale it compensates. Borders copy
    through unsharpened; there is no neighbor to compare them against.
    """
    tmp, sharp = state.tmp, state.sharp
    np.add(tmp[1:-1, :-2, :], tmp[1:-1, 2:, :], out=sharp[1:-1, 1:-1, :])
    np.add(sharp[1:-1, 1:-1, :], tmp[:-2, 1:-1, :], out=sharp[1:-1, 1:-1, :])
    np.add(sharp[1:-1, 1:-1, :], tmp[2:, 1:-1, :], out=sharp[1:-1, 1:-1, :])
    sharp[1:-1, 1:-1, :] /= 4.0
    sharp[0, :, :] = tmp[0, :, :]
    sharp[-1, :, :] = tmp[-1, :, :]
    sharp[:, 0, :] = tmp[:, 0, :]
    sharp[:, -1, :] = tmp[:, -1, :]
    np.subtract(tmp, sharp, out=sharp)
    np.multiply(sharp, amount, out=sharp)
    np.add(tmp, sharp, out=tmp)


def _apply_glow(
    state: FluidState, glow: tuple[float, float, float, float]
) -> None:
    """Multiply a radial brightness lift into the quantized scratch in place."""
    cx, cy, radius, strength = glow
    if strength <= 0.0 or radius <= 0.0:
        return
    np.subtract(state.xx, cx, out=state.sx)
    np.subtract(state.yy, cy, out=state.sy)
    np.square(state.sx, out=state.sx)
    np.square(state.sy, out=state.sy)
    np.add(state.sx, state.sy, out=state.sx)
    np.sqrt(state.sx, out=state.sx)
    state.sx /= radius
    np.subtract(1.0, state.sx, out=state.sx)
    np.clip(state.sx, 0.0, 1.0, out=state.sx)
    np.multiply(state.sx, strength, out=state.sx)
    np.add(state.sx, 1.0, out=state.sx)
    np.multiply(state.tmp, state.sx[..., None], out=state.tmp)


def frame_bytes(
    state: FluidState,
    glow: tuple[float, float, float, float] | None = None,
    sharpen: float = 0.0,
) -> bytes:
    """Quantize at sim res and bilinear-upscale for display, via Pillow."""
    return _pack(state, state.dye, glow, sharpen)


def crossfade_bytes(
    state: FluidState,
    base: np.ndarray,
    t: float,
    glow: tuple[float, float, float, float] | None = None,
    sharpen: float = 0.0,
) -> bytes:
    """One handover frame: `base` dissolving into the live dye by `t` in 0..1.

    `base` is the downscaled still the session started from. At `t == 0` this
    is the still; at `t == 1` it is byte-identical to `frame_bytes(state)`.
    The blend runs into the `mix` scratch so neither input is disturbed.
    """
    clamped = min(1.0, max(0.0, t))
    np.multiply(state.dye, clamped, out=state.mix)
    np.multiply(base, 1.0 - clamped, out=state.tmp)
    np.add(state.mix, state.tmp, out=state.mix)
    return _pack(state, state.mix, glow, sharpen)

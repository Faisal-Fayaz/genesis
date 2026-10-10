"""The live simulation core: field, fluid dye, cursor, upload.

These run headlessly at small sizes. They assert the properties the frame
budget depends on: the field is built once and cached, cursor stir is bounded,
relaxation anchors the dye to the still, and an idle session costs a fixed
small amount of work per frame.
"""

from __future__ import annotations

import numpy as np

from genesis.live import field as field_module
from genesis.live import fluid as fluid_module
from genesis.live.config import LiveConfig
from genesis.live.service import field_params
from genesis.recipe import Recipe


def flowfield_params(**overrides):
    params = {
        "scale": 1.5,
        "detail": 3,
        "turns": 3.0,
        "swirl": True,
        "step_size": 0.004,
    }
    params.update(overrides)
    return params


def striped_base(width: int = 64, height: int = 36) -> np.ndarray:
    """Left half near-white, right half near-black: structure with a plan."""
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:, : width // 2] = (0.9, 0.9, 0.9)
    base[:, width // 2 :] = (0.02, 0.02, 0.02)
    return base


def make_fluid(width: int = 320, height: int = 200, seed: int = 7):
    """A small fluid session state on a striped still."""
    base = striped_base(width, height)
    return fluid_module.build_fluid(base, flowfield_params(), seed, width, height)


def test_field_params_read_only_field_shaping_keys() -> None:
    recipe = Recipe(
        generator="flowfield",
        params={"trails": 12000, "steps": 220, "supersample": 2, **flowfield_params()},
        width=1920,
        height=1080,
        seed=3,
    )
    params = field_params(recipe)
    assert sorted(params) == ["detail", "scale", "step_size", "swirl", "turns"]


def test_field_is_cached_and_clamped() -> None:
    field = field_module.build_field(flowfield_params(), seed=3, width=320, height=180)
    assert field.cos_a.shape == (180, 320)
    assert field.sin_a.shape == (180, 320)
    assert field.cos_a.dtype == np.float32
    # A step that turns gently is left alone; the clamp only fires on tight fields.
    assert field.step_x <= 0.004 * 320 + 1e-9


def test_tight_field_clamps_the_step() -> None:
    gentle = field_module.build_field(flowfield_params(), seed=3, width=320, height=180)
    tight = field_module.build_field(
        flowfield_params(scale=13.8, turns=6.13), seed=3, width=320, height=180
    )
    assert tight.gradient > gentle.gradient
    assert tight.step_x < gentle.step_x


def test_build_fluid_starts_from_the_still() -> None:
    state = make_fluid()
    assert state.dye.shape == (state.height, state.width, 3)
    assert state.dye.dtype == np.float32
    assert state.stir.shape == (state.height, state.width, 2)
    assert bool((state.stir == 0.0).all())
    # The dye is the still downscaled to sim resolution, not a blank canvas.
    assert float(state.dye[..., 0].mean()) > 0.2


def test_build_fluid_accepts_a_smaller_still() -> None:
    """A preview smaller than the sim grid upscales instead of going blank.

    Block-averaging a 96 px preview into a 160-wide grid used to average
    empty slices: NaN dye, black frames, no error anywhere.
    """
    base = striped_base(96, 54)
    state = fluid_module.build_fluid(base, flowfield_params(), 7, 320, 200)
    assert state.dye.shape == (state.height, state.width, 3)
    assert bool(np.isfinite(state.dye).all())
    assert float(state.dye[..., 0].mean()) > 0.2


def test_build_fluid_is_deterministic() -> None:
    first = make_fluid(seed=11)
    second = make_fluid(seed=11)
    assert np.array_equal(first.dye, second.dye)
    assert np.array_equal(first.vel_base, second.vel_base)


def test_stir_off_adds_nothing() -> None:
    state = make_fluid()
    before = state.stir.copy()
    fluid_module.stir(state, 80.0, 50.0, 3.0, 1.0, radius=40.0, strength=8.0, mode="off")
    assert np.array_equal(state.stir, before)
    fluid_module.stir(state, 80.0, 50.0, 3.0, 1.0, radius=40.0, strength=0.0, mode="repel")
    assert np.array_equal(state.stir, before)


def test_stir_is_bounded() -> None:
    state = make_fluid()
    for _ in range(50):
        fluid_module.stir(
            state, 80.0, 50.0, 5.0, 0.0, radius=60.0, strength=99.0, mode="repel"
        )
    assert bool((state.stir <= 60.0).all())
    assert bool((state.stir >= -60.0).all())
    # ...but a real stir still lands: the cursor leaves a mark.
    assert bool(np.abs(state.stir).max() > 0.0)


def test_repel_and_attract_push_opposite_ways() -> None:
    repelled = make_fluid()
    attracted = make_fluid()
    fluid_module.stir(
        repelled, 80.0, 50.0, 0.0, 0.0, radius=40.0, strength=8.0, mode="repel"
    )
    fluid_module.stir(
        attracted, 80.0, 50.0, 0.0, 0.0, radius=40.0, strength=8.0, mode="attract"
    )
    # With no pointer velocity only the radial term paints, mirrored by mode.
    assert np.allclose(repelled.stir, -attracted.stir, atol=1e-6)


def test_advance_moves_the_dye() -> None:
    state = make_fluid()
    before = state.dye.copy()
    fluid_module.advance(state, relax=0.0)
    assert not np.array_equal(state.dye, before)


def test_advance_is_deterministic() -> None:
    first = make_fluid(seed=5)
    second = make_fluid(seed=5)
    for _ in range(4):
        fluid_module.advance(first, relax=0.004)
        fluid_module.advance(second, relax=0.004)
    assert np.array_equal(first.dye, second.dye)


def test_relax_pulls_dye_back_toward_the_still() -> None:
    anchored = make_fluid(seed=9)
    drifting = make_fluid(seed=9)
    for _ in range(20):
        fluid_module.advance(anchored, relax=0.1)
        fluid_module.advance(drifting, relax=0.0)
    anchored_gap = np.abs(anchored.dye - anchored.base).mean()
    drifting_gap = np.abs(drifting.dye - drifting.base).mean()
    assert anchored_gap < drifting_gap


def test_frame_bytes_cover_the_output() -> None:
    state = make_fluid(width=320, height=200)
    data = fluid_module.frame_bytes(state)
    assert isinstance(data, bytes)
    assert len(data) == 320 * 200 * 3


def test_crossfade_endpoints_match_still_and_sim() -> None:
    first = make_fluid(seed=4)
    second = make_fluid(seed=4)
    for state in (first, second):
        for _ in range(5):
            fluid_module.advance(state, relax=0.004)
    base = first.base
    fresh = make_fluid(seed=4)  # dye still == base: what packing the still means
    # t=0 is the still packed; t=1 is byte-identical to a normal frame.
    assert fluid_module.crossfade_bytes(first, base, 0.0) == fluid_module.frame_bytes(fresh)
    assert fluid_module.crossfade_bytes(first, base, 1.0) == fluid_module.frame_bytes(first)
    assert fluid_module.crossfade_bytes(first, base, 0.5) != fluid_module.frame_bytes(first)
    # The handover never disturbs the simulation itself.
    assert np.array_equal(first.dye, second.dye)


def test_glow_off_leaves_bytes_untouched() -> None:
    state = make_fluid()
    plain = fluid_module.frame_bytes(state)
    assert fluid_module.frame_bytes(state, glow=(80.0, 50.0, 40.0, 0.0)) == plain
    assert fluid_module.frame_bytes(state, glow=None) == plain


def test_glow_lifts_nearby_ink_and_is_bounded() -> None:
    state = make_fluid()
    plain = np.frombuffer(
        fluid_module.frame_bytes(state), dtype=np.uint8
    ).reshape(200, 320, 3)
    lit = np.frombuffer(
        fluid_module.frame_bytes(state, glow=(40.0, 25.0, 40.0, 0.5)),
        dtype=np.uint8,
    ).reshape(200, 320, 3)
    diff = lit.astype(np.int16) - plain.astype(np.int16)
    assert diff.max() > 0  # ink under the cursor lifts
    assert diff.min() == 0  # far ink is byte-identical
    assert diff.max() <= 255 * 0.5 + 2  # bounded by strength, plus quant slop


def test_advance_scales_with_wall_clock_dt() -> None:
    """dt scaling keeps motion in the same ballpark across frame rates.

    Without it, halving the fps halves wall-clock speed exactly (a 6x swing
    across the governor's range). With it, 30 frames at dt=2 drift about as
    far as 60 at dt=1. Exact trajectory match is impossible by construction
    — curved flow integrates differently at 2x steps, and chaotic advection
    amplifies it — so this pins the drift ratio, not the pixels. The pixels
    stay exactly replayable at a fixed dt (below).
    """

    def smooth_base(width: int = 192, height: int = 108) -> np.ndarray:
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        field = 0.45 + 0.3 * np.sin(xx / 24.0) * np.cos(yy / 18.0)
        return np.stack([field, field * 0.9, field * 1.1], axis=-1).astype(np.float32)

    def drift(n: int, dt: float, seed: int = 7) -> float:
        base = smooth_base()
        state = fluid_module.build_fluid(base, flowfield_params(), seed, 192, 108)
        for _ in range(n):
            fluid_module.advance(state, relax=0.004, dt=dt)
        return float(np.abs(state.dye - state.base).mean())

    drift_60, drift_30 = drift(60, 1.0), drift(30, 2.0)
    assert drift_60 > 0.05  # the test actually moves dye
    assert 0.5 < drift_30 / drift_60 < 1.5


def test_advance_is_exactly_replayable_at_fixed_dt() -> None:
    base = striped_base(192, 108)
    first = fluid_module.build_fluid(base, flowfield_params(), 7, 192, 108)
    second = fluid_module.build_fluid(base, flowfield_params(), 7, 192, 108)
    for _ in range(10):
        fluid_module.advance(first, relax=0.004, dt=2.0)
        fluid_module.advance(second, relax=0.004, dt=2.0)
    assert np.array_equal(first.dye, second.dye)


def test_sharpen_lifts_edge_contrast_without_touching_flats() -> None:
    base = np.zeros((360, 640, 3), dtype=np.float32)
    base[:, :320] = (0.9, 0.9, 0.9)
    base[:, 320:] = (0.02, 0.02, 0.02)
    state = fluid_module.build_fluid(base, flowfield_params(), 7, 640, 360)
    plain = np.frombuffer(
        fluid_module.frame_bytes(state), dtype=np.uint8
    ).reshape(360, 640, 3).astype(float)
    sharp = np.frombuffer(
        fluid_module.frame_bytes(state, sharpen=0.6), dtype=np.uint8
    ).reshape(360, 640, 3).astype(float)
    edge = lambda frame: np.abs(np.diff(frame[:, 310:330, :], axis=1)).mean()
    assert edge(sharp) > edge(plain) * 1.1  # the boundary steepens
    flat = np.abs(plain[:, 50:150, :] - sharp[:, 50:150, :]).mean()
    assert flat == 0.0  # no edge nearby: byte-identical
    assert fluid_module.frame_bytes(state, sharpen=0.0) == fluid_module.frame_bytes(state)


def test_cursor_tracker_smooths_and_detects_idle() -> None:
    tracker = fluid_module.CursorTracker()
    assert tracker.position() is None
    tracker.observe(100.0, 100.0, frame=0)
    tracker.observe(200.0, 100.0, frame=60)
    x, _ = tracker.position()
    assert 100.0 < x < 200.0
    assert tracker.idle_seconds(60 + 120, fps=60) == 2.0


def test_live_config_frame_budget_matches_fps() -> None:
    assert LiveConfig(fps=30).frame_budget == 1 / 30


def test_relaxed_session_stays_near_the_still() -> None:
    """Ambient motion must not homogenize the dye into mush.

    With the still as the relax target, the worst case is the still itself:
    300 frames of ambient flow stay within a band of the base mean, and a
    stirred session with relax stays closer than one without.
    """
    from genesis import presets as presets_module

    recipe = presets_module.get("silk").recipe(1920, 1080)
    base = striped_base(480, 270)
    ambient = fluid_module.build_fluid(
        base, field_params(recipe), seed=7, out_width=480, out_height=270
    )
    for _ in range(300):
        fluid_module.advance(ambient, relax=0.004)
    assert bool(np.isfinite(ambient.dye).all())
    assert abs(float(ambient.dye.mean() - ambient.base.mean())) < 0.1

    anchored = fluid_module.build_fluid(
        base, field_params(recipe), seed=7, out_width=480, out_height=270
    )
    drifting = fluid_module.build_fluid(
        base, field_params(recipe), seed=7, out_width=480, out_height=270
    )
    for _ in range(60):
        for state in (anchored, drifting):
            fluid_module.stir(
                state, 120.0, 67.0, 2.0, 1.0, radius=60.0, strength=8.0,
                mode="repel",
            )
        fluid_module.advance(anchored, relax=0.1)
        fluid_module.advance(drifting, relax=0.0)
    anchored_gap = np.abs(anchored.dye - anchored.base).mean()
    drifting_gap = np.abs(drifting.dye - drifting.base).mean()
    assert anchored_gap < drifting_gap

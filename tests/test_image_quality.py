"""Every generator must produce a picture, not a void.

The determinism tests confirm that a render repeats, which says nothing about
whether the output is any good. Every visual defect found during development -
flat fields, edge banding, pure-black output, structure painted in the
background color - passed the whole suite, because the values were deterministic
and wrong.

These assertions encode what "a picture" means here. They are deliberately
about *composition and coverage* rather than appearance, so they do not fail
every time a default is retuned.
"""

from __future__ import annotations

import numpy as np
import pytest

from genesis import render
from genesis.recipe import Recipe

# How much brighter the busiest interior line may be than the average one.
# These attractors are sparse curves and are legitimately uneven, so this is
# generous; banding produced ratios far above it.
MAX_LINE_CONCENTRATION = 3.0

# A large share of pure-white pixels means the tone response is clipping rather
# than rendering structure.
MAX_CLIPPED_SHARE = 0.02

# Standard deviation floor, in 8-bit levels. A blank or near-uniform canvas sits
# near 0; anything with real structure clears this easily.
MIN_CONTRAST = 4.0

GENERATORS = ["attractor", "flowfield", "splash"]

# Splash films the whole canvas at any reasonable blob count (3-sigma skirts
# overlap everywhere), so its quiet areas are dim ink, not bare background.
# The floor contract below therefore covers only generators with visible
# background; splash pins the crush direction in a dedicated test instead.
BACKGROUND_GENERATORS = ["attractor", "flowfield"]

# The recipe that produced the coarse mottled noise this suite now guards
# against: a tightly turning field sampled with a step long enough to jump
# across the curvature it is meant to follow. Recorded verbatim so the failure
# it describes stays reproducible.
ALIASED_PARAMS: dict[str, object] = {
    "scale": 13.8,
    "detail": 5,
    "turns": 6.13,
    "step_size": 0.01206,
}

# A field that turns gently, where the guard must not engage. If it did, this
# would fail rather than quietly changing the curated look.
CLEAN_PARAMS: dict[str, object] = {
    "scale": 4.66,
    "detail": 5,
    "turns": 3.84,
    "step_size": 0.00443,
}


def quick(name: str, seed: int = 3, size: tuple[int, int] = (192, 108)) -> Recipe:
    """A small recipe that still exercises a generator's real behavior."""
    params: dict[str, object] = {"supersample": 1}
    if name == "attractor":
        # Clifford and a known-good coefficient set: the two attractor forms
        # whose output is verified to be well-formed.
        params.update({"form": "clifford", "points": 120000, "zoom": 0.5})
    elif name == "splash":
        # Few blobs, low detail: enough structure to judge coverage by,
        # cheap enough for a suite that renders every case twice.
        params.update({"blobs": 12, "detail": 2})
    else:
        params.update({"trails": 3000, "steps": 60})
    return Recipe(
        generator=name,
        params=params,
        width=size[0],
        height=size[1],
        seed=seed,
        palette_name="tide",
    )


def luminance(image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float64)


@pytest.fixture(params=GENERATORS)
def rendered(request):
    return render.render(quick(request.param))


@pytest.fixture(params=BACKGROUND_GENERATORS)
def rendered_on_background(request):
    return render.render(quick(request.param))


class TestImagesAreNotBlank:
    def test_contrast_is_above_the_floor(self, rendered) -> None:
        assert luminance(rendered.image).std() > MIN_CONTRAST

    def test_output_is_not_a_single_flat_color(self, rendered) -> None:
        """More than a handful of distinct values, however the palette is used.

        The floor is deliberately low: a small canvas traced with few samples
        legitimately yields a small number of flat regions, and the point here is
        only to reject a uniform fill.
        """
        data = np.asarray(rendered.image)
        assert len(np.unique(data.reshape(-1, data.shape[-1]), axis=0)) > 4

    def test_dimensions_are_exact(self, rendered) -> None:
        assert rendered.image.size == (192, 108)
        assert rendered.image.mode == "RGB"


class TestBackgroundIsThePaletteFloor:
    """Unpainted pixels must be the palette's darkest stop, not black.

    A contrast pivot in linear light can crush a dark palette color to pure
    black, which reads as a hole in the image rather than as its background.
    """

    def test_quiet_corner_matches_palette_floor(self, rendered_on_background) -> None:
        recipe = rendered_on_background.recipe
        palette = recipe.effective_palette()
        # Compare in sRGB, where the stored hex values live.
        from genesis.color import linear_to_srgb

        floor = np.round(linear_to_srgb(palette.to_linear(0.0)) * 255.0)

        # The darkest 5% of the frame is background by construction.
        flat = np.asarray(rendered_on_background.image, dtype=np.float64).reshape(-1, 3)
        darkest = flat[np.argsort(flat.sum(axis=1))[: max(len(flat) // 20, 1)]]
        # Nothing may sit below the palette floor: that would mean the grade
        # crushed the background to black.
        assert np.all(darkest >= floor - 6)
        # Nor may it drift far above, which would mean the background is being
        # lifted by something other than the palette.
        assert np.all(darkest.mean(axis=0) < floor + 24)


class TestSplashBackgroundNeverCrushes:
    """Splash's gain pivots around the palette floor, so no pixel may sit
    below it. The historical failure composited over black while gaining
    around the floor, which crushed untouched background to pure black
    whenever exposure exceeded 1 — a hole where the palette should be.
    Splash films the whole canvas at working blob counts, so the lift half
    of the contract above does not apply; this pins the crush half at a
    sparse config (background genuinely shows) and a dense one (no pixel
    anywhere may crush).
    """

    def test_sparse_render_keeps_the_floor(self) -> None:
        recipe = Recipe(
            generator="splash",
            params={
                "blobs": 8,
                "spread": 0.05,
                "turbulence": 0.0,
                "fray": 0.0,
                "detail": 1,
                "supersample": 1,
            },
            width=192,
            height=108,
            seed=11,
            palette_name="tide",
        )
        values = np.asarray(render.render(recipe).image, dtype=np.float64)
        from genesis.color import linear_to_srgb

        floor = np.round(
            linear_to_srgb(recipe.effective_palette().to_linear(0.0)) * 255.0
        )
        flat = values.reshape(-1, 3)
        assert np.all(flat.min(axis=0) >= floor - 6)
        assert float((np.abs(flat - floor).max(axis=1) <= 6).mean()) > 0.5

    def test_dense_render_crushes_nothing(self) -> None:
        recipe = quick("splash", seed=7)
        values = np.asarray(render.render(recipe).image, dtype=np.float64)
        from genesis.color import linear_to_srgb

        floor = np.round(
            linear_to_srgb(recipe.effective_palette().to_linear(0.0)) * 255.0
        )
        assert np.all(values.reshape(-1, 3).min(axis=0) >= floor - 6)


class TestNoEdgeBanding:
    """Samples clamped to the border pile up into axis-aligned bands."""

    @pytest.mark.parametrize("axis", ["rows", "cols"])
    def test_no_single_line_dominates(self, rendered, axis) -> None:
        """No interior row or column may hold a disproportionate share of the ink.

        Measured over interior lines only, since the border is covered by the
        dedicated outlier test below. These attractors are sparse curves, so some
        concentration is inherent; the bound catches banding rather than
        ordinary unevenness.
        """
        values = luminance(rendered.image)
        profile = values.mean(axis=0 if axis == "rows" else 1)
        interior = profile[4:-4]
        if interior.size == 0 or interior.mean() <= 0:
            pytest.skip("image too small to have an interior")
        # A perfectly even frame gives exactly 1.0.
        assert float(interior.max() / interior.mean()) < MAX_LINE_CONCENTRATION

    def test_edge_lines_are_not_outliers(self, rendered) -> None:
        """The outermost rows and columns must not be far brighter than the rest."""
        values = luminance(rendered.image)
        inner = values[4:-4, 4:-4]
        if inner.size == 0:
            pytest.skip("image too small to have an interior")
        border = np.concatenate([values[0], values[-1], values[:, 0], values[:, -1]])
        # A clamped band would put the border well above the interior mean.
        assert border.mean() < inner.mean() + 2.0 * (inner.std() + 1e-9)


class TestNoClipping:
    def test_pure_white_is_rare(self, rendered) -> None:
        values = np.asarray(rendered.image)
        clipped = float((values.min(axis=2) >= 254).mean())
        assert clipped < MAX_CLIPPED_SHARE

    def test_pure_black_is_rare(self, rendered) -> None:
        values = np.asarray(rendered.image)
        clipped = float((values.max(axis=2) <= 1).mean())
        assert clipped < 0.5


class TestStructureIsInkNotBackground:
    """Some pixels must differ from the background, by a visible margin.

    Coloring strokes with the same color as the field they sit on produces an
    image whose histogram looks fine and whose content is invisible.
    """

    def test_a_meaningful_share_differs_from_the_background(self, rendered) -> None:
        from genesis.color import linear_to_srgb

        palette = rendered.recipe.effective_palette()
        floor = linear_to_srgb(palette.to_linear(0.0)) * 255.0
        values = np.asarray(rendered.image, dtype=np.float64)
        distance = np.abs(values - floor).sum(axis=2)
        assert float((distance > 24).mean()) > 0.05


class TestPreviewMatchesFullSize:
    """A preview must look like the same composition, not a different one.

    This is what makes parameter tuning affordable: if brightness shifts between
    sizes, every tuned value has to be re-tuned at full resolution.
    """

    @pytest.mark.parametrize("name", GENERATORS)
    def test_coverage_survives_resolution_change(self, name: str) -> None:
        small = render.render(quick(name, size=(192, 108)))
        large = render.render(quick(name, size=(576, 324)))

        def coverage(image) -> float:
            from genesis.color import linear_to_srgb

            palette = image.recipe.effective_palette()
            floor = linear_to_srgb(palette.to_linear(0.0)) * 255.0
            values = np.asarray(image.image, dtype=np.float64)
            return float((np.abs(values - floor).sum(axis=2) > 24).mean())

        small_share = coverage(small)
        large_share = coverage(large)
        # The same recipe should cover a comparable fraction of the frame at both
        # sizes, so a value tuned on a preview carries over. Not equality: a
        # small canvas discretizes a curve more coarsely, and a sparse field
        # spreads differently. This bound is what an unscaled sample count or a
        # mis-scaled ink budget would fall outside of.
        assert 0.6 < (small_share / large_share) < 1.7


def jagged_fraction(
    scale: float,
    detail: int,
    turns: float,
    step_size: float,
    substeps: int,
    width: int = 960,
    height: int = 540,
    trails: int = 3000,
    steps: int = 60,
) -> float:
    """Share of trail segments that turn more than 45 degrees.

    Measured on the advection itself rather than on the rendered pixels, because
    a raster metric cannot separate "the trail corners" from "the field is
    legitimately high-frequency". The angular field is smooth by construction, so
    a sharp turn in a trail can only come from a step that jumped past the
    curvature it was supposed to follow.

    Runs one particle per trail and no rendering, which keeps it to a few
    thousand segments per second. Segments are only compared within a single
    uninterrupted run: a particle that leaves the canvas respawns somewhere
    unrelated, and treating that jump as a turn measures nothing but the respawn.
    """
    from genesis import noise
    from genesis.generators import flowfield
    from genesis.rng import stream

    aspect = height / width
    xs, ys = noise.coordinate_grid(width, height, aspect)
    angle = flowfield._angle_field(
        noise.permutation_table(stream(3, "field")),
        xs,
        ys,
        {"detail": detail, "scale": scale, "turns": turns, "swirl": True},
    )

    gradient = flowfield.angular_gradient(angle)
    clamped = min(step_size, flowfield.max_step_for_gradient(gradient, width))
    sub_step_x = clamped * width / substeps
    sub_step_y = clamped * height / substeps

    rng = stream(3, "particles")
    pos_x = rng.random(trails) * width
    pos_y = rng.random(trails) * height
    generation = np.zeros(trails, dtype=np.int64)

    previous: np.ndarray | None = None
    previous_generation: np.ndarray | None = None
    turns_seen: list[np.ndarray] = []

    for _ in range(steps):
        for _ in range(substeps):
            col = np.clip(pos_x.astype(np.int64), 0, width - 1)
            row = np.clip(pos_y.astype(np.int64), 0, height - 1)
            heading = angle[row, col]
            next_x = pos_x + np.cos(heading) * sub_step_x
            next_y = pos_y + np.sin(heading) * sub_step_y

            segment = np.stack([next_x - pos_x, next_y - pos_y], axis=-1)
            if previous is not None:
                same_run = generation == previous_generation
                if same_run.any():
                    a = previous[same_run]
                    b = segment[same_run]
                    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
                    turns_seen.append(np.abs(np.arctan2(cross, (a * b).sum(-1))))

            previous = segment
            previous_generation = generation
            pos_x, pos_y = next_x, next_y

        escaped = (pos_x < 0) | (pos_x >= width) | (pos_y < 0) | (pos_y >= height)
        if escaped.any():
            pos_x = np.where(escaped, rng.random(trails) * width, pos_x)
            pos_y = np.where(escaped, rng.random(trails) * height, pos_y)
            generation = generation + escaped

    observed = np.concatenate(turns_seen)
    assert observed.size > 0
    return float((observed > np.pi / 4).mean())


class TestFlowFieldTrailsFollowTheCurve:
    """A trail must follow the field, not jump across it.

    A step long enough to step over the curvature between two samples turns the
    trail into a zigzag whose frequency comes from `step_size` rather than from
    the field. The output is still deterministic, still has contrast, and still
    passes every structural check, which is why this is asserted on the
    trajectory instead of on pixels.
    """

    def test_aliased_recipe_is_split_into_smoother_segments(self) -> None:
        """The recipe that produced mottled noise, at the count auto chooses.

        Without the clamp and the split this recipe turns more than 45 degrees
        on 39% of its segments, and the trail becomes a diagonal hatch whose
        frequency comes from `step_size` rather than from the field.

        The assertion is on the improvement, not on an absolute fraction,
        because there is a floor that splitting cannot reach: the angle field is
        sampled at integer pixel positions, so a particle crossing less than a
        pixel per segment still snaps to whichever cell it lands in. Measured at
        960x540 the fraction goes 0.387 to 0.220 to 0.157 for 1, 2 and 4 samples,
        and then flattens out around 0.11 even at 16. What matters is that the
        failure mode is substantially gone, and the bound below is set from the
        measured 4-sample value rather than from an idealized zero.
        """
        unsplit = jagged_fraction(**ALIASED_PARAMS, substeps=1)
        split = jagged_fraction(**ALIASED_PARAMS, substeps=4)
        assert unsplit > 0.30
        assert split < 0.20
        # Halved, which is the actual claim being made about the fix.
        assert split < unsplit * 0.55

    def test_the_split_helps_a_gentle_field_too(self) -> None:
        """The fix is not only a safety net for pathological recipes.

        The curated recipe that motivated this started at 0.0173 and reaches
        0.0006 with two samples, so auto-sub-stepping is what lets it be smooth
        without lowering its step size and losing its fine detail.
        """
        assert jagged_fraction(**CLEAN_PARAMS, substeps=1) > 0.005
        assert jagged_fraction(**CLEAN_PARAMS, substeps=2) < 0.005

    def test_a_gentle_field_needs_only_one_sample(self) -> None:
        """One sample is enough where the field turns slowly.

        This is what makes the auto count free in the common case: if a gentle
        field needed extra samples, every curated recipe would get slower for
        nothing.
        """
        assert jagged_fraction(**CLEAN_PARAMS, substeps=1) < 0.02


class TestStepGuardEngagesOnlyWhenNeeded:
    """The clamp is a safety net, so it must stay out of the way otherwise."""

    def _gradient(self, params: dict[str, object], width: int = 960, height: int = 540):
        from genesis import noise
        from genesis.generators import flowfield
        from genesis.rng import stream

        xs, ys = noise.coordinate_grid(width, height, height / width)
        angle = flowfield._angle_field(
            noise.permutation_table(stream(3, "field")),
            xs,
            ys,
            {
                "detail": int(params["detail"]),
                "scale": float(params["scale"]),
                "turns": float(params["turns"]),
                "swirl": True,
            },
        )
        return flowfield.angular_gradient(angle)

    def test_aliased_recipe_is_clamped(self) -> None:
        from genesis.generators import flowfield

        gradient = self._gradient(ALIASED_PARAMS)
        allowed = flowfield.max_step_for_gradient(gradient, 960)
        assert float(ALIASED_PARAMS["step_size"]) > allowed

    def test_gentle_recipe_is_left_alone(self) -> None:
        from genesis.generators import flowfield

        gradient = self._gradient(CLEAN_PARAMS)
        allowed = flowfield.max_step_for_gradient(gradient, 960)
        assert float(CLEAN_PARAMS["step_size"]) <= allowed

    def test_defaults_are_left_alone(self) -> None:
        """The curated defaults must survive the guard unchanged.

        Golden hashes would catch a changed default digest, but only for the one
        case that happens to be recorded. This asserts the property directly.
        """
        from genesis.generators import flowfield

        spec = next(s for s in flowfield.SCHEMA if s.name == "step_size")
        gradient = self._gradient(
            {"detail": 5, "scale": float(spec.default) / 0.001, "turns": 3.0}
        )
        allowed = flowfield.max_step_for_gradient(gradient, 960)
        assert float(spec.default) <= allowed


class TestFlowFieldContrastIsLive:
    """`contrast` must actually change the rendered image.

    This parameter existed in the schema without being read by the renderer, so
    changing it from minimum to maximum left the digest untouched. Two distinct
    digests at either side of the default prove the control is wired through.
    """

    def test_contrast_changes_output(self) -> None:
        base = quick("flowfield")
        digests = {
            render.render(
                base.with_changes(
                    params={**base.params, "contrast": value}
                )
            ).digest
            for value in (0.5, 1.0, 2.0)
        }
        assert len(digests) == 3


class TestSubstepsAreDerivedFromCurvature:
    def test_auto_picks_one_for_a_gentle_field(self) -> None:
        from genesis.generators import flowfield

        # 0.034 radians per step, well inside one sample's budget.
        assert flowfield.resolve_substeps(0, 1.92, 0.0179) == 1

    def test_auto_splits_a_tight_field(self) -> None:
        from genesis.generators import flowfield

        assert flowfield.resolve_substeps(0, 8.5, 0.0485) >= 3

    def test_auto_is_capped(self) -> None:
        from genesis.generators import flowfield

        # Even a field that turns 40 radians per step stays within the cap,
        # because past the cap the samples refine nothing visible.
        assert flowfield.resolve_substeps(0, 1920.0, 1.0) == flowfield.MAX_AUTO_SUBSTEPS

    def test_an_explicit_request_wins(self) -> None:
        from genesis.generators import flowfield

        # A deliberate 1 must not be raised to the auto count, or there would be
        # no way to ask for the cheaper single-sample render.
        assert flowfield.resolve_substeps(1, 8.5, 0.0485) == 1
        assert flowfield.resolve_substeps(6, 1.92, 0.0179) == 6


class TestSeedsProduceDifferentPictures:
    def test_different_seeds_differ_visually(self) -> None:
        for name in GENERATORS:
            first = luminance(render.render(quick(name, seed=3)).image)
            second = luminance(render.render(quick(name, seed=4)).image)
            assert not np.array_equal(first, second), f"{name} ignored the seed"

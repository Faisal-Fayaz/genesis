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

GENERATORS = ["attractor", "flowfield"]


def quick(name: str, seed: int = 3, size: tuple[int, int] = (192, 108)) -> Recipe:
    """A small recipe that still exercises a generator's real behavior."""
    params: dict[str, object] = {"supersample": 1}
    if name == "attractor":
        # Clifford and a known-good coefficient set: the two attractor forms
        # whose output is verified to be well-formed.
        params.update({"form": "clifford", "points": 120000, "zoom": 0.5})
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

    def test_quiet_corner_matches_palette_floor(self, rendered) -> None:
        recipe = rendered.recipe
        palette = recipe.effective_palette()
        # Compare in sRGB, where the stored hex values live.
        from genesis.color import linear_to_srgb

        floor = np.round(linear_to_srgb(palette.to_linear(0.0)) * 255.0)

        # The darkest 5% of the frame is background by construction.
        flat = np.asarray(rendered.image, dtype=np.float64).reshape(-1, 3)
        darkest = flat[np.argsort(flat.sum(axis=1))[: max(len(flat) // 20, 1)]]
        # Nothing may sit below the palette floor: that would mean the grade
        # crushed the background to black.
        assert np.all(darkest >= floor - 6)
        # Nor may it drift far above, which would mean the background is being
        # lifted by something other than the palette.
        assert np.all(darkest.mean(axis=0) < floor + 24)


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


class TestSeedsProduceDifferentPictures:
    def test_different_seeds_differ_visually(self) -> None:
        for name in GENERATORS:
            first = luminance(render.render(quick(name, seed=3)).image)
            second = luminance(render.render(quick(name, seed=4)).image)
            assert not np.array_equal(first, second), f"{name} ignored the seed"

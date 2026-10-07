"""Procedural noise fields.

The property that matters most is tiling: a wallpaper may be shown on a wider
monitor than the one it was rendered for, so the field must repeat without a
seam. A field that does not tile produces a visible edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from genesis import noise
from genesis.rng import stream


def perm(seed: int = 7):
    return noise.permutation_table(stream(seed, "field"))


def grid(width: int = 64, height: int = 36):
    return noise.coordinate_grid(width, height, height / width)


class TestPermutationTable:
    def test_has_the_doubled_length_gradients_index(self) -> None:
        assert len(perm()) == 512

    def test_is_two_identical_halves(self) -> None:
        table = perm()
        assert np.array_equal(table[:256], table[256:])

    def test_is_a_permutation(self) -> None:
        assert sorted(perm()[:256].tolist()) == list(range(256))

    def test_is_reproducible(self) -> None:
        assert np.array_equal(perm(5), perm(5))

    def test_differs_between_seeds(self) -> None:
        assert not np.array_equal(perm(5), perm(6))


class TestPerlin:
    def test_stays_in_the_documented_range(self) -> None:
        xs, ys = grid(128, 96)
        values = noise.perlin_2d(perm(), xs, ys, 4.0)
        assert values.min() >= -1.5
        assert values.max() <= 1.5

    def test_is_reproducible(self) -> None:
        xs, ys = grid()
        first = noise.perlin_2d(perm(), xs, ys, 4.0)
        second = noise.perlin_2d(perm(), xs, ys, 4.0)
        assert np.array_equal(first, second)

    def test_wraps_at_the_lattice_period(self) -> None:
        """Sampling one full lattice period apart must give the same value.

        The wrap is at 256 *lattice units*, and `perlin_2d` multiplies the
        coordinate by frequency before indexing. So the period in coordinate
        space is `256 / frequency`, not 1.0. Getting this wrong is what puts a
        visible seam down a tiled wallpaper.
        """
        frequency = 4.0
        period = 256.0 / frequency
        for x, y in [(0.3, 0.5), (0.25, 0.75), (0.5, 0.1)]:
            a = noise.perlin_2d(perm(), np.array([x]), np.array([y]), frequency)
            b = noise.perlin_2d(
                perm(), np.array([x + period]), np.array([y]), frequency
            )
            assert np.allclose(a, b, atol=1e-9), f"seam at {x},{y}"

    @pytest.mark.parametrize("frequency", [1.0, 3.0, 8.0])
    def test_wrap_period_scales_with_frequency(self, frequency: float) -> None:
        period = 256.0 / frequency
        a = noise.perlin_2d(perm(), np.array([0.4]), np.array([0.6]), frequency)
        b = noise.perlin_2d(
            perm(), np.array([0.4 + period]), np.array([0.6]), frequency
        )
        assert np.allclose(a, b, atol=1e-9)

    def test_different_frequencies_differ(self) -> None:
        xs, ys = grid()
        low = noise.perlin_2d(perm(), xs, ys, 1.0)
        high = noise.perlin_2d(perm(), xs, ys, 16.0)
        assert not np.allclose(low, high)

    def test_has_actual_variation(self) -> None:
        xs, ys = grid()
        assert noise.perlin_2d(perm(), xs, ys, 4.0).std() > 0.05


class TestFbm:
    def test_is_reproducible(self) -> None:
        xs, ys = grid()
        first = noise.fbm(perm(), xs, ys, octaves=4, frequency=3.0)
        second = noise.fbm(perm(), xs, ys, octaves=4, frequency=3.0)
        assert np.array_equal(first, second)

    def test_output_stays_near_zero_mean(self) -> None:
        xs, ys = grid(128, 128)
        assert abs(float(noise.fbm(perm(), xs, ys).mean())) < 0.1

    def test_rejects_fewer_than_one_octave(self) -> None:
        xs, ys = grid()
        with pytest.raises(ValueError, match="octaves"):
            noise.fbm(perm(), xs, ys, octaves=0)

    def test_more_octaves_add_detail(self) -> None:
        xs, ys = grid()
        coarse = noise.fbm(perm(), xs, ys, octaves=1)
        fine = noise.fbm(perm(), xs, ys, octaves=7)
        assert fine.std() != coarse.std()

    def test_wraps_at_the_lattice_period(self) -> None:
        """Every octave must wrap, not just the first.

        Each octave multiplies the frequency by the lacunarity, so all of them
        wrap at the same coordinate distance only when the base period is used.
        """
        frequency = 2.0
        period = 256.0 / frequency
        a = noise.fbm(perm(), np.array([0.3]), np.array([0.7]), octaves=3, frequency=frequency)
        b = noise.fbm(
            perm(), np.array([0.3 + period]), np.array([0.7]), octaves=3, frequency=frequency
        )
        assert np.allclose(a, b, atol=1e-9)


class TestRidged:
    def test_is_bounded_to_zero_one(self) -> None:
        xs, ys = grid()
        values = noise.ridged(perm(), xs, ys)
        assert values.min() >= 0.0
        assert values.max() <= 1.0

    def test_is_reproducible(self) -> None:
        xs, ys = grid()
        first = noise.ridged(perm(), xs, ys)
        second = noise.ridged(perm(), xs, ys)
        assert np.array_equal(first, second)

    def test_has_sharp_crests(self) -> None:
        """Ridged noise exists to differ from plain fBm; check it does."""
        xs, ys = grid(128, 128)
        ridged = noise.ridged(perm(), xs, ys)
        # Crests mean a heavy concentration near the top of the range.
        assert float((ridged > 0.9).mean()) < 0.2


class TestCoordinateGrid:
    def test_spans_the_expected_range(self) -> None:
        xs, ys = noise.coordinate_grid(10, 10, 1.0)
        assert xs.min() == 0.0 and xs.max() == pytest.approx(1.0)
        assert ys.min() == 0.0 and ys.max() == pytest.approx(1.0)

    def test_y_is_stretched_by_aspect(self) -> None:
        _, ys = noise.coordinate_grid(10, 20, 2.0)
        assert ys.max() == pytest.approx(2.0)

    def test_shape_matches_the_canvas(self) -> None:
        xs, ys = noise.coordinate_grid(32, 16, 0.5)
        assert xs.shape == (16, 32)
        assert ys.shape == (16, 32)

    def test_tileable_aspect_reduces_to_whole_units(self) -> None:
        """A fractional aspect would break the lattice wrap."""
        assert noise.tileable_aspect(1920, 1080) == pytest.approx(16 / 9)
        assert noise.tileable_aspect(1000, 1000) == pytest.approx(1.0)
        assert noise.tileable_aspect(800, 600) == pytest.approx(4 / 3)

    def test_tileable_aspect_is_exact(self) -> None:
        """The reduced ratio must reproduce the canvas aspect precisely.

        A reduced fraction like 16/9 is not exactly representable in binary, so
        "exact" here means the reduction preserves the ratio to within float
        precision, which is all the lattice arithmetic needs.
        """
        for width, height in [(1920, 1080), (800, 600), (2560, 1440)]:
            value = noise.tileable_aspect(width, height)
            assert value == pytest.approx(width / height, rel=1e-15)

    def test_tileable_aspect_reduces_shared_factors(self) -> None:
        assert noise.tileable_aspect(1920, 1080) == pytest.approx(16 / 9)
        assert noise.tileable_aspect(800, 600) == pytest.approx(4 / 3)
        assert noise.tileable_aspect(1000, 1000) == pytest.approx(1.0)

    def test_tileable_aspect_survives_a_coprime_canvas(self) -> None:
        """A canvas with no shared factors still yields a usable ratio."""
        value = noise.tileable_aspect(1000, 667)
        assert np.isfinite(value)
        assert value == pytest.approx(1000 / 667, rel=1e-9)


class TestWorley:
    def test_is_non_negative(self) -> None:
        assert noise.worley(stream(1, "w"), 32, 32, count=20).min() >= 0.0

    def test_is_reproducible(self) -> None:
        first = noise.worley(stream(2, "w"), 32, 32, count=20)
        second = noise.worley(stream(2, "w"), 32, 32, count=20)
        assert np.array_equal(first, second)

    def test_rejects_zero_points(self) -> None:
        with pytest.raises(ValueError, match="count"):
            noise.worley(stream(3, "w"), 8, 8, count=0)

    def test_shape_matches_the_requested_canvas(self) -> None:
        assert noise.worley(stream(4, "w"), 17, 9, count=10).shape == (9, 17)


class TestNormalize:
    def test_maps_onto_zero_one(self) -> None:
        values = np.array([1.0, 5.0, 3.0])
        assert np.allclose(noise.normalize(values), [0.0, 1.0, 0.5])

    def test_flat_input_does_not_divide_by_zero(self) -> None:
        assert np.all(noise.normalize(np.full((3, 3), 7.0)) == 0.0)

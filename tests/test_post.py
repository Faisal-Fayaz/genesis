"""Post-processing.

Post runs in linear light on the generator's output, and converts to sRGB once
at the end. Getting that wrong is invisible in the numbers and obvious in the
picture: a grade pivoted on sRGB values crushes a dark palette to black.
"""

from __future__ import annotations

import numpy as np
import pytest

from genesis import post


def image(seed: int = 0, height: int = 48, width: int = 64) -> np.ndarray:
    """A smooth gradient with real tonal range, in linear light."""
    ys = np.linspace(0.02, 0.9, height)[:, None]
    xs = np.linspace(0.01, 0.7, width)[None, :]
    base = np.stack([ys * np.ones_like(xs), xs * np.ones_like(ys), ys + xs], axis=-1)
    return np.clip(base + seed * 0.01, 0.0, 1.0)


class TestResolve:
    def test_fills_in_defaults(self) -> None:
        settings = post.resolve({"grain": 0.05})
        assert settings["grain"] == 0.05
        assert settings["contrast"] == 1.0

    def test_empty_input_is_all_defaults(self) -> None:
        assert post.resolve({}) == post.SCHEMA.defaults()

    def test_rejects_unknown_options(self) -> None:
        with pytest.raises(ValueError, match="unknown post option"):
            post.resolve({"sepia": 1.0})

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="maximum"):
            post.resolve({"grain": 99.0})

    def test_rejects_non_mapping(self) -> None:
        with pytest.raises(TypeError, match="object"):
            post.resolve([("grain", 0.1)])

    def test_identity_detection(self) -> None:
        assert post.is_identity(post.resolve({}))
        assert not post.is_identity(post.resolve({"grain": 0.05}))


class TestApplyIsTransparentByDefault:
    def test_defaults_change_nothing(self) -> None:
        """An all-default chain must be a no-op, or every render is graded."""
        source = image()
        result = post.apply(source, 1, post.resolve({}))
        assert np.allclose(result, source, atol=1e-9)

    def test_output_stays_in_range(self) -> None:
        result = post.apply(image(), 1, post.resolve({"contrast": 2.5, "lift": 0.2}))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_shape_is_preserved(self) -> None:
        source = image()
        result = post.apply(source, 1, post.resolve({"grain": 0.1, "vignette": 0.4}))
        assert result.shape == source.shape

    def test_input_is_not_mutated(self) -> None:
        source = image()
        before = source.copy()
        post.apply(source, 1, post.resolve({"contrast": 1.5, "grain": 0.1}))
        assert np.array_equal(source, before)


class TestDeterminism:
    def test_same_seed_gives_identical_output(self) -> None:
        settings = post.resolve({"grain": 0.08, "bloom": 0.2})
        first = post.apply(image(), 99, settings)
        second = post.apply(image(), 99, settings)
        assert np.array_equal(first, second)

    def test_different_seed_changes_grain(self) -> None:
        """Grain must come from the recipe seed, not a global."""
        settings = post.resolve({"grain": 0.08})
        first = post.apply(image(), 1, settings)
        second = post.apply(image(), 2, settings)
        assert not np.array_equal(first, second)

    def test_grain_is_reproducible_across_calls(self) -> None:
        settings = post.resolve({"grain": 0.08})
        source = image()
        assert np.array_equal(
            post.apply(source, 7, settings), post.apply(source, 7, settings)
        )


class TestGrade:
    def test_brightness_scales(self) -> None:
        source = np.full((4, 4, 3), 0.5)
        result = post.apply(source, 1, post.resolve({"brightness": 1.5}))
        assert result.mean() == pytest.approx(0.75, rel=0.01)

    def test_contrast_pushes_away_from_mid(self) -> None:
        source = np.full((4, 4, 3), 0.6)
        result = post.apply(source, 1, post.resolve({"contrast": 2.0}))
        assert result.mean() > 0.6

    def test_saturation_zero_greys_the_image(self) -> None:
        source = np.stack(
            [np.full((8, 8), 0.8), np.full((8, 8), 0.2), np.full((8, 8), 0.2)], axis=-1
        )
        result = post.apply(source, 1, post.resolve({"saturation": 0.0}))
        channels = result.reshape(-1, 3)
        assert np.allclose(channels[:, 0], channels[:, 1])
        assert np.allclose(channels[:, 1], channels[:, 2])

    def test_saturation_one_is_a_no_op(self) -> None:
        source = image()
        result = post.apply(source, 1, post.resolve({"saturation": 1.0}))
        assert np.allclose(result, source, atol=1e-9)

    def test_lift_raises_the_black_point(self) -> None:
        source = np.zeros((4, 4, 3))
        result = post.apply(source, 1, post.resolve({"lift": 0.1}))
        assert result.mean() == pytest.approx(0.1, rel=0.01)


class TestVignette:
    def test_darkens_the_corners(self) -> None:
        source = np.ones((32, 32, 3)) * 0.8
        result = post.apply(source, 1, post.resolve({"vignette": 0.8}))
        assert result[0, 0].mean() < result[16, 16].mean()

    def test_leaves_the_centre_intact(self) -> None:
        source = np.ones((32, 32, 3)) * 0.8
        result = post.apply(source, 1, post.resolve({"vignette": 0.8}))
        assert result[16, 16].mean() == pytest.approx(0.8, rel=0.02)


class TestBloom:
    def test_brightens_near_bright_areas(self) -> None:
        source = np.zeros((32, 32, 3))
        source[16, 16] = 1.0
        result = post.apply(source, 1, post.resolve({"bloom": 0.5, "bloom_radius": 4.0}))
        assert result[14, 14].mean() > 0.0

    def test_leaves_dark_areas_alone(self) -> None:
        """Bloom is a highlight effect; a flat black frame must not glow."""
        source = np.zeros((32, 32, 3))
        result = post.apply(source, 1, post.resolve({"bloom": 0.5, "bloom_radius": 4.0}))
        assert result.max() == pytest.approx(0.0, abs=1e-9)


class TestGrain:
    def test_adds_variation(self) -> None:
        source = np.full((32, 32, 3), 0.5)
        result = post.apply(source, 1, post.resolve({"grain": 0.1}))
        assert result.std() > 0.0

    def test_is_stronger_in_midtones_than_in_blacks(self) -> None:
        """Film grain is weak in the toe; matching that keeps blacks clean."""
        settings = post.resolve({"grain": 0.12})
        black = post.apply(np.zeros((48, 48, 3)), 1, settings)
        midtone = post.apply(np.full((48, 48, 3), 0.5), 1, settings)
        assert midtone.std() > black.std()

    def test_grain_size_changes_the_texture_scale(self) -> None:
        fine = post.apply(image(), 3, post.resolve({"grain": 0.1, "grain_size": 1.0}))
        coarse = post.apply(image(), 3, post.resolve({"grain": 0.1, "grain_size": 4.0}))
        assert not np.array_equal(fine, coarse)


class TestBlur:
    def test_preserves_a_constant_image(self) -> None:
        flat = np.ones((16, 16, 3)) * 0.4
        assert np.allclose(post._blur(flat, 3), flat, atol=1e-9)

    def test_smooths_a_single_hot_pixel(self) -> None:
        source = np.zeros((16, 16, 3))
        source[8, 8] = 1.0
        result = post._blur(source, 2)
        assert result[8, 8].mean() < 1.0
        assert result[8, 9].mean() > 0.0

    def test_small_radius_is_skipped(self) -> None:
        source = np.ones((8, 8, 3))
        assert np.array_equal(post._blur(source, 0), source)

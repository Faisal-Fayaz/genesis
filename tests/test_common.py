"""Pixel-buffer helpers.

`scatter_add` is the one place in the codebase where a silent failure is
plausible: numpy will happily accept an assignment that writes to a temporary
view, discarding it without raising. Every case below is a regression test for
a variant of that mistake.
"""

from __future__ import annotations

import numpy as np
import pytest

from genesis.generators._common import downsample, scatter_add


def test_two_dimensional_buffer_accumulates() -> None:
    buffer = np.zeros((4, 4))
    scatter_add(buffer, np.array([0, 0, 5, 15]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert buffer.sum() == 10.0
    assert buffer[0, 0] == 3.0
    assert buffer[1, 1] == 3.0
    assert buffer[3, 3] == 4.0


def test_three_dimensional_buffer_accumulates_every_channel() -> None:
    buffer = np.zeros((4, 4, 3))
    scatter_add(buffer, np.array([0, 5]), np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    assert buffer.sum() == 9.0
    assert np.allclose(buffer[0, 0], [1.0, 1.0, 1.0])
    assert np.allclose(buffer[1, 1], [2.0, 2.0, 2.0])


def test_channels_are_independent() -> None:
    """A one-channel contribution must not leak into its neighbours."""
    buffer = np.zeros((4, 4, 3))
    scatter_add(buffer, np.array([0]), np.array([[1.0, 2.0, 3.0]]))
    assert np.allclose(buffer[0, 0], [1.0, 2.0, 3.0])


def test_float32_buffers() -> None:
    three = np.zeros((4, 4, 3), dtype=np.float32)
    scatter_add(three, np.array([0]), np.array([[1.0, 2.0, 3.0]]))
    assert three.sum() == pytest.approx(6.0)

    two = np.zeros((4, 4), dtype=np.float32)
    scatter_add(two, np.array([0, 0, 5]), np.array([1.0, 2.0, 3.0]))
    assert two.sum() == pytest.approx(6.0)


def test_repeated_calls_accumulate() -> None:
    buffer = np.zeros((3, 3))
    scatter_add(buffer, np.array([4]), np.array([2.0]))
    scatter_add(buffer, np.array([4]), np.array([3.0]))
    assert buffer[1, 1] == 5.0


def test_adjacent_pixels_stay_distinct() -> None:
    """Guards against collapsing indices onto a single cell."""
    buffer = np.zeros((10, 10))
    scatter_add(buffer, np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]))
    assert buffer[0, 0] == 1.0
    assert buffer[0, 1] == 2.0
    assert buffer[0, 2] == 3.0


def test_non_square_buffer() -> None:
    buffer = np.zeros((5, 7, 3))
    scatter_add(buffer, np.array([0, 34]), np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    assert buffer.sum() == pytest.approx(9.0)


def test_matches_naive_accumulation_on_a_realistic_case() -> None:
    """The fast path must equal a straightforward reference implementation."""
    rng = np.random.default_rng(4)
    width, height = 64, 48
    indices = rng.integers(0, width * height, 5000)
    values = rng.random((5000, 3))

    buffer = np.zeros((height, width, 3))
    scatter_add(buffer, indices, values)

    reference = np.zeros((height, width, 3))
    for index, value in zip(indices, values, strict=True):
        reference.reshape(-1, 3)[index] += value

    assert np.allclose(buffer, reference, atol=1e-12)


def test_matches_reference_for_single_channel() -> None:
    rng = np.random.default_rng(5)
    width, height = 40, 30
    indices = rng.integers(0, width * height, 2000)
    values = rng.random(2000)

    buffer = np.zeros((height, width))
    scatter_add(buffer, indices, values)

    reference = np.zeros(width * height)
    np.add.at(reference, indices, values)

    assert np.allclose(buffer.reshape(-1), reference, atol=1e-12)


class TestDownsample:
    def test_factor_of_one_is_a_no_op(self) -> None:
        array = np.random.default_rng(6).random((8, 8))
        assert downsample(array, 1) is array

    def test_averages_blocks(self) -> None:
        array = np.arange(16, dtype=float).reshape(4, 4)
        result = downsample(array, 2)
        assert result.shape == (2, 2)
        assert result[0, 0] == pytest.approx((0 + 1 + 4 + 5) / 4)

    def test_preserves_channel_axis(self) -> None:
        """A 2D buffer must stay 2D; a silent extra axis breaks indexing."""
        assert downsample(np.zeros((8, 8)), 2).ndim == 2
        assert downsample(np.zeros((8, 8, 3)), 2).shape == (4, 4, 3)

    def test_preserves_alpha_axis(self) -> None:
        assert downsample(np.zeros((8, 8, 4)), 2).shape == (4, 4, 4)

    def test_drops_incomplete_edge(self) -> None:
        array = np.ones((5, 5))
        assert downsample(array, 2).shape == (2, 2)

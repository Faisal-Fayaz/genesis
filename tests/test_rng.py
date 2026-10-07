"""Seeded randomness.

Two properties matter and both are load-bearing. Drawing the same named substream
twice must give the same values, or nothing reproduces. And drawing from one
substream must not disturb another, or adding a parameter to one generator would
change another's output.
"""

from __future__ import annotations

import numpy as np
import pytest

from genesis.rng import (
    bernoulli,
    jitter,
    rand_int,
    random_seed,
    scramble,
    stream,
    substream_seed,
    uniform,
    weighted_choice,
)


class TestSubstreams:
    def test_same_name_gives_the_same_values(self) -> None:
        assert np.array_equal(stream(42, "a").random(10), stream(42, "a").random(10))

    def test_different_names_are_independent(self) -> None:
        assert not np.array_equal(stream(42, "a").random(10), stream(42, "b").random(10))

    def test_different_seeds_are_independent(self) -> None:
        assert not np.array_equal(stream(1, "a").random(10), stream(2, "a").random(10))

    def test_consuming_one_does_not_shift_another(self) -> None:
        """The property that lets generators evolve independently."""
        before = stream(42, "alpha").random(5)
        stream(42, "beta").random(10_000)
        assert np.array_equal(before, stream(42, "alpha").random(5))

    def test_consuming_in_a_different_order_gives_the_same_values(self) -> None:
        first = stream(7, "x").random(4)
        stream(7, "y").random(100)
        second = stream(7, "x").random(4)
        assert np.array_equal(first, second)

    def test_entropy_is_injectively_derived(self) -> None:
        assert substream_seed(5, "a") != substream_seed(5, "b")

    def test_seed_is_masked_to_64_bits(self) -> None:
        assert stream(1 << 70, "a").random(3).shape == (3,)


class TestHelpers:
    def test_uniform_respects_bounds(self) -> None:
        rng = stream(1, "t")
        values = [uniform(rng, 2.0, 3.0) for _ in range(500)]
        assert all(2.0 <= v <= 3.0 for v in values)

    def test_uniform_is_reproducible(self) -> None:
        a = [uniform(stream(1, "t"), 0, 1) for _ in range(5)]
        b = [uniform(stream(1, "t"), 0, 1) for _ in range(5)]
        assert a == b

    def test_rand_int_is_inclusive_on_both_ends(self) -> None:
        rng = stream(2, "t")
        values = {rand_int(rng, 1, 3) for _ in range(500)}
        assert values == {1, 2, 3}

    def test_rand_int_handles_a_single_value(self) -> None:
        assert rand_int(stream(2, "t"), 7, 7) == 7

    def test_weighted_choice_reaches_every_option(self) -> None:
        rng = stream(3, "t")
        seen = {weighted_choice(rng, ["a", "b", "c"], [1, 1, 1]) for _ in range(300)}
        assert seen == {"a", "b", "c"}

    def test_weighted_choice_respects_weights(self) -> None:
        rng = stream(4, "t")
        picks = [weighted_choice(rng, ["rare", "common"], [1, 99]) for _ in range(1000)]
        assert picks.count("rare") < picks.count("common")

    def test_weighted_choice_rejects_empty_weights(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            weighted_choice(stream(5, "t"), ["a"], [0.0])

    def test_weighted_choice_accepts_unnormalized_weights(self) -> None:
        rng = stream(6, "t")
        picks = {weighted_choice(rng, ["a", "b"], [5, 5]) for _ in range(100)}
        assert picks == {"a", "b"}

    def test_bernoulli_bounds(self) -> None:
        rng = stream(7, "t")
        assert bernoulli(rng, 0.0) is False
        assert bernoulli(rng, 1.0) is True

    def test_jitter_stays_within_bounds(self) -> None:
        rng = stream(8, "t")
        values = [jitter(rng, 0.5, 10.0, 0.0, 1.0) for _ in range(500)]
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_jitter_with_zero_amount_is_a_no_op(self) -> None:
        assert jitter(stream(9, "t"), 0.25, 0.0, 0.0, 1.0) == 0.25

    def test_random_seed_is_non_negative(self) -> None:
        assert 0 <= random_seed(stream(10, "t")) < 1 << 31

    def test_random_seed_repeats(self) -> None:
        assert random_seed(stream(10, "t")) == random_seed(stream(10, "t"))

    def test_scramble_is_a_permutation(self) -> None:
        items = list(range(20))
        shuffled = scramble(stream(11, "t"), items)
        assert sorted(shuffled) == items

    def test_scramble_does_not_mutate_the_input(self) -> None:
        items = [1, 2, 3]
        scramble(stream(12, "t"), items)
        assert items == [1, 2, 3]

    def test_scramble_actually_reorders(self) -> None:
        items = list(range(50))
        assert scramble(stream(13, "t"), items) != items

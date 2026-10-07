"""Seed derivation and canonical serialization.

This module is the root of the reproducibility guarantee: a recipe is reduced to
one canonical string, and that string is hashed into the seed that drives the
render. If canonicalization were unstable, identical recipes would derive
different seeds and identical inputs would produce different images.
"""

from __future__ import annotations

import json

import pytest

from genesis.canon import canonical_json, derive_seed, stable_hash


class TestCanonicalJson:
    def test_key_order_does_not_matter(self) -> None:
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

    def test_nested_key_order_does_not_matter(self) -> None:
        first = {"params": {"z": 1, "a": {"y": 2, "b": 3}}}
        second = {"params": {"a": {"b": 3, "y": 2}, "z": 1}}
        assert canonical_json(first) == canonical_json(second)

    def test_has_no_insignificant_whitespace(self) -> None:
        assert canonical_json({"a": 1, "b": [1, 2]}) == '{"a":1,"b":[1,2]}'

    def test_distinguishes_different_values(self) -> None:
        assert canonical_json({"a": 1}) != canonical_json({"a": 2})
        assert canonical_json({"a": 1}) != canonical_json({"b": 1})

    def test_list_order_is_significant(self) -> None:
        """Unlike dicts, list order carries meaning and must be preserved."""
        assert canonical_json([1, 2]) != canonical_json([2, 1])

    def test_rejects_nan(self) -> None:
        """NaN is not equal to itself, which would break content addressing."""
        with pytest.raises(ValueError):
            canonical_json({"a": float("nan")})

    def test_handles_non_ascii_deterministically(self) -> None:
        assert canonical_json({"k": "café"}) == canonical_json({"k": "café"})


class TestStableHash:
    def test_is_deterministic(self) -> None:
        value = {"generator": "flowfield", "params": {"trails": 100}}
        assert stable_hash(value) == stable_hash(dict(value))

    def test_distinguishes_different_content(self) -> None:
        assert stable_hash({"a": 1}) != stable_hash({"a": 2})

    def test_size_controls_length(self) -> None:
        assert stable_hash({"a": 1}, size=4) < 1 << 32
        assert stable_hash({"a": 1}, size=8) < 1 << 64

    def test_is_not_python_hash(self) -> None:
        """Python's str and tuple hashing is salted per process; this is not."""
        # The same value must hash identically in any interpreter run.
        assert stable_hash(["form", 1.5, -1.5, 3.0]) == stable_hash(
            ["form", 1.5, -1.5, 3.0]
        )


class TestDeriveSeed:
    def test_absent_seed_is_derived_from_content(self) -> None:
        recipe = {"generator": "attractor", "params": {"points": 100}}
        assert derive_seed(recipe) == derive_seed(dict(recipe))

    def test_explicit_seed_wins(self) -> None:
        assert derive_seed({"generator": "x", "seed": 99}) == 99

    def test_explicit_seed_survives_other_edits(self) -> None:
        """Pinning is pinning: the seed must not track unrelated changes."""
        base = {"generator": "attractor", "params": {"points": 1}, "seed": 7}
        edited = {**base, "params": {"points": 999}, "width": 4000}
        assert derive_seed(base) == derive_seed(edited)

    def test_null_seed_behaves_as_absent(self) -> None:
        assert derive_seed({"a": 1, "seed": None}) == derive_seed({"a": 1})

    def test_content_changes_move_an_derived_seed(self) -> None:
        assert derive_seed({"params": {"trails": 1}}) != derive_seed(
            {"params": {"trails": 2}}
        )

    def test_result_fits_in_64_bits(self) -> None:
        assert 0 <= derive_seed({"generator": "x"}) < 1 << 64

    def test_large_pinned_seed_is_masked_not_wrapped(self) -> None:
        huge = 1 << 70
        assert derive_seed({"seed": huge}) == huge & ((1 << 64) - 1)

    def test_rejects_non_integer_seed(self) -> None:
        for bad in ("7", 7.0, True, [7]):
            with pytest.raises(TypeError):
                derive_seed({"generator": "x", "seed": bad})

    def test_key_order_does_not_change_the_seed(self) -> None:
        first = {"generator": "flowfield", "params": {"trails": 5, "steps": 9}}
        second = {"params": {"steps": 9, "trails": 5}, "generator": "flowfield"}
        assert derive_seed(first) == derive_seed(second)

    def test_survives_a_json_round_trip(self) -> None:
        recipe = {"generator": "x", "params": {"a": [1, 2]}, "width": 10}
        assert derive_seed(recipe) == derive_seed(json.loads(json.dumps(recipe)))

"""Canonical serialization and content-addressed seeding.

Every reproducibility guarantee in genesis rests on this module. A recipe is
reduced to a single canonical byte string; that string is hashed into the seed
that drives the render. Identical recipes therefore imply identical seeds, and
the render contains no other source of entropy.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

_HASH_BYTES = 8
_MASK64 = (1 << 64) - 1


def canonical_json(value: Any) -> str:
    """Serialize `value` to the one string form genesis considers identical."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def stable_hash(value: Any, *, size: int = _HASH_BYTES) -> int:
    """Content-address `value` with BLAKE2b, returned as a big-endian int."""
    digest = hashlib.blake2b(
        canonical_json(value).encode("utf-8"), digest_size=size
    ).digest()
    return int.from_bytes(digest, "big")


def derive_seed(recipe: dict[str, Any]) -> int:
    """Derive the render seed for a recipe.

    Recipes may pin an explicit ``seed``; when absent the seed is a pure
    function of the recipe's content, which is what makes unseeded generation
    still reproducible.
    """
    pinned = recipe.get("seed")
    if pinned is not None:
        if isinstance(pinned, bool) or not isinstance(pinned, int):
            raise TypeError("recipe seed must be an integer or null")
        return pinned & _MASK64
    body = {key: value for key, value in recipe.items() if key != "seed"}
    return stable_hash(body) & _MASK64

"""Seeded randomness.

Randomness is never drawn from module or interpreter state. Callers request a
named substream of the recipe's root seed, so unrelated draws cannot shift each
other and adding a parameter to one generator does not perturb another's output.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .canon import stable_hash

_MASK32 = (1 << 32) - 1
_MASK64 = (1 << 64) - 1


def substream_seed(seed: int, name: str) -> list[int]:
    """Entropy for the named substream of `seed`, independent of other names."""
    return [seed & _MASK64, stable_hash(name, size=4) & _MASK32]


def stream(seed: int, name: str = "root") -> np.random.Generator:
    """A PCG64 generator for the named substream of `seed`."""
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(substream_seed(seed, name)))
    )


def uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def rand_int(rng: np.random.Generator, low: int, high: int) -> int:
    """Uniform integer in the inclusive range [low, high]."""
    return int(rng.integers(low, high + 1))


def weighted_choice(
    rng: np.random.Generator, options: Sequence[str], weights: Sequence[float]
) -> str:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    draws = rng.choice(len(options), p=np.asarray(weights, dtype=float) / total)
    return options[int(draws)]


def bernoulli(rng: np.random.Generator, probability: float) -> bool:
    return bool(rng.random() < probability)


def jitter(
    rng: np.random.Generator, value: float, amount: float, low: float, high: float
) -> float:
    """Nudge `value` by up to +/- `amount`, clamped to the valid range."""
    return float(np.clip(value + rng.uniform(-amount, amount), low, high))


def random_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 1 << 31))


def scramble(rng: np.random.Generator, items: Sequence[Any]) -> list[Any]:
    pool = list(items)
    order = rng.permutation(len(pool))
    return [pool[int(index)] for index in order]

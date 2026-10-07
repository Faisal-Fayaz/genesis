"""Choosing what to show on a given day.

The scheduler's job is variety without randomness: on any given day it should
pick something you have not seen recently, and given the same library it should
pick the same thing, so a rerun does not surprise you.

Scoring is explicit rather than a shuffle. Each candidate earns points for being
unseen for a while, being from a generator you use less often, being a size or
palette you have seen less of, and a large flat bonus for being starred. The
winner is the highest score among a shortlist, with ties broken by seed, so the
choice is a pure function of the library's state.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from . import registry
from .library import Entry, Library
from .recipe import PRESETS, Recipe, random_recipe
from .rng import stream

# A piece not seen for this long is treated as new again.
FORGET_AFTER_DAYS = 21.0

# How much variety each dimension contributes. Generators get the most weight:
# seeing the same kind of picture every day is the thing to avoid.
WEIGHT_UNSEEN = 3.0
WEIGHT_GENERATOR = 1.5
WEIGHT_PALETTE = 1.0
WEIGHT_THEME = 0.5
WEIGHT_SIZE = 0.5
STAR_BONUS = 12.0

# A pick always comes from this many candidates, so the score decides among
# plausible options rather than being dominated by one.
SHORTLIST = 24


@dataclass(frozen=True)
class Choice:
    """Why a candidate won."""

    entry: Entry
    score: float
    reasons: dict[str, float]


@dataclass(frozen=True)
class Plan:
    """What to render today, and what to do with it.

    `existing` is a scored `Choice` rather than a bare `Entry`, so a caller that
    reuses a piece can also show why it was picked.
    """

    recipe: Recipe
    apply_after: bool
    reason: str
    existing: Choice | None = None


def _days_since(timestamp: float | None, now: float) -> float:
    if timestamp is None:
        return FORGET_AFTER_DAYS + 1.0
    return max((now - timestamp) / 86400.0, 0.0)


def score_entry(
    entry: Entry,
    entries: Sequence[Entry],
    now: float,
) -> Choice:
    """Score one piece against the rest of the library."""
    reasons: dict[str, float] = {}

    unseen = _days_since(entry.applied_at, now)
    reasons["unseen_days"] = unseen
    unseen_points = WEIGHT_UNSEEN * min(unseen / FORGET_AFTER_DAYS, 1.0)
    # Applied long ago and then starred means "keep this one", which the day
    # counter alone would keep demoting.
    if unseen > FORGET_AFTER_DAYS:
        unseen_points += 1.0
    reasons["unseen"] = unseen_points

    total = unseen_points

    if entry.starred:
        reasons["starred"] = STAR_BONUS
        total += STAR_BONUS

    # Diversity: prefer the generator, palette and theme used least often.
    for key, weight, attribute in (
        ("generator", WEIGHT_GENERATOR, "generator"),
        ("palette", WEIGHT_PALETTE, "palette_name"),
        ("theme", WEIGHT_THEME, "theme"),
    ):
        if not entries:
            continue
        counts: dict[str, int] = {}
        for other in entries:
            counts[getattr(other, attribute)] = counts.get(getattr(other, attribute), 0) + 1
        share = counts.get(getattr(entry, attribute), 0) / len(entries)
        points = weight * (1.0 - share)
        reasons[key] = points
        total += points

    size_counts: dict[str, int] = {}
    for other in entries:
        key = f"{other.width}x{other.height}"
        size_counts[key] = size_counts.get(key, 0) + 1
    if len(size_counts) > 1:
        share = size_counts.get(f"{entry.width}x{entry.height}", 0) / len(entries)
        points = WEIGHT_SIZE * (1.0 - share)
        reasons["size"] = points
        total += points

    return Choice(entry=entry, score=total, reasons=reasons)


def rank(
    entries: Sequence[Entry],
    now: float | None = None,
    starred_only: bool = False,
    limit: int = SHORTLIST,
) -> list[Choice]:
    """Rank the library, best candidate first.

    Sorted by score with the digest as a tiebreaker, so equal-scoring candidates
    are ordered deterministically rather than by whatever order the database
    happened to return.
    """
    now = time.time() if now is None else now
    candidates = [e for e in entries if e.image_path.exists()]
    if starred_only:
        candidates = [e for e in candidates if e.starred]
    if not candidates:
        return []

    scored = [score_entry(entry, candidates, now) for entry in candidates]
    scored.sort(key=lambda choice: (-choice.score, choice.entry.digest))
    return scored[:limit]


def pick_existing(
    library: Library,
    now: float | None = None,
    starred_only: bool = False,
) -> Choice | None:
    """Choose a piece already in the library, or None if there is nothing to pick.

    This is the cheap path: no render, so a daily run can re-apply something and
    still be near-instant.
    """
    entries = library.list(limit=1000)
    if not entries:
        return None
    ranked = rank(entries, now=now, starred_only=starred_only)
    return ranked[0] if ranked else None


def pick_new(
    library: Library,
    day: datetime.date | None = None,
    width: int = 1920,
    height: int = 1080,
    seed: int | None = None,
    theme: str = "dark",
    post: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """Invent a recipe biased away from what is already in the library.

    The day seeds the draw, so the same day always produces the same recipe and
    a rerun is harmless. Generators and palettes already well represented in the
    library are drawn less often.
    """
    # Local midnight is the right boundary: "today's wallpaper" should roll over
    # when the clock does, not at UTC.
    day = day if day is not None else datetime.now(UTC).astimezone().date()
    entropy = stream(seed if seed is not None else _day_seed(day), "schedule")

    generator_counts: dict[str, int] = {}
    palette_counts: dict[str, int] = {}
    for entry in library.iter_all():
        generator_counts[entry.generator] = generator_counts.get(entry.generator, 0) + 1
        palette_counts[entry.palette_name] = palette_counts.get(entry.palette_name, 0) + 1

    total_entries = max(sum(generator_counts.values()), 1)

    # A library holding only one generator still gets variety: the weights only
    # bend toward the rare choice, never exclude it.
    from .rng import weighted_choice

    generator_names = sorted(registry.names())
    if not generator_names:
        raise ValueError("no generators are registered")

    recipe = random_recipe(
        weighted_choice(
            entropy, generator_names, _generator_weights(generator_names, generator_counts, total_entries)
        ),
        rng=entropy,
        width=width,
        height=height,
        palette_name=weighted_choice(
            entropy, sorted(PRESETS), _palette_weights(palette_counts)
        ),
        theme=theme,
        post=dict(post or {}),
    )

    reason = _describe_intent(generator_counts, palette_counts, recipe)
    return recipe, reason


def _day_seed(day: datetime.date) -> int:
    """A stable seed for a calendar day."""
    from .canon import stable_hash

    return stable_hash(day.isoformat()) & ((1 << 64) - 1)


def _generator_weights(
    names: Sequence[str], counts: dict[str, int], total: int
) -> list[float]:
    """Weight each generator by how much of the library it already fills."""
    return [
        1.0 / (1.0 + counts.get(name, 0) / max(total, 1) * 4.0) for name in names
    ]


def _palette_weights(counts: dict[str, int]) -> list[float]:
    names = sorted(PRESETS)
    return [1.0 / (1.0 + counts.get(name, 0)) for name in names]


def _describe_intent(
    generator_counts: dict[str, int], palette_counts: dict[str, int], recipe: Any
) -> str:
    """A short account of why this recipe was chosen, for logs and the gallery."""
    bits = [f"generator {recipe.generator}"]
    if not generator_counts.get(recipe.generator):
        bits.append("unused here")
    elif recipe.palette_name not in palette_counts:
        bits.append(f"palette {recipe.palette_name} unused here")
    else:
        bits.append(f"palette {recipe.palette_name}")
    return ", ".join(bits)


def choose(
    library: Library,
    now: float | None = None,
    day: datetime.date | None = None,
    width: int = 1920,
    height: int = 1080,
    mode: str = "auto",
    seed: int | None = None,
    theme: str = "dark",
    post: dict[str, Any] | None = None,
) -> Plan:
    """Decide what to show today.

    `mode` selects the source:
      - ``reuse``  prefer something already rendered, and only invent if empty
      - ``new``    always invent a recipe
      - ``auto``   invent one, unless the library is empty or everything is
                   already stale, in which case reusing avoids pointless renders

    Reuse requires the image to still exist, so a library of deleted files falls
    through to generating rather than pointing the desktop at nothing.
    """
    now = time.time() if now is None else now
    entries = library.list(limit=1000)
    existing = pick_existing(library, now=now) if entries else None

    if mode == "reuse" and existing is not None:
        return Plan(
            recipe=existing.entry.recipe_object,
            apply_after=True,
            reason=f"reusing {existing.entry.slug} ({_describe_choice(existing)})",
            existing=existing,
        )

    # In auto mode, anything applied in the last day is still fresh, so showing
    # it again is the cheaper and less repetitive option.
    if (
        mode == "auto"
        and existing is not None
        and _days_since(existing.entry.applied_at, now) < 1.0
    ):
        return Plan(
            recipe=existing.entry.recipe_object,
            apply_after=True,
            reason=f"reusing {existing.entry.slug} ({_describe_choice(existing)})",
            existing=existing,
        )

    recipe, reason = pick_new(
        library,
        day=day,
        width=width,
        height=height,
        seed=seed,
        theme=theme,
        post=post,
    )
    return Plan(recipe=recipe, apply_after=True, reason=f"new: {reason}")


def _describe_choice(choice: Choice) -> str:
    """A one-line explanation of a pick's score."""
    top = sorted(choice.reasons.items(), key=lambda item: -item[1])[:2]
    return ", ".join(f"{name} {value:+.1f}" for name, value in top)


def summary(choices: Iterable[Choice]) -> str:
    """A compact table of ranked candidates, for `daily --dry-run`."""
    lines = []
    for choice in list(choices)[:10]:
        entry = choice.entry
        lines.append(
            f"  {choice.score:6.1f}  {entry.digest[:12]}  "
            f"{entry.generator:10s} {entry.palette_name:6s} "
            f"{_describe_choice(choice)}"
        )
    return "\n".join(lines)
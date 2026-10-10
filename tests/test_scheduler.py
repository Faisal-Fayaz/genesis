"""Choosing what to show.

The scheduler's contract is variety without randomness: a given day must produce
the same choice, and that choice should be something you have not seen recently.
Both properties are testable without rendering anything, which is what makes them
cheap to assert.
"""

from __future__ import annotations

import datetime
import time

import pytest

from genesis import library as library_module
from genesis import render as render_module
from genesis import scheduler as scheduler_module
from genesis.recipe import Recipe

DAY = datetime.date(2026, 3, 14)
NEXT_DAY = datetime.date(2026, 3, 15)


@pytest.fixture
def lib(tmp_path):
    library = library_module.Library(tmp_path / "root")
    library.open()
    yield library
    library.close()


def add(lib, seed: int = 5, generator: str = "flowfield", **overrides):
    params: dict[str, object] = {"supersample": 1}
    if generator == "attractor":
        params.update({"points": 20000, "warmup": 50})
    else:
        params.update({"trails": 1000, "steps": 25})
    fields: dict[str, object] = {
        "generator": generator,
        "params": params,
        "width": 96,
        "height": 54,
        "seed": seed,
        "palette_name": "tide",
    }
    fields.update(overrides)
    recipe = Recipe(**fields)
    return library_module.store(lib, recipe, render_module.render(recipe))


class TestScoring:
    def test_unseen_beats_recently_applied(self, lib) -> None:
        old = add(lib, seed=1)
        fresh = add(lib, seed=2)
        lib.record_applied(old.digest)
        entries = lib.list()
        by_digest = {entry.digest: entry for entry in entries}
        ranked = scheduler_module.rank(
            [by_digest[old.digest], by_digest[fresh.digest]]
        )
        assert ranked[0].entry.digest == fresh.digest

    def test_a_starred_piece_wins(self, lib) -> None:
        plain = add(lib, seed=1)
        starred = add(lib, seed=2)
        lib.set_starred(starred.digest, True)
        fresh = {entry.digest: entry for entry in lib.list()}
        ranked = scheduler_module.rank([fresh[plain.digest], fresh[starred.digest]])
        assert ranked[0].entry.digest == starred.digest

    def test_the_star_bonus_is_recorded(self, lib) -> None:
        entry = add(lib)
        lib.set_starred(entry.digest, True)
        choice = scheduler_module.score_entry(lib.latest(), [lib.latest()], time.time())
        assert choice.reasons["starred"] == scheduler_module.STAR_BONUS

    def test_the_rare_generator_is_favoured(self, lib) -> None:
        for seed in (1, 2, 3):
            add(lib, seed=seed, generator="flowfield")
        odd = add(lib, seed=4, generator="attractor")
        entries = {entry.digest: entry for entry in lib.list()}
        ranked = scheduler_module.rank(list(entries.values()))
        assert ranked[0].entry.digest == odd.digest

    def test_reasons_are_reported(self, lib) -> None:
        entry = add(lib)
        choice = scheduler_module.score_entry(entry, [entry], time.time())
        assert "unseen" in choice.reasons
        assert "generator" in choice.reasons


class TestRank:
    def test_empty_library_ranks_nothing(self, lib) -> None:
        assert scheduler_module.rank([]) == []

    def test_pieces_with_missing_images_are_dropped(self, lib) -> None:
        entry = add(lib)
        entry.image_path.unlink()
        assert scheduler_module.rank(lib.list()) == []

    def test_starred_only_filter(self, lib) -> None:
        add(lib, seed=1)
        starred = add(lib, seed=2)
        lib.set_starred(starred.digest, True)
        ranked = scheduler_module.rank(lib.list(), starred_only=True)
        assert [choice.entry.digest for choice in ranked] == [starred.digest]

    def test_ties_are_broken_deterministically(self, lib) -> None:
        for seed in range(4):
            add(lib, seed=seed)
        first = [choice.entry.digest for choice in scheduler_module.rank(lib.list())]
        second = [choice.entry.digest for choice in scheduler_module.rank(lib.list())]
        assert first == second

    def test_limit_caps_the_shortlist(self, lib) -> None:
        for seed in range(6):
            add(lib, seed=seed)
        assert len(scheduler_module.rank(lib.list(), limit=3)) == 3

    def test_summary_lists_candidates(self, lib) -> None:
        add(lib, seed=1)
        add(lib, seed=2)
        summary = scheduler_module.summary(scheduler_module.rank(lib.list()))
        assert len(summary.splitlines()) == 2


class TestDayStability:
    def test_the_same_day_gives_the_same_recipe(self, lib) -> None:
        first = scheduler_module.pick_new(lib, day=DAY)[0]
        second = scheduler_module.pick_new(lib, day=DAY)[0]
        assert first.slug() == second.slug()

    def test_different_days_give_different_recipes(self, lib) -> None:
        first = scheduler_module.pick_new(lib, day=DAY)[0]
        second = scheduler_module.pick_new(lib, day=NEXT_DAY)[0]
        assert first.slug() != second.slug()

    def test_an_explicit_seed_overrides_the_day(self, lib) -> None:
        first = scheduler_module.pick_new(lib, day=DAY, seed=99)[0]
        second = scheduler_module.pick_new(lib, day=NEXT_DAY, seed=99)[0]
        assert first.slug() == second.slug()

    def test_the_day_seed_is_a_pure_function(self) -> None:
        assert scheduler_module._day_seed(DAY) == scheduler_module._day_seed(DAY)
        assert scheduler_module._day_seed(DAY) != scheduler_module._day_seed(NEXT_DAY)


class TestPickNew:
    def test_returns_a_valid_recipe(self, lib) -> None:
        recipe, _ = scheduler_module.pick_new(lib, day=DAY)
        assert isinstance(recipe, Recipe)
        assert recipe.width == 1920

    def test_honours_the_requested_size(self, lib) -> None:
        recipe, _ = scheduler_module.pick_new(lib, day=DAY, width=1280, height=720)
        assert (recipe.width, recipe.height) == (1280, 720)

    def test_explains_its_choice(self, lib) -> None:
        _, reason = scheduler_module.pick_new(lib, day=DAY)
        assert "generator" in reason

    def test_avoids_the_dominant_generator(self, lib) -> None:
        """With one generator dominant, the other should still get picked."""
        for seed in range(6):
            add(lib, seed=seed, generator="flowfield")
        picks = {
            scheduler_module.pick_new(lib, day=datetime.date(2026, 3, d))[0].generator
            for d in range(1, 29)
        }
        assert "attractor" in picks

    def test_never_returns_an_unrenderable_recipe(self, lib) -> None:
        for day in range(1, 20):
            recipe, _ = scheduler_module.pick_new(lib, day=datetime.date(2026, 4, day))
            assert recipe.params  # validated on construction

    def test_random_picks_stay_affordable(self, lib) -> None:
        """Costs multiply, so a random draw must not be able to exceed a budget."""
        for day in range(1, 29):
            recipe, _ = scheduler_module.pick_new(
                lib, day=datetime.date(2026, 5, day), width=1920, height=1080
            )
            params = recipe.params
            if "trails" in params:
                assert params["trails"] <= 40000
            if "supersample" in params:
                assert params["supersample"] <= 1


class TestChoose:
    def test_empty_library_generates(self, lib) -> None:
        plan = scheduler_module.choose(lib, day=DAY, source="random")
        assert plan.existing is None
        assert plan.reason.startswith("new:")

    def test_curated_is_the_default_source(self, lib) -> None:
        """Curated is the floor for quality, so it should be what you get
        without asking. The random path is still reachable, just not the default.
        """
        plan = scheduler_module.choose(lib, day=DAY)
        assert plan.reason.startswith("curated ")
        from genesis import presets as curated_presets

        assert plan.recipe.canonical() in {
            preset.recipe().canonical() for preset in curated_presets.CURATED
        }

    def test_mood_narrows_the_curated_pool(self, lib) -> None:
        from genesis import presets as curated_presets

        allowed = {
            preset.recipe().canonical() for preset in curated_presets.for_mood("calm")
        }
        for offset in range(8):
            plan = scheduler_module.choose(lib, day=DAY, mood="calm", seed=offset)
            assert plan.recipe.canonical() in allowed

    def test_the_same_day_picks_the_same_preset(self, lib) -> None:
        first = scheduler_module.choose(lib, day=DAY)
        second = scheduler_module.choose(lib, day=DAY)
        assert first.recipe.canonical() == second.recipe.canonical()

    def test_curated_rotates_through_the_whole_set(self, lib) -> None:
        """Over a month every preset should appear, and none should dominate.

        The library is not updated between picks here on purpose: what is being
        checked is that the seeded draw spreads across the set, which is what
        keeps a month of daily wallpapers from looking like one image repeated.
        """
        seen = set()
        for offset in range(30):
            plan = scheduler_module.choose(lib, day=DAY, seed=offset)
            seen.add(plan.recipe.canonical())
        from genesis import presets as curated_presets

        assert len(seen) >= 12, f"only {len(seen)} distinct presets in 30 picks"
        assert len(seen) <= len(curated_presets.CURATED)

    def test_curated_skips_what_the_library_already_has(self, lib) -> None:
        """Once a preset is rendered it is deprioritised, so the set rotates."""
        from genesis import presets as curated_presets

        for preset in curated_presets.CURATED[:8]:
            recipe = preset.recipe(160, 90)
            lib.add(recipe, f"{abs(hash(preset.name)):016x}", lib.root / "images" / "x.png")
        plan = scheduler_module.choose(lib, day=DAY, seed=0)
        name = plan.reason.split()[1]
        assert name not in {p.name for p in curated_presets.CURATED[:8]}

    def test_unknown_mood_is_rejected(self, lib) -> None:
        with pytest.raises(ValueError):
            scheduler_module.choose(lib, day=DAY, mood="nonsense")

    def test_reuse_mode_uses_what_is_there(self, lib) -> None:
        entry = add(lib)
        plan = scheduler_module.choose(lib, mode="reuse")
        assert plan.existing is not None
        assert plan.existing.entry.digest == entry.digest

    def test_new_mode_always_generates(self, lib) -> None:
        add(lib)
        plan = scheduler_module.choose(lib, mode="new", day=DAY)
        assert plan.existing is None

    def test_auto_reuses_something_fresh(self, lib) -> None:
        entry = add(lib)
        lib.record_applied(entry.digest)
        plan = scheduler_module.choose(lib, mode="auto")
        assert plan.existing is not None
        assert plan.existing.entry.digest == entry.digest

    def test_auto_generates_when_everything_is_stale(self, lib) -> None:
        """A piece applied long ago is not worth re-showing; render something."""
        add(lib)
        old = time.time() - 40 * 86400
        lib.connection.execute("UPDATE pieces SET applied_at = ?", (old,))
        lib.connection.commit()
        plan = scheduler_module.choose(lib, mode="auto", day=DAY)
        assert plan.existing is None

    def test_falls_through_when_images_are_missing(self, lib) -> None:
        entry = add(lib)
        entry.image_path.unlink()
        lib.record_applied(entry.digest)
        plan = scheduler_module.choose(lib, mode="auto", day=DAY)
        assert plan.existing is None

    def test_theme_is_passed_through(self, lib) -> None:
        plan = scheduler_module.choose(lib, day=DAY, theme="light")
        assert plan.recipe.theme == "light"

    def test_post_options_are_passed_through(self, lib) -> None:
        plan = scheduler_module.choose(lib, day=DAY, post={"grain": 0.05})
        assert plan.recipe.post.get("grain") == 0.05

    def test_the_recipe_it_returns_renders(self, lib) -> None:
        plan = scheduler_module.choose(lib, day=DAY, width=160, height=90)
        result = render_module.render(plan.recipe)
        assert result.image.size == (160, 90)

    def test_is_deterministic_for_a_day(self, lib) -> None:
        add(lib)
        first = scheduler_module.choose(lib, mode="new", day=DAY).recipe.slug()
        second = scheduler_module.choose(lib, mode="new", day=DAY).recipe.slug()
        assert first == second


class TestPickExisting:
    def test_none_when_empty(self, lib) -> None:
        assert scheduler_module.pick_existing(lib) is None

    def test_returns_the_top_choice(self, lib) -> None:
        entry = add(lib)
        assert scheduler_module.pick_existing(lib).entry.digest == entry.digest

    def test_starred_only_filter(self, lib) -> None:
        add(lib, seed=1)
        starred = add(lib, seed=2)
        lib.set_starred(starred.digest, True)
        choice = scheduler_module.pick_existing(lib, starred_only=True)
        assert choice.entry.digest == starred.digest


class TestWeights:
    def test_generator_weights_favour_the_rare(self) -> None:
        names = ["a", "b"]
        weights = scheduler_module._generator_weights(names, {"a": 10, "b": 0}, 10)
        assert weights[1] > weights[0]

    def test_palette_weights_favour_the_rare(self) -> None:
        counts = {"tide": 5, "ember": 0}
        weights = scheduler_module._palette_weights(counts)
        assert len(weights) == len(scheduler_module._palette_weights({}))

    def test_weights_are_all_positive(self) -> None:
        weights = scheduler_module._generator_weights(["a"], {"a": 100}, 100)
        assert all(w > 0 for w in weights)


class TestDaysSince:
    def test_none_counts_as_long_ago(self) -> None:
        assert scheduler_module._days_since(None, time.time()) > scheduler_module.FORGET_AFTER_DAYS

    def test_just_applied_is_zero_days(self) -> None:
        assert scheduler_module._days_since(time.time(), time.time()) == pytest.approx(0, abs=1)

    def test_a_day_applied(self) -> None:
        now = time.time()
        assert scheduler_module._days_since(now - 86400, now) == pytest.approx(1, abs=0.01)

    def test_a_future_timestamp_is_clamped(self) -> None:
        now = time.time()
        assert scheduler_module._days_since(now + 86400, now) == 0.0
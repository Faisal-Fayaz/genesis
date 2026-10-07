"""The catalog of rendered pieces.

The library is derived state: every image on disk has its recipe written beside
it, so the database can be rebuilt. Two behaviours matter enough to pin down.
Storing the same recipe twice must not create a duplicate, and a record whose
image has been deleted must be detectable rather than quietly served to the
gallery as a broken thumbnail.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from genesis import library as library_module
from genesis import render as render_module
from genesis.recipe import Recipe


@pytest.fixture
def lib(tmp_path):
    library = library_module.Library(tmp_path / "root")
    library.open()
    yield library
    library.close()


def sample_recipe(seed: int = 5, generator: str = "flowfield", **overrides) -> Recipe:
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
    return Recipe(**fields)


def add(lib, recipe=None, **overrides):
    recipe = recipe or sample_recipe(**overrides)
    result = render_module.render(recipe)
    return library_module.store(lib, recipe, result, render_seconds=0.01)


class TestSetup:
    def test_creates_the_directory_layout(self, lib) -> None:
        assert lib.root.is_dir()
        assert lib.images.is_dir()
        assert lib.previews.is_dir()
        assert lib.database.exists()

    def test_open_is_idempotent(self, lib) -> None:
        lib.open()
        lib.open()
        assert lib.count() == 0

    def test_default_root_follows_xdg(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert library_module.default_root() == tmp_path / "genesis"

    def test_default_root_falls_back_to_home(self, monkeypatch) -> None:
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        assert library_module.default_root().parts[-3:] == (".local", "share", "genesis")

    def test_context_manager_closes(self, tmp_path) -> None:
        with library_module.library_at(tmp_path / "ctx") as library:
            library.connection.execute("SELECT 1")
        assert getattr(library, "_connection", None) is None

    def test_connection_is_usable_from_another_thread(self, lib) -> None:
        """The server is threaded, so the connection must not be thread-bound."""
        import threading

        errors: list[Exception] = []

        def read() -> None:
            try:
                assert lib.count() == 0
            except Exception as exc:  # noqa: BLE001 - recorded and re-raised
                errors.append(exc)

        thread = threading.Thread(target=read)
        thread.start()
        thread.join()
        assert not errors


class TestStore:
    def test_records_the_piece(self, lib) -> None:
        entry = add(lib)
        assert lib.count() == 1
        assert lib.get(entry.digest) is not None

    def test_writes_image_and_recipe(self, lib) -> None:
        entry = add(lib)
        assert entry.image_path.exists()
        assert entry.image_path.with_suffix(".json").exists()

    def test_recipe_beside_the_image_is_loadable(self, lib) -> None:
        entry = add(lib)
        loaded = Recipe.load(entry.image_path.with_suffix(".json"))
        assert loaded.to_dict() == entry.recipe

    def test_reuses_the_filename_for_the_same_recipe(self, lib) -> None:
        first = add(lib)
        second = add(lib)
        assert first.image_path == second.image_path

    def test_storing_twice_does_not_duplicate(self, lib) -> None:
        add(lib)
        add(lib)
        assert lib.count() == 1

    def test_preserves_the_star_across_re_stores(self, lib) -> None:
        entry = add(lib)
        lib.set_starred(entry.digest, True)
        add(lib)
        assert lib.get(entry.digest).starred is True

    def test_preserves_created_at_across_re_stores(self, lib) -> None:
        entry = add(lib)
        before = lib.get(entry.digest).created_at
        time.sleep(0.01)
        add(lib)
        assert lib.get(entry.digest).created_at == before

    def test_generates_a_preview_for_a_large_recipe(self, lib) -> None:
        entry = add(lib, width=1920, height=1080)
        assert entry.preview_path is not None
        assert entry.preview_path.exists()

    def test_no_preview_when_the_recipe_is_already_small(self, lib) -> None:
        assert add(lib, width=96, height=54).preview_path is None

    def test_different_seeds_produce_separate_entries(self, lib) -> None:
        add(lib, seed=1)
        add(lib, seed=2)
        assert lib.count() == 2


class TestStarring:
    def test_set_and_clear(self, lib) -> None:
        entry = add(lib)
        lib.set_starred(entry.digest, True)
        assert lib.get(entry.digest).starred is True
        lib.set_starred(entry.digest, False)
        assert lib.get(entry.digest).starred is False

    def test_toggle_returns_the_new_value(self, lib) -> None:
        entry = add(lib)
        assert lib.toggle_starred(entry.digest) is True
        assert lib.toggle_starred(entry.digest) is False

    def test_toggle_unknown_returns_none(self, lib) -> None:
        assert lib.toggle_starred("nope") is None

    def test_setting_unknown_reports_no_rows(self, lib) -> None:
        assert lib.set_starred("nope", True) is False

    def test_filters_by_starred(self, lib) -> None:
        starred = add(lib, seed=1)
        add(lib, seed=2)
        lib.set_starred(starred.digest, True)
        assert [e.digest for e in lib.list(starred=True)] == [starred.digest]


class TestApplied:
    def test_records_the_application(self, lib) -> None:
        entry = add(lib)
        lib.record_applied(entry.digest)
        stored = lib.get(entry.digest)
        assert stored.applied_count == 1
        assert stored.applied_at is not None

    def test_counts_repeat_applications(self, lib) -> None:
        entry = add(lib)
        lib.record_applied(entry.digest)
        lib.record_applied(entry.digest)
        assert lib.get(entry.digest).applied_count == 2

    def test_most_recently_applied(self, lib) -> None:
        first = add(lib, seed=1)
        second = add(lib, seed=2)
        lib.record_applied(first.digest)
        time.sleep(0.01)
        lib.record_applied(second.digest)
        assert lib.most_recently_applied().digest == second.digest

    def test_none_when_nothing_was_applied(self, lib) -> None:
        add(lib)
        assert lib.most_recently_applied() is None


class TestQueries:
    @pytest.fixture
    def populated(self, lib):
        add(lib, seed=1, generator="flowfield", palette_name="tide")
        add(lib, seed=2, generator="attractor", palette_name="ember")
        add(lib, seed=3, generator="flowfield", palette_name="ember")
        return lib

    def test_count(self, populated) -> None:
        assert populated.count() == 3

    def test_count_by_generator(self, populated) -> None:
        assert populated.count(generator="flowfield") == 2

    def test_count_by_palette(self, populated) -> None:
        assert populated.count(palette="ember") == 2

    def test_list_is_newest_first(self, populated) -> None:
        digests = [entry.digest for entry in populated.list()]
        assert digests[0] != digests[-1]

    def test_list_filters(self, populated) -> None:
        assert len(populated.list(generator="attractor")) == 1
        assert len(populated.list(palette="tide")) == 1

    def test_list_limit_and_offset(self, populated) -> None:
        assert len(populated.list(limit=2)) == 2
        assert len(populated.list(limit=2, offset=2)) == 1

    def test_latest(self, populated) -> None:
        assert populated.latest() is not None

    def test_latest_on_an_empty_library(self, lib) -> None:
        assert lib.latest() is None

    def test_get_unknown_returns_none(self, lib) -> None:
        assert lib.get("deadbeef") is None

    def test_find_by_slug(self, populated) -> None:
        entry = populated.latest()
        assert populated.find(entry.slug).digest == entry.digest

    def test_find_unknown_returns_none(self, lib) -> None:
        assert lib.find("no-such-slug") is None

    def test_generators_lists_what_is_present(self, populated) -> None:
        assert populated.generators() == ["attractor", "flowfield"]

    def test_iter_all_yields_everything(self, populated) -> None:
        assert len(list(populated.iter_all())) == 3

    def test_entry_round_trips_to_a_recipe(self, populated) -> None:
        entry = populated.latest()
        assert entry.recipe_object.to_dict() == entry.recipe

    def test_to_json_is_serializable(self, populated) -> None:
        payload = populated.latest().to_json()
        assert json.loads(json.dumps(payload))["generator"] in ("flowfield", "attractor")

    def test_to_json_exposes_urls(self, populated) -> None:
        entry = add(populated, width=1920, height=1080)
        payload = entry.to_json()
        assert payload["image"].startswith("/image/")
        assert payload["preview"].startswith("/preview/")


class TestDelete:
    def test_removes_the_record(self, lib) -> None:
        entry = add(lib)
        assert lib.delete(entry.digest) is True
        assert lib.count() == 0

    def test_keeps_files_by_default(self, lib) -> None:
        entry = add(lib)
        lib.delete(entry.digest)
        assert entry.image_path.exists()

    def test_removes_files_on_request(self, lib) -> None:
        entry = add(lib)
        lib.delete(entry.digest, remove_files=True)
        assert not entry.image_path.exists()

    def test_unknown_digest_reports_false(self, lib) -> None:
        assert lib.delete("nope") is False


class TestHousekeeping:
    def test_detects_a_missing_image(self, lib) -> None:
        entry = add(lib)
        entry.image_path.unlink()
        assert [e.digest for e in lib.missing_images()] == [entry.digest]

    def test_prune_removes_orphan_records(self, lib) -> None:
        entry = add(lib)
        entry.image_path.unlink()
        assert lib.prune_orphans() == 1
        assert lib.count() == 0

    def test_rebuild_recovers_from_recipes_on_disk(self, lib) -> None:
        entry = add(lib)
        # Lose the database but keep the images and their recipes.
        lib.close()
        lib.database.unlink()
        lib.open()
        assert lib.count() == 0
        assert lib.rebuild() == 1
        assert lib.get(entry.digest) is not None

    def test_rebuild_recovers_the_star(self, lib) -> None:
        entry = add(lib)
        lib.set_starred(entry.digest, True)
        lib.close()
        lib.database.unlink()
        lib.open()
        lib.rebuild()
        assert lib.latest().starred is False

    def test_rebuild_skips_unreadable_recipes(self, lib) -> None:
        (lib.images / "garbage.json").write_text("{not json")
        assert lib.rebuild() == 0

    def test_rebuild_skips_recipes_without_images(self, lib) -> None:
        (lib.images / "orphan.json").write_text(
            json.dumps(sample_recipe().to_dict())
        )
        assert lib.rebuild() == 0


class TestHelpers:
    def test_elapsed_formats_by_magnitude(self) -> None:
        assert library_module.elapsed(0.25).endswith("ms")
        assert library_module.elapsed(5.0).endswith("s")
        assert library_module.elapsed(120.0).endswith("m")

    def test_describe_entry_marks_stars(self, lib) -> None:
        starred = add(lib, seed=1)
        add(lib, seed=2)
        lib.set_starred(starred.digest, True)
        lines = [library_module.describe_entry(e) for e in lib.list()]
        assert any(line.startswith("*") for line in lines)
        assert any(line.startswith(" ") for line in lines)

    def test_parse_filter(self) -> None:
        assert library_module.parse_filter("starred=1") == (True, None)
        assert library_module.parse_filter("palette=ember") == (None, "ember")
        assert library_module.parse_filter(None) == (None, None)

    def test_sort_entries_by_each_key(self, lib) -> None:
        """Sorting reads from the database, so entries are re-read first.

        `Entry` is an immutable snapshot, not a live view: the one returned by
        `store` still shows `starred=False` after the row is updated. Callers
        that sort should pass entries from `list`, not from `store`.
        """
        first = add(lib, seed=1)
        time.sleep(0.01)
        second = add(lib, seed=2)
        lib.set_starred(first.digest, True)
        lib.record_applied(first.digest)

        fresh = {entry.digest: entry for entry in lib.list()}

        by_starred = library_module.sort_entries(
            [fresh[second.digest], fresh[first.digest]], "starred"
        )
        assert by_starred[0].digest == first.digest
        assert by_starred[-1].digest == second.digest

        by_recent = library_module.sort_entries(
            [fresh[first.digest], fresh[second.digest]], "recent"
        )
        assert by_recent[0].digest == second.digest

        by_applied = library_module.sort_entries(
            [fresh[second.digest], fresh[first.digest]], "applied"
        )
        assert by_applied[0].digest == first.digest

        by_rendered = library_module.sort_entries(
            [fresh[second.digest], fresh[first.digest]], "rendered"
        )
        assert len(by_rendered) == 2

    def test_entry_is_an_immutable_snapshot(self, lib) -> None:
        """Re-reading is required to observe an update; the object is not live."""
        entry = add(lib, seed=1)
        lib.set_starred(entry.digest, True)
        assert entry.starred is False
        assert lib.get(entry.digest).starred is True


class TestPersistence:
    def test_survives_reopening(self, tmp_path) -> None:
        root = tmp_path / "persist"
        with library_module.library_at(root) as library:
            recipe = sample_recipe()
            library_module.store(library, recipe, render_module.render(recipe))
            digest = library.latest().digest

        with library_module.library_at(root) as library:
            assert library.get(digest) is not None

    def test_stored_recipes_load_in_a_fresh_process(self, tmp_path) -> None:
        """A recipe file must be readable without the library in memory."""
        recipe = sample_recipe()
        with library_module.library_at(tmp_path / "fresh") as library:
            library_module.store(library, recipe, render_module.render(recipe))
            path = library.find(recipe.slug()).image_path.with_suffix(".json")

        loaded = Recipe.load(Path(path))
        assert loaded.to_dict() == recipe.to_dict()
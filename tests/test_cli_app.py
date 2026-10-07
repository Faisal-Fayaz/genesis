"""The new CLI verbs.

`apply`, `daily`, `gallery` and `library` are the operational surface: what a
timer runs, what a person types. Each test drives `cli.main` with real arguments
and asserts on the exit code and output, because that is what a shell script
sees. Nothing here touches the real desktop: the gsettings helper is stubbed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from genesis import cli
from genesis import render as render_module
from genesis.recipe import Recipe


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path / "lib"


@pytest.fixture(autouse=True)
def no_wallpaper(monkeypatch, tmp_path):
    """Never touch the real desktop.

    `apply` and `daily` write gsettings; without this they would change the
    developer's wallpaper whenever the suite ran.
    """
    from genesis import wallpaper as wallpaper_module

    applied: list[tuple[str, str]] = []

    def fake_apply(source, destination=None, theme="dark", stage_copy=True):
        applied.append((str(source), theme))
        return wallpaper_module.AppliedState(
            dark=wallpaper_module.uri_for(Path(source)), light=None
        )

    monkeypatch.setattr(wallpaper_module, "apply", fake_apply)
    monkeypatch.setattr(wallpaper_module, "apply_pair", fake_apply)
    monkeypatch.setattr(wallpaper_module, "has_session", lambda: True)
    monkeypatch.setattr(wallpaper_module, "_display_size", lambda: (1920, 1080))
    yield applied


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def small_recipe_file(tmp_path: Path, name: str = "r.json", **overrides) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(recipe_dict(**overrides), indent=2))
    return path


def recipe_dict(**overrides) -> dict:
    fields: dict = {
        "generator": "flowfield",
        "params": {"trails": 1000, "steps": 25, "supersample": 1},
        "width": 96,
        "height": 54,
        "seed": 5,
        "palette_name": "tide",
    }
    fields.update(overrides)
    return fields


def populate(root: Path, count: int = 1) -> None:
    """Fill a library with rendered pieces, leaving captured output empty.

    Rendering writes through the CLI, which prints progress. Tests read stdout
    with `capsys`, so anything printed here would otherwise be prepended to
    their output and break JSON parsing.
    """
    import contextlib
    import io

    from genesis import library as library_module

    for index in range(count):
        with contextlib.redirect_stdout(io.StringIO()):
            code = cli.main(
                [
                    "gen",
                    str(small_recipe_file(root.parent, f"r{index}.json", seed=index + 1)),
                    "-o",
                    str(root.parent / f"o{index}.png"),
                ]
            )
        assert code == 0
        with library_module.library_at(root) as library:
            recipe = Recipe(**recipe_dict(seed=index + 1))
            library_module.store(library, recipe, render_module.render(recipe))


class TestApply:
    def test_applies_the_latest_piece(self, capsys, root, no_wallpaper) -> None:
        populate(root)
        code, out, _err = run(capsys, "apply", "--root", str(root))
        assert code == 0
        assert "wallpaper set" in out

    def test_reaches_the_stub_with_both_themes(self, capsys, root, no_wallpaper) -> None:
        populate(root)
        run(capsys, "apply", "--root", str(root), "--theme", "both")
        assert no_wallpaper[-1][1] == "both"

    def test_empty_library_is_an_error(self, capsys, root) -> None:
        code, _out, err = run(capsys, "apply", "--root", str(root))
        assert code == 1
        assert "empty" in err

    def test_unknown_digest_is_an_error(self, capsys, root) -> None:
        populate(root)
        code, _out, _err = run(capsys, "apply", "deadbeef", "--root", str(root))
        assert code == 1

    def test_renders_a_recipe_then_applies(self, capsys, root, no_wallpaper, tmp_path) -> None:
        recipe_path = small_recipe_file(tmp_path)
        code, out, _err = run(capsys, "apply", "--recipe", str(recipe_path), "--root", str(root))
        assert code == 0
        assert "wallpaper set" in out

    def test_recipe_overrides_are_applied(self, capsys, root, no_wallpaper, tmp_path) -> None:
        recipe_path = small_recipe_file(tmp_path)
        code, _out, err = run(
            capsys, "apply", "--recipe", str(recipe_path),
            "--param", "trails=2000", "--root", str(root),
        )
        assert code == 0, err

    def test_bad_override_is_reported(self, capsys, root, no_wallpaper, tmp_path) -> None:
        recipe_path = small_recipe_file(tmp_path)
        code, _out, err = run(
            capsys, "apply", "--recipe", str(recipe_path),
            "--param", "trails=0", "--root", str(root),
        )
        assert code == 1
        assert "minimum" in err

    def test_missing_recipe_file_is_an_error(self, capsys, root, tmp_path) -> None:
        code, _out, err = run(capsys, "apply", "--recipe", str(tmp_path / "nope.json"), "--root", str(root))
        assert code == 1
        assert "genesis:" in err


class TestDaily:
    def test_renders_into_an_empty_library(self, capsys, root, no_wallpaper) -> None:
        code, out, _err = run(
            capsys, "daily", "--root", str(root), "--width", "96", "--height", "54"
        )
        assert code == 0
        assert "rendered" in out
        assert "wallpaper set" in out

    def test_render_only_skips_the_desktop(self, capsys, root, no_wallpaper) -> None:
        code, _out, _err = run(
            capsys, "daily", "--render-only", "--root", str(root), "--width", "96", "--height", "54"
        )
        assert code == 0
        assert no_wallpaper == []

    def test_dry_run_touches_nothing(self, capsys, root, no_wallpaper) -> None:
        """A dry run must not render and must not change the desktop."""
        code, _out, _err = run(capsys, "daily", "--dry-run", "--root", str(root))
        assert code == 0
        assert no_wallpaper == []
        assert list(root.glob("images/*.png")) == []

    def test_dry_run_on_an_empty_library_explains(self, capsys, root) -> None:
        code, out, _err = run(capsys, "daily", "--dry-run", "--root", str(root))
        assert code == 0
        assert "empty" in out

    def test_dry_run_lists_candidates(self, capsys, root) -> None:
        populate(root, 3)
        code, out, _err = run(capsys, "daily", "--dry-run", "--root", str(root))
        assert code == 0
        assert "candidates" in out

    def test_dry_run_json_is_parseable(self, capsys, root) -> None:
        populate(root, 2)
        _code, out, _err = run(capsys, "daily", "--dry-run", "--root", str(root), "--json")
        payload = json.loads(out)
        assert isinstance(payload, list)
        assert "score" in payload[0]

    def test_reuse_mode_applies_without_rendering(self, capsys, root, no_wallpaper) -> None:
        populate(root)
        code, out, _err = run(capsys, "daily", "--mode", "reuse", "--root", str(root))
        assert code == 0
        assert "reused" in out

    def test_reuse_on_an_empty_library_generates(self, capsys, root, no_wallpaper) -> None:
        code, out, _err = run(
            capsys, "daily", "--mode", "reuse", "--root", str(root),
            "--width", "96", "--height", "54",
        )
        assert code == 0
        assert "rendered" in out

    def test_post_overrides_are_applied(self, capsys, root, no_wallpaper) -> None:
        code, _out, err = run(
            capsys, "daily", "--render-only", "--root", str(root),
            "--width", "96", "--height", "54", "--post", "grain=0.04",
        )
        assert code == 0, err

    def test_bad_post_override_is_reported(self, capsys, root) -> None:
        code, _out, err = run(
            capsys, "daily", "--root", str(root), "--post", "sepia=1"
        )
        assert code == 1
        assert "unknown post option" in err

    def test_seed_pins_the_day(self, capsys, root, no_wallpaper) -> None:
        code, out, _err = run(
            capsys, "daily", "--render-only", "--root", str(root),
            "--seed", "7", "--width", "96", "--height", "54",
        )
        assert code == 0
        assert "rendered" in out

    def test_reports_missing_session_without_failing(self, capsys, root, monkeypatch) -> None:
        """A timer firing before login must not look like success."""
        from genesis import wallpaper as wallpaper_module

        def refuse(*args, **kwargs):
            raise wallpaper_module.WallpaperUnavailable("no graphical session available")

        monkeypatch.setattr(wallpaper_module, "apply", refuse)
        code, out, err = run(
            capsys, "daily", "--root", str(root), "--width", "96", "--height", "54"
        )
        assert code == 0
        assert "skipped wallpaper change" in err
        assert "rendered" in out


class TestGallery:
    def test_refuses_a_non_loopback_bind(self, capsys, tmp_path) -> None:
        code, _out, err = run(capsys, "gallery", "--host", "0.0.0.0", "--root", str(tmp_path))
        assert code == 1
        assert "localhost" in err

    def test_serves_until_interrupted(self, capsys, root, monkeypatch) -> None:
        """`gallery` builds a server, prints where it is, and serves until stopped.

        Only the blocking loop is replaced. Calling `shutdown()` from a thread
        instead would deadlock, since `serve_forever` sets the flag that
        `shutdown` waits on and is not yet running.

        Real request handling is covered in `test_server.py`, against a real
        server; repeating it here would only reintroduce the race between the
        accept loop and the client.
        """
        from genesis import server as server_module

        served = threading.Event()

        monkeypatch.setattr(
            server_module.GalleryServer,
            "serve_forever",
            lambda self, **kwargs: served.set(),
        )

        code, out, _err = run(capsys, "gallery", "--root", str(root))
        assert served.is_set(), "the serve loop was never entered"
        assert code == 0
        assert "gallery on http://127.0.0.1" in out
        assert "library" in out
        assert "ctrl-c to stop" in out

    def test_reports_a_bad_port(self, capsys, tmp_path) -> None:
        code, _out, err = run(capsys, "gallery", "--port", "1", "--root", str(tmp_path))
        assert code == 1
        assert "cannot bind" in err


class TestLibrary:
    def test_lists_an_empty_library(self, capsys, root) -> None:
        code, out, _err = run(capsys, "library", "list", "--root", str(root))
        assert code == 0
        assert "empty" in out

    def test_lists_populated(self, capsys, root) -> None:
        populate(root, 2)
        code, out, _err = run(capsys, "library", "list", "--root", str(root))
        assert code == 0
        assert "2 of 2 pieces" in out

    def test_json_output_is_parseable(self, capsys, root) -> None:
        populate(root)
        _code, out, _err = run(capsys, "library", "list", "--root", str(root), "--json")
        payload = json.loads(out)
        assert len(payload) == 1
        assert payload[0]["generator"] == "flowfield"

    def test_star_and_unstar(self, capsys, root) -> None:
        populate(root)
        digest = _first_digest(root)
        code, out, _err = run(capsys, "library", "star", digest, "--root", str(root))
        assert code == 0
        assert "starred" in out
        _code, out, _err = run(capsys, "library", "unstar", digest, "--root", str(root))
        assert "unstarred" in out

    def test_starred_filter(self, capsys, root) -> None:
        populate(root)
        digest = _first_digest(root)
        run(capsys, "library", "star", digest, "--root", str(root))
        _code, out, _err = run(
            capsys, "library", "list", "--root", str(root), "--starred", "--json"
        )
        listed = json.loads(out)
        assert len(listed) == 1
        assert listed[0]["starred"] is True

    def test_show_reports_details(self, capsys, root) -> None:
        populate(root)
        digest = _first_digest(root)
        code, out, _err = run(capsys, "library", "show", digest, "--root", str(root))
        assert code == 0
        assert "digest" in out
        assert "generator" in out

    def test_show_json_is_parseable(self, capsys, root) -> None:
        populate(root)
        _code, out, _err = run(
            capsys, "library", "show", _first_digest(root), "--root", str(root), "--json"
        )
        assert json.loads(out)["generator"] == "flowfield"

    def test_delete_removes_the_record(self, capsys, root) -> None:
        populate(root)
        digest = _first_digest(root)
        code, out, _err = run(capsys, "library", "delete", digest, "--root", str(root))
        assert code == 0
        assert "deleted" in out
        _code, out, _err = run(capsys, "library", "list", "--root", str(root))
        assert "empty" in out

    def test_delete_with_files_removes_the_image(self, capsys, root) -> None:
        populate(root)
        digest = _first_digest(root)
        from genesis import library as library_module

        with library_module.library_at(root) as library:
            image = library.get(digest).image_path
        run(capsys, "library", "delete", digest, "--root", str(root), "--files")
        assert not image.exists()

    def test_prune_reports_nothing_to_do(self, capsys, root) -> None:
        populate(root)
        code, out, _err = run(capsys, "library", "prune", "--root", str(root))
        assert code == 0
        assert "removed 0" in out

    def test_prune_clears_missing_images(self, capsys, root) -> None:
        populate(root)
        from genesis import library as library_module

        with library_module.library_at(root) as library:
            library.latest().image_path.unlink()
        code, out, _err = run(capsys, "library", "prune", "--root", str(root))
        assert code == 0
        assert "removed 1" in out

    def test_unknown_digest_is_an_error(self, capsys, root) -> None:
        populate(root)
        code, _out, _err = run(capsys, "library", "show", "deadbeef", "--root", str(root))
        assert code == 1

    def test_import_registers_local_recipes(self, capsys, tmp_path) -> None:
        source = tmp_path / "elsewhere"
        source.mkdir()
        (source / "mine.json").write_text(json.dumps(recipe_dict()))
        (source / "broken.json").write_text("{not json")

        code, out, _err = run(capsys, "library", "import", str(source), "--root", str(tmp_path / "lib"))
        assert code == 0
        assert "imported 1" in out

    def test_filters_are_accepted(self, capsys, root) -> None:
        populate(root, 2)
        code, _out, _err = run(
            capsys, "library", "list", "--root", str(root),
            "--generator", "flowfield", "--palette", "tide",
        )
        assert code == 0

    def test_limit_is_accepted(self, capsys, root) -> None:
        populate(root, 3)
        code, out, _err = run(capsys, "library", "list", "--root", str(root), "--limit", "1")
        assert code == 0
        assert "1 of 3" in out


def _first_digest(root: Path) -> str:
    from genesis import library as library_module

    with library_module.library_at(root) as library:
        return library.latest().digest


class TestHelpText:
    @pytest.mark.parametrize(
        "command", ["apply", "daily", "gallery", "library", "gen", "list", "describe"]
    )
    def test_every_command_has_help(self, command) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.main([command, "--help"])
        assert excinfo.value.code == 0

    def test_the_overview_lists_every_command(self, capsys) -> None:
        with pytest.raises(SystemExit):
            cli.main(["--help"])
        out = capsys.readouterr().out
        for command in ("apply", "daily", "gallery", "library"):
            assert command in out


class TestModuleEntryPoint:
    def test_runs_as_a_process(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "genesis.cli", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0
        assert "flowfield" in completed.stdout
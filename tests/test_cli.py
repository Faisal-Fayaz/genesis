"""The command-line surface.

These run the real `main` with real arguments, so they cover argument parsing,
override handling, file writes, and the exit codes a shell script would see.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from genesis import cli


@pytest.fixture
def recipe_file(tmp_path) -> Path:
    """A small, fast recipe on disk."""
    path = tmp_path / "recipe.json"
    path.write_text(
        json.dumps(
            {
                "generator": "flowfield",
                "params": {"trails": 1000, "steps": 30, "supersample": 1},
                "width": 64,
                "height": 48,
                "seed": 5,
                "palette_name": "tide",
            },
            indent=2,
        )
    )
    return path


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestList:
    def test_lists_generators_by_default(self, capsys) -> None:
        code, out, _ = run(capsys, "list")
        assert code == 0
        assert "flowfield" in out
        assert "attractor" in out

    def test_json_output_is_parseable(self, capsys) -> None:
        code, out, _ = run(capsys, "list", "--json")
        assert code == 0
        payload = json.loads(out)
        assert {item["name"] for item in payload} >= {"flowfield", "attractor"}

    def test_params_flag_shows_bounds(self, capsys) -> None:
        _, out, _ = run(capsys, "list", "--params")
        assert "range" in out

    def test_palettes_are_listed(self, capsys) -> None:
        code, out, _ = run(capsys, "list", "palettes")
        assert code == 0
        assert "tide" in out

    def test_palette_json_is_parseable(self, capsys) -> None:
        code, out, _ = run(capsys, "list", "palettes", "--json")
        assert code == 0
        assert "tide" in json.loads(out)


class TestGen:
    def test_renders_a_recipe(self, capsys, recipe_file, tmp_path) -> None:
        target = tmp_path / "out.png"
        code, out, err = run(capsys, "gen", str(recipe_file), "-o", str(target))
        assert code == 0
        assert target.exists()
        assert "wrote" in out or "wrote" in err

    def test_json_output_carries_the_digest(self, capsys, recipe_file, tmp_path) -> None:
        target = tmp_path / "out.png"
        _, out, _ = run(capsys, "gen", str(recipe_file), "-o", str(target), "--json")
        payload = json.loads(out)
        assert len(payload["digest"]) == 64
        assert payload["generator"] == "flowfield"

    def test_missing_recipe_file_is_an_error(self, capsys, tmp_path) -> None:
        code, _, err = run(capsys, "gen", str(tmp_path / "nope.json"))
        assert code == 1
        assert "genesis:" in err

    def test_missing_recipe_argument_is_an_error(self, capsys) -> None:
        """Handled in the handler, so it returns rather than raising SystemExit."""
        code, _, err = run(capsys, "gen")
        assert code == 2
        assert "recipe path is required" in err

    def test_default_output_is_beside_the_recipe(self, capsys, recipe_file) -> None:
        code, _, _ = run(capsys, "gen", str(recipe_file))
        assert code == 0
        assert list(recipe_file.parent.glob("*.png"))

    def test_quiet_suppresses_progress(self, capsys, recipe_file, tmp_path) -> None:
        code, out, _ = run(
            capsys, "gen", str(recipe_file), "-o", str(tmp_path / "o.png"), "-q"
        )
        assert code == 0
        assert out.strip() == ""

    def test_render_is_reproducible_across_invocations(
        self, capsys, recipe_file, tmp_path
    ) -> None:
        first = tmp_path / "a.png"
        second = tmp_path / "b.png"
        run(capsys, "gen", str(recipe_file), "-o", str(first), "--json")
        run(capsys, "gen", str(recipe_file), "-o", str(second), "--json")
        assert first.read_bytes() == second.read_bytes()


class TestOverrides:
    def test_param_override_changes_the_output(self, capsys, recipe_file, tmp_path) -> None:
        plain = tmp_path / "plain.png"
        tweaked = tmp_path / "tweaked.png"
        _, _, _ = run(capsys, "gen", str(recipe_file), "-o", str(plain))
        _, _, _ = run(
            capsys, "gen", str(recipe_file), "-o", str(tweaked), "--param", "trails=2000"
        )
        assert plain.read_bytes() != tweaked.read_bytes()

    def test_recipe_field_override(self, capsys, recipe_file, tmp_path) -> None:
        target = tmp_path / "out.png"
        code, out, _ = run(
            capsys, "gen", str(recipe_file), "-o", str(target),
            "--set", "palette_name=ember", "--json",
        )
        assert code == 0
        assert json.loads(out)["width"] == 64

    def test_post_override(self, capsys, recipe_file, tmp_path) -> None:
        target = tmp_path / "out.png"
        code, _, _ = run(
            capsys, "gen", str(recipe_file), "-o", str(target), "--post", "grain=0.05"
        )
        assert code == 0

    def test_save_writes_the_recipe_back(self, capsys, recipe_file, tmp_path) -> None:
        target = tmp_path / "out.png"
        run(capsys, "gen", str(recipe_file), "-o", str(target), "--set", "seed=99", "--save")
        written = json.loads(recipe_file.read_text())
        assert written["seed"] == 99

    def test_without_save_the_recipe_is_untouched(self, capsys, recipe_file, tmp_path) -> None:
        before = recipe_file.read_text()
        run(capsys, "gen", str(recipe_file), "-o", str(tmp_path / "o.png"), "--set", "seed=99")
        assert recipe_file.read_text() == before

    def test_scale_resizes(self, capsys, recipe_file, tmp_path) -> None:
        _, out, _ = run(
            capsys, "gen", str(recipe_file), "-o", str(tmp_path / "o.png"),
            "--scale", "0.5", "--json",
        )
        payload = json.loads(out)
        assert payload["width"] == 32
        assert payload["height"] == 24

    def test_preview_shrinks(self, capsys, recipe_file, tmp_path) -> None:
        _, out, _ = run(
            capsys, "gen", str(recipe_file), "-o", str(tmp_path / "o.png"),
            "--preview", "--preview-edge", "32", "--json",
        )
        payload = json.loads(out)
        assert max(payload["width"], payload["height"]) == 32


class TestOverrideErrors:
    @pytest.mark.parametrize(
        "argv, expected",
        [
            (["--param", "trails=0"], "minimum"),
            (["--param", "trails=99999999"], "maximum"),
            (["--param", "nonexistent=1"], "unknown parameter"),
            (["--param", "malformed"], "KEY=VALUE"),
            (["--post", "sepia=1"], "unknown post option"),
            (["--set", "width=notanumber"], "width"),
            (["--set", "theme=neon"], "theme"),
        ],
    )
    def test_bad_overrides_exit_nonzero_with_a_message(
        self, capsys, recipe_file, tmp_path, argv, expected
    ) -> None:
        code, _, err = run(
            capsys, "gen", str(recipe_file), "-o", str(tmp_path / "o.png"), *argv
        )
        assert code == 1
        assert expected in err

    def test_override_errors_do_not_write_a_file(
        self, capsys, recipe_file, tmp_path
    ) -> None:
        target = tmp_path / "o.png"
        run(capsys, "gen", str(recipe_file), "-o", str(target), "--param", "trails=0")
        assert not target.exists()

    def test_value_parsing(self) -> None:
        assert cli._coerce("true") is True
        assert cli._coerce("false") is False
        assert cli._coerce("7") == 7
        assert cli._coerce("1.5") == 1.5
        assert cli._coerce("none") is None
        assert cli._coerce("hello") == "hello"


class TestGenRandom:
    def test_invents_a_recipe(self, capsys, tmp_path) -> None:
        target = tmp_path / "out.png"
        code, out, _ = run(
            capsys, "gen-random", "--seed", "7", "--width", "64", "--height", "48",
            "-o", str(target),
        )
        assert code == 0
        payload = json.loads(out)
        assert target.exists()
        assert payload["recipe"]

    def test_pins_the_seed_when_asked(self, capsys, tmp_path) -> None:
        target = tmp_path / "out.png"
        _, out, _ = run(
            capsys, "gen-random", "--seed", "11", "--width", "64", "--height", "48",
            "-o", str(target),
        )
        assert json.loads(out)["seed"] == 11

    def test_same_seed_gives_the_same_image(self, capsys, tmp_path) -> None:
        first, second = tmp_path / "a.png", tmp_path / "b.png"
        argv = ["gen-random", "--seed", "5", "--width", "64", "--height", "48"]
        run(capsys, *argv, "-o", str(first))
        run(capsys, *argv, "-o", str(second))
        assert first.read_bytes() == second.read_bytes()

    def test_different_seeds_give_different_images(self, capsys, tmp_path) -> None:
        first, second = tmp_path / "a.png", tmp_path / "b.png"
        base = ["gen-random", "--width", "64", "--height", "48"]
        run(capsys, *base, "--seed", "1", "-o", str(first))
        run(capsys, *base, "--seed", "2", "-o", str(second))
        assert first.read_bytes() != second.read_bytes()

    def test_named_generator_is_used(self, capsys, tmp_path) -> None:
        _, out, _ = run(
            capsys, "gen-random", "attractor", "--seed", "3",
            "--width", "64", "--height", "48", "-o", str(tmp_path / "o.png"),
        )
        assert json.loads(out)["generator"] == "attractor"

    def test_unknown_generator_is_an_error(self, capsys, tmp_path) -> None:
        code, _, err = run(
            capsys, "gen-random", "nonexistent", "-o", str(tmp_path / "o.png")
        )
        assert code == 1
        assert "available" in err

    def test_palette_is_honoured(self, capsys, tmp_path) -> None:
        _, out, _ = run(
            capsys, "gen-random", "--seed", "3", "--palette", "ember",
            "--width", "64", "--height", "48", "-o", str(tmp_path / "o.png"),
        )
        assert json.loads(out)["palette"] == "ember"

    def test_param_override_applies(self, capsys, tmp_path) -> None:
        target = tmp_path / "o.png"
        code, _, _ = run(
            capsys, "gen-random", "flowfield", "--seed", "3", "--width", "64",
            "--height", "48", "--param", "trails=5000", "--param", "steps=30", "-o", str(target),
        )
        assert code == 0

    def test_param_override_errors_are_reported(self, capsys, tmp_path) -> None:
        code, _, err = run(
            capsys, "gen-random", "flowfield", "--seed", "3",
            "--param", "trails=0", "-o", str(tmp_path / "o.png"),
        )
        assert code == 1
        assert "minimum" in err

    def test_no_save_skips_the_recipe(self, capsys, tmp_path) -> None:
        target = tmp_path / "o.png"
        _, out, _ = run(
            capsys, "gen-random", "--seed", "3", "--width", "64", "--height", "48",
            "--no-save", "-o", str(target),
        )
        assert json.loads(out)["recipe"] is None
        assert not target.with_suffix(".json").exists()

    def test_quiet_prints_only_the_path(self, capsys, tmp_path) -> None:
        target = tmp_path / "o.png"
        _, out, _ = run(
            capsys, "gen-random", "--seed", "3", "--width", "64", "--height", "48",
            "-q", "-o", str(target),
        )
        assert out.strip() == str(target)

    def test_gen_random_alias_works(self, capsys, tmp_path) -> None:
        """`genesis gen random` reads better than `gen-random`."""
        target = tmp_path / "o.png"
        code, _, _ = run(
            capsys, "gen", "random", "--seed", "3", "--width", "64", "--height", "48",
            "-o", str(target),
        )
        assert code == 0
        assert target.exists()


class TestDescribe:
    def test_prints_resolved_settings(self, capsys, recipe_file) -> None:
        code, out, _ = run(capsys, "describe", str(recipe_file))
        assert code == 0
        assert "flowfield" in out
        assert "seed" in out

    def test_json_output_is_parseable(self, capsys, recipe_file) -> None:
        _, out, _ = run(capsys, "describe", str(recipe_file), "--json")
        assert json.loads(out)["generator"] == "flowfield"

    def test_missing_file_is_an_error(self, capsys, tmp_path) -> None:
        code, _, _ = run(capsys, "describe", str(tmp_path / "nope.json"))
        assert code == 1


class TestTopLevel:
    def test_version_exits_cleanly(self, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["--version"])
        assert excinfo.value.code == 0

    def test_help_exits_cleanly(self, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["--help"])
        assert excinfo.value.code == 0

    def test_no_subcommand_is_an_error(self, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.main([])
        assert excinfo.value.code != 0

    def test_unknown_subcommand_is_an_error(self, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["nonsense"])
        assert excinfo.value.code != 0


class TestEntryPoint:
    """The module must be runnable as a real process."""

    def test_module_runs(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "genesis.cli", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0
        assert "genesis" in completed.stdout

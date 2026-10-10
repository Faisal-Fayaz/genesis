"""The live CLI: planning, refusal paths, and service installation.

Dry runs never open a window and never touch the desktop. The refusal tests
pin the two ways live mode says no: no display at all, and Wayland instead of
X11.
"""

from __future__ import annotations

import json

import pytest

from genesis import cli as cli_module
from genesis import systemd as systemd_module


def run(capsys, monkeypatch, *argv: str, display=":99", wayland=None):
    monkeypatch.setenv("DISPLAY", display) if display else monkeypatch.delenv(
        "DISPLAY", raising=False
    )
    if wayland:
        monkeypatch.setenv("WAYLAND_DISPLAY", wayland)
    else:
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    code = cli_module.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_live_dry_run_plans_full_hd(capsys, monkeypatch, tmp_path) -> None:
    code, out, _ = run(
        capsys, monkeypatch, "live", "--preset", "silk", "--dry-run",
        "--resolution", "1080p", "--root", str(tmp_path / "lib"),
    )
    assert code == 0
    assert "1920x1080" in out
    assert "silk" in out or "flowfield" in out


def test_live_resolution_auto_uses_the_display(capsys, monkeypatch, tmp_path) -> None:
    import json

    from genesis import wallpaper as wallpaper_module

    monkeypatch.setattr(wallpaper_module, "display_size", lambda: (2560, 1440))
    code, out, _ = run(
        capsys, monkeypatch, "live", "--preset", "silk",
        "--dry-run", "--json", "--root", str(tmp_path / "lib"),
    )
    assert code == 0
    plan = json.loads(out)
    assert plan["output"] == [2560, 1440]
    assert plan["output_source"] == "display"


def test_live_outside_world_policies_reach_the_plan(
    capsys, monkeypatch, tmp_path
) -> None:
    import json

    code, out, _ = run(
        capsys, monkeypatch, "live", "--preset", "silk",
        "--fullscreen", "off", "--no-pause-on-lock", "--no-battery-cap",
        "--resolution", "1080p", "--dry-run", "--json",
        "--root", str(tmp_path / "lib"),
    )
    assert code == 0
    plan = json.loads(out)
    assert plan["yield_to_fullscreen"] == "off"
    assert plan["pause_on_lock"] is False
    assert plan["cap_on_battery"] is False


def test_live_plan_carries_policy_defaults(capsys, monkeypatch, tmp_path) -> None:
    import json

    code, out, _ = run(
        capsys, monkeypatch, "live", "--preset", "silk",
        "--resolution", "1080p", "--dry-run", "--json",
        "--root", str(tmp_path / "lib"),
    )
    assert code == 0
    plan = json.loads(out)
    assert plan["yield_to_fullscreen"] == "slow"
    assert plan["pause_on_lock"] is True
    assert plan["cap_on_battery"] is True


def test_live_cursor_glow_reaches_the_plan(capsys, monkeypatch, tmp_path) -> None:
    import json

    code, out, _ = run(
        capsys, monkeypatch, "live", "--preset", "silk", "--cursor-glow", "0.25",
        "--ambient-speed", "2.0",
        "--resolution", "1080p", "--dry-run", "--json",
        "--root", str(tmp_path / "lib"),
    )
    assert code == 0
    plan = json.loads(out)
    assert plan["cursor_glow"] == 0.25
    assert plan["ambient_speed"] == 2.0


def test_live_relax_flags_layer(capsys, monkeypatch, tmp_path) -> None:
    import json

    def plan_of(*extra: str) -> dict:
        code, out, _ = run(
            capsys, monkeypatch, "live", "--preset", "silk",
            *extra, "--resolution", "1080p", "--dry-run", "--json",
            "--root", str(tmp_path / "lib"),
        )
        assert code == 0
        return json.loads(out)

    defaults = plan_of()
    assert (defaults["relax"], defaults["relax_idle"]) == (0.004, 0.02)
    both = plan_of("--relax", "0.01")
    assert (both["relax"], both["relax_idle"]) == (0.01, 0.01)
    split = plan_of("--relax", "0.01", "--relax-idle", "0.05")
    assert (split["relax"], split["relax_idle"]) == (0.01, 0.05)


def test_live_motion_bundle_layers_under_explicit_flags(
    capsys, monkeypatch, tmp_path
) -> None:
    import json

    def plan_of(*extra: str) -> dict:
        code, out, _ = run(
            capsys, monkeypatch, "live", "--preset", "silk",
            *extra, "--resolution", "1080p", "--dry-run", "--json",
            "--root", str(tmp_path / "lib"),
        )
        assert code == 0
        return json.loads(out)

    calm = plan_of("--motion", "calm")
    assert calm["ambient_speed"] == 0.8
    assert calm["stir_strength"] == 3.0
    assert calm["cursor_glow"] == 0.15
    mixed = plan_of("--motion", "calm", "--stir-strength", "1.0")
    assert mixed["stir_strength"] == 1.0  # explicit wins its piece
    assert mixed["ambient_speed"] == 0.8  # the rest stays bundle


def test_live_resolution_auto_falls_back_without_a_display(
    capsys, monkeypatch, tmp_path
) -> None:
    import json

    from genesis import wallpaper as wallpaper_module

    monkeypatch.setattr(wallpaper_module, "display_size", lambda: None)
    code, out, _ = run(
        capsys, monkeypatch, "live", "--preset", "silk",
        "--dry-run", "--json", "--root", str(tmp_path / "lib"),
    )
    assert code == 0
    plan = json.loads(out)
    assert plan["output"] == [1920, 1080]
    assert plan["output_source"] == "fallback"


def test_live_dry_run_json_is_parseable(capsys, monkeypatch, tmp_path) -> None:
    code, out, _ = run(
        capsys, monkeypatch, "live", "--mood", "calm", "--resolution", "720p",
        "--dry-run", "--json", "--root", str(tmp_path / "lib"),
    )
    assert code == 0
    plan = json.loads(out)
    assert plan["output"] == [1280, 720]
    assert plan["fps"] == 60


def test_live_from_digest_animates_that_piece(capsys, monkeypatch, tmp_path) -> None:
    from genesis import library as library_module
    from genesis import render as render_module
    from genesis.recipe import Recipe

    root = tmp_path / "lib"
    recipe = Recipe(
        generator="flowfield",
        params={"trails": 1000, "steps": 25, "supersample": 1},
        width=96,
        height=54,
        seed=7,
        palette_name="ember",
    )
    with library_module.library_at(root) as library:
        entry = library_module.store(library, recipe, render_module.render(recipe))
    code, out, _ = run(
        capsys, monkeypatch, "live", "--from", entry.digest,
        "--dry-run", "--json", "--root", str(root),
    )
    assert code == 0
    plan = json.loads(out)
    assert plan["recipe"]["palette"] == "ember"


def test_live_from_latest_and_unknown(capsys, monkeypatch, tmp_path) -> None:
    from genesis import library as library_module
    from genesis import render as render_module
    from genesis.recipe import Recipe

    root = tmp_path / "lib"
    recipe = Recipe(
        generator="flowfield",
        params={"trails": 1000, "steps": 25, "supersample": 1},
        width=96,
        height=54,
        seed=9,
    )
    with library_module.library_at(root) as library:
        library_module.store(library, recipe, render_module.render(recipe))
    code, out, _ = run(
        capsys, monkeypatch, "live", "--from", "latest",
        "--dry-run", "--json", "--root", str(root),
    )
    assert code == 0
    plan = json.loads(out)
    assert plan["recipe"]["palette"] == "tide"
    assert plan["recipe"]["generator"] == "flowfield"

    code, _, err = run(
        capsys, monkeypatch, "live", "--from", "deadbeef",
        "--dry-run", "--root", str(root),
    )
    assert code == 1
    assert "no piece" in err


def test_live_from_applied_animates_the_desktop_wallpaper(
    capsys, monkeypatch, tmp_path
) -> None:
    from genesis import library as library_module
    from genesis import render as render_module
    from genesis.recipe import Recipe

    root = tmp_path / "lib"
    first = Recipe(
        generator="flowfield",
        params={"trails": 1000, "steps": 25, "supersample": 1},
        width=96,
        height=54,
        seed=21,
        palette_name="ember",
    )
    second = Recipe(
        generator="flowfield",
        params={"trails": 1000, "steps": 25, "supersample": 1},
        width=96,
        height=54,
        seed=22,
        palette_name="tide",
    )
    with library_module.library_at(root) as library:
        library_module.store(library, first, render_module.render(first))
        entry = library_module.store(library, second, render_module.render(second))
        library.record_applied(entry.digest)
    code, out, _ = run(
        capsys, monkeypatch, "live", "--from", "applied",
        "--dry-run", "--json", "--root", str(root),
    )
    assert code == 0
    plan = json.loads(out)
    assert plan["recipe"]["palette"] == "tide"


def test_live_from_applied_without_any_application_explains_itself(
    capsys, monkeypatch, tmp_path
) -> None:
    from genesis import library as library_module
    from genesis import render as render_module
    from genesis.recipe import Recipe

    root = tmp_path / "lib"
    recipe = Recipe(
        generator="flowfield",
        params={"trails": 1000, "steps": 25, "supersample": 1},
        width=96,
        height=54,
        seed=23,
    )
    with library_module.library_at(root) as library:
        library_module.store(library, recipe, render_module.render(recipe))
    code, _, err = run(
        capsys, monkeypatch, "live", "--from", "applied",
        "--dry-run", "--root", str(root),
    )
    assert code == 1
    assert "genesis daily" in err


def test_live_rejects_bad_resolutions(capsys, monkeypatch, tmp_path) -> None:
    code, _, err = run(
        capsys, monkeypatch, "live", "--resolution", "banana",
        "--dry-run", "--root", str(tmp_path / "lib"),
    )
    assert code == 1
    assert "resolution" in err


def test_live_refuses_without_a_display(capsys, monkeypatch, tmp_path) -> None:
    code, _, err = run(
        capsys, monkeypatch, "live", "--dry-run", "--root", str(tmp_path / "lib"),
        display=None,
    )
    assert code == 1
    assert "DISPLAY" in err


def test_live_refuses_wayland(capsys, monkeypatch, tmp_path) -> None:
    code, _, err = run(
        capsys, monkeypatch, "live", "--dry-run", "--root", str(tmp_path / "lib"),
        display=None, wayland="wayland-0",
    )
    assert code == 1
    assert "Wayland" in err


def test_live_help_mentions_cursor_modes(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli_module.main(["live", "--help"])
    assert excinfo.value.code == 0
    assert "repel" in capsys.readouterr().out


def test_install_live_copies_but_never_enables(tmp_path) -> None:
    copied = systemd_module.install(tmp_path, live=True)
    assert systemd_module.LIVE_SERVICE_NAME in copied
    assert (tmp_path / systemd_module.LIVE_SERVICE_NAME).exists()
    assert (tmp_path / systemd_module.TIMER_NAME).exists()


def test_live_service_animates_the_applied_wallpaper() -> None:
    """The persistent service must follow the user's wallpaper, not its own pick."""
    from genesis import systemd as systemd_module_inner

    text = systemd_module_inner.unit_path(
        systemd_module_inner.LIVE_SERVICE_NAME
    ).read_text()
    assert "genesis.cli live --from applied" in text


def test_live_service_is_crash_loop_guarded() -> None:
    """A recipe that crashes on startup must stay down, not hot-loop."""
    import configparser

    from genesis import systemd as systemd_module_inner

    parser = configparser.RawConfigParser(strict=True)
    parser.read_string(
        systemd_module_inner.unit_path(
            systemd_module_inner.LIVE_SERVICE_NAME
        ).read_text()
    )
    assert parser["Unit"].getint("StartLimitBurst") <= 5
    assert "StartLimitIntervalSec" in parser["Unit"]


def test_render_unit_preserves_the_live_from_flag() -> None:
    """Templating rewrites only the executable prefix, never the arguments."""
    from genesis import systemd as systemd_module_inner

    rendered = systemd_module_inner.render_unit(
        systemd_module_inner.LIVE_SERVICE_NAME, "/home/u/.local/bin/genesis"
    )
    assert "ExecStart=/home/u/.local/bin/genesis live --from applied" in rendered


def test_install_commands_cover_the_live_unit() -> None:
    commands = " ".join(systemd_module.install_commands(live=True))
    assert systemd_module.LIVE_SERVICE_NAME in commands
    assert "enable --now genesis-live" not in commands


def test_install_print_lists_live_with_flag(capsys) -> None:
    code = cli_module.main(["install", "--print", "--live"])
    assert code == 0
    assert "genesis-live.service" in capsys.readouterr().out

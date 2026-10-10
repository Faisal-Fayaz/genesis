"""The native interface, without requiring a display server.

`genesis.app` keeps GTK imports inside `run_app`, so these tests can exercise
the controller, Tune layout, persisted settings, and CLI wiring headlessly. The
fake renderer returns a small synthetic image: what matters here is token
handling, validation, storage, and wallpaper actions, not pixels.
"""

from __future__ import annotations

import datetime
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from genesis import app as app_module
from genesis import cli as cli_module
from genesis.recipe import Recipe
from genesis.render import RenderResult


def tiny_recipe(**overrides) -> Recipe:
    """A recipe small enough for fast tests that still validates."""
    fields = {
        "generator": "flowfield",
        "params": {"trails": 1000, "steps": 25, "supersample": 1},
        "width": 96,
        "height": 54,
        "seed": 7,
        "palette_name": "tide",
    }
    fields.update(overrides)
    return Recipe(**fields)


def fake_result(recipe: Recipe) -> RenderResult:
    image = Image.new("RGB", (recipe.width, recipe.height), (4, 18, 31))
    return RenderResult(recipe, image, f"fake-{recipe.slug()}")


@pytest.fixture
def controller(tmp_path: Path, monkeypatch) -> app_module.AppController:
    """A controller with an isolated library and settings file."""
    # The live tests simulate an X11 session; the stub session never opens
    # it. Refusal tests override this env explicitly.
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    state = app_module.AppState(
        root=tmp_path / "lib",
        recipe=tiny_recipe(),
        preview_edge=160,
    )
    controller = app_module.AppController(
        state=state,
        renderer=fake_result,
        seed_source=lambda: 11,
        app_config_path=tmp_path / "app.json",
    )
    yield controller
    controller.close()


def test_default_state_uses_a_valid_curated_recipe(tmp_path: Path) -> None:
    state = app_module.default_state(tmp_path / "lib", 96, 54, "calm")
    state.recipe.validate()
    assert state.recipe.palette_name == "tide"
    assert state.mood == "calm"


def test_default_state_rejects_an_unknown_mood(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown mood"):
        app_module.default_state(tmp_path / "lib", 96, 54, "loud")


def test_tune_panel_has_five_stable_groups() -> None:
    groups = app_module.tune_layout(tiny_recipe())
    assert [group.title for group in groups] == [
        "Shape",
        "Motion",
        "Colour",
        "Frame",
        "Finish",
    ]
    controls = {(control.scope, control.name) for group in groups for control in group.controls}
    assert ("params", "scale") in controls
    assert ("params", "trails") in controls
    assert ("post", "vignette") in controls


def test_engine_and_grade_contrast_have_different_labels() -> None:
    labels = {
        control.label
        for group in app_module.tune_layout(tiny_recipe())
        for control in group.controls
        if control.name == "contrast"
    }
    assert labels == {"Contrast (engine)", "Contrast (grade)"}


def test_splash_recipe_exposes_its_shape_controls() -> None:
    """A splash preset in the app must offer its own knobs, not an empty Tune.

    Tune groups are name-driven, so a new generator's parameters only appear
    if someone lists them. This pins all four splash-specific knobs to Shape.
    """
    recipe = tiny_recipe().with_changes(generator="splash", params={})
    shape = next(
        group for group in app_module.tune_layout(recipe) if group.key == "shape"
    )
    names = {control.name for control in shape.controls}
    assert {"blobs", "spread", "turbulence", "fray"} <= names


def test_conditional_controls_carry_their_reasons() -> None:
    recipe = tiny_recipe().with_changes(generator="attractor", params={})
    controls = {
        control.name: control
        for group in app_module.tune_layout(recipe)
        for control in group.controls
    }
    assert controls["pan_x"].available is False
    assert controls["pan_x"].reason == "needs zoom below 1"
    assert controls["extent"].available is False
    assert controls["extent"].reason == "needs fit set to fixed"
    assert controls["zoom"].available is True
    assert controls["zoom"].reason is None


def test_settings_round_trip(tmp_path: Path) -> None:
    state = app_module.AppState(root=tmp_path / "lib", recipe=tiny_recipe(), mood="soft")
    path = app_module.save_state(state, tmp_path / "settings" / "app.json")
    restored = app_module.load_state(path)
    assert restored.mood == "soft"
    assert restored.recipe.to_dict() == state.recipe.to_dict()


def test_settings_reject_bad_versions_and_moods(tmp_path: Path) -> None:
    state = app_module.default_state(tmp_path / "lib", 96, 54)
    payload = app_module.state_to_json(state)
    payload["version"] = 999
    path = tmp_path / "app.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported app state version"):
        app_module.load_state(path)

    payload["version"] = app_module.CONFIG_VERSION
    payload["mood"] = "loud"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unknown mood"):
        app_module.load_state(path)


def test_missing_settings_create_a_file_but_bad_settings_do_not_overwrite(
    tmp_path: Path, controller: app_module.AppController
) -> None:
    assert controller.restore_saved_state() is False
    assert controller._config_path.exists()

    controller._config_path.write_text("{not json")
    assert controller.restore_saved_state() is False
    assert "unreadable" in (controller.state.error or "")
    assert controller._config_path.read_text() == "{not json"


def test_stale_preview_completions_are_ignored(
    tmp_path: Path, controller: app_module.AppController
) -> None:
    first_token, _first = controller.request_preview()
    second_token, second = controller.request_preview()
    assert controller.complete_preview(first_token, fake_result(second), 0.01) is None
    assert controller.state.preview_path is None

    path = controller.complete_preview(second_token, fake_result(second), 0.02)
    assert path is not None and path.exists()
    assert controller.state.preview_digest == f"fake-{second.slug()}"


def test_full_completion_stores_without_a_second_preview_render(
    tmp_path: Path, controller: app_module.AppController, monkeypatch
) -> None:
    from genesis import library as library_module

    calls = []

    def fail_preview(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("the controller must not render a separate preview")

    monkeypatch.setattr(library_module, "store", fail_preview)
    token, recipe = controller.request_full()
    entry = controller.complete_full(token, fake_result(recipe), 0.03)
    assert calls == []
    assert entry is not None
    assert entry.image_path.exists()
    assert entry.preview_path is not None and entry.preview_path.exists()
    assert controller.state.current_digest == entry.digest


def test_thumbnail_star_and_apply_use_saved_pieces(
    tmp_path: Path, controller: app_module.AppController
) -> None:
    token, recipe = controller.request_full()
    entry = controller.complete_full(token, fake_result(recipe), 0.03)
    assert entry is not None

    rows = controller.refresh_rows()
    assert [row.digest for row in rows] == [entry.digest]
    assert rows[0].starred is False
    assert controller.toggle_star(entry.digest) is True
    assert controller.toggle_star("missing") is None

    applied: list[tuple[Path, str]] = []

    def fake_apply(path: Path, theme: str = "both"):
        applied.append((Path(path), theme))
        return object()

    controller._wallpaper = type("Wallpaper", (), {"apply": staticmethod(fake_apply)})()
    controller.apply_entry(entry.digest)
    assert applied == [(entry.image_path, "both")]

    with pytest.raises(KeyError, match="no piece"):
        controller.apply_entry("missing")


def test_tuned_engine_settings_survive_an_engine_round_trip(
    controller: app_module.AppController,
) -> None:
    controller.set_tuned_value("params", "trails", 12345)
    controller.set_generator("attractor")
    assert controller.state.recipe.generator == "attractor"
    controller.set_generator("flowfield")
    assert controller.state.recipe.params["trails"] == 12345


def test_new_wallpaper_picks_a_curated_preset_in_the_current_mood(
    controller: app_module.AppController,
) -> None:
    controller.set_mood("calm")
    recipe, reason = controller.new_wallpaper(day=datetime.date(2026, 3, 4))
    assert reason.startswith("curated ")
    assert controller.state.recipe == recipe
    # Picking begins no render job: a job nobody runs would wedge `_active`
    # forever behind a stuck "busy". Rendering is the caller's job.
    assert controller.busy is False


def test_app_command_starts_the_native_interface(monkeypatch, tmp_path: Path, capsys) -> None:
    calls: dict[str, object] = {}

    def fake_run_app(*, root=None, config_path=None) -> int:
        calls["root"] = root
        calls["config_path"] = config_path
        return 7

    monkeypatch.setattr(app_module, "run_app", fake_run_app)
    code = cli_module.main(
        ["app", "--root", str(tmp_path / "lib"), "--config", str(tmp_path / "app.json")]
    )
    assert code == 7
    assert calls == {
        "root": tmp_path / "lib",
        "config_path": tmp_path / "app.json",
    }
    assert capsys.readouterr().out == ""

    def fail_app(*, root=None, config_path=None) -> int:
        raise ModuleNotFoundError("No module named 'gi'")

    monkeypatch.setattr(app_module, "run_app", fail_app)
    code = cli_module.main(
        ["app", "--root", str(tmp_path / "lib"), "--config", str(tmp_path / "app.json")]
    )
    assert code == 1
    assert "gi" in capsys.readouterr().err


class FakeLiveSession:
    """A session the tests fully control: blocks until stopped."""

    def __init__(self, config, recipe, seed, base_image=None, gate=None, fail_at=None):
        self.config = config
        self.recipe = recipe
        self.seed = seed
        self.base_image = base_image
        self.gate = gate
        self.fail_at = fail_at
        self.ran = False
        self.stop_event = threading.Event()
        self.stats = SimpleNamespace(
            frames=10, ema_ms=16.6, target_fps=60, degraded=False
        )

    def run(self):
        if self.fail_at == "run":
            raise RuntimeError("display vanished")
        self.ran = True
        self.stop_event.wait(timeout=30)

    def request_stop(self):
        self.stop_event.set()


def live_factory(sessions: list, **kwargs):
    def build(config, recipe, seed, base_image=None):
        if kwargs.get("raise_at_build"):
            raise RuntimeError("no display")
        session = FakeLiveSession(config, recipe, seed, base_image=base_image, **kwargs)
        sessions.append(session)
        gate = kwargs.get("gate")
        if gate is not None:
            gate.wait(timeout=30)
        return session

    return build


def wait_for(predicate, timeout: float = 10.0) -> bool:
    """Poll until predicate holds; thread timing must not flake the suite."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_live_settings_round_trip(tmp_path: Path) -> None:
    live = app_module.LiveSettings(
        cursor_mode="attract", idle_mode="ambient", fps=30, stir_strength=6.5
    )
    state = app_module.AppState(root=tmp_path / "lib", recipe=tiny_recipe(), live=live)
    restored = app_module.load_state(app_module.save_state(state, tmp_path / "app.json"))
    assert restored.live == live


def test_live_settings_reject_bad_values() -> None:
    with pytest.raises(ValueError, match="cursor mode"):
        app_module.LiveSettings(cursor_mode="vibrate")
    with pytest.raises(ValueError, match="idle mode"):
        app_module.LiveSettings(idle_mode="party")
    with pytest.raises(ValueError, match="fps"):
        app_module.LiveSettings(fps=0)
    with pytest.raises(ValueError, match="stir strength"):
        app_module.LiveSettings(stir_strength=-1)
    with pytest.raises(ValueError, match="cursor mode"):
        app_module.LiveSettings.from_dict({"cursor_mode": "loud"})


def test_live_start_stop_cycle(controller: app_module.AppController) -> None:
    sessions: list = []
    controller._live_session_factory = live_factory(sessions)
    controller.start_live()
    assert controller.live_status in ("starting", "running")
    assert wait_for(lambda: controller.live_status == "running")
    stats = controller.live_stats()
    assert stats is not None and stats["frames"] == 10
    controller.stop_live()
    assert controller.live_status == "stopped"
    assert controller.live_stats() is None
    assert sessions[0].ran is True


def test_live_start_twice_is_rejected(
    controller: app_module.AppController,
) -> None:
    sessions: list = []
    controller._live_session_factory = live_factory(sessions)
    controller.start_live()
    assert wait_for(lambda: controller.live_status == "running")
    with pytest.raises(ValueError, match="already running"):
        controller.start_live()
    controller.stop_live()


def test_live_build_failure_surfaces_instead_of_hanging(
    controller: app_module.AppController,
) -> None:
    controller._live_session_factory = live_factory([], raise_at_build=True)
    controller.start_live()
    assert controller.live_status in ("starting", "failed")
    assert wait_for(lambda: controller.live_status == "failed")
    assert "no display" in (controller.live_error or "")
    assert controller.live_stats() is None


def test_live_run_failure_surfaces_instead_of_vanishing(
    controller: app_module.AppController,
) -> None:
    sessions: list = []

    def build(config, recipe, seed, base_image=None):
        session = FakeLiveSession(config, recipe, seed, fail_at="run")
        sessions.append(session)
        return session

    controller._live_session_factory = build
    controller.start_live()
    assert wait_for(lambda: controller.live_status == "failed")
    assert "display vanished" in (controller.state.error or "")


def test_live_stop_during_startup_never_runs(
    controller: app_module.AppController,
) -> None:
    gate = threading.Event()
    sessions: list = []
    controller._live_session_factory = live_factory(sessions, gate=gate)
    controller.start_live()
    controller.stop_live()
    gate.set()
    assert wait_for(lambda: controller.live_status == "stopped")
    assert sessions and sessions[0].ran is False


def test_live_stop_when_stopped_is_a_no_op(
    controller: app_module.AppController,
) -> None:
    controller.stop_live()
    assert controller.live_status == "stopped"


def test_live_preferences_validate_and_persist(
    controller: app_module.AppController,
) -> None:
    controller.set_live_mode("attract")
    controller.set_live_idle("ambient")
    controller.set_live_fps(30)
    controller.set_live_stir_strength(6.0)
    assert controller.state.live == app_module.LiveSettings(
        cursor_mode="attract", idle_mode="ambient", fps=30, stir_strength=6.0
    )
    payload = json.loads(controller._config_path.read_text())
    assert payload["live"]["cursor_mode"] == "attract"
    with pytest.raises(ValueError, match="cursor mode"):
        controller.set_live_mode("vibrate")
    with pytest.raises(ValueError, match="idle mode"):
        controller.set_live_idle("party")
    with pytest.raises(ValueError, match="fps"):
        controller.set_live_fps(240)
    with pytest.raises(ValueError, match="stir strength"):
        controller.set_live_stir_strength(-1)


def test_live_output_size_uses_the_display_when_known(
    controller: app_module.AppController,
) -> None:
    controller._wallpaper = SimpleNamespace(display_size=lambda: (2560, 1440))
    assert controller._live_output_size() == (2560, 1440)
    assert (controller.live_config().width, controller.live_config().height) == (
        2560,
        1440,
    )


def test_live_output_size_falls_back_without_a_probe(
    controller: app_module.AppController,
) -> None:
    controller._wallpaper = SimpleNamespace()  # e.g. a stub without the probe
    assert controller._live_output_size() == (1920, 1080)


def test_start_live_refuses_wayland_upfront(
    controller: app_module.AppController, monkeypatch
) -> None:
    """No worker thread, no failed status: the refusal happens synchronously."""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    with pytest.raises(ValueError, match="X11"):
        controller.start_live()
    assert controller.live_status == "stopped"
    assert controller.live_available()[0] is False


def test_start_live_refuses_without_any_display(
    controller: app_module.AppController, monkeypatch
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(ValueError, match="DISPLAY is unset"):
        controller.start_live()
    assert controller.live_status == "stopped"


def test_live_available_is_true_on_x11(
    controller: app_module.AppController,
) -> None:
    assert controller.live_available() == (True, "")


def test_live_preview_needs_a_completed_preview(
    controller: app_module.AppController,
) -> None:
    assert controller.displayed_recipe is None
    assert controller.live_preview_frame() is None


def structured_result(recipe) -> RenderResult:
    """A preview with real structure: flat ink would never visibly move."""
    import numpy as np

    w, h = recipe.width, recipe.height
    row = (128 + 100 * np.sin(np.arange(w) / 6.0)).astype(np.uint8)
    band = np.tile(row, (h, 1))
    image = Image.fromarray(
        np.stack([band, band[::-1], (band // 2).astype(np.uint8)], axis=-1)
    )
    return RenderResult(recipe, image, f"structured-{recipe.slug()}")


def test_live_preview_frames_are_small_deterministic_motion(
    controller: app_module.AppController,
) -> None:
    token, _preview = controller.request_preview()
    controller.complete_preview(token, structured_result(controller.state.recipe), 0.01)
    first = controller.live_preview_frame()
    assert first is not None
    assert len(first) == 480 * 270 * 3
    second = controller.live_preview_frame()
    assert second is not None and second != first  # motion between ticks
    controller._live_preview = None  # same wallpaper replays the same motion
    assert controller.live_preview_frame() == first


def test_full_supersedes_a_pending_preview(
    controller: app_module.AppController,
) -> None:
    token, _preview = controller.request_preview()
    assert controller.busy is True
    token2, recipe, started = controller.supersede_with_full(
        controller.state.recipe, False
    )
    assert started is True
    assert token2 != token
    # The superseded preview completes into a token mismatch: ignored.
    assert controller.complete_preview(token, fake_result(recipe), 0.01) is None
    entry = controller.complete_full(token2, fake_result(recipe), 0.01)
    assert entry is not None
    assert controller.busy is False


def test_full_queues_behind_a_running_full(
    controller: app_module.AppController,
) -> None:
    token1, recipe = controller.request_full()
    _t2, _r2, started = controller.supersede_with_full(
        controller.state.recipe, True
    )
    assert started is False  # no doubling up; the request waits its turn
    controller.complete_full(token1, fake_result(recipe), 0.01)
    pending = controller.take_pending_full()  # the UI launches this next
    assert pending is not None and pending[1] is True
    _t3, _r3, started3 = controller.supersede_with_full(*pending)
    assert started3 is True


def test_full_completion_refreshes_preview_card_and_clears_error(
    controller: app_module.AppController,
) -> None:
    controller.state.error = "stale gripe"
    token, recipe = controller.request_full()
    entry = controller.complete_full(token, fake_result(recipe), 0.01)
    assert entry is not None
    assert controller.state.error is None
    assert controller.displayed_recipe == recipe
    assert controller.state.preview_path == entry.image_path
    assert controller.state.preview_digest == entry.digest


def test_new_wallpaper_restarts_a_running_live_session(
    controller: app_module.AppController,
) -> None:
    sessions: list = []
    controller._live_session_factory = live_factory(sessions)
    controller.start_live()
    assert wait_for(lambda: controller.live_status == "running")
    first = sessions[0]
    controller.new_wallpaper(day=datetime.date(2026, 3, 4))
    assert wait_for(lambda: controller.live_status == "running" and len(sessions) == 2)
    assert first.stop_event.is_set()
    assert sessions[1] is not first
    controller.stop_live()


def test_new_wallpaper_without_live_does_not_start_it(
    controller: app_module.AppController,
) -> None:
    sessions: list = []
    controller._live_session_factory = live_factory(sessions)
    controller.new_wallpaper(day=datetime.date(2026, 3, 4))
    assert sessions == []
    assert controller.live_status == "stopped"


def test_start_live_with_an_explicit_recipe_animates_that_composition(
    controller: app_module.AppController,
) -> None:
    sessions: list = []
    controller._live_session_factory = live_factory(sessions)
    recipe = tiny_recipe(seed=99, palette_name="ember")
    controller.start_live(recipe)
    assert wait_for(lambda: controller.live_status == "running")
    assert sessions[0].recipe.palette_name == "ember"
    assert sessions[0].recipe.width == 1920
    assert sessions[0].base_image is None
    # Going live tracks the animated recipe separately: the composition
    # being edited must survive untouched.
    assert controller.state.recipe.palette_name == "tide"
    assert controller.state.live_recipe is not None
    assert controller.state.live_recipe.palette_name == "ember"
    controller.stop_live()


def test_make_live_preserves_the_editing_recipe(
    controller: app_module.AppController,
) -> None:
    """Animating a saved piece must not replace the current composition."""
    sessions: list = []
    controller._live_session_factory = live_factory(sessions)
    before = controller.state.recipe
    controller.start_live(tiny_recipe(seed=99, palette_name="ember"))
    assert wait_for(lambda: controller.live_status == "running")
    assert controller.state.recipe == before
    assert controller.state.live_recipe is not None
    assert controller.state.live_recipe.palette_name == "ember"
    controller.stop_live()


def test_live_recipe_round_trips_through_saved_state(tmp_path: Path) -> None:
    live_recipe = tiny_recipe(seed=77, palette_name="ember")
    state = app_module.AppState(
        root=tmp_path / "lib", recipe=tiny_recipe(), live_recipe=live_recipe
    )
    restored = app_module.load_state(app_module.save_state(state, tmp_path / "app.json"))
    assert restored.live_recipe == live_recipe


def test_live_tuning_while_running_records_a_per_wallpaper_override(
    controller: app_module.AppController,
) -> None:
    sessions: list = []
    controller._live_session_factory = live_factory(sessions)
    controller.start_live(tiny_recipe(seed=99, palette_name="ember"))
    assert wait_for(lambda: controller.live_status == "running")
    controller.set_live_stir_strength(6.0)
    key = controller.state.live_recipe.composition_key()
    assert controller.state.live_overrides[key].stir_strength == 6.0
    assert controller.state.live.stir_strength == 5.0  # global untouched
    controller.stop_live()


def test_live_tuning_while_stopped_moves_the_global_default(
    controller: app_module.AppController,
) -> None:
    controller.set_live_stir_strength(6.0)
    assert controller.state.live.stir_strength == 6.0
    assert controller.state.live_overrides == {}


def test_live_overrides_follow_the_composition_across_sizes(
    controller: app_module.AppController,
) -> None:
    """An override keyed by composition applies at any render size."""
    controller._wallpaper = SimpleNamespace(display_size=lambda: (1280, 720))
    sessions: list = []
    controller._live_session_factory = live_factory(sessions)
    controller.start_live(tiny_recipe(seed=41, palette_name="ember"))
    assert wait_for(lambda: controller.live_status == "running")
    controller.set_live_mode("attract")
    key = controller.state.live_recipe.composition_key()
    controller.stop_live()

    big = tiny_recipe(seed=41, palette_name="ember", width=1920, height=1080)
    assert big.composition_key() == key
    assert big.slug() != controller.state.live_recipe.slug()
    controller.start_live(big)
    assert wait_for(lambda: controller.live_status == "running")
    assert sessions[-1].config.cursor_mode == "attract"
    controller.stop_live()


def test_live_config_prefers_override_over_global(
    controller: app_module.AppController,
) -> None:
    key = tiny_recipe(seed=5).composition_key()
    controller.state.live_overrides[key] = app_module.LiveSettings(fps=30)
    assert controller.live_config(key).fps == 30
    assert controller.live_config("unknown-key").fps == 60


def test_live_overrides_round_trip(tmp_path: Path) -> None:
    key = tiny_recipe(seed=5).composition_key()
    state = app_module.AppState(
        root=tmp_path / "lib",
        recipe=tiny_recipe(),
        live_overrides={key: app_module.LiveSettings(fps=30)},
    )
    restored = app_module.load_state(app_module.save_state(state, tmp_path / "app.json"))
    assert restored.live_overrides == state.live_overrides


def test_live_motion_bundle_applies_and_names_itself(
    controller: app_module.AppController,
) -> None:
    controller.set_live_motion("calm")
    assert controller.state.live.stir_strength == 3.0
    assert controller.state.live.ambient_speed == 0.8
    assert app_module.AppController.live_motion_name(controller.state.live) == "calm"
    controller.set_live_stir_strength(7.0)  # any divergence becomes custom
    assert app_module.AppController.live_motion_name(controller.state.live) == "custom"
    with pytest.raises(ValueError, match="unknown motion"):
        controller.set_live_motion("turbo")


def test_live_motion_bundles_are_valid_engine_configs() -> None:
    from genesis.live.config import MOTION_PRESETS, LiveConfig

    for bundle in MOTION_PRESETS.values():
        config = LiveConfig(**bundle)
        assert config.stir_strength > 0


def test_live_feel_settings_reject_bad_values() -> None:
    with pytest.raises(ValueError, match="glow"):
        app_module.LiveSettings(cursor_glow=0.9)
    with pytest.raises(ValueError, match="ambient"):
        app_module.LiveSettings(ambient_speed=9.0)
    with pytest.raises(ValueError, match="relax"):
        app_module.LiveSettings(relax=0.5)


def test_live_config_plumbs_feel_to_the_engine(
    controller: app_module.AppController,
) -> None:
    controller.set_live_glow(0.25)
    controller.set_live_ambient(2.5)
    config = controller.live_config()
    assert config.cursor_glow == 0.25
    assert config.ambient_speed == 2.5


def test_effective_settings_follow_the_running_override(
    controller: app_module.AppController,
) -> None:
    sessions: list = []
    controller._live_session_factory = live_factory(sessions)
    assert controller.effective_live_settings() == controller.state.live
    controller.start_live(tiny_recipe(seed=99, palette_name="ember"))
    assert wait_for(lambda: controller.live_status == "running")
    controller.set_live_glow(0.4)
    assert controller.effective_live_settings().cursor_glow == 0.4
    assert controller.state.live.cursor_glow == 0.1  # global default untouched
    controller.stop_live()


def test_start_live_from_entry_uses_the_stored_png(
    controller: app_module.AppController,
) -> None:
    import numpy as np
    from PIL import Image

    from genesis import library as library_module

    recipe = tiny_recipe(seed=21, palette_name="ember")
    image = Image.new("RGB", (96, 54), (200, 100, 50))
    with library_module.library_at(controller.state.root) as library:
        image_path = library.images / "manual.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        entry = library.add(recipe, "digest-stored-png", image_path)
    sessions: list = []
    controller._live_session_factory = live_factory(sessions)
    controller.start_live_from_entry(entry.digest)
    assert wait_for(lambda: controller.live_status == "running")
    passed = sessions[0].base_image
    assert passed is not None
    expected = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    assert passed.shape == expected.shape
    assert np.array_equal(passed, expected)
    controller.stop_live()


def test_start_live_from_entry_rejects_missing_pieces(
    controller: app_module.AppController,
) -> None:
    with pytest.raises(KeyError, match="no piece"):
        controller.start_live_from_entry("missing")


def test_library_entry_fetch_and_missing(
    controller: app_module.AppController,
) -> None:
    from genesis import library as library_module
    from genesis import render as render_module

    recipe = tiny_recipe(seed=21)
    with library_module.library_at(controller.state.root) as library:
        entry = library_module.store(
            library, recipe, render_module.render(recipe)
        )
    assert controller.library_entry(entry.digest).digest == entry.digest
    with pytest.raises(KeyError, match="no piece"):
        controller.library_entry("missing")


def test_start_live_animates_the_displayed_preview_not_newer_state(
    controller: app_module.AppController,
) -> None:
    """Start-live must animate what the user is looking at.

    If a newer preview was requested but has not completed, the on-screen
    image still shows the older recipe — and that is what must animate.
    Animating state.recipe instead is how a green preview turns into a gray
    desktop.
    """
    sessions: list = []
    controller._live_session_factory = live_factory(sessions)
    token, _preview = controller.request_preview()
    controller.complete_preview(token, fake_result(controller.state.recipe), 0.01)
    displayed = controller.displayed_recipe
    assert displayed is not None
    assert displayed.palette_name == "tide"

    # State moves on (a newer preview was requested) while the screen still
    # shows the completed one.
    controller.state.recipe = tiny_recipe(seed=99, palette_name="ember")
    controller.start_live()
    assert wait_for(lambda: controller.live_status == "running")
    assert sessions[0].recipe.palette_name == "tide"
    # The newer editing state survives; the live recipe tracks the animation.
    assert controller.state.recipe.palette_name == "ember"
    assert controller.state.live_recipe is not None
    assert controller.state.live_recipe.palette_name == "tide"
    assert sessions[0].recipe.slug() in controller.state.status
    controller.stop_live()


def test_stale_preview_completion_keeps_the_displayed_recipe(
    controller: app_module.AppController,
) -> None:
    old_token, _old_preview = controller.request_preview()
    controller.state.recipe = tiny_recipe(seed=99, palette_name="ember")
    new_token, new_preview = controller.request_preview()
    # The older completion arrives late: rejected, and must not touch what
    # is displayed.
    assert controller.complete_preview(old_token, fake_result(new_preview), 0.01) is None
    assert controller.displayed_recipe is None
    controller.complete_preview(new_token, fake_result(new_preview), 0.01)
    assert controller.displayed_recipe is not None
    assert controller.displayed_recipe.palette_name == "ember"


def test_new_wallpaper_restart_animates_the_fresh_recipe(
    controller: app_module.AppController,
) -> None:
    """The restart after New wallpaper must use the fresh pick, never a stale
    preview: new_wallpaper passes its recipe explicitly for exactly this."""
    from types import SimpleNamespace

    sessions: list = []
    controller._live_session_factory = live_factory(sessions)
    controller.start_live()
    assert wait_for(lambda: controller.live_status == "running")
    assert sessions[0].recipe.palette_name == "tide"

    fresh = tiny_recipe(seed=99, palette_name="ember")
    controller._scheduler = SimpleNamespace(
        pick_curated=lambda *args, **kwargs: (fresh, "stubbed")
    )
    controller.new_wallpaper(day=datetime.date(2026, 3, 4))
    assert wait_for(lambda: controller.live_status == "running" and len(sessions) == 2)
    assert sessions[1].recipe.palette_name == "ember"
    assert sessions[1].recipe.params == fresh.params
    controller.stop_live()

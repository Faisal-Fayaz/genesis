"""The native desktop interface.

The browser gallery remains available, but this module is now the primary way
to choose a wallpaper. It separates the parts that need a display from the
parts that do not:

- `AppState` and `AppController` know nothing about GTK. They own the current
  recipe, the Tune layout, library rows, wallpaper actions, and persisted
  settings, so tests can exercise them without a display server.
- `run_app` imports GTK and Adwaita only when the interface actually starts.
  Importing this module must stay safe on a headless machine.

The interface itself is deliberately small. Above the fold there is a mood row,
a large preview, and one primary `New wallpaper` action. Star, Set, and Make
live sit directly on the library thumbnails. All generator and post controls
live in a collapsed Tune panel, grouped by intent rather than by where they
happen to be stored.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import post as post_module
from . import presets as presets_module
from . import registry
from . import render as render_module
from . import scheduler as scheduler_module
from . import wallpaper as wallpaper_module
from .library import Entry, Library
from .live import config as live_config_module
from .recipe import PRESETS, Recipe
from .render import RenderResult

Scope = Literal["params", "post"]
Renderer = Callable[[Recipe], RenderResult]
SeedSource = Callable[[], int]

APP_ID = "org.genesis.wallpapers"
APP_TITLE = "Genesis"
CONFIG_VERSION = 1
MOOD_ALL = "all"
MOODS = (MOOD_ALL, *presets_module.MOODS)
DEFAULT_SIZE = (1920, 1080)
PREVIEW_EDGE = 640
# In-app motion preview: small, slow, sim-only (no cursor, no window).
LIVE_PREVIEW_WIDTH = 480
LIVE_PREVIEW_HEIGHT = 270
LIVE_PREVIEW_STEPS_PER_TICK = 4
PREVIEW_THUMBNAIL_EDGE = 480
LIBRARY_ROW_LIMIT = 24
OUTPUT_SIZES = ((1280, 720), DEFAULT_SIZE, (2560, 1440))
CONFIG_ENVIRONMENT_VARIABLE = "XDG_CONFIG_HOME"


@dataclass(frozen=True)
class TuneControl:
    """One control in the Tune panel."""

    scope: Scope
    name: str
    label: str
    kind: str
    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class TuneGroup:
    """One collapsed section of related controls."""

    key: str
    title: str
    controls: tuple[TuneControl, ...]


TUNE_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "shape",
        "Shape",
        ("form", "a", "b", "c", "d", "scale", "detail", "turns", "swirl", "extent",
         "blobs", "spread", "turbulence", "fray"),
    ),
    (
        "motion",
        "Motion",
        ("trails", "points", "steps", "step_size", "substeps", "warmup"),
    ),
    (
        "colour",
        "Colour",
        (
            "color_count",
            "hue_bias",
            "exposure",
            "contrast",
            "saturation",
            "brightness",
            "lift",
        ),
    ),
    ("frame", "Frame", ("zoom", "pan_x", "pan_y", "fit")),
    (
        "finish",
        "Finish",
        ("grain", "grain_size", "vignette", "bloom", "bloom_radius", "supersample"),
    ),
)


@dataclass(frozen=True)
class PendingRender:
    """A render that has been requested but has not completed."""

    token: int
    recipe: Recipe
    kind: str
    apply_after: bool = False


@dataclass(frozen=True)
class LibraryRow:
    """The thumbnail state the interface needs for one saved piece."""

    digest: str
    slug: str
    generator: str
    palette: str
    starred: bool
    created_at: float
    image_path: Path
    preview_path: Path | None


@dataclass(frozen=True)
class LiveSettings:
    """Persisted live-wallpaper preferences.

    Validated eagerly so a hand-edited settings file fails loudly at load
    instead of starting a session with nonsense values. Runtime status (is a
    session running? what is it doing?) lives on the controller, not here.
    Motion feel (glow, ambient, relax) lives here too so per-wallpaper
    overrides and bundles cover the whole look, not just the stir.
    """

    cursor_mode: str = "repel"
    idle_mode: str = "pause"
    fps: int = 60
    stir_strength: float = 5.0
    cursor_glow: float = 0.1
    ambient_speed: float = 1.5
    relax: float = 0.004
    relax_idle: float = 0.02

    def __post_init__(self) -> None:
        if self.cursor_mode not in live_config_module.CURSOR_MODES:
            raise ValueError(
                f"unknown cursor mode {self.cursor_mode!r}; "
                f"expected one of {list(live_config_module.CURSOR_MODES)}"
            )
        if self.idle_mode not in live_config_module.IDLE_MODES:
            raise ValueError(
                f"unknown idle mode {self.idle_mode!r}; "
                f"expected one of {list(live_config_module.IDLE_MODES)}"
            )
        if not 1 <= self.fps <= 120:
            raise ValueError(f"fps must be in 1..120, got {self.fps!r}")
        if self.stir_strength < 0:
            raise ValueError(
                f"stir strength must be non-negative, got {self.stir_strength!r}"
            )
        if not 0.0 <= self.cursor_glow <= live_config_module.MAX_CURSOR_GLOW:
            raise ValueError(
                f"cursor glow must be in 0..{live_config_module.MAX_CURSOR_GLOW}, "
                f"got {self.cursor_glow!r}"
            )
        if not 0.0 <= self.ambient_speed <= live_config_module.MAX_AMBIENT_SPEED:
            raise ValueError(
                f"ambient speed must be in 0..{live_config_module.MAX_AMBIENT_SPEED}, "
                f"got {self.ambient_speed!r}"
            )
        if not 0.0 <= self.relax <= 0.1:
            raise ValueError(f"relax must be in 0..0.1, got {self.relax!r}")
        if not 0.0 <= self.relax_idle <= 0.1:
            raise ValueError(f"relax idle must be in 0..0.1, got {self.relax_idle!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cursor_mode": self.cursor_mode,
            "idle_mode": self.idle_mode,
            "fps": self.fps,
            "stir_strength": self.stir_strength,
            "cursor_glow": self.cursor_glow,
            "ambient_speed": self.ambient_speed,
            "relax": self.relax,
            "relax_idle": self.relax_idle,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LiveSettings:
        if not isinstance(data, dict):
            raise TypeError("live settings must be a JSON object")
        return cls(
            cursor_mode=data.get("cursor_mode", "repel"),
            idle_mode=data.get("idle_mode", "pause"),
            fps=data.get("fps", 60),
            stir_strength=data.get("stir_strength", 5.0),
            cursor_glow=data.get("cursor_glow", 0.1),
            ambient_speed=data.get("ambient_speed", 1.5),
            relax=data.get("relax", 0.004),
            relax_idle=data.get("relax_idle", 0.02),
        )


LIVE_STATUSES = ("stopped", "starting", "running", "failed")

# Every LiveSettings field a control may write: the preference setters merge
# over exactly these, so adding a knob here is enough to make it routable
# (global vs per-wallpaper) and persistable.
LIVE_TUNABLES = (
    "cursor_mode",
    "idle_mode",
    "fps",
    "stir_strength",
    "cursor_glow",
    "ambient_speed",
    "relax",
    "relax_idle",
)


@dataclass
class AppState:
    """Everything the interface needs to restart where it left off."""

    root: Path
    recipe: Recipe
    mood: str = MOOD_ALL
    preview_edge: int = PREVIEW_EDGE
    preview_path: Path | None = None
    preview_digest: str | None = None
    current_digest: str | None = None
    status: str = "Choose a mood, then make a wallpaper."
    error: str | None = None
    live: LiveSettings = field(default_factory=LiveSettings)
    live_recipe: Recipe | None = None
    live_overrides: dict[str, LiveSettings] = field(default_factory=dict)


def default_recipe(width: int = DEFAULT_SIZE[0], height: int = DEFAULT_SIZE[1]) -> Recipe:
    """The first image shown before anything has been generated or saved."""
    return presets_module.get("silk").recipe(width, height)


def default_state(
    root: str | Path | None = None,
    width: int = DEFAULT_SIZE[0],
    height: int = DEFAULT_SIZE[1],
    mood: str = MOOD_ALL,
) -> AppState:
    """A valid initial state, even when the library is empty."""
    if mood not in MOODS:
        raise ValueError(f"unknown mood {mood!r}; expected one of {list(MOODS)}")
    return AppState(
        root=Path(root) if root is not None else Library().root,
        recipe=default_recipe(width, height),
        mood=mood,
    )


def config_path(config_root: str | Path | None = None) -> Path:
    """Where the interface persists its non-image settings."""
    base = config_root or os.environ.get(CONFIG_ENVIRONMENT_VARIABLE) or Path.home() / ".config"
    return Path(base) / "genesis" / "app.json"


def state_to_json(state: AppState) -> dict[str, Any]:
    """A JSON-serializable form of the interface state."""
    return {
        "version": CONFIG_VERSION,
        "root": str(state.root),
        "mood": state.mood,
        "preview_edge": state.preview_edge,
        "preview_path": str(state.preview_path) if state.preview_path else None,
        "preview_digest": state.preview_digest,
        "current_digest": state.current_digest,
        "status": state.status,
        "live": state.live.to_dict(),
        "live_recipe": state.live_recipe.to_dict() if state.live_recipe else None,
        "live_overrides": {
            slug: settings.to_dict()
            for slug, settings in state.live_overrides.items()
        },
        "recipe": state.recipe.to_dict(),
    }


def state_from_json(data: dict[str, Any], root: str | Path | None = None) -> AppState:
    """Rebuild interface state, validating every field a person could edit."""
    if not isinstance(data, dict):
        raise TypeError("app state must be a JSON object")
    if data.get("version") != CONFIG_VERSION:
        raise ValueError(f"unsupported app state version {data.get('version')!r}")
    mood = data.get("mood", MOOD_ALL)
    if mood not in MOODS:
        raise ValueError(f"unknown mood {mood!r}; expected one of {list(MOODS)}")
    preview_edge = data.get("preview_edge", PREVIEW_EDGE)
    if not isinstance(preview_edge, int) or not 160 <= preview_edge <= 1280:
        raise ValueError(f"preview edge must be an integer in 160..1280, got {preview_edge!r}")

    state = AppState(
        root=Path(root or data.get("root") or Library().root),
        recipe=Recipe.from_dict(data["recipe"]),
        mood=mood,
        preview_edge=preview_edge,
        preview_path=Path(data["preview_path"]) if data.get("preview_path") else None,
        preview_digest=data.get("preview_digest"),
        current_digest=data.get("current_digest"),
        status=str(data.get("status") or ""),
        live=LiveSettings.from_dict(data.get("live") or {}),
        live_recipe=(
            Recipe.from_dict(data["live_recipe"]) if data.get("live_recipe") else None
        ),
        live_overrides=_live_overrides_from_json(data.get("live_overrides")),
    )
    if state.preview_digest is not None and not isinstance(state.preview_digest, str):
        raise ValueError("preview digest must be a string or null")
    if state.current_digest is not None and not isinstance(state.current_digest, str):
        raise ValueError("current digest must be a string or null")
    return state


def _live_overrides_from_json(data: Any) -> dict[str, LiveSettings]:
    """Per-wallpaper live preferences from saved state, validated."""
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError("live overrides must be a JSON object")
    overrides = {}
    for slug, payload in data.items():
        if not isinstance(slug, str):
            raise TypeError("live override keys must be composition keys")
        overrides[slug] = LiveSettings.from_dict(payload)
    return overrides


def save_state(state: AppState, path: str | Path) -> Path:
    """Write interface state atomically, so a crash cannot leave half a file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    staging.write_text(json.dumps(state_to_json(state), indent=2, sort_keys=True) + "\n")
    staging.replace(target)
    return target


def load_state(path: str | Path, root: str | Path | None = None) -> AppState:
    """Read previously saved interface state."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read app state {path}: {exc}") from exc
    try:
        return state_from_json(data, root)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid app state {path}: {exc}") from exc


def tune_layout(recipe: Recipe) -> tuple[TuneGroup, ...]:
    """Group the active controls by intent, not by storage location.

    The Tune panel must look the same whether the current recipe uses
    `flowfield`, `attractor`, or `splash`: Shape always means composition, Motion always
    means how ink moves, Colour always means tone, Frame always means cropping,
    and Finish always means final processing. Generator-specific controls appear
    only when their generator is active, and unavailable controls carry the
    reason so the interface can explain rather than silently ignore them.
    """
    registry.ensure_loaded()
    generator = registry.get(recipe.generator)
    generator_specs = {spec.name: spec for spec in generator.schema}
    post_specs = {spec.name: spec for spec in post_module.SCHEMA}
    values = {**recipe.params, **recipe.post}

    groups: list[TuneGroup] = []
    for key, title, members in TUNE_GROUPS:
        controls: list[TuneControl] = []
        for name in members:
            candidates: list[tuple[Scope, Any]] = []
            if name in generator_specs:
                candidates.append(("params", generator_specs[name]))
            if name in post_specs:
                candidates.append(("post", post_specs[name]))
            for scope, spec in candidates:
                label = spec.display_label
                if name == "contrast":
                    label = f"{label} ({'engine' if scope == 'params' else 'grade'})"
                controls.append(
                    TuneControl(
                        scope=scope,
                        name=name,
                        label=label,
                        kind=spec.kind,
                        available=spec.is_available(values),
                        reason=spec.unavailable_reason(values),
                    )
                )
        groups.append(TuneGroup(key=key, title=title, controls=tuple(controls)))
    return tuple(groups)


def _default_live_session(
    config: Any, recipe: Recipe, seed: int | None, base_image: Any = None
) -> Any:
    """Build a real live session. Imported lazily so tests stay headless."""
    from .live import service as live_service_module

    return live_service_module.LiveSession(config, recipe, seed, base_image)


class AppController:
    """Own every state transition the native interface can make.

    Rendering itself stays outside this class so the interface can run it on a
    worker thread: `request_preview` and `request_full` return a token and a
    recipe snapshot, and the matching `complete_*` method is called later on
    the main thread. Stale completions return `None` instead of clobbering a
    newer request, which is what keeps rapid slider movement from showing an
    old image.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        state: AppState | None = None,
        *,
        library: Library | None = None,
        renderer: Renderer = render_module.render,
        scheduler: Any = scheduler_module,
        wallpaper: Any = wallpaper_module,
        seed_source: SeedSource | None = None,
        clock: Callable[[], float] = time.perf_counter,
        app_config_path: str | Path | None = None,
        live_session_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.state = state or default_state(root)
        if root is not None:
            self.state.root = Path(root)
        self._library = library or Library(self.state.root)
        self._library.open()
        self._renderer = renderer
        self._scheduler = scheduler
        self._wallpaper = wallpaper
        self._seed_source = seed_source or (
            lambda: int.from_bytes(os.urandom(4), "big")
        )
        self._clock = clock
        self._config_path = Path(app_config_path) if app_config_path else config_path()
        self._active: PendingRender | None = None
        self._next_token = 1
        # One queued full render, set when a full is requested while another
        # full runs. Preview jobs are superseded outright instead of queued:
        # at most one preview matters (the newest), while a requested full
        # render is a promise the interface must keep.
        self._pending_full: tuple[Recipe, bool] | None = None
        self._tuned_params: dict[str, dict[str, Any]] = {}
        self._live_session_factory = live_session_factory or _default_live_session
        self._live_session: Any | None = None
        self._live_thread: threading.Thread | None = None
        self._live_stop_requested = False
        self.live_status: str = "stopped"
        self.live_error: str | None = None
        # The full-size recipe the on-screen preview was rendered from, or
        # None if no preview has completed yet this run. Start-live animates
        # this — what the user is actually looking at — rather than whatever
        # state.recipe has moved on to while a newer preview was rendering.
        self.displayed_recipe: Recipe | None = None
        self._preview_source_recipe: Recipe | None = None
        # Cached motion-preview fluid state as (composition key, state).
        # Keyed so a new wallpaper rebuilds it; a few kilobytes that free
        # themselves by replacement.
        self._live_preview: tuple[str, Any] | None = None

    def close(self) -> None:
        """Release the library connection held for the interface lifetime."""
        self._library.close()

    @property
    def busy(self) -> bool:
        """Whether a render is currently in flight."""
        return self._active is not None

    def save(self) -> Path:
        """Persist the current interface state."""
        return save_state(self.state, self._config_path)

    def restore_saved_state(self) -> bool:
        """Load settings saved by a previous run, when they exist and parse.

        A corrupt settings file does not prevent startup: the interface falls
        back to defaults and reports the problem in its status line.
        """
        try:
            saved = load_state(self._config_path, self.state.root)
        except FileNotFoundError:
            self.save()
            return False
        except ValueError as exc:
            self.state.error = f"Saved settings were unreadable and were ignored: {exc}"
            return False
        self.state = saved
        self._remember_tuned_params()
        return True

    def render_snapshot(self, recipe: Recipe) -> RenderResult:
        """Run the injected renderer for one recipe snapshot."""
        return self._renderer(recipe)

    def set_mood(self, mood: str) -> None:
        """Choose which curated mood `New wallpaper` draws from."""
        if mood not in MOODS:
            raise ValueError(f"unknown mood {mood!r}; expected one of {list(MOODS)}")
        self.state.mood = mood
        self.state.error = None
        self.save()

    def set_output_size(self, width: int, height: int) -> None:
        """Change the full render size without touching the composition."""
        self.state.recipe = self.state.recipe.with_changes(width=width, height=height)
        self._mark_preview_stale("Output size changed; rendering a new preview.")
        self.save()

    def set_palette(self, palette_name: str) -> None:
        """Change palettes without touching the composition."""
        if palette_name not in PRESETS:
            raise ValueError(f"unknown palette {palette_name!r}")
        self.state.recipe = self.state.recipe.with_changes(palette_name=palette_name)
        self._mark_preview_stale("Palette changed; rendering a new preview.")
        self.save()

    def set_generator(self, generator_name: str) -> None:
        """Change engines while keeping the settings that still apply."""
        registry.ensure_loaded()
        target = registry.get(generator_name)
        current = self.state.recipe
        remembered = self._tuned_params.get(generator_name, {})
        params: dict[str, Any] = {}
        for spec in target.schema:
            if spec.name in remembered:
                raw = remembered[spec.name]
            elif spec.name in current.params:
                raw = current.params[spec.name]
            else:
                continue
            try:
                params[spec.name] = spec.coerce(raw)
            except (TypeError, ValueError):
                continue
        merged = target.schema.validate(params)
        self.state.recipe = current.with_changes(generator=generator_name, params=merged)
        self._remember_tuned_params()
        self._mark_preview_stale("Engine changed; rendering a new preview.")
        self.save()

    def set_tuned_value(self, scope: Scope, name: str, value: Any) -> None:
        """Apply one Tune control and invalidate the current preview."""
        recipe = self.state.recipe
        if scope == "params":
            registry.ensure_loaded()
            spec = registry.get(recipe.generator).schema.get(name)
            params = {**recipe.params, name: spec.coerce(value)}
            self.state.recipe = recipe.with_changes(params=params)
        elif scope == "post":
            spec = post_module.SCHEMA.get(name)
            post = {**recipe.post, name: spec.coerce(value)}
            self.state.recipe = recipe.with_changes(post=post)
        else:
            raise ValueError(f"unknown tune scope {scope!r}")
        self._remember_tuned_params()
        self._mark_preview_stale("Settings changed; rendering a new preview.")
        self.save()

    def new_wallpaper(self, day: Any = None) -> tuple[Recipe, str]:
        """Pick a fresh curated recipe in the current mood.

        Curated is the only source here because it is the quality floor: every
        preset has been looked at and kept. Random draws remain available
        through `gen-random` for explorations the curated set cannot cover.

        Returns the recipe and the reason; rendering is the caller's job.
        This deliberately begins no preview job: a job nobody runs would sit
        in `_active` forever, wedging every later render behind a stuck
        "busy". (That is exactly the wedge this used to be.)
        """
        mood = None if self.state.mood == MOOD_ALL else self.state.mood
        recipe, reason = self._scheduler.pick_curated(
            self._library,
            day=day,
            width=self.state.recipe.width,
            height=self.state.recipe.height,
            mood=mood,
            seed=self._seed_source(),
            theme=self.state.recipe.theme,
        )
        self.state.recipe = recipe
        self._remember_tuned_params()
        self.state.error = None
        self.state.status = f"New wallpaper: {reason}."
        if self.live_status == "running":
            # A new still means a new live scene: restart the session on the
            # fresh recipe rather than animating something stale or stopping
            # silently. The fresh recipe is passed explicitly so the restart
            # cannot fall back to an older displayed preview. If the restart
            # fails, the error surfaces and live is left stopped, never
            # half-running.
            self.stop_live()
            try:
                self.start_live(recipe)
            except (OSError, ValueError, KeyError) as exc:
                self.live_status = "failed"
                self.live_error = str(exc)
        self.save()
        return recipe, reason

    def _live_output_size(self) -> tuple[int, int]:
        """Live output size: the real display when known, else full HD.

        `getattr` (not a direct call) because tests swap in a stub wallpaper
        module carrying only `apply`; a stub without the probe gets the
        fallback rather than an AttributeError.
        """
        probe = getattr(self._wallpaper, "display_size", None)
        size = probe() if callable(probe) else None
        return size if size is not None else (1920, 1080)

    def live_config(self, composition_key: str | None = None) -> Any:
        """The engine configuration matching current live preferences.

        A per-wallpaper override wins when one exists for `composition_key`:
        silk wants gentle repel while ember wants a strong attract, and one
        global setting fights every wallpaper but one. Unknown keys fall back
        to the global preferences. The key is size-independent, so an
        override survives display changes and the preview/full-size split.
        """
        settings = self.state.live
        if composition_key is not None:
            settings = self.state.live_overrides.get(composition_key, settings)
        width, height = self._live_output_size()
        return live_config_module.LiveConfig(
            width=width,
            height=height,
            fps=settings.fps,
            cursor_mode=settings.cursor_mode,
            idle_mode=settings.idle_mode,
            stir_strength=settings.stir_strength,
            cursor_glow=settings.cursor_glow,
            ambient_speed=settings.ambient_speed,
            relax=settings.relax,
            relax_idle=settings.relax_idle,
        )

    @staticmethod
    def live_available() -> tuple[bool, str]:
        """Whether this session can host live wallpapers, and why not.

        Static and GTK-free so the interface can disable live controls
        upfront instead of failing seconds later inside a worker thread.
        Mirrors the CLI's refusal wording.
        """
        if not os.environ.get("DISPLAY"):
            if os.environ.get("WAYLAND_DISPLAY"):
                reason = (
                    "live wallpapers need an X11 session; Wayland has no "
                    "equivalent mechanism (use New wallpaper for a static one)"
                )
                return False, reason
            return False, "live wallpapers need an X11 display (DISPLAY is unset)"
        return True, ""

    def start_live(self, recipe: Recipe | None = None, base_image: Any = None) -> None:
        """Start animating a recipe.

        With no explicit recipe this animates the displayed preview — what the
        user is actually looking at — falling back to the current recipe only
        when no preview has completed yet. Passing a recipe is how a saved
        library piece becomes live: the thumbnail's "Make live" action hands
        over that exact composition instead of whatever happens to be current.
        `base_image` is an already-loaded still (float32 sRGB) used verbatim
        as the base layer, skipping the one-time re-render. Raises if already
        running.

        Session construction (field build plus pre-roll) happens on a worker
        thread so the interface never blocks; status reads `starting` until the
        session is running.

        The animated recipe is tracked separately as `state.live_recipe`: going
        live must never rewrite the recipe being edited, or a click on an old
        thumbnail would silently replace the current composition.
        """
        if self._live_session is not None or self._live_thread_is_active():
            raise ValueError("live wallpaper is already running")
        available, reason = self.live_available()
        if not available:
            raise ValueError(reason)
        if recipe is None:
            recipe = self.displayed_recipe or self.state.recipe
        width, height = self._live_output_size()
        live_recipe = recipe.with_changes(width=width, height=height)
        self.state.live_recipe = live_recipe
        self._live_stop_requested = False
        self.live_status = "starting"
        self.live_error = None
        self.state.status = f"Starting live wallpaper: {live_recipe.slug()}."
        snapshot = live_recipe
        thread = threading.Thread(
            target=self._start_live_session, args=(snapshot, base_image), daemon=True
        )
        self._live_thread = thread
        thread.start()
        self.save()

    def stop_live(self) -> None:
        """Stop the live session. A no-op when nothing is running."""
        self._live_stop_requested = True
        session, self._live_session = self._live_session, None
        thread, self._live_thread = self._live_thread, None
        if session is not None:
            session.request_stop()
        if thread is not None:
            thread.join(timeout=30.0)
        self.live_status = "stopped"
        self.state.status = "Live wallpaper stopped."
        self.save()

    def _live_thread_is_active(self) -> bool:
        thread = self._live_thread
        return thread is not None and thread.is_alive()

    def start_live_from_entry(self, digest: str) -> None:
        """Animate a saved piece using its stored PNG as the base layer.

        Loading the file instead of re-rendering makes startup instant and
        guarantees the session starts from those exact pixels.
        """
        entry = self.library_entry(digest)
        if not entry.image_path.exists():
            raise FileNotFoundError(
                f"missing image for {digest!r}: {entry.image_path}"
            )
        from PIL import Image

        with Image.open(entry.image_path) as picture:
            base_image = np.asarray(picture.convert("RGB"), dtype=np.float32) / 255.0
        self.stop_live()
        self.start_live(entry.recipe_object, base_image=base_image)

    def _start_live_session(self, recipe: Recipe, base_image: Any = None) -> None:
        """Worker-thread entry: build the session, then run it to completion."""
        try:
            session = self._live_session_factory(
                self.live_config(recipe.composition_key()), recipe, None, base_image
            )
        except Exception as exc:  # noqa: BLE001 - a stillborn session must surface
            if not self._live_stop_requested:
                self.live_status = "failed"
                self.live_error = str(exc)
                self.state.error = f"Live wallpaper failed to start: {exc}"
            else:
                self.live_status = "stopped"
            self._live_thread = None
            return
        if self._live_stop_requested:
            self.live_status = "stopped"
            self._live_thread = None
            return
        self._live_session = session
        self.live_status = "running"
        try:
            session.run()
        except Exception as exc:  # noqa: BLE001 - a dead session must surface, not vanish
            self.live_status = "failed"
            self.live_error = str(exc)
            self.state.error = f"Live wallpaper failed: {exc}"
        else:
            self.live_status = "stopped"
        finally:
            if self._live_session is session:
                self._live_session = None
            self._live_thread = None

    def _set_live_settings(self, **fields: Any) -> None:
        """Write live preferences: per-wallpaper while running, global else.

        Tuning mid-session records an override for exactly the composition on
        screen (layered over the previous override for that wallpaper, if
        any); tuning while stopped moves the global default everything new
        starts from.
        """
        current = self.state.live
        key = None
        if self.live_status == "running" and self.state.live_recipe:
            key = self.state.live_recipe.composition_key()
            current = self.state.live_overrides.get(key, current)
        merged = {
            field: getattr(current, field) for field in LIVE_TUNABLES
        } | fields
        settings = LiveSettings(**merged)
        if key is None:
            self.state = replace(self.state, live=settings)
        else:
            overrides = {**self.state.live_overrides, key: settings}
            self.state = replace(self.state, live_overrides=overrides)
        self.save()

    def set_live_mode(self, mode: str) -> None:
        """Choose how the cursor pushes fluid around."""
        self._set_live_settings(cursor_mode=mode)

    def set_live_idle(self, mode: str) -> None:
        """Choose what happens when the pointer is idle."""
        self._set_live_settings(idle_mode=mode)

    def set_live_fps(self, fps: int) -> None:
        """Choose the live frame-rate target; restarts a running session."""
        self._set_live_settings(fps=fps)

    def set_live_stir_strength(self, stir_strength: float) -> None:
        """Choose how hard the cursor stirs the fluid; restarts a session."""
        self._set_live_settings(stir_strength=stir_strength)

    def set_live_glow(self, glow: float) -> None:
        """Choose the brightness lift near the pointer."""
        self._set_live_settings(cursor_glow=glow)

    def set_live_ambient(self, speed: float) -> None:
        """Choose the ambient drift speed."""
        self._set_live_settings(ambient_speed=speed)

    def set_live_motion(self, motion: str) -> None:
        """Apply a named motion bundle (calm/lively) to the live feel."""
        try:
            bundle = live_config_module.MOTION_PRESETS[motion]
        except KeyError:
            raise ValueError(
                f"unknown motion {motion!r}; "
                f"expected one of {sorted(live_config_module.MOTION_PRESETS)}"
            ) from None
        self._set_live_settings(**bundle)

    def effective_live_settings(self) -> LiveSettings:
        """The preferences a (re)start would use right now.

        While running with an override, that is the override — the controls
        must show what the session actually runs, not the global default it
        no longer follows.
        """
        if self.live_status == "running" and self.state.live_recipe:
            return self.state.live_overrides.get(
                self.state.live_recipe.composition_key(), self.state.live
            )
        return self.state.live

    @staticmethod
    def live_motion_name(settings: LiveSettings) -> str:
        """Which named bundle `settings` matches, or 'custom'.

        The motion combo displays this: a bundle name when the feel matches
        one exactly, custom as soon as any knob diverges.
        """
        for name, bundle in live_config_module.MOTION_PRESETS.items():
            if all(
                getattr(settings, field) == value
                for field, value in bundle.items()
            ):
                return name
        return "custom"

    def live_stats(self) -> dict[str, Any] | None:
        """The latest session measurements, or None when stopped."""
        session = self._live_session
        if session is None:
            return None
        stats = session.stats
        return {
            "frames": stats.frames,
            "fps": 1000.0 / max(stats.ema_ms, 1e-9) if stats.ema_ms else 0.0,
            "ema_ms": stats.ema_ms,
            "target_fps": stats.target_fps,
            "degraded": stats.degraded,
        }

    def live_preview_frame(self) -> bytes | None:
        """One small animated frame of what going live would look like.

        Returns raw RGB bytes at `LIVE_PREVIEW_WIDTH x LIVE_PREVIEW_HEIGHT,
        or None when no preview has completed yet. The still comes from the
        displayed preview file, so starting a motion preview never triggers a
        render; the cached fluid state advances a few steps per call and the
        interface polls on a timer. Deterministic per composition (the
        recipe's own seed): the same wallpaper always previews the same
        motion. Sim-only — no cursor, no window, Wayland-safe.
        """
        from PIL import Image

        from .live import fluid as fluid_module
        from .live.service import field_params

        recipe = self.displayed_recipe
        if recipe is None:
            return None
        path = self.state.preview_path
        if path is None or not Path(path).exists():
            return None
        key = recipe.composition_key()
        cached = self._live_preview
        if cached is None or cached[0] != key:
            with Image.open(path) as picture:
                base = np.asarray(picture.convert("RGB"), dtype=np.float32) / 255.0
            fluid = fluid_module.build_fluid(
                base,
                field_params(recipe),
                int(recipe.effective_seed),
                LIVE_PREVIEW_WIDTH,
                LIVE_PREVIEW_HEIGHT,
            )
            cached = (key, fluid)
            self._live_preview = cached
        fluid = cached[1]
        for _ in range(LIVE_PREVIEW_STEPS_PER_TICK):
            fluid_module.advance(fluid, 0.004)
        return fluid_module.frame_bytes(fluid)

    def request_preview(self) -> tuple[int, Recipe]:
        """Start a low-resolution preview render on the caller’s worker thread."""
        preview_recipe = render_module.preview_size(self.state.recipe, self.state.preview_edge)
        token = self._begin(PendingRender(token=0, recipe=preview_recipe, kind="preview"))
        self._preview_source_recipe = self.state.recipe
        self.state.status = "Rendering preview."
        self.state.error = None
        return token, preview_recipe

    def complete_preview(
        self, token: int, result: RenderResult, seconds: float
    ) -> Path | None:
        """Accept a finished preview unless a newer request replaced it."""
        job = self._take_if_current(token, "preview")
        if job is None:
            return None
        # Only the accepted completion updates what is displayed, and only it
        # may promote its source recipe: _preview_source_recipe always belongs
        # to the latest request, and the latest request is the accepted one.
        self.displayed_recipe = self._preview_source_recipe
        cache = self.state.root / "app-cache"
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"preview-{result.digest[:16]}.png"
        result.image.save(path, format="PNG", optimize=True)
        self.state.preview_path = path
        self.state.preview_digest = result.digest
        self.state.status = f"Preview ready in {seconds:.1f}s."
        self.state.error = None
        self.save()
        return path

    def request_full(self, apply_after: bool = False) -> tuple[int, Recipe]:
        """Start a full-resolution render on the caller’s worker thread."""
        token = self._begin(
            PendingRender(token=0, recipe=self.state.recipe, kind="full", apply_after=apply_after)
        )
        self.state.status = "Rendering full wallpaper."
        self.state.error = None
        return token, self.state.recipe

    def supersede_with_full(
        self, recipe: Recipe, apply_after: bool
    ) -> tuple[int, Recipe, bool]:
        """Begin a full render now, or queue it behind a running full.

        Returns (token, recipe, started). A pending *preview* job is replaced
        outright — its worker, if any, completes into a token mismatch and is
        ignored, which is exactly what that path is for. An actually-running
        *full* render is never doubled up: the request waits in
        `_pending_full` and `take_pending_full` hands it out on completion.
        """
        if self._active is not None and self._active.kind == "full":
            self._pending_full = (recipe, apply_after)
            return self._active.token, recipe, False
        token = self._begin(
            PendingRender(token=0, recipe=recipe, kind="full", apply_after=apply_after)
        )
        self._pending_full = None
        self.state.status = "Rendering full wallpaper."
        self.state.error = None
        return token, recipe, True

    def take_pending_full(self) -> tuple[Recipe, bool] | None:
        """The queued full request, if any. Called after a job completes."""
        pending, self._pending_full = self._pending_full, None
        return pending

    def complete_full(
        self, token: int, result: RenderResult, seconds: float
    ) -> Entry | None:
        """Store a finished full render and optionally set it as wallpaper."""
        job = self._take_if_current(token, "full")
        if job is None:
            return None
        entry = self._store_full_result(job.recipe, result, seconds)
        self.state.current_digest = entry.digest
        # The preview card follows the finished full: what you see is what
        # rendered. The full-res file displays fine through a CONTAIN fit,
        # and pointing at it (rather than a second copy) costs nothing.
        self.displayed_recipe = job.recipe
        self.state.preview_path = entry.image_path
        self.state.preview_digest = entry.digest
        self.state.status = f"Saved {entry.slug} in {seconds:.1f}s."
        self.state.error = None
        if job.apply_after:
            self._wallpaper.apply(entry.image_path, theme="both")
            self._library.record_applied(entry.digest)
            self.state.status = f"Wallpaper set from {entry.image_path.name}."
        self.save()
        return entry

    def fail_active(self, message: str) -> None:
        """Record a worker-thread failure on the main thread."""
        if self._active is None:
            return
        self._active = None
        self.state.error = message
        self.state.status = "Render failed."
        self.save()

    def refresh_rows(self, limit: int = LIBRARY_ROW_LIMIT) -> tuple[LibraryRow, ...]:
        """Snapshot the newest library pieces for the thumbnail strip."""
        return tuple(
            LibraryRow(
                digest=entry.digest,
                slug=entry.slug,
                generator=entry.generator,
                palette=entry.palette_name,
                starred=entry.starred,
                created_at=entry.created_at,
                image_path=entry.image_path,
                preview_path=entry.preview_path,
            )
            for entry in self._library.list(limit=limit)
        )

    def toggle_star(self, digest: str) -> bool | None:
        """Star or unstar one saved piece; `None` means it is unknown."""
        return self._library.toggle_starred(digest)

    def library_entry(self, digest: str) -> Entry:
        """Fetch one saved piece for actions that need its recipe."""
        entry = self._library.get(digest)
        if entry is None:
            raise KeyError(f"no piece matching {digest!r}")
        return entry

    def apply_entry(self, digest: str, theme: str = "both") -> Any:
        """Set one saved piece as the wallpaper."""
        entry = self._library.get(digest)
        if entry is None:
            raise KeyError(f"no piece matching {digest!r}")
        if not entry.image_path.exists():
            raise FileNotFoundError(f"missing image for {digest!r}: {entry.image_path}")
        state = self._wallpaper.apply(entry.image_path, theme=theme)
        self._library.record_applied(entry.digest)
        self.state.current_digest = entry.digest
        self.state.status = f"Wallpaper set from {entry.image_path.name}."
        self.state.error = None
        self.save()
        return state

    def _begin(self, job: PendingRender) -> int:
        self._next_token += 1
        self._active = PendingRender(
            token=self._next_token, recipe=job.recipe, kind=job.kind, apply_after=job.apply_after
        )
        return self._active.token

    def _take_if_current(self, token: int, kind: str) -> PendingRender | None:
        if self._active is None or self._active.token != token or self._active.kind != kind:
            return None
        job, self._active = self._active, None
        return job

    def _mark_preview_stale(self, status: str) -> None:
        self.state.preview_path = None
        self.state.preview_digest = None
        self.state.error = None
        self.state.status = status

    def _remember_tuned_params(self) -> None:
        self._tuned_params[self.state.recipe.generator] = dict(self.state.recipe.params)

    def _store_full_result(
        self, recipe: Recipe, result: RenderResult, seconds: float
    ) -> Entry:
        """Store a full result without rendering a second preview.

        `library.store` renders the preview separately, which would double the
        cost of every full render from the interface. The finished image already
        contains everything a thumbnail needs, so derive it directly.
        """
        image_path = self._library.images / f"{result.digest[:16]}-{recipe.slug()}.png"
        result.image.save(image_path, format="PNG", optimize=True)
        recipe.save(image_path.with_suffix(".json"))

        thumbnail = result.image.copy()
        thumbnail.thumbnail((PREVIEW_THUMBNAIL_EDGE, PREVIEW_THUMBNAIL_EDGE))
        preview_path = self._library.previews / f"{result.digest[:16]}.png"
        thumbnail.save(preview_path, format="PNG", optimize=True)
        return self._library.add(recipe, result.digest, image_path, preview_path, seconds)


def run_app(
    argv: list[str] | None = None,
    root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> int:
    """Start the native interface.

    GTK is imported here, not at module scope, so importing `genesis.app` for
    tests or CLI wiring never requires a display server.
    """
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    from gi.repository import Adw, Gio, GLib, Gtk

    class GenesisApplication(Adw.Application):
        def __init__(self) -> None:
            super().__init__(
                application_id=APP_ID, flags=Gio.ApplicationFlags.FLAGS_NONE
            )
            self.controller = AppController(
                root=root, app_config_path=config_path
            )

        def do_activate(self) -> None:
            self.controller.restore_saved_state()
            window = self.props.active_window
            if window is None:
                window = GenesisWindow(self, self.controller)
            window.present()

    class GenesisWindow(Adw.ApplicationWindow):
        def __init__(self, application: Adw.Application, controller: AppController) -> None:
            super().__init__(application=application, title=APP_TITLE)
            self.controller = controller
            self.set_default_size(1180, 820)
            self._tune_widgets: list[tuple[TuneControl, Any]] = []
            self._tune_container: Gtk.Box | None = None
            self._updating_controls = False
            self._preview_timer = 0
            self._live_poller = 0
            self._build()
            self.refresh_all(auto_preview=True)

        def _build(self) -> None:
            toolbar = Adw.ToolbarView()
            header = Adw.HeaderBar()
            new_button = Gtk.Button(label="New wallpaper")
            new_button.add_css_class("suggested-action")
            new_button.connect("clicked", lambda _button: self.on_new_wallpaper())
            header.pack_start(new_button)
            apply_button = Gtk.Button(label="Apply wallpaper")
            apply_button.connect("clicked", lambda _button: self.on_apply_current())
            header.pack_end(apply_button)
            toolbar.add_top_bar(header)

            page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
            page.set_margin_top(12)
            page.set_margin_bottom(12)
            page.set_margin_start(12)
            page.set_margin_end(12)

            page.append(self._mood_row())
            page.append(self._preview_card())
            page.append(self._live_panel())
            page.append(self._tune_panel())
            page.append(self._library_panel())

            scroller = Gtk.ScrolledWindow()
            scroller.set_child(page)
            toolbar.set_content(scroller)
            self.set_content(toolbar)

        def _mood_row(self) -> Gtk.Widget:
            row = Gtk.Box(spacing=0)
            row.get_style_context().add_class("segmented")
            row.get_style_context().add_class("linked")
            first: Gtk.ToggleButton | None = None
            for mood in MOODS:
                button = Gtk.ToggleButton(label=mood.title())
                if first is None:
                    first = button
                else:
                    button.set_group(first)
                button.set_active(self.controller.state.mood == mood)
                button.connect("toggled", self.on_mood_toggled, mood)
                row.append(button)
            return row

        def _preview_card(self) -> Gtk.Widget:
            card = Gtk.Frame()
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            self.preview = Gtk.Picture()
            self.preview.set_content_fit(Gtk.ContentFit.CONTAIN)
            self.preview.set_size_request(640, 360)
            self.preview.set_hexpand(True)
            self.preview.set_vexpand(True)
            box.append(self.preview)
            status_row = Gtk.Box(spacing=8)
            self.spinner = Gtk.Spinner()
            status_row.append(self.spinner)
            self.status_label = Gtk.Label(xalign=0.0)
            self.status_label.set_hexpand(True)
            status_row.append(self.status_label)
            box.append(status_row)
            card.set_child(box)
            return card

        def _live_panel(self) -> Gtk.Widget:
            frame = Gtk.Frame()
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            box.set_margin_top(6)
            box.set_margin_bottom(6)
            box.set_margin_start(8)
            box.set_margin_end(8)
            box.append(Gtk.Label(label="Live wallpaper", xalign=0.0))

            row = Gtk.Box(spacing=12)
            self.live_toggle = Gtk.Button(label="Start live")
            self.live_toggle.connect("clicked", self.on_live_toggled)
            row.append(self.live_toggle)

            mode_box = Gtk.Box(spacing=0)
            mode_box.get_style_context().add_class("segmented")
            mode_box.get_style_context().add_class("linked")
            first: Gtk.ToggleButton | None = None
            for mode in live_config_module.CURSOR_MODES:
                button = Gtk.ToggleButton(label=mode.title())
                if first is None:
                    first = button
                else:
                    button.set_group(first)
                button.set_active(self.controller.state.live.cursor_mode == mode)
                button.connect("toggled", self.on_live_mode_toggled, mode)
                mode_box.append(button)
            row.append(Gtk.Label(label="Cursor"))
            row.append(mode_box)

            row.append(Gtk.Label(label="When idle"))
            self.live_idle_combo = Gtk.ComboBoxText()
            for mode in live_config_module.IDLE_MODES:
                self.live_idle_combo.append(mode, mode.title())
            self.live_idle_combo.set_active_id(self.controller.state.live.idle_mode)
            self.live_idle_combo.connect("changed", self.on_live_idle_changed)
            row.append(self.live_idle_combo)

            row.append(Gtk.Label(label="FPS"))
            self.live_fps_spin = Gtk.SpinButton.new_with_range(1, 120, 1)
            self.live_fps_spin.set_value(self.controller.state.live.fps)
            self.live_fps_spin.set_tooltip_text("Restarts the session if running")
            self.live_fps_spin.connect("value-changed", self.on_live_fps_changed)
            row.append(self.live_fps_spin)

            row.append(Gtk.Label(label="Stir"))
            self.live_stir_spin = Gtk.SpinButton.new_with_range(0.0, 30.0, 0.5)
            self.live_stir_spin.set_digits(1)
            self.live_stir_spin.set_value(self.controller.state.live.stir_strength)
            self.live_stir_spin.set_tooltip_text("How hard the cursor stirs the fluid")
            self.live_stir_spin.connect("value-changed", self.on_live_stir_changed)
            row.append(self.live_stir_spin)

            row.append(Gtk.Label(label="Glow"))
            self.live_glow_spin = Gtk.SpinButton.new_with_range(0.0, 0.5, 0.05)
            self.live_glow_spin.set_digits(2)
            self.live_glow_spin.set_value(self.controller.state.live.cursor_glow)
            self.live_glow_spin.set_tooltip_text("Brightness lift near the pointer")
            self.live_glow_spin.connect("value-changed", self.on_live_glow_changed)
            row.append(self.live_glow_spin)

            row.append(Gtk.Label(label="Drift"))
            self.live_ambient_spin = Gtk.SpinButton.new_with_range(0.0, 5.0, 0.1)
            self.live_ambient_spin.set_digits(1)
            self.live_ambient_spin.set_value(self.controller.state.live.ambient_speed)
            self.live_ambient_spin.set_tooltip_text("Ambient current underneath the cursor")
            self.live_ambient_spin.connect("value-changed", self.on_live_ambient_changed)
            row.append(self.live_ambient_spin)

            row.append(Gtk.Label(label="Feel"))
            self.live_motion_combo = Gtk.ComboBoxText()
            for name in ("calm", "lively", "custom"):
                self.live_motion_combo.append(name, name.title())
            self.live_motion_combo.set_active_id(
                self.controller.live_motion_name(self.controller.state.live)
            )
            self.live_motion_combo.set_tooltip_text(
                "A named motion feel; any manual change becomes Custom"
            )
            self.live_motion_combo.connect("changed", self.on_live_motion_changed)
            row.append(self.live_motion_combo)
            box.append(row)

            preview_row = Gtk.Box(spacing=12)
            self.live_preview_toggle = Gtk.ToggleButton(label="Preview motion")
            self.live_preview_toggle.set_tooltip_text(
                "Animate a small preview; needs a completed still preview first"
            )
            self.live_preview_toggle.connect("toggled", self.on_live_preview_toggled)
            preview_row.append(self.live_preview_toggle)
            self.live_preview_picture = Gtk.Picture()
            self.live_preview_picture.set_size_request(
                LIVE_PREVIEW_WIDTH, LIVE_PREVIEW_HEIGHT
            )
            self.live_preview_picture.set_visible(False)
            preview_row.append(self.live_preview_picture)
            self._live_preview_timer = None
            box.append(preview_row)

            self.live_status_label = Gtk.Label(xalign=0.0)
            box.append(self.live_status_label)
            frame.set_child(box)
            return frame

        def on_live_toggled(self, _button: Gtk.Button) -> None:
            try:
                if self.controller.live_status in ("stopped", "failed"):
                    self.live_preview_toggle.set_active(False)
                    self.controller.start_live()
                else:
                    self.controller.stop_live()
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
            self._refresh_live()

        def on_live_mode_toggled(self, button: Gtk.ToggleButton, mode: str) -> None:
            if not button.get_active():
                return
            try:
                self.controller.set_live_mode(mode)
                self._restart_live_if_running()
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
            self._refresh_live()

        def on_live_idle_changed(self, combo: Gtk.ComboBoxText) -> None:
            mode = combo.get_active_id()
            if not mode:
                return
            try:
                self.controller.set_live_idle(mode)
                self._restart_live_if_running()
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
            self._refresh_live()

        def on_live_fps_changed(self, spin: Gtk.SpinButton) -> None:
            try:
                self.controller.set_live_fps(spin.get_value_as_int())
                self._restart_live_if_running()
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
            self._refresh_live()

        def on_live_stir_changed(self, spin: Gtk.SpinButton) -> None:
            try:
                self.controller.set_live_stir_strength(spin.get_value())
                self._restart_live_if_running()
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
            self._refresh_live()

        def on_live_glow_changed(self, spin: Gtk.SpinButton) -> None:
            try:
                self.controller.set_live_glow(spin.get_value())
                self._restart_live_if_running()
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
            self._refresh_live()

        def on_live_ambient_changed(self, spin: Gtk.SpinButton) -> None:
            try:
                self.controller.set_live_ambient(spin.get_value())
                self._restart_live_if_running()
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
            self._refresh_live()

        def on_live_motion_changed(self, combo: Gtk.ComboBoxText) -> None:
            name = combo.get_active_id()
            if not name or name == "custom":
                return
            try:
                self.controller.set_live_motion(name)
                self._restart_live_if_running()
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
            self._refresh_live()

        def on_live_preview_toggled(self, button: Gtk.ToggleButton) -> None:
            if button.get_active():
                self.live_preview_picture.set_visible(True)
                if self._live_preview_timer is None:
                    self._live_preview_timer = GLib.timeout_add(
                        100, self._poll_live_preview
                    )
            else:
                self._stop_live_preview()

        def _poll_live_preview(self) -> bool:
            try:
                frame = self.controller.live_preview_frame()
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
                frame = None
            if frame is None:
                self._stop_live_preview()
                self.live_preview_toggle.set_active(False)
                self._refresh_status()
                return False
            from gi.repository import GdkPixbuf

            pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
                GLib.Bytes(frame),
                GdkPixbuf.Colorspace.RGB,
                False,
                8,
                LIVE_PREVIEW_WIDTH,
                LIVE_PREVIEW_HEIGHT,
                LIVE_PREVIEW_WIDTH * 3,
            )
            self.live_preview_picture.set_pixbuf(pixbuf)
            return True

        def _stop_live_preview(self) -> None:
            if self._live_preview_timer is not None:
                GLib.source_remove(self._live_preview_timer)
                self._live_preview_timer = None
            self.live_preview_picture.set_visible(False)

        def _restart_live_if_running(self) -> None:
            """Preference changes take effect by restarting a live session.

            Restarting is predictable: the alternative is a preference that
            silently does nothing until next start, which is exactly the kind
            of dead control this interface was built to avoid. The live
            recipe is passed explicitly so a restart re-animates what was
            running rather than whatever happens to be current.
            """
            if self.controller.live_status == "running":
                self.controller.stop_live()
                self.controller.start_live(self.controller.state.live_recipe)

        def _refresh_live(self) -> None:
            status = self.controller.live_status
            labels = {
                "stopped": "Start live",
                "starting": "Starting…",
                "running": "Stop live",
                "failed": "Retry live",
            }
            self.live_toggle.set_label(labels.get(status, "Start live"))
            available, reason = self.controller.live_available()
            self.live_toggle.set_sensitive(available)
            self.live_toggle.set_tooltip_text("" if available else reason)
            live = self.controller.effective_live_settings()
            self.live_glow_spin.set_value(live.cursor_glow)
            self.live_ambient_spin.set_value(live.ambient_speed)
            self.live_motion_combo.set_active_id(
                self.controller.live_motion_name(live)
            )
            stats = self.controller.live_stats()
            if stats is None:
                detail = self.controller.live_error or "Live wallpaper is stopped."
            else:
                detail = (
                    f"{stats['fps']:.0f} fps, target {stats['target_fps']}"
                    f"{', degraded' if stats['degraded'] else ''}"
                )
            self.live_status_label.set_text(detail)

        def _tune_panel(self) -> Gtk.Widget:
            self.tune_expander = Gtk.Expander(label="Tune")
            self.tune_expander.set_expanded(False)
            container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            container.append(self._engine_row())
            self._tune_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            container.append(self._tune_container)
            self._rebuild_tune_groups()
            self.tune_expander.set_child(container)
            return self.tune_expander

        def _engine_row(self) -> Gtk.Widget:
            row = Gtk.Box(spacing=12)
            generator_label = Gtk.Label(label="Engine")
            self.generator_combo = Gtk.ComboBoxText()
            for name in registry.names():
                self.generator_combo.append(name, name)
            self.generator_combo.set_active_id(self.controller.state.recipe.generator)
            self.generator_combo.connect("changed", self.on_generator_changed)
            palette_label = Gtk.Label(label="Palette")
            self.palette_combo = Gtk.ComboBoxText()
            for name in sorted(PRESETS):
                self.palette_combo.append(name, name)
            self.palette_combo.set_active_id(self.controller.state.recipe.palette_name)
            self.palette_combo.connect("changed", self.on_palette_changed)
            size_label = Gtk.Label(label="Size")
            self.size_combo = Gtk.ComboBoxText()
            for width, height in OUTPUT_SIZES:
                label = f"{width}x{height}"
                self.size_combo.append(label, label)
            current_size = (
                f"{self.controller.state.recipe.width}x{self.controller.state.recipe.height}"
            )
            if self.size_combo.get_active_id() != current_size:
                self.size_combo.set_active_id(current_size)
            self.size_combo.connect("changed", self.on_size_changed)
            for widget in (
                generator_label,
                self.generator_combo,
                palette_label,
                self.palette_combo,
                size_label,
                self.size_combo,
            ):
                row.append(widget)
            return row

        def _library_panel(self) -> Gtk.Widget:
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            box.append(Gtk.Label(label="Library", xalign=0.0))
            scroller = Gtk.ScrolledWindow()
            scroller.set_min_content_height(170)
            scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.NEVER)
            self.thumbnail_row = Gtk.FlowBox()
            self.thumbnail_row.set_selection_mode(Gtk.SelectionMode.NONE)
            self.thumbnail_row.set_max_children_per_line(8)
            scroller.set_child(self.thumbnail_row)
            box.append(scroller)
            return box

        def refresh_all(self, auto_preview: bool = False) -> None:
            self._refresh_preview_image()
            self._refresh_rows()
            self._refresh_status()
            self._refresh_live()
            if auto_preview:
                self.schedule_preview()
            if self._live_poller == 0:
                self._live_poller = GLib.timeout_add(1000, self._poll_live)

        def _poll_live(self) -> bool:
            """Keep the live status line current while a session runs."""
            self._refresh_live()
            return True

        def on_mood_toggled(self, button: Gtk.ToggleButton, mood: str) -> None:
            if not button.get_active() or self.controller.state.mood == mood:
                return
            try:
                self.controller.set_mood(mood)
                self.controller.new_wallpaper()
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
            self.refresh_all(auto_preview=False)
            self.schedule_full(apply_after=True)

        def on_new_wallpaper(self) -> None:
            try:
                self.controller.new_wallpaper()
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
                self._refresh_status()
                return
            self.refresh_all(auto_preview=False)
            self.schedule_full(apply_after=True)

        def on_apply_current(self) -> None:
            self.schedule_full(apply_after=True)

        def on_generator_changed(self, combo: Gtk.ComboBoxText) -> None:
            name = combo.get_active_id()
            if name and name != self.controller.state.recipe.generator:
                try:
                    self.controller.set_generator(name)
                except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                    self.controller.state.error = str(exc)
                self._rebuild_tune_groups()
                self.refresh_all(auto_preview=True)

        def on_palette_changed(self, combo: Gtk.ComboBoxText) -> None:
            name = combo.get_active_id()
            if name and name != self.controller.state.recipe.palette_name:
                try:
                    self.controller.set_palette(name)
                except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                    self.controller.state.error = str(exc)
                self.refresh_all(auto_preview=True)

        def on_size_changed(self, combo: Gtk.ComboBoxText) -> None:
            size = combo.get_active_id()
            if not size:
                return
            try:
                width, height = (int(part) for part in size.split("x"))
                current = self.controller.state.recipe
                if (width, height) != (current.width, current.height):
                    self.controller.set_output_size(width, height)
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
            self.refresh_all(auto_preview=True)

        def _rebuild_tune_groups(self) -> None:
            assert self._tune_container is not None
            child = self._tune_container.get_first_child()
            while child is not None:
                following = child.get_next_sibling()
                self._tune_container.remove(child)
                child = following
            self._tune_widgets = []
            for group in tune_layout(self.controller.state.recipe):
                expander = Gtk.Expander(label=group.title)
                expander.set_expanded(True)
                grid = Gtk.Grid(column_spacing=12, row_spacing=6)
                grid.set_margin_top(6)
                grid.set_margin_bottom(6)
                for row, control in enumerate(group.controls):
                    label = Gtk.Label(label=control.label, xalign=0.0)
                    widget = self._control_widget(control)
                    widget.set_sensitive(control.available)
                    if control.reason:
                        widget.set_tooltip_text(control.reason)
                    grid.attach(label, 0, row, 1, 1)
                    grid.attach(widget, 1, row, 1, 1)
                    self._tune_widgets.append((control, widget))
                expander.set_child(grid)
                self._tune_container.append(expander)

        def _control_widget(self, control: TuneControl) -> Gtk.Widget:
            recipe = self.controller.state.recipe
            raw = recipe.params.get(control.name, recipe.post.get(control.name))
            if control.kind == "bool":
                switch = Gtk.Switch()
                switch.set_active(bool(raw))
                switch.connect("notify::active", self.on_bool_changed, control)
                return switch
            if control.kind == "enum":
                combo = Gtk.ComboBoxText()
                spec = self._spec_for(control)
                for choice in spec.choices:
                    combo.append(choice, choice)
                combo.set_active_id(str(raw))
                combo.connect("changed", self.on_enum_changed, control)
                return combo
            adjustment, digits = self._spin_adjustment(control, raw)
            spin = Gtk.SpinButton()
            spin.set_adjustment(adjustment)
            spin.set_digits(digits)
            spin.set_numeric(True)
            spin.connect("value-changed", self.on_number_changed, control)
            return spin

        def _spec_for(self, control: TuneControl):
            recipe = self.controller.state.recipe
            if control.scope == "params":
                return registry.get(recipe.generator).schema.get(control.name)
            return post_module.SCHEMA.get(control.name)

        def _spin_adjustment(self, control: TuneControl, raw: Any) -> tuple[Gtk.Adjustment, int]:
            spec = self._spec_for(control)
            minimum = float(spec.minimum) if spec.minimum is not None else 0.0
            maximum = float(spec.maximum) if spec.maximum is not None else 1.0
            value = min(max(float(raw if raw is not None else spec.default), minimum), maximum)
            if control.kind == "int":
                step = float(spec.step or 1)
                digits = 0
            else:
                span = max(maximum - minimum, 0.0)
                step = float(spec.step or max(span / 100.0, 0.0001))
                digits = 4 if span < 0.1 else 2
            return (
                Gtk.Adjustment(
                    value=value,
                    lower=minimum,
                    upper=maximum,
                    step_increment=step,
                    page_increment=step * 10,
                ),
                digits,
            )

        def on_bool_changed(self, switch: Gtk.Switch, _state: Any, control: TuneControl) -> None:
            if self._updating_controls:
                return
            self._apply_tune(control, switch.get_active())

        def on_enum_changed(self, combo: Gtk.ComboBoxText, control: TuneControl) -> None:
            if self._updating_controls:
                return
            value = combo.get_active_id()
            if value is not None:
                self._apply_tune(control, value)

        def on_number_changed(self, spin: Gtk.SpinButton, control: TuneControl) -> None:
            if self._updating_controls:
                return
            value = spin.get_value_as_int() if control.kind == "int" else spin.get_value()
            self._apply_tune(control, value)

        def _apply_tune(self, control: TuneControl, value: Any) -> None:
            try:
                self.controller.set_tuned_value(control.scope, control.name, value)
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
            self._refresh_availability()
            self.refresh_all(auto_preview=False)
            self.schedule_preview()

        def _refresh_availability(self) -> None:
            values = {
                **self.controller.state.recipe.params,
                **self.controller.state.recipe.post,
            }
            for control, widget in self._tune_widgets:
                spec = self._spec_for(control)
                available = spec.is_available(values)
                widget.set_sensitive(available)
                widget.set_tooltip_text(spec.unavailable_reason(values))

        def _refresh_rows(self) -> None:
            child = self.thumbnail_row.get_first_child()
            while child is not None:
                following = child.get_next_sibling()
                self.thumbnail_row.remove(child)
                child = following
            for row in self.controller.refresh_rows():
                self.thumbnail_row.append(self._thumbnail(row))

        def _thumbnail(self, row: LibraryRow) -> Gtk.Widget:
            from gi.repository import GdkPixbuf

            overlay = Gtk.Overlay()
            path = row.preview_path if row.preview_path and row.preview_path.exists() else None
            source = path or (row.image_path if row.image_path.exists() else None)
            if source is None:
                overlay.set_child(Gtk.Label(label="missing image"))
            else:
                picture = Gtk.Picture()
                picture.set_content_fit(Gtk.ContentFit.COVER)
                picture.set_size_request(180, 110)
                pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(str(source), 240, 135, True)
                picture.set_pixbuf(pixbuf)
                overlay.set_child(picture)
            star = Gtk.ToggleButton(label="★" if row.starred else "☆")
            star.set_active(row.starred)
            star.set_halign(Gtk.Align.END)
            star.set_valign(Gtk.Align.START)
            star.connect("toggled", self.on_star_toggled, row.digest)
            overlay.add_overlay(star)
            apply_button = Gtk.Button(label="Set")
            apply_button.connect("clicked", self.on_thumbnail_apply, row.digest)
            live_button = Gtk.Button(label="Make live")
            live_button.connect("clicked", self.on_thumbnail_make_live, row.digest)
            available, reason = self.controller.live_available()
            live_button.set_sensitive(available)
            if not available:
                live_button.set_tooltip_text(reason)
            actions = Gtk.Box(spacing=6)
            actions.set_halign(Gtk.Align.CENTER)
            actions.set_valign(Gtk.Align.END)
            actions.append(apply_button)
            actions.append(live_button)
            overlay.add_overlay(actions)
            item = Gtk.FlowBoxChild()
            item.set_child(overlay)
            return item

        def on_star_toggled(self, button: Gtk.ToggleButton, digest: str) -> None:
            try:
                starred = self.controller.toggle_star(digest)
                button.set_label("★" if starred else "☆")
                button.set_active(bool(starred))
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
                button.set_active(not button.get_active())
            self._refresh_status()

        def on_thumbnail_apply(self, _button: Gtk.Button, digest: str) -> None:
            try:
                self.controller.apply_entry(digest)
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
            self._refresh_status()

        def on_thumbnail_make_live(self, _button: Gtk.Button, digest: str) -> None:
            """Animate exactly this saved composition, not the current one."""
            try:
                self.controller.start_live_from_entry(digest)
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.state.error = str(exc)
            self._refresh_status()
            self._refresh_live()

        def _refresh_preview_image(self) -> None:
            from gi.repository import GdkPixbuf

            path = self.controller.state.preview_path
            if path and path.exists():
                pixbuf = GdkPixbuf.Pixbuf.new_from_file(str(path))
                self.preview.set_pixbuf(pixbuf)
            else:
                self.preview.set_pixbuf(None)

        def _refresh_status(self) -> None:
            state = self.controller.state
            text = state.error or state.status
            self.status_label.set_text(text)
            busy = self.controller.busy
            self.spinner.set_spinning(busy)

        def schedule_preview(self) -> None:
            if self._preview_timer:
                GLib.source_remove(self._preview_timer)
                self._preview_timer = 0
            self._preview_timer = GLib.timeout_add(220, self._start_scheduled_preview)

        def _start_scheduled_preview(self) -> bool:
            self._preview_timer = 0
            if self.controller.busy:
                self.schedule_preview()
                return False
            token, recipe = self.controller.request_preview()
            self._refresh_status()
            self._start_worker(token, recipe, "preview", apply_after=False)
            return False

        def schedule_full(self, apply_after: bool = False) -> None:
            token, recipe, started = self.controller.supersede_with_full(
                self.controller.state.recipe, apply_after
            )
            if not started:
                self.controller.state.error = (
                    "Finishing the current render first — yours is queued."
                )
                self._refresh_status()
                return
            self._refresh_status()
            self._start_worker(token, recipe, "full", apply_after=apply_after)

        def _start_queued_full(self) -> None:
            """Launch the full render queued behind a just-finished one, if any."""
            pending = self.controller.take_pending_full()
            if pending is None:
                return
            recipe, apply_after = pending
            token, recipe, _started = self.controller.supersede_with_full(
                recipe, apply_after
            )
            self._refresh_status()
            self._start_worker(token, recipe, "full", apply_after=apply_after)

        def _start_worker(self, token: int, recipe: Recipe, kind: str, apply_after: bool) -> None:
            thread = threading.Thread(
                target=self._render_in_background,
                args=(token, recipe, kind, apply_after),
                daemon=True,
            )
            thread.start()

        def _render_in_background(
            self, token: int, recipe: Recipe, kind: str, apply_after: bool
        ) -> None:
            started = time.perf_counter()
            try:
                result = self.controller.render_snapshot(recipe)
                seconds = time.perf_counter() - started
                GLib.idle_add(self._finish_in_main, token, kind, result, seconds)
            except Exception as exc:  # noqa: BLE001 - worker threads report, never raise
                GLib.idle_add(self._fail_in_main, str(exc))

        def _finish_in_main(self, token: int, kind: str, result: RenderResult, seconds: float) -> bool:
            try:
                if kind == "preview":
                    self.controller.complete_preview(token, result, seconds)
                else:
                    self.controller.complete_full(token, result, seconds)
            except Exception as exc:  # noqa: BLE001 - show the failure, keep the app up
                self.controller.fail_active(str(exc))
            self._refresh_preview_image()
            self._refresh_rows()
            self._refresh_status()
            self._start_queued_full()
            return False

        def _fail_in_main(self, message: str) -> bool:
            self.controller.fail_active(message)
            self._refresh_status()
            self._start_queued_full()
            return False

    application = GenesisApplication()
    return application.run(argv)


__all__ = [
    "APP_ID",
    "APP_TITLE",
    "LIBRARY_ROW_LIMIT",
    "LIVE_PREVIEW_HEIGHT",
    "LIVE_PREVIEW_STEPS_PER_TICK",
    "LIVE_PREVIEW_WIDTH",
    "MOODS",
    "MOOD_ALL",
    "OUTPUT_SIZES",
    "PREVIEW_EDGE",
    "TUNE_GROUPS",
    "AppController",
    "AppState",
    "LibraryRow",
    "LiveSettings",
    "PendingRender",
    "TuneControl",
    "TuneGroup",
    "config_path",
    "default_recipe",
    "default_state",
    "load_state",
    "run_app",
    "save_state",
    "state_from_json",
    "state_to_json",
    "tune_layout",
]

"""The live wallpaper main loop.

One session is: resolve a recipe, acquire its still as the dye all motion
starts from, precompute the ambient flow, then run a paced loop until asked
to stop. Every pixel participates every frame, so the screen reads as fluid
rather than as sparse lines drawn over a picture.

Adaptivity is load-bearing, not decorative. Each frame is timed with an
exponential moving average; when the average exceeds the budget, the target
frame rate steps down through notches and climbs back when headroom returns.
A session that cannot hold its rate slows down gracefully instead of
stuttering.

Stopping is clean by construction: the desktop wallpaper underneath was never
changed (the live window only covered it), so teardown destroys the window and
returns. No gsettings writes, nothing to restore.
"""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .. import wallpaper as wallpaper_module
from ..recipe import Recipe
from . import fluid as fluid_module
from .backend import X11Backend
from .config import LiveConfig
from .fluid import CursorTracker
from .sensors import LockMonitor, PowerMonitor


@dataclass
class LiveStats:
    """Rolling measurements reported with `--stats`."""

    frames: int = 0
    sim_ms: float = 0.0
    up_ms: float = 0.0
    blit_ms: float = 0.0
    ema_ms: float = 0.0
    target_fps: int = 60
    degraded: bool = False


# Frame-rate notches the governor steps through. The configured fps selects
# the starting notch; sustained over-budget frames step down, headroom steps
# back up. Notches rather than a continuum so the rate a user sees is always
# a round, explainable number.
FPS_NOTCHES = (120, 90, 60, 45, 30, 20, 15, 10)

# Seconds over which the still hands over to the live dye. A hard cut at
# frame 1 is the most visible artifact a session has; a two-second dissolve
# hides it. Wall-clock, not frame-counted: at 10 fps a 120-frame dissolve
# would smear for twelve seconds. Past the handover every frame is pure
# simulation; resizes and fidelity steps restart a fresh handover.
CROSSFADE_SECONDS = 2.0

# Minimum frames between fidelity changes (sim-level steps). Rebuilding the
# grid hitches for ~100 ms, so the governor must not flap between levels on
# noisy measurements.
FIDELITY_MIN_GAP_FRAMES = 300


def field_params(recipe: Recipe) -> dict:
    """The still-image field parameters a live session reuses."""
    params = dict(recipe.params)
    return {
        "scale": float(params.get("scale", 1.5)),
        "detail": int(params.get("detail", 5)),
        "turns": float(params.get("turns", 3.0)),
        "swirl": bool(params.get("swirl", True)),
        "step_size": float(params.get("step_size", 0.001)),
    }


def acquire_base(
    recipe: Recipe,
    width: int,
    height: int,
    image_path: Any = None,
    renderer: Any = None,
) -> np.ndarray:
    """The still image a session starts from, as float32 sRGB at live size.

    A stored PNG is preferred: it is pixel-exact and instant. Without one, the
    recipe is rendered once at live size on the caller's thread. Either way
    the first presented frame is the generated wallpaper, not a blank canvas.
    """
    from PIL import Image

    if image_path is not None and Path(image_path).exists():
        picture = Image.open(image_path).convert("RGB")
    else:
        if renderer is None:
            from .. import render as render_module

            renderer = render_module.render
        sized = recipe.with_changes(width=width, height=height)
        picture = renderer(sized).image
    if picture.size != (width, height):
        picture = picture.resize((width, height), Image.LANCZOS)
    return np.asarray(picture, dtype=np.float32) / 255.0


def _fit_base(base: np.ndarray, width: int, height: int) -> np.ndarray:
    """The still at exactly live size, as float32 sRGB.

    A stored PNG can be any resolution while the session presents at config
    size, so resize once up front (LANCZOS, the same filter `acquire_base`
    uses) rather than per frame.
    """
    if base.shape[:2] == (height, width):
        return np.asarray(base, dtype=np.float32)
    from PIL import Image

    picture = Image.fromarray(
        np.clip(np.asarray(base) * 255.0, 0, 255).astype(np.uint8)
    ).resize((width, height), Image.LANCZOS)
    return np.asarray(picture, dtype=np.float32) / 255.0


def quantize_fullres(base: np.ndarray) -> bytes:
    """Pack a float32 sRGB frame for upload: the exact bytes frame 0 shows."""
    frame = np.clip(np.asarray(base, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    return frame.tobytes()


class LiveSession:
    """One running live wallpaper: simulation plus presentation."""

    def __init__(
        self,
        config: LiveConfig,
        recipe: Recipe,
        seed: int | None = None,
        base_image: np.ndarray | None = None,
        base_path: Any = None,
        renderer: Any = None,
        lock_monitor: LockMonitor | None = None,
        power_monitor: PowerMonitor | None = None,
    ) -> None:
        self.config = config
        self.recipe = recipe
        self.seed = seed if seed is not None else int(recipe.effective_seed)
        if config.still:
            base = (
                base_image
                if base_image is not None
                else acquire_base(recipe, config.width, config.height, base_path, renderer)
            )
        else:
            palette = recipe.effective_palette()
            flat = np.asarray(palette.to_srgb(0.0), dtype=np.float32)
            base = np.tile(flat, (config.height, config.width, 1))
        # The still's origin, for refits after a display change: the stored
        # PNG path when there is one (re-acquired exact), else the passed
        # image, else a re-render. One extra full-res frame in memory is the
        # price of an exact resize.
        self._base_path = base_path
        self._base_image_orig = base_image
        self._renderer = renderer
        # Fidelity ladder position (index into fluid.SIM_LEVELS): the
        # governor sheds sim resolution before fps. Set before the first
        # grid build, which reads it. The last-move marker starts a full gap
        # in the past so the first degradation acts promptly; after that,
        # moves are debounced.
        self._sim_level = 0
        self._last_fidelity_frame = -FIDELITY_MIN_GAP_FRAMES
        # The full-resolution still, kept for frame 0. The simulation runs at
        # half resolution, so its first upscaled frame would soften detail;
        # presenting the still itself first makes "starts pixel-identical"
        # literally true.
        self.base_full = np.ascontiguousarray(
            _fit_base(base, config.width, config.height)
        )
        self.fluid = fluid_module.build_fluid(
            base, field_params(recipe), self.seed,
            config.width, config.height,
            ambient_speed=config.ambient_speed,
            sim_scale=fluid_module.SIM_LEVELS[self._sim_level],
        )
        # The downscaled still the handover dissolves from: the dye's own
        # initial content, copied before the first frame moves it.
        self._base_sim = self.fluid.dye.copy()
        self._blend_elapsed = 0.0
        self.tracker = CursorTracker()
        self._pointer_velocity = (0.0, 0.0)
        self.backend: X11Backend | None = None
        self.stats = LiveStats(target_fps=config.fps)
        self._idle = False
        self._shm_note_printed = False
        self._stop = False
        # Outside-world sensors, injectable for headless tests. Cached states
        # refresh on a seconds cadence from the loop, never per frame.
        self._lock_monitor = lock_monitor or LockMonitor()
        self._power_monitor = power_monitor or PowerMonitor()
        self._sensor_tick = 0
        self._locked = False
        self._fullscreen_focused = False
        self._on_battery = False
        # The dye as last uploaded, for skipping pixel-identical frames.
        self._last_upload: np.ndarray | None = None
        # No pre-roll: t=0 is the still itself, and ambient flow starts moving
        # it on the very first frame.

    def request_stop(self, *args: object) -> None:
        self._stop = True

    def run(self) -> LiveStats:
        """Run until stopped; always tears the window down.

        Signal handlers are installed only on the main thread, where they are
        legal. The native app runs sessions on a worker thread and stops them
        with `request_stop()` instead; installing handlers there would raise
        `ValueError` and kill every app-started session at startup.
        """
        self.backend = X11Backend(self.config.width, self.config.height)
        self._shm_note_printed = not self.backend.using_shm
        if self._shm_note_printed:
            print(
                "live: shared memory unavailable on this server or visual; "
                "using plain transfers, expect lower frame rates"
            )
        previous = self.backend
        main_thread = threading.current_thread() is threading.main_thread()
        old_term = old_int = None
        if main_thread:
            old_term = signal.getsignal(signal.SIGTERM)
            old_int = signal.getsignal(signal.SIGINT)

            def stop(*args: object) -> None:
                self._stop = True

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
        try:
            # Frame 0 is the still itself, byte-exact, before the sim takes
            # over from frame 1. Uploading here (inside the try) keeps the
            # teardown guarantee: a failing blit still destroys the window.
            self.backend.blit_full(quantize_fullres(self.base_full))
            self._last_upload = self.fluid.dye.copy()
            self._loop()
            return self.stats
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, old_term)
                signal.signal(signal.SIGINT, old_int)
            # A mid-session resize swaps the backend: tear down whichever
            # window is current, plus the original if it differs.
            if self.backend is not None:
                self.backend.close()
            if previous is not None and previous is not self.backend:
                previous.close()
            self.backend = None

    def _notches(self) -> list[int]:
        """Frame-rate steps at or below the configured rate."""
        steps = [n for n in FPS_NOTCHES if n <= self.config.fps]
        return steps or [self.config.fps]

    def _poll_sensors(self) -> None:
        """Refresh cached outside-world states: lock, fullscreen, battery.

        Every probe fails open to "unknown", and unknown leaves the cached
        value alone — a sensor must never pause a healthy session. Called on
        a seconds cadence from the loop, never per frame: each probe forks
        (`gdbus`) or round-trips to the X server.
        """
        if self.config.pause_on_lock:
            locked = self._lock_monitor.locked()
            if locked is not None:
                self._locked = locked
        if self.config.yield_to_fullscreen != "off" and self.backend is not None:
            focused = self.backend.query_fullscreen_focused()
            if focused is not None:
                self._fullscreen_focused = focused
        if self.config.cap_on_battery:
            battery = self._power_monitor.on_battery()
            if battery is not None:
                self._on_battery = battery

    def _upload_unchanged(self, fluid: Any) -> bool:
        """Whether the dye moved less than half a quantum since last upload.

        Half a quantum rounds to the same byte, so skipping the upload is
        pixel-perfect, not approximate: the server keeps showing exactly what
        the sim would have sent. Compares into the `tmp` scratch — free at
        this point in the frame — so the check itself allocates nothing.
        Never skips without a previous upload to be identical *to*.
        """
        if self._last_upload is None:
            return False
        np.subtract(fluid.dye, self._last_upload, out=fluid.tmp)
        np.abs(fluid.tmp, out=fluid.tmp)
        return bool(fluid.tmp.max() < 0.5 / 255.0)

    def _maybe_resize(self) -> None:
        """Refit the session when the display size changed mid-run.

        A monitor hotplug otherwise strands a wrongly-sized window until
        restart. Re-acquires the still at the new size (the stored PNG when
        there is one — exact — else the original passed image, else a
        re-render), rebuilds the fluid, and re-opens the window; a short
        re-blend hides the cut. A no-op when the size is unknown (headless,
        unprobed) or unchanged. The old window only closes after the new
        one exists, so a failed resize keeps the session alive.
        """
        size = wallpaper_module.display_size()
        if size is None or (int(size[0]), int(size[1])) == (
            self.config.width,
            self.config.height,
        ):
            return
        width, height = int(size[0]), int(size[1])
        if self.config.still:
            if self._base_path is not None:
                base = acquire_base(
                    self.recipe, width, height, self._base_path, self._renderer
                )
            elif self._base_image_orig is not None:
                base = _fit_base(
                    np.asarray(self._base_image_orig, dtype=np.float32),
                    width,
                    height,
                )
            else:
                base = acquire_base(
                    self.recipe, width, height, None, self._renderer
                )
        else:
            palette = self.recipe.effective_palette()
            flat = np.asarray(palette.to_srgb(0.0), dtype=np.float32)
            base = np.tile(flat, (height, width, 1))
        new_backend = X11Backend(width, height)
        old_backend = self.backend
        self.backend = new_backend
        if old_backend is not None:
            old_backend.close()
        self.config = replace(self.config, width=width, height=height)
        self.base_full = np.ascontiguousarray(_fit_base(base, width, height))
        self._rebuild_sim()  # also resets the upload baseline and re-blends
        new_backend.blit_full(quantize_fullres(self.base_full))
        self._last_upload = self.fluid.dye.copy()

    def _rebuild_sim(self) -> None:
        """Rebuild the fluid grid at the current fidelity level and size.

        Shared by resizes and governor fidelity steps: same still, same seed,
        new grid. Resets the upload baseline (nothing on screen matches the
        new grid yet) with a short re-blend to hide the cut.
        """
        self.fluid = fluid_module.build_fluid(
            self.base_full,
            field_params(self.recipe),
            self.seed,
            self.config.width,
            self.config.height,
            ambient_speed=self.config.ambient_speed,
            sim_scale=fluid_module.SIM_LEVELS[self._sim_level],
        )
        self._base_sim = self.fluid.dye.copy()
        self._last_upload = None
        self._blend_elapsed = 0.0

    def _frame_relax(self, cursor: Any) -> float:
        """Active relax while the cursor stirs; idle relax recovers detail.

        Untouched dye drifts home faster, regaining sharpness the diffusion
        eats; cursor-driven motion stays loose. `cursor is None` covers both
        never-observed and ambient-idled pointers.
        """
        return self.config.relax if cursor is not None else self.config.relax_idle

    def _fps_cap(self) -> int:
        """The fastest rate the outside world currently allows."""
        cap = self.stats.target_fps
        if self._fullscreen_focused and self.config.yield_to_fullscreen == "slow":
            cap = min(cap, 10)
        if self._on_battery and self.config.cap_on_battery:
            cap = min(cap, 30)
        return cap

    def _loop(self) -> None:
        assert self.backend is not None
        fluid = self.fluid
        last_pointer: tuple[float, float] | None = None
        prev_smooth: tuple[float, float] | None = None
        last_stats = time.perf_counter()
        while not self._stop:
            frame_start = time.perf_counter()
            event = self.backend.drain_events()
            if event == "quit":
                break
            if event == "resized":
                self._maybe_resize()
                # The resize rebuilt the fluid state: rebind the loop local,
                # which otherwise keeps advecting the dead grid.
                fluid = self.fluid
            # Wall-clock step in sixtieths (1.0 at 60 fps, capped): motion
            # integrates per second, not per frame, so the governor's fps
            # notches never change how fast the wallpaper moves.
            dt = min(60.0 / self.stats.target_fps, 4.0)
            sim_scale = fluid.width / self.config.width
            self._sensor_tick += 1
            if self._sensor_tick % 120 == 0:
                self._poll_sensors()
            if self._locked and self.config.pause_on_lock:
                # Behind a lock screen nobody watches: freeze everything,
                # dye included, and wake only to re-check the lock.
                time.sleep(1.0)
                continue
            if (
                self._fullscreen_focused
                and self.config.yield_to_fullscreen == "pause"
            ):
                time.sleep(0.5)
                continue
            pointer = self.backend.query_pointer()
            moved = pointer is not None and (
                last_pointer is None or (pointer.x, pointer.y) != last_pointer
            )
            if moved:
                assert pointer is not None
                last_pointer = (pointer.x, pointer.y)
                self.tracker.observe(pointer.x, pointer.y, self.stats.frames)
            idle = self.tracker.idle_seconds(self.stats.frames, self.config.fps)
            self._idle = bool(
                idle > self.config.idle_after_seconds
                and self.config.idle_mode in ("pause", "still", "ambient")
            )
            if idle > self.config.idle_after_seconds and self.config.idle_mode in ("pause", "still"):
                # `still` holds its frame with a short sleep rather than a hot
                # spin: quit and unlock events are still noticed promptly.
                sleep_for = 0.5 if self.config.idle_mode == "pause" else 0.1
                time.sleep(sleep_for)
                continue
            cursor = self.tracker.position()
            if idle > self.config.idle_after_seconds and self.config.idle_mode == "ambient":
                cursor = None

            t0 = time.perf_counter()
            if cursor is not None and moved:
                # Pointer velocity in sim pixels per frame, clamped so a
                # teleporting cursor shoves rather than shreds the dye. The
                # clamp scales with dt (per-frame deltas already grow at low
                # fps); the painted strength scales too, keeping paint per
                # second constant.
                base = prev_smooth if prev_smooth is not None else cursor
                vx = (cursor[0] - base[0]) * sim_scale
                vy = (cursor[1] - base[1]) * sim_scale
                speed = float(np.sqrt(vx * vx + vy * vy))
                vmax = 40.0 * dt
                if speed > vmax:
                    vx, vy = vx / speed * vmax, vy / speed * vmax
                fluid_module.stir(
                    fluid, cursor[0] * sim_scale, cursor[1] * sim_scale,
                    vx, vy, self.config.cursor_radius * sim_scale,
                    self.config.stir_strength * dt, self.config.cursor_mode,
                )
            prev_smooth = cursor
            fluid_module.advance(fluid, self._frame_relax(cursor), dt)
            t1 = time.perf_counter()
            sim_ms = (t1 - t0) * 1000.0

            t2 = time.perf_counter()
            glow = None
            if (
                cursor is not None
                and self.config.cursor_glow > 0.0
                and self.config.cursor_mode != "off"
            ):
                glow = (
                    cursor[0] * sim_scale,
                    cursor[1] * sim_scale,
                    self.config.cursor_radius * sim_scale,
                    self.config.cursor_glow,
                )
            if self._blend_elapsed < CROSSFADE_SECONDS:
                data = fluid_module.crossfade_bytes(
                    fluid, self._base_sim,
                    self._blend_elapsed / CROSSFADE_SECONDS, glow,
                    self.config.sharpen,
                )
                self._blend_elapsed += dt / 60.0
            elif cursor is None and self._upload_unchanged(fluid):
                # Settled dye quantizes to the bytes already on screen:
                # spare the server (and the battery) this upload.
                data = None
            else:
                data = fluid_module.frame_bytes(fluid, glow, self.config.sharpen)
            t3 = time.perf_counter()
            up_ms = (t3 - t2) * 1000.0
            if data is not None:
                self.backend.blit_full(data)
                self._last_upload = fluid.dye.copy()
            blit_ms = (time.perf_counter() - t3) * 1000.0

            frame_ms = (time.perf_counter() - frame_start) * 1000.0
            self._adapt(frame_ms, sim_ms, up_ms, blit_ms)
            if self.config.stats and time.perf_counter() - last_stats >= 5.0:
                last_stats = time.perf_counter()
                if not self._shm_note_printed and not self.backend.using_shm:
                    self._shm_note_printed = True
                    reason = self.backend.last_rejected_error
                    print(
                        "live: server rejected shared memory "
                        f"(X error {reason}); using plain transfers, "
                        "expect lower frame rates"
                    )
                cursor = self.tracker.position()
                print(
                    f"live fps={1000.0 / max(self.stats.ema_ms, 1e-9):5.1f} "
                    f"ema={self.stats.ema_ms:5.2f}ms "
                    f"sim={self.stats.sim_ms:5.2f}ms up={self.stats.up_ms:5.2f}ms "
                    f"blit={self.stats.blit_ms:5.2f}ms "
                    f"target={self.stats.target_fps}fps "
                    f"shm={'on' if self.backend.using_shm else 'off'}"
                    f"{' DEGRADED' if self.stats.degraded else ''} "
                    f"cursor={'unknown' if cursor is None else f'{cursor[0]:.0f},{cursor[1]:.0f}'}"
                    f"{' idle' if self._idle else ''}"
                    f"{' locked' if self._locked else ''}"
                    f"{' fullscreen' if self._fullscreen_focused else ''}"
                    f"{' battery' if self._on_battery else ''}"
                )
            budget = 1.0 / self._fps_cap()
            spare = budget - (time.perf_counter() - frame_start)
            if spare > 0.001:
                time.sleep(spare)

    def _adapt(
        self, frame_ms: float, sim_ms: float, up_ms: float, blit_ms: float
    ) -> None:
        stats = self.stats
        stats.frames += 1
        stats.sim_ms = sim_ms
        stats.up_ms = up_ms
        stats.blit_ms = blit_ms
        alpha = 0.05
        stats.ema_ms = frame_ms if stats.ema_ms == 0.0 else (1 - alpha) * stats.ema_ms + alpha * frame_ms
        notches = self._notches()
        try:
            level = notches.index(stats.target_fps)
        except ValueError:
            level = 0
            stats.target_fps = notches[0]
        budget_ms = 1000.0 / stats.target_fps
        if stats.frames % 30 == 0:
            if stats.ema_ms > budget_ms * 1.15:
                self._shed_load(level, notches)
            elif stats.ema_ms < budget_ms * 0.6:
                self._recover_headroom(level, notches)

    def _shed_load(self, level: int, notches: list[int]) -> None:
        """Shed load under sustained over-budget frames: fidelity first.

        Dropping a sim level roughly halves gather traffic for a small
        softness cost; stepping fps down makes everything choppier. Fps
        notches are the last resort, reached only at the bottom rung.
        Fidelity moves are debounced: a rebuild hitches, so noisy
        measurements must not flap levels.
        """
        stats = self.stats
        if (
            self._sim_level + 1 < len(fluid_module.SIM_LEVELS)
            and stats.frames - self._last_fidelity_frame >= FIDELITY_MIN_GAP_FRAMES
        ):
            self._sim_level += 1
            self._last_fidelity_frame = stats.frames
            self._rebuild_sim()
            stats.degraded = True
        elif level + 1 < len(notches):
            stats.target_fps = notches[level + 1]
            stats.degraded = True

    def _recover_headroom(self, level: int, notches: list[int]) -> None:
        """Climb back when sustained headroom returns: fidelity before fps.

        Restoring motion detail first is the right order for the same reason
        shedding it first is: fps is the coarsest control.
        """
        stats = self.stats
        if (
            self._sim_level > 0
            and stats.frames - self._last_fidelity_frame >= FIDELITY_MIN_GAP_FRAMES
        ):
            self._sim_level -= 1
            self._last_fidelity_frame = stats.frames
            self._rebuild_sim()
            if stats.target_fps >= self.config.fps and self._sim_level == 0:
                stats.degraded = False
        elif level > 0:
            stats.target_fps = notches[level - 1]
            if stats.target_fps >= self.config.fps and self._sim_level == 0:
                stats.degraded = False


def run(
    config: LiveConfig,
    recipe: Recipe,
    seed: int | None = None,
    base_image: np.ndarray | None = None,
    base_path: Any = None,
) -> LiveStats:
    """Run one live session to completion (until stopped)."""
    return LiveSession(config, recipe, seed, base_image, base_path).run()


def describe(config: LiveConfig, recipe: Recipe, source: str | None = None) -> dict:
    """Session plan without creating any window: for `--dry-run`."""
    plan = {
        "recipe": {
            "generator": recipe.generator,
            "palette": recipe.palette_name,
            "size": [recipe.width, recipe.height],
            "seed": int(recipe.effective_seed),
        },
        "output": [config.width, config.height],
        "fps": config.fps,
        "stir_strength": config.stir_strength,
        "relax": config.relax,
        "cursor_mode": config.cursor_mode,
        "idle_mode": config.idle_mode,
        "backend": config.backend,
        "still": config.still,
        "yield_to_fullscreen": config.yield_to_fullscreen,
        "pause_on_lock": config.pause_on_lock,
        "cap_on_battery": config.cap_on_battery,
        "cursor_glow": config.cursor_glow,
        "ambient_speed": config.ambient_speed,
        "sharpen": config.sharpen,
        "relax_idle": config.relax_idle,
    }
    if source is not None:
        # How `--resolution auto` resolved: the display, or the fallback.
        plan["output_source"] = source
    return plan

"""Live session lifecycle: threads, signals, and teardown balance.

The session runs on the main thread from the CLI and on a worker thread from
the native app. Signal handlers are legal only on the main thread, so `run()`
must install them there and skip them elsewhere — starting live from the app
used to die instantly with `ValueError: signal only works in main thread`,
leaving a black window on the desktop.

The backend is stubbed throughout: these tests prove lifecycle contracts, not
pixels. Real window behavior is covered in `test_live_backend.py` under Xvfb.
"""

from __future__ import annotations

import signal
import threading

import pytest

from genesis.live import service as service_module
from genesis.live.config import LiveConfig
from genesis.recipe import Recipe


def tiny_config(**overrides) -> LiveConfig:
    fields = {"width": 320, "height": 200, "fps": 60}
    fields.update(overrides)
    return LiveConfig(**fields)


def tiny_recipe() -> Recipe:
    return Recipe(
        generator="flowfield",
        params={"trails": 1000, "steps": 25, "supersample": 1},
        width=320,
        height=200,
        seed=7,
    )


class StubBackend:
    """Records its lifecycle; never touches a display."""

    created = 0
    closed = 0

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.using_shm = False
        self.last_rejected_error = None
        self.blits = 0
        self.frames: list = []
        type(self).created += 1

    def query_pointer(self):
        return None

    def query_fullscreen_focused(self):
        return None

    def drain_events(self):
        return "quit"

    def blit_full(self, data):
        self.blits += 1
        self.frames.append(bytes(data))

    def close(self):
        type(self).closed += 1


@pytest.fixture
def stubbed_backend(monkeypatch):
    StubBackend.created = 0
    StubBackend.closed = 0
    monkeypatch.setattr(service_module, "X11Backend", StubBackend)
    return StubBackend


def test_run_on_worker_thread_needs_no_signal_handlers(stubbed_backend) -> None:
    """The app starts sessions off the main thread; that must not raise."""
    session = service_module.LiveSession(tiny_config(), tiny_recipe())
    errors: list = []
    thread = threading.Thread(target=lambda: errors.append(_run(session)))
    thread.start()
    thread.join(timeout=60)
    assert not thread.is_alive()
    assert errors == ["ok"]
    assert StubBackend.created == 1
    assert StubBackend.closed == 1


def _run(session) -> str:
    try:
        session.run()
    except ValueError:
        return "signal ValueError"
    return "ok"


def test_failed_start_destroys_its_window(stubbed_backend, monkeypatch) -> None:
    """A session that dies mid-run must not orphan its window."""

    def fail_blit(self, data):
        raise RuntimeError("backend blew up")

    monkeypatch.setattr(StubBackend, "drain_events", lambda self: "none")
    monkeypatch.setattr(StubBackend, "blit_full", fail_blit)
    session = service_module.LiveSession(
        tiny_config(idle_mode="ambient"), tiny_recipe()
    )
    with pytest.raises(RuntimeError, match="backend blew up"):
        session.run()
    assert StubBackend.created == 1
    assert StubBackend.closed == 1
    assert session.backend is None


def test_first_frame_is_pixel_identical(stubbed_backend) -> None:
    """Frame 0 shows the still byte-exact; the sim only takes over after."""
    import numpy as np

    from genesis.live.service import quantize_fullres

    base = np.zeros((200, 320, 3), dtype=np.float32)
    base[:, :160] = (0.9, 0.9, 0.9)
    base[:, 160:] = (0.02, 0.02, 0.02)
    seen: list[bytes] = []
    original_blit = StubBackend.blit_full

    def capture(self, data):
        seen.append(bytes(data))
        return original_blit(self, data)

    StubBackend.blit_full = capture
    try:
        session = service_module.LiveSession(
            tiny_config(), tiny_recipe(), base_image=base
        )
        session.run()
    finally:
        StubBackend.blit_full = original_blit
    # drain_events says "quit" at once, so the only upload is frame 0.
    assert len(seen) == 1
    assert seen[0] == quantize_fullres(base)


def test_main_thread_run_restores_signal_handlers(stubbed_backend) -> None:
    before_term = signal.getsignal(signal.SIGTERM)
    before_int = signal.getsignal(signal.SIGINT)
    session = service_module.LiveSession(tiny_config(), tiny_recipe())
    session.run()
    assert signal.getsignal(signal.SIGTERM) is before_term
    assert signal.getsignal(signal.SIGINT) is before_int


def script_monitors(locked=False, battery=False):
    """Injectable sensor doubles with scriptable answers."""
    from types import SimpleNamespace

    states = {"locked": locked, "battery": battery}
    lock = SimpleNamespace(locked=lambda: states["locked"])
    power = SimpleNamespace(on_battery=lambda: states["battery"])
    return states, lock, power


def test_poll_sensors_caches_probe_values(stubbed_backend) -> None:
    _states, lock, power = script_monitors(locked=True, battery=False)
    session = service_module.LiveSession(
        tiny_config(), tiny_recipe(), lock_monitor=lock, power_monitor=power
    )
    session._poll_sensors()
    assert session._locked is True
    assert session._on_battery is False


def test_poll_sensors_keeps_cached_values_when_probes_fail(stubbed_backend) -> None:
    states, lock, power = script_monitors(locked=True, battery=True)
    session = service_module.LiveSession(
        tiny_config(), tiny_recipe(), lock_monitor=lock, power_monitor=power
    )
    session._poll_sensors()
    states["locked"] = None  # the bus starts failing: keep, don't clear
    states["battery"] = None
    session._poll_sensors()
    assert session._locked is True
    assert session._on_battery is True


def test_locked_session_uploads_only_frame_zero(stubbed_backend, monkeypatch) -> None:
    """Behind a lock screen nobody watches: no sim, no uploads past frame 0."""
    plans = ["none", "none", "quit"]
    monkeypatch.setattr(StubBackend, "drain_events", lambda self: plans.pop(0))
    session = service_module.LiveSession(tiny_config(), tiny_recipe())
    session._locked = True
    session.run()
    assert StubBackend.created == 1
    assert session.stats.frames == 0


def test_fullscreen_pause_freezes_uploads(stubbed_backend, monkeypatch) -> None:
    plans = ["none", "none", "quit"]
    monkeypatch.setattr(StubBackend, "drain_events", lambda self: plans.pop(0))
    session = service_module.LiveSession(
        tiny_config(yield_to_fullscreen="pause"), tiny_recipe()
    )
    session._fullscreen_focused = True
    session.run()
    assert session.stats.frames == 0


def test_fps_cap_follows_sensors(stubbed_backend) -> None:
    session = service_module.LiveSession(tiny_config(), tiny_recipe())
    assert session._fps_cap() == 60
    session._fullscreen_focused = True  # slow policy: crawl at 10
    assert session._fps_cap() == 10
    session._fullscreen_focused = False
    session._on_battery = True
    assert session._fps_cap() == 30


def test_loop_frames_dissolve_from_the_still(stubbed_backend, monkeypatch) -> None:
    """The handover progresses across frames: still, blend, blend, sim-ward."""
    plans = ["none", "none", "none", "quit"]
    monkeypatch.setattr(StubBackend, "drain_events", lambda self: plans.pop(0))
    seen: list[bytes] = []
    original_blit = StubBackend.blit_full

    def capture(self, data):
        seen.append(bytes(data))
        return original_blit(self, data)

    StubBackend.blit_full = capture
    try:
        session = service_module.LiveSession(tiny_config(), tiny_recipe())
        session.run()
    finally:
        StubBackend.blit_full = original_blit
    assert len(seen) == 4  # frame 0 plus three loop frames
    assert seen[1] != seen[3]  # the dissolve visibly progresses
    assert session._blend_elapsed == pytest.approx(3 / 60.0)


def test_upload_unchanged_needs_a_previous_upload(stubbed_backend) -> None:
    session = service_module.LiveSession(tiny_config(), tiny_recipe())
    assert session._upload_unchanged(session.fluid) is False


def test_upload_unchanged_detects_settled_dye(stubbed_backend) -> None:
    session = service_module.LiveSession(tiny_config(), tiny_recipe())
    session._last_upload = session.fluid.dye.copy()
    assert session._upload_unchanged(session.fluid) is True
    session.fluid.dye[:] += 0.25 / 255.0  # sub-quantum: identical on screen
    assert session._upload_unchanged(session.fluid) is True
    session.fluid.dye[:] += 2.0 / 255.0  # visible: must upload
    assert session._upload_unchanged(session.fluid) is False


def test_settled_frames_skip_their_uploads(stubbed_backend, monkeypatch) -> None:
    """A frozen sim uploads frame 0 and then nothing: pure savings."""
    from genesis.live import fluid as fluid_module

    plans = ["none"] * 5 + ["quit"]
    monkeypatch.setattr(StubBackend, "drain_events", lambda self: plans.pop(0))
    monkeypatch.setattr(fluid_module, "advance", lambda *args, **kwargs: None)
    seen: list[bytes] = []
    original_blit = StubBackend.blit_full

    def capture(self, data):
        seen.append(bytes(data))
        return original_blit(self, data)

    monkeypatch.setattr(StubBackend, "blit_full", capture)
    session = service_module.LiveSession(tiny_config(), tiny_recipe())
    session._blend_elapsed = service_module.CROSSFADE_SECONDS  # handover done
    session.run()
    assert session.stats.frames == 5
    assert len(seen) == 1  # frame 0 only; five settled frames skipped


def test_still_idle_holds_its_frame_without_spinning(
    stubbed_backend, monkeypatch
) -> None:
    plans = ["none", "quit"]
    monkeypatch.setattr(StubBackend, "drain_events", lambda self: plans.pop(0))
    session = service_module.LiveSession(
        tiny_config(idle_mode="still"), tiny_recipe()
    )
    session.stats.frames = 1000  # long idle, without running a thousand frames
    session.run()
    assert session.stats.frames == 1000  # the idle frame ran no sim at all


def striped_base_image(width: int = 320, height: int = 200):
    import numpy as np

    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:, : width // 2] = (0.9, 0.9, 0.9)
    base[:, width // 2 :] = (0.02, 0.02, 0.02)
    return base


def test_resize_refits_from_the_passed_image(stubbed_backend, monkeypatch) -> None:
    from genesis import wallpaper as wallpaper_module

    session = service_module.LiveSession(
        tiny_config(), tiny_recipe(), base_image=striped_base_image()
    )
    monkeypatch.setattr(wallpaper_module, "display_size", lambda: (640, 360))
    session._maybe_resize()
    assert (session.config.width, session.config.height) == (640, 360)
    assert (session.fluid.out_width, session.fluid.out_height) == (640, 360)
    assert session.backend.width == 640
    assert session.backend.height == 360
    assert session._blend_elapsed == 0.0  # a fresh handover hides the cut
    assert session._last_upload is not None


def test_resize_is_a_noop_when_unchanged_or_unknown(
    stubbed_backend, monkeypatch
) -> None:
    from genesis import wallpaper as wallpaper_module

    session = service_module.LiveSession(
        tiny_config(), tiny_recipe(), base_image=striped_base_image()
    )
    monkeypatch.setattr(wallpaper_module, "display_size", lambda: (320, 200))
    session._maybe_resize()
    assert (session.config.width, session.config.height) == (320, 200)
    assert session.backend is None  # no window was ever touched
    monkeypatch.setattr(wallpaper_module, "display_size", lambda: None)
    session._maybe_resize()
    assert (session.config.width, session.config.height) == (320, 200)


def test_resized_event_refits_mid_run(stubbed_backend, monkeypatch) -> None:
    from genesis import wallpaper as wallpaper_module

    plans = ["resized", "quit"]
    monkeypatch.setattr(StubBackend, "drain_events", lambda self: plans.pop(0))
    monkeypatch.setattr(wallpaper_module, "display_size", lambda: (640, 360))
    session = service_module.LiveSession(
        tiny_config(), tiny_recipe(), base_image=striped_base_image()
    )
    session.run()
    assert (session.config.width, session.config.height) == (640, 360)
    assert session.backend is None  # torn down, not leaked
    assert StubBackend.created == 2  # exactly one window swap happened
    # Both windows torn down (>= covers the idempotent re-close of the
    # superseded one in run()'s teardown, which real backends absorb).
    assert StubBackend.closed >= 2


def test_frame_relax_is_active_only_with_a_cursor(stubbed_backend) -> None:
    session = service_module.LiveSession(tiny_config(), tiny_recipe())
    assert session._frame_relax((10.0, 10.0)) == session.config.relax
    assert session._frame_relax(None) == session.config.relax_idle
    assert session.config.relax_idle > session.config.relax


def overworked(session, frames: int = 29) -> None:
    """Drive the governor with sustained over-budget frames, headlessly.

    `frames` is one below a 30-multiple: `_adapt` counts the frame first.
    """
    session.stats.frames = frames
    session.stats.ema_ms = 1000.0 / session.stats.target_fps * 2.0
    session._adapt(
        session.stats.ema_ms, session.stats.ema_ms, 0.0, 0.0,
    )


def rested(session, frames: int = 29) -> None:
    """Drive the governor with sustained headroom, headlessly."""
    session.stats.frames = frames
    session.stats.ema_ms = 1000.0 / session.stats.target_fps * 0.2
    session._adapt(
        session.stats.ema_ms, session.stats.ema_ms, 0.0, 0.0,
    )


def test_governor_sheds_fidelity_before_fps(stubbed_backend) -> None:
    session = service_module.LiveSession(tiny_config(), tiny_recipe())
    overworked(session, frames=29)
    assert session._sim_level == 1
    assert session.stats.target_fps == 60  # fps untouched so far
    assert session.stats.degraded is True


def test_governor_steps_fps_only_at_the_bottom_rung(stubbed_backend) -> None:
    session = service_module.LiveSession(tiny_config(), tiny_recipe())
    overworked(session, frames=29)  # -> level 1
    overworked(session, frames=329)  # gap ok -> level 2 (bottom)
    assert session._sim_level == 2
    assert session.stats.target_fps == 60
    overworked(session, frames=629)  # nowhere finer to go: fps steps
    assert session.stats.target_fps == 45


def test_governor_debounces_fidelity_changes(stubbed_backend) -> None:
    session = service_module.LiveSession(tiny_config(), tiny_recipe())
    overworked(session, frames=29)  # -> level 1
    overworked(session, frames=59)  # 30 frames later: too soon to move again
    assert session._sim_level == 1
    assert session.stats.target_fps == 45  # ...so the fps notch takes it


def test_governor_recovers_fidelity_before_fps(stubbed_backend) -> None:
    session = service_module.LiveSession(tiny_config(), tiny_recipe())
    overworked(session, frames=29)  # -> level 1, degraded
    overworked(session, frames=329)  # -> level 2
    overworked(session, frames=629)  # -> 45 fps
    rested(session, frames=929)  # headroom: fidelity climbs first
    assert session._sim_level == 1
    assert session.stats.target_fps == 45
    rested(session, frames=1229)
    assert session._sim_level == 0
    assert session.stats.degraded is False or session.stats.target_fps < 60
    rested(session, frames=1529)  # top rung restored: fps climbs now
    assert session.stats.target_fps == 60
    assert session.stats.degraded is False

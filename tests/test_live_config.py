"""Live session configuration validates before any window exists."""

from __future__ import annotations

import pytest

from genesis.live.config import LiveConfig


def test_defaults_describe_full_hd_sixty() -> None:
    config = LiveConfig()
    assert (config.width, config.height) == (1920, 1080)
    assert config.fps == 60
    assert config.frame_budget == pytest.approx(1 / 60)
    assert config.cursor_mode == "repel"
    assert config.idle_mode == "pause"


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", 100),
        ("height", 100),
        ("fps", 0),
        ("fps", 240),
        ("cursor_mode", "vibrate"),
        ("cursor_radius", 0),
        ("stir_strength", -1),
        ("relax", -0.01),
        ("relax", 0.5),
        ("idle_mode", "party"),
        ("idle_after_seconds", -1),
        ("backend", "xwinwrap"),
    ],
)
def test_bad_values_are_rejected(field, value) -> None:
    with pytest.raises(ValueError):
        LiveConfig(**{field: value})


def test_relax_accepts_both_edges() -> None:
    assert LiveConfig(relax=0.0).relax == 0.0
    assert LiveConfig(relax=0.1).relax == 0.1


def test_outside_world_defaults_are_conservative() -> None:
    config = LiveConfig()
    assert config.yield_to_fullscreen == "slow"
    assert config.pause_on_lock is True
    assert config.cap_on_battery is True


def test_bad_fullscreen_policy_is_rejected() -> None:
    with pytest.raises(ValueError):
        LiveConfig(yield_to_fullscreen="minimize")


def test_bad_glow_is_rejected() -> None:
    with pytest.raises(ValueError):
        LiveConfig(cursor_glow=-0.1)
    with pytest.raises(ValueError):
        LiveConfig(cursor_glow=0.6)


def test_bad_ambient_speed_is_rejected() -> None:
    with pytest.raises(ValueError):
        LiveConfig(ambient_speed=-1.0)
    with pytest.raises(ValueError):
        LiveConfig(ambient_speed=9.0)


def test_bad_relax_idle_is_rejected() -> None:
    with pytest.raises(ValueError):
        LiveConfig(relax_idle=-0.01)
    with pytest.raises(ValueError):
        LiveConfig(relax_idle=0.5)


def test_relax_idle_defaults_stronger_than_active() -> None:
    config = LiveConfig()
    assert config.relax == 0.004
    assert config.relax_idle == 0.02


def test_config_round_trips_through_a_dict() -> None:
    config = LiveConfig(cursor_mode="attract", idle_mode="ambient")
    assert LiveConfig(**config.to_dict()) == config

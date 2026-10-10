"""Live wallpaper configuration.

A `LiveConfig` is everything the live service needs besides the recipe: output
size, cursor behaviour, frame budget, and idle policy. It validates eagerly so
a bad invocation fails before any window is created.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CURSOR_MODES = ("repel", "attract", "off")
IDLE_MODES = ("pause", "ambient", "still")
BACKENDS = ("direct",)
# What to do when a fullscreen window has focus: ignore it, slow to a crawl,
# or pause outright. Gamers and movie-watchers should not pay for wallpaper.
FULLSCREEN_POLICIES = ("off", "slow", "pause")

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 60
DEFAULT_CURSOR_RADIUS = 220.0
# Velocity painted into the fluid per frame at the cursor, in sim pixels.
# Gentle by default: earlier 8.0 smeared hard around the pointer.
DEFAULT_STIR_STRENGTH = 5.0
# Relaxation toward the still per frame. Anchors the composition so hours of
# churning cannot homogenize the dye into mush; small enough to stay out of
# the way of visible motion.
DEFAULT_RELAX = 0.004
# Idle recovery pull, per frame at 60 fps: stronger than the active relax so
# untouched dye drifts home and regains detail, while cursor-driven motion
# stays loose. `--relax` sets both; `--relax-idle` overrides just this one.
DEFAULT_RELAX_IDLE = 0.02
# Ambient drift in sim pixels per frame at 60 fps: the wallpaper's own slow
# current underneath the cursor. Scales with wall-clock dt, so it reads the
# same at any frame rate.
DEFAULT_AMBIENT_SPEED = 1.5
MAX_AMBIENT_SPEED = 5.0
# Brightness lift near the pointer, as a fraction. On low by default: bending
# reads better with a little lift, and zero remains one flag away.
DEFAULT_CURSOR_GLOW = 0.1
MAX_CURSOR_GLOW = 0.5
# Sim-space unsharp amount countering upscale softness, 0..1. Off by default:
# it also sharpens upscale artifacts, so it is a taste knob, not a fix.
DEFAULT_SHARPEN = 0.0
MAX_SHARPEN = 1.0
# Named motion feels: bundles of (ambient, stir, relax, idle-relax, glow).
# A bundle is defaults with taste — any explicit flag overrides its piece.
MOTION_PRESETS: dict[str, dict[str, float]] = {
    "calm": {
        "ambient_speed": 0.8,
        "stir_strength": 3.0,
        "relax": 0.004,
        "relax_idle": 0.03,
        "cursor_glow": 0.15,
    },
    "lively": {
        "ambient_speed": 2.0,
        "stir_strength": 9.0,
        "relax": 0.002,
        "relax_idle": 0.01,
        "cursor_glow": 0.2,
    },
}
MOTION_NAMES = ("calm", "lively")


@dataclass(frozen=True)
class LiveConfig:
    """Validated settings for one live session."""

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    cursor_mode: str = "repel"
    cursor_radius: float = DEFAULT_CURSOR_RADIUS
    stir_strength: float = DEFAULT_STIR_STRENGTH
    relax: float = DEFAULT_RELAX
    idle_mode: str = "pause"
    idle_after_seconds: float = 4.0
    backend: str = "direct"
    stats: bool = False
    still: bool = True
    yield_to_fullscreen: str = "slow"
    pause_on_lock: bool = True
    cap_on_battery: bool = True
    cursor_glow: float = DEFAULT_CURSOR_GLOW
    ambient_speed: float = DEFAULT_AMBIENT_SPEED
    sharpen: float = DEFAULT_SHARPEN
    relax_idle: float = DEFAULT_RELAX_IDLE

    def __post_init__(self) -> None:
        if not 320 <= self.width <= 7680:
            raise ValueError(f"width must be in 320..7680, got {self.width!r}")
        if not 200 <= self.height <= 4320:
            raise ValueError(f"height must be in 200..4320, got {self.height!r}")
        if not 1 <= self.fps <= 120:
            raise ValueError(f"fps must be in 1..120, got {self.fps!r}")
        if self.cursor_mode not in CURSOR_MODES:
            raise ValueError(
                f"cursor mode must be one of {CURSOR_MODES}, got {self.cursor_mode!r}"
            )
        if self.cursor_radius <= 0:
            raise ValueError(f"cursor radius must be positive, got {self.cursor_radius!r}")
        if self.stir_strength < 0:
            raise ValueError(
                f"stir strength must be non-negative, got {self.stir_strength!r}"
            )
        if not 0.0 <= self.relax <= 0.1:
            raise ValueError(f"relax must be in 0..0.1, got {self.relax!r}")
        if not 0.0 <= self.relax_idle <= 0.1:
            raise ValueError(f"relax idle must be in 0..0.1, got {self.relax_idle!r}")
        if self.idle_mode not in IDLE_MODES:
            raise ValueError(
                f"idle mode must be one of {IDLE_MODES}, got {self.idle_mode!r}"
            )
        if self.idle_after_seconds < 0:
            raise ValueError(
                f"idle delay must be non-negative, got {self.idle_after_seconds!r}"
            )
        if self.backend not in BACKENDS:
            raise ValueError(
                f"backend must be one of {BACKENDS}, got {self.backend!r}"
            )
        if self.yield_to_fullscreen not in FULLSCREEN_POLICIES:
            raise ValueError(
                f"fullscreen policy must be one of {FULLSCREEN_POLICIES}, "
                f"got {self.yield_to_fullscreen!r}"
            )
        if not 0.0 <= self.cursor_glow <= MAX_CURSOR_GLOW:
            raise ValueError(
                f"cursor glow must be in 0..{MAX_CURSOR_GLOW}, "
                f"got {self.cursor_glow!r}"
            )
        if not 0.0 <= self.ambient_speed <= MAX_AMBIENT_SPEED:
            raise ValueError(
                f"ambient speed must be in 0..{MAX_AMBIENT_SPEED}, "
                f"got {self.ambient_speed!r}"
            )
        if not 0.0 <= self.sharpen <= MAX_SHARPEN:
            raise ValueError(
                f"sharpen must be in 0..{MAX_SHARPEN}, got {self.sharpen!r}"
            )

    @property
    def frame_budget(self) -> float:
        """Seconds allowed per frame at the configured rate."""
        return 1.0 / self.fps

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "cursor_mode": self.cursor_mode,
            "cursor_radius": self.cursor_radius,
            "stir_strength": self.stir_strength,
            "relax": self.relax,
            "relax_idle": self.relax_idle,
            "idle_mode": self.idle_mode,
            "idle_after_seconds": self.idle_after_seconds,
            "backend": self.backend,
            "stats": self.stats,
            "still": self.still,
            "yield_to_fullscreen": self.yield_to_fullscreen,
            "pause_on_lock": self.pause_on_lock,
            "cap_on_battery": self.cap_on_battery,
            "cursor_glow": self.cursor_glow,
            "ambient_speed": self.ambient_speed,
            "sharpen": self.sharpen,
        }

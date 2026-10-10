# Changelog

All notable changes to Genesis are recorded here, newest first. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
aims for [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-10-11

### Added

- `genesis doctor`: one report covering interpreter, dependencies, library,
  session, display, timer, app files, GTK, and X11; exits non-zero only
  when rendering itself is impossible.
- `genesis uninstall`: disables the timer and removes installed units and
  app files without touching the library; with `--print` like `install`.
- CI (ruff plus fast and slow test jobs) and a Flatpak manifest with AppData
  metainfo as the first distribution steps.
- Runtime dependency floors widened (`numpy>=2.5,<3`, `pillow>=11.3,<12`);
  requirements.txt stays the exact dev/golden lockfile.

### Fixed

- App render queue wedge: every New-wallpaper/mood click orphaned a preview
  job nobody ran, then refused the full render as "busy" — permanently,
  until restart. `new_wallpaper()` no longer begins phantom jobs; full
  renders supersede preview work outright and queue (one deep) behind a
  running full instead of dying with a sticky error. Full completions now
  refresh the preview card and clear stale errors.
- Live engine tests rewritten for the fluid/dye simulation (`build_fluid`,
  `stir`, `advance`, `frame_bytes`). The old suite still targeted the retired
  particle-trail engine (`genesis.live.particles`, `FrameBuffer`,
  `palette_ramp`) and failed at collection.
- `genesis/live/service.py`: fixed two crash bugs on the live path — an
  undefined `moved` name (a `NameError` on any frame with a visible cursor)
  and `self.buffer.frame` (no such attribute; now `self.stats.frames`).
- Removed a duplicated `FPS_NOTCHES` definition in the live service.
- `LiveConfig` / `LiveSettings` tests updated: `particles`, `min_particles`,
  `cursor_strength`, and `decay` were replaced by `cursor_radius`,
  `stir_strength`, `relax`, and `still` when the engine moved to dense dye
  advection; the tests now pin the real fields.
- Live session failure test now injects into `blit_full` (the path the
  service actually calls). The old injection into `blit_tiles` never fired,
  which left the session looping forever and hung the suite.
- README live section: removed `--evolve`, `--pulse-depth`, and
  `--cursor-glow` (documented but never implemented) and rewrote the
  how-it-works paragraph for the fluid engine.
- Live sessions now present the still byte-exact as frame 0 before the
  half-resolution simulation takes over; previously the first displayed
  frame was already a softened upscale.
- `genesis app`: going live no longer rewrites the recipe being edited.
  The animated composition is tracked as a separate `live_recipe`, so
  Make-live on an old thumbnail leaves current work untouched.
- `genesis app`: `start_live` now refuses synchronously on Wayland or
  without a display (mirroring the CLI), and the live controls disable
  themselves with the reason instead of failing seconds later.
- Live output size follows the display: the app sizes sessions via the
  display probe with a 1080p fallback, and `genesis live --resolution`
  defaults to `auto` (`--dry-run` reports whether it used the display
  or the fallback via `output_source`).
- Live sessions now present the still byte-exact as frame 0, then dissolve
  into the simulation over two seconds instead of cutting hard at frame 1.
- New `genesis live --cursor-glow 0..0.5`: a bounded brightness lift near
  the pointer (off by default).
- The app plays a small motion preview of the current still before going
  fullscreen, and remembers live preferences per composition (keyed by a
  new size-independent `Recipe.composition_key()`), layered over the
  global defaults.
- Live sessions now freeze behind the lock screen, yield to fullscreen
  apps (`--fullscreen slow|pause|off`, default `slow`), and cap at 30 fps
  on battery (`--battery-cap`, on by default). All probes fail open: an
  unreadable sensor never pauses a healthy session.
- Settled frames whose dye quantizes to the bytes already on screen skip
  their upload; the still-idle path no longer hot-spins.
- The live window re-fits when the display size changes mid-session,
  re-acquiring the still at the new size with a short re-blend.
- The live service unit now bounds crash-loops (`StartLimitBurst=3`).
- Removed the retired dirty-tile upload path: the fluid engine moves every
  pixel, so tiles saved nothing; `blit_full` is the one upload path.
- Zero-allocation frame loop: index arithmetic, the relax blend, and frame
  quantization all run in preallocated buffers (bit-identical output,
  ~25% faster per frame on the reference machine).
- Motion integrates per second, not per frame: ambient drift, stir decay,
  relax pull, and cursor clamps scale with wall-clock dt, so governor fps
  steps no longer change how fast the wallpaper moves.
- The governor sheds sim resolution (0.5 → 0.4 → 0.35) before stepping fps,
  recovering fidelity first on headroom; fidelity moves are debounced.
- The still-to-live handover is wall-clock timed (2 s at any fps), and idle
  dye relaxes home faster than stirred dye (new `--relax-idle`).
- New `--sharpen 0..1` (sim-space unsharp, off by default) and `--motion
  calm|lively` bundles with per-piece explicit-flag override.
- Kinder live defaults: stir 8 → 5, glow off → 0.1.
- The app exposes glow/drift/motion-feel controls, remembers the full feel
  per composition, and plays a headless motion preview before going live.

### Added

### Changed

- The persistent live service now runs `live --from applied`, animating
  whatever `daily`/`apply` last put on the desktop instead of an
  independent curated pick. Reinstall (`genesis install --live`) to pick
  up the new unit; existing installs keep the old one until reinstalled.
- New `genesis live --from applied` selector for the same: animate the
  wallpaper currently on the desktop.

### Added

- `LICENSE`: Genesis is now GPL-3.0-or-later.
- Flatpak manifest (`flatpak/org.genesis.wallpapers.yml`) and AppData
  metainfo (`assets/org.genesis.wallpapers.metainfo.xml`) as the first step
  toward distribution outside a developer checkout.

## [0.1.0] — 2026-10-10

Initial snapshot (untagged). Deterministic still renderer (`flowfield`,
`attractor`, `splash`), recipe-as-source-of-truth, SQLite library, scoring
scheduler, daily systemd timer, GTK native app with browser-gallery fallback,
and the X11-only cursor-reactive live engine (dense dye advection).

Known limitations carried forward, not regressions: live mode refuses on
Wayland, wallpaper setting supports GNOME only, install assumes a developer
checkout (see `genesis install --print` outside systemd).

# Genesis

A deterministic procedural wallpaper engine. Every image is described by a JSON
recipe, and the same recipe always produces a byte-identical PNG.

The recipe is the unit of work, not the image. Editing a parameter changes the
result in a diff you can read and revert, and re-running years later reproduces
the original pixels exactly.

## Quick start

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

# Open the native interface: mood row, large preview, New wallpaper, Tune.
.venv/bin/python -m genesis.cli app

# Change the wallpaper from the command line. Renders or reuses, then applies.
.venv/bin/python -m genesis.cli daily
```

## The four commands you will actually use

```sh
genesis app              # the native UI: mood, preview, New wallpaper, Tune
genesis daily            # today's wallpaper, rendered or reused
genesis daily --dry-run  # see what it would pick, and why
genesis apply            # set an existing image as the wallpaper
```

Everything else is for making a particular image. The browser gallery remains
available as `genesis gallery`, but it is now the fallback interface rather than
the primary one. Something wrong? `genesis doctor` checks the setup and says
what needs attention; `genesis uninstall` removes what `install` put down
without touching your wallpapers.

A Flatpak is in preparation for distribution outside developer checkouts;
see `flatpak/README.md` for the build, what is verified, and the Flathub
submission checklist. Until it is submitted, the venv flow above is the
supported install.

The virtual environment intentionally includes system packages. The renderer,
tests, and library remain pinned and headless; only the native interface uses
the operating system’s GTK and Adwaita packages, which are not pinned.

### The native app

`genesis app` opens the primary interface. The first screen offers a mood row,
a large preview, and one `New wallpaper` action. Star, Set, and Make live sit
directly on the library thumbnails: Make live animates that exact saved
composition with cursor interaction, without disturbing the composition you
are editing — the animated recipe is tracked separately. Every generator and
post control lives behind the collapsed Tune panel, grouped as Shape, Motion,
Colour, Frame, and Finish.

Tune controls that cannot currently affect the image are shown with their
reason—for example, panning before zooming below 1.0—rather than as sliders
that silently do nothing.

The Live wallpaper section starts and stops animation, with cursor mode
(Repel/Attract/Off), idle behavior, frame rate, stir, glow, and drift
controls, plus a motion feel (Calm/Lively/Custom) bundling the look into one
choice — any manual change flips it to Custom. `Preview motion` plays a small
headless simulation of the current preview — no window, no cursor — so the
handover holds no surprises. Changing any live preference while running
restarts the session on the same composition and remembers it for that
wallpaper: each composition keeps its own live settings, layered over the
global defaults. Generating a new wallpaper restarts it on the fresh recipe.
Switching moods or asking for a new wallpaper never wedges the interface:
a full render supersedes preview work, and at most one render waits behind
a running one. On Wayland the live controls are disabled with the reason
shown: live mode needs an X11 session. Installing the persistent live
service stays a terminal operation (see below).

### The browser gallery

`genesis gallery` serves `http://127.0.0.1:8765` as a fallback. The parameter
form is built from the generator schemas the server sends, so adding a
parameter to a generator makes it appear in the UI with no change to the
frontend. It is loopback-only by design: it renders and writes to the library,
which is more capability than belongs on a network interface.

### The library

Everything rendered is catalogued in SQLite under `~/.local/share/genesis`, with
each image and its recipe written beside it as ordinary files.

```sh
genesis library list                    # what is there
genesis library list --starred          # the keepers
genesis library show <digest>           # one piece in detail
genesis library star <digest>           # never show something else instead
genesis library prune                   # forget pieces whose image was deleted
```

The database is derived state. Every image has its recipe next to it, so the
catalog can be rebuilt from the files, and losing it loses nothing.

### The daily timer

```sh
genesis install          # copy the units, enable the timer
genesis install --print  # the same commands, if you have no user manager
genesis uninstall        # disable the timer, remove what install put down
genesis doctor           # check the whole setup and report problems
```

The timer runs `genesis daily` once a day, a little after 8am with a random
delay, and catches up on the next boot if the machine was asleep. A retry on the
same day produces the same wallpaper: the day seeds the choice, so a rerun is
harmless rather than a second surprise.

Starring is the main dial. A starred piece gets a bonus large enough to outrank
any amount of recency, so starring two or three images turns the library into a
shortlist you would be happy to see daily.

### Installing the app on your system

```sh
.venv/bin/pip install -e .   # put `genesis` on PATH
genesis install --app        # desktop entry, icon, and templated units
```

`install --app` writes a `Genesis` launcher into your applications menu and
points the installed systemd units at the resolved `genesis` executable,
instead of the development checkout the shipped unit files reference. Combine
with `--live` to also install the live service, which stays disabled until you
enable it:

```sh
genesis install --app --live
systemctl --user enable --now genesis-live.service
```

### Live wallpapers

`genesis live` animates a composition behind your windows, with a cursor that
stirs the image. It needs an X11 session; on Wayland it refuses upfront and
suggests `genesis daily` instead (the app disables the live controls with the
same explanation).

```sh
genesis live                         # today's curated pick, cursor repels
genesis live --preset silk            # animate one preset
genesis live --mood calm --mode attract
genesis live --from latest            # animate the newest library piece
genesis live --from applied           # animate the wallpaper on the desktop now
genesis live --from <digest|slug>     # animate exactly that composition
genesis live --resolution 720p --fps 30   # lighter load (default: auto = display)
genesis live --no-still               # skip the still base, animate from flat
genesis live --stir-strength 12       # stronger cursor wake
genesis live --relax 0.01             # pull back toward the still faster
genesis live --cursor-glow 0.25       # brightness lift near the pointer
genesis live --motion calm            # a named feel; explicit flags override it
genesis live --sharpen 0.4            # counter upscale softness (taste knob)
genesis live --fullscreen pause       # pause for fullscreen apps (slow/off too)
genesis live --dry-run                # show the plan without opening a window
genesis install --live                # install the service, left disabled
systemctl --user enable --now genesis-live.service  # animate the desktop wallpaper
```

The installed service animates `--from applied`: whatever `daily` or `apply`
last put on the desktop, so the live layer always follows your wallpaper
rather than an independent pick.

How it works: frame 0 is your generated still, pixel-identical — the stored
PNG when animating a library piece, otherwise one render up front. From frame
1 a full-frame dye field takes over through a two-second dissolve, advected
every frame through an ambient flow plus a stir layer the cursor paints into.
A small per-frame relax pulls the dye back toward the still, so hours of
motion cannot churn the composition into mush. The simulation runs at half
output resolution and upscales for display; frames go out over X11 shared
memory where available, plain transfers where not. Output size follows the
display automatically (`--resolution auto`) and re-fits if the display
changes mid-session; pass an explicit size to override.

The session is a good citizen: it freezes behind the lock screen, yields to
fullscreen apps (`--fullscreen slow|pause|off`), and caps itself at 30 fps
on battery. Frames that would upload byte-identical pixels are skipped, so a
settled scene costs almost nothing. Motion integrates per second, not per
frame, so the speed you see never depends on the speed your machine manages.
The frame rate itself adapts in two stages — sim resolution steps down first
(softer motion, same smoothness), and only then do round-number fps notches
(60 → 45 → 30 …) kick in, climbing back when headroom returns. Untouched dye
relaxes home faster than stirred dye, so idle detail recovers while motion
stays loose. Idle behavior is configurable: `pause` sleeps while the pointer
is away, `ambient` keeps drifting without the cursor, `still` holds the last
frame while staying responsive.

Limits, stated plainly: X11 only, no lock-screen wallpaper, no new Python
dependencies (the X11 layer is `ctypes` against system libraries), and live
output is interactive rather than byte-identical to still renders.

## Choosing what to show

The scheduler scores rather than shuffles, so the choice is a pure function of
the library's state. A piece earns points for not having been shown recently,
for its generator and palette being under-used, and a large bonus for being
starred. The highest score wins, with ties broken by digest.

`genesis daily --dry-run` prints the ranked shortlist with the score breakdown,
which is the fastest way to understand what it is doing:

```
7 candidates:
     8.5  047a1932af2e  flowfield  ember   unseen_days +22.0, starred +12.0
     5.2  9107ea5cc12a  attractor  tide    unseen_days +22.0, generator +1.5
```

## Composing

```sh
genesis gen random --seed 3 --palette tide      # invent a recipe
genesis gen mine.json --param trails=20000      # render a saved one
genesis gen mine.json --post grain=0.04 --save  # tweak and keep the change
genesis list --params                           # what can be tuned
genesis describe mine.json                      # how a recipe resolves
```

Overrides are layered onto a recipe and revalidated, so a bad value fails before
rendering rather than partway through. A recipe written next to its image looks
like this:

```json
{
  "generator": "flowfield",
  "params": { "trails": 12000, "steps": 220, "scale": 1.5, "supersample": 1 },
  "width": 1920,
  "height": 1080,
  "palette_name": "ember",
  "seed": 11,
  "post": { "grain": 0.04, "vignette": 0.3 }
}
```

A recipe is the unit of work, not the image. Editing a parameter changes the
result in a diff you can read and revert, and re-running years later reproduces
the original pixels exactly.

## Where files live

```
~/.local/share/genesis/
  library.db          the catalog
  images/             full-size PNGs, each with its recipe beside it
  previews/           gallery thumbnails
  wallpaper.png       the staged image the desktop points at
```

## How reproducibility works

A recipe is serialized to one canonical JSON string (sorted keys, no incidental
whitespace) and hashed with BLAKE2b to derive a 64-bit seed. That seed feeds
named PCG64 substreams, so each generator draws from an independent stream and
adding a parameter to one cannot shift another's output.

Nothing else in the render path introduces entropy: no clock, no hostname, no
unseeded randomness. Pin a `seed` and the image is fixed; leave it null and the
image is still fixed, because the seed is derived from the recipe's own content.

`render.render` returns a digest of the output pixels, which is what makes a
render verifiable rather than merely repeatable.

## Generators

**`flowfield`** — particles advected through a noise vector field, leaving
tapering trails. Ink scales with the canvas, so a preview at 480p and a render
at 1080p are the same composition rather than the same parameters over different
amounts of space.

**`attractor`** — chaotic maps (Clifford, de Jong, Hopalong, Svensson) plotted
as density. Most of the parameter space is either a fixed point, a short cycle,
or a divergence, all of which render as a flat field, so coefficients are probed
before use and replaced from a verified table when they fail. Note that
`hopalong` and `svensson` produce weak results: those maps concentrate badly, and
`svensson`'s usable coefficient region is very small. `clifford` and `dejong` are
the strong ones.

**`splash`** — billowing ink clouds blooming on a dark field. Soft radial
splats displaced by a turbulence field and frayed by noise, composited
back-to-front. Blob count stays fixed across canvas sizes (radii scale instead)
so a composition tuned on a preview survives at full resolution.

## Testing

```sh
.venv/bin/python -m pytest      # 948 tests
.venv/bin/ruff check genesis tests conftest.py
```

Four layers, each answering a different question:

- **Determinism** (`test_determinism.py`, `test_golden.py`) — the same recipe
  repeats, and it repeats *the intended image*. Golden digests turn any change
  in output into a named failure. Regenerate deliberately with
  `pytest --update-golden`; never to silence a failure.
- **Composition** (`test_image_quality.py`, `test_attractor_forms.py`) — each
  generator produces a picture rather than a void: real contrast, a background
  that matches the palette floor, no edge banding from clamped samples, no
  clipping, and coverage that survives a change of resolution.
- **Behaviour** (`test_library.py`, `test_scheduler.py`, `test_wallpaper.py`,
  `test_server.py`) — the catalog does not duplicate, the scheduler picks
  something unseen, a missing session is refused rather than silently ignored, and
  concurrent requests do not corrupt each other.
- **Units** — per module: color conversion in Oklab, noise tiling, post
  processing in linear light, parameter validation, the CLI surface, the systemd
  units.

The pinned dependencies are load-bearing. Pillow's resampling is not
byte-stable across releases, so `test_environment.py` fails if the installed
version drifts from the pin.

## Design notes

- **Color is mixed in Oklab.** Interpolating in sRGB drags saturated hues through
  muddy midpoints. The palette converts to linear light; most grading remains
  there, while `post.contrast` is applied in encoded sRGB space so reduced
  contrast does not lift a dark background into mid-grey. Conversion back to
  8-bit happens once, at the boundary.
- **Ink is specified relatively.** Trail counts and point counts are ratios
  against a reference canvas, not absolute numbers. Absolute counts make a
  4K render a quarter as dense as a 1080p one.
- **Generators return linear-light float arrays.** Conversion to 8-bit happens
  once, at the boundary.
- **Random recipes are drawn from a fraction of each range.** Three separate
  caps, because three separate failure modes appeared: expensive parameters
  multiply, so they are held low; tone parameters wash out at the top of their
  range; and structure parameters produce uninteresting compositions rather than
  variations of good ones.
- **The gallery is standard library only.** A wallpaper gallery is a
  single-user tool on localhost. A framework would mean a dependency to pin and
  keep current for one page.
- **The native app keeps GTK at the boundary.** Importing `genesis.app` never
  imports GTK or requires a display; those imports happen only inside
  `run_app`. Controller state, persistence, Tune grouping, and wallpaper
  actions are therefore testable headlessly.
- **Nothing assumes a session.** `gsettings` exits 0 while changing nothing when
  there is no DBus session, which is exactly the case a timer hits at login, so
  the check is explicit and happens before any call.

## License

Genesis is free software under the GPL-3.0-or-later; see `LICENSE`. It runs
entirely locally — no accounts, no network calls, no telemetry — so your
wallpapers and your library never leave the machine.
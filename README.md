# Genesis

A deterministic procedural wallpaper engine. Every image is described by a JSON
recipe, and the same recipe always produces a byte-identical PNG.

The recipe is the unit of work, not the image. Editing a parameter changes the
result in a diff you can read and revert, and re-running years later reproduces
the original pixels exactly.

## Quick start

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Change the wallpaper. Renders or reuses, then applies.
.venv/bin/python -m genesis.cli daily

# Browse and compose in the browser.
.venv/bin/python -m genesis.cli gallery
```

## The four commands you will actually use

```sh
genesis gallery          # the browser UI: compose, star, set as wallpaper
genesis daily            # today's wallpaper, rendered or reused
genesis daily --dry-run  # see what it would pick, and why
genesis apply            # set an existing image as the wallpaper
```

Everything else is for making a particular image.

### The gallery

`genesis gallery` serves `http://127.0.0.1:8765`. The parameter form is built
from the generator schemas the server sends, so adding a parameter to a
generator makes it appear in the UI with no change to the frontend. It is
loopback-only by design: it renders and writes to the library, which is more
capability than belongs on a network interface.

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
```

The timer runs `genesis daily` once a day, a little after 8am with a random
delay, and catches up on the next boot if the machine was asleep. A retry on the
same day produces the same wallpaper: the day seeds the choice, so a rerun is
harmless rather than a second surprise.

Starring is the main dial. A starred piece gets a bonus large enough to outrank
any amount of recency, so starring two or three images turns the library into a
shortlist you would be happy to see daily.

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

## Testing

```sh
.venv/bin/python -m pytest      # 614 tests
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
  muddy midpoints. The palette converts to linear light, grading happens there,
  and the conversion to sRGB happens once at the end.
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
- **Nothing assumes a session.** `gsettings` exits 0 while changing nothing when
  there is no DBus session, which is exactly the case a timer hits at login, so
  the check is explicit and happens before any call.
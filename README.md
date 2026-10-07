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

# Invent a recipe and render it.
.venv/bin/python -m genesis.cli gen random --seed 3 --palette tide

# Render a saved recipe, with overrides.
.venv/bin/python -m genesis.cli gen myrecipe.json --param trails=20000 --post grain=0.04

# See what exists.
.venv/bin/python -m genesis.cli list --params
.venv/bin/python -m genesis.cli list palettes
```

A recipe written next to its image looks like this:

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
.venv/bin/python -m pytest      # 333 tests
.venv/bin/ruff check genesis tests conftest.py
```

Three layers, each answering a different question:

- **Determinism** (`test_determinism.py`, `test_golden.py`) — the same recipe
  repeats, and it repeats *the intended image*. Golden digests turn any change
  in output into a named failure. Regenerate deliberately with
  `pytest --update-golden`; never to silence a failure.
- **Composition** (`test_image_quality.py`) — each generator produces a picture
  rather than a void: real contrast, a background that matches the palette floor,
  no edge banding from clamped samples, no clipping, and coverage that survives a
  change of resolution.
- **Units** — per module: color conversion in Oklab, noise tiling, post
  processing in linear light, parameter validation, the CLI surface.

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
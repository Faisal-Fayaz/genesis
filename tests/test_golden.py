"""Golden references: the tripwire that turns any visible change into a failure.

Determinism tests prove a recipe repeats byte-identically on one machine.
These prove that what it repeats still *looks* like the intended image — on
every machine. Exact pixel digests turned out to be host-lottery:
transcendental math rounds differently across CPUs and libms, so identical
code renders ulp-shifted pixels on different runners (proven the hard way:
CI went green and red on the same commit, on different hosts, with
determinism green in both runs).

So the references are PNGs and the comparison is tolerant: a couple of LSBs
at most anywhere, near-zero on average. Any real change — generator, color
pipeline, post, framing — moves thousands of pixels by tens of steps, orders
of magnitude above these bounds. The bounds are deliberately wide because the
true cross-machine variance is unmeasurable from any single box; what they
must catch is changes a person could see.

Regenerate deliberately, never to silence a failure:

    pytest tests/test_golden.py --update-golden

A failure here is only acceptable when the change was intended. Otherwise it
means an edit changed the look of every image of that kind, which is a bigger
decision than a code change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from genesis import render
from genesis.canon import canonical_json
from genesis.recipe import Recipe

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_FILE = GOLDEN_DIR / "golden.json"

# Small enough to keep the suite fast, large enough that a structural change
# cannot hide in resampling.
SIZE = (192, 108)

# Comparison bounds, in 8-bit steps. Cross-machine float noise lands far
# below one quantum almost everywhere; a visible change moves large areas
# by whole steps. See the module docstring for why this is tolerant at all.
MAX_ABS_LSB = 3
MEAN_ABS_LSB = 0.5

CASES: dict[str, dict] = {
    "flowfield_default": {
        "generator": "flowfield",
        "params": {"trails": 3000, "steps": 60, "supersample": 1},
        "seed": 3,
        "palette_name": "tide",
    },
    "flowfield_ember": {
        "generator": "flowfield",
        "params": {"trails": 3000, "steps": 60, "supersample": 1},
        "seed": 11,
        "palette_name": "ember",
    },
    "flowfield_post": {
        "generator": "flowfield",
        "params": {"trails": 3000, "steps": 60, "supersample": 1},
        "seed": 3,
        "palette_name": "tide",
        "post": {"grain": 0.04, "vignette": 0.3},
    },
    "attractor_clifford": {
        "generator": "attractor",
        "params": {
            "form": "clifford",
            "points": 120000,
            "zoom": 0.5,
            "supersample": 1,
        },
        "seed": 3,
        "palette_name": "tide",
    },
    "attractor_dejong": {
        "generator": "attractor",
        "params": {
            "form": "dejong",
            "points": 120000,
            "zoom": 0.5,
            "supersample": 1,
        },
        "seed": 3,
        "palette_name": "solar",
    },
    # Same coefficients at full extent, so a change to view framing shows up.
    "attractor_unscaled": {
        "generator": "attractor",
        "params": {"form": "clifford", "points": 120000, "supersample": 1},
        "seed": 3,
        "palette_name": "tide",
    },
    # Supersampling must change the image; if it stops doing so, it is dead code.
    "flowfield_supersampled": {
        "generator": "flowfield",
        "params": {"trails": 1000, "steps": 30, "supersample": 2},
        "seed": 3,
        "palette_name": "tide",
    },
    "splash_default": {
        "generator": "splash",
        "params": {"blobs": 60, "spread": 0.12, "supersample": 1},
        "seed": 3,
        "palette_name": "neon",
    },
    # Zero turbulence and fray pins the smooth-blob path; if the fray texture
    # stops affecting output, only this case moves.
    "splash_smooth": {
        "generator": "splash",
        "params": {
            "blobs": 60,
            "spread": 0.12,
            "turbulence": 0.0,
            "fray": 0.0,
            "supersample": 1,
        },
        "seed": 3,
        "palette_name": "neon",
    },
}


def recipe_for(name: str) -> Recipe:
    return Recipe(width=SIZE[0], height=SIZE[1], **CASES[name])


def reference_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.png"


def recipe_hash(recipe: Recipe) -> str:
    """A platform-independent fingerprint of the case definition itself."""
    return hashlib.sha256(canonical_json(recipe.to_dict()).encode()).hexdigest()


def load_hashes() -> dict[str, str]:
    if not GOLDEN_FILE.exists():
        return {}
    return json.loads(GOLDEN_FILE.read_text())


def as_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.int16)


def test_golden_references_exist() -> None:
    missing = [name for name in sorted(CASES) if not reference_path(name).exists()]
    assert not missing, (
        f"golden references missing for {missing}; regenerate with --update-golden"
    )
    assert GOLDEN_FILE.exists(), (
        "golden.json is missing; regenerate with --update-golden"
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_render_matches_golden(name: str) -> None:
    expected = as_array(Image.open(reference_path(name)))
    actual = as_array(render.render(recipe_for(name)).image)
    assert expected.shape == actual.shape, (
        f"{name} rendered {actual.shape}, reference is {expected.shape}"
    )
    gap = np.abs(actual - expected)
    worst, mean = float(gap.max()), float(gap.mean())
    assert worst <= MAX_ABS_LSB and mean <= MEAN_ABS_LSB, (
        f"{name} no longer renders as recorded "
        f"(worst {worst:.1f} steps, mean {mean:.3f}; "
        f"bounds are {MAX_ABS_LSB} and {MEAN_ABS_LSB}).\n"
        "If this change was intended, regenerate with --update-golden."
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_entries_are_current(name: str) -> None:
    """Guard against a stale reference that happens to match nothing.

    Confirms the recorded recipe hash belongs to *this* case definition, so
    renaming or editing a case cannot leave a passing but meaningless golden
    behind. Recipe hashes are pure data: identical on every machine.
    """
    hashes = load_hashes()
    assert name in hashes, f"{name} missing from golden.json"
    assert hashes[name] == recipe_hash(recipe_for(name))


class TestCasesAreDistinct:
    """Each case must cover ground the others do not."""

    def _images(self) -> dict[str, np.ndarray]:
        return {
            name: as_array(render.render(recipe_for(name)).image)
            for name in sorted(CASES)
        }

    def test_every_case_differs_from_every_other(self) -> None:
        images = self._images()
        names = sorted(images)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                gap = float(
                    np.abs(images[first] - images[second]).mean()
                )
                assert gap > 1.0, f"{first} and {second} render alike ({gap:.3f})"

    def test_supersampling_matters(self) -> None:
        images = self._images()
        gap = float(
            np.abs(
                images["flowfield_supersampled"] - images["flowfield_default"]
            ).mean()
        )
        assert gap > 1.0

    def test_framing_matters(self) -> None:
        images = self._images()
        gap = float(
            np.abs(
                images["attractor_clifford"] - images["attractor_unscaled"]
            ).mean()
        )
        assert gap > 1.0


def _update() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in sorted(CASES):
        recipe = recipe_for(name)
        render.render(recipe).image.save(reference_path(name), format="PNG")
        hashes[name] = recipe_hash(recipe)
    GOLDEN_FILE.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")


@pytest.fixture(autouse=True)
def regenerate_if_requested(request):
    """Recreate golden references when invoked with --update-golden."""
    if request.config.getoption("--update-golden", default=False):
        _update()
        pytest.skip("golden references regenerated")

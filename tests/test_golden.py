"""Golden hashes: the tripwire that turns any change in output into a failure.

Determinism tests prove a recipe repeats. These prove that what it repeats is
still the *intended* image. Any edit to a generator, the color pipeline, or the
post chain shifts one of these digests, which names the change that caused it.

Regenerate deliberately, never to silence a failure:

    pytest tests/test_golden.py --update-golden

A failure here is only acceptable when the change was intended. Otherwise it
means an edit changed the look of every image of that kind, which is a bigger
decision than a code change.

The values below are tied to the pinned numpy and Pillow in requirements.txt.
Resampling and some image encodings are not byte-stable across library versions,
so a dependency bump may legitimately change them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genesis import render
from genesis.recipe import Recipe

GOLDEN_FILE = Path(__file__).parent / "golden.json"

# Small enough to keep the suite fast, large enough that a structural change
# cannot hide in resampling.
SIZE = (192, 108)

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
}


def recipe_for(name: str) -> Recipe:
    return Recipe(width=SIZE[0], height=SIZE[1], **CASES[name])


def load_golden() -> dict[str, str]:
    if not GOLDEN_FILE.exists():
        return {}
    return json.loads(GOLDEN_FILE.read_text())


def test_golden_file_exists() -> None:
    assert GOLDEN_FILE.exists(), (
        "golden.json is missing; regenerate with --update-golden"
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_render_matches_golden(name: str) -> None:
    golden = load_golden()
    assert name in golden, f"{name} missing from golden.json"
    actual = render.render(recipe_for(name)).digest
    assert actual == golden[name], (
        f"{name} no longer renders as recorded.\n"
        f"  expected {golden[name]}\n"
        f"  actual   {actual}\n"
        "If this change was intended, regenerate with --update-golden."
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_entries_are_current(name: str) -> None:
    """Guard against a stale entry that happens to match nothing.

    Confirms the recorded digest belongs to *this* recipe, so renaming or
    editing a case cannot leave a passing but meaningless golden behind.
    """
    golden = load_golden()
    assert golden[name] == render.render(recipe_for(name)).digest


class TestCasesAreDistinct:
    """Each case must cover ground the others do not."""

    def test_every_case_differs_from_every_other(self) -> None:
        digests = {
            name: render.render(recipe_for(name)).digest for name in sorted(CASES)
        }
        assert len(set(digests.values())) == len(digests), (
            f"two cases render identically: {digests}"
        )

    def test_supersampling_matters(self) -> None:
        assert (
            render.render(recipe_for("flowfield_supersampled")).digest
            != render.render(recipe_for("flowfield_default")).digest
        )

    def test_framing_matters(self) -> None:
        assert (
            render.render(recipe_for("attractor_clifford")).digest
            != render.render(recipe_for("attractor_unscaled")).digest
        )


def _update() -> None:
    golden = {name: render.render(recipe_for(name)).digest for name in sorted(CASES)}
    GOLDEN_FILE.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")


@pytest.fixture(autouse=True)
def regenerate_if_requested(request):
    """Recreate golden.json when the suite is invoked with --update-golden."""
    if request.config.getoption("--update-golden", default=False):
        _update()
        pytest.skip("golden hashes regenerated")

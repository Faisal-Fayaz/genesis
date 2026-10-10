"""The curated presets must stay renderable, distinct, and usable as wallpapers.

A preset is a hand-picked composition, so the risk is not that it breaks at
runtime - that is caught by validation - but that it quietly stops being any
good. A retuned default, a reordered palette, or a mood rebalance can turn a
chosen composition into a clone of its neighbour or into something unreadable
behind desktop icons, and neither shows up in a digest check.

The render here is deliberately small. The point is not to inspect the picture,
it is to catch a preset that has stopped covering the canvas or stopped having
any contrast, both of which are cheap to measure and impossible to notice from
a diff.
"""

from __future__ import annotations

import numpy as np
import pytest

from genesis import presets as curated
from genesis import render
from genesis.color import linear_to_srgb
from genesis.recipe import PRESETS

SIZE = (192, 108)

# Rec.709 luma weights, for deciding whether a pixel is "quiet" relative to the
# palette's background.
_LUMA = np.array([0.2126, 0.7152, 0.0722])

# A wallpaper needs quiet regions for desktop icons and text. This is the share
# of the frame that must stay near the background colour.
MIN_QUIET = 0.30

# And the background itself has to still be there. A render whose darkest pixel
# is far above the palette's background is washed out, which is how every "soft"
# preset failed its first draft: `post.contrast` pivots at 0.5 in linear light,
# so 0.9 moved a near-black background from 10/255 to 65/255.
MAX_BACKGROUND_LIFT = 30.0

# A blank or single-hue fill has no structure to look at.
MIN_CONTRAST = 8.0


def measure(preset: curated.Curated, width: int = SIZE[0], height: int = SIZE[1]) -> dict[str, float]:
    """Luminance statistics for one preset at the requested size."""
    result = render.render(preset.recipe(width, height))
    values = np.asarray(result.image, dtype=np.float64)
    luma = values @ _LUMA
    background = float(
        (linear_to_srgb(result.recipe.effective_palette().to_linear(0.0)) * 255.0) @ _LUMA
    )
    return {
        "luma": luma,
        "background": background,
        # How far above the true background the darkest pixel sits. Zero means
        # the background is present; large means the whole frame was lifted.
        "lift": float(luma.min() - background),
        "quiet": float((luma < background + 45).mean()),
        "contrast": float(luma.std()),
    }


class TestPresetsAreWellFormed:
    def test_names_are_unique(self) -> None:
        names = curated.names()
        assert len(names) == len(set(names))

    def test_every_mood_has_presets(self) -> None:
        for mood in curated.MOODS:
            assert curated.for_mood(mood), f"no presets tagged {mood}"

    def test_every_preset_names_a_real_palette(self) -> None:
        for preset in curated.CURATED:
            assert preset.palette in PRESETS, preset.name

    def test_every_preset_names_a_real_mood(self) -> None:
        for preset in curated.CURATED:
            assert preset.mood in curated.MOODS, preset.name

    def test_every_preset_builds_a_valid_recipe(self) -> None:
        for preset in curated.CURATED:
            recipe = preset.recipe(*SIZE)
            recipe.validate()

    def test_every_preset_records_why_it_is_here(self) -> None:
        """A preset without a note is one nobody will know when to retire."""
        for preset in curated.CURATED:
            assert preset.note.strip(), preset.name

    def test_unknown_preset_lists_the_known_ones(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            curated.get("does-not-exist")
        assert "silk" in str(excinfo.value)


class TestPresetsLookLikeWallpapers:
    @pytest.mark.parametrize("preset", curated.CURATED, ids=curated.names())
    def test_leaves_room_for_desktop_icons(self, preset: curated.Curated) -> None:
        """Enough of the frame must stay near the background to read text on."""
        assert measure(preset)["quiet"] >= MIN_QUIET, (
            f"{preset.name} has only {measure(preset)['quiet']:.0%} quiet area"
        )

    @pytest.mark.parametrize("preset", curated.CURATED, ids=curated.names())
    def test_background_survives_to_the_output(self, preset: curated.Curated) -> None:
        """The darkest pixel must still be near the palette's background.

        This remains the pixel-level arbiter for washed-out presets. Post grades
        act on the background as well as on trails; corrected sRGB contrast is
        gentler than the old linear pivot, but it can still lift a dark frame if
        combined with lift, brightness, bloom, or excessive exposure.
        """
        stats = measure(preset)
        assert stats["lift"] < MAX_BACKGROUND_LIFT, (
            f"{preset.name} lifted its background by {stats['lift']:.0f}/255"
        )

    @pytest.mark.parametrize("preset", curated.CURATED, ids=curated.names())
    def test_has_real_contrast(self, preset: curated.Curated) -> None:
        assert measure(preset)["contrast"] > MIN_CONTRAST, (
            f"{preset.name} contrast {measure(preset)['contrast']:.1f}"
        )

    @pytest.mark.parametrize("preset", curated.CURATED, ids=curated.names())
    def test_does_not_clip_to_white(self, preset: curated.Curated) -> None:
        """Clipped pixels mean the exposure is pushing the palette off the end."""
        result = render.render(preset.recipe(*SIZE))
        values = np.asarray(result.image, dtype=np.float64)
        clipped = float((values.min(axis=2) > 250).mean())
        assert clipped < 0.02, f"{preset.name} clips {clipped:.1%} of the frame"


class TestPresetsAreDistinct:
    """Two presets that render the same image are one preset with two names."""

    def test_no_two_presets_share_a_digest(self) -> None:
        seen: dict[str, str] = {}
        for preset in curated.CURATED:
            digest = render.render(preset.recipe(*SIZE)).digest
            assert digest not in seen, f"{preset.name} renders identically to {seen[digest]}"
            seen[digest] = preset.name

    def test_each_mood_uses_more_than_one_palette(self) -> None:
        """A mood shown as one colour reads as a category, not a choice."""
        for mood in curated.MOODS:
            palettes = {preset.palette for preset in curated.for_mood(mood)}
            assert len(palettes) >= 3, f"{mood} uses only {sorted(palettes)}"


class TestPresetsSurviveResolutionChange:
    """A preset tuned on a preview must still work at full size.

    Ink scales with the canvas inside the generator, so a preset's composition
    is meant to carry across resolutions. This is what makes them usable as
    daily wallpapers: the same preset renders correctly at whatever size the
    display is.
    """

    @pytest.mark.parametrize("preset", curated.CURATED, ids=curated.names())
    def test_quiet_area_is_stable_across_sizes(self, preset: curated.Curated) -> None:
        """A preset tuned on a preview must still work at full size.

        Ink scales with the canvas inside the generator, so a preset's
        composition is meant to carry across resolutions. This is what makes them
        usable as daily wallpapers: the same preset renders correctly at whatever
        size the display is.
        """
        small = measure(preset, 192, 108)["quiet"]
        large = measure(preset, 576, 324)["quiet"]
        assert 0.6 < (small / large) < 1.7, f"{preset.name}: {small:.3f} vs {large:.3f}"


class TestCuratedLookup:
    def test_get_returns_the_preset(self) -> None:
        assert curated.get("silk").mood == "rich"

    def test_moods_of_reports_in_canonical_order(self) -> None:
        selected = curated.CURATED[::3]
        assert curated.moods_of(selected) == [
            mood for mood in curated.MOODS if mood in {p.mood for p in selected}
        ]

    def test_for_mood_of_an_unknown_mood_is_empty(self) -> None:
        assert curated.for_mood("nope") == []
"""The floor, the void behind it and the actors must be three different things.

Two constants in `scenarios.base` describe this and neither was checked by
anything, which is how both came to be wrong at once:

  * `MIN_SCENERY_SEPARATION` survived a rewrite. `_recolour_scenery` used to
    reject-sample the floor against it, gave up when the whole palette landed
    inside the exclusion zone, and fell back to the hard-coded grey it was
    replacing. The rewrite scans for the most separated value instead, which
    is strictly better -- and left the constant naming a bar nobody applied.
  * `BACKDROP_VALUE` went dead when the ground became an 80 m dome, because a
    dome curves up behind the scene and IS the backdrop, so one colour was the
    whole surround. When the dome went back to being a render-only backdrop at
    the HDRI levels, the slab returned and the void behind it returned with
    it -- still being painted the floor's own colour. Measured on `drop` at
    seed 91731: floor and void both (0.795, 0.812, 0.82). A 6 m slab the same
    shade as the sky behind it has no horizon.

So this asks the two questions those constants are for, over every scenario at
several seeds, at the levels that HAVE a chosen colour -- from L2 up an HDRI
supplies the ground and the sky together and `_recolour_scenery` returns.
"""
import numpy as np
import pytest

from physloc import scenarios
from physloc.residuals.laws import _srgb_to_lab
from physloc.scenarios.base import MIN_SCENERY_SEPARATION

#: How far the void has to sit from the floor it meets at the horizon, in Lab.
#: LOWER THAN THE ACTOR BAR, and measured rather than chosen: the void is
#: squeezed from both sides -- its band is the darker one, it must clear every
#: actor, and on `shadow_track` one of those "actors" is a near-black cast
#: shadow. The worst case over thirteen scenarios and eight seeds is 20.8, on
#: `shadow_track` at L1, seed 0. Widening `BACKDROP_VALUE` buys 5 Lab there and
#: costs the void being recognisably a void, so the band stays as it is and
#: this is the honest bar.
MIN_HORIZON_SEPARATION = 18.0

SEEDS = range(8)
#: The levels that choose their own colours. `_recolour_scenery` returns at
#: `background == "hdri"`, so L2 and L3 have nothing to check here.
SOLID = ("L0", "L1")


def _lab(rgb) -> np.ndarray:
    return _srgb_to_lab(np.asarray(rgb, np.float64))


def _specs(name):
    tier = scenarios.TIERS["debug"]
    for cx in SOLID:
        for seed in SEEDS:
            yield cx, seed, scenarios.get(name).sample(seed, tier, cx)


@pytest.mark.parametrize("name", sorted(scenarios.available()))
def test_the_floor_stands_clear_of_every_actor(name):
    """What `MIN_SCENERY_SEPARATION` says, actually applied.

    The shadow counts as an actor: `shadow_track` stages its cast shadow as a
    real near-black body, and the family is about whether that shadow tracks
    the object faithfully -- which nobody can judge on a floor the same
    darkness.
    """
    for cx, seed, spec in _specs(name):
        floors = [b for b in spec.bodies if b.role == "floor"]
        if not floors:
            continue
        lf = _lab(floors[0].color)
        for body in spec.bodies:
            if body.role not in ("actor", "shadow"):
                continue
            gap = float(np.linalg.norm(lf - _lab(body.color)))
            assert gap >= MIN_SCENERY_SEPARATION, (
                "%s %s seed %d: floor is %.1f Lab from %s"
                % (name, cx, seed, gap, body.name))


@pytest.mark.parametrize("name", sorted(scenarios.available()))
def test_the_horizon_reads(name):
    """The void behind the slab is not the slab's own colour."""
    for cx, seed, spec in _specs(name):
        floors = [b for b in spec.bodies if b.role == "floor"]
        if not floors:
            continue
        gap = float(np.linalg.norm(_lab(floors[0].color)
                                   - _lab(spec.background_color)))
        assert gap >= MIN_HORIZON_SEPARATION, (
            "%s %s seed %d: floor %s and void %s are %.1f Lab apart"
            % (name, cx, seed, floors[0].color, spec.background_color, gap))


@pytest.mark.parametrize("name", sorted(scenarios.available()))
def test_the_backdrop_is_drawn_from_its_own_band(name):
    """And not, as it was, copied from the floor.

    A separation test alone would pass on a void drawn from the FLOOR's band
    that happened to land far away, which is not what `BACKDROP_VALUE` means.
    """
    from physloc.scenarios.base import BACKDROP_SATURATION, BACKDROP_VALUE
    import colorsys

    for cx, seed, spec in _specs(name):
        if not any(b.role == "floor" for b in spec.bodies):
            continue
        h, s, v = colorsys.rgb_to_hsv(*spec.background_color)
        assert BACKDROP_VALUE[0] - 1e-6 <= v <= BACKDROP_VALUE[1] + 1e-6, (
            "%s %s seed %d: void value %.3f outside %s"
            % (name, cx, seed, v, (BACKDROP_VALUE,)))
        assert s <= BACKDROP_SATURATION[1] + 1e-6, (
            "%s %s seed %d: void saturation %.3f above %.3f"
            % (name, cx, seed, s, BACKDROP_SATURATION[1]))

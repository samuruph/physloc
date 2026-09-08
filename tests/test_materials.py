"""What a body is made of: visible, weighed, and drawn in a sane proportion.

Materials exist so that mass is legible -- a steel cube looks like steel and is
dense -- and every property here protects one half of that bargain.
"""
import collections
import statistics

import numpy as np
import pytest

from physloc.scenarios import materials as M

#: The seven-material set the scenarios' contact parameters were tuned against.
#: Its mean density is the number the palette must not drift away from.
TUNED_MEAN = 2313.0


def test_the_palette_is_visibly_varied():
    """The reason L1 looked like L0.

    Seven near-matte dielectrics under one sun are seven flat colours with
    slightly different sheen, which is what L0 already is. Metal reads
    instantly, and from L2 up it also REFLECTS the environment, which is what
    ties an object to its background.
    """
    metals = [m for m in M.MATERIALS.values() if m.metallic]
    glassy = [m for m in M.MATERIALS.values() if m.transmission]
    assert len(M.MATERIALS) >= 12
    assert len(metals) >= 3, "one metal in seven left six clips in seven matte"
    assert glassy, "nothing transmissive at all"
    specs = {round(m.specular, 2) for m in M.MATERIALS.values()}
    assert len(specs) >= 4, (
        "specular was left at Blender's 0.5 for everything, so ceramic had the "
        "same highlight as rubber: %s" % sorted(specs))


def test_transmission_is_not_a_cue_for_mass():
    """At least two transmissive materials, at different densities.

    With glass (2500) as the only see-through option, "transparent" and
    "middling heavy" were the same fact, and a model could read one off the
    other. Ice at 920 breaks it.
    """
    d = sorted(m.density for m in M.MATERIALS.values() if m.transmission)
    assert len(d) >= 2, d
    assert max(d) / min(d) >= 2.0, "transmissive materials cluster at one mass: %s" % d


def test_the_draw_keeps_the_mass_regime_it_was_tuned_in():
    """Weighted, because the palette is not uniform in density.

    Adding three dense metals to make L1 legible also moved a UNIFORM draw's
    mean density from 2313 to 3458 and doubled the median -- a change to the
    physics smuggled in by a change to the appearance. Every scenario's contact
    tuning assumes a mass regime; drifting out of it while the labels say
    nothing changed is the kind of thing nobody notices for months.
    """
    w = {n: M.MATERIALS[n].weight for n in M.ACTOR_MATERIALS}
    total = sum(w.values())
    mean = sum(M.MATERIALS[n].density * x for n, x in w.items()) / total
    assert abs(mean - TUNED_MEAN) < 0.20 * TUNED_MEAN, (
        "weighted mean density %.0f has drifted from the tuned %.0f" % (mean, TUNED_MEAN))

    uniform = statistics.mean(M.MATERIALS[n].density for n in M.ACTOR_MATERIALS)
    assert uniform > mean, (
        "the weights should pull the mean DOWN -- uniform %.0f, weighted %.0f"
        % (uniform, mean))


def test_pick_actually_follows_the_weights():
    """A weight nothing reads is a comment. This is the test that would have
    caught `pick` still drawing uniformly."""
    rng = np.random.RandomState(0)
    got = collections.Counter(M.pick(rng) for _ in range(20000))
    total = sum(got.values())
    w = sum(M.MATERIALS[n].weight for n in M.ACTOR_MATERIALS)
    for name in M.ACTOR_MATERIALS:
        want = M.MATERIALS[name].weight / w
        assert abs(got[name] / total - want) < 0.01, (
            "%s drawn %.1f%% of the time, wanted %.1f%%"
            % (name, 100 * got[name] / total, 100 * want))


def test_the_mass_spread_stays_solver_friendly():
    """Cork to copper, and no wider.

    Real foam is ~60 kg/m^3, which next to a dense metal is 150:1 -- and a body
    that light skitters under contact parameters tuned near 1 kg. The palette
    may grow; the RATIO may not.
    """
    d = [m.density for m in M.MATERIALS.values()]
    assert max(d) / min(d) <= 45.0, "density spread %.0fx is past the solver's comfort" % (
        max(d) / min(d))


def test_every_material_has_a_light_and_a_heavy_neighbour():
    """No gaps big enough to make the distribution bimodal -- a palette of four
    light and three very heavy materials is two palettes."""
    d = sorted(m.density for m in M.MATERIALS.values())
    gaps = [(b / a, a, b) for a, b in zip(d, d[1:])]
    worst = max(gaps)
    assert worst[0] <= 3.5, (
        "a %.1fx gap between %.0f and %.0f splits the palette in two"
        % worst)


def test_scenery_draws_from_a_narrower_set():
    """A glass ramp is a puzzle rather than a surface, and a mirror floor
    throws the actor's reflection across the scene. Neither is the difficulty
    this dataset measures."""
    assert set(M.SCENERY_MATERIALS) <= set(M.MATERIALS)
    assert len(M.SCENERY_MATERIALS) < len(M.ACTOR_MATERIALS)
    for name in M.SCENERY_MATERIALS:
        assert not M.MATERIALS[name].transmission, name


@pytest.mark.parametrize("name", sorted(M.MATERIALS))
def test_a_material_is_self_consistent(name):
    m = M.MATERIALS[name]
    assert m.density > 0 and m.weight > 0
    assert 0.0 <= m.roughness <= 1.0
    assert m.metallic in (0.0, 1.0), "partial metallic reads as neither"
    assert 0.0 <= m.specular <= 1.0
    assert 0.0 <= m.transmission <= 1.0
    if m.transmission:
        assert 1.0 < m.ior < 2.5, "%s: ior %.2f is not a real refractive index" % (name, m.ior)
        assert not m.metallic, "%s is both a metal and transparent" % name

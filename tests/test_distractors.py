"""Distractors must distract and nothing else.

Every property here was a bug first, found by rendering a clip and reading the
numbers rather than by anything the suite was checking. They are cheap to
assert and expensive to notice.
"""
import numpy as np
import pytest

from physloc import scenarios
from physloc.scenarios import TIERS
from physloc.scenarios._common import (DISTRACTOR_SIZE, KEEP_CLEAR_RADII,
                                       MAX_ASPECT, SIGHTLINE_RADII)
from physloc.scenarios.base import (CONDITION_CYCLE, EXTRA_OBJECTS,
                                    condition_for, has_distractors)

NAMES = sorted(scenarios.available())

#: Distractors are a CONDITION, not a level: every level places them on the
#: clips whose condition asks for them. Tested at L0, where the rest of the
#: scene is held plainest, so anything the clutter does is the only thing that
#: changed.
LEVEL = "L0"
SEEDS = range(6)


#: The variant indices that DO and DO NOT carry clutter, so a placement test
#: can ask the same question on every seed instead of checking empty scenes
#: five times out of six. `test_distractors_land_only_on_their_declared_share`
#: is what pins the ratio itself.
PERIOD = len(CONDITION_CYCLE)
CLUTTERED = [v for v in range(PERIOD) if has_distractors(condition_for(v))]
CLEAN = [v for v in range(PERIOD) if not has_distractors(condition_for(v))]


def _specs(name, level=LEVEL, seeds=None, cluttered=True):
    """One spec per seed, all on variants that carry clutter (or that do not).

    The condition is chosen by variant INDEX, so a placement test picks
    indices rather than forcing a probability: `PERIOD * k + CLUTTERED[0]` is
    a cluttered variant for every k, which gives as many independent scenes as
    wanted while every one of them actually contains distractors.
    """
    sc = scenarios.get(name)
    want = CLUTTERED if cluttered else CLEAN
    picks = [PERIOD * k + want[k % len(want)]
             for k in range(len(SEEDS) if seeds is None else seeds)]
    return [sc.sample(777 + v, TIERS["debug"], level, variant=v,
                      n_variants=PERIOD) for v in picks]


def _split(spec):
    actors = [b for b in spec.bodies if b.role == "actor" and not b.dormant]
    return actors, [b for b in spec.bodies if b.role == "distractor"]


def test_distractors_land_only_on_their_declared_share():
    """The ratio, at the declared value, on every scenario.

    Clutter is a CONDITION rather than a level, so what pins it is not "which
    level" but "which variants, and how many of them". The condition cycle
    decides by variant INDEX, so every scenario gets the same share instead of
    each flipping its own coin -- measured with independent draws, a nominal
    20% axis ranged from 13% to 31% across thirteen scenarios, and any
    per-scenario comparison inherited that as a confound.
    """
    n = 2 * PERIOD
    want = [v for v in range(n) if has_distractors(condition_for(v))]
    assert len(want) == 2 * len(CLUTTERED), "the cycle owes its declared count"
    for name in NAMES:
        sc = scenarios.get(name)
        got = [v for v in range(n)
               if any(b.role == "distractor" for b in
                      sc.sample(777 + v, TIERS["debug"], LEVEL,
                                variant=v, n_variants=PERIOD).bodies)]
        assert got == want, "%s cluttered variants %s, expected %s" % (
            name, got, want)


def test_a_short_run_is_entirely_uncluttered():
    """Asked for explicitly: below the threshold a run is the easy case.

    The plain clips come FIRST in `CONDITION_CYCLE`, so a run shorter than the
    first `distractors` index never reaches one.
    """
    first = min(CLUTTERED)
    for n in range(1, first + 1):
        for name in NAMES:
            sc = scenarios.get(name)
            assert not any(
                b.role == "distractor"
                for v in range(n)
                for b in sc.sample(777 + v, TIERS["debug"], LEVEL,
                                   variant=v).bodies), (
                "%s cluttered a %d-variant run" % (name, n))


@pytest.mark.parametrize("name", NAMES)
def test_the_scene_gets_the_distractors_it_asks_for(name):
    """Placement has to actually succeed.

    The constraints pull against each other -- clear of the actor's path, not
    in front of it, inside the frame -- and a placement band pinned to a fixed
    fraction of the frame could sit entirely inside the exclusion, so the
    sampler drew from a region where it could never succeed. `drop` and
    `pyramid_impact` placed ZERO on some seeds that way.
    """
    lo, hi = EXTRA_OBJECTS
    counts = set()
    for sp in _specs(name, seeds=8):
        got = int(sp.notes.get("n_distractors_placed") or 0)
        assert lo <= got <= hi, (
            "%s seed %d placed %d, outside %s -- placement gave up"
            % (name, sp.seed, got, EXTRA_OBJECTS))
        counts.add(got)
    assert len(counts) >= 2, (
        "%s placed the same count every time (%s) -- the draw is not varying"
        % (name, sorted(counts)))


@pytest.mark.parametrize("name", NAMES)
def test_a_distractor_never_stands_in_the_actors_way(name):
    """Physical clearance. A distractor that joins the collision changes the
    event the clip is labelled for, and no annotation says it did."""
    for sp in _specs(name):
        actors, dis = _split(sp)
        for d in dis:
            p = np.asarray(d.position, np.float64)
            for a in actors:
                keep = float(a.bounding_radius) * KEEP_CLEAR_RADII
                gap = float(np.linalg.norm(p - np.asarray(a.position)))
                assert gap >= keep, (
                    "%s: %s is %.2f m from %s, inside its %.2f m keep-clear"
                    % (name, d.name, gap, a.name, keep))


@pytest.mark.parametrize("name", NAMES)
def test_a_distractor_never_stands_in_front_of_the_actor(name):
    """The one that cost a severity map.

    Distance in WORLD space is not enough: a distractor three metres from the
    actor can sit squarely between it and the camera. When that happened the
    residual was untouched -- 45.57, score 1.0 -- while `severity_map` went
    from 0.229 to entirely empty, because severity is painted through the
    segmentation of a body the camera can no longer see. Reading the residual
    alone would have called that clip healthy.
    """
    for sp in _specs(name):
        actors, dis = _split(sp)
        eye = np.asarray(sp.camera_position, np.float64)
        for d in dis:
            to_d = np.asarray(d.position, np.float64) - eye
            dd = float(np.linalg.norm(to_d))
            for a in actors:
                to_a = np.asarray(a.position, np.float64) - eye
                da = float(np.linalg.norm(to_a))
                if dd >= da or da < 1e-6 or dd < 1e-6:
                    continue          # behind the actor: a harmless backdrop
                sep = np.arccos(np.clip(np.dot(to_d / dd, to_a / da), -1, 1))
                need = (np.arctan2(float(d.bounding_radius), dd)
                        + np.arctan2(float(a.bounding_radius) * SIGHTLINE_RADII, da))
                assert sep >= need * 0.999, (
                    "%s: %s occludes %s (%.3f rad apart, needs %.3f)"
                    % (name, d.name, a.name, sep, need))


@pytest.mark.parametrize("name", NAMES)
def test_a_distractor_is_big_enough_to_be_one(name):
    """Sized against the ACTOR, not the frame.

    Sizing off the frustum gave distractors 6 to 34 pixels against the actor's
    83 at debug resolution -- specks rather than distractions.

    The band applies to the size a distractor is DRAWN at; `vary_dims` then
    varies its axes around that, so a box can end up half an aspect step
    outside on its longest side. The bound below allows for exactly that and no
    more -- widening it further would stop it catching the bug it exists for.
    """
    lo, hi = DISTRACTOR_SIZE
    slack = MAX_ASPECT ** 0.5
    for sp in _specs(name):
        actors, dis = _split(sp)
        med = float(np.median([a.bounding_radius for a in actors]))
        for d in dis:
            ratio = float(d.bounding_radius) / med
            assert lo / slack - 1e-6 <= ratio <= hi * slack + 1e-6, (
                "%s: %s is %.2fx the actor, outside %s widened for aspect"
                % (name, d.name, ratio, DISTRACTOR_SIZE))


@pytest.mark.parametrize("name", NAMES)
def test_no_family_can_target_a_distractor(name):
    """They are scenery. Every injector selects on `role == "actor"`, so a
    distractor is excluded by construction -- this pins that the role stays
    distinct and that ids cannot collide with a scenario's own."""
    from physloc.injectors import _geom

    for sp in _specs(name):
        actors, dis = _split(sp)
        ids = {int(b.segmentation_id) for b in sp.bodies if b.role != "distractor"}
        for d in dis:
            assert d.role == "distractor"
            assert int(d.segmentation_id) not in ids, (
                "%s: %s reuses segmentation id %d" % (name, d.name, d.segmentation_id))
        assert not any(b.role == "distractor" for b in _geom.actors(sp))


@pytest.mark.parametrize("name", NAMES)
def test_adding_distractors_changes_nothing_else(name):
    """A cluttered scene is a clean scene with more bodies in it.

    Distractors draw from their own salted stream, so they cannot shift a
    physics or appearance value the scenario already sampled. Without that, the
    clip that happened to draw clutter would also be a different clip in every
    other respect, and the condition would stop being a condition.

    Compared at the SAME seed with the condition switched -- which is the only
    way to hold everything else fixed now that clutter is chosen by variant
    index rather than by a probability.
    """
    import physloc.scenarios.base as B

    sc = scenarios.get(name)
    clean_v, dirty_v = min(CLEAN), min(CLUTTERED)
    for seed in (777, 812, 4242):
        a = sc.sample(seed, TIERS["debug"], LEVEL, variant=clean_v)
        # Same variant index, so every other draw matches; only the condition
        # lookup is overridden.
        saved = B.has_distractors
        B.has_distractors = lambda v: True
        try:
            b = sc.sample(seed, TIERS["debug"], LEVEL, variant=clean_v)
        finally:
            B.has_distractors = saved
        common = [x for x in b.bodies if x.role != "distractor"]
        assert len(common) == len(a.bodies), name
        assert len(b.bodies) > len(a.bodies), "%s placed nothing" % name
        for x, y in zip(a.bodies, common):
            for f in ("kind", "position", "scale", "mass", "material",
                      "velocity", "color", "friction", "restitution"):
                assert getattr(x, f) == getattr(y, f), (
                    "%s: %s.%s changed when distractors were added"
                    % (name, x.name, f))
        assert a.camera_position == b.camera_position
        assert a.camera_motion_kind == b.camera_motion_kind

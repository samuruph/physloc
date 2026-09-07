"""Distractors must distract and nothing else.

Every property here was a bug first, found by rendering a clip and reading the
numbers rather than by anything the suite was checking. They are cheap to
assert and expensive to notice.
"""
import numpy as np
import pytest

import dataclasses

from physloc import scenarios
from physloc.scenarios import TIERS
from physloc.scenarios._common import (DISTRACTOR_SIZE, KEEP_CLEAR_RADII,
                                       MAX_ASPECT, SIGHTLINE_RADII)
from physloc.scenarios.base import COMPLEXITY, stratify

NAMES = sorted(scenarios.available())

#: Distractors are an ORTHOGONAL AXIS, not a rung: every level places them on
#: `distractor_share` of its clips. Tested at L0, where the rest of the scene is
#: held plainest, so anything the clutter does is the only thing that changed.
LEVEL = "L0"
SEEDS = range(6)


def _forced(level=LEVEL, share=1.0):
    """`COMPLEXITY` with one level's distractor share overridden.

    The axis is stratified by variant index, so at its declared 30% only some
    variants carry clutter -- and a test that wants to check placement on every
    seed would otherwise be checking empty scenes most of the time. Forcing the
    share to 1.0 asks the same placement code the same question on every seed;
    `test_distractors_land_only_on_their_declared_share` is what pins the ratio
    itself.
    """
    return dict(COMPLEXITY,
                **{level: dataclasses.replace(COMPLEXITY[level],
                                              distractor_share=share)})


def _specs(name, level=LEVEL, share=1.0, seeds=SEEDS, monkeypatch=None):
    import physloc.scenarios.base as B

    sc = scenarios.get(name)
    saved = B.COMPLEXITY
    B.COMPLEXITY = _forced(level, share) if share is not None else saved
    try:
        return [sc.sample(777 + v, TIERS["debug"], level, variant=v)
                for v in seeds]
    finally:
        B.COMPLEXITY = saved


def _split(spec):
    actors = [b for b in spec.bodies if b.role == "actor" and not b.dormant]
    return actors, [b for b in spec.bodies if b.role == "distractor"]


def test_distractors_land_only_on_their_declared_share():
    """The ratio, at the declared value, on every scenario.

    Clutter is an axis rather than a rung, so what pins it is not "which level"
    but "which fraction, spread how". `stratify` decides, by variant INDEX, so
    every scenario gets the same share instead of each flipping its own coin --
    measured with independent draws, a nominal 20% axis ranged from 13% to 31%
    across thirteen scenarios, and any per-scenario comparison inherited that
    as a confound.
    """
    share = COMPLEXITY[LEVEL].distractor_share
    n = 20
    want = [v for v in range(n) if stratify(v, share)]
    assert len(want) == int(n * share), "the stratifier owes the declared share"
    for name in NAMES:
        got = [sp.variant for sp in _specs(name, share=None, seeds=range(n))
               if any(b.role == "distractor" for b in sp.bodies)]
        assert got == want, "%s cluttered variants %s, expected %s" % (
            name, got, want)


def test_a_short_run_is_entirely_uncluttered():
    """Asked for explicitly: below the threshold a run is the easy case. The
    axis fires on the LAST variant of each block, so a run of three never
    reaches it."""
    share = COMPLEXITY[LEVEL].distractor_share
    for n in range(1, int(1.0 / share)):
        for name in NAMES:
            assert not any(
                b.role == "distractor"
                for sp in _specs(name, share=None, seeds=range(n))
                for b in sp.bodies), "%s cluttered a %d-variant run" % (name, n)


@pytest.mark.parametrize("name", NAMES)
def test_the_scene_gets_the_distractors_it_asks_for(name):
    """Placement has to actually succeed.

    The constraints pull against each other -- clear of the actor's path, not
    in front of it, inside the frame -- and a placement band pinned to a fixed
    fraction of the frame could sit entirely inside the exclusion, so the
    sampler drew from a region where it could never succeed. `drop` and
    `pyramid_impact` placed ZERO on some seeds that way.
    """
    want = COMPLEXITY[LEVEL].n_distractors
    for sp in _specs(name):
        got = sp.notes.get("n_distractors_placed")
        assert got == want, (
            "%s seed %d placed %s of %d distractors"
            % (name, sp.seed, got, want))


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
    physics or appearance value the scenario already sampled. Without that,
    the clip that happened to draw clutter would also be a different clip in
    every other respect, and the axis would stop being an axis.

    Compared at the SAME seed and variant with the share forced off and on,
    which is the only way to hold everything else fixed now that clutter is a
    ratio rather than a level.
    """
    for a, b in zip(_specs(name, share=0.0), _specs(name, share=1.0)):
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

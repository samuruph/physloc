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
                                       SIGHTLINE_RADII)
from physloc.scenarios.base import COMPLEXITY

NAMES = sorted(scenarios.available())
LEVEL = "L3"
SEEDS = range(6)


def _specs(name, level=LEVEL):
    sc = scenarios.get(name)
    return [sc.sample(777 + v, TIERS["debug"], level, variant=v) for v in SEEDS]


def _split(spec):
    actors = [b for b in spec.bodies if b.role == "actor" and not b.dormant]
    return actors, [b for b in spec.bodies if b.role == "distractor"]


def test_no_distractors_below_their_level():
    """They arrive at L3 and not before -- that is what the level means."""
    for level in ("L0", "L1", "L2"):
        assert COMPLEXITY[level].n_distractors == 0
        for name in NAMES:
            assert not any(b.role == "distractor"
                           for sp in _specs(name, level) for b in sp.bodies)


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
    """
    lo, hi = DISTRACTOR_SIZE
    for sp in _specs(name):
        actors, dis = _split(sp)
        med = float(np.median([a.bounding_radius for a in actors]))
        for d in dis:
            ratio = float(d.bounding_radius) / med
            assert lo - 1e-6 <= ratio <= hi + 1e-6, (
                "%s: %s is %.2fx the actor, outside %s"
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
    """An L3 scene is an L2 scene with more bodies in it.

    Distractors draw from their own salted stream, so they cannot shift a
    physics or appearance value the scenario already sampled. Without that,
    turning the level up would silently resample every clip.
    """
    sc = scenarios.get(name)
    for v in SEEDS:
        a = sc.sample(777 + v, TIERS["debug"], "L2", variant=v)
        b = sc.sample(777 + v, TIERS["debug"], "L3", variant=v)
        common = [x for x in b.bodies if x.role != "distractor"]
        assert len(common) == len(a.bodies), name
        for x, y in zip(a.bodies, common):
            for f in ("kind", "position", "scale", "mass", "material",
                      "velocity", "color", "friction", "restitution"):
                assert getattr(x, f) == getattr(y, f), (
                    "%s: %s.%s changed when distractors were added"
                    % (name, x.name, f))
        assert a.camera_position == b.camera_position
        assert a.camera_motion_kind == b.camera_motion_kind

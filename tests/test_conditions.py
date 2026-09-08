"""The difficulty conditions: which clips get harder, how, and in what share.

Camera motion, distractors and multiple culprits are three ways to make a clip
harder, and `CONDITION_CYCLE` says which combinations exist and how often.

**Named cells, not independent coin flips.** Independent per-axis ratios were
the first design and they blur what a benchmark reports: a "moving camera" clip
might also carry clutter, so the marginal comparison mixes two effects and the
per-condition counts are only exact in expectation. One condition per clip
makes every count exact and every comparison against `standard` isolate one
change.
"""
import numpy as np
import pytest

from physloc import injectors, scenarios
from physloc.scenarios import TIERS
from physloc.scenarios.base import (CONDITION_CYCLE, MULTI_ACTORS,
                                    MULTI_CULPRITS, condition_for,
                                    condition_share, has_distractors,
                                    has_moving_camera, has_multi)

NAMES = sorted(scenarios.available())
LEVEL = "L0"
PERIOD = len(CONDITION_CYCLE)
SEED = 777


def _spec(name, variant, level=LEVEL):
    return scenarios.get(name).sample(SEED + variant, TIERS["debug"], level,
                                      variant=variant)


def _actors(spec):
    return [b for b in spec.bodies if b.role == "actor" and not b.dormant]


def test_the_cycle_is_the_agreed_shape():
    """Six plain clips, then one of each condition, then the one combination
    worth having. Marginals: camera 20%, distractors 10%, multi 20%."""
    assert PERIOD == 10
    assert condition_share("standard") == pytest.approx(0.6)
    assert condition_share("camera") == pytest.approx(0.1)
    assert condition_share("distractors") == pytest.approx(0.1)
    assert condition_share("multi") == pytest.approx(0.1)
    assert condition_share("camera+multi") == pytest.approx(0.1)
    moving = sum(1 for c in CONDITION_CYCLE if "camera" in c)
    multi = sum(1 for c in CONDITION_CYCLE if "multi" in c)
    assert moving / PERIOD == pytest.approx(0.2), "camera marginal"
    assert multi / PERIOD == pytest.approx(0.2), "multi marginal"


def test_the_plain_clips_come_first():
    """So a short run is the easy case. A run that stops before the first
    non-standard index gets nothing but `standard`, which is what makes
    `--variants 1` a sane thing to ask for."""
    first_hard = min(v for v in range(PERIOD)
                     if condition_for(v) != "standard")
    assert all(condition_for(v) == "standard" for v in range(first_hard))
    assert first_hard >= 5, "too few plain clips before the first hard one"


def test_distractors_and_multi_are_never_combined():
    """They are the same placement machinery differing in whether the extras
    take part, so a scene with both asks the viewer to sort inert clutter from
    lawful peers from culprits -- three distinctions where the family makes
    one."""
    for v in range(PERIOD):
        assert not (has_distractors(v) and has_multi(v)), condition_for(v)


@pytest.mark.parametrize("name", NAMES)
def test_each_condition_builds_what_it_claims(name):
    """The scene must match its own label. `meta.json` reports the condition
    from the variant index, so a mismatch here would ship a clip claiming a
    condition the sampler did not build."""
    for v in range(PERIOD):
        spec = _spec(name, v)
        cond = condition_for(v)
        moving = spec.camera_motion_kind not in (None, "static")
        clutter = any(b.role == "distractor" for b in spec.bodies)
        peers = int(spec.notes.get("n_peers_placed") or 0)
        # `occluder_pass` opts out of camera motion entirely -- its occlusion
        # interval is computed from one pose at sample time, and a moving
        # camera would change which frames are hidden, so the list would
        # describe a different clip. It may decline; it may not move when the
        # condition says static.
        if name == "occluder_pass":
            assert not moving, (name, v, cond)
        else:
            assert moving == has_moving_camera(v), (name, v, cond, "camera")
        assert clutter == has_distractors(v), (name, v, cond, "distractors")
        # A scenario that is ALREADY a crowd needs no peers -- `pour` stages
        # forty grains and declares its own `group_fraction`. What `multi`
        # promises is several actors with a minority violating, not that this
        # particular helper placed them.
        crowd = len(_actors(spec)) >= MULTI_ACTORS
        if has_multi(v):
            assert crowd, (name, v, cond, "too few actors")
        else:
            assert peers == 0, (name, v, cond, "peers outside multi")


@pytest.mark.parametrize("name", NAMES)
def test_multi_gives_a_lawful_majority(name):
    """N actors, M of them violating, with M < N/2.

    The point of the condition. With one actor, "which object is wrong" has a
    trivial answer -- there is only one candidate -- so a model can score by
    detecting that SOMETHING is off. A lawful majority makes the clip ask
    *which*, and a scene where most things misbehave answers it before it is
    asked.
    """
    v = next(x for x in range(PERIOD) if condition_for(x) == "multi")
    spec = _spec(name, v)
    n = len(_actors(spec))
    assert n >= 4, "%s: %d actors is too few for _group to engage" % (name, n)
    assert n == spec.notes.get("n_actors"), name
    frac = spec.notes.get("group_fraction")
    assert frac is not None, name
    m = int(round(frac * n))
    if frac >= 1.0:
        # A scenario may declare the WHOLE medium on purpose: `pour`'s families
        # act on all forty grains, because one hovering grain is perfectly
        # annotated and impossible to see. That is a considered exception, not
        # this condition's default.
        assert m == n, name
        return
    assert m == MULTI_CULPRITS, "%s: %d culprits of %d" % (name, m, n)
    assert m * 2 < n, "%s: %d of %d is not a lawful majority" % (name, m, n)


@pytest.mark.parametrize("name", NAMES)
def test_a_multi_clip_names_more_than_one_culprit(name):
    """What the annotation actually ships. `_group` picks the bodies and the
    plan names them; if the two disagree the clip bends objects nobody
    labelled, which every mask and residual would then be wrong about."""
    import mockroll

    plain = next(x for x in range(PERIOD) if condition_for(x) == "standard")
    multi = next(x for x in range(PERIOD) if condition_for(x) == "multi")
    checked = 0
    for fam in ("support", "friction", "antigravity", "continuity"):
        inj = injectors.get(fam)
        for variant, want in ((plain, 1), (multi, MULTI_CULPRITS)):
            spec = _spec(name, variant)
            traj = mockroll.roll(spec, scenarios.get(name))
            plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
            if plan is None:
                continue
            got = len(plan.causal_body_ids)
            if variant == multi:
                assert got >= 2, (
                    "%s x %s: %d culprit(s) in a multi scene" % (name, fam, got))
                checked += 1
    assert checked, "%s: no family planned on a multi scene" % name


@pytest.mark.parametrize("name", NAMES)
def test_peers_are_actors_and_distractors_are_not(name):
    """The one line the two conditions differ on. A peer is an eligible
    culprit; a distractor is scenery every injector query excludes. Their
    segmentation blocks are distinct so a reader can tell them apart from ids
    alone."""
    from physloc.injectors import _geom
    from physloc.scenarios._common import SEG_DISTRACTOR_BASE, SEG_PEER_BASE

    multi = next(x for x in range(PERIOD) if condition_for(x) == "multi")
    clutter = next(x for x in range(PERIOD) if condition_for(x) == "distractors")

    spec = _spec(name, multi)
    peers = [b for b in spec.bodies if b.name.startswith("peer_")]
    if not peers:
        assert len(_actors(spec)) >= MULTI_ACTORS, name   # already a crowd
    eligible = {int(b.segmentation_id) for b in _geom.actors(spec)}
    for b in peers:
        assert b.role == "actor"
        assert int(b.segmentation_id) >= SEG_PEER_BASE
        assert int(b.segmentation_id) in eligible, (
            "%s: %s is a peer no injector can pick" % (name, b.name))

    spec = _spec(name, clutter)
    dis = [b for b in spec.bodies if b.role == "distractor"]
    assert dis, name
    eligible = {int(b.segmentation_id) for b in _geom.actors(spec)}
    for b in dis:
        assert SEG_DISTRACTOR_BASE <= int(b.segmentation_id) < SEG_PEER_BASE
        assert int(b.segmentation_id) not in eligible, (
            "%s: %s is scenery an injector can pick" % (name, b.name))

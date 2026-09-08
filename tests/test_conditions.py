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
from physloc.scenarios.base import (CONDITION_CYCLE, EXTRA_OBJECTS,
                                    MULTI_ACTORS,
                                    MULTI_CULPRIT_RANGE, condition_for,
                                    condition_share, has_distractors,
                                    has_moving_camera, has_multi)

NAMES = sorted(scenarios.available())
LEVEL = "L0"
PERIOD = len(CONDITION_CYCLE)
SEED = 777


def _spec(name, variant, level=LEVEL):
    """A spec at the FULL cycle length, so `variant` indexes the cycle directly.

    `condition_for` spreads a level's conditions over the cycle when the level was
    given fewer variants than the cycle has slots -- which is what stops L3
    being 100% `standard`. Passing `n_variants=PERIOD` here asks for the
    unspread form, so these tests keep addressing conditions by index.
    """
    return scenarios.get(name).sample(SEED + variant, TIERS["debug"], level,
                                      variant=variant, n_variants=PERIOD)


def _actors(spec):
    return [b for b in spec.bodies if b.role == "actor" and not b.dormant]


def test_the_cycle_is_the_agreed_shape():
    """Six plain clips, then one of each condition.

    Marginals: camera 20%, distractors 10%, multi 20%.
    """
    assert PERIOD == 10
    assert condition_share("standard") == pytest.approx(0.6)
    for c in ("camera", "distractors", "multi", "camera+multi"):
        assert condition_share(c) == pytest.approx(0.1), c
    moving = sum(1 for c in CONDITION_CYCLE if "camera" in c)
    multi = sum(1 for c in CONDITION_CYCLE if "multi" in c)
    assert moving / PERIOD == pytest.approx(0.2), "camera marginal"
    assert multi / PERIOD == pytest.approx(0.2), "multi marginal"
    assert condition_share("standard") > max(
        condition_share(c) for c in set(CONDITION_CYCLE) if c != "standard"), (
        "the baseline must stay the largest single stratum")


def test_the_plain_clips_come_first():
    """So a short run is the easy case. A run that stops before the first
    non-standard index gets nothing but `standard`, which is what makes
    `--variants 1` a sane thing to ask for."""
    first_hard = min(v for v in range(PERIOD)
                     if condition_for(v) != "standard")
    assert all(condition_for(v) == "standard" for v in range(first_hard))
    assert first_hard >= 4, "too few plain clips before the first hard one"


@pytest.mark.parametrize("name", NAMES)
def test_the_two_crowded_conditions_differ_only_in_culprit_count(name):
    """What actually separates `distractors` from `multi`.

    Both put N extra objects in the scene, both draw N from the same range, and
    both let some of them move. The ONE difference is how many bodies get
    invalid physics: `distractors` has exactly one culprit -- the scenario's
    own actor -- and its extras are scenery no family can target, while `multi`
    makes the extras eligible and 2..N-1 of them violate.

    That is the distinction worth drawing, because it is the one that changes
    the question the clip asks. "Is anything wrong here" and "WHICH of these is
    wrong" are different problems, and only `multi` poses the second.
    """
    lo, hi = EXTRA_OBJECTS
    d_v = next(v for v in range(PERIOD) if condition_for(v) == "distractors")
    m_v = next(v for v in range(PERIOD) if condition_for(v) == "multi")

    d = _spec(name, d_v)
    extras = [b for b in d.bodies if b.role == "distractor"]
    assert lo <= len(extras) <= hi, (name, len(extras))
    # A scenario may declare the WHOLE medium a culprit on purpose, whatever
    # the condition: `pour`'s families act on all forty grains, because one
    # hovering grain is perfectly annotated and impossible to see. The
    # condition's one-culprit rule is about the bodies THIS code adds, and it
    # adds none that can be targeted.
    if not d.notes.get("group_fraction"):
        assert not any(b.role == "actor" and b.name.startswith("peer_")
                       for b in d.bodies), (
            "%s: the distractors condition placed a targetable peer" % name)
    assert all(b.role == "distractor" for b in extras), name

    m = _spec(name, m_v)
    peers = [b for b in m.bodies if b.name.startswith("peer_")]
    n = len(_actors(m))
    assert peers or n >= lo, name          # already a crowd, e.g. `pour`
    assert m.notes.get("group_fraction"), name
    assert int(round(m.notes["group_fraction"] * n)) >= 2, (
        "%s: the multi condition must have at least two culprits" % name)


@pytest.mark.parametrize("name", NAMES)
def test_a_distractor_is_either_moving_or_genuinely_still(name):
    """Moving or inert, decided per body -- not a speed that happens to be
    small. Uniformly drifting clutter is as learnable as uniformly still
    clutter; a mixture is neither, and a continuum of near-zero speeds is
    really the still case wearing noise."""
    import numpy as np

    from physloc.scenarios._common import DISTRACTOR_SPEED

    v = next(x for x in range(PERIOD) if condition_for(x) == "distractors")
    moving = still = 0
    for k in range(6):
        spec = _spec(name, k * PERIOD + v)
        for b in spec.bodies:
            if b.role != "distractor":
                continue
            speed = float(np.linalg.norm(b.velocity))
            if speed > 1e-9:
                moving += 1
                assert speed >= DISTRACTOR_SPEED[0] * 0.6 - 1e-9, (
                    "%s: %s creeps at %.4f -- moving should mean moving"
                    % (name, b.name, speed))
            else:
                still += 1
                assert not any(abs(x) > 1e-9 for x in b.angular_velocity), (
                    "%s: %s is inert but spinning" % (name, b.name))
    assert moving and still, (
        "%s: distractors were all %s over six clips"
        % (name, "moving" if moving else "inert"))


def test_distractors_and_multi_are_never_combined():
    """They are the same placement machinery differing in whether the extras
    take part, so a scene with both asks the viewer to sort inert clutter from
    lawful peers from culprits -- three distinctions where the family makes
    one."""
    for v in range(PERIOD):
        assert not (has_distractors(condition_for(v)) and has_multi(condition_for(v))), condition_for(v)


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
            assert moving == has_moving_camera(condition_for(v)), (name, v, cond, "camera")
        assert clutter == has_distractors(condition_for(v)), (name, v, cond, "distractors")
        # A scenario that is ALREADY a crowd needs no peers -- `pour` stages
        # forty grains and declares its own `group_fraction`. What `multi`
        # promises is several actors with a minority violating, not that this
        # particular helper placed them.
        crowd = len(_actors(spec)) >= MULTI_ACTORS[0]
        if has_multi(condition_for(v)):
            assert crowd, (name, v, cond, "too few actors")
        else:
            assert peers == 0, (name, v, cond, "peers outside multi")


@pytest.mark.parametrize("name", NAMES)
def test_multi_draws_both_of_its_counts(name):
    """N actors and M culprits, both randomised, both inside their bounds.

    The point of the condition, and of randomising it. A fixed count is a cue:
    a model that learns "five objects, two wrong" is reading the layout rather
    than the physics. At least two culprits, because one is what `standard`
    already is, and at most N-1 so there is always a lawful body to contrast
    against -- which is the whole task.
    """
    lo_n, hi_n = MULTI_ACTORS
    seen_n, seen_m = set(), set()
    for k in range(6):
        for v in (x for x in range(PERIOD) if has_multi(condition_for(x))):
            spec = _spec(name, k * PERIOD + v)
            n = len(_actors(spec))
            frac = spec.notes.get("group_fraction")
            assert n == spec.notes.get("n_actors"), name
            assert frac is not None, (name, n)
            m = int(round(frac * n))
            if frac >= 1.0:
                # A scenario may declare the WHOLE medium on purpose: `pour`'s
                # families act on all forty grains, because one hovering grain
                # is perfectly annotated and impossible to see. A considered
                # exception, not this condition's default.
                assert m == n, name
                continue
            assert lo_n <= n <= hi_n, "%s: %d actors outside %s" % (
                name, n, MULTI_ACTORS)
            assert m >= MULTI_CULPRIT_RANGE[0], (
                "%s: %d culprit(s) is what `standard` already is" % (name, m))
            assert m <= n - MULTI_CULPRIT_RANGE[1], (
                "%s: %d of %d leaves nothing lawful to compare against"
                % (name, m, n))
            seen_n.add(n)
            seen_m.add(m)
    if seen_n:
        assert len(seen_n) >= 3, "%s: object count barely varies: %s" % (
            name, sorted(seen_n))
        assert len(seen_m) >= 2, "%s: culprit count barely varies: %s" % (
            name, sorted(seen_m))


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
        for variant in (plain, multi):
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
        assert len(_actors(spec)) >= MULTI_ACTORS[0], name  # already a crowd
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

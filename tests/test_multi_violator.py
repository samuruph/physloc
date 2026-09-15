"""Each violator of a `multi` clip on its own clock.

Every violator in a `multi` clip used to fire on the same frame, from one plan,
with one severity timeline painted into all of them. Most eligible clips now
plan each violator separately and merge the plans; a quarter keep one shared
moment on purpose, so simultaneity is represented rather than accidental.
"""
from __future__ import annotations

import numpy as np
import pytest

import mockroll
from physloc import injectors, scenarios
from physloc.injectors import multi
from physloc.injectors.base import InterventionPlan
from physloc.scenarios import TIERS
from physloc.sim.trajectory import prefix_identical

TIER = TIERS["debug"]
#: The variant slot `CONDITION_CYCLE` gives `multi` in a ten-variant level.
MULTI_VARIANT = 8
FAMILIES = ("phantom_impulse", "antigravity", "continuity", "permanence")


def _multi_spec(seed, name="drop"):
    sc = scenarios.get(name)
    spec = sc.sample(seed, TIER, "L0", variant=MULTI_VARIANT, n_variants=10)
    assert "multi" in spec.condition
    return sc, spec


def _plans(inj, spec, traj, seed=3):
    return multi.violator_plans(inj, spec, traj,
                               lambda: np.random.RandomState(seed), "strong",
                               inj.simulates)


def test_merge_keeps_each_violator_and_unions_the_windows():
    a = InterventionPlan(family="phantom_impulse", kind="instant", t_event=5,
                         windows=[(5, 6)], causal_body_ids=[2], params={},
                         magnitude=1.0, magnitude_unit="m/s", severity_bin="strong")
    b = InterventionPlan(family="phantom_impulse", kind="instant", t_event=11,
                         windows=[(11, 12)], causal_body_ids=[600], params={},
                         magnitude=0.8, magnitude_unit="m/s", severity_bin="strong")
    m = InterventionPlan.merge([a, b], [2, 600])
    assert m.t_event == 5
    assert m.causal_body_ids == [2, 600]
    assert [c.body_id for c in m.violators] == [2, 600]
    assert [c.t_event for c in m.violators] == [5, 11]
    assert sorted(map(tuple, m.windows)) == [(5, 6), (11, 12)]
    d = m.to_dict()
    assert [c["t_event_frame"] for c in d["violators"]] == [5, 11]
    assert d["violators"][1]["magnitude"] == 0.8


def test_sync_share_is_about_a_quarter():
    _, spec = _multi_spec(0)
    modes = []
    for seed in range(800):
        spec.seed = seed
        modes.append(multi.timing_mode(spec, "phantom_impulse"))
    share = modes.count("sync") / float(len(modes))
    assert abs(share - multi.MULTI_SYNC_SHARE) < 0.05


def test_non_multi_clips_are_never_split():
    sc = scenarios.get("drop")
    spec = sc.sample(4, TIER, "L0", variant=0, n_variants=10)
    traj = mockroll.roll(spec, sc)
    plan, subs = _plans(injectors.get("phantom_impulse"), spec, traj)
    assert subs == []
    assert plan is None or plan.notes.get("violator_timing") == "shared"


@pytest.mark.parametrize("family", FAMILIES)
def test_independent_violators_get_their_own_moments(family):
    inj = injectors.get(family)
    split, distinct = 0, 0
    for seed in range(12):
        sc, spec = _multi_spec(seed)
        before = dict(spec.notes.get("family_targets") or {})
        traj = mockroll.roll(spec, sc)
        plan, subs = _plans(inj, spec, traj)
        # The per-violator override never leaks into the scene.
        assert dict(spec.notes.get("family_targets") or {}) == before
        if plan is None or not subs:
            continue
        split += 1
        assert plan.notes["violator_timing"] == "independent"
        assert [c.body_id for c in plan.violators] == [int(s.causal_body_ids[0])
                                                      for s in subs]
        assert plan.t_event == min(s.t_event for s in subs)
        distinct += int(len({s.t_event for s in subs}) > 1)
        for s in subs:
            assert 1 <= s.t_event < traj.num_frames - 1
    if split:
        # Independent draws do not all land on one frame.
        assert distinct >= max(1, split // 2)


@pytest.mark.parametrize("family", ["phantom_impulse", "continuity"])
def test_no_violator_changes_anything_before_its_own_moment(family):
    """The host approximation of what the worker stages, applied in order of
    moment. The clip is the valid rollout until the EARLIEST moment, and adding
    each later violator leaves every frame before its own moment as it was.

    Not "each violator is lawful until its own moment": an earlier violator may
    genuinely disturb a later one before that one fires, and that disturbance
    is a consequence of the first violation, not a leak of the second.
    """
    inj = injectors.get(family)
    checked = 0
    for seed in range(12):
        sc, spec = _multi_spec(seed)
        traj = mockroll.roll(spec, sc)
        plan, subs = _plans(inj, spec, traj)
        if not subs:
            continue
        out = traj
        for sub in multi.by_moment(subs):
            before = out
            out = inj.apply(spec, before, sub)
            ok, why = prefix_identical(before, out, sub.t_event)
            assert ok, (family, seed, sub.t_event, why)
        ok, why = prefix_identical(traj, out, plan.t_event)
        assert ok, why
        checked += 1
    if not checked:
        pytest.skip("no independent multi clip in these seeds")

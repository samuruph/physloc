"""Cells the worker used to decline on every attempt, kept renderable.

Each was "culprit leaves the frame after t_event at every one of 4 event
moments": the keep-in-shot fits counted frames off screen against a budget the
worker's gate does not use, could not weaken far enough, or did not move at all
between attempts, and two staged families threw a medium out of its box.
"""
from __future__ import annotations

import numpy as np
import pytest

import mockroll
from physloc import injectors, scenarios
from physloc.injectors import _geom
from physloc.scenarios import TIERS


def _roll(scenario, seed, tier=None):
    sc = scenarios.get(scenario)
    spec = sc.sample(seed, tier or TIERS["debug"], "L0")
    return spec, mockroll.roll(spec, sc)


def _plan(inj, spec, traj, severity="strong", attempt=0):
    inj.event_attempt = attempt
    try:
        return inj.plan(spec, traj, np.random.RandomState(1), severity)
    finally:
        inj.event_attempt = 0


def _culprits(spec, plan):
    by_id = {int(b.segmentation_id): b for b in spec.bodies}
    return [by_id[i] for i in plan.causal_body_ids
            if i in by_id and not by_id[i].static]


@pytest.mark.parametrize("seed", [777, 778, 779])
def test_superelastic_drop_bounce_stays_in_shot(seed):
    """The shared ladder bottomed out at a 2.4x bounce, which put the ball
    10-13 m up on every seed; the deeper one finds a gain the frame holds."""
    spec, traj = _roll("drop", seed)
    inj = injectors.get("superelastic")
    plan = _plan(inj, spec, traj)
    assert plan is not None
    out = inj.apply(spec, traj, plan)
    assert _geom.culprits_visible(spec, out, _culprits(spec, plan), plan.t_event)
    assert plan.params["speed_gain"] > 1.0


def test_superelastic_bins_stay_ordered_after_a_deep_fit():
    spec, traj = _roll("drop", 777)
    inj = injectors.get("superelastic")
    gains = [_plan(inj, spec, traj, sev).params["speed_gain"]
             for sev in ("weak", "medium", "strong")]
    assert gains[0] < gains[1] < gains[2]


def test_a_retry_fits_no_stronger_than_the_first_attempt():
    """Attempts used to plan the identical variant four times over."""
    spec, traj = _roll("drop", 778)
    inj = injectors.get("superelastic")
    first = _plan(inj, spec, traj, attempt=0).params["frame_fit_scale"]
    retry = _plan(inj, spec, traj, attempt=1).params["frame_fit_scale"]
    assert retry <= first


def test_a_retry_backs_off_from_the_fitted_rung():
    """The preview kept the strongest rung in shot and PyBullet did not; a
    retry that re-checked the same preview planned the same variant again."""
    from test_group_rewrite import _Probe
    from physloc.injectors.base import Injector

    ladder = Injector.FIT_LADDER
    for attempt in (0, 1, 2):
        probe = _Probe(1.0)                   # every rung fits the preview
        with _geom.event_context("phantom_impulse", attempt):
            got, _ = probe._fit_to_frame(None, None, [], 0, 1.0, lambda k: k,
                                         tolerance=0)
        want = ladder[min(len(ladder) - 1,
                          Injector.FIT_RETRY_RUNGS * attempt)]
        assert got == want


def test_a_retry_shortens_the_fitted_window():
    from test_group_rewrite import _Probe
    from physloc.injectors.base import Injector

    for attempt, floor in ((0, 3), (1, 3), (2, 3), (6, 3)):
        probe = _Probe(30)                    # the whole window fits
        with _geom.event_context("antigravity", attempt):
            got = probe._fit_window_to_frame(None, None, [], 0, 30,
                                             lambda n: n, tolerance=0,
                                             floor=floor)
        want = max(floor, int(round(30 * Injector.FIT_RETRY_WINDOW ** attempt)))
        assert got == want


def test_support_hover_is_fitted_into_the_frame():
    """Seed 778 hung the ball at z = 3.7, above the shot, on every attempt."""
    spec, traj = _roll("drop", 778)
    inj = injectors.get("support")
    plan = _plan(inj, spec, traj)
    out = inj.apply(spec, traj, plan)
    assert _geom.culprits_visible(spec, out, _culprits(spec, plan), plan.t_event)
    assert 0.0 < plan.params["clearance_radii"] <= inj.CLEARANCE_RADII["strong"]


def test_support_hangs_a_medium_still():
    spec, traj = _roll("pour", 777)
    plan = _plan(injectors.get("support"), spec, traj)
    assert plan.notes["mode"] == "hover_still"


def test_one_body_of_a_pair_on_screen_counts_as_seen():
    """A colliding pair is one event: the middle block rebounding in view is
    evidence even while the top block tumbles out of the side."""
    spec, traj = _roll("stack_topple", 777)
    pair = [b for b in spec.bodies if not b.static and not b.dormant][:2]
    assert len(pair) == 2
    out = traj.copy() if hasattr(traj, "copy") else injectors.get(
        "superelastic")._clone(traj)
    j = out.index_of(int(pair[0].segmentation_id))
    out.pos[:, j, :] = np.asarray([500.0, 500.0, 500.0], np.float32)
    assert _geom.culprits_on_screen(spec, out, pair).all()
    # ...but not when neither is.
    k = out.index_of(int(pair[1].segmentation_id))
    out.pos[:, k, :] = np.asarray([-500.0, 500.0, 500.0], np.float32)
    assert not _geom.culprits_on_screen(spec, out, pair).any()
    # A medium still needs most of itself in shot.
    trio = [b for b in spec.bodies if not b.static and not b.dormant][:3]
    if len(trio) == 3:
        assert not _geom.culprits_on_screen(spec, out, trio).any()


def test_a_granular_medium_needs_less_of_itself_in_shot():
    """Some grains of a pour leaving the frame is not a lost violation."""
    spec, traj = _roll("pour", 777)
    rigid, _ = _roll("drop", 777)
    assert _geom.visible_share(spec) == _geom.MEDIUM_VISIBLE_SHARE
    assert _geom.visible_share(rigid) == _geom.VISIBLE_SHARE
    grains = [b for b in spec.bodies if not b.static and not b.dormant]
    out = injectors.get("superelastic")._clone(traj)
    gone = int(len(grains) * 0.5)                 # half the pour leaves
    for b in grains[:gone]:
        out.pos[:, out.index_of(int(b.segmentation_id)), :] = 500.0
    assert _geom.culprits_on_screen(spec, out, grains).all()


def test_a_peer_outside_the_shot_is_not_a_split_culprit():
    """One peer spawned out of frame declined whole `multi` clips (drop 786,
    stack_topple 785 x solidity); it must simply not be named a culprit."""
    from physloc.injectors import multi
    from physloc.scenarios.base import CONDITION_CYCLE

    variant = next(i for i, c in enumerate(CONDITION_CYCLE) if c == "multi")
    sc = scenarios.get("drop")
    inj = injectors.get("phantom_impulse")
    for seed in range(40):
        spec = sc.sample(seed, TIERS["debug"], "L0", variant=variant,
                         n_variants=len(CONDITION_CYCLE))
        if multi.timing_mode(spec, inj.family) != "independent":
            continue
        traj = mockroll.roll(spec, sc)
        base, subs = multi.culprit_plans(
            inj, spec, traj, lambda: np.random.RandomState(1), "strong",
            lambda p: inj.simulates(p))
        if len(subs) < 3:
            continue
        # Park one of the culprits far outside the shot for the whole clip.
        hidden = int(subs[-1].causal_body_ids[0])
        out = inj._clone(traj)
        out.pos[:, out.index_of(hidden), :] = np.float32(500.0)
        plan, subs2 = multi.culprit_plans(
            inj, spec, out, lambda: np.random.RandomState(1), "strong",
            lambda p: inj.simulates(p))
        assert hidden not in {int(s.causal_body_ids[0]) for s in subs2}
        assert len(subs2) == len(subs) - 1
        return
    pytest.skip("no seed gave an independently timed multi plan of 3+ culprits")


def test_continuity_retries_turn_the_jump():
    """The same outward jump on every attempt left the frame four times over
    (stack_topple L3 777); a retry must jump somewhere else."""
    spec, traj = _roll("drop", 777)
    inj = injectors.get("continuity")
    unit = []
    for attempt in range(4):
        per_bin = []
        for sev in ("weak", "strong"):
            d = np.asarray(_plan(inj, spec, traj, sev, attempt).params["delta_m"])
            per_bin.append(d / max(np.linalg.norm(d), 1e-9))
        # Bins of one attempt share a heading...
        assert np.allclose(per_bin[0], per_bin[1], atol=1e-6)
        unit.append(per_bin[1])
    # ...attempt 1 reverses attempt 0, and 2 and 3 are either side of it.
    assert float(unit[0] @ unit[1]) < -0.99
    assert abs(float(unit[0] @ unit[2])) < 0.05
    assert abs(float(unit[0] @ unit[3])) < 0.05


def test_non_parabolic_kicks_leave_every_body_at_rest():
    """The interior-only difference left each body drifting sideways."""
    inj = injectors.get("non_parabolic")
    spec, _ = _roll("drop", 777)
    for n in (5, 9, 17):
        from types import SimpleNamespace
        plan = SimpleNamespace(params={"amplitude_m": 0.4, "cycles": inj.CYCLES})
        for k in range(3):
            path = inj._offsets(spec, plan, n, k, 3)
            dt = 1.0 / 30.0
            accel = inj._stage_accel(path, dt)
            assert accel.shape == (n + 2, 3)
            # Velocity after the last kick, and the displacement it leaves.
            v = np.cumsum(accel * dt, axis=0)
            assert np.allclose(v[-1], 0.0, atol=1e-9)
            x = np.cumsum(v * dt, axis=0)
            # The path from the event's own frame, then back where it began.
            assert np.allclose(x[:n], path, atol=1e-9)
            assert np.allclose(x[n:], 0.0, atol=1e-9)


def test_non_parabolic_medium_path_is_speed_capped():
    """~27 m/s sideways on `pour` at release detonated the pile in PyBullet."""
    tier = TIERS["release"].override(num_frames=25)
    spec, traj = _roll("pour", 20260826, tier)
    inj = injectors.get("non_parabolic")
    plan = _plan(inj, spec, traj)
    if plan is None:
        pytest.skip("no airborne run on this sample")
    t0, t1 = plan.windows[0]
    peak = inj._peak_speed(spec, plan.params["amplitude_m"], t1 - t0 + 1,
                           1.0 / traj.dt)
    assert peak <= inj.MEDIUM_SPEED_CAP + 1e-6

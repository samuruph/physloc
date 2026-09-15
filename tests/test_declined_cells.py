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

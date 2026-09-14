"""Stepping a whole medium, fast enough to finish, and the fit searches over it.

A `pour` job at release geometry never got past its variants: `_rewrite_group`
visited every pair of 212 grains on every substep in Python -- 175 s a rollout
-- and the frustum fits re-ran it up to thirty times for each of three bins.
The broad phase and the bisected, remembered fits must change how long that
takes and nothing about what it produces.
"""
from __future__ import annotations

import numpy as np
import pytest

import mockroll
from physloc import injectors, scenarios
from physloc.injectors import _geom
from physloc.injectors.base import Injector
from physloc.scenarios import TIERS


def _reference_rewrite_group(inj, spec, traj, out, bodies, t0, g_by_body=None,
                             v0_by_body=None, substeps=24):
    """The all-pairs loop the broad phase replaced, verbatim."""
    n = traj.num_frames - t0
    ids = [int(b.segmentation_id) for b in bodies]
    obstacles = _geom.Obstacles(spec, traj, exclude_ids=ids)
    idx = [traj.index_of(i) for i in ids]
    rad = [float(traj.radius[j]) for j in idx]
    rest = [float(b.restitution) for b in bodies]
    ground = [_geom.floor_fn(spec, b) for b in bodies]
    g_def = np.tile(traj.gravity.astype(np.float64)[None, :], (n, 1))
    g_seq = [np.asarray((g_by_body or {}).get(i, g_def), np.float64) for i in ids]
    p = [traj.pos[t0 - 1, j].astype(np.float64).copy() for j in idx]
    v = [np.asarray((v0_by_body or {}).get(i, traj.lin_vel[t0 - 1, j]),
                    np.float64).copy() for i, j in zip(ids, idx)]
    h = traj.dt / float(substeps)
    pos = np.zeros((len(idx), n, 3))
    for f in range(n):
        for k in range(substeps):
            t = float(t0 - 1) + f + (k + 1) / float(substeps)
            for a in range(len(idx)):
                v[a] = v[a] + g_seq[a][f] * h
                p[a] = p[a] + v[a] * h
                fz = ground[a](p[a][0], p[a][1])
                if p[a][2] - rad[a] <= fz and v[a][2] < 0.0:
                    p[a][2] = fz + rad[a]
                    v[a][2] = -v[a][2] * rest[a]
                    if abs(v[a][2]) < 0.05:
                        v[a][2] = 0.0
                p[a], v[a] = obstacles.resolve(p[a], v[a], rad[a], rest[a], t)
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    d = p[b] - p[a]
                    dist = float(np.linalg.norm(d))
                    reach = rad[a] + rad[b]
                    if dist >= reach or dist < 1e-9:
                        continue
                    nrm = d / dist
                    push = 0.5 * (reach - dist)
                    p[a] -= nrm * push
                    p[b] += nrm * push
                    rel = float(np.dot(v[b] - v[a], nrm))
                    if rel < 0.0:
                        e = 0.5 * (rest[a] + rest[b])
                        imp = -(1.0 + e) * rel * 0.5
                        v[a] -= nrm * imp
                        v[b] += nrm * imp
        for a in range(len(idx)):
            pos[a, f] = p[a]
    for a, j in enumerate(idx):
        out.pos[t0:, j, :] = pos[a].astype(np.float32)


@pytest.fixture(scope="module")
def pour():
    sc = scenarios.get("pour")
    tier = TIERS["debug"].override(num_frames=13)
    spec = sc.sample(20260826, tier, "L0")
    return spec, mockroll.roll(spec, sc)


@pytest.mark.parametrize("push", [0.0, 3.0])
def test_broad_phase_matches_the_all_pairs_loop(pour, push):
    """Same collisions, same order, on a real pile -- with and without a shove
    that throws the grains into each other."""
    spec, traj = pour
    inj = injectors.get("phantom_impulse")
    bodies = [b for b in spec.bodies if b.role == "actor" and not b.dormant]
    t0 = 3
    v0 = {int(b.segmentation_id):
          traj.lin_vel[t0 - 1, traj.index_of(int(b.segmentation_id))]
          .astype(np.float64) + np.array([push, 0.0, push])
          for b in bodies}
    fast, ref = inj._clone(traj), inj._clone(traj)
    inj._rewrite_group(spec, traj, fast, bodies, t0, v0_by_body=v0)
    _reference_rewrite_group(inj, spec, traj, ref, bodies, t0, v0_by_body=v0)
    assert np.allclose(fast.pos, ref.pos, atol=1e-5), float(
        np.abs(fast.pos - ref.pos).max())


class _Probe(Injector):
    """A fit target whose on-screen count is a known step function of the knob."""

    family = "phantom_impulse"

    def __init__(self, fits_from):
        self.fits_from = fits_from
        self.calls = []

    def _offscreen_frames(self, spec, traj, bodies, from_frame, **_kw):
        # The budget is measured on the lawful trajectory, which is not a knob
        # value; it counts as fully in shot.
        if not isinstance(traj, (int, float)):
            return 0
        return 0 if traj <= self.fits_from else 5


def _scale_search(probe, ladder):
    def build(k):
        probe.calls.append(k)
        return k
    return probe._fit_to_frame(None, None, [], 0, 1.0, build, tolerance=0,
                               ladder=ladder)


@pytest.mark.parametrize("fits_from", [1.0, 0.8, 0.41, 0.2, 0.0])
def test_scale_bisection_finds_the_same_rung_as_a_linear_walk(fits_from):
    ladder = Injector.FIT_LADDER
    want = next((s for s in ladder if s <= fits_from), ladder[-1])
    probe = _Probe(fits_from)
    got, _ = _scale_search(probe, ladder)
    assert got == want
    # The strongest rung, the weakest, then a bisection of the rest: at most
    # six rollouts where the linear walk took up to ten.
    assert len(probe.calls) <= 6


@pytest.mark.parametrize("fits_up_to", [30, 17, 3, 1])
def test_window_bisection_finds_the_longest_window_that_fits(fits_up_to):
    probe = _Probe(fits_up_to)
    calls = []

    def build(n):
        calls.append(n)
        return n

    got = probe._fit_window_to_frame(None, None, [], 0, 30, build, tolerance=0,
                                     floor=3)
    assert got == max(3, min(30, fits_up_to))
    assert len(calls) <= 7


def test_the_three_bins_share_one_fit(pour):
    """A remembered fit is not recomputed for the next bin of the same cell."""
    spec, traj = pour
    probe = _Probe(0.63)
    calls = []

    def build(k):
        calls.append(k)
        return k

    for _bin in ("weak", "medium", "strong"):
        probe._fit_to_frame(spec, traj, [], 0, 1.0, build, tolerance=0,
                            memo=("probe",))
    first = len(calls)
    assert first > 0
    probe._fit_to_frame(spec, traj, [], 0, 1.0, build, tolerance=0,
                        memo=("probe",))
    assert len(calls) == first
    # A different scene is a different fit.
    other = scenarios.get("pour").sample(1, spec.tier, "L0")
    probe._fit_to_frame(other, traj, [], 0, 1.0, build, tolerance=0,
                        memo=("probe",))
    assert len(calls) > first

"""Every culprit stays in shot, at every tier.

The regression this exists for: when the tiers were lengthened (13 -> 25 frames
at the debug tier, 25 -> 49 at the release tier) every camera in the project stayed where it had
been hand-tuned for the short clips, while the distance a body covers grew with
the clip. `toss` and `tumble` were the worst -- a ballistic arc's apex goes as
`g*T^2/8`, so quadrupling -- and ended up with 1% and 0% of their frames still
in view at the release tier. `collision`, `barrier_pass`, `occluder_pass`, `rolling_ramp`
and `shadow_track` all leaked out sideways for the linear version of the same
reason.

None of it was caught by anything. Every cell still planned, still applied,
still validated and still rendered; the violations were simply injected
off-screen, which is the one failure mode a green test suite is least able to
notice. So the framing gets a test of its own, and it runs at all three tiers
rather than the one we happen to be debugging at.
"""
from __future__ import annotations

import numpy as np
import pytest

import physloc.scenarios  # noqa: F401  -- registers every scenario
from physloc.injectors import _geom
from physloc.scenarios import base as B

import mockroll

#: Where violation windows live: `_geom.EVENT_FRACTION` through the end.
BAND = (0.30, 0.95)
#: Culprits may clip the edge briefly -- a pour column starts above frame by
#: design -- but a body that spends a tenth of the window out of shot is a clip
#: whose violation nobody can see.
MIN_VISIBLE = 0.90
SEEDS = 5


def _visible_fraction(spec, traj, body, lo, hi) -> float:
    bi = traj.index_of(int(body.segmentation_id))
    r = float(body.bounding_radius)
    p = np.asarray(traj.pos[:, bi, :], np.float64)
    ok = np.ones(traj.num_frames, bool)
    for d in list(np.eye(3) * r) + list(-np.eye(3) * r):
        ok &= _geom.in_frame(spec, p + d[None], margin=0.0)
    return float(ok[lo:hi].mean())


@pytest.mark.parametrize("tier_name", ["debug", "release"])
@pytest.mark.parametrize("scenario", sorted(B.available()))
def test_culprits_stay_in_frame(scenario, tier_name):
    sc, tier = B.get(scenario), B.TIERS[tier_name]
    for seed in range(SEEDS):
        spec = sc.sample(seed, tier, "L0")
        traj = mockroll.roll(spec, sc)
        lo = int(BAND[0] * traj.num_frames)
        hi = int(BAND[1] * traj.num_frames)
        for body in spec.bodies:
            if body.dormant or body.static or body.role not in ("actor", "shadow"):
                continue
            frac = _visible_fraction(spec, traj, body, lo, hi)
            assert frac >= MIN_VISIBLE, (
                "%s/%s seed %d: %s is in frame for only %.0f%% of the window "
                "band" % (scenario, tier_name, seed, body.name, 100 * frac))


@pytest.mark.parametrize("scenario", ["toss"])
def test_free_flight_is_scale_invariant(scenario):
    """The same seed frames identically at every tier.

    `camera.frame_flight` derives the whole shot from the clip length, so the
    actor should cover the same fraction of the frame whatever the tier. If it
    does not, a bug found at the debug tier is not the same bug at the release tier -- which is
    the entire premise of debugging at the debug tier.
    """
    sc = B.get(scenario)
    sizes = []
    for tier_name in ("debug", "release"):
        spec = sc.sample(3, B.TIERS[tier_name], "L0")
        body = next(b for b in spec.bodies if b.role == "actor" and not b.dormant)
        eye = np.asarray(spec.camera_position, np.float64)
        aim = np.asarray(spec.camera_look_at, np.float64)
        dist = float(np.linalg.norm(aim - eye))
        sizes.append(body.bounding_radius / dist / _geom.TAN_HALF_FOV)
    assert max(sizes) - min(sizes) < 0.01, (
        "%s actor covers %s of the half-frame across tiers D/A/B"
        % (scenario, ["%.3f" % s for s in sizes]))

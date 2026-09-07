"""The single most valuable test in the repo -- docs/PLAN.md Part 4.

Valid and invalid renders must be bit-identical for every frame before
`t_event`. If this fails, the seed plumbing or the render path has drifted and
every downstream annotation is suspect.
"""
import json
import os

import numpy as np
import pytest

from conftest import find_workdir
from physloc.sim.trajectory import Trajectory, prefix_identical

# forward_flow at frame t encodes motion t -> t+1, so its last legal identical
# frame is t_event-2: at t_event-1 it already points into the diverged future.
# Everything else must match right up to t_event-1.
LOOKAHEAD = {"forward_flow": 1}


@pytest.fixture(scope="module")
def work():
    d = find_workdir()
    if d is None:
        pytest.skip("no worker output; run `python -m physloc.cli generate --debug`")
    return d


def test_trajectory_prefix_is_exactly_identical(work):
    with open(os.path.join(work, "plan.json")) as fh:
        te = json.load(fh)["plan"]["t_event_frame"]
    a = Trajectory.load(os.path.join(work, "traj_valid.npz"))
    b = Trajectory.load(os.path.join(work, "traj_invalid.npz"))
    ok, why = prefix_identical(a, b, te, atol=0.0)
    assert ok, why


def test_render_prefix_is_exactly_identical(work):
    with open(os.path.join(work, "plan.json")) as fh:
        te = json.load(fh)["plan"]["t_event_frame"]
    v = np.load(os.path.join(work, "passes_valid.npz"))
    i = np.load(os.path.join(work, "passes_invalid.npz"))
    for key in v.files:
        upto = te - LOOKAHEAD.get(key, 0)
        if upto <= 0:
            continue
        a = v[key][:upto].astype(np.float64)
        b = i[key][:upto].astype(np.float64)
        d = float(np.abs(a - b).max()) if a.size else 0.0
        assert d == 0.0, "%s differs by %.3e before frame %d" % (key, d, upto)


def test_the_twins_actually_diverge_after_t_event(work):
    """Guards the opposite failure: an injector that silently did nothing would
    pass every identity check above."""
    with open(os.path.join(work, "plan.json")) as fh:
        te = json.load(fh)["plan"]["t_event_frame"]
    v = np.load(os.path.join(work, "passes_valid.npz"))["rgba"].astype(np.float64)
    i = np.load(os.path.join(work, "passes_invalid.npz"))["rgba"].astype(np.float64)
    assert float(np.abs(v[te:] - i[te:]).max()) > 0.0


def test_forward_flow_lookahead_is_exactly_one_frame(work):
    """Documents *why* forward_flow gets an exemption, so nobody widens it."""
    with open(os.path.join(work, "plan.json")) as fh:
        te = json.load(fh)["plan"]["t_event_frame"]
    v = np.load(os.path.join(work, "passes_valid.npz"))["forward_flow"]
    i = np.load(os.path.join(work, "passes_invalid.npz"))["forward_flow"]
    per = [float(np.abs(v[f].astype(np.float64) - i[f].astype(np.float64)).max())
           for f in range(v.shape[0])]
    assert all(p == 0.0 for p in per[:te - 1]), per[:te]
    assert per[te - 1] > 0.0, "forward_flow should already differ at t_event-1"


def test_segmentation_ids_match_what_the_scenario_declared(work):
    """Kubric numbers the raw segmentation by scene-asset order and honours the
    declared `segmentation_id` only after `adjust_segmentation_idxs`. Skipping
    that post-process silently relabels instances whenever declaration order
    differs from insertion order, which corrupts every mask and residual
    downstream while still looking plausible."""
    import json as _json
    with open(os.path.join(work, "plan.json")) as fh:
        spec = _json.load(fh)["spec"]
    declared = {int(b["segmentation_id"]) for b in spec["bodies"]}
    for tag in ("valid", "invalid"):
        seg = np.load(os.path.join(work, "passes_%s.npz" % tag))["segmentation"]
        seg = seg[..., 0] if seg.ndim == 4 else seg
        got = {int(v) for v in np.unique(seg) if v != 0}
        assert got <= declared, "rendered ids %s not within declared %s" % (
            sorted(got), sorted(declared))

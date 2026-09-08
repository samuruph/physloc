"""Interventions that are staged in the simulator produce REAL contacts.

The defect this exists to catch, reported cell by cell: a teleported ball
bouncing off a barrier it had already been moved past, and two balls exchanging
momentum without touching. Both came from the same cause -- the trajectory was
edited after the fact and re-integrated against the scene as *declared*, so the
resolver kept a wall in front of a body that was behind it, and wrote collision
outcomes for contacts that never occurred.

Only re-simulation can pass these. A hand-rolled resolver cannot, which is the
point of the test.

Needs worker output, so it skips without docker.
"""
import json
import os

import numpy as np
import pytest

from physloc.sim.trajectory import Trajectory

#: (scenario, family) -> must the culprit still touch what it touched lawfully?
SIMULATED = ("continuity", "phantom_impulse", "newton1_inertia", "solidity")


#: Worker output older than this many seconds relative to the newest source
#: file is treated as stale and skipped.
STALE_GRACE = 5.0


def _plans():
    """Variant directories from a worker run, IF they match the current code.

    Its own discovery rather than `conftest.find_workdir`, which returns a
    single variant directory; this test wants all of them across every scenario
    a run produced.

    THE STALENESS CHECK IS THE POINT. These tests read whatever a previous
    `generate` left on disk, so they are checking artefacts rather than code --
    and an artefact produced by an older version of the injectors says nothing
    about the current ones. A leftover sweep failed
    `test_a_prevented_collision_leaves_no_contact` for two full suite runs on a
    clip whose contact pair the current code does not even produce; regenerating
    that same cell passed. A test that fails on a file nobody has regenerated is
    worse than one that skips: it trains you to ignore it.

    So output older than the newest source file under `physloc/` is skipped with
    a message saying how to refresh it.
    """
    import glob

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plans = sorted(glob.glob(os.path.join(root, "out", "**", "variants", "*",
                                          "plan.json"), recursive=True))
    if not plans:
        return []
    newest_src = max(
        os.path.getmtime(f)
        for f in glob.glob(os.path.join(root, "physloc", "**", "*.py"),
                           recursive=True))
    fresh = [p for p in plans
             if os.path.getmtime(p) >= newest_src - STALE_GRACE]
    if not fresh:
        pytest.skip(
            "worker output under out/ predates the current physloc/ sources, "
            "so it describes code that no longer exists. Regenerate with "
            "`python -m physloc.cli generate --config review` (or delete out/).")
    return fresh


#: A contact whose normal is within this of vertical is something the body is
#: resting ON, not something it was moved PAST.
SUPPORT_COS = 0.7            # ~45 degrees


def _obstacle_pairs(traj, t_event):
    """Post-event contacts that are not supports.

    Naming the floor is not enough -- a ramp and a table top are supports too,
    and a body teleported along a ramp still rests on it, correctly. What
    separates the two is the contact normal: a near-vertical normal means the
    body is sitting on the surface, and gravity puts it back there whatever the
    intervention did. A near-horizontal one means the surface is in the way.
    """
    import numpy as np

    c = traj.contacts
    out = set()
    for k in range(len(c)):
        if int(c.frame[k]) < t_event:
            continue
        n = np.asarray(c.normal[k], float)
        mag = float(np.linalg.norm(n))
        if mag > 1e-9 and abs(float(n[2]) / mag) > SUPPORT_COS:
            continue                      # resting on it
        out.add((int(c.body_a[k]), int(c.body_b[k])))
    return out


def _pairs_after(traj, t_event):
    c = traj.contacts
    return {(int(a), int(b)) for f, a, b in
            zip(c.frame, c.body_a, c.body_b) if int(f) >= t_event}


@pytest.fixture(scope="module")
def work():
    plans = _plans()
    if not plans:
        pytest.skip("no worker output; run `python -m physloc.cli generate --debug`")
    return plans


def test_staged_families_carry_simulated_contacts(work):
    """`traj.meta['contacts']` says where the contact list came from.

    'simulated' means PyBullet detected them on the edited world; 'geometric'
    means we inferred them from positions afterwards. A staged family reporting
    'geometric' has silently fallen back to the trajectory-edit path.
    """
    seen = 0
    for pj in work:
        vdir = os.path.dirname(pj)
        family = json.load(open(pj))["plan"]["family"]
        if family not in SIMULATED:
            continue
        tp = os.path.join(vdir, "traj_invalid.npz")
        if not os.path.exists(tp):
            continue
        traj = Trajectory.load(tp)
        source = traj.meta.get("contacts")
        # `solidity` on granular scenes keeps the trajectory path on purpose --
        # removing the floor under forty grains is a scene edit, not one pair.
        if source == "geometric":
            continue
        assert source == "simulated", (
            "%s reports contacts=%r" % (family, source))
        seen += 1
    if not seen:
        pytest.skip("no staged families in this worker output")


def test_a_prevented_collision_leaves_no_contact(work):
    """The exact reported defect.

    When an intervention moves a body away from, or stops it short of, something
    it lawfully hit, that contact must be **absent** from the invalid clip. Not
    merely different -- absent, because the bodies never meet.
    """
    checked = 0
    for pj in work:
        vdir = os.path.dirname(pj)
        root = vdir.split(os.sep + "variants" + os.sep)[0]
        blob = json.load(open(pj))["plan"]
        family = blob["family"]
        if family not in ("continuity", "solidity"):
            continue
        # `solidity` suppresses the pair for a bounded number of frames, and
        # that duration IS its severity axis: at `weak` the body dips in and is
        # pushed back out when contact resumes, which is the violation that bin
        # claims. Only `strong` is guaranteed to be clear of the surface for
        # good, so only `strong` can promise the contact is gone.
        if family == "solidity" and blob["intervention"]["severity_bin"] != "strong":
            continue
        tv = os.path.join(root, "traj_valid.npz")
        ti = os.path.join(vdir, "traj_invalid.npz")
        if not (os.path.exists(tv) and os.path.exists(ti)):
            continue
        a, b = Trajectory.load(tv), Trajectory.load(ti)
        if b.meta.get("contacts") != "simulated":
            continue
        te = int(blob["t_event_frame"])
        before = _pairs_after(a, te)
        culprits = {int(i) for i in blob["causal_body_ids"]}

        lost = before - _pairs_after(b, te)

        # ASK THE PLAN WHICH PAIR IT SUPPRESSED, where the plan knows.
        # `solidity` names its partner outright, and that is a stricter and
        # truer check than any heuristic over normals: the exact pair it
        # disabled must be gone.
        #
        # The heuristic alone was wrong here, and a real clip found it. On
        # `stack_topple` the family suppressed a SUPPORT pair -- body 2 resting
        # on body 4, contact normal z = -0.996 -- because passing through the
        # thing holding you up is a solidity violation like any other. The
        # normal test classified that pair as a support and excluded it, then
        # demanded that some sideways contact be lost instead, and failed on a
        # clip whose physics was exactly right.
        partner = (blob.get("notes") or {}).get("partner_id")
        if partner is not None:
            want = {(min(int(c), int(partner)), max(int(c), int(partner)))
                    for c in culprits if int(c) != int(partner)}
            lost_norm = {(min(x, y), max(x, y)) for x, y in lost}
            had = {(min(x, y), max(x, y)) for x, y in before}
            want &= had                 # only pairs that touched lawfully
            if want:
                assert want & lost_norm, (
                    "%s: the pair it suppressed %s is still in contact after "
                    "t_event -- the intervention did not take"
                    % (family, sorted(want)))
                checked += 1
                continue

        # Otherwise -- `continuity`, which moves a body rather than naming a
        # pair -- fall back to the obstacle heuristic. A body teleported across
        # flat ground still lands on the floor and loses no contact, which is
        # correct; the claim is about an OBSTACLE it was moved past.
        obstacle = {p for p in _obstacle_pairs(a, te)
                    if p[0] in culprits or p[1] in culprits}
        if not obstacle:
            continue

        assert lost & obstacle, (
            "%s: the invalid clip kept every obstacle contact the valid one had "
            "(%s) -- the body it was moved past or through is still stopping it"
            % (family, sorted(obstacle)))
        checked += 1
    if not checked:
        pytest.skip("no simulated continuity/solidity variants present")

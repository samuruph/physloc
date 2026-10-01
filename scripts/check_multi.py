"""Trajectory-level sanity check of invalid variants, aimed at `multi` clips.

    python scripts/check_multi.py <job workdir> [<job workdir> ...] [--family f]

A job workdir is `<workdir>/<level>/<scenario>/<seed>/` -- it holds
`traj_valid.npz` and `variants/<family>_<sev>/{traj_invalid.npz, plan.json}`.
Every check compares the invalid rollout against its lawful twin and reports
what a viewer would call a bug:

  UNCAUSED  a body that is neither a violator nor touched (directly or through
            a chain of contacts) by one leaves its lawful path -- an object
            that jumps, freezes or disappears for no reason in the clip.
  JUMP      a violator or bystander moves further in one frame than it does
            anywhere in the lawful clip, in a family that does not teleport.
  OVERLAP   two bodies interpenetrate where they did not in the lawful clip,
            in a family whose claim is not interpenetration.
  SINK      a body sits below the floor, in a family that does not sink it.
  SINGLE    a `multi` clip whose plan names fewer than two violators.

Reads files only. Exits 1 if anything was flagged.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physloc.sim.trajectory import Trajectory  # noqa: E402

#: Families whose claim IS the anomaly the check would otherwise flag.
TELEPORTS = {"continuity", "time_slip", "fission", "fusion"}
OVERLAPS = {"solidity", "fusion", "fission", "deformation", "immutability"}
SINKS = {"solidity", "support"}
GONE = {"permanence", "dissolve", "fusion"}


def contact_reach(traj, seeds, T):
    """[T, B] bool: bodies reached by a contact chain from `seeds`, by frame."""
    ids = [int(i) for i in traj.body_ids]
    col = {b: j for j, b in enumerate(ids)}
    reached = np.zeros((T, len(ids)), bool)
    live = {col[s] for s in seeds if s in col}
    by_frame = {}
    c = traj.contacts
    for k in range(len(c)):
        by_frame.setdefault(int(c.frame[k]), []).append(
            (int(c.body_a[k]), int(c.body_b[k])))
    for f in range(T):
        for a, b in by_frame.get(f, ()):
            if a in col and b in col:
                if col[a] in live:
                    live.add(col[b])
                if col[b] in live:
                    live.add(col[a])
        for j in live:
            reached[f, j] = True
    return reached


def check_variant(vdir, tv, spec_bodies, statics):
    plan_doc = json.load(open(os.path.join(vdir, "plan.json")))
    plan = plan_doc["plan"]
    family = plan["family"]
    spec = plan_doc["spec"]
    ti = Trajectory.load(os.path.join(vdir, "traj_invalid.npz"))
    T = int(tv.pos.shape[0])
    ids = [int(i) for i in tv.body_ids]
    col = {b: j for j, b in enumerate(ids)}
    viol = [int(v["body_id"]) for v in (plan.get("violators") or [])] or \
        [int(i) for i in plan["causal_body_ids"]]
    t_event = int(plan["t_event_frame"])
    if plan.get("violators"):
        t_event = min(int(v["t_event_frame"]) for v in plan["violators"])
    out = []
    multi = "multi" in str(spec.get("condition", ""))
    if multi and len(plan.get("violators") or plan["causal_body_ids"]) < 2:
        out.append(("SINGLE", "plan has %d violator(s) %s, timing %s"
                    % (len(viol), viol, plan.get("violator_timing"))))
    # SCRIPTED bodies move because the scenario writes their pose -- a
    # pendulum's rod follows its bob, `shadow_track`'s caster follows its actor
    # (`Scenario.rescript`) -- never by contact, so contact reach says nothing
    # about them. Not checked.
    ghosts = {int(b["segmentation_id"]) for b in spec["bodies"]
              if b.get("scripted") or b.get("collides") is False
              or b.get("role") in ("shadow", "shadow_caster")}
    dyn = [j for j, b in enumerate(ids) if not bool(tv.is_static[j])
           and b not in statics and b not in ghosts]
    # Reach through the INVALID clip's contacts, plus every violator from its
    # own event onwards -- the bodies that are allowed to differ.
    reach = contact_reach(ti, viol, T)
    for v in viol:
        if v in col:
            reach[:, col[v]] = True
    dpos = np.linalg.norm(ti.pos - tv.pos, axis=2)          # [T, B]
    rad = np.maximum(np.asarray(tv.radius, np.float64), 1e-3)
    for j in dyn:
        if ids[j] in viol:
            continue
        bad = np.flatnonzero((dpos[:, j] > 0.25 * rad[j]) & ~reach[:, j]
                             & (np.arange(T) >= t_event))
        gone = np.flatnonzero(~ti.present[:, j] & tv.present[:, j])
        if bad.size:
            out.append(("UNCAUSED", "%s (id %d) leaves its lawful path at "
                        "frame %d by %.2f m with no contact from a violator"
                        % (tv.body_names[j], ids[j], bad[0],
                           dpos[bad[0], j])))
        if gone.size and family not in GONE:
            out.append(("UNCAUSED", "%s (id %d) vanishes at frame %d"
                        % (tv.body_names[j], ids[j], gone[0])))
    if family not in TELEPORTS:
        # A TELEPORT is a displacement the body's own velocity cannot explain.
        # Fast lawful motion -- a kick, reversed gravity -- moves far in a frame
        # and is not one; a body dropped somewhere new by a stale stage is.
        dt = 1.0 / float(tv.fps or 30.0)
        for j in dyn:
            step = np.diff(ti.pos[:, j], axis=0)
            v = np.asarray(ti.lin_vel[:, j], np.float64)
            # Distance from the step to the SEGMENT between the displacements
            # the velocity before and after the frame imply, not to their
            # mean: at a bounce the velocity flips, the mean is ~0, and a
            # lawful bounce would read as a teleport.
            a0, b0 = v[:-1] * dt, v[1:] * dt
            ab = b0 - a0
            u = np.clip(np.einsum("ij,ij->i", step - a0, ab)
                        / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12),
                        0.0, 1.0)
            err = np.linalg.norm(step - (a0 + u[:, None] * ab), axis=1)
            # CALIBRATED ON THE LAWFUL CLIP. Velocity is sampled once a frame
            # while contacts happen between substeps, so a lawful impact also
            # leaves an unexplained step; a jump is only one that dwarfs the
            # worst this body ever shows lawfully.
            vstep = np.diff(tv.pos[:, j], axis=0)
            vv = np.asarray(tv.lin_vel[:, j], np.float64)
            va, vb = vv[:-1] * dt, vv[1:] * dt
            vab = vb - va
            vu = np.clip(np.einsum("ij,ij->i", vstep - va, vab)
                         / np.maximum(np.einsum("ij,ij->i", vab, vab), 1e-12),
                         0.0, 1.0)
            lawful = float(np.linalg.norm(vstep - (va + vu[:, None] * vab),
                                          axis=1).max())
            lim = max(0.5 * rad[j], 0.05, 2.0 * lawful)
            hit = np.flatnonzero((err > lim) & ti.present[1:, j]
                                 & ti.present[:-1, j])
            hit = hit[hit + 1 >= t_event]
            if hit.size:
                out.append(("JUMP", "%s (id %d) moves %.2f m in one frame at "
                            "%d, %.2f m more than its velocity explains"
                            % (tv.body_names[j], ids[j],
                               np.linalg.norm(step[hit[0]]), hit[0] + 1,
                               err[hit[0]])))
    if family not in OVERLAPS:
        for a in range(len(dyn)):
            for b in range(a + 1, len(dyn)):
                ja, jb = dyn[a], dyn[b]
                lim = 0.6 * (rad[ja] + rad[jb])
                both = ti.present[:, ja] & ti.present[:, jb]
                di = np.linalg.norm(ti.pos[:, ja] - ti.pos[:, jb], axis=1)
                dv = np.linalg.norm(tv.pos[:, ja] - tv.pos[:, jb], axis=1)
                hit = np.flatnonzero(both & (di < lim) & (dv >= lim))
                if hit.size:
                    out.append(("OVERLAP", "%s and %s interpenetrate from "
                                "frame %d (%.2f m apart, radii %.2f+%.2f)"
                                % (tv.body_names[ja], tv.body_names[jb],
                                   hit[0], di[hit[0]], rad[ja], rad[jb])))
    floor = next((b for b in spec["bodies"] if b.get("role") == "floor"), None)
    if family not in SINKS and floor is not None and floor.get("kind") == "cube":
        fz = float(tv.pos[0, col[int(floor["segmentation_id"])], 2])
        top = fz + float(floor["scale"][2])
        for j in dyn:
            # The centre below the floor surface plus a quarter radius: inside
            # the floor, which no resting or rolling body is.
            # At the body's CURRENT size: `immutability` and `deformation`
            # shrink it, and a smaller body rests with its centre lower.
            shrink = np.asarray(ti.scale_mul[:, j], np.float64).min(axis=1)
            lim_i = top + 0.25 * rad[j] * shrink
            lim = top + 0.25 * rad[j]
            low = np.flatnonzero((ti.pos[:, j, 2] < lim_i) & ti.present[:, j]
                                 & ~(tv.pos[:, j, 2] < lim))
            if low.size:
                out.append(("SINK", "%s (id %d) is inside the floor at frame "
                            "%d (centre z %.2f, floor %.2f)"
                            % (tv.body_names[j], ids[j], low[0],
                               ti.pos[low[0], j, 2], top)))
    return family, viol, plan.get("violator_timing"), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--family", help="only these families (comma list)")
    ap.add_argument("--quiet", action="store_true", help="flagged variants only")
    a = ap.parse_args()
    want = {f for f in (a.family or "").split(",") if f}
    flagged = 0
    for job in a.jobs:
        tv = Trajectory.load(os.path.join(job, "traj_valid.npz"))
        for vdir in sorted(glob.glob(os.path.join(job, "variants", "*"))):
            if not os.path.isfile(os.path.join(vdir, "traj_invalid.npz")):
                continue
            spec = json.load(open(os.path.join(vdir, "plan.json")))["spec"]
            statics = {int(b["segmentation_id"]) for b in spec["bodies"]
                       if b.get("static")}
            fam, viol, timing, out = check_variant(vdir, tv, spec["bodies"],
                                                   statics)
            if want and fam not in want:
                continue
            name = os.path.relpath(vdir, os.path.dirname(os.path.dirname(job)))
            if out:
                flagged += 1
            if out or not a.quiet:
                print("%s %s violators=%s timing=%s" % (
                    "!!" if out else "ok", name, viol, timing))
                for kind, msg in out:
                    print("     %-8s %s" % (kind, msg))
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())

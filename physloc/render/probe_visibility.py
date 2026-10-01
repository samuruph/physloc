"""Why does the visibility gate decline a cell, and by how much?

The worker drops a (family, severity) whose violators leave the shot at every
event moment. This replays exactly that path -- same scene sample, same framing
loop, same valid rollout, same `_invalid_variant` per event attempt -- with no
renderer, and reports what `violators_visible` measured on each attempt: whether
the opening `EVIDENCE_SECONDS` were all on screen (`head`) and the share of the
`min_visible_frames` span that was (`share`). That is what a threshold change
would act on.

    bash docker/kubric.sh physloc/render/probe_visibility.py \
        --scenario drop --seed 20260824 --variant 0 --n-variants 10 \
        --complexity L0 --frames 61 --params /path/params.json \
        --cells non_parabolic/strong,antigravity/strong

One JSON line per (family, severity): `attempts` is a list of
`[head, share, t_event]` (or an error string), indexed by event attempt.
"""
import argparse
import gc
import json
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from physloc import injectors, scenarios
from physloc.injectors import _geom
from physloc.render import worker as W


def _gate(spec, family, plan, traj_valid, traj_invalid):
    """(head_ok, share) as `violators_stay_visible` sees them; worst violator."""
    traj = traj_valid if family in W.ABSENCE_FAMILIES else traj_invalid
    by_id = {int(b.segmentation_id): b for b in spec.bodies}
    if plan.violators:
        groups = [([by_id[c.body_id]], c.t_event) for c in plan.violators
                  if c.body_id in by_id and not by_id[c.body_id].static]
    else:
        ids = {int(i) for i in plan.causal_body_ids}
        groups = [([b for b in spec.bodies
                    if int(b.segmentation_id) in ids and not b.static],
                   plan.t_event)]
    T = int(traj.num_frames)
    fps = float(spec.tier.fps)
    heads, shares = [], []
    for bodies, t_event in groups:
        if not bodies:
            continue
        t = int(np.clip(t_event, 0, T - 1))
        span = min(_geom.min_visible_frames(spec, T), T - t)
        if span <= 0:
            heads.append(False)
            shares.append(0.0)
            continue
        on = _geom.violators_on_screen(spec, traj, bodies)[t:t + span]
        head = max(1, min(span, int(round(_geom.EVIDENCE_SECONDS * fps))))
        heads.append(bool(on[:head].all()))
        shares.append(float(on.mean()))
    if not shares:
        return True, 1.0
    return all(heads), min(shares)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--n-variants", type=int, default=None)
    ap.add_argument("--complexity", default="L0")
    ap.add_argument("--tier", default="release")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--params")
    ap.add_argument("--cells", required=True,
                    help="comma list of family/severity")
    ap.add_argument("--scratch", default="/tmp/probe_visibility")
    ap.add_argument("--dump-spec", help="write the sampled spec here (json)")
    ap.add_argument("--audit-state", action="store_true",
                    help="after every cell, report any simulator state that "
                         "differs from before the first one")
    a = ap.parse_args(W._blender_argv())

    if a.params:
        from physloc import params as _params
        _params.apply(_params.read(a.params))
    tier = scenarios.TIERS[a.tier].override(num_frames=a.frames)
    scenario = scenarios.get(a.scenario)
    os.makedirs(a.scratch, exist_ok=True)

    # The worker's framing loop, verbatim in effect.
    for attempt in range(W.FRAMING_ATTEMPTS):
        spec = scenario.sample(a.seed, tier, a.complexity, variant=a.variant,
                               n_variants=a.n_variants, attempt=attempt)
        sc, sim, _r, objs = W.build_scene(spec, a.scratch, render=False)
        probe = W.simulate(spec, sc, sim, objs,
                           tuple(scenario.sim_hooks(spec, sim, objs) or ()))
        scenario.script(spec, probe)
        framed = scenario.framing_ok(spec, probe)
        # As the worker does: a second live PyBullet beside this one records
        # no contacts at all, which silently changes every rollout after it.
        del sc, sim, objs, probe
        gc.collect()
        if framed or attempt == W.FRAMING_ATTEMPTS - 1:
            break
    scene, simulator, _r, objs = W.build_scene(spec, a.scratch, render=False)
    scen_hooks = tuple(scenario.sim_hooks(spec, simulator, objs) or ())
    traj_valid = W.simulate(spec, scene, simulator, objs, scen_hooks)
    scenario.script(spec, traj_valid)
    if a.dump_spec:
        with open(a.dump_spec, "w") as fh:
            json.dump(spec.to_dict(), fh, indent=1, sort_keys=True, default=str)
        traj_valid.save(os.path.splitext(a.dump_spec)[0] + "_traj_valid.npz")

    def _state():
        import pybullet as pb
        from physloc.render import stepper
        out = {"n_bodies": pb.getNumBodies()}
        for b in spec.bodies:
            idx = stepper.pybullet_index(simulator, objs, spec,
                                         int(b.segmentation_id))
            if idx is None:
                out[b.name] = None
                continue
            info = pb.getDynamicsInfo(idx, -1)
            pos, _q = pb.getBasePositionAndOrientation(idx)
            out[b.name] = {"idx": idx, "mass": round(info[0], 4),
                           "friction": round(info[1], 4),
                           "restitution": round(info[5], 4),
                           "z": round(pos[2], 3)}
        return out
    base_state = _state() if a.audit_state else None

    for cell in [c.strip() for c in a.cells.split(",") if c.strip()]:
        family, sev = cell.split("/")
        inj = injectors.get(family)
        tag = "%s/%s" % (family, sev)
        rng_seed = (a.seed + 7919 + zlib.crc32(tag.encode())) % (2 ** 31 - 1)
        rows = []
        for attempt in range(W.EVENT_ATTEMPTS):
            inj.event_attempt = attempt
            try:
                made = W._invalid_variant(spec, scenario, inj, sev, rng_seed,
                                          traj_valid, simulator, scene, objs,
                                          scen_hooks)
                if not made.get("ok"):
                    rows.append(str(made.get("error"))[:120])
                else:
                    head, share = _gate(spec, family, made["plan"], traj_valid,
                                        made["traj"])
                    # The same window on the LAWFUL twin: was the violator in
                    # shot there at all? Separates "the violation threw it
                    # out" from "it was never on screen at that moment".
                    vhead, vshare = _gate(spec, family, made["plan"],
                                          traj_valid, traj_valid)
                    rows.append([head, round(share, 3),
                                 int(made["plan"].t_event),
                                 bool(made["visible"]),
                                 vhead, round(vshare, 3)])
            except Exception as exc:                          # noqa: BLE001
                rows.append("raised %r" % (exc,))
            finally:
                inj.event_attempt = 0
        if base_state is not None:
            now = _state()
            diff = {k: [base_state.get(k), now.get(k)] for k in now
                    if now.get(k) != base_state.get(k)}
            if diff:
                print("STATE after %s: %s" % (cell, json.dumps(diff)),
                      flush=True)
        print("PROBE " + json.dumps({
            "scenario": a.scenario, "seed": a.seed, "level": a.complexity,
            "variant": a.variant, "family": family, "severity": sev,
            "attempts": rows}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

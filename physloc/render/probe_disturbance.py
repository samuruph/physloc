"""How far does each family actually move the scene, at one severity bin?

The question `pour` raised: at `strong`, `superelastic` scattered the medium
27 m out of a 0.68 m box while `phantom_impulse` shifted it 0.84 m -- so "the
strong bin" meant two wildly different things depending on which family drew
it. Comparing those needs the STAGED rollout, because a granular scene is
exactly where the host-side preview stops describing what the solver does, and
it does not need a single frame rendered.

    bash docker/kubric.sh physloc/render/probe_disturbance.py --scenario pour
    bash docker/kubric.sh physloc/render/probe_disturbance.py \
        --scenario collision --families superelastic,dissolve --verbose

Reports, per family, how far the culprit bodies end up from where the lawful
twin left them, in metres and in units of the scene's own frame extent -- the
second is the comparable number, since a metre means something different in a
0.68 m box and a 6 m room.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from physloc import camera as cam
from physloc import injectors, scenarios, taxonomy
from physloc.render import stepper
from physloc.render.worker import build_scene, simulate


def _staged(spec, sim, objs, scene, traj_valid, inj, plan):
    """The worker's staged path, minus everything that draws."""
    scripted = {int(b.segmentation_id) for b in spec.bodies if b.scripted}
    scripted -= {int(i) for i in inj.revives(spec, plan)}
    if not (inj.simulates(plan) and not scripted.intersection(plan.causal_body_ids)):
        return inj.apply(spec, traj_valid, plan), False
    try:
        stepper.reset_to(spec, objs, traj_valid, plan.t_event)
        hooks = tuple(inj.stage(spec, sim, objs, plan) or ())
        tail = stepper.run_from(sim, scene, spec, objs, plan.t_event,
                                spec.tier.num_frames - 1, hooks)
        out = stepper.splice(traj_valid, tail, plan.t_event)
    finally:
        inj.unstage(spec, sim, objs, plan)
        stepper.reset_to(spec, objs, traj_valid, spec.tier.num_frames - 1)
    return inj.post_simulate(spec, traj_valid, out, plan), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="pour")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--severity", default="strong")
    ap.add_argument("--complexity", default="L0")
    ap.add_argument("--families", default="",
                    help="comma list; default is every compatible family")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    tier = scenarios.TIERS["debug"]
    spec = scenarios.get(a.scenario).sample(a.seed, tier, a.complexity)
    scene, sim, renderer, objs = build_scene(spec, "out/_probe_dist")
    traj = simulate(spec, scene, sim, objs)
    scenarios.get(a.scenario).script(spec, traj)
    del renderer

    extent = float(cam.frame_extent(spec.camera_position, spec.camera_look_at))
    fams = ([f for f in a.families.split(",") if f] or
            sorted(f for f in taxonomy.FAMILIES
                   if taxonomy.is_compatible(a.scenario, f)))
    print("scenario=%s seed=%d %s frames=%d frame_extent=%.2f m"
          % (a.scenario, a.seed, a.complexity, tier.num_frames, extent))
    print("%-18s %6s %8s %9s %9s %7s" % ("family", "staged", "max_m",
                                         "mean_m", "max/extent", "n_body"))
    rows = []
    for fam in fams:
        try:
            inj = injectors.get(fam)
        except Exception as exc:                              # noqa: BLE001
            print("%-18s  -- no injector (%s)" % (fam, exc))
            continue
        rng = np.random.RandomState(0)
        try:
            plan = inj.plan(spec, traj, rng, a.severity)
        except Exception as exc:                              # noqa: BLE001
            print("%-18s  -- plan raised %s" % (fam, exc))
            continue
        if plan is None:
            print("%-18s  -- no plan" % fam)
            continue
        try:
            inv, staged = _staged(spec, sim, objs, scene, traj, inj, plan)
        except Exception as exc:                              # noqa: BLE001
            print("%-18s  -- rollout raised %s" % (fam, exc))
            continue
        idx = [traj.index_of(int(i)) for i in plan.causal_body_ids]
        d = np.linalg.norm(np.asarray(inv.pos)[:, idx, :]
                           - np.asarray(traj.pos)[:, idx, :], axis=2)
        row = (fam, staged, float(d.max()), float(d[-1].mean()),
               float(d.max()) / max(extent, 1e-9), len(idx))
        rows.append(row)
        print("%-18s %6s %8.2f %9.3f %9.2f %7d"
              % (fam, "yes" if staged else "EDIT", row[2], row[3], row[4], row[5]))
        if a.verbose:
            print("     magnitude=%.3f %s params=%s"
                  % (plan.magnitude, plan.magnitude_unit,
                     json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                 for k, v in plan.params.items()
                                 if not isinstance(v, (list, dict))})))
    if rows:
        r = sorted(rows, key=lambda x: -x[4])
        print("\nspread: strongest %s at %.1f extents, weakest %s at %.2f"
              % (r[0][0], r[0][4], r[-1][0], r[-1][4]))


if __name__ == "__main__":
    main()

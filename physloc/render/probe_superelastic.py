"""Why does the superelastic boost not take effect on `collision`?

Four attempts have been ruled out by measuring the OUTPUT. This measures the
hook itself: every substep it sees, whether it thinks the bodies are touching,
and what it does when contact ends.

    bash docker/kubric.sh physloc/render/probe_superelastic.py --scenario collision
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from physloc import injectors, scenarios
from physloc.render import stepper
from physloc.render.worker import build_scene, simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="collision")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--severity", default="strong")
    a = ap.parse_args()

    import pybullet as pb

    tier = scenarios.TIERS["debug"]
    spec = scenarios.get(a.scenario).sample(a.seed, tier, "L0")
    scene, sim, renderer, objs = build_scene(spec, "out/_probe_se")
    traj = simulate(spec, scene, sim, objs)
    scenarios.get(a.scenario).script(spec, traj)

    inj = injectors.get("superelastic")
    rng = np.random.RandomState(0)
    plan = inj.plan(spec, traj, rng, a.severity)
    print("PLAN t_event=%s partner=%s gain=%s approach=%s causal=%s"
          % (plan.t_event, plan.notes.get("partner_id"),
             plan.params.get("speed_gain"), plan.notes.get("approach_speed"),
             plan.causal_body_ids))

    stepper.reset_to(spec, objs, traj, plan.t_event)
    import inspect
    print("INJ class=%s stage=%s simulated=%s"
          % (type(inj).__name__, inj.stage.__qualname__,
             getattr(inj, "simulated", None)))
    src = inspect.getsource(inj.stage)
    print("stage source has DEBUG marker:", "PHYSLOC_DEBUG_SE" in src)
    print("stage source has moving_partner:", "moving_partner" in src)
    hooks = inj.stage(spec, sim, objs, plan) or ()
    print("hooks returned:", len(hooks))

    idx = stepper.pybullet_index(spec=spec, simulator=sim, objs=objs,
                                 seg_id=int(plan.causal_body_ids[0]))
    partner = stepper.pybullet_index(spec=spec, simulator=sim, objs=objs,
                                     seg_id=int(plan.notes["partner_id"]))
    print("pybullet idx actor=%s partner=%s" % (idx, partner))

    spf = stepper.steps_per_frame(scene)
    log = []

    def spy(_c, step, frame):
        pts = [c for c in pb.getContactPoints(bodyA=idx)
               if partner in (c[1], c[2])]
        va = np.asarray(pb.getBaseVelocity(idx)[0], np.float64)
        vb = np.asarray(pb.getBaseVelocity(partner)[0], np.float64)
        log.append((step, frame, len(pts),
                    float(np.linalg.norm(va - vb))))

    all_hooks = list(hooks) + [spy]
    stepper.run_from(sim, scene, spec, objs, plan.t_event,
                     spec.tier.num_frames - 1, all_hooks)

    print("\nstep frame contacts |v_rel|")
    prev = None
    for step, frame, n, rel in log[:spf * 4]:
        mark = " <-- contact ends" if prev and prev > 0 and n == 0 else ""
        print("%4d %5d %8d %7.3f%s" % (step, frame, n, rel, mark))
        prev = n


if __name__ == "__main__":
    main()

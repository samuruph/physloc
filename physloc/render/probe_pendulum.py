"""How well does each way of holding a pendulum bob on its rod keep the swing?

    bash docker/kubric.sh physloc/render/probe_pendulum.py < /dev/null

Three candidates on the real `pendulum_swing` release scene, 61 frames:

  A  the scenario's own hook (projection + energy clamp), as shipped
  B  PyBullet's native point-to-point joint, default solver
  C  the same joint, stiff solver (erp, iterations) and no body damping

Reports rod-length drift and the energy kept, per candidate.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pybullet as pb

from physloc import scenarios
from physloc.render import stepper
from physloc.render import worker as W

SEED = int(os.environ.get("SEED", "20260824"))


def run(mode):
    tier = scenarios.TIERS["release"].override(num_frames=61)
    sc = scenarios.get("pendulum_swing")
    spec = sc.sample(SEED, tier, "L0", variant=0, n_variants=10)
    scene, sim, _r, objs = W.build_scene(spec, "/tmp/probe_pend", render=False)
    pivot = np.asarray(spec.notes["pivot"], np.float64)
    arm = float(spec.notes["arm"])
    idx = stepper.pybullet_index(sim, objs, spec, sc.SEG_BOB)
    if mode == "A":
        hooks = tuple(sc.sim_hooks(spec, sim, objs) or ())
    else:
        pos, quat = pb.getBasePositionAndOrientation(idx)
        ip, iq = pb.invertTransform(pos, quat)
        local, _ = pb.multiplyTransforms(ip, iq, pivot.tolist(), [0, 0, 0, 1])
        anchor = pb.createMultiBody(0.0, -1, -1, pivot.tolist())
        cid = pb.createConstraint(idx, -1, anchor, -1, pb.JOINT_POINT2POINT,
                                  [0, 0, 0], local, [0, 0, 0])
        if mode == "C":
            pb.changeConstraint(cid, maxForce=1e7, erp=0.9)
            pb.changeDynamics(idx, -1, linearDamping=0.0, angularDamping=0.0)
            pb.setPhysicsEngineParameter(numSolverIterations=200)
        hooks = ()
    tail = stepper.run_from(sim, scene, spec, objs, 0, 60, hooks)
    p = np.asarray(tail["pos"])[:, [b.name for b in spec.bodies].index(
        next(b.name for b in spec.bodies if int(b.segmentation_id) == sc.SEG_BOB))]
    v = np.asarray(tail["lin_vel"])[:, [b.name for b in spec.bodies].index(
        next(b.name for b in spec.bodies if int(b.segmentation_id) == sc.SEG_BOB))]
    L = np.linalg.norm(p - pivot, axis=1)
    E = 0.5 * (v ** 2).sum(1) + 9.81 * p[:, 2]
    ang = np.degrees(np.arctan2(p[:, 0] - pivot[0], -(p[:, 2] - pivot[2])))
    print("PEND %s length %.3f..%.3f (arm %.3f)  energy kept %.1f%%  "
          "angle 0/15/30/45/60: %s" % (
              mode, L.min(), L.max(), arm,
              100.0 * (E[-1] - 9.81 * (pivot[2] - arm))
              / max(E[0] - 9.81 * (pivot[2] - arm), 1e-9),
              np.round(ang[[0, 15, 30, 45, 60]], 1).tolist()), flush=True)
    pb.disconnect()


for m in ("A",):
    run(m)

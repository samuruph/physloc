"""What timestep does PyBullet actually integrate with, per tier?

    bash docker/kubric.sh physloc/render/probe_timestep.py < /dev/null

A frame must advance the physics by exactly 1/fps. Kubric steps
`step_rate / frame_rate` substeps per frame; whether each substep is
1/step_rate seconds depends on PyBullet's `fixedTimeStep`, which this prints.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import pybullet as pb

from physloc import scenarios
from physloc.render import worker as W

for tier_name, frames in (("debug", None), ("release", 61)):
    tier = scenarios.TIERS[tier_name].override(num_frames=frames)
    spec = scenarios.get("drop").sample(20260824, tier, "L0", variant=0,
                                        n_variants=10)
    scene, sim, _r, objs = W.build_scene(spec, "/tmp/probe_ts", render=False)
    p = pb.getPhysicsEngineParameters()
    n = int(scene.step_rate) // int(scene.frame_rate)
    print("TIMESTEP %s fps=%s step_rate=%s fixedTimeStep=%.6f substeps=%d "
          "-> seconds/frame=%.4f (should be %.4f)"
          % (tier_name, scene.frame_rate, scene.step_rate, p["fixedTimeStep"],
             n, p["fixedTimeStep"] * n, 1.0 / float(scene.frame_rate)),
          flush=True)
    del scene, sim, objs

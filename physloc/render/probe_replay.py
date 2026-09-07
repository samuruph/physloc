"""Is `replay()` idempotent? Render the SAME trajectory twice through it.

The valid render is the first replay a scene ever sees; every invalid render is
a replay laid on top of one. If replaying identical values a second time does
not reproduce the first render, that difference is present in every invalid
clip whether or not its injector touched anything -- which is exactly the shape
of the prefix-identity failures found in the review sweep (29 of 176 clips
differing a frame or two *before* t_event, on families that touch no material
channel at all).

    bash docker/kubric.sh physloc/render/probe_replay.py --scenario pendulum_swing
"""
import argparse
import json
import os

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from physloc import scenarios
from physloc.render.worker import build_scene, render_and_save, replay, simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="pendulum_swing")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--tier", default="debug")
    ap.add_argument("--complexity", default="L0")
    ap.add_argument("--outdir", default="out/probe_replay")
    a = ap.parse_args()

    tier = scenarios.TIERS[a.tier]
    spec = scenarios.get(a.scenario).sample(a.seed, tier, a.complexity)
    scratch = os.path.join(a.outdir, "_scratch")
    os.makedirs(scratch, exist_ok=True)

    scene, simulator, renderer, objs = build_scene(spec, scratch)
    traj = simulate(spec, scene, simulator, objs)
    scenarios.get(a.scenario).script(spec, traj)

    replay(spec, objs, traj, renderer)
    os.makedirs(os.path.join(a.outdir, "a"), exist_ok=True)
    render_and_save(renderer, scene, spec, objs, os.path.join(a.outdir, "a"), "one")
    replay(spec, objs, traj, renderer)          # identical values, second time
    os.makedirs(os.path.join(a.outdir, "b"), exist_ok=True)
    render_and_save(renderer, scene, spec, objs, os.path.join(a.outdir, "b"), "two")

    ra = np.load(os.path.join(a.outdir, "a", "passes_one.npz"))["rgba"]
    rb = np.load(os.path.join(a.outdir, "b", "passes_two.npz"))["rgba"]
    d = np.abs(ra.astype(int) - rb.astype(int))
    print("PROBE " + json.dumps({
        "scenario": a.scenario, "frames": int(ra.shape[0]),
        "idempotent": bool(d.max() == 0), "max_abs_diff": int(d.max()),
        "per_frame_max": d.max(axis=(1, 2, 3)).tolist()}))


if __name__ == "__main__":
    main()

"""Where does a release frame's time actually go?

`SECONDS_PER_CLIP` says 637 s for a solid background and 2930 s for an HDRI at
release geometry -- 4.6x -- and that ratio decides whether a full v0 run is
days or weeks. What it does NOT say is WHY, and the answer changes what is
worth doing about it:

  * if the HDRI cost is SAMPLING, halving `spp` nearly halves it, and the
    existing 256sq fit (`T = 1.29 + 0.0074*spp`, only ~26% sampling) simply
    does not transfer to a scene lit by an environment map;
  * if it is fixed per-frame work -- loading a 4k .hdr, shading a 80 m dome --
    then spp buys nothing and the lever is elsewhere.

Fits `T = a + b*spp` for both backgrounds, at the real release resolution, and
prints the split. Deliberately a handful of frames per point: the fit only has
to separate a constant from a slope.

One measurement per invocation -- a PyBullet connection does not survive a
second scene in the same interpreter -- so loop over the points outside:

    for L in L0 L2; do for S in 16 64; do
      bash docker/kubric.sh physloc/render/probe_cost.py \
           --complexity $L --spp $S --resolution 512 --frames 2
    done; done

Each line prints `COST <level> res spp build per_frame`. Fit `per_frame`
against `spp` per level: a steep slope means sampling dominates and `spp` is
the lever; a large intercept means it does not.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--complexity", default="L0")
    a = ap.parse_args()

    # ONE MEASUREMENT PER PROCESS, on purpose. `kb.simulator.PyBullet` connects
    # to a physics server on construction and the connection does not survive a
    # second scene in the same interpreter -- the first version built four
    # scenes in a loop and died on the second with "Not connected to physics
    # server". The caller loops instead; see the docstring.
    from physloc import scenarios
    from physloc.render.worker import build_scene

    tier = scenarios.TIERS["debug"].override(resolution=a.resolution,
                                             samples_per_pixel=a.spp)
    spec = scenarios.get("drop").sample(777, tier, a.complexity)

    t0 = time.perf_counter()
    scene, sim, renderer, objs = build_scene(spec, "out/_probe_cost")
    build = time.perf_counter() - t0

    scene.frame_start, scene.frame_end = 0, max(0, a.frames - 1)
    t0 = time.perf_counter()
    renderer.render(return_layers=("rgba",))
    render = time.perf_counter() - t0

    print("COST %s res=%d spp=%d build=%.2f per_frame=%.3f"
          % (a.complexity, a.resolution, a.spp, build,
             render / max(1, a.frames)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

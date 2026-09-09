"""Is the KuBasic dome's floor the same surface as our cube floor?

SUPERSEDED, and kept because its measurement is still the reason the code looks
the way it does. The answer below is yes -- the dome's inner floor really is
flat to within Bullet's collision margin -- and the dome duly became the ground
at every level. Then `probe_cost.py` priced it: **8.69 s/frame on the cube
against 27.54 on the dome**, because a dome encloses the scene and a slab does
not. So the dome went back to being a backdrop (`_common.backdrop`, HDRI levels
only, collisions disabled) and the cube went back to being the ground. The
level comparison this probe was run to enable is preserved by the collider
being uniform, which it is.

The original question follows.

The one thing blocking L2. `_common.ground` returns a cube below the HDRI level
and the dome at it, and a dome is a genuinely different collision shape -- so
the same seed does not roll the same way and L1/L2 are two independent releases
that happen to share a seed rather than a pair.

The roadmap's preferred fix is to use the dome at EVERY level, shaded flat
below L2 instead of with an HDRI, which makes the geometry identical by
construction. That only works if the dome's inner floor is flat, level, and at
z = 0 over the few metres a scenario actually uses. Measured here rather than
assumed:

    bash docker/kubric.sh physloc/render/probe_dome.py

Drops the same sphere on each surface from the same height, at several offsets
from the origin, and prints where it comes to rest. Equal columns mean the swap
is free; a drift with radius means the dome is a bowl and the cube has to stay.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

DT = 1.0 / 240.0
STEPS = 1200


def settle(pb, body):
    for _ in range(STEPS):
        pb.stepSimulation()
    pos, _ = pb.getBasePositionAndOrientation(body)
    vel, _ = pb.getBaseVelocity(body)
    return float(pos[2]), float(np.linalg.norm(vel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.35)
    ap.add_argument("--drop-from", type=float, default=1.5)
    ap.add_argument("--offsets", default="0,0.5,1,2,3,4,5")
    a = ap.parse_args()

    import kubric as kb
    import pybullet as pb

    from physloc.render.worker import KUBASIC

    src = kb.AssetSource.from_manifest(KUBASIC)
    dome = src.create(asset_id="dome", name="dome", static=True,
                      friction=0.6, restitution=0.4, background=True)
    print("dome bounds:", np.asarray(dome.bounds).tolist())

    offs = [float(x) for x in a.offsets.split(",")]
    print("\n%-8s %-22s %-22s %s" % ("offset", "on the CUBE floor",
                                     "on the DOME floor", "difference"))
    rows = []
    for surface in ("cube", "dome"):
        scene = kb.Scene(frame_start=0, frame_end=1)
        sim = kb.simulator.PyBullet(scene)
        if surface == "cube":
            floor = kb.Cube(name="floor", scale=(6, 6, 0.1),
                            position=(0, 0, -0.1), static=True,
                            friction=0.6, restitution=0.4)
        else:
            floor = src.create(asset_id="dome", name="floor", static=True,
                               friction=0.6, restitution=0.4, background=True)
        scene += floor
        sim.add(floor)
        pb.setGravity(0, 0, -9.81)
        pb.setTimeStep(DT)

        got = []
        for d in offs:
            ball = kb.Sphere(name="b%s" % d, scale=a.radius,
                             position=(d, 0.0, a.drop_from), mass=1.0,
                             friction=0.6, restitution=0.4)
            scene += ball
            sim.add(ball)
            z, v = settle(pb, ball.linked_objects[sim])
            got.append((z, v))
            # park it far away so the next drop lands on bare floor
            pb.resetBasePositionAndOrientation(
                ball.linked_objects[sim], (0, 0, -500), (0, 0, 0, 1))
            pb.resetBaseVelocity(ball.linked_objects[sim], [0, 0, 0], [0, 0, 0])
        rows.append(got)

    for d, (zc, vc), (zd, vd) in zip(offs, rows[0], rows[1]):
        print("%-8.2f z=%-7.4f |v|=%-8.4f z=%-7.4f |v|=%-8.4f %+.4f"
              % (d, zc, vc, zd, vd, zd - zc))
    worst = max(abs(b[0] - a_[0]) for a_, b in zip(rows[0], rows[1]))
    print("\nlargest disagreement over the tested span: %.4f m "
          "(a sphere of radius %.2f)" % (worst, a.radius))
    print("VERDICT:", "the dome can replace the cube" if worst < 1e-3
          else "the surfaces differ -- the swap is not free")


if __name__ == "__main__":
    main()

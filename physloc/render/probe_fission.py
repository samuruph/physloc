"""Where does a fission half actually come to rest?

You reported the two cones floating. The cause is an ORIGIN and EXTENT
mismatch, not a hull-shape one: `ShapeSwap` built its proxy from a hull in unit
coordinates spanning [-1, +1] on every axis, and stood it at the pose the
declared body had. No KuBasic mesh spans that. A cylinder is half of it, a
torus is a disc a fifth as thick as it is wide, and a cone is not centred on
its own origin at all -- base at z = -0.306, apex at +0.900.

Measured here, in the pinned image, dropping onto a plane at scale 0.5 from
0.30 m (high enough to settle, low enough not to topple):

    native  rest z = 0.1560     the KuBasic cone itself
    table1  rest z = 0.1541     `hull_for("cone")` at scale 1  -- agrees
    proxyk  rest z = 0.1225     the same at the cleave scale, correct 0.1238
    sphere  rest z = 0.3948     the OLD hull -- 3.2x too high, the float

Run it:

    bash docker/kubric.sh physloc/render/probe_fission.py --drop-from 0.30

`--drop-from 2.0` is the original release height and is worth watching too, but
a cone dropped that far topples, and a toppled cone's resting height says
nothing about whether its collider matches its mesh.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

STEPS = 900
DT = 1.0 / 240.0


def _settle(pb, body):
    for _ in range(STEPS):
        pb.stepSimulation()
    pos, quat = pb.getBasePositionAndOrientation(body)
    vel, _ = pb.getBaseVelocity(body)
    aabb = pb.getAABB(body)
    return (float(pos[2]), float(np.linalg.norm(vel)),
            float(aabb[0][2]), np.round(quat, 3).tolist())


def bounce(a):
    """Native asset vs proxy, dropped identically, traced at the clip's fps.

    The second half of the report: the halves were not only resting too high,
    they were BOUNCING -- to z = 1.11 where the valid cone's own bounce reached
    0.34, and still bouncing when the clip ended. Same declared restitution on
    both, so if the traces differ the proxy is not the body it stands in for.
    """
    import pybullet as pb
    import kubric as kb

    from physloc.render import stepper
    from physloc.render.worker import KUBASIC

    r, rest, fric = a.scale, a.restitution, a.friction
    fps, frames = a.fps, a.frames
    sub = max(1, int(round(1.0 / (fps * DT))))

    src = kb.AssetSource.from_manifest(KUBASIC)
    scene = kb.Scene(frame_start=0, frame_end=1)
    sim = kb.simulator.PyBullet(scene)
    floor = kb.Cube(name="floor", scale=(6, 6, 0.1), position=(0, 0, -0.1),
                    static=True, friction=0.6, restitution=0.4)
    cone = src.create(asset_id="cone", name="cone", scale=r,
                      position=(0, 0, a.drop_from), mass=a.mass,
                      friction=fric, restitution=rest)
    for o in (floor, cone):
        scene += o
        sim.add(o)
    pb.setGravity(0, 0, -9.81)
    pb.setTimeStep(DT)

    shape = pb.createCollisionShape(
        pb.GEOM_MESH,
        vertices=(stepper.hull_for("cone", np.asarray(cone.bounds))
                  * (r * a.k)).tolist())
    proxy = pb.createMultiBody(a.mass, shape, -1, (2.0, 0, a.drop_from),
                               (0, 0, 0, 1), useMaximalCoordinates=True)
    pb.changeDynamics(proxy, -1, contactProcessingThreshold=0,
                      lateralFriction=fric, restitution=rest)

    idx = cone.linked_objects[sim]
    zn, zp = [], []
    for _ in range(frames):
        zn.append(pb.getBasePositionAndOrientation(idx)[0][2])
        zp.append(pb.getBasePositionAndOrientation(proxy)[0][2])
        for _ in range(sub):
            pb.stepSimulation()
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("substeps/frame:", sub, " restitution:", rest, " floor:", 0.4)
    print("native z:", np.array(zn))
    print("proxy  z:", np.array(zp), "  (drawn at %.3f scale)" % a.k)
    print("native  apex after landing: %.3f   rest %.3f"
          % (max(zn[np.argmin(zn):]), zn[-1]))
    print("proxy   apex after landing: %.3f   rest %.3f"
          % (max(zp[np.argmin(zp):]), zp[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--drop-from", type=float, default=2.0,
                    help="release height; use ~0.3 to test geometry without "
                         "giving the body enough energy to topple")
    ap.add_argument("--mass", type=float, default=0.3001)
    ap.add_argument("--restitution", type=float, default=0.5624)
    ap.add_argument("--friction", type=float, default=0.4)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--k", type=float, default=0.5 ** (1.0 / 3.0))
    ap.add_argument("--bounce", action="store_true",
                    help="trace native vs proxy over a whole clip instead")
    a = ap.parse_args()

    if a.bounce:
        return bounce(a)

    import pybullet as pb
    import kubric as kb

    from physloc.injectors.identity import Fission
    from physloc.render import stepper
    from physloc.render.worker import KUBASIC

    r = float(a.scale)
    k = float(Fission.CLEAVE_SCALE)

    # --- what the mesh IS, before any physics -------------------------------
    kubasic = kb.AssetSource.from_manifest(KUBASIC)
    cone = kubasic.create(asset_id="cone", name="cone", scale=r,
                          position=(0, 0, a.drop_from), mass=1.0,
                          friction=0.5, restitution=0.2)
    print("asset bounds (unit mesh) :", np.asarray(cone.bounds).tolist())
    print("aabbox at scale %.3f     :" % r, np.asarray(cone.aabbox).tolist())
    for kind in ("sphere", "cube"):
        o = kb.Sphere(name="s") if kind == "sphere" else kb.Cube(name="c")
        print("%s bounds (unit mesh)  :" % kind,
              np.asarray(o.bounds).tolist())

    # --- the native asset, dropped ------------------------------------------
    sim = kb.simulator.PyBullet(kb.Scene(frame_start=0, frame_end=1))
    scene = kb.Scene(frame_start=0, frame_end=1)
    floor = kb.Cube(name="floor", scale=(6, 6, 0.1), position=(0, 0, -0.1),
                    static=True, friction=0.5, restitution=0.2)
    for o in (floor, cone):
        scene += o
        sim.add(o)
    pb.setGravity(0, 0, -9.81)
    pb.setTimeStep(DT)
    idx = cone.linked_objects[sim]
    z_native, v_native, b_native, q_native = _settle(pb, idx)
    print("native  rest z = %.4f  |v| = %.4f  aabb_z0 = %+.4f  q = %s"
          % (z_native, v_native, b_native, q_native))

    # --- proxies, dropped from the same height ------------------------------
    bnd = np.asarray(cone.bounds, np.float64)
    for label, verts, s in (
            ("proxy1", stepper.hull_for("cone", bnd), 1.0),
            ("proxyk", stepper.hull_for("cone", bnd), k),
            ("table1", stepper.hull_for("cone"), 1.0),
            ("sphere", stepper.unit_hull(), k)):
        half = np.full(3, r * s)
        shape = pb.createCollisionShape(
            pb.GEOM_MESH, vertices=(np.asarray(verts) * half[None, :]).tolist())
        body = pb.createMultiBody(1.0, shape, -1, (0, 0, a.drop_from), (0, 0, 0, 1),
                                  useMaximalCoordinates=True)
        pb.changeDynamics(body, -1, contactProcessingThreshold=0,
                          lateralFriction=0.5, restitution=0.2)
        z, v, b, q = _settle(pb, body)
        print("%s  rest z = %.4f  |v| = %.4f  aabb_z0 = %+.4f  q = %s"
              "   (native x scale = %.4f)"
              % (label, z, v, b, q, z_native * s))
        pb.removeBody(body)


if __name__ == "__main__":
    main()

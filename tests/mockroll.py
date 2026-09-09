"""A crude rigid-body rollout, so every cell can be smoke-tested on the host.

**Not a physics engine and not a substitute for one.** PyBullet lives in the
container and a real rollout costs a docker round trip per cell; at 48 build
cells that is most of an hour to discover that one injector has a typo. This
produces a trajectory of roughly the right shape -- things fall, land, collide
and are recorded as contacts -- which is enough to exercise every `plan()` and
`apply()` path end to end. Physical fidelity is the container's job.

Ramps are treated as axis-aligned boxes, so a block "slides" by dropping onto
the slab's bounding top. That is wrong, and deliberately so: the test asks
whether the code runs and the annotations are structurally sound, never whether
the numbers are right.
"""
from __future__ import annotations

import numpy as np

from physloc.sim.trajectory import Contacts, Trajectory

SUBSTEPS = 8


def _tops(spec):
    """Static surfaces as (top_z, cx, cy, hx, hy, seg_id, mu), highest first."""
    out = []
    for b in spec.bodies:
        if not b.static or not b.collides:
            continue
        if b.kind == "cube":
            out.append((b.position[2] + b.scale[2], b.position[0], b.position[1],
                        b.scale[0], b.scale[1], int(b.segmentation_id),
                        float(b.friction)))
        else:
            out.append((spec.floor_level, 0.0, 0.0, 1e6, 1e6,
                        int(b.segmentation_id), float(b.friction)))
    return sorted(out, key=lambda r: -r[0])


def _boxes(spec):
    """Static cubes that stand up in the world, as (centre, half-extents, id).

    Slabs the bodies rest *on* are handled by `_ground`; this is for the ones
    they run into. A cube taller than it is thin in some horizontal direction
    is a wall, not a floor.
    """
    out = []
    for b in spec.bodies:
        if not b.static or not b.collides or b.kind != "cube":
            continue
        sx, sy, sz = (float(x) for x in b.scale)
        if sz <= min(sx, sy):
            continue                       # lies flat: it is a floor
        out.append((np.asarray(b.position, np.float64),
                    np.array([sx, sy, sz]), int(b.segmentation_id)))
    return out


def _box_hit(box, point, radius):
    """Outward unit normal if the sphere overlaps the box, else (None, None)."""
    centre, half, seg = box
    d = np.asarray(point, np.float64) - centre
    clamped = np.clip(d, -half, half)
    delta = d - clamped
    dist = float(np.linalg.norm(delta))
    if dist > radius:
        return None, None
    if dist > 1e-9:
        return delta / dist, seg
    # Centre inside the box: push out along the shallowest axis.
    slack = half - np.abs(d)
    axis = int(np.argmin(slack))
    n = np.zeros(3)
    n[axis] = 1.0 if d[axis] >= 0 else -1.0
    return n, seg


def _ground(tops, x, y, floor_level):
    for top, cx, cy, hx, hy, seg, mu in tops:
        if abs(x - cx) <= hx and abs(y - cy) <= hy:
            return top, seg, mu
    return floor_level, 0, 0.6


def roll(spec, scenario=None) -> Trajectory:
    T = spec.tier.num_frames
    bodies = spec.bodies
    B = len(bodies)
    dt = 1.0 / float(spec.tier.fps)
    h = dt / SUBSTEPS
    g = np.asarray(spec.gravity, np.float64)
    tops = _tops(spec)
    boxes = _boxes(spec)

    pos = np.zeros((T, B, 3), np.float64)
    quat = np.zeros((T, B, 4), np.float64)
    lvel = np.zeros((T, B, 3), np.float64)
    avel = np.zeros((T, B, 3), np.float64)
    p = np.array([b.position for b in bodies], np.float64)
    v = np.array([b.velocity for b in bodies], np.float64)
    w = np.array([b.angular_velocity for b in bodies], np.float64)
    q = np.array([b.quaternion for b in bodies], np.float64)
    r = np.array([b.bounding_radius for b in bodies], np.float64)
    free = np.array([not b.sim_static for b in bodies], bool)
    seg = [int(b.segmentation_id) for b in bodies]

    # The scenario's own distance constraint, if it declares one. The mock has
    # no joints and no hooks, but a pivot is three lines of projection and
    # leaving it out is not a harmless approximation: without it
    # `pendulum_swing`'s bob free-falls to the floor here, so every host-side
    # test sees a ball lying on the ground where the container sees a swinging
    # pendulum -- and a family whose whole subject is the swing then "changes
    # nothing", which is what `angular_momentum` reported.
    pivot = (np.asarray(spec.notes["pivot"], np.float64)
             if spec.notes.get("constraint") == "pivot" else None)
    arm = float(spec.notes.get("arm", 0.0))
    held = [i for i, b in enumerate(bodies)
            if pivot is not None and b.role == "actor"]

    cf, ca, cb, cn, cp = [], [], [], [], []

    def note(frame, a, b_, normal, point):
        cf.append(frame)
        ca.append(a)
        cb.append(b_)
        cn.append(normal)
        cp.append(point)

    for f in range(T):
        for _ in range(SUBSTEPS):
            v[free] += g * h
            p[free] += v[free] * h
            for i in range(B):
                if not free[i]:
                    continue
                top, sid, mu_s = _ground(tops, p[i, 0], p[i, 1], spec.floor_level)
                if p[i, 2] - r[i] <= top and v[i, 2] < 0.0:
                    p[i, 2] = top + r[i]
                    v[i, 2] = -v[i, 2] * float(bodies[i].restitution)
                    if abs(v[i, 2]) < 0.06:
                        v[i, 2] = 0.0
                    note(f, seg[i], sid, [0.0, 0.0, 1.0], p[i].tolist())
                # Coulomb friction while in contact with the surface below.
                # The mock ran frictionless until a framing audit used it to ask
                # "does the actor stay in shot?", and got 0% for `rolling_ramp`
                # at the release tier because nothing here ever slowed a sliding body
                # down. Physical fidelity is still the container's job; this is
                # only enough that a body which should coast to a halt does.
                if p[i, 2] - r[i] <= top + 1e-4:
                    mu = float(bodies[i].friction) * mu_s   # PyBullet's default
                    vt = v[i, :2]
                    sp = float(np.hypot(vt[0], vt[1]))
                    if sp > 1e-9:
                        dv = mu * float(abs(g[2])) * h
                        v[i, :2] = vt * max(0.0, 1.0 - dv / sp)
                # Sphere against a static box, treated as axis-aligned. Needed
                # because a scenario whose central event the mock cannot
                # produce is a blind spot in the only test that covers every
                # cell: without it `barrier_pass` never touches its wall, and
                # `solidity` there quietly falls back to sinking through the
                # floor -- exactly the failure that scenario exists to avoid.
                for box in boxes:
                    n, sid_b = _box_hit(box, p[i], r[i])
                    if n is None:
                        continue
                    vn = float(np.dot(v[i], n))
                    if vn >= 0.0:
                        continue
                    v[i] -= (1.0 + float(bodies[i].restitution)) * vn * n
                    note(f, seg[i], sid_b, n.tolist(), p[i].tolist())
            for i in held:
                # Same rule the scenario gives the simulator: back onto the
                # sphere of radius `arm` about the pivot, radial velocity out.
                d = p[i] - pivot
                dist = float(np.linalg.norm(d))
                if dist < 1e-9:
                    continue
                n = d / dist
                p[i] = pivot + n * arm
                v[i] -= float(np.dot(v[i], n)) * n
            for i in range(B):
                for j in range(i + 1, B):
                    if not (free[i] and free[j]):
                        continue
                    d = p[j] - p[i]
                    dist = float(np.linalg.norm(d))
                    overlap = r[i] + r[j] - dist
                    if dist < 1e-6 or overlap <= 0.0:
                        continue
                    n = d / dist
                    p[i] -= n * overlap * 0.5
                    p[j] += n * overlap * 0.5
                    rel = float(np.dot(v[j] - v[i], n))
                    if rel < 0.0:
                        e = 0.5 * (bodies[i].restitution + bodies[j].restitution)
                        imp = -(1.0 + e) * rel * 0.5
                        v[i] -= n * imp
                        v[j] += n * imp
                        note(f, seg[i], seg[j], n.tolist(),
                             (p[i] + n * r[i]).tolist())
        pos[f], lvel[f], avel[f], quat[f] = p, v, w, q

    n = len(cf)
    contacts = Contacts(
        np.asarray(cf, np.int32) if n else np.zeros((0,), np.int32),
        np.asarray(ca, np.int32) if n else np.zeros((0,), np.int32),
        np.asarray(cb, np.int32) if n else np.zeros((0,), np.int32),
        np.asarray(cp, np.float32).reshape(n, 3) if n else np.zeros((0, 3), np.float32),
        np.asarray(cn, np.float32).reshape(n, 3) if n else np.zeros((0, 3), np.float32),
        np.ones((n,), np.float32), np.zeros((n,), np.float32))

    present = np.ones((T, B), bool)
    colour = np.zeros((T, B, 3), np.float32)
    for j, b in enumerate(bodies):
        if b.dormant:
            present[:, j] = False
        colour[:, j, :] = np.asarray(b.color, np.float32)

    traj = Trajectory(
        body_ids=np.asarray(seg, np.int32),
        body_names=[b.name for b in bodies],
        pos=pos.astype(np.float32), quat=quat.astype(np.float32),
        lin_vel=lvel.astype(np.float32), ang_vel=avel.astype(np.float32),
        present=present, colour=colour,
        opacity=np.ones((T, B), np.float32),
        mass=np.asarray([b.mass for b in bodies], np.float32),
        radius=r.astype(np.float32),
        is_static=np.asarray([b.static for b in bodies], bool),
        contacts=contacts, fps=float(spec.tier.fps),
        gravity=np.asarray(spec.gravity, np.float32),
        meta={"scenario": spec.scenario, "seed": spec.seed, "label": "valid",
              "spec": spec.to_dict()})
    if scenario is not None:
        scenario.script(spec, traj)
    return traj

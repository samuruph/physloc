"""A crude rigid-body rollout, so every cell can be smoke-tested on the host.

**Not a physics engine and not a substitute for one.** PyBullet lives in the
container and a real rollout costs a docker round trip per cell; at 48 build
cells that is most of an hour to discover that one injector has a typo. This
produces a trajectory of roughly the right shape -- things fall, land, collide
and are recorded as contacts -- which is enough to exercise every `plan()` and
`apply()` path end to end. Physical fidelity is the container's job.

A tilted slab -- a ramp -- is a slope: a body resting on it is pushed along its
face by gravity and slowed by friction along it. It used to be an axis-aligned
box at its bounding top, a flat plateau a block crept across and stopped on,
which quietly decided whether `rolling_ramp`'s in-flight families could plan at
all: once its block was enlarged it stopped short of the lip on every seed the
tests try. The test still asks whether the code runs and the annotations are
structurally sound, never whether the numbers are right.
"""
from __future__ import annotations

import numpy as np

from physloc.sim.trajectory import Contacts, Trajectory

SUBSTEPS = 8


def _held(spec):
    """Ids a scenario holds still, as `sim_hooks` does in the container.

    The mock has no hooks, so a held body would fall here and take whatever
    rests on it down too -- the same reason the pivot below is mirrored.
    """
    return {int(i) for i in (spec.notes.get("held_body_ids") or ())}


def _fixed(body, held) -> bool:
    return bool(body.static) or int(body.segmentation_id) in held


def _rot(q):
    w, x, y, z = (float(c) for c in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _tops(spec):
    """Static surfaces as (top_z, surface, seg_id, mu), highest first.

    `surface(x, y)` is `(z, normal)` of the top face over (x, y), or None when
    (x, y) is off it -- a plane for a tilted slab, a flat top otherwise.
    """
    out = []
    held = _held(spec)
    for b in spec.bodies:
        if not _fixed(b, held) or not b.collides:
            continue
        if b.kind == "cube":
            R = _rot(b.quaternion or (1.0, 0.0, 0.0, 0.0))
            c = np.asarray(b.position, np.float64)
            half = np.asarray(b.scale, np.float64)
            n = R[:, 2] if R[2, 2] >= 0 else -R[:, 2]
            if abs(n[2]) < 0.2:
                continue                      # a wall, handled by `_boxes`
            face = c + n * half[2]

            def surface(x, y, R=R, c=c, half=half, n=n, face=face):
                z = face[2] - (n[0] * (x - face[0]) + n[1] * (y - face[1])) / n[2]
                local = R.T @ (np.array([x, y, z]) - c)
                if abs(local[0]) > half[0] or abs(local[1]) > half[1]:
                    return None
                return float(z), n
            out.append((float(face[2]), surface, int(b.segmentation_id),
                        float(b.friction)))
        else:
            level = float(spec.floor_level)
            up = np.array([0.0, 0.0, 1.0])
            out.append((level, lambda x, y, level=level, up=up: (level, up),
                        int(b.segmentation_id), float(b.friction)))
    return sorted(out, key=lambda r: -r[0])


def _boxes(spec):
    """Static cubes that stand up in the world, as (centre, half-extents, id).

    Slabs the bodies rest *on* are handled by `_ground`; this is for the ones
    they run into. A cube taller than it is thin in some horizontal direction
    is a wall, not a floor.
    """
    out = []
    held = _held(spec)
    for b in spec.bodies:
        if not _fixed(b, held) or not b.collides or b.kind != "cube":
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


def _ground(tops, x, y, floor_level, below=np.inf):
    """(z, normal, seg_id, mu) of the highest surface under (x, y) that is not
    above `below` -- a body beneath a ramp stands on the floor, not the ramp."""
    for _top, surface, seg, mu in tops:
        hit = surface(x, y)
        if hit is not None and hit[0] <= below + 1e-6:
            return hit[0], hit[1], seg, mu
    return floor_level, np.array([0.0, 0.0, 1.0]), 0, 0.6


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
    held_ids = _held(spec)
    free = np.array([not (b.sim_static or int(b.segmentation_id) in held_ids)
                     for b in bodies], bool)
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
                top, n, sid, mu_s = _ground(tops, p[i, 0], p[i, 1],
                                            spec.floor_level,
                                            below=p[i, 2] + r[i])
                # The centre stands `r` off the face along its normal, which on
                # a slope is `r / n_z` above the face's height under it.
                lift = r[i] / max(float(n[2]), 0.2)
                vn = float(v[i] @ n)
                if p[i, 2] - lift <= top and vn < 0.0:
                    p[i, 2] = top + lift
                    v[i] -= (1.0 + float(bodies[i].restitution)) * vn * n
                    if abs(float(v[i] @ n)) < 0.06:
                        v[i] -= float(v[i] @ n) * n
                    note(f, seg[i], sid, n.tolist(), p[i].tolist())
                # Coulomb friction while in contact with the surface below.
                # The mock ran frictionless until a framing audit used it to ask
                # "does the actor stay in shot?", and got 0% for `rolling_ramp`
                # at the release tier because nothing here ever slowed a sliding body
                # down. Physical fidelity is still the container's job; this is
                # only enough that a body which should coast to a halt does.
                if p[i, 2] - lift <= top + 1e-4:
                    mu = float(bodies[i].friction) * mu_s   # PyBullet's default
                    # Along the FACE: on a slope that is the direction a body
                    # slides in, and the normal force is g's share across it.
                    vt = v[i] - float(v[i] @ n) * n
                    sp = float(np.linalg.norm(vt))
                    if sp > 1e-9:
                        dv = mu * float(abs(g[2])) * float(n[2]) * h
                        v[i] = v[i] - vt + vt * max(0.0, 1.0 - dv / sp)
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

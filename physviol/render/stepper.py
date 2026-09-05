"""Step PyBullet forward from a state we choose -- the simulated-intervention seam.

Until now every injector edited a *finished* trajectory, and everything after
`t_event` was re-integrated by our own resolver against sphere and box
approximations taken from the scene spec. That resolver does not know about
`scale_mul`, does not know that a teleported body is already past the wall, and
does not produce real contacts. So a clip could show a ball bouncing off a
barrier it had been moved beyond, or two bodies "colliding" without touching.

Kubric's `simulator.run(frame_start, frame_end)` continues from whatever state
the physics client is currently in rather than resetting it, so the honest fix
was available all along: put the world back to the valid state at `t_event`,
make the intervention a change the *simulator* honours, and let PyBullet run.
Prefix identity is untouched, because frames before `t_event` are copied from
the valid rollout verbatim.

This module is the loop. It exists rather than calling `simulator.run` because
`run` has no per-step hook, and a per-body gravity change -- `antigravity`,
`support` -- is a counter-force that must be applied on every substep.

Container-side only: it imports nothing, but it is handed a live Kubric
simulator.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

#: Signature of a per-step hook: `hook(physics_client, step, frame)`.
Hook = Callable[[object, int, int], None]


def steps_per_frame(scene) -> int:
    return max(int(scene.step_rate) // max(int(scene.frame_rate), 1), 1)


def substeps_of(simulator, default: int = 20) -> int:
    """Substeps per frame, asked of a live simulator rather than a scene.

    A staged hook is handed `(client, step, frame)` and nothing else, so a hook
    that wants to act several times *within* a frame has to know how long a
    frame is. Kubric's simulator keeps its scene, and the fallback is the rate
    the worker builds every scene at.
    """
    scene = getattr(simulator, "scene", None)
    if scene is None:
        return int(default)
    try:
        return steps_per_frame(scene)
    except Exception:                                         # noqa: BLE001
        return int(default)


def reset_to(spec, objs, traj, frame: int) -> None:
    """Put every dynamic body back to its state at `frame`.

    Kubric wires `position`, `quaternion`, `velocity` and `angular_velocity`
    through to `resetBasePositionAndOrientation` and `resetBaseVelocity`, and
    is careful to restore velocities after the pose reset zeroes them -- so
    assigning the four attributes is a complete state restore.

    Called before every variant, so one variant's staged intervention cannot
    leak into the next. That is the same discipline `worker._clear_animation`
    enforces on the render side, and for the same reason: this scene is reused.
    """
    for j, body in enumerate(spec.bodies):
        if body.static:
            continue
        obj = objs[body.name]
        obj.position = tuple(float(x) for x in traj.pos[frame, j])
        obj.quaternion = tuple(float(x) for x in traj.quat[frame, j])
        obj.velocity = tuple(float(x) for x in traj.lin_vel[frame, j])
        obj.angular_velocity = tuple(float(x) for x in traj.ang_vel[frame, j])


def _body_index_map(simulator, objs, order) -> Dict[int, str]:
    """PyBullet body id -> our body name.

    `asset.linked_objects[simulator]` is the direct route and is what the newer
    Kubric documents; the installed one does not always populate it, so fall
    back to walking the client's bodies through `_obj_idx_to_asset`.
    """
    out: Dict[int, str] = {}
    for name in order:
        obj = objs[name]
        idx = getattr(obj, "linked_objects", {}).get(simulator)
        if isinstance(idx, int):
            out[idx] = name
    if out:
        return out

    import pybullet as pc

    asset_to_name = {id(objs[n]): n for n in order}
    for i in range(pc.getNumBodies()):
        idx = pc.getBodyUniqueId(i)
        try:
            asset = simulator._obj_idx_to_asset(idx)
        except Exception:                                     # noqa: BLE001
            continue
        name = asset_to_name.get(id(asset))
        if name is not None:
            out[idx] = name
    return out


def pybullet_index(simulator, objs, spec, seg_id: int):
    """PyBullet body id for a declared `segmentation_id`, or None.

    The bridge a staged intervention needs: our ids are the ones the taxonomy
    and the masks speak, PyBullet's are whatever the client assigned.
    """
    body = next((b for b in spec.bodies
                 if int(b.segmentation_id) == int(seg_id)), None)
    if body is None:
        return None
    order = [b.name for b in spec.bodies]
    for idx, name in _body_index_map(simulator, objs, order).items():
        if name == body.name:
            return idx
    return None


def run_from(simulator, scene, spec, objs, t0: int, t_end: int,
             hooks: Sequence[Hook] = ()) -> Dict[str, np.ndarray]:
    """Simulate frames `t0..t_end` inclusive from the client's current state.

    Returns per-frame poses and the contacts actually detected, with contact
    force **averaged over the substeps of its frame** rather than summed -- the
    same correction `worker.simulate` makes, and for the same reason: one row
    per substep collapsed onto one integer frame inflates a resting body's
    contact force twentyfold.
    """
    # The INSTALLED Kubric (2022.4.1) keeps a bare connection id in
    # `physics_client` and calls the `pybullet` module directly; the newer
    # reference checkout wraps it in a `_BulletClient` at `_physics_client`.
    # The image is the API authority, and with one DIRECT connection the module
    # functions address it, so go through the module and work on both.
    import pybullet as pc

    spf = steps_per_frame(scene)
    n_frames = t_end - t0 + 1
    order = [b.name for b in spec.bodies]
    idx_of = _body_index_map(simulator, objs, order)
    seg_of = {b.name: int(b.segmentation_id) for b in spec.bodies}

    pos = np.zeros((n_frames, len(order), 3), np.float32)
    quat = np.zeros((n_frames, len(order), 4), np.float32)
    lvel = np.zeros((n_frames, len(order), 3), np.float32)
    avel = np.zeros((n_frames, len(order), 3), np.float32)
    cf: List[int] = []
    ca: List[int] = []
    cb: List[int] = []
    cp: List[Sequence[float]] = []
    cn: List[Sequence[float]] = []
    ci: List[float] = []

    for step in range(n_frames * spf):
        frame = t0 + step // spf

        for hook in hooks:
            hook(pc, step, frame)

        # RE-READ the mapping after the hooks have run, every substep. A hook
        # may have changed which PyBullet body stands for one of ours --
        # `ShapeSwap` does exactly that, because resizing a collision shape
        # means replacing the body that carries it, and it does so several
        # times within a frame. Built once outside the loop, the map went stale
        # the instant a body was resized and every pose and contact belonging to
        # the resized actor was silently dropped. A dict of five entries per
        # substep does not show up against a step of the solver.
        idx_of = _body_index_map(simulator, objs, order)

        # NOT on the first substep. `getContactPoints` returns whatever the
        # last `stepSimulation` computed, and on step 0 that call belongs to a
        # different run -- the valid rollout's final frame, or the previous
        # variant's. `reset_to` moves the bodies but does not recompute the
        # manifold, so the stale one is read out and stamped onto `t_event`.
        #
        # It cost a whole family. `drop x angular_momentum` fires at frame 3
        # with the actor in free fall; the leftover manifold said it was resting
        # on the floor, and `laws.angular_momentum` gates out every frame within
        # one of a contact -- which is exactly the frame the spin changes on. The
        # residual came out identically zero and the severity map with it.
        for contact in (pc.getContactPoints() if step else ()):
            (_flag, body_a, body_b, _la, _lb, _pa, position_b, normal_b,
             _dist, normal_force, *_rest) = contact
            if normal_force <= 1e-6:
                continue
            a_name = idx_of.get(body_a)
            b_name = idx_of.get(body_b)
            if a_name is None or b_name is None:
                continue
            cf.append(frame)
            # Kubric's own reader swaps the pair so the stored normal points
            # from body_a towards body_b; keep that convention so the residual
            # laws see the same thing whichever path produced the trajectory.
            ca.append(seg_of[b_name])
            cb.append(seg_of[a_name])
            cp.append(position_b)
            cn.append(normal_b)
            ci.append(float(normal_force) / spf)

        if step % spf == 0:
            f = step // spf
            for obj_idx, name in idx_of.items():
                j = order.index(name)
                p, q = simulator.get_position_and_rotation(obj_idx)
                v, w = simulator.get_velocities(obj_idx)
                pos[f, j] = p
                quat[f, j] = q
                lvel[f, j] = v
                avel[f, j] = w
        pc.stepSimulation()

    n = len(cf)
    return {
        "pos": pos, "quat": quat, "lin_vel": lvel, "ang_vel": avel,
        "contact_frame": np.asarray(cf, np.int32),
        "contact_a": np.asarray(ca, np.int32),
        "contact_b": np.asarray(cb, np.int32),
        "contact_point": (np.asarray(cp, np.float32).reshape(n, 3) if n
                          else np.zeros((0, 3), np.float32)),
        "contact_normal": (np.asarray(cn, np.float32).reshape(n, 3) if n
                           else np.zeros((0, 3), np.float32)),
        "contact_impulse": np.asarray(ci, np.float32),
    }


def splice(traj_valid, tail: Dict[str, np.ndarray], t0: int):
    """Valid prefix + simulated tail, as one trajectory.

    Frames before `t0` are copied verbatim, which is what makes prefix identity
    hold by construction rather than by luck. Contacts are taken from the tail
    for `>= t0` and from the valid rollout before it, so the contact list
    describes the trajectory it is attached to.
    """
    import copy as _copy

    from ..sim.trajectory import Contacts

    out = _copy.copy(traj_valid)
    for key in ("pos", "quat", "lin_vel", "ang_vel"):
        arr = np.asarray(getattr(traj_valid, key)).copy()
        arr[t0:] = tail[key][: arr.shape[0] - t0]
        setattr(out, key, arr)
    for key in ("present", "scale_mul", "colour", "opacity"):
        setattr(out, key, np.asarray(getattr(traj_valid, key)).copy())

    old = traj_valid.contacts
    keep = np.asarray(old.frame, int) < t0 if len(old) else np.zeros((0,), bool)
    out.contacts = Contacts(
        np.concatenate([np.asarray(old.frame, np.int32)[keep], tail["contact_frame"]]),
        np.concatenate([np.asarray(old.body_a, np.int32)[keep], tail["contact_a"]]),
        np.concatenate([np.asarray(old.body_b, np.int32)[keep], tail["contact_b"]]),
        np.concatenate([np.asarray(old.point, np.float32)[keep], tail["contact_point"]]),
        np.concatenate([np.asarray(old.normal, np.float32)[keep], tail["contact_normal"]]),
        np.concatenate([np.asarray(old.impulse, np.float32)[keep], tail["contact_impulse"]]),
        np.zeros((int(keep.sum()) + len(tail["contact_frame"]),), np.float32))
    out.meta = dict(traj_valid.meta)
    out.meta["contacts"] = "simulated"
    return out


# ------------------------------------------------------------- resizing --
#: Where a parked body waits while a proxy stands in for it. Far outside any
#: scenario's frustum and far below its floor, so it can neither be seen nor
#: touch anything even with its collision mask still on.
PARKED_Z = -2000.0

_HULL = None


def unit_hull(subdivisions: int = 2) -> np.ndarray:
    """Vertices of a unit sphere, for building an ellipsoid collision shape.

    PyBullet has a box and a sphere and nothing in between: `GEOM_SPHERE` takes
    one radius, so a body squashed along one axis has no primitive to be. It
    does accept an explicit vertex list for a convex hull, and an icosphere
    scaled per axis is exactly an ellipsoid -- measured in the pinned image, a
    0.3 x 0.3 x 0.6 hull dropped on a plane comes to rest at z = 0.601.

    Cached, because the vertices depend on nothing.
    """
    global _HULL
    if _HULL is not None:
        return _HULL
    t = (1.0 + 5.0 ** 0.5) / 2.0
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], np.float64)
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(int(subdivisions)):
        out, cache = [], {}
        for tri in faces:
            mids = []
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (min(a, b), max(a, b))
                if key not in cache:
                    v = np.vstack([v, (v[a] + v[b]) / 2.0])
                    cache[key] = len(v) - 1
                mids.append(cache[key])
            a, b, c = tri
            ab, bc, ca = mids
            out += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = out
    _HULL = v / np.linalg.norm(v, axis=1, keepdims=True)
    return _HULL


class ShapeSwap:
    """Change a body's SIZE in the simulator while the simulation is running.

    The one thing PyBullet genuinely cannot do in place: there is no API to
    rescale a collision shape, which is why `deformation` and `immutability`
    were the two families that edited a finished trajectory instead of staging
    themselves. That cost exactly what the trajectory path always costs -- the
    render knew the body had changed size and the physics did not, so a
    shrunken cube stopped touching the floor it was standing on and a swollen
    ball grew half-way into the barrier beside it.

    What PyBullet *can* do is create a new body and remove an old one, mid-run,
    carrying the pose and both velocities across. So this parks the declared
    body out of the world and stands a correctly-sized **proxy** in its place,
    swapping the proxy again on each frame of a resize ramp. From `t_event` on
    the actor's contacts, its resting height and everything it runs into come
    from the solver at its *current* size.

    The declared body is parked rather than removed because Kubric's traitlet
    setters close over its PyBullet index at scene-build time: remove it and the
    next `obj.position = ...` writes to a body that no longer exists. Parking
    keeps every one of those setters valid, and `linked_objects` -- a plain
    dict, checked in the image -- is retargeted so that `_body_index_map`, and
    therefore the recorded poses and contacts, follow the proxy.

    `restore()` is **not optional**, for the reason every `unstage` is not: one
    scene serves every family in a run.
    """

    def __init__(self, simulator, objs, spec, body, dynamic: bool = False):
        self.simulator = simulator
        self.spec = spec
        self.body = body
        self.obj = objs[body.name]
        # `dynamic` overrides the declared `sim_static`. `fission`'s understudy
        # is declared scripted, so the simulator holds it at mass 0 and
        # `changeDynamics(mass=...)` does not bring it back to life -- measured
        # in the pinned image, a body created with mass 0 and
        # `useMaximalCoordinates` stays put however its mass is later changed.
        # Standing a genuinely dynamic proxy in its place is what makes the
        # second half of a split a body the solver actually moves.
        self.dynamic = bool(dynamic)
        self.original = pybullet_index(simulator, objs, spec,
                                       int(body.segmentation_id))
        self.proxy = None
        self._scale = None

    @property
    def ok(self) -> bool:
        return self.original is not None

    @property
    def live(self):
        """The PyBullet body currently standing for this one."""
        return self.original if self.proxy is None else self.proxy

    def set_scale(self, scale, pose=None, velocity=None) -> None:
        """Give the body a collision shape `scale` times its declared size.

        `pose` and `velocity` override what the outgoing body was doing, for the
        case where the proxy is not continuing that body's motion at all -- a
        `fission` understudy starts from the *original's* pose, not from the
        parking spot it has been sitting in since frame 0.
        """
        import pybullet as pb

        if not self.ok:
            return
        scale = np.asarray(scale, np.float64).reshape(3)
        if (self._scale is not None and pose is None
                and float(np.abs(scale - self._scale).max()) < 1e-4):
            return
        if pose is None:
            pos, quat = pb.getBasePositionAndOrientation(self.live)
        else:
            pos, quat = pose
        if velocity is None:
            vel, ang = pb.getBaseVelocity(self.live)
        else:
            vel, ang = velocity

        half = np.asarray(self.body.scale, np.float64) * scale
        if self.body.kind == "cube":
            shape = pb.createCollisionShape(pb.GEOM_BOX,
                                            halfExtents=half.tolist())
        else:
            shape = pb.createCollisionShape(
                pb.GEOM_MESH, vertices=(unit_hull() * half[None, :]).tolist())

        if self.proxy is None:
            self._park()
        else:
            pb.removeBody(self.proxy)

        mass = (float(self.body.mass)
                if (self.dynamic or not self.body.sim_static) else 0.0)
        self.proxy = pb.createMultiBody(mass, shape, -1, pos, quat,
                                        useMaximalCoordinates=True)
        pb.changeDynamics(self.proxy, -1, contactProcessingThreshold=0,
                          lateralFriction=float(self.body.friction),
                          restitution=float(self.body.restitution))
        pb.resetBaseVelocity(self.proxy, list(vel), list(ang))
        self.obj.linked_objects[self.simulator] = self.proxy
        self._scale = scale

    def restore(self) -> None:
        """Put the declared body back, wherever the proxy left off."""
        import pybullet as pb

        if not self.ok or self.proxy is None:
            return
        pos, quat = pb.getBasePositionAndOrientation(self.proxy)
        vel, ang = pb.getBaseVelocity(self.proxy)
        pb.removeBody(self.proxy)
        self.proxy = None
        self._scale = None
        pb.setCollisionFilterGroupMask(self.original, -1, 1, 1)
        pb.changeDynamics(self.original, -1,
                          mass=0.0 if self.body.sim_static
                          else float(self.body.mass))
        pb.resetBasePositionAndOrientation(self.original, pos, quat)
        pb.resetBaseVelocity(self.original, list(vel), list(ang))
        self.obj.linked_objects[self.simulator] = self.original

    def _park(self) -> None:
        import pybullet as pb

        pb.setCollisionFilterGroupMask(self.original, -1, 0, 0)
        pb.changeDynamics(self.original, -1, mass=0.0)
        pb.resetBasePositionAndOrientation(
            self.original, (0.0, 0.0, PARKED_Z), (0.0, 0.0, 0.0, 1.0))
        pb.resetBaseVelocity(self.original, [0.0] * 3, [0.0] * 3)


class Vanish:
    """Take a body out of the simulation, and put it back.

    What `permanence` and `dissolve` actually claim: the body is not there. A
    body that is not there does not hold anything up and does not strike
    anything, and those consequences belong to the solver rather than to a
    correction applied afterwards.

    Both families used to edit the trajectory and then repair the scene by hand
    -- `_settle_bystanders` re-integrating whatever the vanished body would have
    struck. That produces a different answer from the simulator's, and on
    `pyramid_impact` you could see the difference: `dissolve` displaced three
    spheres by 0.12 to 0.49 m through the hand correction while the staged
    families displaced a different set by different amounts. Same scene, same
    claim, two mechanisms.

    Parked rather than removed, for the reason `ShapeSwap` parks: Kubric's
    traitlet setters close over the body's index, so removing it makes the next
    `obj.position = ...` write to a body that no longer exists.
    """

    def __init__(self, simulator, objs, spec, body):
        self.simulator = simulator
        self.spec = spec
        self.body = body
        self.idx = pybullet_index(simulator, objs, spec,
                                  int(body.segmentation_id))
        self.saved = None

    @property
    def ok(self) -> bool:
        return self.idx is not None

    def hide(self) -> None:
        """Remove it from the world, remembering where it was."""
        import pybullet as pb

        if not self.ok or self.saved is not None:
            return
        pos, quat = pb.getBasePositionAndOrientation(self.idx)
        vel, ang = pb.getBaseVelocity(self.idx)
        self.saved = (pos, quat, vel, ang)
        pb.setCollisionFilterGroupMask(self.idx, -1, 0, 0)
        pb.changeDynamics(self.idx, -1, mass=0.0)
        pb.resetBasePositionAndOrientation(self.idx, (0.0, 0.0, PARKED_Z), quat)
        pb.resetBaseVelocity(self.idx, [0.0] * 3, [0.0] * 3)

    def show(self) -> None:
        """Put it back exactly as it left.

        Where it left, not where its lawful twin would be by now: a body that
        blinks out and returns somewhere else has teleported, and that is a
        different family.
        """
        import pybullet as pb

        if not self.ok or self.saved is None:
            return
        pos, quat, vel, ang = self.saved
        self.saved = None
        pb.setCollisionFilterGroupMask(self.idx, -1, 1, 1)
        pb.changeDynamics(self.idx, -1,
                          mass=0.0 if self.body.sim_static
                          else float(self.body.mass))
        pb.resetBasePositionAndOrientation(self.idx, pos, quat)
        pb.resetBaseVelocity(self.idx, list(vel), list(ang))

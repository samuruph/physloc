"""Mechanical energy from the trajectory seam -- docs/energy.md.

Exact, not estimated: every term comes straight off `traj.npz` and the scene
spec. Runs host-side, needs no renderer and no simulator.

The point of the module is the two *anomaly* channels, which are what a
benchmark scores against. A passive rigid-body scene under gravity, contacts and
friction can only lose mechanical energy, and only at a contact -- so energy
that moves on a contact-free frame, or moves at a contact by more than
restitution and friction permit, is unphysical by construction rather than by a
tuned threshold. Measured floor on valid clips: 0.005% of E0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..injectors import _geom

#: Frames on either side of a reported contact that still count as "at contact".
#: A step's energy change is attributable to a contact at either of its ends, and
#: PyBullet reports the frame it *noticed* a contact, which can lag the touch.
CONTACT_SLACK = 1


@dataclass
class EnergyTrace:
    """Per-frame mechanical energy and its two anomaly channels."""

    total: np.ndarray                 # [T]  joules
    kinetic_translational: np.ndarray  # [T]
    kinetic_rotational: np.ndarray    # [T]
    potential: np.ndarray             # [T]
    by_body: np.ndarray               # [T, B]
    body_ids: np.ndarray              # [B]
    dissipated: np.ndarray            # [T]  cumulative E0 - E(t), >= 0 lawfully
    in_frame: np.ndarray              # [T, B] bool -- body the camera can SEE
    energy_in_frame: np.ndarray       # [T]  the total over visible bodies only
    free_anomaly: np.ndarray          # [T]  |dE| on contact-free frames, / E0
    contact_anomaly: np.ndarray       # [T]  energy GAINED at a contact, / E0
    excess_loss: np.ndarray           # [T]  energy lost beyond the dissipation
                                      #      budget, / E0

    def to_npz(self) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in self.__dict__.items():
            if k == "body_ids":
                out[k] = np.asarray(v, np.int32)
            elif k == "in_frame":
                out[k] = np.asarray(v, bool)
            else:
                out[k] = np.asarray(v, np.float32)
        return out

    def summary(self) -> Dict[str, float]:
        e0 = float(self.total[0])
        denom = max(abs(e0), 1e-9)
        return {
            "E0": e0, "E_end": float(self.total[-1]),
            "total_dissipated": float(self.dissipated[-1]),
            "total_dissipated_fraction": float(self.dissipated[-1] / denom),
            "peak_free_anomaly": float(self.free_anomaly.max()),
            "peak_contact_anomaly": float(self.contact_anomaly.max()),
            "peak_excess_loss": float(self.excess_loss.max()),
            "E_end_in_frame": float(self.energy_in_frame[-1]),
            "frames_all_bodies_in_frame": int(self.in_frame.all(axis=1).sum()),
        }


def inertia_diag(kind: str, scale, mass: float) -> np.ndarray:
    """Body-frame principal moments for the primitives this project stages.

    `scale` is the half-extent triple Kubric uses, so a cube of `scale=(h,h,h)`
    has side `2h` and a sphere of `scale=(r,r,r)` has radius `r`.
    """
    s = np.asarray(scale, np.float64).reshape(3)
    if kind == "sphere":
        r = float(s.mean())
        return np.full(3, 0.4 * mass * r * r)
    if kind in ("cube", "box"):
        a, b, c = s
        return mass / 3.0 * np.array([b * b + c * c, a * a + c * c,
                                      a * a + b * b])
    if kind == "cylinder":                 # axis = z
        r, h = float(s[0]), float(s[2])
        return np.array([mass * (3 * r * r + 4 * h * h) / 12.0,
                         mass * (3 * r * r + 4 * h * h) / 12.0,
                         0.5 * mass * r * r])
    if kind == "cone":                     # axis = z, apex up, half-height h
        r, h = float(s[0]), float(s[2])
        # Standard solid cone about its centre of mass, which sits a quarter of
        # the way up from the base.
        lat = mass * (3.0 / 20.0 * r * r + 3.0 / 80.0 * (2.0 * h) ** 2)
        return np.array([lat, lat, 0.3 * mass * r * r])
    if kind == "torus":                    # axis = z
        # KuBasic's torus has an outer radius of `scale` and a tube radius of
        # about a third of it. Close enough: the energy law reads this as a
        # ratio between rotational and linear terms, not as a spec sheet.
        R, a = float(s[0]) * 0.75, float(s[0]) * 0.25
        lat = mass * (5.0 / 8.0 * a * a + 0.5 * R * R)
        return np.array([lat, lat, mass * (0.75 * a * a + R * R)])
    r = float(s.mean())                    # anything else: solid-sphere stand-in
    return np.full(3, 0.4 * mass * r * r)


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    """[T,4] (w,x,y,z), Kubric's order -> [T,3,3] rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    n = np.where(n < 1e-12, 1.0, n)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def _is_shadow(body) -> bool:
    """A cast shadow is a picture, not a body, and carries no energy.

    `shadow_track` hands the cast shadow to a scripted stand-in body so it can
    have a segmentation id and therefore a mask -- see that scenario's
    docstring. The consequence nobody chased down is that the stand-in is
    declared with a mass like anything else, so it contributed a constant 0.33 J
    to every clip in the scenario, and MOVING it moved the scene's energy:
    `shadow x strong` slid the shadow a body-width over two frames and the
    total jumped from 4.12 J to 54.63 J. A shadow is where light is not. It has
    no mass, no momentum and no energy, and the energy channel must say so --
    otherwise the three optical families read as the most violent energy
    violations in the dataset while the object they are about never moves.

    A violation OF the shadow is still fully annotated: it has a mask, a
    severity and its own residual law. It just does not have joules.
    """
    return str(getattr(body, "role", "")) == "shadow"


def contact_frames(traj, slack: int = CONTACT_SLACK) -> np.ndarray:
    """[T] bool -- frames on or beside a reported contact."""
    T = traj.num_frames
    out = np.zeros(T, bool)
    c = traj.contacts
    if len(c):
        idx = np.clip(np.asarray(c.frame, int), 0, T - 1)
        out[idx] = True
    # Padded, never `np.roll` -- rolling wraps the LAST frame's contact onto
    # frame 0, which on a clip that ends at rest marks the opening frame as
    # "in contact" and hides any free-energy anomaly there.
    #
    # The widening is not cosmetic: PyBullet labels a contact with the frame at
    # the START of the interval it occurred in, so the energy a collision
    # removes lands on the step *after* the reported frame. Measured on `drop`:
    # contact reported at frame 8, ball still 0.13 m clear at frame 8, energy
    # drops between 8 and 9.
    if slack > 0:
        pad = np.concatenate([np.zeros(slack, bool), out, np.zeros(slack, bool)])
        wide = pad.copy()
        for k in range(1, slack + 1):
            wide[k:] |= pad[:-k]
            wide[:-k] |= pad[k:]
        out = wide[slack:slack + T]
    return out


def compute(traj, spec, floor_level: Optional[float] = None,
            seg: Optional[np.ndarray] = None) -> EnergyTrace:
    """Mechanical energy of every dynamic body, frame by frame.

    `seg` is the rendered segmentation, `[T, H, W]` of declared ids. Pass it
    whenever there is one: it is what decides `in_frame`, and the frustum test
    that stands in for it when there is not is a strictly weaker answer -- see
    the note beside `visible` below.
    """
    bodies = {b.name: b for b in spec.bodies}
    g = np.asarray(traj.gravity, np.float64)
    g_norm = float(np.linalg.norm(g))
    g_hat = g / max(g_norm, 1e-12)
    datum = float(getattr(spec, "floor_level", 0.0)
                  if floor_level is None else floor_level)

    T, B = traj.num_frames, traj.num_bodies
    ke_t = np.zeros(T); ke_r = np.zeros(T); pe = np.zeros(T)
    by_body = np.zeros((T, B))
    ke_body = np.zeros((T, B)); pe_body = np.zeros((T, B))
    present = np.zeros((T, B), bool)

    for j, name in enumerate(traj.body_names):
        body = bodies.get(name)
        if body is None or body.static or _is_shadow(body):
            continue
        m0 = float(getattr(body, "mass", 1.0))
        # Mass follows volume -- see docs/energy.md. Volume-preserving squash
        # leaves this at exactly 1, uniform growth does not, which is the
        # taxonomy's mass-vs-shape split falling out of the arithmetic.
        vol = np.prod(np.asarray(traj.scale_mul[:, j, :], np.float64), axis=1)
        m = m0 * np.clip(vol, 1e-9, None)

        v = np.asarray(traj.lin_vel[:, j, :], np.float64)
        w = np.asarray(traj.ang_vel[:, j, :], np.float64)
        p = np.asarray(traj.pos[:, j, :], np.float64)
        # HOW MUCH OF THE BODY IS STILL THERE, not merely whether it is. A
        # dissolving body loses its matter over several frames, and `present`
        # is a single flip at the end of the fade -- so the energy fell off a
        # cliff on one frame however slowly the body faded, and the weak bin
        # (an eleven-frame fade) produced exactly the same step as the strong
        # (four frames), just later. Weighting by opacity makes the trace
        # follow the intervention's own profile, which is what makes the three
        # bins of `dissolve` distinguishable in the energy channel at all.
        #
        # Everything else is unaffected: only `dissolve` writes opacity, and
        # `permanence` still steps, because a cut IS a step.
        here = (np.asarray(traj.present[:, j], bool).astype(np.float64)
                * np.clip(np.asarray(traj.opacity[:, j], np.float64), 0.0, 1.0))

        # Inertia scales as m*r^2, so a resized body's tensor moves with the
        # cube of the linear factor times the mass factor.
        lin = np.asarray(traj.scale_mul[:, j, :], np.float64)
        I0 = inertia_diag(body.kind, getattr(body, "scale", (0.5,) * 3), 1.0)
        I_body = m[:, None] * I0[None, :] * lin ** 2
        R = _quat_matrix(np.asarray(traj.quat[:, j, :], np.float64))
        w_body = np.einsum("tji,tj->ti", R, w)      # world -> body frame

        e_kt = 0.5 * m * (v ** 2).sum(1)
        e_kr = 0.5 * (I_body * w_body ** 2).sum(1)
        e_pe = m * g_norm * (-(p @ g_hat) - datum)

        ke_t += e_kt * here
        ke_r += e_kr * here
        pe += e_pe * here
        by_body[:, j] = (e_kt + e_kr + e_pe) * here
        ke_body[:, j] = (e_kt + e_kr) * here
        pe_body[:, j] = e_pe * here
        present[:, j] = here > 0

    total = ke_t + ke_r + pe

    # WHERE THE EVIDENCE IS. Energy is computed from the trajectory, not from
    # the pixels, so a body that has left the view still has a real energy and a
    # superelastic bounce really does add that much -- the jump is physically
    # correct. What was missing is any way to tell that the evidence has gone
    # off screen. `energy_in_frame` sums only the bodies a camera can see, so a
    # consumer can score what is visible while `total` stays consistent with the
    # trajectory a consumer could recompute it from.
    #
    # THE PIXELS DECIDE THIS, not the frustum. A frustum test answers "is it in
    # shot", and the question is "can the camera see it" -- which differs by
    # every occluder in the scene, starting with the ground. On `pour` x
    # `solidity` the grains drop through the floor and are hidden by it from
    # frame 8, while the frustum keeps calling them visible until frame 13
    # because falling out of the world takes a while to leave the bottom of the
    # shot. So the energy curve went on accounting for the whole pour for five
    # frames after nobody could see any of it -- reported as the energy lagging
    # the disappearance. Every other occluder in the dataset has the same
    # shape, and `occluder_pass` stages one on purpose.
    #
    # The segmentation is the exact answer and it is already rendered. The
    # frustum stays as the fallback for callers that have no render yet -- the
    # mock rollout in `tests/`, and `Injector` fit ladders, which run before
    # anything is drawn.
    visible = np.zeros((T, B), bool)
    if seg is not None:
        seg_arr = np.asarray(seg)
        ids = [int(x) for x in np.asarray(traj.body_ids)]
        for j in range(B):
            n = min(T, seg_arr.shape[0])
            visible[:n, j] = (seg_arr[:n] == ids[j]).any(axis=(1, 2))
    else:
        for j in range(B):
            try:
                visible[:, j] = _geom.in_frame(
                    spec, np.asarray(traj.pos[:, j, :], np.float64))
            except Exception:                                 # noqa: BLE001
                visible[:, j] = True
    visible &= np.asarray(traj.present, bool)
    energy_visible = (by_body * visible).sum(axis=1)

    denom = max(abs(float(total[0])), 1e-9)
    dE = np.diff(total, prepend=total[0])
    free = ~contact_frames(traj)

    free_anom = np.where(free, np.abs(dE), 0.0) / denom
    # At a contact, a *gain* is unconditionally unphysical -- no budget can
    # justify one, because contacts and friction only ever remove mechanical
    # energy. So the gain side needs no model at all.
    contact_anom = np.where(~free, np.maximum(dE, 0.0), 0.0) / denom

    # The loss side does need a budget, and the honest one is simple: a contact
    # can dissipate at most the kinetic energy actually available that frame. A
    # perfectly inelastic stop takes all of it and no more. Potential energy
    # cannot be dissipated at all -- it can only be converted by the body
    # moving, which the balance already accounts for.
    #
    # This is deliberately *not* tuned to catch everything. `newton1_inertia`
    # halts a body on the ground, which costs exactly its kinetic energy and is
    # therefore within budget: a body stopping dead is energetically identical
    # to one that hit something. That family is a momentum violation and
    # `linear_momentum` is what scores it. What this catches is energy leaving
    # with no carrier at all -- `permanence` and `dissolve` taking a body's
    # potential energy with it.
    #
    # Two terms, and both were learned by frame-by-frame audit of a lawful clip:
    #
    # 1. The kinetic energy available at the START of the step, not at its end.
    #    Using the end reads 77% excess loss on a perfectly lawful `drop`: at
    #    the impact frame the ball has already stopped, so its remaining kinetic
    #    energy is nearly zero and every joule the floor legitimately absorbed
    #    looks unexplained.
    #
    # 2. Potential energy RELEASED during the step. A body in contact can settle
    #    downward and have that energy absorbed immediately, within the frame.
    #    On `drop` the ball sinks 4.7 cm into the solver's contact tolerance as
    #    it comes to rest and loses 1.310 J while holding only 0.850 J of
    #    kinetic energy -- the missing 0.461 J is exactly m*g*0.047. Without
    #    this term that settling reads as a 1.7% violation on every clip that
    #    ends at rest, which is most of them.
    #
    # The allowance is per body and requires the body to still EXIST at both
    # ends of the step. That is what keeps `permanence` and `dissolve`
    # detectable: a body that vanishes takes its potential energy with it, and
    # nothing absorbed it.
    excess = np.zeros(T)
    for j in range(B):
        e_j = by_body[:, j]
        d_j = np.diff(e_j, prepend=e_j[0])
        ke_before = np.concatenate([ke_body[:1, j], ke_body[:-1, j]])
        pe_before = np.concatenate([pe_body[:1, j], pe_body[:-1, j]])
        released = np.maximum(pe_before - pe_body[:, j], 0.0)
        survived = np.concatenate([present[:1, j], present[:-1, j]]) & present[:, j]
        budget = ke_before + np.where(survived, released, 0.0)
        excess += np.maximum(-d_j - budget, 0.0)
    excess = excess / denom

    return EnergyTrace(
        total=total, kinetic_translational=ke_t, kinetic_rotational=ke_r,
        potential=pe, by_body=by_body,
        body_ids=np.asarray(traj.body_ids, np.int32),
        dissipated=total[0] - total,
        in_frame=visible, energy_in_frame=energy_visible,
        free_anomaly=free_anom, contact_anomaly=contact_anom,
        excess_loss=excess)


def per_body_free_anomaly(traj, spec) -> np.ndarray:
    """[T, B] -- each body's own energy change on frames where it has no contact.

    The per-body counterpart of `free_anomaly`, and the only one of the three
    channels that is honestly attributable to a single body. A body's energy
    may legitimately jump at a contact, because a partner supplied it; with
    nothing touching it, only gravity acts and gravity is already in the
    balance. So "this body's energy moved while nothing was touching it" needs
    no budget and no partner bookkeeping.

    Narrower than the scene-level trace by design. `permanence` and `dissolve`
    take a resting body's potential energy with it, which is a scene-level
    excess loss and not a free-energy event; `energy.npz` carries those and this
    law does not claim them.
    """
    trace = compute(traj, spec)
    T, B = traj.num_frames, traj.num_bodies
    out = np.zeros((T, B))
    denom = max(abs(float(trace.total[0])), 1e-9)
    c = traj.contacts
    for j in range(B):
        touched = np.zeros(T, bool)
        if len(c):
            bid = int(traj.body_ids[j])
            m = (np.asarray(c.body_a, int) == bid) | (np.asarray(c.body_b, int) == bid)
            if m.any():
                touched[np.clip(np.asarray(c.frame, int)[m], 0, T - 1)] = True
        for k in range(1, CONTACT_SLACK + 1):
            touched |= np.roll(touched, k) | np.roll(touched, -k)
        d = np.diff(trace.by_body[:, j], prepend=trace.by_body[0, j])
        out[:, j] = np.where(~touched, np.abs(d), 0.0) / denom
    return out


def energy_map(trace: EnergyTrace, seg: np.ndarray) -> np.ndarray:
    """[T, H, W] -- each body's energy painted onto its own pixels.

    The same mechanism `severity_map` uses, and the same honest limit: energy is
    a per-body quantity, so it is constant inside a rigid body's silhouette.
    Resolving it further would invent structure the physics does not have.
    """
    T = trace.by_body.shape[0]
    out = np.zeros((T,) + seg.shape[1:], np.float32)
    denom = max(abs(float(trace.total[0])), 1e-9)
    for j, bid in enumerate(np.asarray(trace.body_ids, int)):
        e = trace.by_body[:, j] / denom
        for t in range(T):
            if e[t] != 0.0:
                out[t][seg[t] == bid] = np.float32(e[t])
    return out


def body_state(traj, spec) -> Dict[str, np.ndarray]:
    """Per-body physical quantities, as the energy computation saw them.

    Derived from the trajectory, not additional to it: `traj.npz` ships in every
    clip directory and already carries pos, quat, both velocities, radius,
    gravity and the contacts. Two things make this worth its own file.

    `traj.mass` is the declared mass, one constant per body. `bodies["mass"]` is
    `[T,B]` and **follows volume**, which is what the energy is computed
    against and what makes `immutability` a mass violation while `deformation`
    is not.

    And every column comes off the same code path as the energy trace --
    `inertia_diag`, the same datum, the same volume rule -- so the two cannot
    disagree about what a body weighed or how it was spinning.

    A consumer can recompute E from these columns and get the shipped trace back
    to floating-point noise, which `tests/test_energy.py` pins.
    """
    bodies = {b.name: b for b in spec.bodies}
    g = np.asarray(traj.gravity, np.float64)
    g_norm = float(np.linalg.norm(g))
    g_hat = g / max(g_norm, 1e-12)
    datum = float(getattr(spec, "floor_level", 0.0))
    T, B = traj.num_frames, traj.num_bodies

    mass = np.zeros((T, B)); inertia = np.zeros((T, B, 3))
    height = np.zeros((T, B)); is_static = np.zeros(B, bool)
    for j, name in enumerate(traj.body_names):
        body = bodies.get(name)
        if body is None:
            continue
        # Reported as static, for the same reason `compute` skips it: a cast
        # shadow is a picture of an absence and has no mechanics. Leaving it
        # dynamic here would let a consumer recompute an energy the shipped
        # trace does not contain.
        is_static[j] = bool(body.static or _is_shadow(body))
        if _is_shadow(body):
            continue
        lin = np.asarray(traj.scale_mul[:, j, :], np.float64)
        m = float(getattr(body, "mass", 1.0)) * np.clip(
            np.prod(lin, axis=1), 1e-9, None)
        mass[:, j] = m
        I0 = inertia_diag(body.kind, getattr(body, "scale", (0.5,) * 3), 1.0)
        inertia[:, j, :] = m[:, None] * I0[None, :] * lin ** 2
        height[:, j] = -(np.asarray(traj.pos[:, j, :], np.float64) @ g_hat) - datum

    v = np.asarray(traj.lin_vel, np.float64)
    w = np.asarray(traj.ang_vel, np.float64)
    R = np.stack([_quat_matrix(np.asarray(traj.quat[:, j, :], np.float64))
                  for j in range(B)], axis=1)                  # [T,B,3,3]
    w_body = np.einsum("tbji,tbj->tbi", R, w)
    kinetic = 0.5 * mass * (v ** 2).sum(-1) + 0.5 * (inertia * w_body ** 2).sum(-1)

    return {
        "body_ids": np.asarray(traj.body_ids, np.int32),
        # Names beside ids so the file is self-describing: `seg == body_ids[j]`
        # is the mask for `body_names[j]` over the whole clip, without having to
        # cross-reference meta.json.
        "body_names": np.asarray([str(n) for n in traj.body_names], dtype="U32"),
        "static": is_static,
        "present": np.asarray(traj.present, bool),
        # HOW MUCH OF THE BODY IS LEFT. `compute` weights each body's energy by
        # this, so it has to ship or the "recompute the trace from this file
        # alone" guarantee stops holding on the one family that moves it --
        # `dissolve`, where the body's matter goes away over several frames
        # rather than at one. 1.0 everywhere else.
        "opacity": np.clip(np.asarray(traj.opacity, np.float32), 0.0, 1.0),
        "mass": mass,                                  # [T,B]  kg, follows volume
        "radius": np.asarray(traj.radius, np.float32),  # [B]   m
        "inertia": inertia,                            # [T,B,3] body-frame diag
        "position": np.asarray(traj.pos, np.float32),   # [T,B,3] m
        "quaternion": np.asarray(traj.quat, np.float32),  # [T,B,4] w,x,y,z
        "velocity": np.asarray(traj.lin_vel, np.float32),   # [T,B,3] m/s
        "speed": np.linalg.norm(v, axis=-1),                # [T,B]
        "angular_velocity": np.asarray(traj.ang_vel, np.float32),
        "angular_speed": np.linalg.norm(w, axis=-1),
        "momentum": (mass[..., None] * v),              # [T,B,3] kg m/s
        "momentum_magnitude": np.linalg.norm(mass[..., None] * v, axis=-1),
        "angular_momentum": (inertia * w_body),         # [T,B,3] body frame
        "height": height,                               # [T,B] above floor datum
        "kinetic": kinetic,                             # [T,B] J
        "potential": mass * g_norm * height,            # [T,B] J
        "gravity": np.asarray(traj.gravity, np.float32),
        "dt": np.float32(traj.dt),
    }

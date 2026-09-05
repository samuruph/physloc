"""Physical-law residuals -- docs/PLAN.md 1.2 and 3.4 step 1.

Every residual is computed from `traj.npz` alone, never from pixels, and is
dimensionless so that families with different physical units stay comparable
after the noise-floor normalisation in 3.4 step 2.

Signature: law(traj, body_index, ctx) -> np.ndarray of shape [T], >= 0.
`ctx` carries scenario geometry the law needs (surface height, etc).
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from ..sim.trajectory import Trajectory

Ctx = Dict[str, Any]
_LAWS: Dict[str, Callable[[Trajectory, int, Ctx], np.ndarray]] = {}


def register(name: str):
    def deco(fn):
        _LAWS[name] = fn
        return fn
    return deco


def get(name: str):
    if name not in _LAWS:
        raise KeyError("no residual law %r; have %s" % (name, sorted(_LAWS)))
    return _LAWS[name]


def available():
    return sorted(_LAWS)


# --------------------------------------------------------------------------
@register("penetration")
def penetration(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Solidity: how far the body's surface is inside another solid, in radii.

    Two cases, because "inside" means different things depending on what it is
    inside of. Against a static surface it is depth below the surface plane;
    against another *moving* body it is the overlap of the two along the line
    of centres, which is the quantity that actually goes to zero when two balls
    are merely touching and grows as one passes through the other. Measuring a
    ball-ball pass-through as height below a floor would report nothing at all.
    """
    r = max(float(traj.radius[b]), 1e-9)
    partner = ctx.get("partner_id")
    if partner is not None and ctx.get("pass_through"):
        try:
            j = traj.index_of(int(partner))
        except KeyError:
            j = None
        if j is not None:
            # Overlap along the collision axis. Projecting onto the contact
            # normal rather than using centre distance is what makes one
            # expression cover both a ball entering another ball and a ball
            # entering a wall: the second has no meaningful centre distance,
            # and its "depth below the top face" is a height in mid-air.
            n = np.asarray(ctx.get("contact_normal", (0.0, 0.0, 1.0)), np.float64)
            nn = float(np.linalg.norm(n))
            n = n / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
            delta = (traj.pos[:, b, :].astype(np.float64)
                     - traj.pos[:, j, :].astype(np.float64))
            gap = np.abs(delta @ n)
            reach = float(ctx.get("partner_extent", traj.radius[j]))
            return np.maximum(0.0, r + reach - gap) / r
    top = float(ctx["surface_top"])
    lowest = traj.pos[:, b, 2] - r
    depth = np.maximum(0.0, top - lowest)
    bounds = ctx.get("support_bounds")
    if bounds is not None:
        # Only where the body is actually over the surface. A raised support is
        # finite, and a mug knocked off a table is *below table height* for the
        # rest of the clip without having passed through anything -- which made
        # `phantom_impulse` and `newton1_inertia` on `resting_table` read as
        # solidity violations.
        cx, cy, hx, hy = (float(x) for x in bounds)
        over = ((np.abs(traj.pos[:, b, 0] - cx) <= hx + r)
                & (np.abs(traj.pos[:, b, 1] - cy) <= hy + r))
        depth = depth * over
    return (depth / max(r, 1e-9)).astype(np.float64)


@register("free_fall")
def free_fall(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Kinematics: |a - g| / |g| for a body that should be in free flight.

    Non-zero during legitimate contact too, so callers gate it on the frames
    where the body is actually unsupported (`ctx['unsupported']`, optional).
    """
    g = traj.gravity.astype(np.float64)
    a = traj.acceleration(b).astype(np.float64)
    r = np.linalg.norm(a - g[None, :], axis=1) / max(float(np.linalg.norm(g)), 1e-9)
    gate = ctx.get("unsupported")
    if gate is not None:
        r = r * np.asarray(gate, dtype=np.float64)
    return r


@register("mass_continuity")
def mass_continuity(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Permanence: the fraction of the body's mass missing from the scene.

    Discrete by nature -- 1.0 while absent, 0.0 while present. The IntPhys 2
    categories are all essentially discrete; that is exactly why PLAN Part 2
    adds continuously dialable families alongside them.
    """
    present = traj.present[:, b].astype(np.float64)
    return 1.0 - present


@register("position_continuity")
def position_continuity(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Continuity: a step larger than any velocity in play could produce.

    Scored against the distance the body could have covered given its speed on
    *either* side of the step, not against the previous velocity alone. The
    difference matters more than it looks: extrapolating from `v[t-1]` means
    every uncaused shove registers as a teleport, because a body that speeds up
    always lands further along than its old velocity predicted. That made
    `phantom_impulse`, `newton1_inertia` and half the other families read as
    continuity violations too, and a benchmark cannot ask "can the model spot a
    teleport" using clips where six other families also trip the teleport
    detector.

    A real teleport is a step no plausible speed explains, so that is what this
    measures.
    """
    dt = traj.dt
    r = max(float(traj.radius[b]), 1e-9)
    out = np.zeros((traj.num_frames,), np.float64)
    if traj.num_frames < 2:
        return out
    p = traj.pos[:, b, :].astype(np.float64)
    v = np.linalg.norm(traj.lin_vel[:, b, :].astype(np.float64), axis=1)
    step = np.linalg.norm(p[1:] - p[:-1], axis=1)
    # Whichever end of the step was faster, with room for acceleration inside
    # the frame.
    reach = np.maximum(v[:-1], v[1:]) * dt * float(ctx.get("step_tolerance", 1.6))
    out[1:] = np.maximum(0.0, step - reach) / r
    return out


@register("linear_momentum")
def linear_momentum(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Newton 1/2/3, phantom impulse: momentum change unaccounted for by forces.

    ||m*dv - (F_contact + m*g)*dt|| / (m*|g|*dt). With no contact recorded the
    only legal force is gravity, so this reduces to the free-fall residual.
    """
    g = traj.gravity.astype(np.float64)
    m = float(traj.mass[b])
    dt = traj.dt
    v = traj.lin_vel[:, b, :].astype(np.float64)
    dv = np.zeros_like(v)
    dv[1:] = v[1:] - v[:-1]

    f_contact = np.zeros_like(v)
    c = traj.contacts
    bid = int(traj.body_ids[b])
    if len(c):
        for k in range(len(c)):
            f = int(c.frame[k])
            if not (0 <= f < traj.num_frames):
                continue
            # SIGN. PyBullet's `contactNormalOnB` points from B towards A, and
            # Kubric hands the pair back swapped, so our stored normal points
            # from `body_a` towards `body_b`. The force that contact exerts *on
            # body_a* is therefore along MINUS the stored normal.
            #
            # With the signs the other way round the contact force added to
            # gravity instead of cancelling it, and a body simply resting on the
            # floor read `2*m*g*dt / (m*g*dt)` = 2.0 on every frame of a
            # perfectly lawful clip. That constant bias is what the noise floor
            # was calibrated against, and it swamped real signals of 1-3.
            if int(c.body_a[k]) == bid:
                f_contact[f] -= c.normal[k].astype(np.float64) * float(c.impulse[k])
            elif int(c.body_b[k]) == bid:
                f_contact[f] += c.normal[k].astype(np.float64) * float(c.impulse[k])

    resid = m * dv - (f_contact + m * g[None, :]) * dt
    denom = max(m * float(np.linalg.norm(g)) * dt, 1e-9)
    out = (np.linalg.norm(resid, axis=1) / denom).astype(np.float64)

    return out


@register("energy_at_contact")
def energy_at_contact(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Super-elastic: fractional gain in *total mechanical* energy across a frame.

    Kinetic energy alone is the wrong quantity, and wrong in a way that quietly
    destroys the signal: a body in free fall gains kinetic energy on every
    single frame, perfectly lawfully, by trading height for speed. Scored on
    kinetic energy the valid twin registers a large residual throughout its
    fall, the noise floor calibrated on it swallows the real bounce, and a
    genuine super-elastic clip comes out with a severity field of exactly zero.

    Potential energy is measured from `surface_top`, so free flight conserves
    the total, a lawful bounce loses some to restitution, and only energy
    arriving from nowhere shows up. No contact gate is needed -- the law is now
    true on every frame, which is what a conservation law should be.
    """
    g = float(np.linalg.norm(traj.gravity))
    datum = float(ctx.get("surface_top", 0.0))

    # THE PAIR, when the collision has two moving participants. A ball-ball
    # bounce that returns more than it took shows up in neither body alone: the
    # striker still LOSES energy, just less of it, and the target's larger gain
    # looks like an ordinary transfer. Only their sum rises. Scored on the
    # striker this read exactly 0.000 on a clip whose total mechanical energy
    # went from 6.32 J to 7.42 J.
    ids = [int(x) for x in (ctx.get("pair_ids") or ())]
    members = []
    for seg in ids:
        try:
            members.append(traj.index_of(seg))
        except Exception:                                     # noqa: BLE001
            continue
    if len(members) < 2:
        members = [b]

    m = float(traj.mass[b])
    r = max(float(traj.radius[b]), 1e-9)
    e = np.zeros((traj.num_frames,), np.float64)
    for j in members:
        mj = float(traj.mass[j])
        vj = traj.lin_vel[:, j, :].astype(np.float64)
        hj = traj.pos[:, j, 2].astype(np.float64) - datum
        e += 0.5 * mj * np.sum(vj * vj, axis=1) + mj * g * hj

    out = np.zeros_like(e)
    if e.shape[0] < 2:
        return out
    # Scale by the body's own gravitational energy over one radius, so a body
    # momentarily at rest at the datum does not divide a small gain by ~0.
    prev = np.maximum(e[:-1], m * g * r)
    out[1:] = np.maximum(0.0, e[1:] - e[:-1]) / prev
    return out


@register("trajectory_shape")
def trajectory_shape(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Non-parabolic flight: per-frame deviation from the fitted g-parabola,
    in metres, normalised by the body radius."""
    g = traj.gravity.astype(np.float64)
    dt = traj.dt
    frames = np.asarray(ctx.get("flight_frames", np.arange(traj.num_frames)), int)
    out = np.zeros((traj.num_frames,), np.float64)
    if frames.size < 3:
        return out
    p = traj.pos[frames, b, :].astype(np.float64)
    t = (frames - frames[0])[:, None] * dt
    # Least squares for p0, v0 given known g.
    q = p - 0.5 * g[None, :] * t ** 2
    A = np.concatenate([np.ones_like(t), t], axis=1)
    coef, *_ = np.linalg.lstsq(A, q, rcond=None)
    fit = A @ coef + 0.5 * g[None, :] * t ** 2
    out[frames] = np.linalg.norm(p - fit, axis=1) / max(float(traj.radius[b]), 1e-9)
    return out


@register("shape_continuity")
def shape_continuity(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Immutability: how far the body's volume has drifted from its own frame 0.

    Reported as |V/V0 - 1|, so a body that doubles in volume scores 1.0 and one
    that halves scores 0.5. Asymmetric on purpose -- growing is unbounded,
    shrinking bottoms out at total disappearance, which is `permanence`'s law,
    not this one.
    """
    s = np.asarray(traj.scale_mul[:, b, :], np.float64)
    vol = np.prod(np.maximum(s, 1e-9), axis=1)
    return np.abs(vol / max(float(vol[0]), 1e-9) - 1.0)


@register("mass_dissolution")
def mass_dissolution(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """How much of the body has ceased to exist, as a fraction, over time.

    Continuous where `mass_continuity` is a step: it follows the body's opacity
    down from solid to invisible and stays at 1.0 once it is removed. That
    difference is the whole distinction between `dissolve` and `permanence` --
    the same end state, reached in a way a model has to notice as a *trend*
    rather than as a single-frame discontinuity.
    """
    opaque = np.clip(np.asarray(traj.opacity[:, b], np.float64), 0.0, 1.0)
    gone = 1.0 - opaque / max(float(opaque[0]), 1e-9)
    return np.maximum(np.clip(gone, 0.0, 1.0),
                      1.0 - traj.present[:, b].astype(np.float64))


@register("object_count")
def object_count(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Fission: how many bodies exist where one should.

    Counted over a declared sibling group -- the body and any understudy that
    could stand in for it -- and reported relative to frame 0, so one becoming
    two scores 1.0. Like `mass_continuity` this is discrete by construction:
    an object either split or it did not, and the severity bins vary how far
    the halves fly apart rather than how true the residual is.
    """
    ids = [int(i) for i in ctx.get("sibling_ids", [])] or [int(traj.body_ids[b])]
    idx = []
    for i in ids:
        try:
            idx.append(traj.index_of(i))
        except KeyError:
            continue
    if not idx:
        return np.zeros((traj.num_frames,), np.float64)
    n = traj.present[:, idx].sum(axis=1).astype(np.float64)
    return np.abs(n / max(float(n[0]), 1.0) - 1.0)


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0,1] -> CIE-Lab (D65). Vectorised over a trailing axis of 3.

    Perceptual distance, not RGB distance, because the two disagree badly: the
    same numeric step is glaring between two greens and invisible between two
    dark blues. A severity that claims to be comparable across clips has to be
    measured in a space where equal steps look equally different.
    """
    c = np.clip(np.asarray(rgb, np.float64), 0.0, 1.0)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0,
                     500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], axis=-1)


@register("colour_continuity")
def colour_continuity(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Appearance: how far the body's colour has drifted from its own frame 0.

    CIE-Lab distance, scaled by 100 so a value of 1.0 is about as different as
    two colours get. Measured against the body's *own* first frame rather than
    an absolute reference, so a scene of differently coloured objects does not
    register a violation just for containing them.
    """
    lab = _srgb_to_lab(np.asarray(traj.colour[:, b, :], np.float64))
    return np.linalg.norm(lab - lab[0][None, :], axis=1) / 100.0


@register("shape_anisotropy")
def shape_anisotropy(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """How far the body has been stretched out of its own proportions.

    Distinct from `shape_continuity`, which scores a change in *volume* and is
    blind to a body that squashes flat while conserving it. This scores the
    aspect ratio: the widest axis over the narrowest, relative to frame 0. A
    body that doubles in size uniformly reads zero here and 7.0 there, and a
    body that squashes to half its height at constant volume reads the reverse.

    Used both by `deformation`, where the body itself distorts, and by
    `shadow_shape`, where the shadow stops matching the shape of the thing
    casting it.
    """
    s = np.asarray(traj.scale_mul[:, b, :], np.float64)
    s = np.maximum(np.abs(s), 1e-9)
    ratio = s.max(axis=1) / s.min(axis=1)
    return np.abs(ratio / max(float(ratio[0]), 1e-9) - 1.0)


@register("angular_momentum")
def angular_momentum(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Spin that changed with nothing to change it.

    ||dw|| * r / (|g| * dt): the frame-to-frame change in angular velocity,
    scaled to the linear units the other laws use so severities stay
    comparable. Gated to frames with no recorded contact for this body --
    a tumbling body swaps angular momentum with the ground legally, and only
    a change while free of contact is unexplained.
    """
    r = float(traj.radius[b])
    g = max(float(np.linalg.norm(traj.gravity)), 1e-9)
    pivot = ctx.get("pivot")
    if pivot is not None:
        # ON A CONSTRAINT the angular momentum that matters is about the PIVOT,
        # not about the body's own centre. A bob on a rod carries its angular
        # momentum as `r x v`; its own spin is zero throughout and stays zero
        # however violently the swing is changed, so measuring the spin reports
        # nothing at all -- which is exactly what `pendulum_swing x
        # angular_momentum` did, shipping a reversed swing at severity 0.000.
        #
        # Written in the same units as the free case -- an effective angular
        # rate times a lever arm over `g*dt` -- so the two are comparable and a
        # single `r_strong` means the same thing on both.
        arm_vec = (np.asarray(traj.pos[:, b, :], np.float64)
                   - np.asarray(pivot, np.float64)[None, :])
        vel = np.asarray(traj.lin_vel[:, b, :], np.float64)
        span = np.maximum(np.sum(arm_vec * arm_vec, axis=1), 1e-12)
        w = np.cross(arm_vec, vel) / span[:, None]
        r = float(ctx.get("arm", np.sqrt(span).mean()))
    else:
        w = np.asarray(traj.ang_vel[:, b, :], np.float64)
    dw = np.zeros_like(w)
    dw[1:] = w[1:] - w[:-1]
    out = np.linalg.norm(dw, axis=1) * r / (g * traj.dt)

    touching = np.zeros((traj.num_frames,), bool)
    c = traj.contacts
    bid = int(traj.body_ids[b])
    if len(c):
        for k in range(len(c)):
            f = int(c.frame[k])
            if 0 <= f < traj.num_frames and bid in (int(c.body_a[k]),
                                                    int(c.body_b[k])):
                touching[max(0, f - 1):f + 2] = True
    gate = ctx.get("contact_free")
    if gate is not None:
        touching |= ~np.asarray(gate, bool)
    return out * (~touching).astype(np.float64)


@register("support")
def support(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Equilibrium: a body held up by nothing.

    Clearance between the body's lowest point and the surface below it, in
    radii, counted only on frames where the body is *not* in free flight. Both
    halves are needed: a ball mid-arc has metres of clearance and is perfectly
    lawful, while a hovering one has the same clearance and is not. What
    separates them is whether it is accelerating at g.
    """
    top = float(ctx["surface_top"])
    r = max(float(traj.radius[b]), 1e-9)
    clearance = np.maximum(0.0, (traj.pos[:, b, 2] - r) - top) / r
    g = traj.gravity.astype(np.float64)
    a = traj.acceleration(b).astype(np.float64)
    falling = (np.linalg.norm(a - g[None, :], axis=1)
               / max(float(np.linalg.norm(g)), 1e-9)) < float(ctx.get("free_fall_tol", 0.35))
    return clearance * (~falling).astype(np.float64)


@register("friction")
def friction(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Along-track acceleration that gravity does not explain, over |g|.

    Signed component of (a - g) along the direction of travel, absolute-valued:
    it catches both "slid to a halt with nothing to stop it" and "sped up on
    its own". Legal sliding friction also registers here -- which is precisely
    what the noise floor measured on the valid twin subtracts out, so the score
    reflects only the part the intervention added.
    """
    g = traj.gravity.astype(np.float64)
    a = traj.acceleration(b).astype(np.float64)
    v = np.asarray(traj.lin_vel[:, b, :], np.float64)
    speed = np.linalg.norm(v, axis=1)
    moving = speed > float(ctx.get("min_speed", 0.05))
    vhat = v / np.maximum(speed, 1e-9)[:, None]
    along = np.sum((a - g[None, :]) * vhat, axis=1)
    return np.abs(along) / max(float(np.linalg.norm(g)), 1e-9) * moving.astype(np.float64)


@register("phase_consistency")
def phase_consistency(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Temporal: frames of its own motion the body has lost, cumulative.

    A frame counts as lost in proportion to how far short of `ctx["v_ref"]` the
    body's step falls -- 1.0 when it is frozen, 0.0 when it is travelling at
    reference speed. Cumulated, so the residual reads directly as "how many
    frames behind its own phase is it", which is the unit `time_slip`'s
    magnitude is in.

    **Not zero on a lawful clip**, deliberately: a body coming to rest under
    friction loses frames by this measure too. It is scored against the valid
    twin frame by frame (`severity.bounded_score(baseline=...)`), and against
    that baseline a stall of `n` frames reads as exactly `n` from the end of the
    stall onward, while a lawful deceleration reads as zero.

    A *literally* temporal residual -- "these frames are in the wrong order" --
    is not computable here: a law sees one trajectory and one body, with no
    access to the twin and no notion of frame identity. This measures the
    consequence instead, which is that the body ends up behind its own phase.
    That consequence is localisable to one body over one interval, where frame
    shuffling is localisable to neither.
    """
    T = traj.num_frames
    out = np.zeros((T,), np.float64)
    if T < 2:
        return out
    p = np.asarray(traj.pos[:, b, :], np.float64)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)

    v_ref = float(ctx.get("v_ref", 0.0))
    if v_ref <= 1e-6:
        moving = step[step > 1e-9]
        v_ref = float(np.median(moving)) / traj.dt if moving.size else 0.0
    expected = v_ref * traj.dt
    if expected <= 1e-9:
        return out
    lost = np.clip(1.0 - step / expected, 0.0, 1.0)
    out[1:] = np.cumsum(lost)
    return out


@register("energy_balance")
def energy_balance(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Energy that moved while nothing was touching this body.

    Fraction of the clip's initial mechanical energy, per frame. Needs the scene
    spec (for mass and inertia), which the annotation pipeline puts in
    `ctx["spec"]`; without it the law is silent rather than wrong.

    Measured floor on valid clips: 0.000% on `occluder_pass`, 1.7% on `drop`,
    where a hard impact strains the discrete-time balance. Violating families
    land between 7% and 768%. See docs/energy.md.
    """
    spec = ctx.get("spec")
    if spec is None:
        return np.zeros((traj.num_frames,), np.float64)
    from . import energy as _energy
    return _energy.per_body_free_anomaly(traj, spec)[:, b]


@register("shadow_consistency")
def shadow_consistency(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Optical: how far the cast shadow sits from where its caster puts it.

    `b` is the *shadow*, not the object. The expected position is the caster's
    centre projected along the light direction onto the ground plane; the
    residual is the miss distance in caster radii, so it is comparable with
    every other length-based law in the table.
    """
    ci = traj.index_of(int(ctx["caster_id"]))
    L = np.asarray(ctx["light_dir"], np.float64)
    L = L / max(float(np.linalg.norm(L)), 1e-9)
    top = float(ctx.get("surface_top", 0.0))
    p = np.asarray(traj.pos[:, ci, :], np.float64)
    denom = -L[2] if abs(L[2]) > 1e-6 else 1e-6
    t = (p[:, 2] - top) / denom
    expected = p[:, :2] + t[:, None] * L[None, :2]
    got = np.asarray(traj.pos[:, b, :2], np.float64)
    r = max(float(traj.radius[ci]), 1e-9)
    return np.linalg.norm(got - expected, axis=1) / r


# --------------------------------------------------------------------------
def compute_all(traj: Trajectory, ctx_for_body) -> Dict[str, np.ndarray]:
    """Every registered law for every body -> {law: [T, B]}."""
    T, B = traj.num_frames, traj.num_bodies
    out = {}
    for name, fn in _LAWS.items():
        arr = np.zeros((T, B), np.float64)
        for b in range(B):
            if bool(traj.is_static[b]):
                continue
            try:
                arr[:, b] = fn(traj, b, ctx_for_body(b))
            except Exception:                       # noqa: BLE001
                arr[:, b] = 0.0
        out[name] = arr
    return out

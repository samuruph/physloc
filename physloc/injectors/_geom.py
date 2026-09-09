"""Geometry every injector needs, in one place.

Injectors compose with scenarios by asking questions about *state* -- "what is
under this body?", "when is it airborne?" -- never about scenario names. These
helpers are the vocabulary for those questions, and keeping them here is what
stops thirteen scenarios times sixteen families from becoming per-combination
code.

py3.9-compatible: runs inside the container.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .. import camera as _cam
from ..scenarios import base


# ----------------------------------------------------------------- bodies --
def actors(spec) -> List:
    """Every body a violation may act on. Several scenarios have more than one
    (`collision`, `pour`), which is why this is a list."""
    return [b for b in spec.bodies if b.role == "actor"]


def actor(spec):
    a = actors(spec)
    return a[0] if a else None


def dynamic_bodies(spec) -> List:
    return [b for b in spec.bodies if not b.static]


def top_of(spec, body) -> float:
    """World z of the walkable top face of `body`."""
    if body is None:
        return float(spec.floor_level)
    if body.kind == "dome":
        # KuBasic's dome is a bowl whose interior floor sits at the origin.
        return float(spec.floor_level)
    # THE MESH'S OWN TOP, not its scale. `scale` is a scale FACTOR, and no
    # KuBasic mesh reaches 1.0 in its own coordinates: a cylinder stops at 0.5,
    # a torus at 0.15, and a cone's tip is at 0.900 while its base is at
    # -0.306, so it is not even centred on its origin. This used to return
    # `position + scale` for all of them, which puts a cylinder's support
    # surface at TWICE its real height and a torus's at nearly seven times.
    #
    # It matters because this is the datum `support` scores against: a body
    # resting on a cylinder was measured as floating half the cylinder's height
    # on a perfectly lawful frame.
    lo, hi = base.KIND_BOUNDS.get(body.kind, base.KIND_BOUNDS["sphere"])
    return float(body.position[2] + float(hi[2]) * float(body.scale[2]))


def _over(surface, position, radius: float) -> bool:
    """Is a body at `position` actually above this surface's footprint?

    Height alone is not enough, and `pour` is where that shows: its
    box walls are tall thin slabs whose tops sit below the falling grains, so a
    search that only compares heights decided a grain was being held up by a
    wall standing half a metre to one side. The `penetration` residual is then
    gated to frames the grain is over that wall -- which is never -- and a whole
    pour dropping through the floor scored exactly zero.
    """
    if surface.kind != "cube":
        return True                       # a dome or plane is under everything
    cx, cy, _ = surface.position
    hx, hy, _ = surface.scale
    return (abs(float(position[0]) - float(cx)) <= float(hx) + radius
            and abs(float(position[1]) - float(cy)) <= float(hy) + radius)


def support_under(spec, body) -> Tuple[Optional[object], float]:
    """The static surface directly beneath `body`, and its top z.

    "Directly beneath" is decided by height, not by declaration order: a
    scenario with both a floor and a table has two static surfaces, and a mug
    on the table is supported by the higher one. Picking the first static body
    in the list -- which is what a naive helper does -- silently annotates the
    mug as floating a table's height above its support.
    """
    lowest = float(body.position[2]) - float(body.bounding_radius) + 1e-3
    best, best_top = None, None
    for other in spec.bodies:
        if other is body or not other.static:
            continue
        # A DISTRACTOR IS NEVER A SUPPORT SURFACE. It is scenery: the
        # scenario stages the floor, the table, the ramp that a violation is
        # defined against, and letting a decoration redefine that changes what
        # the violation MEANS. Measured on `drop x support`: with distractors
        # the actor came to rest near a cylinder and `surface_top` jumped from
        # 0.0 to 0.87 -- the cylinder's top -- so "held up with nothing under
        # it" was scored against a reference 0.87 m in the air and the residual
        # fell from 3.60 to 1.65.
        if other.role in ("distractor", "backdrop"):
            continue
        if not _over(other, body.position, body.bounding_radius):
            continue
        t = top_of(spec, other)
        if t <= lowest and (best_top is None or t > best_top):
            best, best_top = other, t
    if best_top is None:
        return None, float(spec.floor_level)
    return best, float(best_top)


def surface_top(spec, body=None) -> float:
    """Height of whatever `body` would land on. Defaults to the first actor."""
    body = body if body is not None else actor(spec)
    if body is None:
        return float(spec.floor_level)
    return support_under(spec, body)[1]


# ----------------------------------------------------------------- timing --
def occluded_midpoint(spec) -> Optional[int]:
    """The middle of the scenario's fully-occluded run, if it has one.

    Firing there is what makes `observability_lag` non-zero; the midpoint gives
    the widest margin against the geometric occlusion test being off by a frame
    at either edge.
    """
    occ = spec.notes.get("occluded_frames") or []
    return int(occ[len(occ) // 2]) if occ else None


#: Where a violation fires when nothing physical dictates the moment, as a
#: fraction of the clip. Everything downstream is expressed this way so that
#: lengthening a tier lengthens the violation rather than leaving it a brief
#: event followed by a long static shot.
EVENT_FRACTION = 0.35

#: How much of what remains after the event a sustained violation occupies.
WINDOW_FRACTION = 0.55


#: How far the event moment is jittered per scene, as a fraction of the clip.
#:
#: Without it every clip that has no physical cue fires at exactly the same
#: frame: measured across twelve seeds, `immutability` and `colour_shift`
#: produced ONE distinct `t_event` and `continuity` three. A benchmark whose
#: violations all begin a third of the way in teaches when to look rather than
#: what to look for.
EVENT_JITTER = 0.10


def scene_fraction(spec) -> float:
    """A number in [0, 1) that is a property of the SCENE, not the family.

    One draw, reused wherever a family has a *choice* of moment, so that the
    three severities of a cell fire together -- otherwise their magnitudes stop
    being comparable -- and two families on the same scene sit at the same
    relative point in whatever window each of them can use.

    Salted away from `Scenario.rng` so that consulting it cannot shift any
    physics draw.
    """
    seed = (int(spec.seed) * 2654435761 + 0x51ED) % (2 ** 31 - 1)
    return float(np.random.RandomState(seed).uniform())


def frame_in_band(spec, lo: int, hi: int) -> int:
    """Place the event INSIDE the band that is actually available.

    The difference between this and jittering a fraction of the clip and then
    clamping it: a clamp collapses to the band's edge whenever the band is
    narrow or offset, and the randomness silently disappears. `_event_frame`
    clamped to `num_frames // 4`, and for a body airborne from frame 0 -- every
    `drop`, `toss` and `pyramid_impact` -- that edge was binding on every seed,
    so `continuity` fired on frame 6 of 25 in ten clips out of ten while the
    jitter upstream was working perfectly.

    Interpolating instead means the choice survives however narrow the band is.
    """
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        return lo
    return lo + int(round(scene_fraction(spec) * (hi - lo)))


def default_event_frame(spec, num_frames: int) -> Optional[int]:
    """When to fire, absent a physical cue: hidden if possible, else a third in.

    Roughly a third of the way in leaves the opening frames untouched -- so the
    prefix is long enough to establish what lawful motion looks like -- and
    still leaves most of the clip for the consequences to play out. Jittered per
    SCENE, not per family or per bin: the three severities of one cell must fire
    together or their magnitudes stop being comparable, and two families on one
    scene sharing a moment is what makes them comparable to each other.
    """
    t0 = occluded_midpoint(spec)
    if t0 is None:
        frac = (EVENT_FRACTION
                + (2.0 * scene_fraction(spec) - 1.0) * EVENT_JITTER)
        t0 = max(1, int(round(frac * num_frames)))
    return int(t0) if 1 <= t0 < num_frames - 1 else None


def window_frames(num_frames: int, t0: int, fraction: float = None,
                  minimum: int = 3) -> int:
    """A sustained window as a share of what is left after the event."""
    frac = WINDOW_FRACTION if fraction is None else fraction
    return max(minimum, int(round(frac * (num_frames - t0))))


def airborne(traj, bi: int, top: float, slack: float = 1e-3) -> np.ndarray:
    """[T] bool: frames where body `bi`'s lowest point clears the surface."""
    r = float(traj.radius[bi])
    return (traj.pos[:, bi, 2] - r) > (top + slack)


def first_contact_any(traj, body_id: int, exclude=()) -> Optional[Tuple[int, int]]:
    """(frame, other_body_id) of this body's earliest recorded contact.

    Static partners win ties -- a ball arriving at the floor is a cleaner event
    to hang a violation on than a graze against another moving body on the same
    frame.
    """
    c = traj.contacts
    if not len(c):
        return None
    best = None
    for k in range(len(c)):
        a, b = int(c.body_a[k]), int(c.body_b[k])
        if body_id not in (a, b):
            continue
        other = b if a == body_id else a
        if other in exclude:
            continue
        f = int(c.frame[k])
        if best is None or f < best[0]:
            best = (f, other)
    return best


def first_dynamic_pair_contact(spec, traj, prefer: Optional[int] = None,
                               only=None, min_approach: float = 0.35
                               ) -> Optional[Tuple[int, int, int]]:
    """(frame, id_a, id_b) of the first contact between two *dynamic* bodies.

    The event `newton2_mass` and `newton3_reaction` need: two things that both
    ought to respond, so that only one responding is visibly wrong.

    `only` restricts both bodies to a declared set; `prefer` is the actor, and
    contacts involving it win outright rather than merely breaking ties. In `pyramid_impact` the base spheres settle against
    each other on frame 1, so the earliest dynamic pair is a nudge between two
    props -- an eventless violation with a one-frame lawful prefix -- while the
    collision the scenario exists to stage is the cube landing on the apex five
    frames later.
    """
    dyn = {b.segmentation_id for b in spec.bodies
           if not b.static and not b.dormant}
    if only:
        # Restrict to a declared set, so a family that needs two bodies the
        # viewer cannot tell apart is not handed a cube and a sphere.
        dyn &= {int(i) for i in only}
    c = traj.contacts
    best, best_with_actor = None, None
    for k in range(len(c)):
        a, b = int(c.body_a[k]), int(c.body_b[k])
        if not (a in dyn and b in dyn and a != b):
            continue
        f = int(c.frame[k])
        if f < 1:
            continue
        # A real impact, not two bodies settling against each other. The
        # spheres in a pyramid touch from frame 0 and jostle for the whole
        # clip, so "first contact between two dynamic bodies" found a nudge
        # with no momentum in it -- and a Newton family staged on that has
        # almost no reaction to suppress. Scored 0.01.
        ia, ib = traj.index_of(a), traj.index_of(b)
        # Snap to the frame the surfaces actually meet. PyBullet resolves
        # contacts at its own sub-step rate and reports the frame it noticed,
        # which can be a frame before the two are visibly touching -- so the
        # clip showed one ball reacting to another it had not reached yet.
        reach = float(traj.radius[ia] + traj.radius[ib])
        gap = np.linalg.norm(
            traj.pos[:, ib, :].astype(np.float64)
            - traj.pos[:, ia, :].astype(np.float64), axis=1)
        touching = np.flatnonzero(gap <= reach * 1.02)
        touching = touching[(touching >= 1) & (touching < traj.num_frames - 1)]
        if touching.size:
            f = int(touching[0])
        sep = traj.pos[f, ib].astype(np.float64) - traj.pos[f, ia].astype(np.float64)
        mag = float(np.linalg.norm(sep))
        if mag > 1e-9:
            n = sep / mag
            closing = float(np.dot(
                traj.lin_vel[f - 1, ia].astype(np.float64)
                - traj.lin_vel[f - 1, ib].astype(np.float64), n))
            if closing < min_approach:
                continue
        if best is None or f < best[0]:
            best = (f, a, b)
        if prefer is not None and prefer in (a, b):
            if best_with_actor is None or f < best_with_actor[0]:
                best_with_actor = (f, a, b)
    return best_with_actor or best


# ------------------------------------------------------------- quaternion --
def integrate_quaternion(q0: np.ndarray, omega: np.ndarray, dt: float,
                         n: int) -> np.ndarray:
    """[n,4] (w,x,y,z) from a constant body-frame angular velocity.

    Exact rather than the first-order `q + 0.5*w*q*dt` update: at 12 fps a
    tumbling body turns a large fraction of a radian per frame, and the
    first-order form both drifts off the unit sphere and visibly under-rotates.
    """
    w = np.asarray(omega, np.float64)
    speed = float(np.linalg.norm(w))
    out = np.zeros((n, 4), np.float32)
    q = np.asarray(q0, np.float64).copy()
    if speed < 1e-9:
        out[:] = q.astype(np.float32)
        return out
    axis = w / speed
    half = 0.5 * speed * dt
    dq = np.concatenate([[np.cos(half)], np.sin(half) * axis])
    for f in range(n):
        q = _qmul(dq, q)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        out[f] = q.astype(np.float32)
    return out


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], np.float64)


def quat_about_y(angle: float) -> Tuple[float, float, float, float]:
    """(w,x,y,z) for a rotation of `angle` about +Y."""
    return (float(np.cos(angle / 2.0)), 0.0, float(np.sin(angle / 2.0)), 0.0)


def support_plane(spec, body):
    """(top, fallback, bounds) describing what `body` can land on.

    `bounds` is the supporting surface's (cx, cy, hx, hy) footprint, or None for
    an unbounded ground plane. `fallback` is what is underneath *that*.
    """
    surface, top = support_under(spec, body)
    if surface is None or surface.kind != "cube":
        return float(top), float(spec.floor_level), None
    cx, cy = float(surface.position[0]), float(surface.position[1])
    hx, hy = float(surface.scale[0]), float(surface.scale[1])
    return float(top), float(spec.floor_level), (cx, cy, hx, hy)


def floor_fn(spec, body):
    """A ground-height function (x, y) -> z for the integrators.

    A raised surface is finite. Without this a body pushed off a table keeps
    gliding along an invisible plane at table height, which renders as a
    hovering object -- a `support` violation smuggled into some other family's
    clip and annotated as neither.
    """
    top, fallback, bounds = support_plane(spec, body)
    if bounds is None or abs(top - fallback) < 1e-6:
        return lambda x, y: top
    cx, cy, hx, hy = bounds

    def ground(x, y):
        if abs(x - cx) <= hx and abs(y - cy) <= hy:
            return top
        return fallback
    return ground


def longest_airborne_run(traj, bi: int, top: float) -> Optional[Tuple[int, int]]:
    """The longest stretch of frames where body `bi` is off the surface.

    Families that bend free flight have to fire *inside* free flight. Picking a
    fixed fraction of the clip instead lands `drop` mid-bounce as often as
    not, and a parabola fitted across a bounce is not a parabola.
    """
    air = airborne(traj, bi, top)
    best, run_start = None, None
    for f in range(int(air.shape[0])):
        if air[f] and run_start is None:
            run_start = f
        elif not air[f] and run_start is not None:
            if best is None or (f - run_start) > (best[1] - best[0] + 1):
                best = (run_start, f - 1)
            run_start = None
    if run_start is not None:
        end = int(air.shape[0]) - 1
        if best is None or (end - run_start + 1) > (best[1] - best[0] + 1):
            best = (run_start, end)
    return best


def last_dynamic_contact(spec, traj, body_id: int) -> int:
    """Last frame this body touches another *dynamic* body, or -1 if it never does.

    Static contacts do not count, and the difference matters. A scenario's wall
    or ramp is there on every frame, so a body that meets it later than it
    should still meets it in a lawful way. Another moving body is not: shifting
    one participant of a collision in time and leaving the other alone makes the
    struck body react before it is struck, which is a causality inversion
    nobody labelled.
    """
    dyn = {int(b.segmentation_id) for b in spec.bodies
           if not b.static and int(b.segmentation_id) != int(body_id)}
    c = traj.contacts
    last = -1
    for k in range(len(c.frame)):
        a, b_ = int(c.body_a[k]), int(c.body_b[k])
        if body_id not in (a, b_):
            continue
        other = b_ if a == body_id else a
        if other in dyn:
            last = max(last, int(c.frame[k]))
    return last


def geometric_contacts(spec, traj, tol: float = 0.02):
    """Recompute the contact set from the trajectory's own geometry.

    **An edited trajectory inherits the valid rollout's contact list verbatim**
    -- `_clone` copies the poses and the simulator is never re-run -- so every
    contact-based check on an invalid clip has been reading the *valid* clip's
    contacts. Measured on `collision x fission`: 980 rows, byte-identical,
    including a ball-to-ball contact at the frame where the striker has already
    been diverted a body-width away.

    That is not a small bookkeeping issue. It is what let a struck body keep
    reacting to a collision that no longer happens, and it silently defeats any
    detector built to catch exactly that.

    Spheres against spheres and bodies against their own support surface, which
    is what the analysis needs; `impulse` and `penetration` are left at zero
    because a geometric test cannot know them. `traj.meta["contacts"]` records
    that these are geometric so nothing mistakes them for the simulator's.
    """
    from ..sim.trajectory import Contacts

    dyn = [b for b in spec.bodies if not b.static]
    idx = {int(b.segmentation_id): traj.index_of(int(b.segmentation_id))
           for b in dyn}
    T = traj.num_frames
    frames, a_ids, b_ids = [], [], []

    for f in range(T):
        for i, ba in enumerate(dyn):
            ia = idx[int(ba.segmentation_id)]
            if not bool(traj.present[f, ia]):
                continue
            pa = np.asarray(traj.pos[f, ia], np.float64)
            ra = float(traj.radius[ia]) * float(np.mean(traj.scale_mul[f, ia]))
            # against the surface it rests on
            ground = floor_fn(spec, ba)
            if pa[2] - ra <= ground(pa[0], pa[1]) + tol:
                frames.append(f)
                a_ids.append(int(ba.segmentation_id))
                b_ids.append(int(_support_id(spec, ba)))
            # against every other dynamic body
            for bb in dyn[i + 1:]:
                ib = idx[int(bb.segmentation_id)]
                if not bool(traj.present[f, ib]):
                    continue
                pb = np.asarray(traj.pos[f, ib], np.float64)
                rb = float(traj.radius[ib]) * float(np.mean(traj.scale_mul[f, ib]))
                if np.linalg.norm(pa - pb) <= ra + rb + tol:
                    frames.append(f)
                    a_ids.append(int(ba.segmentation_id))
                    b_ids.append(int(bb.segmentation_id))

    n = len(frames)
    z3 = np.zeros((n, 3), np.float32)
    return Contacts(np.asarray(frames, np.int32), np.asarray(a_ids, np.int32),
                    np.asarray(b_ids, np.int32), z3, z3.copy(),
                    np.zeros((n,), np.float32), np.zeros((n,), np.float32))


def _support_id(spec, body) -> int:
    surface, _ = support_under(spec, body)
    if surface is not None:
        return int(surface.segmentation_id)
    ground = next((b for b in spec.bodies if b.static), None)
    return int(ground.segmentation_id) if ground is not None else 0


def contact_free_run(traj, body_id: int, min_len: int = 2) -> Optional[Tuple[int, int]]:
    """The longest stretch of frames on which nothing touches this body.

    What `angular_momentum` needs: a body in contact can change its spin
    lawfully, so a spin edit made during contact is not a violation of anything
    the residual can prove.
    """
    T = int(traj.num_frames)
    touched = np.zeros((T,), bool)
    c = traj.contacts
    for k in range(len(c)):
        if body_id in (int(c.body_a[k]), int(c.body_b[k])):
            f = int(c.frame[k])
            if 0 <= f < T:
                touched[max(0, f - 1):f + 2] = True
    best, start = None, None
    for f in range(T):
        if not touched[f] and start is None:
            start = f
        elif touched[f] and start is not None:
            if best is None or (f - start) > (best[1] - best[0] + 1):
                best = (start, f - 1)
            start = None
    if start is not None:
        if best is None or (T - start) > (best[1] - best[0] + 1):
            best = (start, T - 1)
    if best is None or (best[1] - best[0] + 1) < min_len:
        return None
    return best


def support_under_any(spec, body, traj=None, frame: int = 0):
    """(body, top) like `support_under`, but a *moving* body counts too.

    `support_under` deliberately ignores dynamic bodies because it feeds the
    integrators, and bouncing off something that is itself falling is not a
    plane. The `support` residual needs the opposite answer: the top block of a
    stack rests on the block below, and measuring its clearance against the
    floor two blocks down would report five radii of hover on a perfectly
    lawful clip and drown the real violation in the noise floor.
    """
    if traj is not None:
        bi = traj.index_of(int(body.segmentation_id))
        lowest = float(traj.pos[frame, bi, 2] - traj.radius[bi]) + 1e-3
    else:
        lowest = float(body.position[2]) - float(body.bounding_radius) + 1e-3
    best, best_top = None, None
    ref = (traj.pos[frame, traj.index_of(int(body.segmentation_id))]
           if traj is not None else body.position)
    for other in spec.bodies:
        if other is body or other.dormant:
            continue
        if not _over(other, ref, body.bounding_radius):
            continue
        if other.static:
            t = top_of(spec, other)
        elif traj is not None:
            oi = traj.index_of(int(other.segmentation_id))
            t = float(traj.pos[frame, oi, 2] + traj.radius[oi])
        else:
            t = float(other.position[2]) + float(other.bounding_radius)
        if t <= lowest and (best_top is None or t > best_top):
            best, best_top = other, t
    if best_top is None:
        return None, float(spec.floor_level)
    return best, float(best_top)


def path_sample(pos: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Positions along a recorded path at fractional frame indices `u`.

    Lets an injector slow, stop or reverse a body *along its own lawful path*
    instead of re-integrating it. On a ramp that is the difference between a
    block that stops on the slope and a block that stops in mid-air next to it:
    the scenario's geometry is baked into the path already, so following it
    cannot produce a surface violation the annotation does not know about.
    """
    T = int(pos.shape[0])
    u = np.clip(np.asarray(u, np.float64), 0.0, T - 1.0)
    i0 = np.floor(u).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    f = (u - i0)[:, None]
    return pos[i0].astype(np.float64) * (1.0 - f) + pos[i1].astype(np.float64) * f


# ------------------------------------------------------------------ frame --
# Kubric's PerspectiveCamera defaults: 50 mm lens on a 36 mm sensor, so the
# half-angle tangent is 18/50. Square renders make the vertical the same.
#: Re-exported from `physloc.camera`, which is the single definition. Scene
#: sampling frames a shot with the same frustum the injectors judge it by, and
#: two copies of the lens constant is a quiet way for those to disagree.
TAN_HALF_FOV = _cam.TAN_HALF_FOV


def camera_basis(spec, frame: Optional[int] = None,
                 num_frames: Optional[int] = None):
    """(eye, forward, right, up) for the scenario's camera.

    Pass the frame the intervention acts on when the camera may move. The
    consumers all pick a direction to act ALONG -- the axis a body is squashed
    on, the line two halves separate across, the plane a snake wiggles in --
    and those only need to read correctly at the moment of the intervention.
    Frame 0 would aim them for a shot the clip has already left.
    """
    if frame is None or num_frames is None:
        return _cam.camera_basis(spec.camera_position, spec.camera_look_at)
    eye, aim = spec.camera_at(int(frame), int(num_frames))
    return _cam.camera_basis(eye, aim)


def in_frame(spec, points: np.ndarray, margin: float = 0.04,
             from_frame: Optional[int] = None,
             num_frames: Optional[int] = None) -> np.ndarray:
    """[...] bool: are these world points inside the camera's view?

    Used to keep an intervention's strongest bin from throwing the actor out of
    shot. A body that leaves frame has an empty mask for the rest of the clip,
    and the clip then *depicts* an object vanishing while being labelled
    `antigravity` or `continuity` -- a mislabelled clip, not merely a dull one.

    Pass `from_frame` and `num_frames` when `points` has a LEADING TIME AXIS
    and the camera may move. Without them every frame is tested against the
    camera's opening pose, which for a moving camera answers a question nobody
    asked: a body the camera pans away from reads as visible, and one it pans
    towards reads as lost. The fit ladder is measured against this, so it would
    be weakening interventions to fit a frustum the clip never had.
    """
    if (from_frame is None or num_frames is None
            or not getattr(spec, "camera_moves", False)
            or np.asarray(points).ndim < 2):
        return _cam.visible(spec.camera_position, spec.camera_look_at, points,
                            margin=margin)
    pts = np.asarray(points, np.float64)
    out = np.zeros(pts.shape[:-1], bool)
    for k in range(pts.shape[0]):
        eye, aim = spec.camera_at(int(from_frame) + k, int(num_frames))
        out[k] = _cam.visible(eye, aim, pts[k], margin=margin)
    return out


def first_impact(traj, body_id: int, exclude=(), min_speed: float = 0.3,
                 min_normal_fraction: float = 0.3):
    """(frame, other_id, normal) of this body's first real *impact*.

    Distinct from `first_contact_any`, which returns the first contact of any
    kind -- including the every-frame contact of a ball rolling along the floor.
    `superelastic` needs a collision it can reflect: on a rolling contact the
    normal is perpendicular to the motion, so flipping the normal component
    changes nothing at all and the clip ships with a violation that never
    happened. An impact is a contact the body is moving *into*.
    """
    c = traj.contacts
    if not len(c):
        return None
    for k in np.argsort(np.asarray(c.frame)):
        f = int(c.frame[k])
        if f < 1 or f >= traj.num_frames:
            continue
        a, b = int(c.body_a[k]), int(c.body_b[k])
        if body_id not in (a, b):
            continue
        other = b if a == body_id else a
        if other in exclude:
            continue
        n = np.asarray(c.normal[k], np.float64)
        if float(np.linalg.norm(n)) < 1e-6:
            continue
        n = n / float(np.linalg.norm(n))
        bi = traj.index_of(body_id)
        v = np.asarray(traj.lin_vel[f - 1, bi], np.float64)
        speed = float(np.linalg.norm(v))
        if speed < min_speed:
            continue
        if abs(float(np.dot(v, n))) < min_normal_fraction * speed:
            continue
        return f, other, n
    return None


# ------------------------------------------------------------- obstacles --
class Obstacles:
    """Everything a re-integrated body must not pass through.

    The integrators originally knew about one thing: the ground. That was fine
    while every family bent a single body in open air, and quietly wrong as
    soon as scenes had furniture in them. A `fission` half pushed sideways went
    straight through the occluding screen; `global_gravity` on a pyramid sent
    spheres through the cube that struck them; two balls under `newton3` ended
    up sharing the same space. Each of those clips carried one violation label
    and depicted two, which makes the families useless for measuring anything
    independently.

    Two kinds of obstacle, both approximate on purpose:

    * **static boxes**, axis-aligned. Rotated ones -- ramps -- are skipped
      rather than approximated, because an AABB around a tilted slab is much
      bigger than the slab and would push bodies out of mid-air. A body sliding
      on a ramp is held up by `floor_fn` instead.
    * **other dynamic bodies**, as spheres of their bounding radius, following
      their *valid* trajectory. Exact for the bodies an injector leaves alone,
      which is the case that matters; bodies being re-integrated together are
      handled by stepping them jointly instead.
    """

    def __init__(self, spec, traj, exclude_ids=()):
        exclude = {int(i) for i in exclude_ids}
        self.boxes = []
        self.spheres = []
        for body in spec.bodies:
            bid = int(body.segmentation_id)
            if bid in exclude or body.dormant:
                continue
            if body.static:
                if body.kind != "cube":
                    continue
                q = np.asarray(body.quaternion, np.float64)
                if abs(float(q[0])) < 0.999:
                    continue                      # rotated: see the docstring
                half = np.asarray(body.scale, np.float64)
                if half[2] <= min(half[0], half[1]):
                    continue                      # lies flat: it is a floor
                self.boxes.append((np.asarray(body.position, np.float64), half))
            else:
                try:
                    j = traj.index_of(bid)
                except KeyError:
                    continue
                self.spheres.append((j, float(traj.radius[j])))
        self._pos = traj.pos
        self._present = traj.present
        self._n = int(traj.num_frames)

    def resolve(self, p, v, radius, restitution, t):
        """Push `p` out of anything it is inside, and reflect `v`. In place-ish.

        `t` is a fractional frame index, so a moving obstacle is sampled where
        it actually is between two rendered frames rather than at the nearest
        one.
        """
        for centre, half in self.boxes:
            d = p - centre
            clamped = np.clip(d, -half, half)
            delta = d - clamped
            dist = float(np.linalg.norm(delta))
            if dist >= radius:
                continue
            if dist > 1e-9:
                n = delta / dist
                p = centre + clamped + n * radius
            else:
                slack = half - np.abs(d)
                axis = int(np.argmin(slack))
                n = np.zeros(3)
                n[axis] = 1.0 if d[axis] >= 0 else -1.0
                p = centre + n * (half[axis] + radius)
            vn = float(np.dot(v, n))
            if vn < 0.0:
                v = v - (1.0 + restitution) * vn * n

        if self.spheres:
            lo = int(np.clip(np.floor(t), 0, self._n - 1))
            hi = int(np.clip(lo + 1, 0, self._n - 1))
            frac = float(np.clip(t - lo, 0.0, 1.0))
            for j, rj in self.spheres:
                if not (bool(self._present[lo, j]) or bool(self._present[hi, j])):
                    continue
                q = (self._pos[lo, j].astype(np.float64) * (1.0 - frac)
                     + self._pos[hi, j].astype(np.float64) * frac)
                d = p - q
                dist = float(np.linalg.norm(d))
                reach = radius + rj
                if dist >= reach or dist < 1e-9:
                    continue
                n = d / dist
                p = q + n * reach
                vn = float(np.dot(v, n))
                if vn < 0.0:
                    v = v - (1.0 + restitution) * vn * n
        return p, v


# ------------------------------------------------------- resized bodies --
def reseat(spec, traj, out, body, bi: int, t0: int,
           half_z: np.ndarray) -> None:
    """Keep a resized body's *clearance* to its support, not its centre height.

    A family that changes a body's size and leaves `pos` alone is writing a
    second, unlabelled violation into the clip: shrink the vertical half-extent
    of a ball resting on the floor and it now hovers by the difference; grow it
    and it sinks into the ground. Both were visible in the released
    `deformation` clips -- the cube stopped touching the floor -- and the old
    guard only handled the growing half, with a `maximum` clamp that could push
    a body up but never bring it back down.

    The invariant that survives a resize is the **gap**, not the centre:
    whatever clearance the lawful rollout had at frame `t`, the resized body
    keeps at frame `t`. A body seated on a surface has gap 0 and stays seated
    through the whole ramp; a body in flight keeps its clearance and therefore
    still lands on the frame it lawfully landed on, seated correctly when it
    does. `half_z` is the body's vertical half-extent per frame from `t0`.
    """
    ground = floor_fn(spec, body)
    r0 = float(traj.radius[bi])
    z = np.asarray(out.pos[t0:, bi, 2], np.float64)
    xy = np.asarray(out.pos[t0:, bi, :2], np.float64)
    z_valid = np.asarray(traj.pos[t0:, bi, 2], np.float64)
    xy_valid = np.asarray(traj.pos[t0:, bi, :2], np.float64)
    hz = np.asarray(half_z, np.float64)
    for k in range(z.shape[0]):
        base = float(ground(xy_valid[k, 0], xy_valid[k, 1]))
        gap = max(0.0, float(z_valid[k]) - r0 - base)
        here = float(ground(xy[k, 0], xy[k, 1]))
        z[k] = here + float(hz[k]) + gap
    out.pos[t0:, bi, 2] = z.astype(np.float32)


def push_out(spec, traj, out, body, bi: int, t0: int,
             radius_t: np.ndarray) -> None:
    """Move a resized body horizontally out of whatever its new size overlaps.

    The counterpart to `reseat` on the other two axes. `immutability` swells a
    ball to 2.3x beside a wall and leaves its centre where it was, so the ball
    ends up half inside the barrier -- a solidity failure inside an identity
    clip. Growing is the only direction that can create an overlap, so a
    shrinking body is untouched and passes through here unchanged.

    Horizontal only, and deliberately: the vertical axis belongs to `reseat`,
    and resolving both here would let a body grow itself up the face of a wall.
    """
    obst = Obstacles(spec, out, exclude_ids=[int(body.segmentation_id)])
    pos = np.asarray(out.pos[t0:, bi, :], np.float64)
    rad = np.asarray(radius_t, np.float64)
    for k in range(pos.shape[0]):
        p = pos[k].copy()
        moved, _ = obst.resolve(p.copy(), np.zeros(3), float(rad[k]), 0.0,
                                float(t0 + k))
        pos[k, 0:2] = moved[0:2]
    out.pos[t0:, bi, :] = pos.astype(np.float32)


def lateral_axis(spec, prefer_horizontal: bool = True) -> int:
    """The world axis a size change shows up best on, from the camera's view.

    A deformation nobody can see is not a deformation. Stretching along the
    camera's viewing direction changes almost nothing on screen -- and on
    `occluder_pass` it did something worse than nothing, pushing the actor's
    new bulk straight out of the front of the screen it was supposed to be
    hidden behind, so the clip read as the body spawning in front of the
    occluder.

    So the axis is chosen by how much of it lies across the image plane rather
    than drawn at random: the world axis with the largest projection onto the
    camera's right vector wins.
    """
    _, _, right, up = camera_basis(spec)
    axes = np.eye(3)
    score = np.abs(axes @ right)
    if not prefer_horizontal:
        score = np.maximum(score, np.abs(axes @ up))
    else:
        score[2] = -1.0                       # vertical is not ours to pick
    return int(np.argmax(score))


def acting_frame(spec, traj, body_id: int, num_frames: int,
                 want: Optional[int] = None, share: float = 0.4,
                 floor_fraction: float = 1.0 / 6.0) -> Optional[int]:
    """A frame the violation has room to play out from.

    `default_event_frame` answers "when is a good moment", and for a body in
    free flight that is not the same question. On `drop` at the debug tier the
    actor is airborne for nine frames and lands on the tenth, so a third of the
    way into the *clip* is a third of the way into the *landing*: the shove
    arrives one frame before impact and the floor absorbs it, and the split
    happens at the moment of contact so both halves are pinned by friction
    before they have moved a body-width. You reported both -- a phantom impulse
    that is barely visible and a fission whose halves end up overlapping.

    So a family that acts on a free body fires inside the contact-free run with
    at least `share` of that run still ahead of it, and no earlier than
    `floor_fraction` of the clip, which is what keeps a lawful prefix.
    """
    if want is None:
        want = default_event_frame(spec, num_frames)
    if want is None:
        return None
    try:
        run = contact_free_run(traj, int(body_id), min_len=2)
    except Exception:                                         # noqa: BLE001
        run = None
    if run is None:
        return int(want)
    lo, hi = int(run[0]), int(run[1])
    latest = hi - int(round(share * max(hi - lo, 0)))
    earliest = max(lo + 1, 1, int(round(floor_fraction * num_frames)))
    # Honour `want` when it already lands inside the usable band; otherwise
    # place it within the band rather than clamping to whichever edge it
    # overshot, which is how a family ends up firing on one frame forever.
    if earliest <= int(want) <= latest:
        t = int(want)
    else:
        t = frame_in_band(spec, earliest, max(earliest, latest))
    if 1 <= t < num_frames - 1:
        return int(t)
    return int(want)


def unoccluded_event_frame(spec, num_frames: int, span: int,
                           want: Optional[int] = None) -> Optional[int]:
    """A start frame whose next `span` frames are all in plain sight.

    The opposite of `default_event_frame`, which *seeks* an occlusion so the
    observability lag is non-zero. Some families need the other thing: a
    `dissolve` that fades out entirely behind a screen is an object that went
    behind a screen and did not come out, which is the picture `permanence`
    already ships -- you noticed the two were indistinguishable on
    `occluder_pass`, and this is what separates them.

    Nearest wins, and ties go earlier: shifting back keeps more of the clip for
    the consequence, and a fade that starts before the actor reaches the screen
    is over before it gets there.
    """
    if want is None:
        # Through `default_event_frame`, not `EVENT_FRACTION` directly, so this
        # path inherits the per-scene jitter. Computing the fraction here again
        # was how `dissolve` and every other family that needs an unoccluded
        # moment kept firing on exactly the same frame in every clip.
        #
        # `default_event_frame` seeks an occlusion when the scene has one, and
        # this function wants the opposite -- but the search below moves off
        # that frame to the nearest clear one, so starting from the jittered
        # moment is right either way.
        want = default_event_frame(spec, num_frames)
        if want is None:
            want = max(1, int(round(EVENT_FRACTION * num_frames)))
    occ = set(int(f) for f in (spec.notes.get("occluded_frames") or []))
    span = max(1, int(span))

    def clear(t: int) -> bool:
        return not any(f in occ for f in range(t, min(t + span, num_frames)))

    if not occ:
        return int(want) if 1 <= want < num_frames - 1 else None
    for delta in range(0, num_frames):
        for t in (want - delta, want + delta):
            if 1 <= t < num_frames - 1 and clear(t):
                return int(t)
    return int(want) if 1 <= want < num_frames - 1 else None


def hidden_behind_static(spec, point) -> bool:
    """Is `point` behind a static upright box, as seen from the camera?

    A segment/AABB slab test from the eye. Used by families that decide *where*
    to leave a body: an intervention that parks the actor out of sight ships a
    clip whose invalid render contains no evidence at all, so `mask_invalid` and
    `severity_map` are empty however large the residual is.

    The declared `occluded_frames` list cannot answer this, and that is not a
    detail. It is computed from the LAWFUL rollout, so it says where the body
    passes on the path it does not take: on the `occluder_pass` seed in the
    review sweep it was empty -- the ball never dwells behind the screen when it
    keeps rolling -- while the friction clip brought it to rest squarely behind
    it. The question is about the invalid path, so it has to be asked of a
    position rather than of a frame index.

    Rotated boxes are skipped rather than approximated, and boxes that lie flat
    are floors: the same two exclusions `Obstacles` makes, for the same reasons.
    """
    eye, _, _, _ = camera_basis(spec)
    p = np.asarray(point, np.float64)
    ray = p - eye
    length = float(np.linalg.norm(ray))
    if length < 1e-9:
        return False
    ray = ray / length

    for body in spec.bodies:
        if not body.static or body.kind != "cube":
            continue
        if getattr(body, "role", "") == "floor":
            continue
        half = np.asarray(body.scale, np.float64)
        if half[2] <= min(half[0], half[1]):
            continue                                  # lies flat: it is a floor
        q = np.asarray(body.quaternion, np.float64)
        if abs(float(q[0])) < 0.999:
            continue                                  # rotated: see the docstring
        centre = np.asarray(body.position, np.float64)
        lo, hi, miss = 0.0, length, False
        for axis in range(3):
            if abs(ray[axis]) < 1e-9:
                if abs(eye[axis] - centre[axis]) > half[axis]:
                    miss = True
                    break
                continue
            a = (centre[axis] - half[axis] - eye[axis]) / ray[axis]
            b = (centre[axis] + half[axis] - eye[axis]) / ray[axis]
            lo = max(lo, min(a, b))
            hi = min(hi, max(a, b))
            if lo > hi:
                miss = True
                break
        if not miss and lo <= hi:
            return True
    return False


def before_medium_lands(spec, traj, bodies, margin: int = 1):
    """The frame just before the FIRST of `bodies` reaches what holds it up.

    For a scene made of many falling bodies, "a third of the way in" is not a
    moment -- it is after the event. Measured on `pour`, the grains land
    between frames 3 and 10 of 25, so the families that fired at frame 7 to 9
    were all intervening on a pile that had already settled: `solidity` let the
    grains bounce and only then sank them, `support` lifted a landed pile off
    the floor instead of catching it in the air, and `superelastic` multiplied
    the restitution of bodies that had stopped moving.

    Returns None when nothing lands, which is the caller's cue to keep whatever
    moment it had chosen.
    """
    first = None
    for body in bodies:
        try:
            bi = traj.index_of(int(body.segmentation_id))
        except Exception:                                     # noqa: BLE001
            continue
        top = surface_top(spec, body)
        r = float(traj.radius[bi])
        down = np.flatnonzero(
            np.asarray(traj.pos[:, bi, 2], np.float64) - r <= top + 1e-2)
        if down.size and (first is None or int(down[0]) < first):
            first = int(down[0])
    if first is None:
        return None
    t = first - int(margin)
    return int(t) if t >= 1 else None

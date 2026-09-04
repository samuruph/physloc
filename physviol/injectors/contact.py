"""Contact-domain injectors: solidity, superelastic, newton3_reaction.

What the three have in common is that they need a *contact event* to hang the
violation on -- you cannot break a collision before the collision happens. That
is why the injector API is two-phase: `plan` gets to look at the finished valid
rollout and pick its moment, and only then does `apply` edit anything.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..sim.trajectory import Trajectory
from . import _geom
from .base import Injector, InterventionPlan, register


def _extent_along(spec, partner_id: int, normal) -> float:
    """How far the partner reaches from its centre along the contact normal.

    A sphere's answer is its radius whichever way you ask; a wall's depends
    entirely on the direction, and using its bounding radius instead would
    claim a thin slab is a metre thick.
    """
    body = next((b for b in spec.bodies
                 if int(b.segmentation_id) == int(partner_id)), None)
    if body is None:
        return 0.0
    if body.kind == "cube":
        return float(np.abs(np.asarray(body.scale, np.float64)
                            * np.asarray(normal, np.float64)).sum())
    return float(body.bounding_radius)


def _top_of_partner(spec, traj, partner_id: int, frame: int) -> float:
    """World z of the top of whatever the actor just landed on.

    Taken at the contact frame and then held fixed. For a static floor that is
    exact; for the apex of a sphere pyramid it is an approximation good for the
    handful of frames a violation window lasts, and the plan records which case
    it was.
    """
    body = next((b for b in spec.bodies
                 if int(b.segmentation_id) == int(partner_id)), None)
    if body is None:
        return float(spec.floor_level)
    if body.static:
        return _geom.top_of(spec, body)
    bi = traj.index_of(int(partner_id))
    f = min(max(int(frame), 0), traj.num_frames - 1)
    return float(traj.pos[f, bi, 2] + traj.radius[bi])


def candidates_for(spec, family: str):
    """Bodies a family should prefer to act on, in order.

    A scenario may declare `notes["family_targets"][family]` when the obvious
    culprit is the wrong one. `pyramid_impact` is the case that forced it: the
    actor is the falling cube, but the interesting solidity failure is a
    *struck ball* driven through the ground, which is what the same collision
    looks like when a video generator gets it wrong.

    This stays inside the composition rule. The scenario declares a fact about
    itself, in the same `notes` dict it already uses for occlusion intervals and
    light directions; no injector learns a scenario's name, and a family with no
    declared preference falls back to the primary actor as before.
    """
    want = (spec.notes.get("family_targets") or {}).get(family)
    by_id = {int(b.segmentation_id): b for b in spec.bodies}
    if want:
        return [by_id[int(i)] for i in want if int(i) in by_id]
    return []


def _riding_on(spec, traj, actor, latest: int):
    """(frame, partner_id, True, normal) if the actor rests on a RAISED surface.

    "Raised" meaning static, and not the ground: a ramp, a table, a plinth. The
    body is on it from the start, so no impact will ever name it, yet it is the
    only surface whose failure is worth showing.
    """
    floors = {int(b.segmentation_id) for b in spec.bodies
              if b.static and getattr(b, "role", "") == "floor"}
    statics = [b for b in spec.bodies
               if b.static and int(b.segmentation_id) not in floors]
    if not statics:
        return None

    t0 = _geom.default_event_frame(spec, traj.num_frames)
    if t0 is None or not (1 <= t0 <= latest):
        return None
    bi = traj.index_of(int(actor.segmentation_id))
    p = np.asarray(traj.pos[t0, bi], np.float64)
    r = float(traj.radius[bi])

    for body in statics:
        centre = np.asarray(body.position, np.float64)
        half = np.asarray(body.scale, np.float64)
        gap = np.linalg.norm(np.maximum(np.abs(p - centre) - half, 0.0))
        if gap <= r * 1.35:
            return (int(t0), int(body.segmentation_id), True,
                    np.array([0.0, 0.0, 1.0]))
    return None


def _approach_speed(traj, actor_id: int, partner_id: int, t0: int) -> float:
    """Closing speed just before contact, from the lawful rollout.

    Relative for two moving bodies, absolute against a static one -- the
    coefficient of restitution is defined on whichever of those the collision
    actually has.
    """
    f = max(0, int(t0) - 1)
    ia = traj.index_of(int(actor_id))
    va = np.asarray(traj.lin_vel[f, ia], np.float64)
    try:
        ib = traj.index_of(int(partner_id))
    except Exception:                                         # noqa: BLE001
        return float(np.linalg.norm(va))
    vb = np.asarray(traj.lin_vel[f, ib], np.float64)
    if float(np.linalg.norm(vb)) < 1e-9:
        return float(np.linalg.norm(va))
    return float(np.linalg.norm(va - vb))


def _contact_event(spec, traj, actor):
    """(frame, partner_id, partner_is_static, normal) for the first collision.

    Prefers a real *impact* -- a contact the body is moving into -- and only
    falls back to any recorded contact, then to geometry. The ordering matters
    for a body that starts at rest: a pyramid sphere touches its neighbours from
    frame 0, so "first contact" is a nudge between two things standing still,
    while the event worth breaking is the one the falling cube causes.
    """
    dormant = tuple(int(b.segmentation_id) for b in spec.bodies if b.dormant)
    # An event needs room after it: a violation that fires two frames from the
    # end has nowhere to show its consequences, and on `stack_topple` the first
    # real impact is the blocks hitting the ground as the tower finishes
    # falling -- long after the interesting moment, which is the top block
    # sinking into the one below while the stack is still standing.
    latest = traj.num_frames - 3

    # A RAISED SURFACE the body is already riding on beats any later impact.
    # `ramp_slide`'s block rests on its ramp from frame 0, so the ramp is never
    # an "impact" and the first one is the block landing on the floor after it
    # has left the ramp entirely -- which made the clip a lawful slide followed
    # by a fall through the ground, rather than a block sinking through the
    # ramp it was sliding on. The floor is excluded because sinking through the
    # floor from the floor is the degenerate case `occluder_pass` was deferred
    # for.
    riding = _riding_on(spec, traj, actor, latest)
    if riding is not None:
        return riding

    impact = _geom.first_impact(traj, int(actor.segmentation_id), exclude=dormant)
    if impact is not None and 1 <= impact[0] <= latest:
        partner = next((b for b in spec.bodies
                        if int(b.segmentation_id) == impact[1]), None)
        return (int(impact[0]), int(impact[1]),
                bool(partner is not None and partner.static),
                np.asarray(impact[2], np.float64))

    hit = _geom.first_contact_any(traj, int(actor.segmentation_id),
                                  exclude=dormant)
    if hit is not None and 1 <= hit[0] <= latest:
        partner = next((b for b in spec.bodies
                        if int(b.segmentation_id) == hit[1]), None)
        return (int(hit[0]), int(hit[1]),
                bool(partner is not None and partner.static),
                np.array([0.0, 0.0, 1.0]))

    # Whatever is directly beneath, moving or not. Restricting this to *static*
    # surfaces meant a block resting on another block had no supporting surface
    # at all -- the search skipped the block under it and found the floor two
    # bodies down, which it is nowhere near -- so `stack_topple` could not host
    # a solidity violation despite being made of things stacked on each other.
    surface, top = _geom.support_under_any(spec, actor, traj, 0)
    if surface is None:
        return None
    bi = traj.index_of(int(actor.segmentation_id))
    below = np.flatnonzero((traj.pos[:, bi, 2] - traj.radius[bi]) <= top + 1e-4)
    if not below.size:
        return None
    if int(below[0]) == 0:
        t = traj.num_frames // 3
        if not (1 <= t < traj.num_frames - 1):
            return None
        return int(t), int(surface.segmentation_id), True, np.array([0.0, 0.0, 1.0])
    return (int(below[0]), int(surface.segmentation_id), True,
            np.array([0.0, 0.0, 1.0]))


class Solidity(Injector):
    """Suppress a collision pair so one body sinks into / through another.

    The knob is a **target penetration depth** in units of the actor's radius:

        weak   -> 0.30 r  (dips in, contact resumes, pushed back out)
        medium -> 0.80 r  (deeply swallowed, still recovered)
        strong -> 2.50 r  (centre passes the surface: through, for good)

    The bins straddle 1.0 r deliberately. Below one radius the actor's centre is
    still above the surface, so restoring contact can push it back out and the
    violation is a *transient* sinking. Past one radius there is nothing left to
    push against. That qualitative jump is what `strong` encodes.

    **Why the descent is prescribed rather than ballistic.** Letting the actor
    free-fall through the surface sounds more principled, but at 12 fps a
    dropped ball covers ~0.6 m per frame at impact, so the reachable depths are
    quantised far more coarsely than a radius: asking for 0.8 r yields 1.8 r and
    the bins collapse into each other. Since the injector already owns the
    trajectory, it places the actor at *exactly* the requested depth instead.
    That is what makes `magnitude` exact by construction, and it means the
    measured penetration residual must reproduce it -- which is a real
    end-to-end check on the residual pipeline, not a tautology.
    """

    family = "solidity"

    DEPTH_BY_BIN = {"weak": 0.30, "medium": 0.80, "strong": 2.50}
    #: How long the collision pair stays suppressed, in SECONDS. This IS the
    #: severity axis once the violation is staged: the pair is either enforced
    #: or it is not, so "how deep" is not a dial the simulator has -- but "for
    #: how long" is, and it produces exactly the qualitative ladder the depth
    #: bins were describing. A moment and the body dips in and is pushed back
    #: out when contact resumes; long enough and it is out the far side before
    #: anything can stop it, which is what `strong` has always meant.
    #:
    #: Left as a whole-clip suppression, every bin passed completely through and
    #: all three scored 1.00 -- three renders of one violation. Written in
    #: frames, it did the opposite at v0: ten frames is 0.83 s at the debug
    #: tier's 12 fps but 0.33 s at 30, and 0.33 s is not long enough for a ball
    #: to clear a wall -- so the pair came back while the ball was still inside
    #: it and the solver ejected it backwards. You reported exactly that: the
    #: strong bin did not pass through.
    SECONDS_BY_BIN = {"weak": 0.17, "medium": 0.34, "strong": 0.85}

    def strong_residual_reference(self, spec) -> float:
        # The penetration law reports depth in radii, which is exactly the unit
        # DEPTH_BY_BIN is expressed in -- so the reference is the bin value.
        return float(self.DEPTH_BY_BIN["strong"])

    # ------------------------------------------------------------------ #
    def plan(self, spec, traj: Trajectory, rng: np.random.RandomState,
             severity_bin: str) -> Optional[InterventionPlan]:
        group = self._group(spec)
        if len(group) > 1:
            return self._plan_group(spec, traj, group, severity_bin)
        choices = candidates_for(spec, self.family) or [self._primary(spec)]
        actor = event = None
        for body in choices:
            if body is None:
                continue
            found = _contact_event(spec, traj, body)
            if found is not None and 1 <= found[0] < traj.num_frames - 1:
                actor, event = body, found
                break
        if actor is None or event is None:
            return None
        t_contact, partner_id, partner_static, normal = event

        radius = actor.bounding_radius
        depth_r = self.DEPTH_BY_BIN[severity_bin]

        # FIRE ONE FRAME BEFORE THE RECORDED CONTACT. `t_contact` is the frame
        # the contact is *reported* on, and by then the solver has already
        # resolved most of it: on `barrier_pass` the ball's vx reads 0.93 m/s at
        # frame 12 and 0.30 m/s at frame 13, because it has bounced in between.
        # The staged path resets the world to `t_event` and disables the pair
        # from there, so firing on the contact frame hands the intervention a
        # ball that has already stopped -- which then creeps into the wall at a
        # quarter of its approach speed and is still inside it when the clip
        # ends. That is the "strong doesn't pass through" you saw; the ball was
        # never given the speed to get out the far side.
        t_fire = max(1, t_contact - 1)
        n_window = self._window_len(
            self._frames_for(spec, self.SECONDS_BY_BIN[severity_bin]),
            t_fire, traj.num_frames)
        t_end = t_fire + n_window - 1

        # The mode follows the contact *geometry*, not whether the partner is
        # static. Landing on something (near-vertical normal) can be expressed
        # as a prescribed depth below its top face; running into something --
        # a wall, another ball -- cannot, and trying to sink a body below the
        # top of a wall it hit side-on lifts it into the air instead.
        head_on = abs(float(normal[2])) < 0.6
        mode = "pass_through" if (head_on or not partner_static) else "sink"

        # A SUPPORT contact needs everything beneath the body disabled, not just
        # the surface it happened to touch first. On `ramp_slide` the block was
        # let through the floor while the ramp still held it, so it slid down
        # the ramp perfectly lawfully and only then fell.
        #
        # A HEAD-ON contact is the opposite case and must keep its floor: a ball
        # sent through a wall should come out the other side, not drop out of
        # the world. That is why this follows the contact normal rather than
        # whether the partner is static.
        extra_ids = ([int(b.segmentation_id) for b in spec.bodies
                      if b.static and int(b.segmentation_id) != int(partner_id)]
                     if (mode == "sink" and partner_static) else [])

        notes = {"radius": float(radius),
                 "surface_top": _top_of_partner(spec, traj, partner_id, t_contact),
                 "partner_id": int(partner_id),
                 "partner_static": bool(partner_static),
                 "partner_dynamic": bool(not partner_static),
                 "contact_normal": [float(x) for x in normal],
                 "pass_through": bool(mode == "pass_through"),
                 "partner_extent": _extent_along(spec, partner_id, normal),
                 "t_contact": int(t_contact),
                 "passes_through": bool(depth_r >= 1.0)}
        if extra_ids:
            # Every surface beneath the body is suppressed, so it falls out of
            # the world rather than off the side of something. `support_bounds`
            # exists to stop a mug knocked off a table reading as a solidity
            # violation for the rest of the clip, and that is the right default
            # -- but here the body really did go through, and bounding the
            # depth to where it is still over the surface returned exactly 0.000
            # on a clip where the block visibly sinks to z = -1.6.
            notes["support_bounds"] = None
            notes["surface_top"] = float(_geom.surface_top(spec, actor))
        if mode == "pass_through":
            # Against another moving body there is no surface to place the
            # actor a prescribed depth below. Suppressing the pair's response
            # for the window is both simpler and closer to what the family
            # claims -- "disable a collision pair" -- and it produces the thing
            # video generators actually get wrong: two objects occupying the
            # same space and coming out the other side.
            strong = self._passed_through(
                spec, traj, actor, partner_id, t_contact,
                self._frames_for(spec, self.SECONDS_BY_BIN["strong"]))
            notes["r_strong"] = self._measure(
                strong, int(actor.segmentation_id), "penetration", notes)

        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t_fire,
            windows=[(t_fire, t_end)],
            causal_body_ids=[int(actor.segmentation_id), int(partner_id)],
            params={"type": "disable_collision_pair",
                    "pair": [int(actor.segmentation_id), int(partner_id)],
                    "frames_disabled": int(n_window),
                    "mode": mode,
                    "also_disable": extra_ids,
                    # A bin that means "through, for good" cannot end on a
                    # clock. Restoring the pair while the two are still
                    # overlapping hands the solver an interpenetration to
                    # resolve, and it resolves it the only way it can: by
                    # ejecting the actor back the way it came. That is what you
                    # saw -- the strong bin bouncing off a wall it was supposed
                    # to pass through -- and no duration tuned for one scene's
                    # approach speed fixes it for the next one. So past a
                    # radius, the pair comes back when the bodies are
                    # geometrically clear of each other, which is a fact about
                    # the run rather than a guess made before it.
                    "restore_when_clear": bool(depth_r >= 1.0),
                    "target_depth_radii": depth_r},
            magnitude=float(depth_r * radius),
            magnitude_unit="m_penetration_depth",
            severity_bin=severity_bin,
            notes=notes,
        )

    def _plan_group(self, spec, traj, bodies, severity_bin):
        """Every body in the group sinks through whatever holds it up, together.

        For a scene made of many interchangeable bodies, one grain dropping
        through the floor is a few pixels nobody will find. A whole pour going
        through at once is unmistakable, and it is also the more honest staging
        of "this surface stopped being solid" -- a floor does not lose its
        solidity for one grain.
        """
        T = traj.num_frames
        # If something *outside* the group runs into it, that arrival is the
        # moment -- the pyramid's spheres should drop through the floor when the
        # cube lands on them, not a third of the way through the clip for no
        # visible reason. Absent such a body, fall back to the usual fraction.
        t0 = self._group_trigger(spec, traj, bodies)
        if t0 is None:
            t0 = _geom.default_event_frame(spec, T)
        if t0 is None:
            return None
        depth_r = self.DEPTH_BY_BIN[severity_bin]
        n_win = self._window_len(
            self._frames_for(spec, self.SECONDS_BY_BIN[severity_bin]), t0, T)
        t_end = min(T - 1, t0 + n_win - 1)
        lead = bodies[0]
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=[(t0, t_end)],
            causal_body_ids=[int(b.segmentation_id) for b in bodies],
            params={"type": "disable_collision_pair", "mode": "sink_group",
                    "frames_disabled": int(n_win),
                    "target_depth_radii": depth_r},
            magnitude=float(depth_r * lead.bounding_radius),
            magnitude_unit="m_penetration_depth", severity_bin=severity_bin,
            notes={"radius": float(lead.bounding_radius),
                   "surface_top": _geom.surface_top(spec, lead),
                   "t_contact": int(t0), "group": True,
                   "passes_through": bool(depth_r >= 1.0)})

    @staticmethod
    def _group_trigger(spec, traj, bodies):
        """First frame a body outside the group touches one inside it."""
        inside = {int(b.segmentation_id) for b in bodies}
        outside = {int(b.segmentation_id) for b in spec.bodies
                   if not b.static and not b.dormant
                   and int(b.segmentation_id) not in inside}
        if not outside:
            return None
        c = traj.contacts
        best = None
        for k in range(len(c)):
            a, b = int(c.body_a[k]), int(c.body_b[k])
            hit = ((a in inside and b in outside)
                   or (b in inside and a in outside))
            if not hit:
                continue
            f = int(c.frame[k])
            if 1 <= f < traj.num_frames - 2 and (best is None or f < best):
                best = f
        return best

    def _sink_group(self, spec, traj, plan):
        """The surface stops being solid; everything else stays exactly as it was.

        The floor is simply removed for these bodies from `t_event` on, and they
        are re-integrated under ordinary gravity. Prescribing a depth for the
        whole group instead -- which is what this did first -- made every grain
        in a pour start sinking on the same frame, so forty grains still in
        mid-air dropped through a floor none of them had reached yet. Taking the
        floor away lets each body arrive and pass through on its own schedule,
        under the same physics it had a frame earlier, which is the only version
        that looks like anything.
        """
        out = self._clone(traj)
        t0 = plan.windows[0][0]
        for bid in plan.causal_body_ids:
            body = next(b for b in spec.bodies
                        if int(b.segmentation_id) == int(bid))
            # `solid=False` for the obstacles and a floor far below: nothing to
            # land on, everything else unchanged.
            self._rewrite_from(spec, traj, out, body, t0,
                               obstacles=None, solid=False, floorless=True)
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out

    def _passed_through(self, spec, traj, actor, partner_id, t0, n_win):
        """Both bodies ignore each other for `n_win` frames, then resume.

        Neither is teleported: each simply carries on with the velocity it had
        the frame before contact, so they interpenetrate at the closing speed
        the scene already had and separate again when the pair is restored.
        """
        out = self._clone(traj)
        partner = next(b for b in spec.bodies
                       if int(b.segmentation_id) == int(partner_id))
        t1 = min(traj.num_frames - 1, t0 + n_win - 1)
        # The whole pair, for both bodies. Excluding only the body being
        # resumed left the *other* one still an obstacle, so a struck ball that
        # had stayed put all clip got shoved half a metre sideways the moment
        # the window closed -- the target of a pass-through moving is precisely
        # what the family says does not happen.
        pair = [int(actor.segmentation_id), int(partner_id)]
        for body in (actor, partner):
            if body.static:
                continue          # a wall does not get out of the way
            bi = traj.index_of(int(body.segmentation_id))
            v_pre = traj.lin_vel[t0 - 1, bi].astype(np.float64)
            # `solid=False` for the window: this is the one family whose whole
            # point is going through things, so handing it the scene's obstacle
            # set would politely undo the intervention. Contact resumes
            # afterwards -- with obstacles back on -- so the body is lawful
            # again rather than permanently ghostly.
            self._rewrite_from(spec, traj, out, body, t0, v0=v_pre, solid=False)
            if t1 + 1 < traj.num_frames:
                # Carry on from wherever the pass-through left it, rather than
                # snapping back onto the lawful path -- and with the body it
                # just went through left out of the obstacle set. Restoring that
                # pair while the two are still overlapping ejects one of them in
                # a single substep, which is a position jump large enough to
                # register as a teleport: the clip would then depict a solidity
                # failure *and* a continuity failure while claiming one.
                self._rewrite_from(
                    spec, out, out, body, t1 + 1,
                    v0=out.lin_vel[t1, bi].astype(np.float64),
                    obstacles=_geom.Obstacles(spec, out, exclude_ids=pair))
        c = out.contacts
        pair = {int(actor.segmentation_id), int(partner_id)}
        if len(c):
            drop = np.array([(t0 <= int(f) <= t1) and ({int(a), int(b)} == pair)
                             for f, a, b in zip(c.frame, c.body_a, c.body_b)], bool)
            keep = ~drop
            out.contacts = type(c)(c.frame[keep], c.body_a[keep], c.body_b[keep],
                                   c.point[keep], c.normal[keep], c.impulse[keep],
                                   c.penetration[keep])
        return out

    # ------------------------------------------------------------------ #
    #: Only the two-body mode. `sink_group` removes the floor under ~40 grains
    #: at once, which is a scene edit rather than one disabled pair, and it
    #: stays on the trajectory path.
    simulated = True

    #: Extra frames kept after the bodies separate, so the clip does not cut
    #: on the exact frame of last contact.
    CLEAR_TAIL = 1

    def refine_windows(self, spec, traj_valid, traj_invalid, plan) -> None:
        """End the window when the bodies stop overlapping, not after N frames.

        You reported the severity mask going to zero while the ball was still
        inside the barrier. It was: the window length came from a per-bin frame
        count, so it expired mid-pass and the mask -- which is gated on the
        window -- went with it. Overlap is a geometric fact about the finished
        trajectory, so it is measured here rather than guessed in `plan()`.
        """
        pair = plan.params.get("pair")
        if not pair or len(pair) != 2:
            return
        a_id, b_id = int(pair[0]), int(pair[1])
        try:
            ia = traj_invalid.index_of(a_id)
        except Exception:                                     # noqa: BLE001
            return
        body_b = next((b for b in spec.bodies
                       if int(b.segmentation_id) == b_id), None)
        if body_b is None:
            return

        T = traj_invalid.num_frames
        t0 = plan.t_event
        ra = float(traj_invalid.radius[ia])
        pos_a = np.asarray(traj_invalid.pos[:, ia, :], np.float64)

        if body_b.static:
            centre = np.asarray(body_b.position, np.float64)
            half = np.asarray(body_b.scale, np.float64)
            delta = np.abs(pos_a - centre[None, :]) - half[None, :]
            outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
            overlapping = outside < ra
        else:
            ib = traj_invalid.index_of(b_id)
            rb = float(traj_invalid.radius[ib])
            gap = np.linalg.norm(
                pos_a - np.asarray(traj_invalid.pos[:, ib, :], np.float64),
                axis=1)
            overlapping = gap < (ra + rb)

        frames = np.flatnonzero(overlapping)
        frames = frames[frames >= t0]
        if not frames.size:
            return
        t_end = int(min(T - 1, int(frames[-1]) + self.CLEAR_TAIL))
        if t_end <= t0:
            return
        plan.windows = [(t0, t_end)]
        plan.intervention_windows = [(t0, t_end)]
        # And the SEVERITY window ends with the overlap too. Widening this to
        # the end of the clip -- which is what the causal mask wanted -- put
        # severity on every frame after the ball emerged, so the field marked
        # where the body ENDED UP rather than where it was passing through. You
        # reported exactly that on `drop`. The causal mask gets its long tail
        # from the measured trajectory divergence in `annotate.pipeline`
        # instead, which does not drag severity along with it.
        plan.consequence_windows = [(t0, t_end)]

    def simulates(self, plan) -> bool:
        return (plan.params.get("mode") != "sink_group"
                and len(plan.params.get("pair", ())) == 2)

    def _stageable(self, plan) -> bool:
        return self.simulates(plan)

    def stage(self, spec, simulator, objs, plan):
        """Turn the collision pair off and let the body go where it goes.

        The violation is that two surfaces did not stop each other, so the
        honest way to say it is to stop PyBullet from enforcing that pair --
        not to write in an overlap and hope the re-integrator does not undo it.
        Every surface beneath a body it was RESTING on goes too, so a block on a
        ramp sinks through the ramp rather than sliding down it and only then
        falling through the floor.
        """
        if not self._stageable(plan):
            return ()
        import pybullet as pb

        pairs = self._filter_pairs(spec, simulator, objs, plan)
        for ia, ib in pairs:
            pb.setCollisionFilterPair(ia, ib, -1, -1, 0)

        # Restore the pair once the window closes, so the bins differ. Without
        # this the suppression lasted the whole clip and weak, medium and strong
        # were the same clip three times.
        t_end = plan.t_event + int(plan.params.get("frames_disabled", 4))
        when_clear = bool(plan.params.get("restore_when_clear"))
        state = {"restored": False}

        def clear() -> bool:
            """True once nothing the pair covers is still overlapping."""
            for ia, ib in pairs:
                if pb.getClosestPoints(ia, ib, distance=0.0):
                    return False
            return True

        def restore(_client, _step, frame):
            if state["restored"] or frame < t_end:
                return
            # The weak and medium bins WANT the pair back while the bodies still
            # overlap -- being shoved back out is the whole shape of a transient
            # sinking. Only the bins that claim to go all the way through wait
            # for the far side.
            if when_clear and not clear():
                return
            for ia, ib in pairs:
                pb.setCollisionFilterPair(ia, ib, -1, -1, 1)
            state["restored"] = True

        return (restore,)

    def _filter_pairs(self, spec, simulator, objs, plan):
        """Every (actor, surface) pair this violation suppresses."""
        from ..render import stepper

        pair = plan.params.get("pair")
        if not pair or len(pair) != 2:
            return []
        a, b = pair
        ia = stepper.pybullet_index(simulator, objs, spec, int(a))
        if ia is None:
            return []
        others = [int(b)] + [int(x) for x in plan.params.get("also_disable", ())]
        out = []
        for other in others:
            ib = stepper.pybullet_index(simulator, objs, spec, other)
            if ib is not None:
                out.append((ia, ib))
        return out

    def unstage(self, spec, simulator, objs, plan) -> None:
        if not self._stageable(plan):
            return
        import pybullet as pb
        from ..render import stepper

        for ia, ib in self._filter_pairs(spec, simulator, objs, plan):
            pb.setCollisionFilterPair(ia, ib, -1, -1, 1)

    def _apply(self, spec, traj: Trajectory, plan: InterventionPlan) -> Trajectory:
        actor_id = int(plan.causal_body_ids[0])
        actor = next(b for b in spec.bodies
                     if int(b.segmentation_id) == actor_id)
        t0, t1 = plan.windows[0]

        if plan.params.get("mode") == "sink_group":
            return self._sink_group(spec, traj, plan)
        if plan.params.get("mode") == "pass_through":
            out = self._passed_through(spec, traj, actor,
                                       int(plan.notes["partner_id"]), t0,
                                       t1 - t0 + 1)
            out.meta = dict(traj.meta)
            out.meta["intervention"] = plan.to_dict()
            out.meta["label"] = "invalid"
            return out

        out = self._clone(traj)
        bi = traj.index_of(actor_id)
        n_win = t1 - t0 + 1

        radius = float(plan.notes["radius"])
        top = float(plan.notes["surface_top"])
        target = float(plan.magnitude)

        p_prev = traj.pos[t0 - 1, bi].astype(np.float64)
        v_prev = traj.lin_vel[t0 - 1, bi].astype(np.float64)

        # Lateral motion carries on undisturbed; only the vertical axis is
        # prescribed, easing from "just touching" to exactly `target` deep.
        tt = (np.arange(1, n_win + 1, dtype=np.float64) * traj.dt)[:, None]
        xy = p_prev[None, :2] + v_prev[None, :2] * tt

        frac = np.linspace(1.0 / n_win, 1.0, n_win) ** 1.6   # ease-in
        depth = target * frac
        z = top + radius - depth

        out.pos[t0:t1 + 1, bi, 0:2] = xy.astype(np.float32)
        out.pos[t0:t1 + 1, bi, 2] = z.astype(np.float32)
        dz = np.diff(np.concatenate([[float(traj.pos[t0 - 1, bi, 2])], z]))
        out.lin_vel[t0:t1 + 1, bi, 0:2] = v_prev[None, :2].astype(np.float32)
        out.lin_vel[t0:t1 + 1, bi, 2] = (dz / traj.dt).astype(np.float32)
        out.ang_vel[t0:, bi, :] = traj.ang_vel[t0 - 1, bi][None, :]

        # ---- after the window, contact is restored -----------------------
        n_rest = traj.num_frames - (t1 + 1)
        if n_rest > 0:
            p_end = np.array([xy[-1, 0], xy[-1, 1], z[-1]], dtype=np.float64)
            v_end = np.array([v_prev[0], v_prev[1], float(dz[-1] / traj.dt)])
            if z[-1] > top:
                # Centre still above the surface: the surface shoves it back
                # out, and it must bounce properly from there or a later frame
                # would sink through again and fake a second violation.
                p_end[2] = top + radius
                v_end[2] = abs(v_end[2]) * float(actor.restitution)
                pos2, vel2 = self._integrate_profile(
                    p_end, v_end,
                    np.tile(traj.gravity.astype(np.float64)[None, :], (n_rest, 1)),
                    traj.dt, top, radius, float(actor.restitution))
            else:
                # Centre is past the surface: nothing left to push against, so
                # it keeps falling and the violation persists by consequence.
                pos2, vel2 = self._ballistic(
                    p_end, v_end, traj.gravity.astype(np.float64), traj.dt, n_rest)
            out.pos[t1 + 1:, bi, :] = pos2
            out.lin_vel[t1 + 1:, bi, :] = vel2

        # Contacts for the suppressed pair no longer occur during the window.
        c = out.contacts
        pair = set(plan.params["pair"])
        if len(c):
            drop = np.array([(t0 <= int(f) <= t1) and ({int(a), int(b)} == pair)
                             for f, a, b in zip(c.frame, c.body_a, c.body_b)], bool)
            keep = ~drop
            out.contacts = type(c)(c.frame[keep], c.body_a[keep], c.body_b[keep],
                                   c.point[keep], c.normal[keep], c.impulse[keep],
                                   c.penetration[keep])

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class SuperElastic(Injector):
    """Restitution greater than one: every bounce comes back faster than it went in.

    The taxonomy's only `repeated` family, and the reason `violation_windows` is
    a *list* rather than an onset. One boosted restitution produces a gain at
    every subsequent contact, so the clip has several disjoint violation
    intervals with lawful flight between them. A schema with a single `t_event`
    would have to pick one and silently discard the rest.

    Both bodies in a two-body collision are boosted. Rescaling only one would
    add a momentum violation on top of the energy one, and a clip carrying two
    violations while annotating one is worse than useless.
    """

    family = "superelastic"
    #: Outgoing speed as a multiple of incoming. Raised twice from 1.2/1.55/2.1,
    #: because the ceiling those numbers were chosen against was the wrong one:
    #: they were low enough that a vertical bounce stayed in shot, which made
    #: every *horizontal* rebound -- `barrier_pass`, `collision` -- barely
    #: distinguishable from its lawful twin. The framing constraint belongs on
    #: the individual clip, not on the constant, so the nominal gain is what
    #: reads clearly and `_fit_to_frame` walks it back per scene wherever the
    #: geometry cannot host it.
    #:
    #: The whole ladder moves together rather than only its top, because you
    #: judged the strongest bin about right for a weak one -- a stronger strong
    #: over an unchanged weak would just stretch the gap and leave the bottom of
    #: the ladder invisible.
    GAIN_BY_BIN = {"weak": 2.20, "medium": 3.60, "strong": 6.00}
    #: Frames the pair may spend off camera before the fit weakens the bounce.
    #:
    #: The default of one was the binding constraint on `barrier_pass`, not the
    #: nominal gain: measured there, gain 3.5 costs one frame off camera and 4.5
    #: costs four, so a budget of one clamped the clip to 3.48 however high the
    #: constant went. A ball that comes back much faster than it arrived SHOULD
    #: leave the shot sooner -- that is what the violation looks like -- and the
    #: cost is a few frames of empty mask at the tail.
    FRAME_TOLERANCE = 4

    def strong_residual_reference(self, spec) -> float:
        # The law reports fractional kinetic-energy gain, and energy goes as
        # speed squared, so a speed gain of k shows up as k^2 - 1.
        return float(self.GAIN_BY_BIN["strong"] ** 2 - 1.0)

    def _boosted(self, spec, traj, targets, t0, gain, normal=None) -> Trajectory:
        """Reflect each body's *incoming* velocity with restitution `gain` > 1.

        Amplifying the outgoing velocity instead is the obvious implementation
        and it is wrong wherever the impact is heavily damped: a 2.2 kg cube
        landing on a pyramid of 0.8 kg spheres has essentially stopped by the
        next sampled frame, and 2.1x of nothing is nothing -- the clip shipped
        with a total-energy residual of exactly zero. Restitution is a statement
        about the ratio of outgoing to *incoming* speed, so that is what this
        computes, and the resulting energy gain is k^2 - 1 no matter how the
        simulator resolved the collision.
        """
        out = self._clone(traj)
        v_by_body = {}
        for body in targets:
            bi = traj.index_of(int(body.segmentation_id))
            v_in = traj.lin_vel[t0 - 1, bi].astype(np.float64)
            n = (np.asarray(normal, np.float64) if normal is not None
                 else self._contact_normal(traj, t0, int(body.segmentation_id),
                                           v_in))
            # Reflection is quadratic in the normal, so its sign does not
            # matter -- only that it is the *impact's* normal.
            v_out = v_in - (1.0 + gain) * float(np.dot(v_in, n)) * n
            out.lin_vel[t0, bi, :] = v_out.astype(np.float32)
            v_by_body[int(body.segmentation_id)] = v_out

        if len(targets) > 2:
            # A whole medium bouncing: step it together. Boosting each grain
            # against the others' *lawful* paths ejected the ones buried in the
            # pile, and an ejection is a position jump large enough to read as
            # a teleport -- a second violation inside an energy clip.
            self._rewrite_group(spec, traj, out, targets, t0 + 1,
                                v0_by_body=v_by_body)
        else:
            for body in targets:
                self._rewrite_from(
                    spec, traj, out, body, t0 + 1,
                    v0=v_by_body[int(body.segmentation_id)],
                    restitution=min(0.98, float(body.restitution) * gain))
        return out

    @staticmethod
    def _contact_normal(traj, frame: int, body_id: int, v_in) -> np.ndarray:
        """The normal of the frame's most head-on contact for this body.

        Taking the *first* matching contact is what made a ball-ball collision
        reflect about the floor: two balls rolling into each other are touching
        the ground on the same frame, and the ground's normal is perpendicular
        to their motion, so the reflection was the identity and the "bouncier"
        ball sailed straight through its partner. Scoring by how much of the
        body's velocity lies along each normal picks the collision that is
        actually happening.
        """
        best, best_score = np.array([0.0, 0.0, 1.0]), -1.0
        c = traj.contacts
        for k in range(len(c)):
            if int(c.frame[k]) != int(frame):
                continue
            if body_id not in (int(c.body_a[k]), int(c.body_b[k])):
                continue
            cand = np.asarray(c.normal[k], np.float64)
            mag = float(np.linalg.norm(cand))
            if mag < 1e-6:
                continue
            cand = cand / mag
            score = abs(float(np.dot(v_in, cand)))
            if score > best_score:
                best, best_score = cand, score
        return best

    def _targets(self, spec, actor, partner_id):
        group = self._group(spec)
        if len(group) > 1:
            # A scene of interchangeable bodies bounces as a whole. One grain
            # in forty coming back higher than it should is not something a
            # viewer -- or a model -- has any way to notice.
            return group
        out = [actor]
        partner = next((b for b in spec.bodies
                        if int(b.segmentation_id) == int(partner_id)), None)
        if partner is not None and not partner.static:
            out.append(partner)
        return out

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = self._primary(spec)
        if actor is None:
            return None
        # An impact, not merely a contact: reflecting the normal component of a
        # rolling ball's velocity along the floor changes nothing, and the clip
        # would ship claiming a super-elastic bounce that never happened.
        dormant = tuple(int(b.segmentation_id) for b in spec.bodies if b.dormant)

        # A BALL-BALL impact beats a floor bounce where the scene has one. On
        # `collision` the family was boosting the landing instead of the meeting
        # -- the ball shot straight up and out of frame, which is neither what
        # the scenario is built around nor something a viewer can follow. The
        # same preference `_CollisionEdit` uses, for the same reason.
        pair = _geom.first_dynamic_pair_contact(
            spec, traj, prefer=int(actor.segmentation_id))
        impact = None
        if pair is not None:
            t_pair, id_a, id_b = pair
            if 1 <= t_pair < traj.num_frames - 2:
                mine = int(actor.segmentation_id)
                other = id_b if mine == id_a else id_a
                if mine in (id_a, id_b):
                    d = (traj.pos[t_pair, traj.index_of(int(other))]
                         - traj.pos[t_pair, traj.index_of(mine)])
                    mag = float(np.linalg.norm(d))
                    if mag > 1e-6:
                        impact = (int(t_pair), int(other),
                                  np.asarray(d, np.float64) / mag)
        if impact is None:
            impact = _geom.first_impact(traj, int(actor.segmentation_id),
                                        exclude=dormant)
        if impact is None:
            return None
        t0, partner_id, normal = impact
        if not (1 <= t0 < traj.num_frames - 2):
            return None

        targets = self._targets(spec, actor, partner_id)
        # KEEP THE BOUNCE IN SHOT. A gain that reads well on a ball rebounding
        # across the frame throws a dropped one clean out of the top of it, and
        # a body that leaves frame has an empty mask for the rest of the clip --
        # the clip then depicts an object vanishing while claiming an energy
        # violation. Fitted on the STRONGEST bin so all three share one scale
        # and stay ordered, and the knob scaled is the *excess* over 1, since a
        # gain of 1 is a lawful bounce rather than no bounce at all.
        excess = self.GAIN_BY_BIN["strong"] - 1.0
        fit, _ = self._fit_to_frame(
            spec, traj, targets, t0, excess,
            lambda k: self._boosted(spec, traj, targets, t0,
                                    1.0 + excess * k, normal),
            tolerance=self.FRAME_TOLERANCE)
        gain = 1.0 + (self.GAIN_BY_BIN[severity_bin] - 1.0) * fit
        # Roll the boosted trajectory forward here so the windows can be the
        # frames energy is *actually* gained on, rather than a guess. `apply`
        # recomputes the same thing deterministically.
        preview = self._boosted(spec, traj, targets, t0, gain, normal)
        windows = self._gain_windows(preview, targets, traj, t0)

        # The reference has to be measured, not declared. Boosting the speed by
        # k multiplies *kinetic* energy by k^2, but the law scores total
        # mechanical energy, so the achievable gain depends on how much of the
        # body's energy is potential when it bounces. A cube striking the apex
        # of a pyramid is most of a metre up and carries a large potential
        # term, so a declared reference of k^2 - 1 is unreachable there and
        # every clip in that scenario scores a fraction of what it deserves.
        top = _top_of_partner(spec, traj, partner_id, t0)
        strong_preview = self._boosted(spec, traj, targets, t0,
                                       1.0 + excess * fit, normal)
        r_strong = self._measure(strong_preview, int(actor.segmentation_id),
                                 "energy_at_contact", {"surface_top": top})

        return InterventionPlan(
            family=self.family, kind="repeated", t_event=windows[0][0],
            windows=windows,
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "restitution_gain", "speed_gain": gain,
                    "energy_gain_ratio": gain ** 2 - 1.0,
                    "frame_fit_scale": float(fit),
                    "first_contact_frame": int(t0)},
            magnitude=float(gain ** 2 - 1.0),
            magnitude_unit="energy_gain_ratio", severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "partner_id": int(partner_id),
                   # Both participants, so `energy_at_contact` can score the
                   # pair rather than one body. A moving partner carries away
                   # part of the gain and neither half shows it alone.
                   "pair_ids": [int(actor.segmentation_id), int(partner_id)],
                   # The approach speed, measured from the VALID rollout rather
                   # than observed at run time. The world is reset to t_event,
                   # and at t_event the bodies are already touching -- so a hook
                   # that waits to see a free frame before contact never sees
                   # one, records an approach of zero, and boosts by nothing.
                   "approach_speed": float(_approach_speed(
                       traj, int(actor.segmentation_id), int(partner_id), t0)),
                   "surface_top": top,
                   "speed_gain": gain, "n_bounces": len(windows),
                   "r_strong": float(r_strong),
                   "impact_normal": [float(x) for x in normal],
                   "targets": [int(b.segmentation_id) for b in targets]})

    @staticmethod
    def _gain_windows(preview, targets, traj, t0) -> List[Tuple[int, int]]:
        """Frames where the boosted rollout gains kinetic energy it should not."""
        T = traj.num_frames
        flag = np.zeros((T,), bool)
        for body in targets:
            bi = traj.index_of(int(body.segmentation_id))
            v = np.linalg.norm(preview.lin_vel[:, bi, :].astype(np.float64), axis=1)
            gained = np.zeros((T,), bool)
            gained[1:] = v[1:] > v[:-1] * 1.05 + 1e-3
            gained[:t0] = False
            flag |= gained
        # The first contact is a superelastic bounce by construction, whether
        # or not the speed test notices it on a slow one. It also has to be in
        # the list for `t_event` to equal the first frame this injector edits --
        # anything later would declare a prefix longer than the one it kept.
        out, f = [(t0, min(T - 1, t0 + 1))], 0
        while f < T:
            if flag[f]:
                end = min(T - 1, f + 1)
                out.append((f, end))
                f = end + 1
            else:
                f += 1
        # Merge anything that touches, so the windows stay sorted and disjoint
        # -- `physviol validate` rejects overlaps, and a bounce two frames after
        # another is one violation, not two.
        merged: List[Tuple[int, int]] = []
        for s, e in sorted(out):
            if merged and s <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    #: Expressed as a per-step velocity boost, not a restitution coefficient.
    #: **PyBullet clamps restitution to 1.** Measured, not assumed: staged as
    #: `changeDynamics(restitution=e*gain)` the clip came out with the valid
    #: twin's energy to two decimals -- E_end 3.73 J against 3.73 J, an energy
    #: gain of 0.00 and a severity of 0.000. A coefficient greater than one is
    #: simply not something the solver will honour, so the extra speed has to
    #: be handed to the body at the moment it separates.
    simulated = True

    def stage(self, spec, simulator, objs, plan):
        """Return the collision's energy instead of absorbing it.

        Superelastic means the coefficient of restitution exceeds one: the pair
        separates faster than it met, so the contact gives back more than it
        took. Two shapes, because the quantity differs:

        * against a STATIC partner it is the body's outgoing speed over its
          incoming speed;
        * against a MOVING partner it is the pair's RELATIVE separation speed
          over their relative approach speed, and the correction is split evenly
          so momentum stays conserved. The violation is that the collision
          returned more energy than it took, not that momentum appeared.

        The contact lasts a single substep -- measured with
        `probe_superelastic.py`: the balls approach at 1.292 m/s, and by the one
        step where PyBullet reports a contact the solver has already resolved it
        to 0.727. So the boost cannot be applied "during" the collision; it is
        applied on the step after contact ends, against the approach speed
        recorded while the bodies were still free.
        """
        import pybullet as pb

        from ..render import stepper

        gain = float(plan.params["speed_gain"])
        actor_id = int(plan.causal_body_ids[0])
        idx = stepper.pybullet_index(simulator, objs, spec, actor_id)
        if idx is None:
            return ()

        partner_seg = plan.notes.get("partner_id")
        partner_body = next((b for b in spec.bodies
                             if int(b.segmentation_id) == int(partner_seg)),
                            None) if partner_seg is not None else None
        partner = (stepper.pybullet_index(simulator, objs, spec,
                                          int(partner_seg))
                   if partner_seg is not None else None)
        moving = bool(partner is not None and partner_body is not None
                      and not partner_body.static)
        state = {"ready": True, "touching": False,
                 "approach": float(plan.notes.get("approach_speed", 0.0))}
        def _points():
            pts = pb.getContactPoints(bodyA=idx)
            if partner is None:
                return pts
            return [c for c in pts if partner in (c[1], c[2])]

        def boost(_client, _step, _frame):
            points = _points()
            va = np.asarray(pb.getBaseVelocity(idx)[0], np.float64)
            wa = pb.getBaseVelocity(idx)[1]

            if moving:
                vb = np.asarray(pb.getBaseVelocity(partner)[0], np.float64)
                wb = pb.getBaseVelocity(partner)[1]
                rel = va - vb
                if points:
                    state["touching"] = True
                    return
                if not state["touching"]:
                    state["approach"] = float(np.linalg.norm(rel))
                    return
                state["touching"] = False
                mag = float(np.linalg.norm(rel))
                extra = state["approach"] * gain - mag
                if not state["ready"]:
                    return
                if mag < 1e-6 or extra <= 0.0:
                    return
                n = rel / mag
                pb.resetBaseVelocity(idx, (va + n * extra * 0.5).tolist(),
                                     list(wa))
                pb.resetBaseVelocity(partner, (vb - n * extra * 0.5).tolist(),
                                     list(wb))
                state["ready"] = False
                return

            if not points:
                state["ready"] = True
                state["approach"] = float(np.linalg.norm(va))
                return
            if not state["ready"]:
                return
            n = np.mean([np.asarray(c[7], np.float64) for c in points], axis=0)
            mag = float(np.linalg.norm(n))
            if mag < 1e-9:
                return
            n /= mag
            out_speed = float(va @ n)
            if out_speed <= 1e-3:
                return
            pb.resetBaseVelocity(
                idx, (va + n * (state["approach"] * gain - out_speed)).tolist(),
                list(wa))
            state["ready"] = False

        return (boost,)

    def _apply(self, spec, traj, plan) -> Trajectory:
        actor = self._primary(spec)
        targets = self._targets(spec, actor, plan.notes["targets"][-1])
        out = self._boosted(spec, traj, targets,
                            int(plan.params["first_contact_frame"]),
                            float(plan.params["speed_gain"]),
                            plan.notes.get("impact_normal"))
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class _CollisionEdit(Injector):
    """Shared machinery for the two ways a collision can come out wrong.

    Both need the same thing -- the first contact between two bodies that both
    ought to respond -- and differ only in what outcome they substitute, so the
    collision-finding lives here and each subclass supplies `_outcome`.

    The normal is taken as the line of centres at the contact frame rather than
    from the recorded contact. That is exact for spheres, close enough for a
    cube landing on one, and it sidesteps the orientation question entirely:
    `Contacts.normal` points from `body_a` toward `body_b`, so using it
    correctly means knowing which of the two you are looking at, and getting
    that backwards silently mirrors the collision.
    """

    STRONG_REFERENCE = 3.0            # |dv| / (g*dt) that the pair fails to show

    def strong_residual_reference(self, spec) -> float:
        return self.STRONG_REFERENCE

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        eligible = (spec.notes.get("family_targets") or {}).get(self.family)
        actor = self._primary(spec)
        actor_id = int(actor.segmentation_id) if actor is not None else None
        hit = _geom.first_dynamic_pair_contact(spec, traj, prefer=actor_id,
                                               only=eligible)
        if hit is None:
            return None
        t0, id_a, id_b = hit
        if not (1 <= t0 < traj.num_frames - 1):
            return None
        if actor_id is None or actor_id not in (id_a, id_b):
            actor_id = id_a
        other_id = id_b if actor_id == id_a else id_a

        ia, ib = traj.index_of(actor_id), traj.index_of(other_id)
        normal = traj.pos[t0, ib] - traj.pos[t0, ia]
        mag = float(np.linalg.norm(normal))
        if mag < 1e-6:
            return None
        normal = (normal / mag).astype(np.float64)

        ratio = float(self.RATIO_BY_BIN[severity_bin])
        preview = self._edited(spec, traj, actor_id, other_id, t0, normal, ratio)
        strong = self._edited(spec, traj, actor_id, other_id, t0, normal,
                              float(self.RATIO_BY_BIN["strong"]))
        r_strong = max(
            self._measure(strong, cid, "linear_momentum", {})
            for cid in (actor_id, other_id))
        del preview

        g_dt = float(np.linalg.norm(traj.gravity)) * traj.dt
        dv = float(np.linalg.norm(traj.lin_vel[t0, ib] - traj.lin_vel[t0 - 1, ib]))
        t1 = min(traj.num_frames - 1, t0 + 1)
        # The exchange is over in a frame or two, and so is the *evidence*: what
        # follows is two bodies travelling lawfully on wrong paths. The causal
        # mask still runs to the end of the clip, because it is gated on the
        # measured divergence between the twins rather than on this window --
        # which is what lets severity stay on the frames that show the breach.
        return InterventionPlan(
            family=self.family, kind="instant", t_event=t0, windows=[(t0, t1)],
            intervention_windows=[(t0, t1)],
            consequence_windows=[(t0, t1)],
            causal_body_ids=self._causal_order(actor_id, other_id),
            params=dict(self._params(ratio), collision_frame=int(t0),
                        actor=actor_id, other=other_id),
            # THE KNOB, not a derived effect. The magnitude used to be the
            # velocity change the pair failed to show, scaled by g*dt -- which
            # is measured *after* the fact, and is measured at the geometric
            # touch frame, where PyBullet has usually not resolved the collision
            # yet. On `collision` that read 7.8e-09 for a staged mass ratio of
            # 25: a maximal violation reporting as none at all. The unit has
            # always said `effective_mass_ratio`; now the number does too, and
            # it is exact and known before anything is simulated, which is what
            # CLAUDE.md asks `intervention.magnitude` to be. The measured effect
            # lives in the residual, where it belongs.
            magnitude=float(ratio),
            magnitude_unit=self.magnitude_unit, severity_bin=severity_bin,
            notes={"radius": float(traj.radius[ia]),
                   "surface_top": _geom.surface_top(spec, actor),
                   "ratio": ratio, "actor_id": actor_id, "other_id": other_id,
                   "normal": [float(x) for x in normal],
                   "r_strong": float(r_strong), "lawful_dv": dv})

    #: Both families are a mass statement, and PyBullet takes mass directly --
    #: so the collision itself can be left to the simulator instead of having
    #: its outcome written in. You reported the balls never touching; they do
    #: now, because the contact is real and only the masses are a lie.
    simulated = True

    def _mass_of(self, spec, seg_id: int) -> float:
        body = next(b for b in spec.bodies
                    if int(b.segmentation_id) == int(seg_id))
        return float(getattr(body, "mass", 1.0))

    def stage(self, spec, simulator, objs, plan):
        import pybullet as pb
        from ..render import stepper

        for seg_id, mass in self._staged_masses(spec, plan).items():
            idx = stepper.pybullet_index(simulator, objs, spec, seg_id)
            if idx is not None:
                pb.changeDynamics(idx, -1, mass=mass)
        return ()

    def unstage(self, spec, simulator, objs, plan) -> None:
        import pybullet as pb
        from ..render import stepper

        for seg_id in self._staged_masses(spec, plan):
            idx = stepper.pybullet_index(simulator, objs, spec, seg_id)
            if idx is not None:
                pb.changeDynamics(idx, -1, mass=self._mass_of(spec, seg_id))

    def _staged_masses(self, spec, plan) -> Dict[int, float]:
        raise NotImplementedError

    def _apply(self, spec, traj, plan) -> Trajectory:
        n = plan.notes
        out = self._edited(spec, traj, int(n["actor_id"]), int(n["other_id"]),
                           int(plan.params["collision_frame"]),
                           np.asarray(n["normal"], np.float64), float(n["ratio"]))
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out

    # ------------------------------------------------------------------ #
    FREEZE = "freeze"

    def _edited(self, spec, traj, actor_id, other_id, t0, normal, ratio):
        out = self._clone(traj)
        # The two are touching at `t0` and their outcome is prescribed, so they
        # must not see each other as obstacles afterwards -- the resolver would
        # dutifully re-solve the collision correctly and undo the intervention.
        # It did exactly that: a declared mass ratio of 25 came out rendering as
        # a ratio of 1. Everything else in the scene stays solid.
        pair = _geom.Obstacles(spec, traj,
                               exclude_ids=[int(actor_id), int(other_id)])
        for body_id, v_new in self._outcome(traj, actor_id, other_id, t0,
                                            normal, ratio).items():
            body = next(b for b in spec.bodies
                        if int(b.segmentation_id) == int(body_id))
            bi = traj.index_of(int(body_id))
            if isinstance(v_new, str) and v_new == self.FREEZE:
                # Held exactly where it was, not re-integrated. Integrating it
                # would hand it straight back to the collision resolver, which
                # would dutifully push it out of the striker's way -- restoring
                # the very reaction the family exists to remove.
                out.pos[t0:, bi, :] = traj.pos[t0 - 1, bi][None, :]
                out.quat[t0:, bi, :] = traj.quat[t0 - 1, bi][None, :]
                out.lin_vel[t0:, bi, :] = 0.0
                out.ang_vel[t0:, bi, :] = 0.0
                continue
            self._rewrite_from(spec, traj, out, body, t0, v0=v_new,
                               obstacles=pair)
        return out

    @staticmethod
    def _split(v, normal):
        """Velocity as (normal component, tangential vector)."""
        vn = float(np.dot(v, normal))
        return vn, v - vn * normal


class Newton2Mass(_CollisionEdit):
    """Two identical-looking bodies collide as though their masses differed.

    The distinction from `newton3_reaction` is the one that makes both families
    worth having. **Both bodies respond, and the exchange is exactly what a
    lawful collision between masses `k*m` and `m` would produce** -- the
    velocity changes come out in the ratio `1:k`, the restitution is the one
    the valid twin used, and the whole outcome is internally consistent. The
    only thing wrong is that the two bodies are visually identical, so nothing
    in the image justifies the mass ratio the collision implies.

    Worth being precise about what is *not* claimed here, because the obvious
    formulation is self-contradicting: this does not conserve momentum for the
    bodies' true masses. It cannot. An outcome inconsistent with equal masses
    and an outcome that conserves equal-mass momentum are the same constraint,
    so "wrong mass ratio but momentum still conserved" describes nothing. What
    separates the two families is how many bodies react: here both do, in a
    fixed and self-consistent ratio; under `newton3_reaction` one simply does
    not react at all.

    That makes it the family most dependent on its valid twin -- "that ball
    barely moved" is not evidence of anything on its own -- and it is why
    `collision` samples both balls with the same radius and the same
    colour. It is also why the family is not offered on `pyramid_impact`: a cube
    and a sphere are plainly different objects, so "the cube is heavy" is a
    lawful reading and there is no violation left to see.
    """

    family = "newton2_mass"
    magnitude_unit = "effective_mass_ratio"
    RATIO_BY_BIN = {"weak": 3.0, "medium": 8.0, "strong": 25.0}

    @staticmethod
    def _params(ratio):
        return {"type": "effective_mass_scale", "mass_ratio": ratio}

    @staticmethod
    def _causal_order(actor_id, other_id):
        """The culprit ALONE, and it is the body whose mass is the lie.

        `_staged_masses` scales `other`, so `other` is the thing that is wrong;
        the striker merely rebounds off it. Both used to be listed, which made
        both level 1 in `causal_mask` -- two red bodies and no way to tell the
        cause from the effect. Naming one leaves the other to be picked up as a
        measured consequence, which is level 2, blue: exactly the reading you
        asked for.
        """
        return [int(other_id)]

    def _outcome(self, traj, actor_id, other_id, t0, normal, ratio):
        """Re-solve the collision for effective masses (ratio*m, m).

        Standard one-dimensional restitution along the line of centres, with the
        tangential components untouched. The restitution is recovered from the
        lawful rollout so the invalid clip differs from its twin in the mass
        ratio and nothing else.
        """
        ia, ib = traj.index_of(actor_id), traj.index_of(other_id)
        ua, ta = self._split(traj.lin_vel[t0 - 1, ia].astype(np.float64), normal)
        ub, tb = self._split(traj.lin_vel[t0 - 1, ib].astype(np.float64), normal)

        # The restitution has to be read from the frame the collision actually
        # *resolves*, which is not always the frame the surfaces meet. `t0` is
        # now the geometric touch -- so that nothing reacts before it is
        # touched -- and PyBullet may only separate the pair a frame or two
        # later. Reading the outcome at `t0` then found the two still
        # approaching, computed a restitution of zero, and produced an outcome
        # where both bodies moved identically: a mass ratio of 25 that rendered
        # as a mass ratio of 1.
        approach = ua - ub
        va, vb = ua, ub
        for t in range(t0, min(t0 + 4, traj.num_frames)):
            ca, _ = self._split(traj.lin_vel[t, ia].astype(np.float64), normal)
            cb, _ = self._split(traj.lin_vel[t, ib].astype(np.float64), normal)
            if (cb - ca) > 1e-3:                  # separating: it has resolved
                va, vb = ca, cb
                break
        e = 0.5 if abs(approach) < 1e-6 else float(np.clip((vb - va) / approach,
                                                           0.05, 1.0))
        ma, mb = ratio * float(traj.mass[ia]), float(traj.mass[ib])
        total = ma + mb
        p = ma * ua + mb * ub
        va_new = (p + mb * e * (ub - ua)) / total
        vb_new = (p + ma * e * (ua - ub)) / total
        return {actor_id: ta + va_new * normal,
                other_id: tb + vb_new * normal}


    def _staged_masses(self, spec, plan) -> Dict[int, float]:
        """One body weighs `ratio` times what it looks like it weighs.

        Momentum is still conserved -- PyBullet conserves it -- so the violation
        is exactly what the family claims: two bodies that look identical
        respond as though they are not.
        """
        other = int(plan.notes["other_id"])
        ratio = float(plan.notes["ratio"])
        return {other: self._mass_of(spec, other) * max(ratio, 1e-6)}


register(Solidity())
register(SuperElastic())
register(Newton2Mass())


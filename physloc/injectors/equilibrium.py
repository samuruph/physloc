"""Equilibrium-domain injectors: support, friction.

Both act on bodies that are *in contact with something* rather than in flight,
which makes them the two families most at risk of accidentally producing a
second violation. A hovering body must not also pass through the table it left;
a block that stops on a slope must stop *on the slope* and not beside it. The
implementations below are shaped almost entirely by those two constraints.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..sim.trajectory import Trajectory
from . import _geom
from .base import Injector, InterventionPlan, register


class Support(Injector):
    """A body rises off its support and hangs there.

    Two shapes, chosen by what the body was doing. One that was at rest hovers
    where it stood. One that was sliding keeps sliding, along exactly the path
    it would have taken, lifted clear of the surface -- freezing that one would
    add a Newton-1 violation on top, and "the box stopped" is a different claim
    from "the box is not touching the ramp".

    The clearance is measured against whatever is directly beneath the body --
    including another moving body. Measuring against the floor instead would
    report the top block of a stack as hovering two blocks' worth of height on a
    perfectly lawful clip, and the noise floor calibrated on that valid arm
    would swallow the real violation whole.
    """

    family = "support"
    persistent = True
    RISE_FRAMES = 3
    CLEARANCE_RADII = {"weak": 0.8, "medium": 2.0, "strong": 3.6}
    #: Natural frequency, rad/s, of the pull holding a hovering body at its
    #: height -- see `stage`.
    HOLD_OMEGA = 8.0

    def strong_residual_reference(self, spec) -> float:
        return float(self.CLEARANCE_RADII["strong"])

    def refine_windows(self, spec, traj_valid, traj_invalid, plan) -> None:
        """The strong bin's residual, from this clip's own.

        The reference was the strong CLEARANCE, 3.6 radii, and the law reads
        clearance above whatever is beneath the body -- which, for a body
        gliding off the end of a ramp, is the floor metres below. Measured on
        `rolling_ramp` 777: 8.37 / 9.57 / 11.2 radii for weak / medium /
        strong, all over 3.6, so all three saturated at 1.000.

        Those numbers are an offset -- where the body is gliding -- plus the
        declared clearance, and the bins share the offset: 8.37 - 0.8 + 3.6 =
        11.17 against 11.2 measured for strong. So this clip's departure plus
        the clearance it is short of strong IS strong's, and each bin is scored
        against it. Measured the way the pipeline scores: the mass-weighted
        departure from the twin over every body named.
        """
        from ..residuals import laws as _laws

        law = _laws.get("support")
        ctx = dict(plan.notes, spec=spec)
        rows, weights = [], []
        for bid in plan.causal_body_ids:
            try:
                bi_v = traj_valid.index_of(int(bid))
                bi_i = traj_invalid.index_of(int(bid))
            except KeyError:
                continue
            rows.append(np.abs(law(traj_invalid, bi_i, ctx)
                               - law(traj_valid, bi_v, ctx)))
            weights.append(float(traj_invalid.mass[bi_i]))
        if not rows:
            return
        w = np.asarray(weights, np.float64)
        w = w / w.sum() if float(w.sum()) > 0.0 else np.full_like(w, 1.0 / len(w))
        measured = float(np.tensordot(w, np.stack(rows), axes=(0, 0)).max())
        short = (float(self.CLEARANCE_RADII["strong"])
                 - float(plan.params.get("clearance_radii", 0.0)))
        if measured > 1e-9:
            plan.notes["r_strong"] = float(measured + max(0.0, short))

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        # WHOLE MEDIUM where the scene is made of interchangeable bodies. One
        # grain of forty hovering is perfectly annotated and impossible to see,
        # and it also breaks the residual: a lifted grain still has grains
        # beneath it, so `support`'s clearance is taken against them and the law
        # reads it as supported. Lifting the whole pour leaves nothing under any
        # of them, which is both visible and measurable.
        targets = self._group(spec)
        actor = targets[0] if targets else self._primary(spec)
        if actor is None:
            return None
        T = traj.num_frames
        # CATCH THE MEDIUM IN THE AIR. A third of the way into `pour` is after
        # the grains have landed, so the violation lifted a settled pile off the
        # floor rather than stopping it mid-fall -- which is a different and far
        # less legible claim. Where the scene is one falling medium, support
        # fails while it is still falling.
        t0 = None
        if len(targets) > 1:
            t0 = _geom.before_medium_lands(spec, traj, targets)
        if t0 is None:
            t0 = max(1, T // 3)
        if t0 >= T - 1:
            return None
        clearance_r = self.CLEARANCE_RADII[severity_bin]
        radius = float(actor.bounding_radius)
        _, top = _geom.support_under_any(spec, actor, traj, 0)
        bi = traj.index_of(int(actor.segmentation_id))
        # HORIZONTAL speed, not total. A body in free fall has a large speed and
        # no horizontal motion at all, and calling that "moving" is what put
        # `drop` into the wrong branch: it kept the downward velocity it had,
        # drifted into the floor at a constant rate and bounced -- the up and
        # down you reported. A body nothing is holding up does not carry on
        # falling; that is `antigravity`. It hangs.
        speed = float(np.linalg.norm(traj.lin_vel[t0, bi][:2]))
        mode = "hover_still" if speed < 0.3 else "hover_moving"
        if len(targets) > 1:
            # A medium hangs where support failed. Its sideways speed is the
            # spout's spread, not a slide along a surface anyone could see it
            # keep -- see `stage`.
            mode = "hover_still"

        def hover(clearance: float) -> InterventionPlan:
            return InterventionPlan(
                family=self.family, kind="sustained", t_event=t0,
                windows=[(t0, T - 1)],
                causal_body_ids=[int(b.segmentation_id) for b in targets] or
                                [int(actor.segmentation_id)],
                params={"type": "hover", "clearance_radii": clearance,
                        "mode": mode},
                magnitude=float(clearance * radius),
                magnitude_unit="m_support_clearance", severity_bin=severity_bin,
                notes={"radius": radius,
                       # With the WHOLE medium lifted there is nothing left
                       # beneath any of it, so the clearance datum is the floor.
                       # Measured against the pile -- which is what
                       # `support_under_any` returns for a grain in a column --
                       # a grain hovering at z = 0.33 sat *below* its recorded
                       # support and the law read clearance 0.000 on an
                       # obviously airborne pour.
                       "surface_top": (float(_geom.surface_top(spec, actor))
                                       if len(targets) > 1 else float(top)),
                       "clearance_radii": clearance, "mode": mode,
                       # The path a SLIDING body lawfully takes from here, as
                       # its horizontal velocity per frame -- see `stage`.
                       "lawful_xy_velocity": (
                           traj.lin_vel[t0:, bi, :2].astype(float).tolist()
                           if mode == "hover_moving" and len(targets) <= 1
                           else None)})

        # KEEP THE HOVER IN SHOT. A body caught a third of the way through a
        # `drop` is still high, and lifting it the strong bin's 3.6 radii from
        # there hung it above the frame for the rest of the clip: on seed 778
        # at z = 3.7, declined on every attempt because the moment never moves.
        # Fitted on the strongest bin so the three stay ordered. A medium is
        # not fitted here -- `_apply` lifts only its first grain, which is not
        # a preview of the whole pour rising.
        if len(targets) <= 1:
            strongest = float(self.CLEARANCE_RADII["strong"])
            scale, _ = self._fit_to_frame(
                spec, traj, [actor], t0, strongest,
                lambda k: self._apply(spec, traj, hover(strongest * k)),
                memo=("hover", strongest))
            clearance_r *= float(scale)
        return hover(clearance_r)

    #: Weightlessness is a force statement, so PyBullet can say it: cancel
    #: gravity on the body and let it keep whatever motion it had.
    simulated = True

    def stage(self, spec, simulator, objs, plan):
        """Lift the body clear and cancel gravity on it.

        The two modes fall out of the physics rather than needing separate code.
        A body at rest hovers where it stood, because nothing is pushing it. A
        body that was sliding keeps sliding in a straight line, because nothing
        is pulling it down -- which is exactly the "keeps moving, lifted clear"
        shape the family wants, and it now happens because it must rather than
        because it was written in.

        It also keeps colliding with everything else while it hovers, which the
        prescribed version could not promise.
        """
        import pybullet as pb
        from ..render import stepper

        want = float(plan.notes["clearance_radii"]) * float(plan.notes["radius"])
        top = float(plan.notes["surface_top"])
        g = np.asarray(spec.gravity, np.float64)
        targets = []
        for bid in plan.causal_body_ids:
            body = next((b for b in spec.bodies
                         if int(b.segmentation_id) == int(bid)), None)
            if body is None or body.static:
                continue
            idx = stepper.pybullet_index(simulator, objs, spec, int(bid))
            if idx is None:
                continue
            pos, quat = pb.getBasePositionAndOrientation(idx)
            radius = float(body.bounding_radius)
            # Only as much lift as the clearance still needs. Adding the full
            # clearance unconditionally teleported a body that was ALREADY
            # airborne three and a half radii further up, which is a continuity
            # violation smuggled into a support clip.
            have = max(0.0, float(pos[2]) - radius - top)
            lifted = (pos[0], pos[1], pos[2] + max(0.0, want - have))
            v, w = pb.getBaseVelocity(idx)
            pb.resetBasePositionAndOrientation(idx, lifted, quat)
            # The vertical component goes. Whatever the body was doing sideways
            # it carries on doing -- a sliding box keeps sliding, and freezing
            # it would add a Newton-1 violation on top of this one -- but a
            # falling body stops falling, because that is what "nothing is
            # holding it up, and yet" looks like.
            #
            # A body the plan calls STILL keeps no sideways motion either,
            # which is what `_apply` has always drawn. It matters for a medium:
            # a pour caught mid-stream carries the spout's spread, and with
            # gravity gone every grain flew on in a straight line -- measured
            # at release on `pour`, out to 4 m and under 60% of the pile in
            # shot within ten frames, on every attempt.
            keep = 0.0 if plan.notes.get("mode") == "hover_still" else 1.0
            pb.resetBaseVelocity(idx, [keep * float(v[0]), keep * float(v[1]),
                                       0.0], list(w))
            targets.append((idx, float(getattr(body, "mass", 1.0)),
                            float(lifted[2])))
        if not targets:
            return ()

        # A SLIDING BODY FOLLOWS ITS LAWFUL PATH, lifted clear -- which is what
        # `_apply` draws and what the family says. Cancelling gravity alone
        # also cancelled the floor friction that was slowing it, so it glided
        # on at its launch speed: on `stack_topple` at 37 frames the top block
        # left the shot 3 m out where the lawful one stopped at 1.94 m, on
        # every attempt, since a lower hover does nothing for a drift.
        lawful = plan.notes.get("lawful_xy_velocity")
        t0 = int(plan.t_event)

        # HELD AT ITS HEIGHT, not merely weightless. With gravity cancelled
        # and nothing else, a knock that gave the body any vertical speed kept
        # it forever: on `collision` 778 the weak bin's striker, hovering low
        # enough to clip its target, was bumped and climbed 6 cm a frame for
        # the rest of the clip -- past the medium bin's hover, so weak scored
        # above medium. A critically damped pull back to the hover height is
        # what "held up by nothing" means; it still collides with whatever it
        # meets.
        omega = float(self.HOLD_OMEGA)

        def weightless(_client, _step, frame):
            for idx, mass, z_hold in targets:
                pos, _ = pb.getBasePositionAndOrientation(idx)
                vz = float(pb.getBaseVelocity(idx)[0][2])
                hold = mass * (-omega * omega * (float(pos[2]) - z_hold)
                               - 2.0 * omega * vz)
                force = (-g * mass) + np.array([0.0, 0.0, hold])
                pb.applyExternalForce(idx, -1, force.tolist(), list(pos),
                                      pb.WORLD_FRAME)
                if lawful:
                    k = min(max(int(frame) - t0, 0), len(lawful) - 1)
                    v, w = pb.getBaseVelocity(idx)
                    pb.resetBaseVelocity(
                        idx, [float(lawful[k][0]), float(lawful[k][1]),
                              float(v[2])], list(w))

        return (weightless,)

    def _apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        actor = self._primary(spec)
        bi = traj.index_of(int(actor.segmentation_id))
        t0 = plan.t_event
        top = float(plan.notes["surface_top"])
        radius = float(plan.notes["radius"])
        lift = float(plan.notes["clearance_radii"]) * radius
        n = traj.num_frames - t0

        # Eased up over a few frames, not snapped. A body that jumps three radii
        # between two frames is a *teleport*, so snapping it made every
        # `support` clip trip the continuity detector as well.
        ramp = max(2, min(self.RISE_FRAMES, n))
        u = np.clip((np.arange(n, dtype=np.float64) + 1.0) / ramp, 0.0, 1.0)
        ease = u * u * (3.0 - 2.0 * u)
        start = traj.pos[t0 - 1, bi].astype(np.float64)

        # ONE rule, three pictures. Horizontal motion carries on exactly as it
        # lawfully would -- freezing a sliding body would add a Newton-1
        # violation on top of this one -- and the height is held at whatever it
        # was when support failed, raised to the clearance the bin asks for.
        #
        # A resting body therefore hovers where it stood, a sliding one keeps
        # sliding with nothing under it, and a falling one stops falling and
        # hangs. That last case is the one you reported: it used to keep its
        # downward velocity, so it sank to the floor and bounced.
        held = traj.pos[t0:, bi, :].astype(np.float64).copy()
        if plan.notes["mode"] == "hover_still":
            held[:, 0:2] = start[None, 0:2]
        have = max(0.0, float(start[2]) - radius - top)
        hold_z = float(start[2]) + max(0.0, lift - have)
        held[:, 2] = start[2] + (hold_z - start[2]) * ease

        out.pos[t0:, bi, :] = held.astype(np.float32)
        out.quat[t0:, bi, :] = traj.quat[t0:, bi, :]
        out.ang_vel[t0:, bi, :] = 0.0
        self._sync_velocity(traj, out, bi, t0)

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class Friction(Injector):
    """A moving body is dragged to a halt by a surface that should barely grip.

    **The direction of this family changed, and it is worth saying why.** It
    used to claim the other half of the axis -- *less* grip than declared, so a
    body fails to slow as it should -- on the reasoning that "more grip" at its
    limit is a body stopping dead, which is `newton1_inertia`. Measured in the
    pinned image, the half it kept is not available: `barrier_pass`,
    `collision` and `occluder_pass` all give their actor a friction of 0.02 to
    0.05 so that it rolls freely, and taking that to 0.001 moves the ball by
    0.11 m over three seconds. You reported all three as barely visible, and
    that is the number behind it -- there is no headroom below a coefficient
    that is already almost zero.

    The other half has plenty, and it is not `newton1_inertia`. That family
    removes a body's velocity *between two frames* with nothing touching it;
    this one decelerates it over half a second while it is in continuous contact
    with a surface, which is exactly the signature of friction and exactly what
    the `friction` law measures. What is wrong is that nothing in the image
    justifies the grip: a ball rolls onto ordinary floor and stops as though it
    had rolled onto carpet.

    Staged as coefficients, and **rolling friction as well as lateral**. That is
    the measurement that decides it: lateral friction alone hardly touches a
    rolling sphere, because a rolling contact is not sliding. The two together
    take the same ball from 2.51 m of travel to 0.80 m and leave it at rest.

    The host-side approximation replays the body's own lawful path at a reduced
    rate rather than re-integrating it, which guarantees it stays on the surface
    it was travelling along: a ramp is not a horizontal plane, so re-integrating
    under gravity puts the block in mid-air beside its ramp -- a support
    violation smuggled into a friction clip and annotated as neither.
    """

    family = "friction"
    persistent = True
    #: Fraction of its lawful speed the body keeps once the surface has gripped
    #: it. Used by the host-side `_retimed` approximation and as the residual
    #: reference; the staged coefficient is solved for separately, below.
    RATE_BY_BIN = {"weak": 0.55, "medium": 0.28, "strong": 0.08}

    #: **How far the body still travels, as a fraction of the distance it
    #: lawfully would.** The knob is a stopping DISTANCE rather than a
    #: coefficient, and that is what keeps this family distinguishable from
    #: `newton1_inertia`.
    #:
    #: You reported the two as near-identical on `barrier_pass`, and they were:
    #: a fixed coefficient stopped the ball inside two frames, which is a step
    #: to zero -- exactly newton-1's picture. Friction is not a step, it is a
    #: curve, and a curve needs room. Solving for the distance gives the body
    #: that room and adapts to whatever speed the scenario happens to give it,
    #: which also answers the other half of your report: `collision`'s friction
    #: clip looked like its valid twin because the same coefficient that halts a
    #: fast ball barely touches a slow one.
    #:
    #: Even `strong` leaves a third of the lawful travel, so the body is still
    #: sliding when the deceleration becomes obvious. A body that stops dead in
    #: one frame is a different family and stays one.
    TRAVEL_BY_BIN = {"weak": 0.70, "medium": 0.50, "strong": 0.32}
    #: Scored on the deceleration the clip ADDS to the lawful one, not on any
    #: difference: see `severity.bounded_score`.
    SCORE_EXCESS_ONLY = True
    #: Ceilings, so a solved coefficient stays inside what Bullet handles well.
    #: Raised from 1.2 / 0.40, which was BINDING and collapsing the ladder: the
    #: per-bin floor below is `MIN_RATIO_BY_BIN * mu`, so on a surface declared
    #: at mu = 0.5 medium wanted 1.3 and strong 2.0 and the old ceiling handed
    #: both of them 1.2 -- two bins, one coefficient, one clip. That is the
    #: "strengths appear all the same" you saw on `pour`. A coefficient above 1
    #: is unusual but perfectly ordinary for Bullet; it means the contact grips
    #: harder than the normal force, which is exactly the claim the family
    #: makes.
    MAX_LATERAL = 2.50
    MAX_ROLLING = 0.60
    #: Floor, as a multiple of the DECLARED coefficient. Whatever the solve
    #: returns, the surface must grip harder than it says it does -- otherwise
    #: the clip depicts the opposite violation from the one it is labelled
    #: with. `ramp_slide` is where that bit: its block is already slowing on a
    #: slope, so the arithmetic asked for *less* grip than declared.
    #:
    #: Per bin, not one number, because on a slope the floor IS the knob: the
    #: stopping-distance solve is below it for every bin there, so a single
    #: floor made weak, medium and strong the same clip three times.
    #: `weak` lowered from 1.6, because on a MEDIUM the floor is what binds and
    #: 1.6x of a declared 0.5 is already enough grip to lock a pile where it
    #: lands. Measured on `pour`: the lawful pile spreads to 0.51 m and all
    #: three bins came out between 0.216 and 0.232 -- the change against the
    #: valid twin was large and the change *between bins* was not, which is the
    #: "strengths appear all the same" you saw. At 1.15 the weak bin still
    #: grips harder than the surface declares, and it still spreads.
    MIN_RATIO_BY_BIN = {"weak": 1.15, "medium": 2.6, "strong": 4.0}

    def strong_residual_reference(self, spec) -> float:
        return 2.0

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        # WHOLE MEDIUM where the scene is made of interchangeable bodies. One
        # grain of forty hovering is perfectly annotated and impossible to see,
        # and it also breaks the residual: a lifted grain still has grains
        # beneath it, so `support`'s clearance is taken against them and the law
        # reads it as supported. Lifting the whole pour leaves nothing under any
        # of them, which is both visible and measurable.
        targets = self._group(spec)
        actor = targets[0] if targets else self._primary(spec)
        if actor is None:
            return None
        T = traj.num_frames
        bi = traj.index_of(int(actor.segmentation_id))
        speed = np.linalg.norm(traj.lin_vel[:, bi, :].astype(np.float64), axis=1)
        moving = np.flatnonzero(speed > 0.35)
        if moving.size < 3:
            return None
        # BRAKE IN THE OPEN. The deceleration takes the better part of a
        # second, so firing a third of the way into `occluder_pass` brings the
        # body to rest behind the screen -- and a body at rest behind a screen
        # has no pixels in the invalid render, so the severity field has nowhere
        # to land and the clip ships a picture scoring zero.
        #
        # Asked of the WHERE, not of the scenario's `occluded_frames`. That list
        # is computed from the lawful rollout, so it describes the path the body
        # does not take: on the review sweep's `occluder_pass` seed it was empty
        # -- a ball that keeps rolling never dwells behind the screen -- while
        # the friction clip parked it squarely behind. So the stopping point is
        # predicted and tested against the actual geometry, and if it is hidden
        # the violation fires earlier, until the body comes to rest in view.
        t0 = int(max(moving[0] + 1, min(moving[-1] - 1, T // 3)))
        # A MEDIUM GRIPS AS IT LANDS. On `pour` the grains are at rest from
        # frame 7, so a brake applied at frame 7 has nothing left to slow -- and
        # the three bins picked frames 4, 6 and 6, which is three different
        # violations wearing one ladder. Catching the pour on the way down means
        # the grip decides how far the pile spreads when it hits, which is what
        # friction looks like on a granular medium, and all three bins share the
        # frame so their magnitudes stay comparable.
        medium = len(targets) > 2
        t0 = self._medium_event_frame(spec, traj, targets, t0)
        ramp_end = None
        ramp_id = spec.notes.get("ramp_id")
        if ramp_id is not None:
            run = _geom.contact_run(traj, int(actor.segmentation_id), int(ramp_id))
            if run is not None:
                lo, hi = max(1, run[0] + 1), min(T - 2, run[1] - 1)
                if lo <= hi:
                    # EARLY IN THE SLIDE, while the block is still slow. Braked
                    # late, near the lip, stopping on what is left of the ramp
                    # took several g, and a cube braked that hard does not
                    # slide to a halt -- it pitches over its leading edge and
                    # tumbles on down, which is what every ramp clip showed.
                    # The first quarter of the run leaves stopping to a
                    # fraction of g; the moment still varies within it.
                    t0 = _geom.frame_in_band(spec, lo,
                                             lo + max(1, (hi - lo) // 4))
                    ramp_end = hi
        if not (1 <= t0 < T - 1):
            return None
        # Outward from the preferred moment in BOTH directions. Searching only
        # earlier does not work on `occluder_pass`: the stopping distance is a
        # share of the path that REMAINS, so starting sooner lengthens the path
        # and leaves the body in much the same place. What works there is
        # braking later, once the ball is past the screen -- and which of the
        # two a scene needs is not something to decide in advance.
        lo, hi = int(max(1, moving[0] + 1)), int(min(T - 2, moving[-1] - 1))
        # A medium has already chosen its moment, and it is not a body that can
        # be parked behind a screen: `pour` has no occluder and the search would
        # only walk the frame back off the arrival it was picked for.
        for delta in ([] if medium
                      else range(0, max(t0 - lo, hi - t0) + 1)):
            for cand in (t0 - delta, t0 + delta):
                if not (lo <= cand <= hi):
                    continue
                stop = self._stopping_point(spec, traj, bi, cand, severity_bin)
                if stop is None or not _geom.hidden_behind_static(spec, stop):
                    t0 = cand
                    break
            else:
                continue
            break

        rate = self.RATE_BY_BIN[severity_bin]
        # THE WHOLE MEDIUM, not just the grain that stands for it. `_retimed`
        # slows one body, which is the entire intervention on a single-actor
        # scene and a fortieth of it on a pour -- so the reference came out at
        # 1.085 against residuals of 2.50 to 3.08 and all three bins clipped to
        # 1.000. Retiming every target and taking the largest is the same
        # measurement the staged clip will be scored by.
        strong = traj
        for body in targets:
            strong = self._retimed(strong, traj.index_of(
                int(body.segmentation_id)), t0, self.RATE_BY_BIN["strong"])
        r_strong = max(
            (self._measure(strong, int(b.segmentation_id), "friction", {})
             for b in targets), default=0.0)
        mu = float(getattr(actor, "friction", 0.5))
        grip, roll, target = self._solve_grip(spec, traj, actor, bi, t0,
                                              severity_bin)
        # ON A RAMP, THE SURFACE IS THE KNOB. PyBullet takes friction as the
        # PRODUCT of the pair, and `ramp_slide` gives its ramp a coefficient of
        # a few hundredths so the block slides at all -- so raising the block's
        # own coefficient by 15% raised the contact's by 15% of almost nothing,
        # and every weak clip there was indistinguishable from its twin. The
        # solve also ignored the slope: stopping on an incline needs the grip to
        # beat the downhill pull too, `(a + g sin t) / (g cos t)` rather than
        # `a / g`. Solved for the contact, the ramp takes the difference -- it
        # is the surface that grips harder than it should, which is the claim.
        #
        # AND IT STOPS ON THE RAMP. The distance was a share of the body's whole
        # remaining path -- ramp, flight and floor -- while the extra grip acts
        # only on the ramp, so the grip asked for was a fraction of what showed:
        # measured, `ramp_slide` travelled 83-110% of its lawful path at bins
        # asking 70 / 50 / 32%. A block stopping partway down a slope it
        # lawfully slides off is the unmistakable form of the claim, so on a
        # ramp the share is of what is left of the RAMP, and the grip stays on
        # -- a block stopped on a slope must not be let go again.
        g = float(np.linalg.norm(traj.gravity)) or 9.81
        v = float(np.linalg.norm(traj.lin_vel[t0, bi]))
        tilt = 0.0
        surface = None
        on_ramp = ramp_end is not None and spec.notes.get("tilt_rad") is not None
        if on_ramp:
            tilt = float(spec.notes["tilt_rad"])
            ramp_path = np.asarray(traj.pos[t0:ramp_end + 1, bi, :], np.float64)
            d_ramp = float(np.linalg.norm(np.diff(ramp_path, axis=0), axis=1).sum())
            target = max(1e-3, float(self.TRAVEL_BY_BIN[severity_bin]) * d_ramp)
            need = ((v * v / (2.0 * target) + g * np.sin(tilt))
                    / (g * np.cos(tilt)))
            ramp_body = next((b for b in spec.bodies
                              if int(b.segmentation_id) == int(ramp_id)), None)
            declared = float(getattr(ramp_body, "friction", 0.0) or 0.0)
            surface = float(min(self.MAX_LATERAL,
                                max(declared, need / max(grip, 1e-6))))
            ramp_end = None                  # hold the grip: see above
        # THE REFERENCE, FROM THE PHYSICS OF THE STRONG BIN. The friction law
        # reads unexplained along-track deceleration over g; a body brought to
        # rest over `d` on a slope `t` reads (v^2/2d + g sin t) / g, and the
        # lawful clip's own reading is measured on the twin. Strong's excess is
        # the difference -- independent of this clip, so the three bins are
        # scored against one yardstick rather than each against itself.
        # A medium keeps the retimed measurement: its grains are not one body
        # stopping over one distance.
        if len(targets) <= 2:
            from ..residuals import laws as _laws
            if on_ramp:
                d_strong = max(1e-3, float(self.TRAVEL_BY_BIN["strong"]) * d_ramp)
                end = int(np.clip(t0 + max(3, int(0.5 * (T - t0))), t0 + 1, T - 1))
            else:
                d_strong = max(1e-3, float(target) * float(self.TRAVEL_BY_BIN["strong"])
                               / float(self.TRAVEL_BY_BIN[severity_bin]))
                end = T - 1
            lawful = float(np.median(_laws.get("friction")(traj, bi, {})[t0:end + 1]))
            r_strong = max(1e-3, (v * v / (2.0 * d_strong) + g * np.sin(tilt)) / g
                           - lawful)
            d_bin = (max(1e-3, float(self.TRAVEL_BY_BIN[severity_bin]) * d_ramp)
                     if on_ramp else max(1e-3, float(target)))
            brake = {"v": v, "tilt": tilt, "lawful": lawful,
                     "d_bin": d_bin, "d_strong": d_strong}
        else:
            brake = None
        intervention = [(t0, T - 1 if on_ramp else
                         (ramp_end if ramp_end is not None else T - 1))]
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=[(t0, T - 1)],
            intervention_windows=intervention,
            consequence_windows=[(t0, T - 1)],
            # THE WHOLE MEDIUM. `_group` was already being consulted and then
            # thrown away -- only `targets[0]` was named -- so a pour had one
            # grain of forty gripping while the rest slid past it, which is
            # neither visible nor what a surface losing its slipperiness looks
            # like.
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "friction_scale", "end_rate": rate,
                    "lateral_friction": grip, "rolling_friction": roll,
                    "declared_friction": mu,
                    "surface_id": (int(ramp_id) if surface is not None
                                   else None),
                    "surface_friction": surface,
                    "travel_fraction": float(self.TRAVEL_BY_BIN[severity_bin]),
                    "target_distance_m": float(target)},
            # THE KNOB: how much of its lawful journey the body never makes.
            # Not the coefficient ratio, which is a *derived* quantity here --
            # the solve clamps it against floors and ceilings, so on `collision`
            # all three bins came out at the same ratio while the clips plainly
            # differed. The travel fraction is exact, ordered by construction
            # and the thing a viewer actually sees.
            magnitude=float(1.0 - self.TRAVEL_BY_BIN[severity_bin]),
            magnitude_unit="path_length_deficit",
            severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "surface_top": _geom.surface_top(spec, actor),
                   "end_rate": rate, "lateral_friction": grip,
                   "rolling_friction": roll,
                   "target_distance_m": float(target),
                   "ramp_window_end": ramp_end,
                   "brake": brake,
                   "r_strong": float(r_strong)})

    def refine_windows(self, spec, traj_valid, traj_invalid, plan) -> None:
        """Anchor the strong reference on this clip's measured braking.

        The plan's reference is the physics of the strong bin -- speed,
        stopping distance, slope -- and it is right to within about 15%. That
        is not close enough on a slope, where most of every bin's reading is
        the same thing: a body NOT sliding downhill, g sin t, about 0.33 g on
        `rolling_ramp`. The bins differ by the braking on top of it, 0.06 g
        against 0.11, so a 15% error put weak over the reference and all three
        read 1.000. The model is kept for what it gets right, the ratio of
        strong's braking to this bin's, and this clip's measured excess sets
        the level: strong's is this one's times that ratio.
        """
        brake = plan.notes.get("brake")
        if not brake:
            return
        from ..residuals import laws as _laws

        g = 9.81

        def model(d):
            return max(1e-6, (brake["v"] ** 2 / (2.0 * max(d, 1e-6))
                              + g * np.sin(brake["tilt"])) / g - brake["lawful"])

        law = _laws.get("friction")
        measured = 0.0
        for bid in plan.causal_body_ids:
            try:
                bi_v = traj_valid.index_of(int(bid))
                bi_i = traj_invalid.index_of(int(bid))
            except KeyError:
                continue
            measured = max(measured, float(np.maximum(
                0.0, law(traj_invalid, bi_i, {}) - law(traj_valid, bi_v, {})).max()))
        if measured > 1e-9:
            plan.notes["r_strong"] = float(
                measured * model(brake["d_strong"]) / model(brake["d_bin"]))

    def _stopping_point(self, spec, traj, bi: int, t0: int, severity_bin: str):
        """Where along its lawful path the body will have come to rest.

        The distance is the same share of the remaining path `_solve_grip`
        solves against, so the two agree by construction; the position is read
        off the lawful path rather than integrated, which is exact for the
        question being asked -- the body is being slowed along that path, not
        sent somewhere else.
        """
        pos = np.asarray(traj.pos[t0:, bi, :], np.float64)
        if pos.shape[0] < 2:
            return None
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(step)])
        target = float(self.TRAVEL_BY_BIN[severity_bin]) * float(arc[-1])
        return pos[int(np.searchsorted(arc, target, side="left"))
                   if target < arc[-1] else -1]

    def _solve_grip(self, spec, traj, actor, bi: int, t0: int,
                    severity_bin: str):
        """(lateral, rolling, target_distance) that stop the body where we want.

        Uniform deceleration: a body at `v` covering `d` before stopping needs
        `a = v^2 / 2d`. For a SLIDING body that is `mu*g`, so `mu = v^2/(2 g d)`;
        for a ROLLING one the retarding torque gives `a = mu_r*g/r`, so the
        coefficient carries an extra factor of the radius. Both fall straight
        out of the target distance, which is why the knob is expressed there.

        `d` is a share of the distance the body lawfully still travels --
        measured along its own path, so a scenario where it rebounds off a wall
        and comes back counts the whole journey rather than the net
        displacement.
        """
        pos = np.asarray(traj.pos[t0:, bi, :], np.float64)
        lawful = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
        v = float(np.linalg.norm(traj.lin_vel[t0, bi]))
        target = max(1e-3, float(self.TRAVEL_BY_BIN[severity_bin]) * lawful)
        accel = v * v / (2.0 * target)
        g = float(np.linalg.norm(traj.gravity)) or 9.81
        radius = max(float(traj.radius[bi]), 1e-6)
        mu = float(getattr(actor, "friction", 0.5))
        ratio = self.MIN_RATIO_BY_BIN[severity_bin]
        floor = ratio * mu
        if actor.kind == "sphere":
            # Rolling resistance does the work; lateral friction only has to be
            # enough to keep it rolling rather than sliding.
            #
            # **And it must beat what the SCENE declares.** `pour`'s grains
            # carry rolling friction of their own now, because a medium of
            # frictionless rollers cannot hold a pile -- and the solved values
            # here (0.058 to 0.24) straddle it. The weak bin was therefore
            # setting a rolling friction BELOW the declared one and making the
            # grains slipperier: the opposite violation, scored as this one.
            # The floor is the same multiple of the declared coefficient the
            # lateral floor uses, so "grips harder than it says it does" holds
            # on both axes.
            declared_roll = float(getattr(actor, "rolling_friction", 0.0))
            roll = min(self.MAX_ROLLING,
                       max(accel * radius / g, floor * 0.1,
                           ratio * declared_roll))
            return (min(self.MAX_LATERAL, max(0.4, floor)), roll, target)
        return (min(self.MAX_LATERAL, max(accel / g, floor)), 0.0, target)

    # ------------------------------------------------------------------ #
    def _retimed(self, traj, bi: int, t0: int, end_rate: float) -> Trajectory:
        out = self._clone(traj)
        T = traj.num_frames
        n = T - t0
        # Rate eases from 1 (lawful) to `end_rate` over three frames and then
        # holds, so the body slows visibly instead of stopping between frames.
        ramp = np.linspace(1.0, end_rate, min(3, n) + 1)[1:]
        rate = np.concatenate([ramp, np.full((n - ramp.size,), end_rate)])
        u = float(t0 - 1) + np.cumsum(rate)
        pos = _geom.path_sample(traj.pos[:, bi, :], u)
        out.pos[t0:, bi, :] = pos.astype(np.float32)
        vel = np.zeros_like(pos)
        vel[1:] = (pos[1:] - pos[:-1]) / traj.dt
        vel[0] = (pos[0] - traj.pos[t0 - 1, bi]) / traj.dt
        out.lin_vel[t0:, bi, :] = vel.astype(np.float32)
        out.ang_vel[t0:, bi, :] = (traj.ang_vel[t0:, bi, :]
                                   * rate[:, None].astype(np.float32))
        return out

    #: STAGED. Coefficients are what friction *is*, so this is the family with
    #: the least excuse for prescribing an outcome: whether the body slides,
    #: rolls, stops or keeps creeping is the solver's answer to a surface
    #: property, not ours.
    simulated = True

    def stage(self, spec, simulator, objs, plan):
        """Give the actor's contact more grip than the scene declares.

        Both coefficients, and the rolling one is the load-bearing half: a
        rolling contact is not a sliding one, so `lateralFriction` alone leaves
        a ball rolling almost exactly as far as it lawfully would.
        """
        import pybullet as pb
        from ..render import stepper

        grip = float(plan.params["lateral_friction"])
        roll = float(plan.params["rolling_friction"])
        changed = []
        for bid in plan.causal_body_ids:
            body = next((b for b in spec.bodies
                         if int(b.segmentation_id) == int(bid)), None)
            if body is None or body.static:
                continue
            idx = stepper.pybullet_index(simulator, objs, spec, int(bid))
            if idx is not None:
                pb.changeDynamics(idx, -1, lateralFriction=grip,
                                  rollingFriction=roll, spinningFriction=roll)
                changed.append((idx, body))
        surface = self._surface(spec, simulator, objs, plan)
        if surface is not None:
            pb.changeDynamics(surface[0], -1,
                              lateralFriction=float(plan.params["surface_friction"]))
        end = plan.notes.get("ramp_window_end")
        if end is None or not changed:
            return ()

        state = {"restored": False}

        def restore(_client, _step, frame):
            if state["restored"] or frame <= int(end):
                return
            for idx, body in changed:
                declared_roll = float(getattr(body, "rolling_friction", 0.0))
                pb.changeDynamics(
                    idx, -1,
                    lateralFriction=float(getattr(body, "friction", 0.5)),
                    rollingFriction=declared_roll,
                    spinningFriction=declared_roll)
            if surface is not None:
                pb.changeDynamics(surface[0], -1, lateralFriction=surface[1])
            state["restored"] = True

        return (restore,)

    @staticmethod
    def _surface(spec, simulator, objs, plan):
        """(pybullet index, declared friction) of the surface this plan grips
        through, or None."""
        sid = plan.params.get("surface_id")
        if sid is None or plan.params.get("surface_friction") is None:
            return None
        from ..render import stepper
        body = next((b for b in spec.bodies
                     if int(b.segmentation_id) == int(sid)), None)
        idx = stepper.pybullet_index(simulator, objs, spec, int(sid))
        if body is None or idx is None:
            return None
        return idx, float(getattr(body, "friction", 0.5))

    def unstage(self, spec, simulator, objs, plan) -> None:
        import pybullet as pb
        from ..render import stepper

        surface = self._surface(spec, simulator, objs, plan)
        if surface is not None:
            pb.changeDynamics(surface[0], -1, lateralFriction=surface[1])
        for bid in plan.causal_body_ids:
            body = next((b for b in spec.bodies
                         if int(b.segmentation_id) == int(bid)), None)
            if body is None or body.static:
                continue
            idx = stepper.pybullet_index(simulator, objs, spec, int(bid))
            if idx is not None:
                # Back to what the SCENE declares, not to zero. `pour`'s
                # grains carry rolling friction so the medium can hold a pile
                # at all, and zeroing it here would leave the next variant in
                # the run pouring ball bearings -- exactly the cross-family
                # leak `unstage` exists to prevent.
                roll = float(getattr(body, "rolling_friction", 0.0))
                pb.changeDynamics(
                    idx, -1,
                    lateralFriction=float(getattr(body, "friction", 0.5)),
                    rollingFriction=roll, spinningFriction=roll)

    def _apply(self, spec, traj, plan) -> Trajectory:
        actor = self._primary(spec)
        bi = traj.index_of(int(actor.segmentation_id))
        out = self._retimed(traj, bi, plan.t_event, float(plan.notes["end_rate"]))
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(Support())
register(Friction())

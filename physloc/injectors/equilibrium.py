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

    def strong_residual_reference(self, spec) -> float:
        return float(self.CLEARANCE_RADII["strong"])

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
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=[(t0, T - 1)],
            causal_body_ids=[int(b.segmentation_id) for b in targets] or
                            [int(actor.segmentation_id)],
            params={"type": "hover", "clearance_radii": clearance_r,
                    "mode": mode},
            magnitude=float(clearance_r * radius),
            magnitude_unit="m_support_clearance", severity_bin=severity_bin,
            notes={"radius": radius,
                   # With the WHOLE medium lifted there is nothing left beneath
                   # any of it, so the clearance datum is the floor. Measured
                   # against the pile -- which is what `support_under_any`
                   # returns for a grain in a column -- a grain hovering at
                   # z = 0.33 sat *below* its recorded support and the law read
                   # clearance 0.000 on an obviously airborne pour.
                   "surface_top": (float(_geom.surface_top(spec, actor))
                                   if len(targets) > 1 else float(top)),
                   "clearance_radii": clearance_r, "mode": mode})

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
            pb.resetBaseVelocity(idx, [float(v[0]), float(v[1]), 0.0], list(w))
            targets.append((idx, float(getattr(body, "mass", 1.0))))
        if not targets:
            return ()

        def weightless(_client, _step, _frame):
            for idx, mass in targets:
                pos, _ = pb.getBasePositionAndOrientation(idx)
                pb.applyExternalForce(idx, -1, (-g * mass).tolist(), list(pos),
                                      pb.WORLD_FRAME)

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
        strong = self._retimed(traj, bi, t0, self.RATE_BY_BIN["strong"])
        r_strong = self._measure(strong, int(actor.segmentation_id), "friction", {})
        mu = float(getattr(actor, "friction", 0.5))
        grip, roll, target = self._solve_grip(spec, traj, actor, bi, t0,
                                              severity_bin)
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=[(t0, T - 1)],
            # THE WHOLE MEDIUM. `_group` was already being consulted and then
            # thrown away -- only `targets[0]` was named -- so a pour had one
            # grain of forty gripping while the rest slid past it, which is
            # neither visible nor what a surface losing its slipperiness looks
            # like.
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "friction_scale", "end_rate": rate,
                    "lateral_friction": grip, "rolling_friction": roll,
                    "declared_friction": mu,
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
                   "r_strong": float(r_strong)})

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
        floor = self.MIN_RATIO_BY_BIN[severity_bin] * mu
        if actor.kind == "sphere":
            # Rolling resistance does the work; lateral friction only has to be
            # enough to keep it rolling rather than sliding.
            roll = min(self.MAX_ROLLING, max(accel * radius / g, floor * 0.1))
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
        for bid in plan.causal_body_ids:
            body = next((b for b in spec.bodies
                         if int(b.segmentation_id) == int(bid)), None)
            if body is None or body.static:
                continue
            idx = stepper.pybullet_index(simulator, objs, spec, int(bid))
            if idx is not None:
                pb.changeDynamics(idx, -1, lateralFriction=grip,
                                  rollingFriction=roll, spinningFriction=roll)
        return ()

    def unstage(self, spec, simulator, objs, plan) -> None:
        import pybullet as pb
        from ..render import stepper

        for bid in plan.causal_body_ids:
            body = next((b for b in spec.bodies
                         if int(b.segmentation_id) == int(bid)), None)
            if body is None or body.static:
                continue
            idx = stepper.pybullet_index(simulator, objs, spec, int(bid))
            if idx is not None:
                pb.changeDynamics(
                    idx, -1,
                    lateralFriction=float(getattr(body, "friction", 0.5)),
                    rollingFriction=0.0, spinningFriction=0.0)

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

"""Dynamics-domain injectors: phantom_impulse, angular_momentum.

(`newton2_mass` lives with `newton3_reaction` in contact.py -- they are one
mechanism at two settings of the same dial and share their collision-finding.)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..sim.trajectory import Trajectory
from . import _geom
from .base import Injector, InterventionPlan, register


class PhantomImpulse(Injector):
    """A body is shoved by nothing.

    The cleanest violation in the taxonomy and the one with the least ambiguity
    about *where*: the culprit is one body, the moment is one frame, and there
    is no second object anywhere near it. It fires only on frames with no
    recorded contact, because a shove during a collision is indistinguishable
    from the collision.
    """

    family = "phantom_impulse"
    #: Raised from 1.0 / 2.4 / 4.5. The frustum fit walks these back per scene
    #: wherever the geometry cannot host them, so the nominal value should be
    #: what reads clearly rather than what never overflows -- you judged the
    #: strongest bin too gentle on `drop` and elsewhere.
    DV_BY_BIN = {"weak": 1.6, "medium": 3.6, "strong": 7.0}      # m/s
    #: Frames the culprit may spend off camera before the fit weakens the shove.
    FRAME_TOLERANCE = 4

    def strong_residual_reference(self, spec) -> float:
        g_dt = 9.81 / float(spec.tier.fps)
        return float(self.DV_BY_BIN["strong"] / g_dt)

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        targets = self._group(spec)
        if not targets:
            return None
        actor = targets[0]
        T = traj.num_frames
        # Inside the contact-free run, with room left in it -- see
        # `_geom.acting_frame`. Clamping to the run's LAST frame is what this
        # did before, so on `drop` the shove landed one frame before the actor
        # hit the floor and the floor absorbed it. You reported the strongest
        # bin as barely visible; that was the reason, not the size of the push.
        t0 = _geom.acting_frame(spec, traj, int(actor.segmentation_id), T,
                                want=max(1, T // 3))
        if t0 is None or not (1 <= t0 < T - 1):
            return None

        # Direction is a property of the SCENE, not of the bin. Drawn from
        # the per-bin rng it came out different for weak, medium and strong,
        # so the three were three different violations and their magnitudes
        # stopped being comparable -- phantom_impulse reported 0.87 / 2.08 /
        # 1.95 for bins that scale 1.0 / 2.4 / 4.5, because each heading got
        # its own frustum-fit scale.
        heading = float(self._instance_rng(spec).uniform(0.0, 2.0 * np.pi))
        # AIMED BY THE BODY'S STATE, not one direction for every case.
        #
        # A body on a surface gets a mostly sideways shove with a little lift: a
        # purely horizontal one is easy to mistake for a nudge from off screen,
        # while a visible hop is unmistakably uncaused.
        #
        # A body IN FLIGHT gets a mostly vertical one, and this is the half that
        # was wrong. `drop` is framed tall and narrow around a fall, so a
        # sideways shove -- however hard -- takes the actor out of the side of
        # the frame within a few frames: measured, 5 of 21 frames off camera at
        # the very floor of the fit ladder, and 17 at nominal. The fit could
        # only respond by weakening the shove, so the strongest bin delivered
        # 1.9 m/s of a nominal 7.0 and still left the shot. Sending a falling
        # body UP instead is both unmistakable -- it stops falling and climbs,
        # with nothing to have done it -- and stays in the frame the scenario
        # already sized for the fall.
        bi = traj.index_of(int(actor.segmentation_id))
        top = _geom.surface_top(spec, actor)
        airborne = bool(traj.pos[t0, bi, 2] - float(traj.radius[bi])
                        > top + 0.05)
        if airborne:
            unit = np.array([0.30 * np.cos(heading), 0.30 * np.sin(heading), 1.0])
        else:
            unit = np.array([0.85 * np.cos(heading), 0.85 * np.sin(heading), 0.5])
        unit /= float(np.linalg.norm(unit))
        strongest = unit * self.DV_BY_BIN["strong"]
        # A LOOSER framing budget than the default. This family's whole content
        # is a large uncaused displacement, so the fit is the one thing standing
        # between it and legibility -- on `drop` it clamped to the floor of the
        # ladder and delivered 1.9 m/s of a nominal 7.0, which is the "barely
        # visible" you reported. Letting the actor drift out of shot for the
        # last frame or two costs the tail of the mask; a shove nobody can see
        # costs the whole clip.
        scale, _ = self._fit_to_frame(
            spec, traj, [actor], t0, strongest,
            lambda k: self._shoved(spec, traj, actor, t0, strongest * k),
            tolerance=self.FRAME_TOLERANCE)
        push = unit * self.DV_BY_BIN[severity_bin] * scale
        dv = float(np.linalg.norm(push))
        g_dt = float(np.linalg.norm(traj.gravity)) * traj.dt
        n_win = self._window_len(2, t0, T)
        # The shove lasts two frames; the ball it launched is a consequence for
        # the rest of the clip, and `causal_mask` is gated on that window.
        return InterventionPlan(
            family=self.family, kind="instant", t_event=t0,
            windows=[(t0, T - 1)],
            intervention_windows=[(t0, min(T - 1, t0 + n_win - 1))],
            consequence_windows=[(t0, T - 1)],
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "impulse", "delta_v": push.tolist(),
                    "frame_fit_scale": scale},
            magnitude=float(dv / max(g_dt, 1e-9)),
            magnitude_unit="impulse_over_m_vtyp", severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "surface_top": _geom.surface_top(spec, actor),
                   "delta_v_ms": dv})

    def _shoved(self, spec, traj, actor, t0: int, delta_v) -> Trajectory:
        out = self._clone(traj)
        bi = traj.index_of(int(actor.segmentation_id))
        v0 = (traj.lin_vel[t0 - 1, bi].astype(np.float64)
              + np.asarray(delta_v, np.float64))
        self._rewrite_from(spec, traj, out, actor, t0, v0=v0)
        return out

    simulated = True

    def stage(self, spec, simulator, objs, plan):
        """Add the impulse to the body's live velocity, then let physics run.

        Nothing about the shove needs prescribing beyond the instant it
        happens: where the body goes afterwards is whatever gravity, the floor
        and anything it hits decide, which is the point.
        """
        dv = np.asarray(plan.params["delta_v"], np.float64)
        for bid in plan.causal_body_ids:
            body = next(b for b in spec.bodies
                        if int(b.segmentation_id) == int(bid))
            obj = objs[body.name]
            obj.velocity = tuple(float(x) for x in
                                 np.asarray(obj.velocity, np.float64) + dv)
        return ()

    def _apply(self, spec, traj, plan) -> Trajectory:
        push = np.asarray(plan.params["delta_v"], np.float64)
        by_id = {int(b.segmentation_id): b for b in spec.bodies}
        bodies = [by_id[int(i)] for i in plan.causal_body_ids if int(i) in by_id]
        out = self._clone(traj)
        t0 = plan.t_event
        # Shoved together, so several bodies given the same uncaused push do not
        # each dodge where the others *would* have been and end up overlapping.
        self._rewrite_group(
            spec, traj, out, bodies, t0,
            v0_by_body={int(b.segmentation_id):
                        traj.lin_vel[t0 - 1, traj.index_of(int(b.segmentation_id))]
                        .astype(np.float64) + push for b in bodies})
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class AngularMomentum(Injector):
    """Spin that reverses, or grows, with nothing to torque it.

    Two ways a body can carry angular momentum, and the injector dispatches on
    which one the *scenario declares* -- never on the scenario's name:

    * **Free body** (`tumble`, `rolling_ramp`): the body's own spin is
      rescaled and the orientation re-integrated from there. Position is
      untouched, so the clip contains exactly one anomaly.
    * **Constrained body** (`spec.notes["constraint"] == "pivot"`): the whole
      swing is re-solved from the event frame with its angular rate rescaled,
      by asking the scenario to do it. A pendulum turning around in the middle
      of its arc is the violation; a bob whose *spin* reverses would be
      invisible, since the bob is a sphere.

    Fires inside a contact-free stretch. A body in contact can change its spin
    lawfully, so an edit made during contact is not a violation the residual can
    prove -- which is exactly why `rolling_ramp` has a lip to fly off.
    """

    family = "angular_momentum"
    # New spin as a multiple of the old: nearly stopped, exactly reversed,
    # reversed and faster.
    SPIN_BY_BIN = {"weak": 0.2, "medium": -1.0, "strong": -1.9}
    # ...unless the body is barely spinning, in which case rescaling zero is
    # zero and the clip would carry no violation at all. Then a spin is
    # *imposed* instead, which the family's own description covers: "spin
    # reverses, or torque appears with no contact". rad/s.
    IMPOSED_BY_BIN = {"weak": 1.5, "medium": 4.0, "strong": 8.0}
    SPINNING = 0.5      # rad/s, above which there is a spin worth reversing

    def strong_residual_reference(self, spec) -> float:
        omega = float(spec.notes.get("omega_ref", 0.0))
        if omega <= 0.0:
            omega = max([float(np.linalg.norm(b.angular_velocity))
                         for b in spec.bodies if not b.static] or [0.0])
        omega = max(omega, 2.0)
        radius = max([float(b.bounding_radius)
                      for b in _geom.actors(spec)] or [0.3])
        g_dt = 9.81 / float(spec.tier.fps)
        return abs(self.SPIN_BY_BIN["strong"] - 1.0) * omega * radius / g_dt

    # ------------------------------------------------------------------ #
    def _pivot(self, spec) -> bool:
        return spec.notes.get("constraint") == "pivot"

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = self._primary(spec)
        if actor is None:
            return None
        T = traj.num_frames
        if self._pivot(spec):
            t0 = max(1, T // 2)
        else:
            # Two frames is enough, and insisting on more is what killed
            # `rolling_ramp`: the run is padded by a frame either side of every
            # contact, so a four-frame hop off a ramp offers only two usable
            # frames and the cell produced no plan at all. Fire at the midpoint
            # and clamp *into* the run rather than shrinking it from both ends.
            free = _geom.contact_free_run(traj, int(actor.segmentation_id),
                                          min_len=2)
            if free is None:
                return None
            lo, hi = free
            t0 = min(max((lo + hi) // 2, max(1, lo)), hi)
        if not (1 <= t0 < T - 1):
            return None

        bi = traj.index_of(int(actor.segmentation_id))
        omega0 = float(np.linalg.norm(traj.ang_vel[t0 - 1, bi]))
        targets = self._all_actors(spec) if self._pivot(spec) else [actor]
        spinning = omega0 > self.SPINNING or self._pivot(spec)

        k = self.SPIN_BY_BIN[severity_bin] if spinning else 0.0
        imposed = None if spinning else self._imposed(traj, bi, severity_bin)
        magnitude = (abs(k - 1.0) * omega0 if spinning
                     else float(np.linalg.norm(imposed)))

        strong_k = self.SPIN_BY_BIN["strong"] if spinning else 0.0
        strong_imposed = (None if spinning
                          else self._imposed(traj, bi, "strong"))
        strong = self._preview(spec, traj, t0, strong_k, targets, strong_imposed)
        r_strong = self._measure(strong, int(actor.segmentation_id),
                                 "angular_momentum", {})

        t1 = min(T - 1, t0 + self._window_len(2, t0, T) - 1)
        return InterventionPlan(
            family=self.family, kind="instant", t_event=t0,
            windows=[(t0, t1)],
            intervention_windows=[(t0, t1)],
            # The torque is an impulse and the spin that follows it is lawful --
            # a free body conserves angular momentum, so nothing after `t1`
            # breaks the law again. But the body IS spinning differently for the
            # rest of the clip because of what happened at `t0`, and that is a
            # consequence, which is what `causal_mask` is gated on.
            consequence_windows=[(t0, T - 1)],
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "spin_scale" if spinning else "spin_impose",
                    "omega_scale": k,
                    "omega_imposed": None if imposed is None
                                     else [float(x) for x in imposed],
                    "constraint": "pivot" if self._pivot(spec) else "free"},
            magnitude=float(magnitude),
            magnitude_unit="angular_momentum_defect", severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "surface_top": _geom.surface_top(spec, actor),
                   "omega_scale": k, "omega_at_event": omega0,
                   "omega_imposed": None if imposed is None
                                    else [float(x) for x in imposed],
                   "r_strong": float(r_strong)})

    def _imposed(self, traj, bi: int, severity_bin: str) -> np.ndarray:
        """A spin to give a body that has none, about the axis across its travel.

        Tumbling forward reads as physical; spinning about the direction of
        motion reads as a rendering artefact, so the axis is chosen rather than
        picked at random.
        """
        v = traj.lin_vel[:, bi, :].astype(np.float64)
        speed = np.linalg.norm(v, axis=1)
        f = int(np.argmax(speed))
        heading = v[f, :2]
        n = float(np.linalg.norm(heading))
        axis = (np.array([-heading[1], heading[0], 0.0]) / n if n > 1e-6
                else np.array([0.0, 1.0, 0.0]))
        return axis * self.IMPOSED_BY_BIN[severity_bin]

    # ------------------------------------------------------------------ #
    def _preview(self, spec, traj, t0: int, k: float, targets,
                 imposed=None) -> Trajectory:
        out = self._clone(traj)
        if self._pivot(spec):
            from .. import scenarios as scen_mod
            if scen_mod.get(spec.scenario).rescript(spec, out, t0, k):
                return out
        for body in targets:
            bi = traj.index_of(int(body.segmentation_id))
            omega = (traj.ang_vel[t0 - 1, bi].astype(np.float64) * k
                     if imposed is None else np.asarray(imposed, np.float64))
            n = traj.num_frames - t0
            out.ang_vel[t0:, bi, :] = omega.astype(np.float32)[None, :]
            out.quat[t0:, bi, :] = _geom.integrate_quaternion(
                traj.quat[t0 - 1, bi], omega, traj.dt, n)
        return out

    #: STAGED for a FREE body. You reported the spin running on forever and the
    #: physics never becoming valid again: it could not, because the edited path
    #: wrote one constant angular velocity into every remaining frame, so the
    #: body kept that spin through its landing and along the ground. Handed to
    #: the solver as a one-off change of angular velocity, the torque is an
    #: impulse and everything after it is real -- the spin persists in flight,
    #: because that is what conservation means, and contact damps it when the
    #: body arrives.
    #:
    #: A PIVOT body keeps the edited path: its arc is scripted by the scenario
    #: over a constraint PyBullet has no joint for, so there is nothing in the
    #: simulator to hand the change to.
    simulated = True

    def simulates(self, plan) -> bool:
        return plan.params.get("constraint") != "pivot"

    def stage(self, spec, simulator, objs, plan):
        import pybullet as pb

        from ..render import stepper

        imposed = plan.notes.get("omega_imposed")
        k = float(plan.notes["omega_scale"])
        for bid in plan.causal_body_ids:
            idx = stepper.pybullet_index(simulator, objs, spec, int(bid))
            if idx is None:
                continue
            vel, ang = pb.getBaseVelocity(idx)
            omega = (np.asarray(imposed, np.float64) if imposed is not None
                     else np.asarray(ang, np.float64) * k)
            pb.resetBaseVelocity(idx, list(vel), omega.tolist())
        return ()

    def _apply(self, spec, traj, plan) -> Trajectory:
        """The host-side approximation, for the mock rollout the tests run on."""
        targets = [b for b in spec.bodies
                   if int(b.segmentation_id) in set(plan.causal_body_ids)]
        out = self._preview(spec, traj, plan.t_event,
                            float(plan.notes["omega_scale"]), targets,
                            plan.notes.get("omega_imposed"))
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(PhantomImpulse())
register(AngularMomentum())

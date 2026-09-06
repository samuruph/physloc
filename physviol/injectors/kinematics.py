"""Kinematics-domain injectors, plus the one global-domain family.

    antigravity      per-body gravity scale, bounded in time
    global_gravity   the same knob turned on the whole scene at once
    continuity       a discontinuous position set
    non_parabolic    free flight that no parabola fits
    newton1_inertia  a body that starts or stops with nothing acting on it

All five are pure trajectory edits, so they work on *any* scenario whose actor
is in the right state -- airborne, resting, sliding -- with no per-scenario code.
That composition is the point of the seam: thirteen scenarios times seventeen
families is 48 build cells and 21 files, not 221.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..sim.trajectory import Trajectory
from . import _geom
from .base import Injector, InterventionPlan, register


# ---------------------------------------------------------------------- #
def _event_frame(spec, traj, body, num_frames: int) -> Optional[int]:
    """When to fire on a body that ought to be in free flight.

    Hidden if the scenario offers an occlusion interval -- that is what makes
    the observability lag non-zero. Otherwise inside the longest airborne run,
    no earlier than a quarter of the way in so there is a lawful prefix to
    contradict. Falling back to a fixed fraction of the clip is what puts
    `drop` mid-bounce, and a parabola fitted across a bounce is not a
    parabola.
    """
    occ = _geom.occluded_midpoint(spec)
    if occ is not None and 1 <= occ < num_frames - 1:
        return int(occ)
    bi = traj.index_of(int(body.segmentation_id))
    run = _geom.longest_airborne_run(traj, bi, _geom.surface_top(spec, body))
    lo = max(1, num_frames // 4)
    if run is not None and run[1] > run[0]:
        # ANYWHERE in the airborne run, not at its lower edge. This used to be
        # `min(max(run[0] + 1, lo), run[1])`, and for a body that is airborne
        # from frame 0 -- every `drop`, every `toss` -- `lo` was binding on
        # every seed, so the family fired on frame `num_frames // 4` in every
        # clip it ever produced. `continuity` and `phantom_impulse` were pinned
        # that way across twelve scenarios each.
        t = _geom.frame_in_band(spec, max(run[0] + 1, lo), run[1])
        if 1 <= t < num_frames - 1:
            return int(t)
    # No airborne run at all -- a ball rolling the whole clip, a body at rest
    # on a table. The old fallback was the constant `num_frames // 3`, which is
    # why `continuity` and `phantom_impulse` fired on the same frame in every
    # clip of `collision`, `resting_table`, `stack_topple` and `pour`. Nothing
    # physical dictates the moment for such a body, which is precisely what
    # `default_event_frame` is for, jitter included.
    t = _geom.default_event_frame(spec, num_frames)
    if t is None:
        t = max(1, num_frames // 3)
    return int(t) if 1 <= t < num_frames - 1 else None


class _GravityScale(Injector):
    """Shared machinery for `antigravity` and `global_gravity`.

    They differ in *extent*, not in mechanism: one bends gravity for the culprit,
    the other for everything that moves. Keeping one implementation means the
    trapezoid profile, the solid floor and the bounded window are identical
    across both, so the only thing the label distinguishes is what it claims to.
    """

    ALPHA_BY_BIN = {"weak": -0.6, "medium": -1.6, "strong": -2.8}
    WINDOW_FRACTION = 0.55
    MIN_BODIES = 1
    spatial_extent = "local"

    def strong_residual_reference(self, spec) -> float:
        return abs(1.0 - self.ALPHA_BY_BIN["strong"])

    def _targets(self, spec):
        return self._all_actors(spec)

    #: Per-body gravity is not a PyBullet setting -- there is one world gravity
    #: and no per-object override -- so it is expressed as a counter-force
    #: applied on every substep. That is what the stepper's hooks exist for.
    simulated = True

    def stage(self, spec, simulator, objs, plan):
        """Bend gravity for the chosen bodies by applying `m*g*(alpha-1)`.

        A body under an effective gravity of `alpha*g` needs `m*g*(alpha-1)` on
        top of the world's `m*g`, and the force has to be re-applied every
        substep because PyBullet clears accumulated external forces after each
        step. Staged rather than integrated by hand, so a body that rises and
        falls does it while still colliding with everything around it: the
        trapezoid profile used to be written into the trajectory and then
        re-integrated against approximations of the scene.
        """
        import pybullet as pb
        from ..render import stepper

        # THE SAME TRAPEZOID the plan was fitted and annotated against, not the
        # peak held flat. `plan()` chooses the window length by rolling the
        # `_pulse` profile forward and shortening it until the actor stays in
        # shot; staging a constant `alpha_peak` instead applies roughly twice
        # the mean effect that fit was measured on, so the clip that renders is
        # not the clip that was fitted. Measured on `drop` at the debug tier:
        # the actor reached z = 6.57 in a shot framed for 2.84 and spent 12 of
        # 25 frames off camera -- which is an object leaving the frame, with an
        # empty mask and a severity field to match, inside a clip labelled
        # `antigravity`.
        t0, t1 = plan.windows[0]
        alpha_profile = self._pulse(t1 - t0 + 1,
                                    float(plan.params["alpha_peak"]))
        g = np.asarray(spec.gravity, np.float64)
        targets = []
        for bid in plan.causal_body_ids:
            body = next((b for b in spec.bodies
                         if int(b.segmentation_id) == int(bid)), None)
            if body is None or body.static:
                continue
            idx = stepper.pybullet_index(simulator, objs, spec, int(bid))
            if idx is not None:
                targets.append((idx, float(getattr(body, "mass", 1.0))))
        if not targets:
            return ()

        def bend(_client, _step, frame):
            if not (t0 <= frame <= t1):
                return
            alpha = float(alpha_profile[min(frame - t0,
                                            alpha_profile.shape[0] - 1)])
            for idx, mass in targets:
                force = (g * mass * (alpha - 1.0)).tolist()
                pos, _ = pb.getBasePositionAndOrientation(idx)
                pb.applyExternalForce(idx, -1, force, list(pos), pb.WORLD_FRAME)

        return (bend,)

    def _choose(self, spec, traj):
        """Which bodies to bend, given the rollout. Overridden by `antigravity`."""
        return self._targets(spec)

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        targets = self._choose(spec, traj)
        if len(targets) < self.MIN_BODIES:
            return None
        t0 = _event_frame(spec, traj, targets[0], traj.num_frames)
        if t0 is None:
            return None

        n_left = traj.num_frames - t0
        n_want = self._window_len(max(2, int(round(self.WINDOW_FRACTION * n_left))),
                                  t0, traj.num_frames)
        alpha = self.ALPHA_BY_BIN[severity_bin]
        strongest = self.ALPHA_BY_BIN["strong"]

        # Keep the bodies on screen by **shortening the window**, not by
        # weakening the intervention. The bin is a qualitative claim -- gravity
        # reverses and the body climbs -- and scaling alpha back until the body
        # fits turns that into "gravity is slightly reduced", which is a
        # different violation wearing this one's label. Fitted on the strongest
        # bin so all three share a window and stay comparable.
        n_win = self._fit_window_to_frame(
            spec, traj, targets, t0, n_want,
            lambda n: self._rollout(spec, traj, t0,
                                    min(traj.num_frames - 1, t0 + n - 1),
                                    strongest, targets))
        t1 = min(traj.num_frames - 1, t0 + n_win - 1)

        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0, windows=[(t0, t1)],
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "gravity_scale", "alpha_peak": alpha,
                    "profile": "trapezoid", "frames": int(t1 - t0 + 1),
                    "frames_requested": int(n_want),
                    "extent": self.spatial_extent},
            magnitude=abs(1.0 - alpha),
            magnitude_unit="gravity_scale_deviation",
            severity_bin=severity_bin, spatial_extent=self.spatial_extent,
            notes={"surface_top": _geom.surface_top(spec, targets[0]),
                   "radius": float(targets[0].bounding_radius),
                   "alpha_peak": alpha, "n_targets": len(targets)})

    def _rollout(self, spec, traj, t0: int, t1: int, alpha_peak: float,
                 targets) -> Trajectory:
        """Integrate the targets under a trapezoidal gravity pulse over [t0,t1].

        Shared by `plan` -- which runs it a few times to pick a peak that keeps
        the bodies in shot -- and by `apply`, so the clip that ships is exactly
        the one the plan was chosen from.
        """
        out = self._clone(traj)
        n_win, n_after = t1 - t0 + 1, traj.num_frames - (t1 + 1)
        g = traj.gravity.astype(np.float64)
        alpha = self._pulse(n_win, float(alpha_peak))
        g_seq = np.concatenate([alpha[:, None] * g[None, :],
                                np.tile(g[None, :], (max(n_after, 0), 1))])
        # Stepped together. Under `global_gravity` every moving body in the
        # scene is being re-integrated at once, so doing them one at a time
        # against each other's *lawful* paths made each dodge where its
        # neighbours would have been rather than where they now are -- which is
        # how a pyramid's spheres ended up passing through the cube that struck
        # them, a solidity failure inside a gravity clip.
        self._rewrite_group(spec, traj, out, targets, t0,
                            g_by_body={int(b.segmentation_id): g_seq
                                       for b in targets})
        out.meta = dict(traj.meta)
        out.meta["alpha_profile"] = [float(x) for x in alpha]
        return out

    def _apply(self, spec, traj, plan) -> Trajectory:
        t0, t1 = plan.windows[0]
        # The bodies the PLAN named, never a fresh choice. `_choose` picks the
        # most airborne actor while `_targets` picks a random one, so
        # re-deriving here bent a different grain than the plan had annotated
        # -- and since the annotated grain was then untouched, the clip came out
        # byte-identical to its valid twin with a full set of labels attached.
        by_id = {int(b.segmentation_id): b for b in spec.bodies}
        targets = [by_id[int(i)] for i in plan.causal_body_ids if int(i) in by_id]
        out = self._rollout(spec, traj, t0, t1,
                            float(plan.params["alpha_peak"]), targets)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class AntiGravity(_GravityScale):
    """Bend gravity on the actor for a bounded stretch of the clip.

    Severity is the peak deviation `|1 - alpha|`: 0 is normal gravity, 1 is
    weightless, 2 is gravity fully reversed.

    Two properties worth being explicit about, because the obvious
    implementation gets both wrong:

    * **The floor stays solid.** Changing gravity must not also drop the actor
      through the ground -- that is a *solidity* violation, and a clip carrying
      two violations while annotating one is worse than useless.
    * **The intervention has a shape.** Alpha ramps from 1 up to its peak and
      back to 1 over the window, after which the actor obeys real physics
      again. So the residual, and with it `severity_map`, traces a curve over
      time instead of sitting at a single value for the rest of the clip.

    **Every bin reverses the actor's direction**; they differ in how hard. A
    bin whose only effect is that things fall a bit slower is indistinguishable
    from a shutter-speed change, so all three are qualitative claims -- the body
    turns around and comes back -- and severity is how violently. When the
    geometry cannot host that, the *window* shortens rather than the reversal
    weakening (see `_fit_window_to_frame`).

    **It acts on exactly one body, never the whole scene.** That is the entire
    difference from `global_gravity`, and it is invisible unless something else
    in the frame is still falling normally -- which is why the compatibility
    matrix pairs the two with different scenarios, and why both only become
    fully legible at `--population multi`.
    """

    family = "antigravity"

    def _targets(self, spec):
        live = self._all_actors(spec)
        if not live:
            return []
        return [live[self._instance_rng(spec).randint(len(live))]]

    def _choose(self, spec, traj):
        """One actor, and one that is actually in the air.

        Bending gravity for a body resting on the floor produces almost nothing
        to measure: `free_fall` is gated on the frames where a body is
        unsupported, so a grain sitting in a pile has its residual zeroed for
        the whole window however hard gravity is reversed. That is how
        `pour x antigravity` came out with a clean-looking plan and a
        severity of exactly zero -- it had picked a grain that had already
        landed.

        Preferring the longest airborne run also keeps the choice deterministic
        and stable across severity bins, since it depends only on the valid
        rollout. The instance rng breaks ties, so a scene where every actor is
        equally airborne still varies which one is bent.

        How *many* comes from `_group`, so a scene made of many interchangeable
        bodies can ask for a share of them. One floating grain out of forty is
        a few pixels nobody will find; a dozen rising together is the point.
        """
        live = self._all_actors(spec)
        if not live:
            return []
        want = len(self._group(spec))
        jitter = self._instance_rng(spec)
        scored = []
        for body in live:
            bi = traj.index_of(int(body.segmentation_id))
            run = _geom.longest_airborne_run(
                traj, bi, _geom.surface_top(spec, body))
            span = 0 if run is None else (run[1] - run[0] + 1)
            scored.append(((span, float(jitter.rand())), body))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored[:max(1, want)]]


class GlobalGravity(_GravityScale):
    """The same bend, applied to every moving body at once.

    The label a consumer sees is `spatial_extent: "global"`, and that is the
    whole difference: nothing in the frame contradicts anything else, so the
    only evidence is the viewer's prior about how fast things fall on Earth.
    That makes it the hardest family in the taxonomy and the one where a
    per-pixel mask is least informative -- the mask is every moving body.

    Weak and medium stay inside the "this is a different planet" regime, which
    is the signature of a *global* constant being wrong: everything falls in
    slow motion together, and no object contradicts another. Strong leaves that
    regime and reverses, because `strong` is the development default and a
    strongest bin nobody can see is not worth generating.

    **Needs at least two moving bodies, and `plan` returns None below that.**
    With one object on screen, scaling gravity for the scene and scaling it for
    that object produce identical pixels -- the clip would be an `antigravity`
    clip with a different label, and the two families would be
    indistinguishable everywhere the dataset used them.
    """

    family = "global_gravity"
    ALPHA_BY_BIN = {"weak": 0.45, "medium": 0.05, "strong": -1.2}
    MIN_BODIES = 2
    spatial_extent = "global"

    def _targets(self, spec):
        # Not the SCRIPTED ones. A kinematic prop -- `pendulum_swing`'s rod,
        # which is hung between the pivot and the bob rather than falling -- is
        # not a body gravity acts on, and naming it as a culprit had a worse
        # consequence than a wrong mask: the worker will not stage an
        # intervention that touches a scripted body, so the whole family
        # dropped to the edited path and re-integrated the bob as a free body,
        # which took it straight off its rod.
        return [b for b in spec.bodies
                if not b.static and not b.dormant and not b.scripted]


class Continuity(Injector):
    """Teleport the actor by a fixed distance at a single frame.

    Instant, and the only family whose mask is genuinely two-lobed: the actor is
    both where it should be and where it wrongly is, which is precisely what the
    union rule in PLAN 3.3 exists to capture.
    """

    family = "continuity"
    JUMP_RADII = {"weak": 1.5, "medium": 4.0, "strong": 9.0}

    def strong_residual_reference(self, spec) -> float:
        return float(self.JUMP_RADII["strong"])

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        # THE WHOLE MEDIUM where the scene is made of interchangeable bodies.
        # One grain of forty jumping is perfectly annotated and impossible to
        # find; forty grains jumping together is the same violation said loudly
        # enough to see. `_group` collapses to the single actor everywhere else,
        # so this costs the other scenarios nothing.
        targets = self._group(spec)
        actor = targets[0] if targets else self._primary(spec)
        if actor is None:
            return None
        t0 = _event_frame(spec, traj, actor, traj.num_frames)
        if t0 is None:
            return None
        radius = float(actor.bounding_radius)
        jump_r = self.JUMP_RADII[severity_bin]
        # A FULL HEADING, drawn once per SCENE. Two separate constraints meet
        # here and it is easy to satisfy one by breaking the other:
        #
        # * Per scene, NOT per bin. Drawn from the per-bin rng the direction
        #   came out different for weak, medium and strong, so the three were
        #   three different violations and their magnitudes stopped being
        #   comparable -- phantom_impulse reported 0.87 / 2.08 / 1.95 for bins
        #   that scale 1.0 / 2.4 / 4.5, because each heading got its own
        #   frustum-fit scale.
        # * A heading, NOT a sign. Fixing the first by hard-coding `+x`/`-x`
        #   left twelve seeds producing two distinct teleport directions, where
        #   phantom_impulse's heading draw produced twelve. A model can learn
        #   an axis; it cannot learn a heading.
        heading = float(self._instance_rng(spec).uniform(0.0, 2.0 * np.pi))
        direction = np.array([np.cos(heading), np.sin(heading), 0.0])
        nominal = direction * jump_r * radius
        # A teleport big enough to leave the frame depicts an object vanishing,
        # which is `permanence`, not `continuity`. Shorten it until both lobes
        # of the two-lobed mask are actually in shot -- fitting on the longest
        # jump so all three bins shrink together and stay ordered.
        strongest = direction * self.JUMP_RADII["strong"] * radius
        scale, _ = self._fit_to_frame(
            spec, traj, [actor], t0, strongest,
            lambda k: self._teleport(traj, actor, t0, strongest * k))
        delta = nominal * scale
        # LAND IT CLEAR. A teleport is a claim about position and nothing else,
        # so the body must not arrive inside something: on `collision` it
        # landed slightly overlapping the ball it was supposed to skip past,
        # and the solver's first job on the next frame was to push the two
        # apart -- an ejection the clip neither claims nor annotates. Lengthen
        # the jump first, since that is still the same violation only bigger,
        # and only then try the other direction.
        delta = self._clear_landing(spec, traj, actor, t0, delta)
        return InterventionPlan(
            family=self.family, kind="instant", t_event=t0,
            windows=[(t0, traj.num_frames - 1)],
            intervention_windows=[(t0, t0)],
            consequence_windows=[(t0, traj.num_frames - 1)],
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "position_set", "delta_m": delta.tolist(),
                    "frame_fit_scale": scale},
            magnitude=float(np.linalg.norm(delta)),
            magnitude_unit="m_jump_distance",
            severity_bin=severity_bin,
            notes={"radius": radius, "jump_radii": jump_r,
                   "surface_top": _geom.surface_top(spec, actor)})

    simulated = True

    def stage(self, spec, simulator, objs, plan):
        """Move the body, keep its velocity, and let physics take over.

        You reported the teleported ball bouncing off a barrier it had already
        been moved past. It did: the trajectory was edited and then
        re-integrated against the scene as *declared*, so the resolver kept the
        wall in front of a body that was behind it. Staged into the simulator
        there is nothing to bounce off, because PyBullet is looking at where the
        body actually is.
        """
        delta = np.asarray(plan.params["delta_m"], np.float64)
        for bid in plan.causal_body_ids:
            body = next(b for b in spec.bodies
                        if int(b.segmentation_id) == int(bid))
            obj = objs[body.name]
            obj.position = tuple(float(x) for x in
                                 np.asarray(obj.position, np.float64) + delta)
        return ()

    def _teleport_all(self, traj, bodies, t0: int, delta) -> Trajectory:
        out = self._clone(traj)
        for body in bodies:
            bi = traj.index_of(int(body.segmentation_id))
            out.pos[t0:, bi, :] = (traj.pos[t0:, bi, :]
                                   + np.asarray(delta, np.float32))
        return out

    #: Multiples of the fitted jump to try before giving up, then the same
    #: distances the other way. Extending is preferred: a longer jump is the
    #: same violation at a larger magnitude, where reversing it is a different
    #: clip.
    LANDING_TRIES = (1.0, 1.15, 1.3, 1.5, 1.75, -1.0, -1.15, -1.3, -1.5)

    def _clear_landing(self, spec, traj, actor, t0: int, delta):
        """The nearest version of this jump that does not land inside anything.

        Everything else in the scene is where the VALID rollout puts it at
        `t0`, which is exactly right: the teleport happens before anything has
        had a chance to react to it.
        """
        bi = traj.index_of(int(actor.segmentation_id))
        radius = float(traj.radius[bi])
        start = np.asarray(traj.pos[t0, bi], np.float64)
        obstacles = _geom.Obstacles(spec, traj,
                                    exclude_ids=[int(actor.segmentation_id)])
        for k in self.LANDING_TRIES:
            candidate = np.asarray(delta, np.float64) * float(k)
            target = start + candidate
            moved, _ = obstacles.resolve(target.copy(), np.zeros(3), radius,
                                         0.0, float(t0))
            if float(np.linalg.norm(moved - target)) > 1e-6:
                continue                      # something pushed it: it overlaps
            if not bool(_geom.in_frame(spec, target[None, :])[0]):
                continue
            return candidate
        return np.asarray(delta, np.float64)

    #: How far along its own cycle a constrained body jumps, in seconds. Large
    #: enough to be plainly a different point of the swing, short of the half
    #: period that would merely mirror it.
    PHASE_SHIFT_SECONDS = 0.45

    def _teleport(self, traj, actor, t0: int, delta, spec=None) -> Trajectory:
        out = self._clone(traj)
        bi = traj.index_of(int(actor.segmentation_id))
        # Horizontal only, so the teleport cannot smuggle in a floor violation.
        out.pos[t0:, bi, :] = traj.pos[t0:, bi, :] + np.asarray(delta, np.float32)
        return out

    def _apply(self, spec, traj, plan) -> Trajectory:
        by_id = {int(b.segmentation_id): b for b in spec.bodies}
        bodies = [by_id[int(i)] for i in plan.causal_body_ids if int(i) in by_id]
        out = self._teleport_all(traj, bodies, plan.t_event,
                                 np.asarray(plan.params["delta_m"], np.float32))
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class NonParabolic(Injector):
    """Free flight that snakes: no parabola fits it, and nothing jumps.

    The point of the family is to be wrong in a way the other two kinematic
    families are not. `antigravity` gets the *rate* of fall wrong; `continuity`
    puts the body somewhere it could not have travelled to. This one keeps both
    right -- the mean acceleration over the flight is still g, and the path is
    continuous with a continuous derivative everywhere -- and gets the *shape*
    wrong. A model that only checks how fast things fall sees nothing.

    Earlier this was a single sine bump over five frames, and it read as a
    teleport: at 12 fps a one-cycle wobble large enough to notice is a body
    appearing a body-width away and coming back. The fix is more cycles over
    more frames at a smaller amplitude -- a visible snake rather than a
    displacement.

    **The wobble is laid out in the camera's image plane**, using the same basis
    the frustum test uses. Perturbing a world axis is a gamble on where the
    camera happens to be: the same offset that snakes across the frame from one
    viewpoint is pure depth from another, and depth on a uniformly lit primitive
    is nearly invisible.
    """

    family = "non_parabolic"
    #: Peak lateral excursion in body radii. Small on purpose -- legibility here
    #: comes from the number of cycles, not from the size of each one.
    AMPLITUDE_RADII = {"weak": 0.5, "medium": 0.9, "strong": 1.5}
    #: Fewer cycles than the 2.5 this shipped with. At v0's 30 fps a 2.5-cycle
    #: snake over a short airborne run turns each half-cycle over in a handful
    #: of frames, which reads as a jitter rather than as a path -- you asked for
    #: it smoother, and the amplitude is not the part to change: legibility here
    #: comes from the shape being continuous and obviously not a parabola.
    CYCLES = 1.75

    def strong_residual_reference(self, spec) -> float:
        # The least-squares parabola absorbs a little of a symmetric wobble, so
        # the peak measured deviation lands just under the amplitude asked for.
        return float(self.AMPLITUDE_RADII["strong"]) * 0.9

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        # THE WHOLE MEDIUM -- see `Continuity.plan`. A single grain snaking
        # inside a pour is invisible; forty snaking together is a pour that
        # does not fall the way anything falls.
        targets = self._group(spec)
        actor = targets[0] if targets else self._primary(spec)
        if actor is None:
            return None
        bi = traj.index_of(int(actor.segmentation_id))
        top = _geom.surface_top(spec, actor)
        run = _geom.longest_airborne_run(traj, bi, top)
        if run is None or run[1] - run[0] < 4:
            return None
        # The whole airborne stretch, not a slice of it: a serpentine needs room
        # for several cycles, and the residual is fitted over exactly these
        # frames so a parabola through two arcs never enters the comparison.
        t0, t1 = int(run[0]) + 1, int(run[1])
        if t1 - t0 < 3:
            return None

        radius = float(actor.bounding_radius)
        amp = self.AMPLITUDE_RADII[severity_bin] * radius
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0, windows=[(t0, t1)],
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "path_warp", "profile": "serpentine",
                    "amplitude_m": amp, "cycles": self.CYCLES,
                    "frames": int(t1 - t0 + 1)},
            magnitude=float(amp), magnitude_unit="m_rms_from_parabola",
            severity_bin=severity_bin,
            notes={"radius": radius, "surface_top": top,
                   "flight_frames": list(range(int(run[0]), int(run[1]) + 1))})

    #: STAGED. A path that is not a parabola requires a FORCE, and a force is
    #: precisely what a per-substep hook can apply -- so this family had no
    #: business editing a trajectory. Staged, the body snakes because something
    #: is pushing it, and it still lands on the floor, still collides with what
    #: it meets, and still carries its neighbours with it if it strikes them.
    simulated = True

    def _offsets(self, spec, plan, n: int):
        """The lateral offset per frame, in the camera's image plane."""
        amp = float(plan.params["amplitude_m"])
        cycles = float(plan.params["cycles"])
        _, _, right, up = _geom.camera_basis(spec)
        u = (np.arange(n, dtype=np.float64) + 1.0) / (n + 1.0)
        envelope = np.sin(np.pi * u) ** 2
        phase = 2.0 * np.pi * cycles * u
        return ((amp * envelope * np.sin(phase))[:, None] * up[None, :]
                + (0.6 * amp * envelope * np.cos(phase))[:, None]
                * right[None, :])

    def stage(self, spec, simulator, objs, plan):
        import pybullet as pb

        from ..render import stepper

        t0, t1 = plan.windows[0]
        n = t1 - t0 + 1
        offset = self._offsets(spec, plan, n)
        dt = 1.0 / float(spec.tier.fps)
        # The force that produces that offset: its second derivative, times the
        # mass. Differenced twice rather than differentiated analytically so the
        # staged path and `_apply` describe the same curve.
        accel = np.zeros_like(offset)
        accel[1:-1] = (offset[2:] - 2.0 * offset[1:-1] + offset[:-2]) / (dt * dt)
        targets = []
        for bid in plan.causal_body_ids:
            body = next((b for b in spec.bodies
                         if int(b.segmentation_id) == int(bid)), None)
            if body is None or body.static:
                continue
            idx = stepper.pybullet_index(simulator, objs, spec, int(bid))
            if idx is not None:
                targets.append((idx, float(getattr(body, "mass", 1.0))))
        if not targets:
            return ()

        def snake(_client, _step, frame):
            if not (t0 <= frame <= t1):
                return
            a = accel[min(frame - t0, n - 1)]
            for idx, mass in targets:
                at, _ = pb.getBasePositionAndOrientation(idx)
                pb.applyExternalForce(idx, -1, (a * mass).tolist(), list(at),
                                      pb.WORLD_FRAME)

        return (snake,)

    def _apply(self, spec, traj, plan) -> Trajectory:
        """The host-side approximation, for the mock rollout the tests run on."""
        out = self._clone(traj)
        t0, t1 = plan.windows[0]
        n = t1 - t0 + 1
        amp = float(plan.params["amplitude_m"])
        cycles = float(plan.params["cycles"])
        top = float(plan.notes["surface_top"])
        radius = float(plan.notes["radius"])

        _, _, right, up = _geom.camera_basis(spec)
        # A Hann envelope holds the offset and its slope at zero on both ends,
        # so the body joins and leaves its lawful arc without a corner and the
        # frames outside the window stay untouched.
        u = (np.arange(n, dtype=np.float64) + 1.0) / (n + 1.0)
        envelope = np.sin(np.pi * u) ** 2
        phase = 2.0 * np.pi * cycles * u
        offset = (amp * envelope * np.sin(phase))[:, None] * up[None, :] \
            + (0.6 * amp * envelope * np.cos(phase))[:, None] * right[None, :]

        for bid in plan.causal_body_ids:
            bi = traj.index_of(int(bid))
            pos = traj.pos[t0:t1 + 1, bi, :].astype(np.float64) + offset
            pos[:, 2] = np.maximum(pos[:, 2], top + radius)
            out.pos[t0:t1 + 1, bi, :] = pos.astype(np.float32)

            vel = out.lin_vel[t0:t1 + 1, bi, :].astype(np.float64)
            vel[0] = (pos[0] - traj.pos[t0 - 1, bi]) / traj.dt
            vel[1:] = (pos[1:] - pos[:-1]) / traj.dt
            out.lin_vel[t0:t1 + 1, bi, :] = vel.astype(np.float32)

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class Newton1Inertia(Injector):
    """A moving body stops dead with nothing to stop it, and stays stopped.

    Halting only. It used to do double duty -- halt a moving body *or* shove a
    resting one -- and the second half was `phantom_impulse` wearing a different
    label: an uncaused velocity change on a body with nothing touching it. Two
    families that overlap on half their cases cannot be scored independently,
    which is the whole point of keeping them apart, so the shove belongs to
    `phantom_impulse` and the halt belongs here.

    That also settles the family's `kind`. The two branches disagreed --
    stopping is a state that persists, shoving is an event -- so the taxonomy
    could only be right about one of them. Now the window runs to the end of the
    clip, because a body sitting motionless where it should still be sliding is
    violating Newton 1 for every frame it sits there.
    """

    family = "newton1_inertia"
    persistent = True
    #: How much of the body's speed is removed. Below 1.0 it merely slows,
    #: which reads as friction; at 1.0 it stops dead, which reads as wrong.
    HALT_BY_BIN = {"weak": 0.55, "medium": 0.85, "strong": 1.0}
    MOVING = 0.3                                                # m/s
    STRONG_REFERENCE = 3.0                                      # dv / (g*dt)

    def strong_residual_reference(self, spec) -> float:
        return self.STRONG_REFERENCE

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        # THE WHOLE MEDIUM -- see `Continuity.plan`. Forty grains stopping dead
        # together is Newton 1 said at a volume a viewer can hear; one grain of
        # forty doing it is a few pixels in a pile.
        targets = self._group(spec)
        actor = targets[0] if targets else self._primary(spec)
        if actor is None:
            return None
        T = traj.num_frames
        bi = traj.index_of(int(actor.segmentation_id))
        speed = np.linalg.norm(traj.lin_vel[:, bi, :].astype(np.float64), axis=1)
        moving = np.flatnonzero(speed > self.MOVING)
        moving = moving[(moving >= 1) & (moving < T - 2)]
        if not moving.size:
            return None                 # nothing to halt: not a cell we can build
        t0 = int(max(moving[0], min(moving[-1], T // 3)))

        fraction = float(self.HALT_BY_BIN[severity_bin])
        g_dt = float(np.linalg.norm(traj.gravity)) * traj.dt
        # The halt happens between two frames; the body then simply stays
        # stopped, which is the consequence rather than more intervening.
        union, applied, after = self._split_windows(t0, T, 1)
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=union, intervention_windows=applied,
            consequence_windows=after,
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "velocity_damp", "removed_fraction": fraction},
            magnitude=float(speed[t0] * fraction / max(g_dt, 1e-9)),
            magnitude_unit="dv_over_g_dt", severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "surface_top": _geom.surface_top(spec, actor),
                   "removed_fraction": fraction,
                   "speed_at_event": float(speed[t0])})

    simulated = True

    def stage(self, spec, simulator, objs, plan):
        """Take the body's speed away and leave it there.

        Staged rather than written, so what happens next is real: a body halted
        on a slope is held by friction if friction can hold it and slides if it
        cannot, instead of being pinned in place by an injector that decided
        the answer in advance.
        """
        keep = 1.0 - float(plan.notes["removed_fraction"])
        for bid in plan.causal_body_ids:
            body = next(b for b in spec.bodies
                        if int(b.segmentation_id) == int(bid))
            obj = objs[body.name]
            obj.velocity = tuple(float(x) * keep for x in obj.velocity)
            obj.angular_velocity = tuple(float(x) * keep
                                         for x in obj.angular_velocity)
        return ()

    def _apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        t0 = plan.t_event
        keep = 1.0 - float(plan.notes["removed_fraction"])
        for bid in plan.causal_body_ids:
            self._halt_one(traj, out, traj.index_of(int(bid)), t0, keep)
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out

    def _halt_one(self, traj, out, bi: int, t0: int, keep: float) -> None:
        if True:
            # Frozen exactly where it was. Held rather than re-integrated: a
            # body that stops on a slope must stay on the slope, and gravity
            # would slide it back off.
            out.pos[t0:, bi, :] = traj.pos[t0 - 1, bi][None, :]
            out.quat[t0:, bi, :] = traj.quat[t0 - 1, bi][None, :]
            out.lin_vel[t0:, bi, :] = 0.0
            out.ang_vel[t0:, bi, :] = 0.0
        else:
            # Partly damped: follow the body's own lawful path at a slower rate,
            # so it stays on whatever surface it was travelling along.
            n = traj.num_frames - t0
            u = float(t0 - 1) + np.cumsum(np.full((n,), keep))
            pos = _geom.path_sample(traj.pos[:, bi, :], u)
            out.pos[t0:, bi, :] = pos.astype(np.float32)
            out.quat[t0:, bi, :] = traj.quat[t0 - 1, bi][None, :]
            out.ang_vel[t0:, bi, :] = (traj.ang_vel[t0:, bi, :] * keep)
            self._sync_velocity(traj, out, bi, t0)


register(AntiGravity())
register(GlobalGravity())
register(Continuity())
register(NonParabolic())
register(Newton1Inertia())


class TimeSlip(Injector):
    """A body stalls in place, then resumes exactly where it left off.

    The dataset's one *temporal* violation, and it is deliberately not the one
    LikePhys ships. Their Temporal Continuity row is frame shuffling, which is
    an encoding artifact: it has no per-body residual, it cannot carry a mask
    because every pixel changes at once, and a detector wins on it from
    low-level statistics without doing any physics -- exactly the shortcut the
    artifact-probe control exists to catch. So the principle is staged as
    physics instead.

    Distinct from both its neighbours, and not by a hair:

    - Not `newton1_inertia`, which destroys momentum permanently -- the body
      stops and *stays* stopped. This one hands the momentum back, same
      direction and same magnitude, which no contact can do. There are two
      uncaused events, at the start of the stall and at its end.
    - Not `continuity`, because nothing jumps in space. Same path, same route,
      traversed with a pause in it, so `position_continuity` reads zero
      throughout.

    Two constraints came out of building it, both of which would otherwise ship
    a second unlabelled violation in the clip:

    1. **The body must be supported, never airborne.** A body that freezes in
       mid-air is also hovering, so `free_fall` fires and the clip is a
       gravitation violation too. Hence `requires=('sliding',)`.
    2. **The stall must come after the body's last contact with another dynamic
       body.** The shift persists to the end of the clip, so a collision that
       happens after it would have one participant arriving late while the
       other reacts on time -- the struck ball moving before it is struck.
       Static contacts are fine: a wall is there on every frame, so meeting it
       late is still meeting it lawfully.
    """

    family = "time_slip"
    #: The slip length is the magnitude, so `--window` must not override it.
    persistent = True
    #: Stall duration as a fraction of the clip. Scales with tier, like every
    #: other duration in the project. Capped well short of the clip because the
    #: body has to visibly *resume* -- see RESUME_RADII.
    SLIP_BY_BIN = {"weak": 0.13, "medium": 0.22, "strong": 0.32}
    SLIP_MIN = 2
    MOVING = 0.3                                                # m/s
    #: How far the body must still travel after the stall, in radii. Without
    #: this the first rendered clip stalled the ball behind `occluder_pass`'s
    #: screen and the clip ended before it came out the other side -- which is
    #: an object that went behind a screen and never reappeared, i.e. a
    #: `permanence` clip wearing a `time_slip` label. The whole violation here
    #: is that the motion comes *back*; if the resumption is off the end of the
    #: clip, there is no violation to see, only a different family's.
    RESUME_RADII = 3.0

    def strong_residual_reference(self, spec) -> float:
        return max(float(self.SLIP_MIN),
                   self.SLIP_BY_BIN["strong"] * spec.tier.num_frames)

    def _slip_frames(self, spec, severity_bin: str, T: int) -> int:
        return int(max(self.SLIP_MIN,
                       round(self.SLIP_BY_BIN[severity_bin] * T)))

    def _candidate(self, spec, traj, n: int):
        """(body, t0, v_ref) for a body that can carry the stall, or None.

        Not simply `_primary`. In `collision` the striker has spent its momentum
        by the time its last dynamic contact is behind it, so the only body
        still moving afterwards is the one it struck -- and a family that
        returned no plan there would lose the scenario that makes the
        comparison with `newton1_inertia` interesting.
        """
        T = traj.num_frames
        # The MIDDLE of the clip, not the middle of an occlusion.
        # `default_event_frame` hides an event behind a screen wherever the
        # scenario offers one, and a stall that happens entirely out of sight is
        # a body that emerges late -- which is not something a viewer can read
        # as a stall at all. You reported it as barely visible on
        # `occluder_pass`; this is why.
        want = max(1, int(round(_geom.EVENT_FRACTION * T)))
        occluded = set(int(f) for f in (spec.notes.get("occluded_frames") or []))
        order = _geom.actors(spec)
        primary = self._primary(spec)
        if primary is not None:
            order = [primary] + [b for b in order if b is not primary]
        best = None
        for body in order:
            if body.dormant or body.static:
                continue
            bid = int(body.segmentation_id)
            bi = traj.index_of(bid)
            pos = np.asarray(traj.pos[:, bi, :], np.float64)
            # Path LENGTH, not displacement. `barrier_pass` rolls into a wall
            # and comes back, so its net displacement over the tail is near
            # zero while it has visibly travelled several body-widths -- the
            # displacement version rejected that scenario outright.
            arc = np.concatenate([[0.0], np.cumsum(
                np.linalg.norm(np.diff(pos, axis=0), axis=1))])
            speed = np.linalg.norm(
                traj.lin_vel[:, bi, :].astype(np.float64), axis=1)
            radius = max(float(traj.radius[bi]), 1e-9)
            lo = max(1, _geom.last_dynamic_contact(spec, traj, bid) + 1)
            hi = T - n - 1
            if hi < lo:
                continue
            for t0 in range(int(lo), int(hi) + 1):
                if speed[t0] <= self.MOVING:
                    continue
                # Distance the body still covers along its lawful path after
                # the stall, which is what the viewer sees it resume through.
                tail = T - min(t0 + n, T)
                if tail < 1:
                    continue
                resumed = float(arc[t0 + tail - 1] - arc[t0])
                if resumed < self.RESUME_RADII * radius:
                    continue
                # As close to mid-clip as the window allows, and IN SHOT: a
                # frame spent stalled behind a screen carries none of the
                # evidence, so each one costs more than being a frame further
                # from the ideal moment.
                hidden = sum(1 for f in range(t0, min(t0 + n, T))
                             if f in occluded)
                score = abs(t0 - want) + 3 * hidden
                if best is None or score < best[0]:
                    best = (score, body, t0, float(speed[t0]))
        return None if best is None else best[1:]

    #: Frames the body must be visible for after it re-emerges. The occlusion test
    #: is for FULL occlusion, so partial visibility starts a frame or two after
    #: the last occluded frame -- hence the margin rather than 1.
    SEEN_AFTER = 5

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        T = traj.num_frames
        n = self._slip_frames(spec, severity_bin, T)
        # Where the scenario hides the actor, the whole clip's occlusion shifts
        # by `n` too, so the stall has to be short enough that the body still
        # comes back out with frames to spare. Path length alone is not enough:
        # the first version travelled its three radii entirely behind the
        # screen and appeared only in the final frame, which reads as
        # `permanence` rather than as a stall. Capped rather than refused --
        # a slightly shorter strong bin at the debug tier is honest, because the
        # magnitude is reported.
        occ = spec.notes.get("occluded_frames")
        if occ:
            n = min(n, max(self.SLIP_MIN,
                           T - self.SEEN_AFTER - int(max(occ))))
        found = self._candidate(spec, traj, n)
        if found is None:
            return None
        actor, t0, v_ref = found
        bid = int(actor.segmentation_id)

        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=[(t0, min(t0 + n, T - 1))],
            causal_body_ids=[bid],
            params={"type": "time_slip", "slip_frames": int(n)},
            magnitude=float(n), magnitude_unit="slip_frames",
            severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "surface_top": _geom.surface_top(spec, actor),
                   "slip_frames": int(n), "v_ref": v_ref,
                   "r_strong": self.strong_residual_reference(spec)})

    def _apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        bi = traj.index_of(int(plan.causal_body_ids[0]))
        t0 = plan.t_event
        n = int(plan.notes["slip_frames"])
        T = traj.num_frames

        # Held, not re-integrated: a body stalled on a slope must stay on the
        # slope, and re-integrating it under gravity would slide it back off --
        # which is `newton1_inertia`'s lesson, learned there first.
        end = min(t0 + n, T)
        out.pos[t0:end, bi, :] = traj.pos[t0, bi][None, :]
        out.quat[t0:end, bi, :] = traj.quat[t0, bi][None, :]
        out.lin_vel[t0:end, bi, :] = 0.0
        out.ang_vel[t0:end, bi, :] = 0.0

        # Then the body's own lawful path again, `n` frames late. The tail of
        # the valid motion falls off the end of the clip, which is what "behind
        # in time" means.
        tail = T - end
        if tail > 0:
            out.pos[end:, bi, :] = traj.pos[t0:t0 + tail, bi, :]
            out.quat[end:, bi, :] = traj.quat[t0:t0 + tail, bi, :]
            out.lin_vel[end:, bi, :] = traj.lin_vel[t0:t0 + tail, bi, :]
            out.ang_vel[end:, bi, :] = traj.ang_vel[t0:t0 + tail, bi, :]

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(TimeSlip())

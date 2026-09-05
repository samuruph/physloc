"""Appearance-domain injectors: deformation, colour_shift.

A domain of its own because none of the seven physical ones covers it. A body
that squashes out of proportion has not moved unlawfully, gained energy or
broken a contact -- everything about its *trajectory* is fine. What is wrong is
that it stopped looking like itself, and that is among the most common things
real video generators get wrong.

(`shadow_shape` uses the same mechanism and the same law but lives in `optical`,
because the thing losing its shape there is the shadow rather than the object.)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..sim.trajectory import Trajectory
from . import _geom
from ..residuals.laws import _srgb_to_lab
from .base import Injector, InterventionPlan, register


class _Squash(Injector):
    """Shared machinery: ease a body into a non-uniform scale and hold it.

    Volume is preserved -- one axis grows by `k`, the other two shrink by
    `1/sqrt(k)` -- so `shape_continuity` stays silent and only the aspect ratio
    moves. That separation is the whole reason the two laws exist: a family that
    changed size *and* proportions at once would fire both, and neither
    measurement would mean anything on its own.
    """

    persistent = True
    ASPECT_BY_BIN = {"weak": 1.35, "medium": 1.9, "strong": 2.8}
    #: In SECONDS, not frames. The tiers differ by more than a factor of two in
    #: frame rate, so a constant expressed in frames is a different-looking
    #: violation at each of them -- a five-frame ramp is 0.42 s at the debug
    #: tier and 0.17 s at v0, which is the difference between a body visibly
    #: distorting and a body changing between two frames.
    RAMP_SECONDS = 0.4

    def strong_residual_reference(self, spec) -> float:
        k = self.ASPECT_BY_BIN["strong"]
        return abs(k * np.sqrt(k) - 1.0)

    def _targets(self, spec):
        return self._group(spec)

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        targets = self._targets(spec)
        if not targets:
            return None
        T = traj.num_frames
        t0 = _geom.default_event_frame(spec, T)
        if t0 is None:
            return None

        k = float(self.ASPECT_BY_BIN[severity_bin])
        # Which axis stretches is a property of the scene, not of the bin, or
        # the three strengths would be three different distortions.
        #
        # A body RESTING on a surface may only stretch horizontally. Stretching
        # it vertically makes it taller, which either drives it into the floor
        # or lifts its centre of mass -- and lifting a centre of mass is real
        # work done from nowhere, so the clip would depict a shape violation
        # *and* an energy violation while claiming one. That is the same trap
        # the volume-preserving scaling avoids on the mass axis: this family's
        # whole job is to move the aspect ratio and nothing else.
        #
        # Among the horizontal axes the choice is the CAMERA's, not the dice's.
        # Stretching along the viewing direction is nearly invisible, and on
        # `occluder_pass` it was worse than invisible: the actor's new bulk
        # pushed out of the front face of the screen it was hidden behind, so
        # the clip read as the body appearing in front of the occluder.
        rng = self._instance_rng(spec)
        top = _geom.surface_top(spec, targets[0])
        radius = float(targets[0].bounding_radius)
        resting = bool(traj.pos[t0, traj.index_of(
            int(targets[0].segmentation_id)), 2] - radius <= top + 1e-2)
        lateral = _geom.lateral_axis(spec)
        axis = lateral if (resting or rng.rand() < 0.5) else 2
        ramp = self._frames_for(spec, self.RAMP_SECONDS, minimum=2)
        ramp = max(2, min(ramp, T - t0))
        union, applied, after = self._split_windows(t0, T, ramp)
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=union, intervention_windows=applied,
            consequence_windows=after,
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "aspect_scale", "aspect": k, "axis": axis,
                    "ramp_frames": int(ramp)},
            magnitude=float(k), magnitude_unit=self.magnitude_unit,
            severity_bin=severity_bin,
            notes={"radius": radius, "surface_top": top,
                   "aspect": k, "axis": axis, "ramp_frames": int(ramp),
                   "resting": resting,
                   "r_strong": self.strong_residual_reference(spec)})

    def _profile(self, plan, n: int) -> np.ndarray:
        """[n,3] per-axis scale from `t_event`: one axis grows, two shrink.

        Shared by the staged hook and by `_apply`, so the collision shape the
        simulator is handed and the scale the renderer draws are one number
        rather than two implementations of the same intention.
        """
        k = float(plan.notes["aspect"])
        axis = int(plan.notes["axis"])
        ramp = max(1, int(plan.notes["ramp_frames"]))
        u = np.clip((np.arange(n, dtype=np.float64) + 1.0) / ramp, 0.0, 1.0)
        grow = 1.0 + (k - 1.0) * (u * u * (3.0 - 2.0 * u))
        factor = np.tile((1.0 / np.sqrt(grow))[:, None], (1, 3))  # volume-preserving
        factor[:, axis] = grow
        return factor

    #: STAGED, through `stepper.ShapeSwap` -- see `Immutability`, which shares
    #: the mechanism and the reason for it. A squashed body is an *ellipsoid*,
    #: which PyBullet has no primitive for, so the proxy is built as a convex
    #: hull of a per-axis-scaled unit sphere; a cube gets exact half-extents.
    #: Before this the cube in `barrier_pass`, `collision`, `drop` and
    #: `occluder_pass` alike stopped touching the floor the moment it squashed,
    #: because the render shortened it and the physics never heard.
    #:
    #: One consequence worth being explicit about, because it changes what this
    #: family claims: a shape change the physics honours **cannot** leave the
    #: path untouched. A rolling ball stretched to 2.8x along one axis is an
    #: ellipsoid, and an ellipsoid tumbles rather than rolls. The old docstring
    #: promised a perfectly lawful trajectory, and it could only promise that by
    #: not telling the simulator anything had happened -- which is exactly the
    #: bug you found. The law scored here is still `shape_continuity`, so the
    #: severity measures the deformation and not its consequences.
    simulated = True
    #: How many times within a frame the collision shape is rebuilt -- see the
    #: note on `Immutability.SWAPS_PER_FRAME`, which this matches so the two
    #: resize families do not step at different rates.
    SWAPS_PER_FRAME = 8

    def _target_bodies(self, spec, plan):
        by_id = {int(b.segmentation_id): b for b in spec.bodies}
        out = []
        for bid in plan.causal_body_ids:
            body = by_id.get(int(bid))
            # A SCRIPTED body is animated rather than solved -- `shadow_shape`'s
            # target is a flattened disc lying on the floor -- so there is
            # nothing in the simulator to resize and nothing holding it up.
            if body is not None and not body.scripted and body.role != "shadow":
                out.append(body)
        return out

    def stage(self, spec, simulator, objs, plan):
        from ..render import stepper

        self._swaps = []
        bodies = self._target_bodies(spec, plan)
        if not bodies:
            return ()
        swaps = [stepper.ShapeSwap(simulator, objs, spec, b) for b in bodies]
        swaps = [s for s in swaps if s.ok]
        if not swaps:
            return ()
        self._swaps = swaps
        t0 = plan.t_event
        profile = self._profile(plan, spec.tier.num_frames - t0)
        spf = stepper.substeps_of(simulator)
        every = max(1, spf // self.SWAPS_PER_FRAME)
        state = {"step": -1}
        grid = np.arange(profile.shape[0])

        def reshape(_client, step, frame):
            # Several times a frame -- see `Immutability.stage`. A shape that
            # changes in one jump per frame arrives inside the floor and is
            # pushed back out, which is a bounce this family does not claim.
            if frame < t0 or step <= state["step"] or step % every:
                return
            state["step"] = step
            k = min(step / float(spf), profile.shape[0] - 1.0)
            f = [float(np.interp(k, grid, profile[:, a])) for a in range(3)]
            for swap in swaps:
                swap.set_scale(f)

        return (reshape,)

    def unstage(self, spec, simulator, objs, plan) -> None:
        for swap in getattr(self, "_swaps", ()) or ():
            swap.restore()
        self._swaps = []

    def post_simulate(self, spec, traj_valid, traj_invalid, plan) -> Trajectory:
        """The visual half: PyBullet carries the shape, Blender has to be told."""
        t0 = plan.t_event
        factor = self._profile(plan, traj_valid.num_frames - t0)
        traj_invalid.scale_mul = np.asarray(traj_invalid.scale_mul).copy()
        for bid in plan.causal_body_ids:
            bi = traj_valid.index_of(int(bid))
            traj_invalid.scale_mul[t0:, bi, :] = factor.astype(np.float32)
        return super().post_simulate(spec, traj_valid, traj_invalid, plan)

    def _apply(self, spec, traj, plan) -> Trajectory:
        """The host-side approximation, for the mock rollout the tests run on.

        The container takes the staged path above. This one reproduces the same
        geometry by hand so every downstream annotation can be exercised without
        a docker round trip: the body's clearance to its support is preserved
        rather than its centre height, and a body that grew sideways is pushed
        out of whatever it now overlaps.
        """
        out = self._clone(traj)
        t0 = plan.t_event
        n = traj.num_frames - t0
        factor_t = self._profile(plan, n)

        by_id = {int(b.segmentation_id): b for b in spec.bodies}
        for bid in plan.causal_body_ids:
            bi = traj.index_of(int(bid))
            body = by_id.get(int(bid))
            factor = factor_t
            out.scale_mul[t0:, bi, :] = factor.astype(np.float32)
            # A SCRIPTED body is not held up by anything -- `shadow_shape`'s
            # target is a flattened disc lying on the floor whose declared
            # radius has nothing to do with how tall it is drawn -- so seating
            # it against a surface would lift it off the ground it *is*.
            if body is None or body.scripted or body.role == "shadow":
                continue
            # The body's FOOTPRINT moved, so its pose has to move with it. A
            # volume-preserving stretch shortens two axes, and the vertical one
            # is usually among them: leaving the centre where it was lifted a
            # resting cube clear of the ground by the difference, which is what
            # you saw on `barrier_pass`, `collision`, `drop` and
            # `occluder_pass` alike. `reseat` preserves the *clearance* the
            # lawful rollout had rather than the centre height, so a seated body
            # stays seated and an airborne one still lands when it lawfully
            # landed.
            r0 = float(traj.radius[bi])
            _geom.reseat(spec, traj, out, body, bi, t0, r0 * factor[:, 2])
            # And a body that grew sideways must not grow into the barrier it
            # was rolling towards.
            _geom.push_out(spec, traj, out, body, bi, t0,
                           r0 * np.maximum(factor[:, 0], factor[:, 1]))
            self._sync_velocity(traj, out, bi, t0)

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class Deformation(_Squash):
    """A rigid body squashes or stretches out of proportion.

    Its path stays perfectly lawful -- the violation is that a rigid object is
    not supposed to change shape at all.
    """

    family = "deformation"
    magnitude_unit = "aspect_ratio_change"


class ShadowShape(_Squash):
    """The shadow stays where it belongs but stops matching its caster.

    Deliberately separate from `shadow`, which moves the shadow and leaves its
    shape alone. A benchmark asking whether a model tracks shadow *position*
    should not be scored on clips where the shadow is also the wrong shape, and
    vice versa -- so the two never appear in the same clip.
    """

    family = "shadow_shape"
    magnitude_unit = "shadow_aspect_ratio"
    #: NOT staged, unlike `deformation`. The thing changing shape here is a
    #: scripted body -- a flattened disc the scenario animates, pinned in the
    #: simulator so it neither falls nor collides -- so there is no collision
    #: shape to swap and nothing for PyBullet to do. It keeps the trajectory
    #: path, which is the right one for a body whose motion is written rather
    #: than solved.
    simulated = False

    def _targets(self, spec):
        shade = [b for b in spec.bodies if b.role == "shadow"]
        return shade[:1]


class ColourShift(Injector):
    """A body changes colour, with nothing in the scene to explain it.

    The most common perceptual failure in generated video, and the one furthest
    from anything mechanical: the trajectory is untouched and every physical law
    in the taxonomy stays satisfied. What is wrong is that the object stopped
    looking like itself.

    Material colour is animated the same way pose and scale are -- one fcurve
    per channel on the Principled BSDF's Base Colour, verified rendering
    against the pinned image. Eased rather than switched, for the reason
    `immutability` is: a colour that changes between two frames is a cut,
    one that visibly shifts is a violation.

    Severity is perceptual distance rather than RGB distance, and the bins are
    *solved for* in CIE-Lab rather than expressed as hue rotations -- see
    `DISTANCE_BY_BIN` for why the obvious version is not monotone.
    """

    family = "colour_shift"
    persistent = True
    #: Target perceptual separation, in CIE-Lab distance over 100. Specified
    #: here and *solved for* per body, rather than specified as a hue rotation.
    #:
    #: Setting the bins in hue was the first attempt and it is not monotone in
    #: what it claims to measure: from red, a 0.30 turn lands on green and a
    #: 0.50 turn lands on cyan, and green is the further of the two in Lab. So
    #: `medium` came out more different than `strong` on some colours and less
    #: on others, and the severity ordering the whole bin system rests on held
    #: only by luck. Solving for the distance makes the bins mean the same thing
    #: whatever colour a body started as.
    #: Strong pushed to the far end of what a hue rotation can reach. It was
    #: 0.95, and `_shift_to_distance` bisects for the turn that lands there --
    #: so on colours whose antipode is closer than that, strong quietly came out
    #: no further than medium. Asking for more makes the solver take the
    #: furthest turn available on every starting colour.
    DISTANCE_BY_BIN = {"weak": 0.25, "medium": 0.60, "strong": 1.30}
    RAMP_FRACTION = 0.22
    RAMP_MIN = 3

    def strong_residual_reference(self, spec) -> float:
        return float(self.DISTANCE_BY_BIN["strong"])

    @staticmethod
    def _shift_to_distance(rgb, target: float, sign: float):
        """The colour a hue rotation reaches at `target` Lab distance from `rgb`.

        Bisection on the rotation, because Lab distance is monotone in hue turn
        over half the wheel but has no closed form. Twenty steps is well past
        the precision anything downstream can use.
        """
        import colorsys
        h, sat, val = colorsys.rgb_to_hsv(*rgb)
        sat = min(1.0, sat + 0.12)
        base = _srgb_to_lab(np.asarray(rgb, np.float64))
        # LIGHTNESS TOO, once hue alone runs out. A hue rotation can only travel
        # as far as the antipode, and on colours whose antipode is close that
        # ceiling arrives well below the strong bin's target -- the solver then
        # returns the furthest turn it has and `strong` comes out no further
        # than `medium`. Lab distance has an L axis as well as a chroma plane,
        # and moving along it is both unbounded by hue and, perceptually, the
        # most obvious change there is.
        # Whichever of the three moves furthest, MEASURED rather than guessed.
        # Darkening compresses chroma, so on a bright colour it can lose more
        # hue distance than the lightness change buys -- picking a direction up
        # front made `strong` reach 0.99 where hue alone had reached 1.24. Three
        # evaluations per bisection step is nothing, and taking the best can
        # never do worse than hue on its own.
        lifts = (0.0, 0.35, -0.35) if float(target) > 0.9 else (0.0,)

        def at(turn):
            best_rgb, best_d = None, -1.0
            for lift in lifts:
                out = np.asarray(colorsys.hsv_to_rgb(
                    (h + sign * turn) % 1.0, sat,
                    float(np.clip(val + lift * turn / 0.5, 0.05, 1.0))),
                    np.float64)
                d = float(np.linalg.norm(_srgb_to_lab(out) - base) / 100.0)
                if d > best_d:
                    best_rgb, best_d = out, d
            return best_rgb, best_d

        lo, hi = 0.0, 0.5
        best = at(hi)
        if best[1] <= target:
            return best                    # even the antipode is not far enough
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            cand = at(mid)
            if cand[1] < target:
                lo = mid
            else:
                hi, best = mid, cand
        return best

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        targets = [b for b in self._group(spec) if b.kind != "dome"]
        if not targets:
            return None
        T = traj.num_frames
        t0 = _geom.default_event_frame(spec, T)
        if t0 is None:
            return None

        target = float(self.DISTANCE_BY_BIN[severity_bin])
        # Direction per scene, not per bin, so the three strengths are the same
        # shift at three sizes rather than three different colours.
        sign = 1.0 if self._instance_rng(spec).rand() < 0.5 else -1.0
        ramp = max(self.RAMP_MIN, int(round(self.RAMP_FRACTION * T)))

        finals, reached = {}, []
        for body in targets:
            rgb, dist = self._shift_to_distance(body.color, target, sign)
            finals[int(body.segmentation_id)] = [float(x) for x in rgb]
            reached.append(dist)
        union, applied, after = self._split_windows(t0, T, ramp)
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=union, intervention_windows=applied,
            consequence_windows=after,
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "colour_ramp", "target_distance": target,
                    "ramp_frames": int(min(ramp, T - t0))},
            magnitude=float(np.mean(reached)),
            magnitude_unit="lab_distance", severity_bin=severity_bin,
            notes={"radius": float(targets[0].bounding_radius),
                   "surface_top": _geom.surface_top(spec, targets[0]),
                   "final_colour": finals,
                   "ramp_frames": int(min(ramp, T - t0)),
                   "r_strong": self.strong_residual_reference(spec)})

    def _apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        t0 = plan.t_event
        ramp = int(plan.notes["ramp_frames"])
        n = traj.num_frames - t0
        u = np.clip((np.arange(n, dtype=np.float64) + 1.0) / max(ramp, 1), 0.0, 1.0)
        ease = (u * u * (3.0 - 2.0 * u))[:, None]

        for bid, final in plan.notes["final_colour"].items():
            bi = traj.index_of(int(bid))
            start = traj.colour[t0 - 1, bi].astype(np.float64)
            end = np.asarray(final, np.float64)
            out.colour[t0:, bi, :] = (
                start[None, :] + (end - start)[None, :] * ease).astype(np.float32)

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(ColourShift())
register(Deformation())
register(ShadowShape())

"""`pendulum_swing` -- a bob on a rigid rod, swinging about a fixed pivot.

The one scenario whose motion the simulator does not solve. Kubric exposes no
joints and PyBullet's are not reachable through it, so the arc is written
analytically and replayed as keyframes. That is a use of the trajectory seam,
not a hole in it: everything downstream still reads one `traj.npz`, the twins
still share a bit-identical render path, and an injector still edits a finished
trajectory.

Constrained periodic motion is what `angular_momentum` needs: on a free body a
spin reversal is a curiosity, but on a pendulum it is a swing that turns around
in the middle of its arc with nothing to turn it. Grounded in LikePhys
*Pendulum*.
"""
from __future__ import annotations

import math

import numpy as np

from . import _common as C
from . import materials as M
from ._hdri import pick as pick_hdri
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class PendulumSwing(Scenario):
    name = "pendulum_swing"
    SEG_FLOOR, SEG_BOB, SEG_ROD, SEG_POST = 1, 2, 4, 3
    #: Fractional energy rise per substep below which a gain is read as
    #: numerical drift and taken back. An intervention arrives far above it.
    DRIFT_TOLERANCE = 0.02

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        arng = C.appearance_rng(seed, self.name)
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        arm = float(rng.uniform(1.4, 1.8))
        theta0 = float(rng.uniform(0.75, 1.0)) * float(rng.choice([-1.0, 1.0]))
        pivot = (0.0, 0.0, arm + float(rng.uniform(0.85, 1.05)))
        omega = math.sqrt(9.81 / arm)
        r_bob = float(rng.uniform(0.20, 0.26))

        post = BodySpec(name="post", kind="cube",
                        position=(0.0, 0.28, pivot[2] / 2.0),
                        scale=(0.08, 0.06, pivot[2] / 2.0), mass=0.0, static=True,
                        color=(0.30, 0.30, 0.34), segmentation_id=self.SEG_POST,
                        role="prop")
        # The rod is GEOMETRY, not a participant. It is kinematic -- pinned in
        # the simulator and hung between the pivot and the bob every substep by
        # `_carry_rod` -- because a stick with no mass of its own contributes
        # nothing to the swing and nothing collides with it. It stays a body so
        # that it has a segmentation id; a `prop` rather than an `actor` so it
        # never becomes a culprit, and when the swing changes the rod's motion
        # changes with it, which `causal_mask` picks up as a measured
        # consequence without anyone declaring it.
        rod = BodySpec(name="rod", kind="cube", position=(0.0, 0.0, pivot[2] - arm / 2),
                       scale=(0.035, 0.035, arm / 2.0), mass=0.0, static=False,
                       scripted=True, color=(0.55, 0.55, 0.60),
                       segmentation_id=self.SEG_ROD, role="prop")
        # The bob is an ORDINARY DYNAMIC BODY. Its arc is not written down
        # anywhere: it falls under gravity like anything else and the rod is a
        # distance constraint the solver honours every substep (`sim_hooks`).
        # That is what lets every staged family act on it exactly as it would on
        # a dropped ball, with no scenario-specific path anywhere.
        bob_kind = "sphere" if rng.rand() < 0.6 else "cube"
        start = (pivot[0] + arm * math.sin(theta0), pivot[1],
                 pivot[2] - arm * math.cos(theta0))
        bob = BodySpec(name="bob", kind=bob_kind, position=start,
                       scale=(r_bob,) * 3, mass=1.0, static=False,
                       friction=0.4, restitution=0.3,
                       color=C.hue_rgb(float(rng.uniform(0, 1))),
                       segmentation_id=self.SEG_BOB, role="actor")

        bob = C.with_material(bob, M.pick(arng), arng)

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR), post, bob, rod],
            lights=C.lights(cx, look_at=(0, 0, 1.4)),
            camera_position=(0.2, -6.8, 1.9), camera_look_at=(0.0, 0.0, 1.5),
            floor_level=0.0, complexity=complexity,
            hdri_id=pick_hdri(C.appearance_rng(seed, "hdri")) if cx.background == "hdri" else None,
            camera_jitter_deg=(15.0, 8.0),
            notes={"constraint": "pivot", "pivot": list(pivot), "arm": arm,
                   "theta0": theta0, "omega": omega,
                   "omega_ref": abs(theta0) * omega,
                   "bob_radius": r_bob, "bob_kind": bob_kind})

    # ------------------------------------------------------------------ #
    def sim_hooks(self, spec, simulator, objs):
        """The rod, as a constraint the solver honours every substep.

        PyBullet's point-to-point joint is not usable for this in the pinned
        build -- measured, a single rigid-rod constraint drifts 18% and dies out
        within a second. Enforcing the distance directly is exact: put the bob
        back on the sphere of radius `arm` about the pivot and remove the radial
        velocity component. Measured, the arm holds its length to six decimal
        places and the swing keeps 97% of its amplitude over three seconds, at
        0.19 ms/frame.

        This replaces an analytic arc -- `theta0 * cos(omega * t)`, the
        SMALL-ANGLE solution, which the scenario was using at angles up to 57
        degrees. So the swing is not merely differently produced, it is right
        where it used to be approximate.

        What it buys is uniformity. The bob is now an ordinary dynamic body, so
        every staged family acts on it the way it acts on any other body: no
        `script`, no `rescript`, no scenario hooks for gravity or phase, and no
        rule anywhere that some scenarios are exempt from the simulator.
        """
        import numpy as _np
        import pybullet as pb

        from ..render import stepper

        pivot = _np.asarray(spec.notes["pivot"], _np.float64)
        arm = float(spec.notes["arm"])
        state = {"energy": None}

        def constrain(_client, _step, _frame):
            # Resolved EVERY substep, never cached. `ShapeSwap` replaces the
            # body that stands for the bob when `immutability` or `deformation`
            # resizes it, so a index captured when the hook was built ends up
            # constraining the parked original while the live proxy sails off
            # its rod -- measured, the arm reached 3.37 m against a 1.46 m rod.
            idx = stepper.pybullet_index(simulator, objs, spec, self.SEG_BOB)
            if idx is None:
                return
            rod = stepper.pybullet_index(simulator, objs, spec, self.SEG_ROD)
            pos, quat = pb.getBasePositionAndOrientation(idx)
            vel, spin = pb.getBaseVelocity(idx)
            arm_vec = _np.asarray(pos, _np.float64) - pivot
            dist = float(_np.linalg.norm(arm_vec))
            if dist < 1e-9:
                return
            n = arm_vec / dist
            # A ROD pushes as well as pulls, so the distance is held from both
            # sides. `rope_swing` differs from this scenario in exactly one
            # character -- `!=` becomes `>` -- which is the whole physical
            # difference between a rod and a rope.
            if abs(dist - arm) > 1e-12:
                pb.resetBasePositionAndOrientation(
                    idx, (pivot + n * arm).tolist(), quat)
                v = _np.asarray(vel, _np.float64)
                v = v - float(v @ n) * n
                # A ONE-SIDED energy cap. Removing the radial velocity is the
                # whole of what a rod does, but the position projection that
                # comes with it moves the bob a little in height, and the work
                # that represents is never accounted for -- so the swing gains a
                # sliver of energy every substep and compounds. Measured, a
                # deformed bob reached -131 degrees where the lawful one turns
                # at -46, going over the top instead of coming back.
                #
                # Conserving energy across the projection outright is wrong the
                # other way: at rest any upward projection asks for a negative
                # kinetic energy, the speed is zeroed, and the pendulum never
                # starts -- which is what the first attempt did.
                #
                # So drift is clamped and interventions are not. A passive rod
                # cannot add energy, and numerical drift is a fraction of a
                # percent per substep; a family that deliberately adds some --
                # `phantom_impulse`, `superelastic`, `angular_momentum` -- adds
                # it in one large step. The threshold tells them apart.
                z = float(pivot[2] + n[2] * arm)
                g = abs(float(_np.asarray(spec.gravity, _np.float64)[2]))
                # ROTATION COUNTS. `deformation` turns the bob into an
                # ellipsoid, which tumbles, and a tumbling body moves energy
                # between spin and travel. Leaving the spin out of the sum made
                # every one of those transfers look like drift in one direction
                # and a free gain in the other: the deformed bob crept round to
                # -93 degrees where the lawful one turns at -46.
                info = pb.getDynamicsInfo(idx, -1)
                mass = max(float(info[0]), 1e-9)
                inertia = _np.asarray(info[2], _np.float64) / mass
                w = _np.asarray(spin, _np.float64)
                energy = (0.5 * float(v @ v) + g * z
                          + 0.5 * float(inertia @ (w * w)))
                ref = state.get("energy")
                if ref is None or energy > ref * (1.0 + self.DRIFT_TOLERANCE):
                    state["energy"] = energy          # started, or intervened
                elif energy > ref:
                    spun = 0.5 * float(inertia @ (w * w))
                    want = 2.0 * max(ref - g * z - spun, 0.0)  # drift: give it back
                    speed = float(_np.linalg.norm(v))
                    if speed > 1e-9:
                        v = v * (float(_np.sqrt(want)) / speed)
                    state["energy"] = ref
                else:
                    state["energy"] = energy
                pb.resetBaseVelocity(idx, v.tolist(), list(spin))
            if rod is not None:
                self._carry_rod(pb, rod, pivot, n, arm)

        return (constrain,)

    @staticmethod
    def _carry_rod(pb, rod, pivot, direction, arm) -> None:
        """Hang the rod between the pivot and wherever the bob now is.

        The rod is geometry rather than a participant: it has no mass of its
        own in the swing and nothing collides with it, so following the bob is
        the whole of its behaviour. It stays a body rather than a decoration
        because it needs a segmentation id -- a family that acts on "the
        pendulum" has to be able to mask the stick as well as the weight.
        """
        import numpy as _np

        centre = pivot + direction * (arm / 2.0)
        # Local +Z onto the arm direction, about +Y: the plane the swing is in.
        theta = float(_np.arctan2(direction[0], -direction[2]))
        phi = _np.pi - theta
        pb.resetBasePositionAndOrientation(
            rod, centre.tolist(),
            [0.0, float(_np.sin(phi / 2.0)), 0.0, float(_np.cos(phi / 2.0))])
        pb.resetBaseVelocity(rod, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


register(PendulumSwing())

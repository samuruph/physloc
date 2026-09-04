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
from ._hdri import pick as pick_hdri
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class PendulumSwing(Scenario):
    name = "pendulum_swing"
    SEG_FLOOR, SEG_BOB, SEG_ROD, SEG_POST = 1, 2, 4, 3

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
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
        # Both moving parts are scripted: pinned in the simulator so they
        # neither fall nor collide, animated from the trajectory instead. Both
        # carry role "actor" because they are one rigid assembly -- an
        # intervention on the swing moves the rod too, and the mask must say so.
        rod = BodySpec(name="rod", kind="cube", position=(0.0, 0.0, pivot[2] - arm / 2),
                       scale=(0.035, 0.035, arm / 2.0), mass=1.0, static=False,
                       scripted=True, color=(0.55, 0.55, 0.60),
                       segmentation_id=self.SEG_ROD, role="actor")
        # Scripted, not simulated -- the bob's pose comes from `_place` on every
        # frame regardless of shape, so unlike a free body there is no rolling
        # or contact assumption tied to "sphere" to worry about here.
        bob_kind = "sphere" if rng.rand() < 0.6 else "cube"
        bob = BodySpec(name="bob", kind=bob_kind, position=(0.0, 0.0, pivot[2] - arm),
                       scale=(r_bob,) * 3, mass=1.0, static=False, scripted=True,
                       color=C.hue_rgb(float(rng.uniform(0, 1))),
                       segmentation_id=self.SEG_BOB, role="actor")

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            # Bob before rod: `_primary` takes the first actor, and a family
            # that removes or resizes "the pendulum" should act on the weight,
            # not on the stick holding it. Both stay actors because
            # `angular_momentum` re-solves the whole assembly.
            bodies=[C.ground(cx, self.SEG_FLOOR), post, bob, rod],
            lights=C.lights(cx, look_at=(0, 0, 1.4)),
            camera_position=(0.2, -6.8, 1.9), camera_look_at=(0.0, 0.0, 1.5),
            floor_level=0.0, complexity=complexity,
            hdri_id=pick_hdri(C.appearance_rng(seed)) if cx.background == "hdri" else None,
            camera_jitter_deg=(15.0, 8.0),
            notes={"constraint": "pivot", "pivot": list(pivot), "arm": arm,
                   "theta0": theta0, "omega": omega,
                   "omega_ref": abs(theta0) * omega,
                   "bob_radius": r_bob, "bob_kind": bob_kind})

    # ------------------------------------------------------------------ #
    def script(self, spec, traj) -> None:
        n = traj.num_frames
        t = np.arange(n, dtype=np.float64) * traj.dt
        th0, om = float(spec.notes["theta0"]), float(spec.notes["omega"])
        self._place(spec, traj, th0 * np.cos(om * t), -th0 * om * np.sin(om * t))

    def rescript(self, spec, traj, t0: int, omega_scale: float) -> bool:
        """Continue the swing from frame `t0` with the angular rate rescaled.

        Simple harmonic motion is fully determined by (angle, rate) at one
        instant, so this is exact rather than a re-simulation: keep the angle,
        scale the rate, evolve. `omega_scale = -1` is angular momentum reversed;
        `|omega_scale| > 1` is a swing that gains energy from nowhere.
        """
        if t0 < 1 or t0 >= traj.num_frames:
            return False
        n = traj.num_frames
        om = float(spec.notes["omega"])
        th0, dth0 = float(spec.notes["theta0"]), 0.0
        t = np.arange(n, dtype=np.float64) * traj.dt
        theta = th0 * np.cos(om * t)
        rate = -th0 * om * np.sin(om * t)

        d = (np.arange(n, dtype=np.float64) - t0) * traj.dt
        a, b = theta[t0], rate[t0] * float(omega_scale)
        theta_new = a * np.cos(om * d) + (b / om) * np.sin(om * d)
        rate_new = -a * om * np.sin(om * d) + b * np.cos(om * d)
        theta[t0:] = theta_new[t0:]
        rate[t0:] = rate_new[t0:]
        self._place(spec, traj, theta, rate)
        return True

    def regravity(self, spec, traj, t0: int, alpha: float) -> bool:
        """Continue the swing from `t0` under gravity scaled by `alpha`.

        The period is `2*pi*sqrt(L/g)`, so scaling gravity scales the angular
        frequency by `sqrt(alpha)` -- a slower swing under weaker gravity, which
        is the whole visible content of the violation on a pendulum and is not
        something a rate change can imitate.

        `alpha <= 0` is gravity reversed, and then the pendulum does not
        oscillate at all: the restoring torque becomes a driving one and the
        solution turns hyperbolic, so the bob climbs away from the bottom and
        keeps going. That is the same equation, not a special case -- only the
        sign under the square root changes -- and it is clamped at the
        horizontal so the arm never winds past a right angle.
        """
        if t0 < 1 or t0 >= traj.num_frames:
            return False
        n = traj.num_frames
        om = float(spec.notes["omega"])
        th0 = float(spec.notes["theta0"])
        t = np.arange(n, dtype=np.float64) * traj.dt
        theta = th0 * np.cos(om * t)
        rate = -th0 * om * np.sin(om * t)

        d = (np.arange(n, dtype=np.float64) - t0) * traj.dt
        a, b = theta[t0], rate[t0]
        lam = om * np.sqrt(abs(float(alpha)))
        if float(alpha) > 0.0:
            new_theta = a * np.cos(lam * d) + (b / max(lam, 1e-9)) * np.sin(lam * d)
            new_rate = -a * lam * np.sin(lam * d) + b * np.cos(lam * d)
        else:
            new_theta = a * np.cosh(lam * d) + (b / max(lam, 1e-9)) * np.sinh(lam * d)
            new_rate = a * lam * np.sinh(lam * d) + b * np.cosh(lam * d)
            # A rod, not a string, so it can carry the bob over the top -- and
            # straight up is exactly where a pendulum under reversed gravity
            # belongs: that is its stable point. The arm swings in the x-z plane
            # and the post stands off it in y, so nothing is in the way.
            #
            # Clamped at the inverted position rather than at the horizontal.
            # Stopping it halfway up left the bob hanging sideways and
            # motionless for half the clip, which is the frozen picture this
            # whole change is meant to remove; letting it travel the full arc
            # keeps it moving almost to the end and finishes somewhere no
            # pendulum can be.
            over = np.abs(new_theta) > math.pi
            new_theta = np.clip(new_theta, -math.pi, math.pi)
            new_rate = np.where(over, 0.0, new_rate)
        theta[t0:] = new_theta[t0:]
        rate[t0:] = new_rate[t0:]
        self._place(spec, traj, theta, rate)
        return True

    def rephase(self, spec, traj, t0: int, shift_seconds: float) -> bool:
        """Continue the swing from `t0` as if `shift_seconds` further along.

        The assembly jumps to a different point of the same arc and carries on
        lawfully from there -- a position discontinuity that leaves the rod and
        bob attached, which displacing the bob in space does not.
        """
        if t0 < 1 or t0 >= traj.num_frames:
            return False
        n = traj.num_frames
        om = float(spec.notes["omega"])
        th0 = float(spec.notes["theta0"])
        t = np.arange(n, dtype=np.float64) * traj.dt
        theta = th0 * np.cos(om * t)
        rate = -th0 * om * np.sin(om * t)
        shifted = t + float(shift_seconds)
        theta[t0:] = (th0 * np.cos(om * shifted))[t0:]
        rate[t0:] = (-th0 * om * np.sin(om * shifted))[t0:]
        self._place(spec, traj, theta, rate)
        return True

    # ------------------------------------------------------------------ #
    def _place(self, spec, traj, theta: np.ndarray, rate: np.ndarray) -> None:
        """Write the assembly's pose for a whole angle series onto a trajectory."""
        pivot = np.asarray(spec.notes["pivot"], np.float64)
        arm = float(spec.notes["arm"])
        s, c = np.sin(theta), np.cos(theta)
        # Angle is measured from straight down toward +X, so the arm direction
        # is (sin, 0, -cos) and the rod's local +Z maps onto it under a rotation
        # of (pi - theta) about +Y.
        direction = np.stack([s, np.zeros_like(s), -c], axis=1)
        tangent = np.stack([c, np.zeros_like(c), s], axis=1)
        phi = math.pi - theta
        quat = np.stack([np.cos(phi / 2.0), np.zeros_like(phi),
                         np.sin(phi / 2.0), np.zeros_like(phi)], axis=1)
        ang = np.stack([np.zeros_like(rate), -rate, np.zeros_like(rate)], axis=1)

        for name, dist in (("rod", arm / 2.0), ("bob", arm)):
            j = spec.index_of(name)
            traj.pos[:, j, :] = (pivot[None, :] + dist * direction).astype(np.float32)
            traj.lin_vel[:, j, :] = (dist * rate[:, None] * tangent).astype(np.float32)
            traj.ang_vel[:, j, :] = ang.astype(np.float32)
            if name == "rod":
                traj.quat[:, j, :] = quat.astype(np.float32)


register(PendulumSwing())

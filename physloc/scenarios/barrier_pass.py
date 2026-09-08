"""`barrier_pass` -- a ball rolls into a solid wall and bounces back.

Exists so `solidity` has a scenario where passing through is the whole point.
It was previously staged on `occluder_pass`, where the only surface available
was the floor -- so the violation sank the ball into the ground *while it was
hidden behind the screen*, and the clip read as the ball vanishing. That is
`permanence`'s job, and it is on `occluder_pass` too, so the two families were
producing the same picture under different labels.

Splitting them rather than reshaping `occluder_pass` keeps the thing that makes
that scenario worth having: its screen is the only source of observability lag
in the dataset, and a screen the ball can hit is a screen the ball never gets
behind.

The valid clip is a lawful rebound, which matters more here than usual. "A ball
passes through a wall" is only legible as a violation if the same ball, in the
same scene, is shown bouncing off it.
"""
from __future__ import annotations

from .. import camera as cam
from . import _common as C
from . import materials as M
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


CAMERA = (0.1, -6.4, 1.7)
LOOK_AT = (0.25, 0.0, 0.55)


class BarrierPass(Scenario):
    name = "barrier_pass"
    SEG_FLOOR, SEG_BALL, SEG_WALL, SEG_SPLIT = 1, 2, 3, 4

    #: The ball's own friction, named because the approach is solved through it.
    BALL_FRICTION = 0.05

    #: How fast the ball must still be going when it reaches the wall. Above
    #: `_geom.first_impact`'s 0.3 m/s floor with margin to spare, and fast
    #: enough that the lawful rebound is unmistakable -- which is the whole
    #: premise of the scenario.
    MIN_ARRIVAL_SPEED = 0.75

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        # ONE appearance stream for the whole sample. `appearance_rng` builds a
        # fresh RandomState per call, so asking it once per draw handed every
        # draw the same first number -- material, colour and proportions came
        # out identical across every scenario on a given seed. Threading one
        # stream lets the draws advance.
        arng = C.appearance_rng(seed, self.name)
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        radius = float(rng.uniform(0.24, 0.32))
        thickness = float(rng.uniform(0.07, 0.11))
        wall_x = float(rng.uniform(1.05, 1.35))
        wall_h = float(rng.uniform(0.62, 0.85))
        flight = tier.num_frames / float(tier.fps)

        # Solve the approach for the ARRIVAL, not the launch.
        #
        # It used to pick a launch speed from the frame width and set the gap
        # to `speed * 0.45 * flight` -- constant-velocity arithmetic on a ball
        # that is being slowed by friction the whole way. The error grows with
        # the clip: at the debug tier's 25 frames the ball reached the wall at
        # 0.93 m/s, at the release tier's 89 it arrived at 0.26 and hit on frame 61 of
        # 89 rather than "just under halfway". Below 0.3 m/s `_geom.first_impact`
        # stops calling it an impact at all, so `superelastic x barrier_pass`
        # planned nothing on any seed at v0 -- a cell the matrix claims and the
        # release never contained.
        #
        # So: choose WHEN it should hit and HOW FAST it should still be going,
        # then integrate backwards through the deceleration to get the launch.
        # Both are per-seed draws, which is also what stops every clip of this
        # scenario impacting on the same frame.
        arrive_frac = float(rng.uniform(0.38, 0.58))
        t_travel = arrive_frac * flight
        # Rolling friction bleeds the approach; a = mu*g is the standard
        # first-order model and is what `_integrate_profile` already assumes.
        decel = self.BALL_FRICTION * cam.GRAVITY
        v_arrive = float(rng.uniform(self.MIN_ARRIVAL_SPEED,
                                     self.MIN_ARRIVAL_SPEED * 1.6))
        speed = v_arrive + decel * t_travel
        gap = speed * t_travel - 0.5 * decel * t_travel ** 2
        # Keep the launch inside the shot: the ball has to be visible rolling
        # in, or the approach the rebound is judged against is never seen.
        reach = cam.frame_extent(CAMERA, LOOK_AT) * 0.92
        gap = min(gap, max(0.35, wall_x - thickness - radius + reach))
        x0 = wall_x - thickness - radius - gap

        # Rolling without slipping only reads correctly on a sphere -- see
        # `collision`, which shares this exact reasoning. A cube slides instead
        # (`ramp_slide`'s regime), no spin.
        kind = "sphere" if rng.rand() < 0.6 else "cube"
        spin = (0.0, speed / radius, 0.0) if kind == "sphere" else (0.0, 0.0, 0.0)
        ball = BodySpec(
            name="ball", kind=kind, position=(x0, 0.0, radius),
            scale=(radius,) * 3, velocity=(speed, 0.0, 0.0),
            # Elastic enough that the lawful rebound is unmistakable -- a ball
            # that hits a wall and stops dead makes a pass-through look like
            # the more sensible of the two.
            angular_velocity=spin,
            mass=1.0, friction=BarrierPass.BALL_FRICTION, restitution=0.78,
            color=C.hue_rgb(float(rng.uniform(0, 1))),
            segmentation_id=self.SEG_BALL, role="actor")
        ball = C.with_material(ball, M.pick(arng), arng)
        ball = C.vary_dims(ball, arng)
        wall = BodySpec(
            name="wall", kind="cube",
            position=(wall_x, 0.0, wall_h), scale=(thickness, 1.15, wall_h),
            mass=0.0, static=True, friction=0.4, restitution=0.75,
            color=(0.30, 0.31, 0.37), segmentation_id=self.SEG_WALL,
            role="occluder")

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR), wall, ball,
                    C.understudy(ball, self.SEG_SPLIT)],
            lights=C.lights(cx, look_at=(0, 0, 0.5)),
            camera_position=CAMERA, camera_look_at=LOOK_AT,
            floor_level=0.0, complexity=complexity,
            notes={"radius": radius, "speed": speed, "wall_x": wall_x,
                   "wall_id": self.SEG_WALL, "actor_kind": kind,
                   "family_targets": {"solidity": [self.SEG_BALL]}})


register(BarrierPass())

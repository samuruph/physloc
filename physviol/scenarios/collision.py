"""`collision` -- a rolling sphere strikes an identical one at rest.

The only scenario in v0 with **two bodies that both ought to respond**, which is
what `newton3_reaction` and `newton2_mass` need: a violation where only one body
reacts is invisible unless a lawful reaction is the obvious alternative.
Grounded in LikePhys *Ball Collision*.
"""
from __future__ import annotations

from .. import camera as cam
from . import _common as C
from ._hdri import pick as pick_hdri
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


#: Hand-composed, and now also the reference the ball speed is derived from.
CAMERA = (0.3, -6.0, 1.7)
LOOK_AT = (0.1, 0.0, 0.35)


class Collision(Scenario):
    name = "collision"
    SEG_FLOOR, SEG_A, SEG_B, SEG_SPLIT = 1, 2, 4, 6

    #: Both balls' friction, named because the approach is solved through it.
    #: Low enough that they roll rather than scrub.
    BALL_FRICTION = 0.05

    #: How fast the striker must still be going when it reaches the target.
    #: Comfortably above `_geom.first_impact`'s 0.3 m/s floor, and fast enough
    #: that the struck ball's response is legible -- which is what both Newton
    #: families are built to test.
    MIN_MEET_SPEED = 0.85

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        # The two balls are deliberately IDENTICAL -- same radius, same colour.
        # `newton2_mass` stages a collision whose outcome would be lawful for
        # masses in some ratio k:1, and the only thing that makes it a violation
        # is that nothing in the image justifies that ratio. Give the balls
        # different sizes and "the big one is heavier" becomes a perfectly good
        # reading, and the family stops testing anything.
        radius = float(rng.uniform(0.28, 0.36))
        r_a = r_b = radius
        hue = float(rng.uniform(0, 1))
        # One shared draw, not one each -- see the IDENTICAL comment above:
        # a striker and target that differ in shape are exactly the same
        # confound as differing in size, just spelled a different way.
        kind = "sphere" if rng.rand() < 0.6 else "cube"
        flight = tier.num_frames / float(tier.fps)

        # A striker into a body at REST, not two balls closing head-on. Both
        # Newton families hinge on how the struck ball responds, and a target
        # that is already moving makes that unreadable twice over: "it did not
        # react" is indistinguishable from "it stopped dead", which is a
        # different violation, and a target that keeps coming at the striker
        # has to end up sharing space with it. Against a resting target,
        # `newton3` is simply "it never moved" -- no overlap, nothing to
        # misread.
        target_x = float(rng.uniform(0.15, 0.45))
        # Solve the approach for the MEETING, not the launch -- the same fix
        # `barrier_pass` needed, for the same reason. `gap = speed * 0.45 *
        # flight` is constant-velocity arithmetic on a ball that friction is
        # slowing, so the striker arrived later and slower the longer the clip
        # got, and every family that fires on the impact -- friction, fusion,
        # newton1, newton2, solidity, superelastic -- fired on the same frame
        # in every clip, because the arrival was pinned to whatever that
        # arithmetic produced.
        #
        # Choosing WHEN they should meet and HOW FAST the striker should still
        # be going fixes the drift and varies the moment, both at once.
        meet_frac = float(rng.uniform(0.34, 0.56))
        t_travel = meet_frac * flight
        decel = self.BALL_FRICTION * cam.GRAVITY
        v_meet = float(rng.uniform(self.MIN_MEET_SPEED, self.MIN_MEET_SPEED * 1.7))
        speed = v_meet + decel * t_travel
        gap = speed * t_travel - 0.5 * decel * t_travel ** 2
        # The struck ball has to stay in shot after it is hit, so the pair
        # cannot start further out than the frame allows.
        gap = min(gap, cam.frame_extent(CAMERA, LOOK_AT) * 1.15)
        striker_x = target_x - r_a - r_b - gap

        # Rolling without slipping only reads correctly on a sphere -- a cube
        # given the same omega_y = vx/r would spin at a rate tuned for a round
        # silhouette and visibly skid. `ramp_slide`'s cube slides with no spin
        # at all; do the same here rather than fake a rolling cube.
        striker_spin = (0.0, speed / r_a, 0.0) if kind == "sphere" else (0.0, 0.0, 0.0)
        striker = BodySpec(
            name="ball_a", kind=kind,
            position=(striker_x, 0.0, r_a), scale=(r_a,) * 3,
            velocity=(speed, 0.0, 0.0),
            angular_velocity=striker_spin,
            mass=1.0, friction=Collision.BALL_FRICTION, restitution=0.75,
            color=C.hue_rgb(hue), segmentation_id=self.SEG_A, role="actor")
        target = BodySpec(
            name="ball_b", kind=kind,
            position=(target_x, 0.0, r_b), scale=(r_b,) * 3,
            mass=1.0, friction=Collision.BALL_FRICTION, restitution=0.75,
            color=C.hue_rgb(hue), segmentation_id=self.SEG_B, role="actor")

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR), striker, target,
                    C.understudy(striker, self.SEG_SPLIT)],
            lights=C.lights(cx, look_at=(0, 0, 0.4)),
            camera_position=CAMERA, camera_look_at=LOOK_AT,
            floor_level=0.0, complexity=complexity,
            hdri_id=pick_hdri(C.appearance_rng(seed)) if cx.background == "hdri" else None,
            notes={"radius_a": r_a, "radius_b": r_b, "speed": speed,
                   "identical_actors": True, "target_at_rest": True,
                   "striker_id": self.SEG_A, "target_id": self.SEG_B,
                   "actor_kind": kind})


register(Collision())

"""`collision` -- a rolling sphere strikes an identical one at rest.

The only scenario in v0 with **two bodies that both ought to respond**, which is
what `newton3_reaction` and `newton2_mass` need: a violation where only one body
reacts is invisible unless a lawful reaction is the obvious alternative.
Grounded in LikePhys *Ball Collision*.
"""
from __future__ import annotations

from .. import camera as cam
from . import _common as C
from . import materials as M
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

    #: Room in the frame for the struck ball leaving at this multiple of the
    #: meeting speed -- a super-elastic hit at gain 2.5 -- and for the striker
    #: rolling back at `BACK_SHARE` of that. See the framing in `_sample`.
    HEADROOM = 1.75
    BACK_SHARE = 0.45
    #: The share of the clip after the meeting the bodies are framed for: the
    #: visible span a violation needs, with a margin.
    VISIBLE_SHARE = 0.45
    #: Share of the frame's width the framed box may fill.
    FILL = 0.85
    #: How much of the approach is in shot: the striker is framed from this
    #: far into its roll towards the target.
    APPROACH_SHOWN = 0.4

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        # NEVER DRAWN TOO SMALL TO SEE. The shot is framed on the action, and
        # where that is wide the object shrinks with it; this enlarges it,
        # before anything is simulated, until it is drawn at least
        # `C.MIN_SCREEN_SHARE` of the frame wide -- see `at_least_screen_share`.
        return C.at_least_screen_share(
            lambda boost: self._build(seed, tier, complexity, boost),
            self.SEG_A, C.size_scale(seed, self.name))

    def _build(self, seed: int, tier: Tier, complexity: str,
               boost: float = 1.0) -> SceneSpec:
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

        # The two balls are deliberately IDENTICAL -- same radius, same colour.
        # `newton2_mass` stages a collision whose outcome would be lawful for
        # masses in some ratio k:1, and the only thing that makes it a violation
        # is that nothing in the image justifies that ratio. Give the balls
        # different sizes and "the big one is heavier" becomes a perfectly good
        # reading, and the family stops testing anything.
        radius = float(rng.uniform(0.28, 0.36)) * C.size_scale(seed, self.name) * boost
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
            name=C.shape_name(kind, "a"), kind=kind,
            position=(striker_x, 0.0, r_a), scale=(r_a,) * 3,
            velocity=(speed, 0.0, 0.0),
            angular_velocity=striker_spin,
            mass=1.0, friction=Collision.BALL_FRICTION, restitution=0.75,
            color=C.hue_rgb(hue), segmentation_id=self.SEG_A, role="actor")
        target = BodySpec(
            name=C.shape_name(kind, "b"), kind=kind,
            position=(target_x, 0.0, r_b), scale=(r_b,) * 3,
            mass=1.0, friction=Collision.BALL_FRICTION, restitution=0.75,
            color=C.hue_rgb(hue), segmentation_id=self.SEG_B, role="actor")

        # ONE material for both, and one dimension draw, for the same reason
        # they already share a radius, a colour and a shape: `newton2_mass`
        # claims a mass ratio the image cannot justify, and the moment the two
        # balls differ visibly -- "the steel one is heavier" -- that claim
        # becomes a lawful reading and the family stops testing anything.
        # `tests/test_injectors.py` pins the pair's sameness.
        mat = M.pick(arng)
        look = M.appearance(mat, arng)
        striker = C.with_material(striker, mat, arng, look=look)
        target = C.with_material(target, mat, arng, look=look)

        # FRAMED ON WHERE THE STRUCK BALL GOES, not a fixed eye. The fixed
        # camera lost the struck ball even in the LAWFUL clip -- it rolled out
        # at frame 27 of 37 -- so any family that sends it off faster had no
        # room at all: `superelastic` was fitted down to 1-7% of its gain and
        # its three bins came out as one clip. Frame the striker's start, the
        # struck ball's travel at up to `HEADROOM` x the meeting speed (a
        # strong super-elastic hit), and the striker rolling back, over the
        # stretch a violation must stay visible for.
        v_fast = self.HEADROOM * v_meet
        t_vis = min(flight - t_travel, self.VISIBLE_SHARE * flight)

        def roll(v, t):
            t_stop = v / max(decel, 1e-9)
            t = min(t, t_stop)
            return v * t - 0.5 * decel * t * t

        # The striker from partway into its approach, not its start: the old
        # camera never showed the start either -- the striker rolled in from
        # off shot -- and framing it doubled the width for nothing.
        x_in = striker_x + roll(speed, self.APPROACH_SHOWN * t_travel)
        x_lo = min(x_in, target_x - r_b - r_a
                   - roll(self.BACK_SHARE * v_fast, t_vis)) - r_a
        x_hi = target_x + roll(v_fast, t_vis) + r_b
        # The hand-composed VIEWPOINT is kept -- its elevation and direction --
        # and only moved: centred on the action and pulled back as far as the
        # run-out needs, never closer than it was. `frame_box` would have
        # dropped the eye to table height, a different shot.
        xc = 0.5 * (x_lo + x_hi)
        width = (x_hi - x_lo) / self.FILL
        # `frame_extent` is the HALF-width.
        pull = max(1.0, width / (2.0 * cam.frame_extent(CAMERA, LOOK_AT)))
        camera_look_at = (xc, LOOK_AT[1], LOOK_AT[2])
        camera_position = tuple(camera_look_at[i] + pull * (CAMERA[i] - LOOK_AT[i])
                                for i in range(3))

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR), striker, target,
                    C.understudy(striker, self.SEG_SPLIT)],
            lights=C.lights(cx, look_at=(0, 0, 0.4)),
            camera_position=camera_position, camera_look_at=camera_look_at,
            floor_level=0.0, complexity=complexity,
            notes={"radius_a": r_a, "radius_b": r_b, "speed": speed,
                   "identical_actors": True, "target_at_rest": True,
                   "striker_id": self.SEG_A, "target_id": self.SEG_B,
                   "actor_kind": kind,
                   # Non-contact interventions happen just before the meeting,
                   # leaving time for their causal effect on the second ball.
                   "event_anchor": {"body_ids": [self.SEG_A],
                                    "partner_ids": [self.SEG_B],
                                    "offset": -2, "jitter": 1,
                                    # Keep causal interventions before the
                                    # meeting, but distribute a minority over
                                    # the striker's approach.
                                    "broad_offsets": [-10, 8],
                                    "broad_seconds": [-0.42, 0.28],
                                    "jitter_seconds": 0.10,
                                    "near_probability": 0.6}})


register(Collision())

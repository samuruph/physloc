"""`rolling_ramp` -- a cube tumbles down a raised ramp and off its lip.

Two regimes in one clip, deliberately: sustained contact on the slab, then a
short free flight after the lip. `friction` and `newton1_inertia` fire in the
first; `angular_momentum` needs the second, because a body in contact with the
ground can change its spin lawfully and only a change made in mid-air is
unexplained.

A cube rather than a ball for the same reason `toss` only spins its cube
draws: a uniformly coloured sphere renders identically however it is spinning.
"""
from __future__ import annotations

import math

from .. import camera as cam
from . import _common as C
from . import materials as M
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class RollingRamp(Scenario):
    name = "rolling_ramp"
    SEG_FLOOR, SEG_BLOCK, SEG_RAMP, SEG_SPLIT = 1, 2, 3, 4

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

        # A LONGER RAMP, and a gentler one. The block starts near the top now
        # (see `start_along`), so the tilt comes down to keep the speed at the
        # lip -- and with it the landing distance the camera has to take in --
        # close to what it was.
        tilt = float(rng.uniform(0.30, 0.38))
        half_len, thick = 1.75, 0.08
        sin_t, cos_t = math.sin(tilt), math.cos(tilt)
        # The LIP height is the number that matters, so set it and derive the
        # slab centre -- not the other way round. It sets how long the block is
        # airborne after the lip, which is the only stretch `angular_momentum`
        # can act in: its law excludes a frame either side of every contact.
        #
        # 1.15 m was enough while the block started a short way above the lip.
        # Sliding the full longer ramp it leaves at ~3.9 m/s, already falling at
        # ~2 m/s, and at 1.15 m it hit the floor within three frames at the
        # debug tier -- every `rolling_ramp` x `angular_momentum` job declined,
        # measured on seeds 777-781. A higher lip and the gentler tilt above
        # (less of that speed pointing down) give the fall ~5-6 frames.
        #
        # DRAWN, and higher on average. It was a fixed 1.70 m, so every scene
        # had the same ramp; a longer fall also gives a violation more of the
        # clip to act in and be seen in. The floor of the range stays above the
        # old value, so the airborne stretch `angular_momentum` needs only grows.
        #
        # ITS OWN STREAM. Drawn from `rng` it shifted every later draw -- block
        # size, launch speed, friction, start -- so raising the ramp silently
        # changed every other parameter of every seed too.
        # Raised again, from 1.80-2.30 m, for a longer flight: measured, the
        # block was airborne for 5 frames (0.42 s). At 2.7-3.3 m it flies for
        # about 0.6-0.7 s.
        lip_z = float(C.appearance_rng(seed, self.name + "/lip").uniform(2.70, 3.30))
        centre = (0.0, 0.0, lip_z + half_len * sin_t)
        half = float(rng.uniform(0.17, 0.22)) * C.size_scale(seed, self.name)
        v0 = float(rng.uniform(0.3, 0.8))
        mu = float(rng.uniform(0.25, 0.38))
        d, _ = C.ramp_axes(tilt)
        # NEAR THE TOP, so the slide is a visible stretch of the clip rather
        # than a few frames before the lip. It used to start 0.55-0.78 m above
        # the lip at a launch speed of ~2 m/s, which had the block off the slab
        # within the opening frames: every contact-phase violation (`friction`,
        # `newton1_inertia`) had almost nothing to act on, and what it did act
        # on was over before a viewer had seen what lawful sliding looks like.
        # The longer ramp is what keeps the airborne stretch after the lip that
        # `angular_momentum` needs within reach of the debug tier.
        start_along = -half_len + half + float(rng.uniform(0.10, 0.45))

        # ---- where the block ends up, so the camera can be pointed at it ----
        # A block that coasts out of shot is a violation injected off-screen.
        # Rather than tune an eye position against one tier, work the run out:
        # slide to the lip, fall, land, and decelerate to a stop. The slide is
        # slowed by the slab's friction, which PyBullet takes as the product of
        # the pair -- `mu**2` by construction below.
        a_slide = 9.81 * max(sin_t - mu * mu * cos_t, 0.05)
        v_lip = math.sqrt(v0 ** 2 + 2.0 * a_slide * (half_len - start_along))
        vx_lip, vz_lip = v_lip * cos_t, -v_lip * sin_t
        t_fall = (-vz_lip + math.sqrt(vz_lip ** 2 + 2.0 * 9.81 * lip_z)) / 9.81
        x_lip = half_len * cos_t
        x_land = x_lip + vx_lip * t_fall
        run_out = 0.95

        # Friction is TWO coefficients, not one. PyBullet multiplies the pair,
        # so a single `mu` on both block and slab gave the ramp `mu**2` (~0.09,
        # nicely slippery) and the floor `mu*0.6` (~0.18) -- far too little to
        # stop a block that lands at 3 m/s, which is why it kept going straight
        # out of frame. Solve the block's coefficient for the run-out budget,
        # then pick the slab's to leave the ramp exactly as slippery as before.
        floor_mu = 0.6
        # CAPPED AT 1.0, Kubric's own limit: `PhysicalObject` raises a
        # TraitError for any friction above it, at scene build, in the
        # container. The cap was 1.2 and nothing reached it until the longer
        # ramp raised the speed at the lip -- then every `rolling_ramp` job of
        # a debug sweep died before simulating. The run-out below grows to
        # match whatever the cap costs.
        block_mu = min(1.0, max(0.20, vx_lip ** 2
                                / (2.0 * 9.81 * run_out * floor_mu)))
        # Where the cap binds, the block needs more floor than was budgeted,
        # and the camera must be framed for the run-out it will really have.
        run_out = max(run_out, vx_lip ** 2 / (2.0 * 9.81 * block_mu * floor_mu))
        ramp_mu = min(1.0, max(0.02, mu * mu / block_mu))

        block = BodySpec(
            name="block", kind="cube",
            position=C.on_ramp(centre, tilt, start_along, half + thick),
            scale=(half,) * 3,
            quaternion=(math.cos(tilt / 2.0), 0.0, math.sin(tilt / 2.0), 0.0),
            velocity=tuple(v0 * x for x in d),
            mass=1.0, friction=block_mu, restitution=0.2,
            color=C.hue_rgb(float(rng.uniform(0, 1))),
            segmentation_id=self.SEG_BLOCK, role="actor")

        lip = tuple(centre[i] + half_len * d[i] for i in range(3))
        camera_position, camera_look_at = cam.frame_box(
            x_range=(-x_lip - half, x_land + run_out + half),
            z_range=(0.0, centre[2] + half_len * sin_t + thick + 2 * half))
        block = C.with_material(block, M.pick(arng), arng)

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR),
                    C.ramp(self.SEG_RAMP, tilt, centre, half_len, 0.8, thick,
                           ramp_mu),
                    block, C.understudy(block, self.SEG_SPLIT)],
            lights=C.lights(cx, look_at=(0, 0, 1.0)),
            camera_position=camera_position, camera_look_at=camera_look_at,
            floor_level=0.0, complexity=complexity,
            camera_jitter_deg=(15.0, 8.0),
            notes={"tilt_rad": tilt, "mu": mu, "lip": list(lip),
                   "block_friction": block_mu, "ramp_friction": ramp_mu,
                   "x_land": x_land, "run_out": run_out,
                   "half_len": half_len, "start_along": start_along,
                   "ramp_id": self.SEG_RAMP})


register(RollingRamp())

"""`ramp_slide` -- a block slides down an incline.

Sustained contact with friction for the whole clip, which is what separates it
from every ballistic scenario: `friction` and `newton1_inertia` need a body
whose motion is being *continuously* mediated by a surface, so that "it stopped
on its own" is a statement about a force that is not there. Grounded in LikePhys
*Block Slide*.
"""
from __future__ import annotations

import math

from .. import camera as cam
from . import _common as C
from . import materials as M
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class RampSlide(Scenario):
    name = "ramp_slide"
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

        tilt = float(rng.uniform(0.40, 0.52))
        mu = float(rng.uniform(0.16, 0.26))
        # tan(tilt) > mu is the slide condition. Sampling them independently
        # would silently produce clips where the block never moves and every
        # violation fires against a stationary body.
        assert math.tan(tilt) > mu + 0.1, "ramp is too shallow to slide"

        half_len, thick = 1.35, 0.07
        # THE LOW END IS DRAWN, and set clear of the floor. A fixed centre at
        # 0.95 m put the lip 0.28-0.42 m up across the tilt range -- the block
        # reached the bottom almost at floor level, so the slide was short and
        # a violation fired on it had little run left to show in. Drawing the
        # lip rather than the centre also varies the scene, which a fixed ramp
        # did not. The camera is framed on whatever results, below.
        # Its own stream, so the draw leaves every later physics value of the
        # seed where it was (see `rolling_ramp`).
        # Raised again, from 0.55-0.95 m, for a longer flight off the end:
        # measured, the block was airborne for 1-2 frames (0.08-0.17 s) before
        # it landed, and at 1.2-1.7 m still only 3 -- it keeps touching the
        # lip as it tips off. 1.8-2.3 m.
        lip_z = float(C.appearance_rng(seed, self.name + "/lip").uniform(1.80, 2.30))
        centre = (0.0, 0.0, lip_z + half_len * math.sin(tilt))
        half = float(rng.uniform(0.18, 0.24)) * C.size_scale(seed, self.name)
        v0 = float(rng.uniform(0.3, 0.7))
        d, _ = C.ramp_axes(tilt)
        start_along = -0.85
        sin_t, cos_t = math.sin(tilt), math.cos(tilt)

        # ---- where the block stops, so it stops ON SCREEN ----
        # The camera used to be fixed on the ramp, and the block reached the
        # floor at 3-4 m/s against a floor friction PyBullet takes as the
        # PRODUCT of the pair (~0.1-0.16): it needed several metres and about
        # three seconds to stop. At 25 frames that was past the end of the clip;
        # at 37 the block left the shot at frame 15 and slid off the edge of the
        # ground, and eight families on this scenario were declined in the
        # full-length review. What the block does after the ramp must not
        # depend on how long the clip is, so -- as `rolling_ramp` does -- the
        # run-out is solved for and the camera framed on it.
        #
        # The slide itself is unchanged: the ramp contact stays at `mu**2`,
        # which is what it was with `mu` on both the block and the slab.
        a_slide = 9.81 * max(sin_t - mu * mu * cos_t, 0.05)
        v_lip = math.sqrt(v0 ** 2 + 2.0 * a_slide * (half_len - start_along))
        vx_lip, vz_lip = v_lip * cos_t, -v_lip * sin_t
        t_fall = (-vz_lip + math.sqrt(vz_lip ** 2 + 2.0 * 9.81 * max(lip_z, 0.0))) / 9.81
        x_lip = half_len * cos_t
        x_land = x_lip + vx_lip * t_fall
        run_out = 0.9
        floor_mu = 0.6
        # Capped at Kubric's limit of 1.0 -- see `rolling_ramp` -- and the
        # run-out grows to whatever the cap costs, so the frame still holds it.
        block_mu = min(1.0, max(mu, vx_lip ** 2 / (2.0 * 9.81 * run_out * floor_mu)))
        run_out = max(run_out, vx_lip ** 2 / (2.0 * 9.81 * block_mu * floor_mu))
        ramp_mu = min(1.0, max(0.02, mu * mu / block_mu))

        block = BodySpec(
            name="block", kind="cube",
            position=C.on_ramp(centre, tilt, start_along, half + thick),
            scale=(half,) * 3,
            quaternion=(math.cos(tilt / 2.0), 0.0, math.sin(tilt / 2.0), 0.0),
            velocity=tuple(v0 * x for x in d),
            mass=1.0, friction=block_mu, restitution=0.1,
            color=C.hue_rgb(float(rng.uniform(0, 1))),
            segmentation_id=self.SEG_BLOCK, role="actor")

        block = C.with_material(block, M.pick(arng), arng)

        camera_position, camera_look_at = cam.frame_box(
            x_range=(-x_lip - half, x_land + run_out + half),
            z_range=(0.0, centre[2] + half_len * sin_t + thick + 2 * half))

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR),
                    C.ramp(self.SEG_RAMP, tilt, centre, half_len, 0.85, thick,
                           ramp_mu),
                    block, C.understudy(block, self.SEG_SPLIT)],
            lights=C.lights(cx, look_at=(0, 0, 1.0)),
            camera_position=camera_position, camera_look_at=camera_look_at,
            floor_level=0.0, complexity=complexity,
            notes={"tilt_rad": tilt, "mu": mu, "half_extent": half,
                   "down_slope": list(d), "ramp_id": self.SEG_RAMP,
                   "block_friction": block_mu, "ramp_friction": ramp_mu,
                   "x_land": x_land, "run_out": run_out})


register(RampSlide())

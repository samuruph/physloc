"""`ramp_slide` -- a block slides down an incline.

Sustained contact with friction for the whole clip, which is what separates it
from every ballistic scenario: `friction` and `newton1_inertia` need a body
whose motion is being *continuously* mediated by a surface, so that "it stopped
on its own" is a statement about a force that is not there. Grounded in LikePhys
*Block Slide*.
"""
from __future__ import annotations

import math

from . import _common as C
from . import materials as M
from ._hdri import pick as pick_hdri
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
        centre = (0.0, 0.0, 0.95)
        half = float(rng.uniform(0.18, 0.24))
        v0 = float(rng.uniform(0.3, 0.7))
        d, _ = C.ramp_axes(tilt)

        block = BodySpec(
            name="block", kind="cube",
            position=C.on_ramp(centre, tilt, -0.85, half + thick),
            scale=(half,) * 3,
            quaternion=(math.cos(tilt / 2.0), 0.0, math.sin(tilt / 2.0), 0.0),
            velocity=tuple(v0 * x for x in d),
            mass=1.0, friction=mu, restitution=0.1,
            color=C.hue_rgb(float(rng.uniform(0, 1))),
            segmentation_id=self.SEG_BLOCK, role="actor")

        block = C.with_material(block, M.pick(arng), arng)


        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR),
                    C.ramp(self.SEG_RAMP, tilt, centre, half_len, 0.85, thick, mu),
                    block, C.understudy(block, self.SEG_SPLIT)],
            lights=C.lights(cx, look_at=(0, 0, 1.0)),
            camera_position=(0.5, -6.6, 2.5), camera_look_at=(0.0, 0.0, 1.0),
            floor_level=0.0, complexity=complexity,
            hdri_id=pick_hdri(C.appearance_rng(seed, "hdri")) if cx.background == "hdri" else None,
            notes={"tilt_rad": tilt, "mu": mu, "half_extent": half,
                   "down_slope": list(d), "ramp_id": self.SEG_RAMP})


register(RampSlide())

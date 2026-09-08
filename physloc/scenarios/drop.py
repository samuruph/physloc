"""`drop` -- a sphere falls to a floor and bounces.

Event structure: free fall -> contact -> rebound. The simplest scenario that
still has a well-defined contact instant, which is what `solidity`,
`superelastic` and `antigravity` all need. Grounded in LikePhys *Ball Drop*.
"""
from __future__ import annotations

from . import _common as C
from . import materials as M
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class Drop(Scenario):
    name = "drop"

    # Segmentation ids are fixed per role so downstream annotation never has to
    # guess which instance is the actor.
    SEG_FLOOR, SEG_BALL, SEG_SPLIT = 1, 2, 4

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
        radius = float(rng.uniform(0.35, 0.55))
        drop_height = float(rng.uniform(2.4, 3.4))
        restitution = float(rng.uniform(0.55, 0.75))
        # A little lateral drift so the contact is not perfectly axis-aligned.
        vx, vy = float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5))
        hue = float(rng.uniform(0.0, 1.0))

        floor = C.ground(cx, self.SEG_FLOOR)

        # Sphere or cube: both fall and bounce, so the shape is free to vary
        # and two instances of drop do not look like one clip twice.
        # Any shape it likes: nothing here assumes a rolling contact, so a
        # cone that topples and a ring that rolls away are both fair
        # pictures of the same physics. Drawn off the appearance
        # stream -- see `C.pick_shape`.
        kind = C.pick_shape(arng)
        ball = BodySpec(
            name="ball", kind=kind, position=(0.0, 0.0, drop_height),
            scale=(radius, radius, radius), velocity=(vx, vy, 0.0),
            mass=1.0, friction=0.4, restitution=restitution,
            color=C.hue_rgb(hue), segmentation_id=self.SEG_BALL, role="actor")
        # Made of something, which fixes both its mass and its look. Drawn off
        # the appearance stream so it cannot shift the physics draws above it;
        # see `_common.with_material`.
        ball = C.with_material(ball, M.pick(arng), arng)
        # ...and its own proportions. A no-op on the sphere draws; on the cube
        # draws it is the difference between "a cube" and "a slab", which is
        # variety a single radius could never produce.
        ball = C.vary_dims(ball, arng)

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[floor, ball, C.understudy(ball, self.SEG_SPLIT)],
            lights=C.lights(cx),
            camera_position=(5.2, -4.4, 2.4), camera_look_at=(0.0, 0.0, 1.0),
            floor_level=0.0, complexity=complexity,            notes={"radius": radius, "drop_height": drop_height,
                   "restitution": restitution, "actor_kind": kind},
        )


register(Drop())

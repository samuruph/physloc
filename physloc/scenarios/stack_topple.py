"""`stack_topple` -- a marginally stable stack that falls over on its own.

The valid clip already contains dramatic motion, which is the point: it is the
control for "surprising but lawful". A model that flags every clip where
something collapses is not detecting violations, and this scenario is where
that shows up. Grounded in IntPhys 2 / LikePhys stability.
"""
from __future__ import annotations

from . import _common as C
from . import materials as M
from ._hdri import pick as pick_hdri
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class StackTopple(Scenario):
    name = "stack_topple"
    SEG_FLOOR, SEG_TOP, SEG_MID, SEG_BASE = 1, 2, 4, 5

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        arng = C.appearance_rng(seed, self.name)
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        half = float(rng.uniform(0.24, 0.30))
        # Each block overhangs the one below by a fraction of its width. The
        # stack falls once the combined centre of mass of everything above the
        # base clears the base's support polygon, which for three blocks means
        # `1.5 * lean > half`. The old range straddled that threshold, so the
        # "topple" scenario shipped a stack that stood there for the whole clip
        # -- and a scenario whose event never happens is not a control for
        # surprising-but-lawful, it is just furniture.
        lean = float(rng.uniform(0.80, 1.00)) * half
        side = float(rng.choice([-1.0, 1.0]))
        hue = float(rng.uniform(0, 1))

        # ONE material for the whole stack, and its proportions left alone.
        # This scenario is balanced at the edge of stability on purpose -- that
        # is what makes "it stood when it should have fallen" worth staging --
        # and a per-block density or a per-block aspect ratio would push every
        # seed off that edge one way or the other. The colour still walks by
        # level so the three blocks stay tellable apart.
        stack_mat = M.pick(arng)
        stack_look = M.appearance(stack_mat, arng)

        def block(name, level, seg):
            body = BodySpec(
                name=name, kind="cube",
                position=(side * lean * level, 0.0, half * (2 * level + 1)),
                scale=(half,) * 3, mass=1.0, friction=0.45, restitution=0.05,
                color=C.hue_rgb((hue + 0.11 * level) % 1.0),
                segmentation_id=seg,
                role="actor" if level == 2 else "prop")
            return C.with_material(body, stack_mat, arng, mass_from="ratio",
                                   look=(body.color,) + stack_look[1:])

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR),
                    block("base", 0, self.SEG_BASE),
                    block("mid", 1, self.SEG_MID),
                    block("top", 2, self.SEG_TOP)],
            lights=C.lights(cx, look_at=(0, 0, 0.7)),
            camera_position=(3.4, -4.8, 2.0), camera_look_at=(0.0, 0.0, 0.8),
            floor_level=0.0, complexity=complexity,
            hdri_id=pick_hdri(C.appearance_rng(seed, "hdri")) if cx.background == "hdri" else None,
            notes={"half_extent": half, "lean_per_level": lean,
                   "stack_ids": [self.SEG_BASE, self.SEG_MID, self.SEG_TOP]})


register(StackTopple())

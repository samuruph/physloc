"""`pyramid_impact` -- a cube dropped onto a pyramid of spheres.

A multi-body contact chain: the impact propagates through four bodies, so
`newton3_reaction` and `newton2_mass` have a collision whose *lawful* outcome is
visibly rich. Grounded in LikePhys *Pyramid Impact*.
"""
from __future__ import annotations

import math

from . import _common as C
from . import materials as M
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class PyramidImpact(Scenario):
    name = "pyramid_impact"
    SEG_FLOOR, SEG_CUBE = 1, 2
    SEG_BALLS = (4, 5, 6, 7)

    #: How much denser the striker is than the pile it lands on. The
    #: hand-picked 2.2-vs-0.8 masses were a 2.75, and the band brackets it: too
    #: little and the cube bounces off, too much and there is no pyramid left
    #: to violate anything about.
    STRIKER_RATIO = (2.0, 6.0)

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        arng = C.appearance_rng(seed, self.name)
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        r = float(rng.uniform(0.26, 0.32))
        hue = float(rng.uniform(0, 1))
        # Three spheres on an equilateral base, one nested in the dimple above.
        s = r * 1.02
        base_xy = [(-s, -s / math.sqrt(3.0)), (s, -s / math.sqrt(3.0)),
                   (0.0, 2.0 * s / math.sqrt(3.0))]
        apex_z = r + math.sqrt(max((2 * r) ** 2 - (2 * s / math.sqrt(3.0)) ** 2,
                                   1e-4))

        balls = [BodySpec(name="ball_%d" % i, kind="sphere",
                          position=(x, y, r), scale=(r,) * 3, mass=0.8,
                          friction=0.5, restitution=0.35,
                          color=C.hue_rgb((hue + 0.13 * i) % 1.0),
                          segmentation_id=self.SEG_BALLS[i], role="prop")
                 for i, (x, y) in enumerate(base_xy)]
        balls.append(BodySpec(name="ball_apex", kind="sphere",
                              position=(0.0, 0.0, apex_z), scale=(r,) * 3,
                              mass=0.8, friction=0.5, restitution=0.35,
                              color=C.hue_rgb((hue + 0.39) % 1.0),
                              segmentation_id=self.SEG_BALLS[3], role="prop"))

        half = float(rng.uniform(0.24, 0.30))
        drop = apex_z + r + half + float(rng.uniform(0.9, 1.4))
        cube = BodySpec(
            name="cube", kind="cube", position=(0.0, 0.0, drop),
            scale=(half,) * 3, mass=2.2, friction=0.5, restitution=0.2,
            color=C.hue_rgb((hue + 0.5) % 1.0),
            segmentation_id=self.SEG_CUBE, role="actor")

        # The striker is made of something DENSER than the pile, which is what
        # the hand-picked 2.2-vs-0.8 masses were expressing before materials
        # existed. Keeping that ordering is the scenario: a cube that does not
        # scatter the pyramid stages nothing for `solidity` or `superelastic`
        # to violate. Drawing both from a split of the material list guarantees
        # it rather than hoping the draws land that way.
        # The MASS RATIO is the tuned quantity, and it is what has to be
        # chosen -- not the two materials. Drawing them independently makes the
        # ratio a product of three things (two densities and two volumes): a
        # steel striker on a plastic pile came out 32 to 1 against the 2.75 the
        # hand-picked masses expressed, and a cube that heavy does not scatter
        # a pyramid so much as delete it, leaving `solidity` and `superelastic`
        # staged on a valid clip that is already chaos.
        #
        # So: draw the pile freely, then pick the striker material whose
        # resulting mass -- density x its ACTUAL volume, sizes and all --
        # lands closest to the middle of the band. Both bodies stay honestly
        # volume-derived, so a heavy-looking striker really is the heavy one.
        # The pile never draws the densest material, so there is always a
        # denser one left for the striker. With a steel pile the best available
        # ratio was 1.2, and a cube that barely outweighs the pyramid bounces
        # off it instead of scattering it.
        ball_mat = M.pick(arng, tuple(m for m in M.ACTOR_MATERIALS
                                      if m != max(M.ACTOR_MATERIALS,
                                                  key=M.density_ratio)))
        balls = [C.with_material(b, ball_mat, arng,
                                 look=(b.color,) + M.appearance(ball_mat, arng)[1:])
                 for b in balls]
        cube = C.vary_dims(cube, arng)
        m_ball = max(b.mass for b in balls)
        want = m_ball * sum(self.STRIKER_RATIO) / 2.0
        cube_mat = min(
            M.ACTOR_MATERIALS,
            key=lambda m: abs(M.mass_for(m, cube.scale, cube.kind) - want))
        cube = C.with_material(cube, cube_mat, arng)

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR)] + balls + [cube],
            lights=C.lights(cx, look_at=(0, 0, 0.5)),
            camera_position=(3.0, -4.6, 2.0), camera_look_at=(0.0, 0.0, 0.6),
            floor_level=0.0, complexity=complexity,
            notes={"radius": r, "apex_z": apex_z, "drop_height": drop,
                   "pyramid_ids": list(self.SEG_BALLS),
                   # The falling cube is the actor, but it is not the
                   # interesting body to break. What a video generator gets
                   # wrong here is the *scatter*: a struck sphere driven
                   # through the ground or through its neighbour. The apex goes
                   # first because it takes the impact directly.
                   "family_targets": {
                       "solidity": [self.SEG_BALLS[3], self.SEG_BALLS[0],
                                    self.SEG_BALLS[1], self.SEG_BALLS[2]],
                       # The Newton families need two bodies a viewer cannot
                       # tell apart, so they get the spheres and never the
                       # cube -- otherwise "the cube is heavier" is a perfectly
                       # lawful reading of the clip.
                       "newton2_mass": list(self.SEG_BALLS),
                       "newton3_reaction": list(self.SEG_BALLS)}})


register(PyramidImpact())

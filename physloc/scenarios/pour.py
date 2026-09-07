"""`pour` -- a loose column of small spheres falls into an open box.

**This is not a fluid and must never be labelled one.** True liquid was tested
against the pinned image and does not work: Blender 2.93.4 ships Mantaflow but
headless baking fails (`NameError: liquid_save_data_N` -> `Manta::Error`),
Kubric exposes no fluid object, and a liquid does not fit a pose-based
trajectory seam in any case. So v0 ships the honest neighbour -- a few dozen
rigid grains, which streams, piles and breaks up like a granular medium -- and
labels it `physics_medium: "granular"`. `physloc validate` rejects any clip
claiming `"fluid"` at schema v0. Real fluid and cloth are Phase 3, behind a
newer Blender.

Every grain is an `actor`, so a violation here acts on the whole medium rather
than singling out one bead: a `global_gravity` or `antigravity` clip shows the
pour itself misbehaving, which is the point.
"""
from __future__ import annotations

from . import _common as C
from . import materials as M
from ._hdri import pick as pick_hdri
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)

SEG_GRAIN_BASE = 16


class Pour(Scenario):
    name = "pour"
    SEG_FLOOR = 1
    SEG_WALLS = (3, 4, 5, 6)

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        arng = C.appearance_rng(seed, self.name)
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        # Grain count follows the tier, not taste: at the debug tier the clip is ~1 s
        # and a hundred beads is render cost spent on pixels nobody inspects.
        n_grains = 40 if tier.name == "debug" else 96
        r = float(rng.uniform(0.070, 0.085))
        hue = float(rng.uniform(0, 1))

        # Low walls, and a camera high enough to see over the near one.
        # At 0.34 the front wall hid the entire pile: a grain sinking
        # through the floor was behind it in both twins, so the violation
        # was active, correctly scored and completely unobservable.
        half, wall_t, wall_h = 0.82, 0.05, 0.24
        walls = []
        for i, (cx_, cy_, sx, sy) in enumerate([
                (0.0, half, half + wall_t, wall_t),
                (0.0, -half, half + wall_t, wall_t),
                (half, 0.0, wall_t, half + wall_t),
                (-half, 0.0, wall_t, half + wall_t)]):
            walls.append(BodySpec(
                name="wall_%d" % i, kind="cube",
                position=(cx_, cy_, wall_h), scale=(sx, sy, wall_h),
                mass=0.0, static=True, friction=0.6, restitution=0.1,
                color=(0.34, 0.35, 0.40), segmentation_id=self.SEG_WALLS[i],
                role="prop"))

        grains, heights = [], []
        for i in range(n_grains):
            rad = float(rng.uniform(0.7, 1.0)) * 0.45
            ang = float(rng.uniform(0.0, 6.283185))
            z0 = float(rng.uniform(0.55, 2.25))
            heights.append((z0, SEG_GRAIN_BASE + i))
            grains.append(BodySpec(
                name="grain_%03d" % i, kind="sphere",
                position=(rad * float(rng.uniform(-1, 1)),
                          rad * float(rng.uniform(-1, 1)), z0),
                scale=(r,) * 3,
                velocity=(0.0, 0.0, float(rng.uniform(-0.4, 0.0))),
                mass=0.12, friction=0.5, restitution=0.2,
                color=C.hue_rgb((hue + 0.03 * i) % 1.0, s=0.5, v=0.85),
                segmentation_id=SEG_GRAIN_BASE + i, role="actor"))
            del ang

        # ONE material for the whole medium. The grains are interchangeable by
        # construction -- that is what makes `_group` able to act on all of
        # them at once and what makes a pour read as a substance rather than as
        # ninety-six separate objects -- so giving them different densities
        # would be ninety-six little unlabelled mass violations. The colour
        # still walks per grain, which is what keeps the pile legible.
        grain_mat = M.pick(arng)
        grain_look = M.appearance(grain_mat, arng)
        grains = [C.with_material(g, grain_mat, arng, mass_from="ratio",
                                  look=(g.color, grain_look[1], grain_look[2]))
                  for g in grains]

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR)] + walls + grains,
            lights=C.lights(cx, look_at=(0, 0, 0.3)),
            camera_position=(1.7, -3.2, 3.0), camera_look_at=(0.0, 0.0, 0.15),
            floor_level=0.0, complexity=complexity, physics_medium="granular",
            hdri_id=pick_hdri(C.appearance_rng(seed, "hdri")) if cx.background == "hdri" else None,
            notes={"n_grains": n_grains, "grain_radius": r,
                   "box_half_width": half,
                   "grain_ids": [SEG_GRAIN_BASE + i for i in range(n_grains)],
                   # Every family that can act on a set acts on ALL of the
                   # grains here. A pour is not a scene with one protagonist:
                   # a single grain floating, vanishing or dropping through the
                   # floor is a handful of pixels somewhere in a pile, and the
                   # clip reads as nothing having happened. The whole medium
                   # misbehaving is what the scenario is for.
                   "group_fraction": 1.0})


register(Pour())

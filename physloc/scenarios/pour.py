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

        # **The count and the box are one decision, and the quantity that
        # matters is HOW MANY GRAINS DEEP the medium settles.**
        #
        # A 1.64 m box holds about 140 of these grains in a single layer, so
        # forty of them covered under a third of the floor and the pour settled
        # exactly ONE GRAIN DEEP -- measured, every grain in the lawful clip
        # ended at z = 0.073, heavy and light alike. A medium with no interior
        # is not a medium: `newton2_mass` made half the grains 25x heavier and
        # they had nothing to sink through, so the clip differed from its twin
        # while its three bins did not differ from each other (0.334 / 0.338 /
        # 0.382 m), and `friction` had no heap whose angle it could change.
        #
        # Depth is `n * (2pi/3) * r^2 / (packing * floor area)`, so the count,
        # the radius and the vessel are one knob with three handles. The vessel
        # came down from 1.64 m and the count went up, because grains alone
        # would have had to reach several hundred.
        #
        # **The radius is then set by what a grain should LOOK like**, and the
        # count follows to hold the depth. At 0.069 m in a 0.68 m box the
        # medium is under five beads across and reads as marbles; at 0.059 m it
        # is nearly six, and it is still seven pixels at the debug tier's
        # 128 px -- small enough to read as a medium, large enough to resolve.
        #
        # **How far to go is limited by the residual laws, not by the render.**
        # `penetration` reports depth in RADII and `trajectory_shape` normalises
        # by radius, so shrinking a grain magnifies every residual measured on
        # it. Measured, twice: at 0.048 m -- seven grains across, which looks
        # better still -- `solidity` went to 0.94 / 0.91 / 1.00,
        # `superelastic` to 1.00 / 0.87 / 1.00 and `friction` to a flat 1.000;
        # at 0.056 m `superelastic` still came out 0.29 / 0.19 / 0.42, out of
        # order. 0.063 m is as fine as the ladders currently hold.
        #
        # The count is then chosen WITH the radius to keep `n r^2 / A` -- and
        # therefore the pile depth -- exactly where those ladders were
        # calibrated, so the only thing that changes is how fine the medium
        # looks.
        n_grains = 96 if tier.name == "debug" else 212
        r = float(rng.uniform(0.058, 0.068))
        hue = float(rng.uniform(0, 1))

        # Low walls, and a camera high enough to see over the near one.
        # At 0.34 the front wall hid the entire pile: a grain sinking
        # through the floor was behind it in both twins, so the violation
        # was active, correctly scored and completely unobservable. The wall
        # came down with the box, for the same reason -- what matters is the
        # wall against the PILE, and the pile is about 0.46 m tall.
        #
        # **Deliberately lower than the pile, and raising it was tried and
        # reverted.** A finer medium overflows more, so 0.28 was tested to
        # contain it -- and it did, at the cost of the one family that needs
        # the medium free to spread: confined by a taller wall, `friction` has
        # nothing left to reduce and went from a saturating 1.00 / 1.00 / 1.00
        # to a flat 0.00 / 0.00 / 0.00, which is a clip depicting nothing. The
        # overflow is contained by the FRAME instead -- see the camera below.
        half, wall_t, wall_h = 0.34, 0.04, 0.20
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

        # **A COLUMN, not a curtain.** The grains used to be released across
        # +/-0.45 m -- a 0.9 m spread into a 1.64 m box -- so they arrived
        # already spread out and settled ONE GRAIN DEEP: measured, every grain
        # in the lawful clip ended at z = 0.073, heavy and light alike. A
        # single layer has no interior, so two families had nothing to act on.
        # `newton2_mass` made half the grains 25x heavier and they had nothing
        # to sink through -- the clip differed from its twin (a mean 0.34 m per
        # grain) but the three bins were indistinguishable at 0.334 / 0.338 /
        # 0.382 m, because any mismatch past ~3x fully decorrelates a chaotic
        # packing. `friction` had no heap whose angle it could change.
        #
        # Poured down a narrow column the medium piles up and then avalanches
        # out to its angle of repose, which is the thing friction actually
        # determines and the thing a dense grain actually sinks through.
        grains, heights = [], []
        for i in range(n_grains):
            rad = float(rng.uniform(0.7, 1.0)) * 0.12
            ang = float(rng.uniform(0.0, 6.283185))
            # Lower than the 0.55--2.25 this poured from: a grain arriving at
            # 6.6 m/s does not join a pile, it splashes one. Not much lower,
            # though -- taken to 0.42 the medium was on the floor by frame 2,
            # `before_medium_lands` returned frame 1, and every family here
            # fired with a single frame of lawful prefix behind it.
            z0 = float(rng.uniform(0.68, 1.90))
            heights.append((z0, SEG_GRAIN_BASE + i))
            grains.append(BodySpec(
                name="grain_%03d" % i, kind="sphere",
                position=(rad * float(rng.uniform(-1, 1)),
                          rad * float(rng.uniform(-1, 1)), z0),
                scale=(r,) * 3,
                velocity=(0.0, 0.0, float(rng.uniform(-0.4, 0.0))),
                mass=0.12, friction=0.5, restitution=0.2,
                # What lets the medium HOLD A PILE. Without it the grains are
                # frictionless rollers with no angle of repose, and the pour
                # settled one grain deep across the whole box however narrowly
                # it was poured -- see `BodySpec.rolling_friction`. Well under
                # the 0.08-0.24 the `friction` family solves for, so that
                # violation still has room to be a violation.
                rolling_friction=0.12,
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
                                  look=(g.color,) + tuple(grain_look[1:]))
                  for g in grains]

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR)] + walls + grains,
            lights=C.lights(cx, look_at=(0, 0, 0.3)),
            # Moved in AND up with the box. The old pose framed 1.66 m of
            # world, which around a 0.68 m vessel is mostly empty floor. This
            # frames 1.49 m -- the grains that overflow the low walls reach
            # 1.43 m from the centre at 96 grains, and a grain outside the
            # frame is a culprit with no pixels in a clip that claims one --
            # and looks down at 52
            # degrees rather than 36, so the near wall hides 0.16 m of the box
            # instead of 0.27. That is the trap the wall height already
            # carries a comment about: at a shallow angle a wall tall enough to
            # hold the pour is also tall enough to hide it.
            camera_position=(1.20, -2.23, 3.47), camera_look_at=(0.0, 0.0, 0.20),
            floor_level=0.0, complexity=complexity, physics_medium="granular",
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

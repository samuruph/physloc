"""A rung must change ONE thing, and nothing else -- docs/roadmap.md section 3a.

**These are not twins.** "Twin" in this project means the valid/invalid pair:
one scene, bit-identical prefix up to `t_event`. Two complexity levels are not
twins of each other and the dataset does not ship them as such -- every rung
draws its own scenes from its own seed block, because a ladder whose upper
rungs contain no new physical events adds difficulty but no breadth.

What is tested here is a property of the SAMPLER, not of the release: handed
the same seed, two rungs must differ only on the axis between them. That is
what proves a rung changes its own axis and leaves everything else alone -- the
appearance streams are salted, `pick_hdri` does not steal a physics draw, and
turning the level up does not silently resample the scene. Generation then
hands the rungs different seeds on purpose, and the guarantee still holds: it
is about what a level DOES, not about which seeds it is given.

L0 -> L1 is MATERIALS, so mass is the one physical quantity that is supposed to
move across it -- mass is `density x volume`, and giving an object a material is
giving it a density. L1 -> L2 is the ENVIRONMENT, which changes no physics at
all.
"""
from __future__ import annotations

import numpy as np
import pytest

import mockroll
from physloc import scenarios
from physloc.scenarios import TIERS
from physloc.scenarios.base import COMPLEXITY

SEED = 777
NAMES = sorted(scenarios.available())


@pytest.mark.parametrize("name", NAMES)
def test_appearance_draws_do_not_shift_the_physics_stream(name):
    """Appearance has its own salted stream, so turning materials on does not
    resample the scene.

    `pick_hdri(rng)` used to draw from the PHYSICS stream, and because it only
    fired at the realistic level the extra draw shifted every physics value
    after it -- so the two levels were independent releases wearing the same
    seed. Everything a scenario samples is now the same at both levels except
    what materials own: mass, and how the surface looks.
    """
    sc = scenarios.get(name)
    a = sc.sample(SEED, TIERS["release"], "L0")
    b = sc.sample(SEED, TIERS["release"], "L1")
    assert len(a.bodies) == len(b.bodies)
    for x, y in zip(a.bodies, b.bodies):
        if x.role == "floor":
            continue                      # the known gap, covered below
        for field in ("kind", "position", "scale", "friction",
                      "restitution", "velocity", "quaternion"):
            assert getattr(x, field) == getattr(y, field), (
                "%s: %s.%s differs across complexity" % (name, x.name, field))
    assert a.camera_position == b.camera_position
    assert a.camera_look_at == b.camera_look_at


@pytest.mark.parametrize("name", NAMES)
def test_materials_are_what_l1_changes(name):
    """The other half: the rung has to actually DO something.

    A ladder whose rungs are indistinguishable measures nothing, and a test
    that only checks what stayed the same would pass on a level that changed
    nothing at all. L1 gives every body a material, which gives it a density,
    which gives it a mass.

    OVER SEEDS, not on one, because `wood` is `REFERENCE_MATERIAL` -- the single
    density L0 flattens everything to -- so a body that draws wood at L1 keeps
    exactly its L0 mass. That is right, and on one seed four scenarios drew it
    and looked like a level that did nothing.
    """
    sc = scenarios.get(name)
    moved = 0
    for seed in range(SEED, SEED + 8):
        a = sc.sample(seed, TIERS["release"], "L0")
        b = sc.sample(seed, TIERS["release"], "L1")
        assert not any(x.material for x in a.bodies), (
            "%s: L0 is the one shared density -- see `_flatten_materials`" % name)
        assert any(y.material for y in b.bodies), (
            "%s: L1 gave nothing a material" % name)
        moved += any(x.mass != y.mass for x, y in zip(a.bodies, b.bodies)
                     if x.role != "floor")
    assert moved, "%s: materials changed no mass on any seed" % name


@pytest.mark.parametrize("name", NAMES)
def test_the_whole_built_ladder_rolls_identically(name):
    """Handed one seed, every rung rolls it the same way.

    Not a claim that the release contains these pairs -- it does not, by
    design. A claim that a rung's machinery touches its own axis and nothing
    else, which is the only reason a level comparison means anything at all.

    It used to hold for one scenario across one step, and the blocker was
    geometry: `C.ground` returned a cube below the HDRI level and a KuBasic
    dome at it, which is a genuinely different collision shape, so the same
    seed did not roll the same way and the levels were independent releases
    that happened to share a seed.

    The dome is now the ground at every rung -- shaded flat below L2, lit by an
    HDRI at it -- so the geometry is identical by construction. Measured before
    committing to the swap (`physloc/render/probe_dome.py`): a 0.35 m sphere
    dropped on each surface at 0 to 5 m from the origin rests at 0.3500 on the
    cube and 0.3509 on the dome, flat to within Bullet's collision margin. Once
    both sides use it the difference is zero rather than a millimetre.

    This is the HOST rollout, which is an approximation of PyBullet -- so it
    pins that nothing in the SPEC differs in a way the physics would see, not
    that the container agrees to the bit. `prefix_identical` is what guards the
    render, and it guards it per twin rather than across rungs.
    """
    sc = scenarios.get(name)
    built = [k for k, v in COMPLEXITY.items() if v.implemented]
    rolls = [mockroll.roll(sc.sample(SEED, TIERS["release"], lv), sc)
             for lv in built]
    for lv, roll in zip(built[1:], rolls[1:]):
        assert roll.pos.shape == rolls[0].pos.shape, (name, lv)
        assert np.allclose(rolls[0].pos, roll.pos, atol=1e-9), (
            "%s: %s rolls differently from %s -- max %.3e"
            % (name, lv, built[0], float(np.abs(rolls[0].pos - roll.pos).max())))

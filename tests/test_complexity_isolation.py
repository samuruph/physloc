"""A level must change ONE thing, and nothing else -- docs/roadmap.md section 3a.

**These are not twins.** "Twin" in this project means the valid/invalid pair:
one scene, bit-identical prefix up to `t_event`. Two complexity levels are not
twins of each other and the dataset does not ship them as such -- every level
draws its own scenes from its own seed block, because a ladder whose upper
levels contain no new physical events adds difficulty but no breadth.

What is tested here is a property of the SAMPLER, not of the release: handed
the same seed, two levels must differ only on the axis between them. That is
what proves a level changes its own axis and leaves everything else alone -- the
appearance streams are salted, `pick_hdri` does not steal a physics draw, and
turning the level up does not silently resample the scene. Generation then
hands the levels different seeds on purpose, and the guarantee still holds: it
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
    """The other half: the level has to actually DO something.

    A ladder whose levels are indistinguishable measures nothing, and a test
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
    """Handed one seed, every level with the same OBJECTS rolls it the same way.

    Not a claim that the release contains these pairs -- it does not, by
    design. A claim that a level's machinery touches its own axis and nothing
    else, which is the only reason a level comparison means anything at all.

    The blocker this closed was geometry: `C.ground` returned a cube below the
    HDRI level and a KuBasic dome at it, so the ground changed shape under a
    level that was supposed to be about lighting. The ground is now a cube slab
    everywhere; the dome survives at the HDRI levels as `C.backdrop`, a
    render-only body whose collisions the worker disables. So the COLLIDER is
    uniform -- which is all this test needs -- while only the levels that light
    themselves from an environment map pay for a surface that encloses the
    scene. L0, L1 and L2 roll identically on all thirteen scenarios.

    **L3 IS EXEMPT, and that is the level working.** It replaces primitives with
    scanned GSO meshes, so the collision geometry changes by construction: a
    shark does not roll like a sphere. Asserting otherwise would be asserting
    that the hardest level does nothing. The exemption is derived from
    `actor_assets` rather than hardcoded, so a future level that keeps
    primitives is still held to the rule.

    This is the HOST rollout, an approximation of PyBullet -- so it pins that
    nothing in the SPEC differs in a way the physics would see, not that the
    container agrees to the bit. `prefix_identical` guards the render, per
    twin.
    """
    sc = scenarios.get(name)
    built = [k for k, v in COMPLEXITY.items() if v.implemented]
    base_assets = COMPLEXITY[built[0]].actor_assets
    same = [k for k in built if COMPLEXITY[k].actor_assets == base_assets]
    assert len(same) >= 2, "nothing to compare"
    rolls = {lv: mockroll.roll(sc.sample(SEED, TIERS["release"], lv), sc)
             for lv in built}
    for lv in same[1:]:
        assert rolls[lv].pos.shape == rolls[same[0]].pos.shape, (name, lv)
        assert np.allclose(rolls[same[0]].pos, rolls[lv].pos, atol=1e-9), (
            "%s: %s rolls differently from %s -- max %.3e"
            % (name, lv, same[0],
               float(np.abs(rolls[same[0]].pos - rolls[lv].pos).max())))


@pytest.mark.parametrize("name", NAMES)
def test_the_gso_level_actually_changes_the_objects(name):
    """The other half: a level whose objects are indistinguishable from the
    level below measures nothing.

    L3 swaps every non-static actor and distractor for a scanned asset, sized
    so its LONGEST AXIS matches what the primitive was drawn at -- MOVi's
    normalisation (`movi_c_worker.py:167-170`). Without that the level would
    change the scene's scale as well as its geometry, and two axes would move
    at once.
    """
    gso = [k for k, v in COMPLEXITY.items()
           if v.implemented and v.actor_assets == "gso"]
    if not gso:
        pytest.skip("no GSO level is built")
    sc = scenarios.get(name)
    plain = sc.sample(SEED, TIERS["release"], "L0")
    scanned = sc.sample(SEED, TIERS["release"], gso[0])
    swapped = [b for b in scanned.bodies if b.kind == "gso"]
    assert swapped, "%s: the GSO level swapped nothing" % name
    assert all(b.asset_id for b in swapped), name
    for before, after in zip(plain.bodies, scanned.bodies):
        if after.kind != "gso":
            continue
        # The DRAWN size is preserved; `scale` is a normalising factor and
        # says nothing on its own -- a 30 mm block scaled to 0.4 m carries
        # scale 13.
        assert 2 * max(after.extents) == pytest.approx(
            2 * max(before.extents), rel=0.02), (
            "%s: %s changed size across the level" % (name, after.name))

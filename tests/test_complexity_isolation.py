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
    render-only body declared `collides=False`. So the COLLIDER is
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

    # COMPARED BODY BY BODY, not array against array. The HDRI levels carry one
    # body the levels below do not -- the backdrop dome -- so the position
    # blocks have different widths and a shape assert fails on a difference
    # that is render-only by construction: `_common.backdrop` has its
    # collisions disabled in the worker and cannot touch anything. Matching on
    # segmentation id says what the test means: every body BOTH levels have
    # follows the same path.
    def by_id(level):
        spec = sc.sample(SEED, TIERS["release"], level)
        roll = mockroll.roll(spec, sc)
        return spec, {int(b.segmentation_id): roll.pos[:, i, :]
                      for i, b in enumerate(spec.bodies)}

    base_spec, base = by_id(same[0])
    for lv in same[1:]:
        spec, other = by_id(lv)
        shared = set(base) & set(other)
        assert shared, (name, lv, "no bodies in common")
        # The only body a level may add or drop is the backdrop.
        extra = {int(b.segmentation_id) for b in spec.bodies} ^ {
            int(b.segmentation_id) for b in base_spec.bodies}
        roles = {int(b.segmentation_id): b.role
                 for b in list(spec.bodies) + list(base_spec.bodies)}
        assert all(roles[i] == "backdrop" for i in extra), (
            "%s: %s adds or drops %s, which is not a backdrop"
            % (name, lv, sorted((i, roles[i]) for i in extra)))
        for seg in sorted(shared):
            assert np.allclose(base[seg], other[seg], atol=1e-9), (
                "%s: %s rolls body %d differently from %s -- max %.3e"
                % (name, lv, seg, same[0],
                   float(np.abs(base[seg] - other[seg]).max())))


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

# --------------------------------------------------------------------------
# What a level must NOT change: what a body is standing on, and what a family
# decides to break. Both were shipping wrong at the HDRI and GSO levels.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_the_backdrop_is_never_a_surface(name):
    """`solidity` breaks the same contact at every level.

    The dome is static, 40 m across, and its inner surface is coplanar with the
    slab, so a geometry search for "what is under this body" found it and
    answered with the BACKDROP. Measured on the review sweep: `barrier_pass`,
    `collision`, `drop` and `stack_topple` planned `pass_through` against a wall
    or another body at L0 and L1, and `sink` against segmentation id 900 at L2
    and L3 -- four scenarios where the clip showed a body dropping through the
    ground while the annotation named something else.

    Written against the PLAN rather than against a rendered clip, because the
    plan is where the choice is made and the host can reach it.
    """
    from physloc.injectors import get as get_injector

    sc = scenarios.get(name)
    built = [k for k, v in COMPLEXITY.items() if v.implemented]
    inj = get_injector("solidity")
    modes = {}
    for lv in built:
        spec = sc.sample(SEED, TIERS["release"], lv)
        backdrops = {int(b.segmentation_id) for b in spec.bodies
                     if not b.collides}
        traj = mockroll.roll(spec, sc)
        plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
        if plan is None:
            continue
        touched = (set(int(x) for x in plan.params.get("pair", ()))
                   | set(int(x) for x in plan.params.get("also_disable", ()))
                   | set(int(x) for x in plan.causal_body_ids))
        assert not (touched & backdrops), (
            "%s at %s: solidity acts on a body nothing can touch (%s)"
            % (name, lv, sorted(touched & backdrops)))
        modes[lv] = plan.params.get("mode")
    base = COMPLEXITY[built[0]].actor_assets
    same = [lv for lv in modes if COMPLEXITY[lv].actor_assets == base]
    assert len({modes[lv] for lv in same}) <= 1, (
        "%s: solidity breaks a different contact per level -- %s" % (name, modes))

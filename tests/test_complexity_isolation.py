"""A rung must change ONE thing -- docs/roadmap.md section 3a.

The design is that `(scenario, seed, variant)` names one event, staged at
several realism levels, so a benchmark can ask whether a model's grasp of the
physics survives the realism. That only means anything if each rung changes its
own axis and nothing else.

L0 -> L1 is MATERIALS, so mass is the one physical quantity that is supposed to
differ across it -- mass is `density x volume`, and giving an object a material
is giving it a density. Everything else must not move.
"""
from __future__ import annotations

import numpy as np
import pytest

import mockroll
from physloc import scenarios
from physloc.scenarios import TIERS

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


@pytest.mark.parametrize("name", ["drop"])
def test_complexity_twin_rolls_identically(name):
    """The GEOMETRY is identical across L0 -> L1, so a `drop` rolls the same.

    Mass differs by design at this rung, and a body in free fall does not care
    -- so the one scenario whose rollout is mass-independent is the one that
    can pin "the collision geometry did not change". That is the property that
    matters: it used to fail, because `C.ground` returned a cube at the plain
    level and a KuBasic dome at the realistic one, which is a genuine change of
    shape, and the two levels were independent releases rather than twins.

    The same gap now sits at L1 -> L2, where the HDRI dome arrives. Closing it
    is what would let a release and its harder twin be compared clip for clip.
    """
    sc = scenarios.get(name)
    a, b = sc.sample(SEED, TIERS["release"], "L0"), sc.sample(SEED, TIERS["release"], "L1")
    ta, tb = mockroll.roll(a, sc), mockroll.roll(b, sc)
    assert ta.pos.shape == tb.pos.shape
    assert np.allclose(ta.pos, tb.pos, atol=1e-9)

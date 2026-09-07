"""v0 and v1 must name the same physical event -- docs/roadmap.md section 3a.

The design is that `(scenario, seed)` is one event, rendered plainly at L0 and
photographically at L1, so a benchmark can ask whether a model's grasp of the
physics survives the realism. That only means anything if the rollout is
identical across complexity.
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
    """The half of the twin that already holds.

    `pick_hdri(rng)` used to draw from the physics stream, and because it only
    fires at L1 the extra draw shifted every physics value after it. Appearance
    now has its own salted stream, so everything a scenario samples *before*
    consulting the floor is the same at both levels.
    """
    sc = scenarios.get(name)
    a = sc.sample(SEED, TIERS["release"], "L0")
    b = sc.sample(SEED, TIERS["release"], "L1")
    for x, y in zip(a.bodies, b.bodies):
        if x.role == "floor":
            continue                      # the known gap, covered below
        for field in ("kind", "position", "scale", "mass", "friction",
                      "restitution", "velocity", "quaternion"):
            assert getattr(x, field) == getattr(y, field), (
                "%s: %s.%s differs across complexity" % (name, x.name, field))
    assert a.camera_position == b.camera_position
    assert a.camera_look_at == b.camera_look_at


@pytest.mark.parametrize("name", ["drop"])
def test_complexity_twin_rolls_identically(name):
    """L0 and L1 roll identically -- the ladder reordering is what fixed this.

    It used to be an expected failure: `C.ground` returns a cube at L0 and a
    KuBasic dome from the HDRI level up, which is a genuine collision-geometry
    change, so the same seed did not roll the same way and the two levels were
    independent releases rather than twins.

    L0 and L1 now share a background -- they differ only in whether the camera
    may move -- so the geometry is identical and the physics is too. The gap
    moves up the ladder to L3 -> L4, where the HDRI dome arrives, and it is the
    same gap: making the collision geometry identical across that step is what
    would let a release and its harder twin be compared clip for clip.
    """
    sc = scenarios.get(name)
    a, b = sc.sample(SEED, TIERS["release"], "L0"), sc.sample(SEED, TIERS["release"], "L1")
    ta, tb = mockroll.roll(a, sc), mockroll.roll(b, sc)
    assert ta.pos.shape == tb.pos.shape
    assert np.allclose(ta.pos, tb.pos, atol=1e-9)

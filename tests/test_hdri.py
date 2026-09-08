"""The environment map: chosen once, per level, and every id has to resolve.

The list is baked into the repo so `SceneSpec` sampling works on the host with
no container and no network. That is the right trade, and it has a cost nothing
was paying: the manifest is the authority and nothing checked the copy against
it. Eight of the forty-four curated ids did not exist -- `workshop`,
`dancing_hall`, `garden_nook` and five more -- so about one L2 clip in five
would have died at RENDER time, hundreds of clips into a run, with a `KeyError`
raised from inside Kubric's asset source.

`bash docker/kubric.sh scripts/refresh_hdri_ids.py` is the check against the
live manifest; it needs the container, so it cannot run here. What CAN be
checked here is everything else, and it is what the bug looked like from the
host: a well-formed list of plausible strings.
"""
import pytest

from physloc import scenarios
from physloc.scenarios import TIERS
from physloc.scenarios._hdri import HDRI_IDS
from physloc.scenarios.base import COMPLEXITY

NAMES = sorted(scenarios.available())
SEED = 777


def test_the_id_list_is_well_formed():
    assert len(HDRI_IDS) >= 24, "too few environments to vary a release"
    assert len(set(HDRI_IDS)) == len(HDRI_IDS), "duplicate ids"
    for i in HDRI_IDS:
        assert i and i == i.strip().lower(), i
        assert " " not in i and "/" not in i, i


@pytest.mark.parametrize("name", NAMES)
def test_only_the_hdri_levels_get_an_environment(name):
    """`background` decides, and nothing else. It used to be decided in each
    scenario -- thirteen identical copies of one conditional -- which is a
    property of the level living in thirteen files."""
    sc = scenarios.get(name)
    for level, cx in COMPLEXITY.items():
        if not cx.implemented:
            continue
        spec = sc.sample(SEED, TIERS["debug"], level)
        got = spec.hdri_id
        if cx.background == "hdri":
            assert got in HDRI_IDS, "%s at %s drew %r" % (name, level, got)
        else:
            assert got is None, "%s at %s drew %r on a solid background" % (
                name, level, got)


def test_the_environment_varies_across_scenarios_and_seeds():
    """Salted by scenario, like every other appearance draw.

    All thirteen copies salted the stream with the bare word "hdri", so every
    scenario picked the SAME environment on a given seed and a release shipped
    one backdrop per seed repeated thirteen times. Measured: seed 777 gave
    `killesberg_park` to all thirteen.
    """
    level = next((k for k, v in COMPLEXITY.items()
                  if v.background == "hdri" and v.implemented), None)
    if level is None:
        pytest.skip("no HDRI level is built")
    across = {n: scenarios.get(n).sample(SEED, TIERS["debug"], level).hdri_id
              for n in NAMES}
    assert len(set(across.values())) >= len(NAMES) // 2, across
    seeds = {s: scenarios.get("drop").sample(s, TIERS["debug"], level).hdri_id
             for s in range(12)}
    assert len(set(seeds.values())) >= 8, seeds


def test_choosing_an_environment_does_not_shift_a_physics_draw():
    """Its own salted stream. `pick_hdri(rng)` once drew from the PHYSICS
    stream, and because it only fires on the realistic level the extra draw
    shifted every physics value after it -- so the two levels were independent
    releases wearing the same seed rather than a pair."""
    level = next((k for k, v in COMPLEXITY.items()
                  if v.background == "hdri" and v.implemented), None)
    if level is None:
        pytest.skip("no HDRI level is built")
    for name in NAMES:
        sc = scenarios.get(name)
        plain = sc.sample(SEED, TIERS["debug"], "L1")
        lit = sc.sample(SEED, TIERS["debug"], level)
        assert len(plain.bodies) == len(lit.bodies), name
        for x, y in zip(plain.bodies, lit.bodies):
            for f in ("kind", "position", "scale", "mass", "friction",
                      "restitution", "velocity", "material"):
                assert getattr(x, f) == getattr(y, f), (
                    "%s: %s.%s moved when the environment arrived"
                    % (name, x.name, f))

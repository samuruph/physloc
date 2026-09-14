"""`rolling_ramp` slides a visible stretch of a longer ramp before its lip.

It used to start 0.55-0.78 m above the lip at ~2 m/s and leave the slab in the
opening frames, so every contact-phase violation had almost nothing to act on.
"""
from __future__ import annotations

from physloc import scenarios
from physloc.scenarios import TIERS

TIER = TIERS["debug"]


def test_rolling_ramp_starts_near_the_top():
    sc = scenarios.get("rolling_ramp")
    for seed in range(20):
        n = sc.sample(seed, TIER, "L0").notes
        # The upper third of the slab, measured from its top edge.
        assert n["start_along"] <= -n["half_len"] / 3.0


def test_no_scenario_builds_a_body_kubric_would_reject():
    """Kubric raises for friction or restitution outside [0, 1], at scene build.

    Nothing on the host checked it, so a friction of 1.2 on `rolling_ramp`'s
    block reached the container and failed every job of that scenario before a
    frame was simulated.
    """
    bad = []
    for name in sorted(scenarios.available()):
        sc = scenarios.get(name)
        for tier in ("debug", "release"):
            for seed in range(30):
                spec = sc.sample(seed, TIERS[tier], "L0", variant=seed % 10,
                                 n_variants=10)
                for b in spec.bodies:
                    for field in ("friction", "restitution"):
                        value = float(getattr(b, field))
                        if not 0.0 <= value <= 1.0:
                            bad.append((name, tier, seed, b.name, field, value))
    assert not bad, bad[:10]


def test_the_ramp_is_longer():
    n = scenarios.get("rolling_ramp").sample(0, TIER, "L0").notes
    assert n["half_len"] >= 1.7

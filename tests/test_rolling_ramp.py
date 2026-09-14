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


def test_the_ramp_is_longer():
    n = scenarios.get("rolling_ramp").sample(0, TIER, "L0").notes
    assert n["half_len"] >= 1.7

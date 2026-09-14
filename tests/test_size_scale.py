"""Object size: one multiplier per scene on every scenario's own size draw.

Sizes barely varied within a scenario -- each drew from a band a tenth wide --
so two clips of one scenario never differed much in how big the object was.
"""
from __future__ import annotations

import numpy as np
import pytest

from physloc import scenarios
from physloc.scenarios import _common as C
from physloc.scenarios import TIERS

TIER = TIERS["debug"]
SIZED = sorted(n for n in scenarios.available() if n != "pour")


def test_size_scale_is_deterministic_and_in_band():
    lo, hi = C.SIZE_SCALE
    draws = [C.size_scale(s, "drop") for s in range(200)]
    assert draws == [C.size_scale(s, "drop") for s in range(200)]
    assert all(lo <= d <= hi for d in draws)
    # Actually varies, and spans most of the band rather than a sliver of it.
    assert np.ptp(draws) > 0.8 * (hi - lo)
    # Salted by scenario, so two scenarios on one seed are not the same size.
    assert C.size_scale(7, "drop") != C.size_scale(7, "toss")


def test_size_scale_multiplies_the_size_and_nothing_else(monkeypatch):
    seed = 11
    base = scenarios.get("drop").sample(seed, TIER, "L0")
    monkeypatch.setattr(C, "SIZE_SCALE", (1.0, 1.0))
    flat = scenarios.get("drop").sample(seed, TIER, "L0")
    s = base.notes["size_scale"]
    assert base.notes["radius"] == pytest.approx(flat.notes["radius"] * s)
    # The physics stream advanced exactly as before.
    assert base.notes["drop_height"] == flat.notes["drop_height"]
    assert base.notes["restitution"] == flat.notes["restitution"]


@pytest.mark.parametrize("name", SIZED)
def test_every_sized_scenario_records_its_multiplier(name):
    spec = scenarios.get(name).sample(3, TIER, "L0")
    assert "size_scale" in spec.notes


def test_pour_is_exempt():
    spec = scenarios.get("pour").sample(3, TIER, "L0")
    assert "size_scale" not in spec.notes



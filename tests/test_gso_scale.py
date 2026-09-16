"""How L3 scans are sized: by bounding VOLUME, not by their longest axis.

L3 objects came out visibly smaller than the primitives they replaced. A
sphere fills its whole bounding cube; a scan normalised by its longest axis
matches it on one axis only, and across the 140 curated assets the median scan
occupied 51% of the primitive's bounding volume.
"""
from __future__ import annotations

import numpy as np
import pytest

from physloc import scenarios
from physloc.scenarios import base as B
from physloc.scenarios import TIERS
from physloc.scenarios._gso import GSO_ASSETS

TIER = TIERS["debug"]
SIZED = sorted(n for n in scenarios.available() if n != "pour")


def test_scale_ladder_matches_volume_and_caps_elongation():
    ext = (0.3, 0.3, 0.3)
    ratios = []
    for entry in GSO_ASSETS.values():
        lo, hi = (np.asarray(v, np.float64) for v in entry["bounds"])
        dims = hi - lo
        ladder = B.gso_scale_ladder((lo, hi), ext)
        assert len(ladder) == B.GSO_FIT_RUNGS
        assert ladder == sorted(ladder, reverse=True)
        f_long = 0.6 / float(np.max(dims))
        assert ladder[-1] == pytest.approx(f_long)
        assert ladder[0] * float(np.max(dims)) <= B.GSO_MAX_ELONGATION * 0.6 + 1e-9
        ratios.append(ladder[0] ** 3 * float(np.prod(dims)) / 0.6 ** 3)
    # The old rule's median was 0.51; volume matching brings it to ~1 except
    # where the elongation cap binds.
    assert np.median(ratios) > 0.9


def test_granular_media_keep_the_longest_axis_rule():
    lo, hi = GSO_ASSETS[next(iter(GSO_ASSETS))]["bounds"]
    ladder = B.gso_scale_ladder((lo, hi), (0.06,) * 3, granular=True)
    assert ladder == [pytest.approx(0.12 / float(np.max(np.subtract(hi, lo))))]


@pytest.mark.parametrize("name", SIZED)
def test_l3_scans_are_not_smaller_and_do_not_start_interpenetrating(name):
    grew = []
    for seed in range(6):
        prim = scenarios.get(name).sample(seed, TIER, "L2")
        scan = scenarios.get(name).sample(seed, TIER, "L3")
        # PAIRED BY SEGMENTATION ID, not by name: a swapped body is renamed
        # after the scan it became (`cone` -> `3d_dollhouse_swing`), and the id
        # is what identifies the same body across levels anyway.
        by_id = {b.segmentation_id: b for b in prim.bodies}
        was = {id(b): (by_id[b.segmentation_id].centre,
                       by_id[b.segmentation_id].extents)
               for b in scan.bodies if b.segmentation_id in by_id}
        for b in scan.bodies:
            if b.kind != "gso" or b.segmentation_id not in by_id or b.dormant:
                continue
            p = by_id[b.segmentation_id]
            grew.append(float(np.prod(b.extents)) / float(np.prod(p.extents)))
            assert max(b.extents) <= B.GSO_MAX_ELONGATION * max(p.extents) + 1e-6
        swapped = {id(b): None for b in scan.bodies if b.kind == "gso"}
        assert B._gso_clashes(scan, was, swapped) == []
    if grew:
        # Volume, not the longest axis: the old rule sat near 0.5 here.
        assert np.median(grew) > 0.7

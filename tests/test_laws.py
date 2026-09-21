"""The discrete families measure what their bins vary, against their strong bin.

`permanence`, `dissolve`, `fission` and `fusion` used to score exactly 1.000 at
every bin, because their laws were on/off -- gone or not, split or not -- while
the bins vary how long, how abruptly and how far. Each law now scales by that
measured quantity against a reference that is its injector's strong bin, and
these tests pin the two together: a reference that drifted from its injector
would put strong below 1.0 or saturate the weaker bins.
"""
from physloc import injectors
from physloc.residuals import laws


def test_fission_reference_is_the_strong_separation():
    inj = type(injectors.get("fission"))
    assert laws.FISSION_REF_RADII == inj.SEPARATION_BY_BIN["strong"]


def test_fusion_reference_is_the_strong_reach():
    inj = type(injectors.get("fusion"))
    assert laws.FUSION_REF_RADII == inj.MEET_RADII["strong"]


def test_each_discrete_ladder_orders_by_its_reference():
    """Weak's share of the reference is below medium's, which is below
    strong's -- the ordering the audit requires, stated on the knobs."""
    fis = type(injectors.get("fission")).SEPARATION_BY_BIN
    fus = type(injectors.get("fusion")).MEET_RADII
    for table, ref in ((fus, laws.FUSION_REF_RADII),):
        shares = [min(1.0, table[b] / ref) for b in ("weak", "medium", "strong")]
        assert shares[0] < shares[1] < shares[2] == 1.0, shares
    fade = type(injectors.get("dissolve")).FADE_BY_BIN
    assert fade["weak"] > fade["medium"] > fade["strong"]   # quicker is stronger


def test_fission_launches_its_bins_in_order():
    """Weaker bins launch at strong's speed scaled by the square root of their
    separation share, so the impulse climbs weak -> medium -> strong however
    the strong solve came out -- the cap used to tie medium and strong."""
    import numpy as np
    sep = type(injectors.get("fission")).SEPARATION_BY_BIN
    shares = [np.sqrt(sep[b] / sep["strong"]) for b in ("weak", "medium", "strong")]
    assert shares[0] < shares[1] < shares[2] == 1.0

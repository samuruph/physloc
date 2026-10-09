from physloc import taxonomy as t


def test_internally_consistent():
    t.validate_taxonomy()


def test_every_family_has_a_domain_and_a_law():
    for name, fam in t.FAMILIES.items():
        assert fam.domain in t.DOMAINS
        assert fam.law, name
        assert fam.magnitude_unit, name


def test_build_cells_are_all_compatible():
    for scenario, family in t.build_cells():
        assert t.is_compatible(scenario, family, require_build=True)


def test_novelty_claims_match_prior_art_doc():
    """The prior-art breakdown of the 23 families:
    12 map cleanly onto prior art, 3 exist there only as discrete flags where we
    make them continuous, and 9 are genuinely new. A null cross-reference is a
    novelty claim, so the counts must keep agreeing with that table."""
    intphys = [f for f, v in t.FAMILIES.items() if v.intphys2 is not None]
    any_prior = [f for f, v in t.FAMILIES.items()
                 if v.intphys2 is not None or v.likephys is not None]
    fully_new = [f for f, v in t.FAMILIES.items()
                 if v.intphys2 is None and v.likephys is None]

    assert sorted(intphys) == ["colour_shift", "continuity", "dissolve",
                               "fission", "fusion", "immutability",
                               "permanence", "solidity"]
    # +shadow, +shadow_shape and +shadow_inverted, which LikePhys covers as
    # Optical Consistency but IntPhys 2's four principles do not reach, and
    # +time_slip, whose principle LikePhys covers as Temporal Continuity though
    # it stages it as frame shuffling rather than as physics.
    assert len(any_prior) == 15, sorted(any_prior)
    assert len(fully_new) == 8, sorted(fully_new)
    assert len(intphys) + 4 == 12         # "12 map cleanly"
    assert len(any_prior) - len(intphys) - 4 == 3   # "3 discrete-only"
    assert len(t.FAMILIES) == 23


def test_declared_kind_matches_what_injectors_emit():
    """`FAMILIES[f].kind` is what the taxonomy promises a consumer; the plan's
    `kind` is what the clip actually contains. They drifted apart on
    `permanence`, which is declared instant and produces a sustained absence --
    the removal happens at one instant, but the body being gone is an ongoing
    state, which is the whole reason its strong bin never returns.
    """
    import numpy as np
    import mockroll
    from physloc import injectors, scenarios
    from physloc.scenarios import TIERS

    have = set(scenarios.available())
    checked = 0
    for scenario, family in t.build_cells():
        if scenario not in have:
            continue
        sc = scenarios.get(scenario)
        spec = sc.sample(4242, TIERS["debug"], "L0")
        traj = mockroll.roll(spec, sc)
        inj = injectors.get(family)
        inj.window_frames = None
        plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
        if plan is None:
            continue
        assert plan.kind == t.FAMILIES[family].kind, (
            "%s: taxonomy says %r, %s emits %r"
            % (family, t.FAMILIES[family].kind, scenario, plan.kind))
        checked += 1
    assert checked > 50


def test_a_level_without_flat_colours_declares_colour_shift_absent():
    """`colour_shift.available_at` refuses scanned actors; the taxonomy must
    say so for exactly the levels that use them, or the expected count drifts
    from what the worker can make."""
    from physloc.scenarios.base import COMPLEXITY

    gso = {name for name, cx in COMPLEXITY.items() if cx.actor_assets == "gso"}
    declared = {level for fam, level in t.NOT_AT_LEVEL if fam == "colour_shift"}
    assert declared == gso


def test_expected_cells_drop_only_what_is_declared():
    cells = t.build_cells()
    assert t.expected_cells("L0", "standard") == cells
    assert t.expected_cells("L0", "camera") == cells
    for condition in ("multi", "camera+multi"):
        got = set(t.expected_cells("L0", condition))
        assert set(cells) - got == {(s, f) for f, s in t.NOT_IN_MULTI}
    l3 = set(t.expected_cells("L3", "standard"))
    assert set(cells) - l3 == {(s, f) for s, f in cells if f == "colour_shift"}


def test_every_absence_has_a_reason():
    for why in list(t.NOT_IN_MULTI.values()) + list(t.NOT_AT_LEVEL.values()):
        assert why and "%s" not in why

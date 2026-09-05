"""Each cell must depict ONE violation.

The dataset's whole value proposition is that a benchmark can ask "can this
model detect a teleport" and score it on clips where a teleport is the only
thing wrong. That property is not free and it is not stable: it broke four
separate times while the injectors were being written, always the same way --
an injector re-integrates a body, the integrator does not know about the walls,
and a clip labelled `global_gravity` quietly becomes a clip about solidity too.

So it is asserted rather than hoped for. `taxonomy.EXCLUSIVE_LAWS` names the six
laws with a clean zero baseline on a lawful clip and the families entitled to
move each one; every other family must leave them where the valid twin left
them.
"""
import numpy as np
import pytest

import mockroll
from physviol import injectors, scenarios
from physviol.injectors import _geom
from physviol.residuals import laws
from physviol.scenarios import TIERS
from physviol.taxonomy import (EXCLUSIVE_LAWS, ORTHOGONALITY_TOLERANCE,
                               build_cells)
from conftest import REACHABLE_SEEDS, reachable_cell

CELLS = [(s, f) for s, f in build_cells() if s in set(scenarios.available())]
SEED = 4242


def _residual_ctx(spec, plan, body_id):
    ctx = dict(plan.notes)
    ctx.setdefault("surface_top", spec.floor_level)
    ctx.setdefault("sibling_ids", plan.causal_body_ids)
    body = next((b for b in spec.bodies
                 if int(b.segmentation_id) == int(body_id)), None)
    if body is not None and "support_bounds" not in ctx:
        _, _, bounds = _geom.support_plane(spec, body)
        if bounds is not None:
            ctx["support_bounds"] = list(bounds)
    return ctx


@pytest.mark.parametrize("scenario,family", CELLS,
                         ids=["%s.%s" % (s, f) for s, f in CELLS])
def test_cell_moves_only_its_own_law(scenario, family):
    found = reachable_cell(scenario, family, SEED)
    assert found is not None, (
        "%s x %s has no plan on any of %d seeds"
        % (scenario, family, REACHABLE_SEEDS))
    spec, traj, inj, plan = found
    invalid = inj.apply(spec, traj, plan)

    body_id = int(plan.causal_body_ids[0])
    ctx = _residual_ctx(spec, plan, body_id)
    leaks = []
    for law, owners in EXCLUSIVE_LAWS.items():
        if family in owners:
            continue
        fn = laws.get(law)
        before = float(np.max(fn(traj, traj.index_of(body_id), ctx)))
        after = float(np.max(fn(invalid, invalid.index_of(body_id), ctx)))
        if after - before > ORTHOGONALITY_TOLERANCE:
            leaks.append("%s +%.2f" % (law, after - before))
    assert not leaks, "%s x %s also trips: %s" % (scenario, family, ", ".join(leaks))


def test_every_exclusive_law_has_an_owner_that_exists():
    from physviol.taxonomy import FAMILIES
    for law, owners in EXCLUSIVE_LAWS.items():
        assert law in laws.available(), law
        for fam in owners:
            assert fam in FAMILIES, "%s owns %s but is not a family" % (fam, law)


def test_owners_actually_move_the_law_they_own():
    """The other half of the contract: a tripwire nobody trips is not a test.

    If `penetration` never fired on a `solidity` clip the orthogonality check
    above would pass trivially, and the whole thing would be measuring nothing.
    """
    checked = 0
    for law, owners in EXCLUSIVE_LAWS.items():
        for family in owners:
            cells = [c for c in CELLS if c[1] == family]
            if not cells:
                continue
            moved = False
            for scenario, _ in cells:
                sc = scenarios.get(scenario)
                spec = sc.sample(SEED, TIERS["debug"], "L0")
                traj = mockroll.roll(spec, sc)
                inj = injectors.get(family)
                inj.window_frames = None
                plan = inj.plan(spec, traj,
                                np.random.RandomState(SEED + 7919), "strong")
                if plan is None:
                    continue
                invalid = inj.apply(spec, traj, plan)
                bid = int(plan.causal_body_ids[0])
                ctx = _residual_ctx(spec, plan, bid)
                fn = laws.get(law)
                before = float(np.max(fn(traj, traj.index_of(bid), ctx)))
                after = float(np.max(fn(invalid, invalid.index_of(bid), ctx)))
                if after - before > 0.1:
                    moved = True
                    break
            assert moved, "%s owns %s but never moves it" % (family, law)
            checked += 1
    assert checked >= 6

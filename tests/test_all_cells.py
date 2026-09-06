"""Every build cell plans, applies and keeps its prefix -- without docker.

The coverage check that runs in a second instead of an hour. It cannot say
whether a violation looks right; it says that no cell is dead, that no injector
crashes on a scenario it has never seen, and that the four structural
guarantees hold everywhere: a plan exists, its windows are legal, the invalid
trajectory is bit-identical before `t_event`, and something actually changed
after it.
"""
import os

import numpy as np
import pytest

from physviol import injectors, scenarios
from physviol.sim.trajectory import prefix_identical
from physviol.taxonomy import SEVERITY_BINS, build_cells
from conftest import REACHABLE_SEEDS, reachable_cell, reachable_ladder

CELLS = [(s, f) for s, f in build_cells() if s in set(scenarios.available())]
SEED = 4242

#: Seeds tried per cell at release geometry. Small on purpose -- a mock
#: rollout at 89 frames is what costs, and a healthy cell builds on the first.
V0_SEEDS = 3


def _prepare(scenario_name, family, severity="strong"):
    """A built cell, or a hard failure naming the seeds that declined.

    Not every cell exists on every seed: several scenarios draw their actor's
    shape per seed, and a family may decline the draw -- `angular_momentum`
    declines a sphere, because an untextured sphere's rotation is invisible.
    That is a property of the sample, not a broken cell, so the helper walks
    seeds. Declining ALL of them is the real failure, and is what the assert
    reports.
    """
    found = reachable_cell(scenario_name, family, SEED, severity)
    assert found is not None, (
        "%s x %s produced no plan on any of %d seeds"
        % (scenario_name, family, REACHABLE_SEEDS))
    return found


def test_every_family_has_an_injector():
    from physviol.taxonomy import FAMILIES
    assert set(FAMILIES) == set(injectors.available())


def test_all_build_scenarios_exist():
    missing = sorted({s for s, _ in build_cells()} - set(scenarios.available()))
    # clutter_toss is DEFER-only in the matrix, so it must not appear here.
    assert missing == [], "build cells reference unimplemented scenarios: %s" % missing


@pytest.mark.parametrize("scenario,family", CELLS,
                         ids=["%s.%s" % (s, f) for s, f in CELLS])
def test_cell_plans_and_applies(scenario, family):
    spec, traj, inj, plan = _prepare(scenario, family)

    T = traj.num_frames
    # 0 is legal, and exactly one family uses it: `shadow_inverted` is wrong
    # from the first frame, so its identical prefix is empty. Everything else
    # needs a lawful prefix for the twin structure to mean anything, and the
    # bound below is what stops one going missing by accident.
    assert 0 <= plan.t_event < T, "t_event %d out of range" % plan.t_event
    if plan.t_event == 0:
        assert inj.family in ("shadow_inverted",), (
            "%s claims t_event 0; only families that are wrong from frame 0 "
            "may, and they have to say so here" % inj.family)
    prev = -2
    for s, e in plan.windows:
        assert s <= e and 0 <= s < T and 0 <= e < T, (s, e, T)
        assert s > prev, "windows overlap or are unsorted at (%d,%d)" % (s, e)
        prev = e
    assert plan.causal_body_ids, "no causal bodies"
    for cid in plan.causal_body_ids:
        traj.index_of(int(cid))          # raises if the id is not in the scene

    invalid = inj.apply(spec, traj, plan)
    ok, why = prefix_identical(traj, invalid, plan.t_event)
    assert ok, "%s x %s: %s" % (scenario, family, why)

    changed = any(
        float(np.abs(getattr(invalid, a)[plan.t_event:]
                     - getattr(traj, a)[plan.t_event:]).max()) > 1e-6
        for a in ("pos", "quat", "scale_mul", "colour"))
    changed = changed or not np.array_equal(invalid.present, traj.present)
    assert changed, "%s x %s changed nothing after t_event" % (scenario, family)


@pytest.mark.parametrize("scenario,family", CELLS,
                         ids=["%s.%s" % (s, f) for s, f in CELLS])
def test_cell_magnitude_is_ordered(scenario, family):
    """weak <= medium <= strong, for every cell, on the family's own knob."""
    # ONE seed for all three bins -- see `reachable_ladder`. Comparing a weak
    # plan from one scene against a strong one from another proves nothing.
    plans = reachable_ladder(scenario, family, SEED, SEVERITY_BINS)
    assert plans is not None, (
        "%s x %s has no full severity ladder on any of %d seeds"
        % (scenario, family, REACHABLE_SEEDS))
    mags = [abs(p.magnitude) for p in plans]
    assert mags == sorted(mags), "%s x %s magnitudes not ordered: %s" % (
        scenario, family, mags)


@pytest.mark.parametrize("scenario,family", CELLS,
                         ids=["%s.%s" % (s, f) for s, f in CELLS])
def test_only_declared_culprits_change_appearance(scenario, family):
    """Whatever `apply` makes a body *look* like, `plan` must have declared.

    The invariant that catches plan/apply divergence, which is silent by
    construction: `antigravity` chose its target one way in `plan` and another
    way in `apply`, so it bent a body nobody was annotating and left the
    annotated one alone. The clip came out identical to its valid twin with a
    full set of labels attached, and every existing check passed.

    Narrowed from "no non-culprit array differs" to the appearance channels
    only. The old form asserted that a bystander is never touched at all, and
    that turned out to be the wrong invariant: when an intervention prevents a
    collision, the body that *was* going to be struck has to be re-settled, or
    it departs on schedule hit by nothing. See
    `test_no_body_moves_without_being_touched`, which is the physical claim the
    old test was standing in for.

    Pose and velocity may therefore be settled for a bystander. Size, colour,
    opacity and existence may not: nothing about a collision that failed to
    happen can change what an uninvolved body looks like.
    """
    spec, traj, inj, plan = _prepare(scenario, family)
    invalid = inj.apply(spec, traj, plan)
    declared = {int(i) for i in plan.causal_body_ids}

    restyled, moved = set(), set()
    for j, body in enumerate(spec.bodies):
        bid = int(body.segmentation_id)
        for attr in ("scale_mul", "colour", "opacity"):
            a, b = getattr(traj, attr)[:, j], getattr(invalid, attr)[:, j]
            if float(np.abs(a - b).max()) > 1e-6:
                restyled.add(bid)
        if not np.array_equal(traj.present[:, j], invalid.present[:, j]):
            restyled.add(bid)
        for attr in ("pos", "quat"):
            a, b = getattr(traj, attr)[:, j], getattr(invalid, attr)[:, j]
            if float(np.abs(a - b).max()) > 1e-6:
                moved.add(bid)

    undeclared = sorted(restyled - declared)
    assert not undeclared, (
        "bodies %s changed appearance but are not causal" % undeclared)
    assert restyled or moved, "declared culprits but edited nothing"


@pytest.mark.parametrize("scenario,family", CELLS,
                         ids=["%s.%s" % c for c in CELLS])
def test_no_body_moves_without_being_touched(scenario, family):
    """A body accelerates only if something touches it.

    The physical claim the appearance test above used to stand in for, and the
    one that actually matters. Found in `collision x fission`: the striker
    splits at frame 9 and both halves go elsewhere, yet the target still started
    moving at frame 12 at exactly the speed the original impact would have given
    it -- struck by nothing, in a clip labelled as a fission violation.

    Static geometry does not count as a cause: a floor can hold a body up or
    slow it down, it cannot accelerate one. A resting ball that suddenly departs
    is unexplained even though it has been in contact with the ground the whole
    time, and a detector that accepted any contact reported nothing at all.
    """
    from physviol.injectors import _geom
    from physviol.injectors.base import Injector

    spec, traj, inj, plan = _prepare(scenario, family)
    invalid = inj.apply(spec, traj, plan)

    # An appearance-only family cannot have made anything move without cause,
    # because it made nothing move at all. Checking them exercises only the
    # detector's own false-positive rate against `mockroll`'s crude contacts --
    # which is what the bystander guard used to "fix", displacing a lawfully
    # struck ball by 0.44 m in a clip whose only claim was a colour change.
    if not Injector._changes_dynamics(traj, invalid, plan):
        pytest.skip("%s changes no dynamics" % family)

    culprits = {int(i) for i in plan.causal_body_ids}
    contacts = _geom.geometric_contacts(spec, invalid)

    for body in spec.bodies:
        bid = int(body.segmentation_id)
        if body.static or body.scripted or bid in culprits:
            continue
        bad = np.flatnonzero(Injector._uncaused_frames(
            invalid, spec, contacts, bid, from_frame=max(1, plan.t_event)))
        assert not bad.size, (
            "%s accelerates at frame %d with nothing touching it"
            % (body.name, int(bad[0])))


#: Checks that cost minutes and only matter before a release run. Opt in with
#: `PHYSVIOL_RELEASE_CHECKS=1 pytest tests/`. A guard worth 19 minutes before
#: committing a hundred hours of render is not worth 19 minutes per commit.
RELEASE_CHECKS = os.environ.get("PHYSVIOL_RELEASE_CHECKS") == "1"


@pytest.mark.skipif(not RELEASE_CHECKS,
                    reason="release-geometry sweep; set PHYSVIOL_RELEASE_CHECKS=1")
def test_every_cell_is_reachable_at_release_geometry():
    """Every cell must build at v0's geometry, not only at the debug tier.

    The rest of this file runs at `debug` because that is what makes a full
    sweep take a second instead of an hour. The gap that leaves is real, and it
    shipped a dead cell: `barrier_pass` solved its approach with
    constant-velocity arithmetic on a ball friction was slowing, so the error
    grew with the clip. At 25 frames the ball reached the wall at 0.93 m/s; at
    89 it arrived at 0.26 -- below the speed `_geom.first_impact` will call an
    impact -- and `superelastic x barrier_pass` planned nothing on any seed,
    while every debug-tier test passed.

    A clip length is not a free parameter that only changes the pace. Anything
    integrated over the clip -- friction, drag, a settling pile -- changes
    regime with it, and this is the test that notices.

    One test rather than one per cell, and the rollout is shared by every
    family of a scenario: at 89 frames a mock rollout is the expensive part,
    and re-rolling it 166 times costs minutes.
    """
    from physviol.scenarios import TIERS
    import mockroll

    seeds = range(SEED, SEED + V0_SEEDS)
    dead = []
    for scenario in sorted({s for s, _ in CELLS}):
        sc = scenarios.get(scenario)
        rolls = [(seed, sc.sample(seed, TIERS["v0"], "L0")) for seed in seeds]
        rolls = [(seed, spec, mockroll.roll(spec, sc)) for seed, spec in rolls]
        for fam in sorted({f for s, f in CELLS if s == scenario}):
            inj = injectors.get(fam)
            inj.window_frames = None
            ok = any(inj.plan(spec, traj, np.random.RandomState(seed + 7919),
                              "strong") is not None
                     for seed, spec, traj in rolls)
            if not ok:
                dead.append("%s x %s" % (scenario, fam))
    assert not dead, (
        "these cells build at the debug tier but on none of %d seeds at tier "
        "v0 -- the matrix claims cells the release cannot contain:\n  %s"
        % (V0_SEEDS, "\n  ".join(dead)))


#: Clip lengths a cell must survive. All 4k+1 (the VAE latent stride): the
#: shortest the tiers allow, the debug tier's own 25, and v0's 89. Three rather
#: than a fine sweep because the cost is a mock rollout plus a fit-to-frame
#: ladder per cell per length, and 166 cells at five lengths ran for over half
#: an hour. The ends and the middle are what catch a length-dependent bug.
FRAME_SWEEP = (13, 25, 89)


@pytest.mark.skipif(not RELEASE_CHECKS,
                    reason="clip-length sweep; set PHYSVIOL_RELEASE_CHECKS=1")
def test_every_cell_survives_a_change_of_clip_length():
    """A cell must plan at any clip length, not just the two we ship.

    `num_frames` is a config value, and every timing decision downstream of it
    has to be a FRACTION of the clip rather than a frame count -- otherwise a
    violation that fires a third of the way into a 25-frame clip fires in the
    opening moments of an 89-frame one, or past the end of a 13-frame one.

    This is not hypothetical. Two bugs found at v0 geometry were exactly this
    shape: `barrier_pass` sized its approach with arithmetic that ignored
    friction, so the ball arrived too slowly to count as an impact once the
    clip got long; and `unoccluded_event_frame` computed the event fraction
    itself instead of going through `default_event_frame`, so it fired on the
    same frame in every clip. Both passed every debug-tier test.

    Asserts only what must be true at every length: the cell still builds, the
    event leaves a lawful prefix, and no window runs off either end.
    """
    from physviol.scenarios import TIERS
    import mockroll

    bad = []
    for scenario in sorted({s for s, _ in CELLS}):
        sc = scenarios.get(scenario)
        for frames in FRAME_SWEEP:
            tier = TIERS["v0"].override(num_frames=frames)
            spec = sc.sample(SEED, tier, "L0")
            traj = mockroll.roll(spec, sc)
            T = traj.num_frames
            for fam in sorted({f for s, f in CELLS if s == scenario}):
                inj = injectors.get(fam)
                inj.window_frames = None
                plan = inj.plan(spec, traj, np.random.RandomState(SEED + 7919),
                                "strong")
                if plan is None:
                    continue          # a declined sample is not a length bug
                where = "%s x %s @ %df" % (scenario, fam, frames)
                if not (0 <= plan.t_event < T):
                    bad.append("%s: t_event %d outside [0,%d)"
                               % (where, plan.t_event, T))
                for s_, e_ in plan.windows:
                    if not (0 <= s_ <= e_ < T):
                        bad.append("%s: window (%d,%d) outside [0,%d)"
                                   % (where, s_, e_, T))
                        break
    assert not bad, "clip length breaks these cells:\n  " + "\n  ".join(bad)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_CANDIDATES = ["out/sev_medium/drop/91731", "out/work/drop/91731"]
RELEASE_CANDIDATES = ["out/release_medium", "out/release"]


def find_workdir():
    for c in WORK_CANDIDATES:
        p = os.path.join(REPO, c)
        if os.path.exists(os.path.join(p, "plan.json")):
            return p
    return None


def find_release():
    import glob
    for c in RELEASE_CANDIDATES:
        p = os.path.join(REPO, c)
        if glob.glob(os.path.join(p, "clips", "**", "meta.json"), recursive=True):
            return p
    return None


#: Seeds tried before a cell is declared unbuildable, when a family declines the
#: first one.
REACHABLE_SEEDS = 12


def reachable_cell(scenario_name, family, seed, severity="strong"):
    """(spec, traj, injector, plan) on a seed this cell can be built on.

    A family may decline a *sample* without the cell being wrong. Several
    scenarios draw their actor's shape per seed, and `angular_momentum` declines
    a sphere because untextured primitives make its violation invisible -- so
    `toss x angular_momentum` is a real cell that happens not to exist on seed
    777.

    Returns None only when every seed tried declines, which IS a failure: the
    compatibility matrix would be claiming a cell that cannot be built.

    Lives here rather than in one test file because three of them need it, and
    the first two copies were written separately -- which is how
    `test_orthogonality` came to be the one place still asserting a plan exists
    on the first seed, and the one place that failed.
    """
    import numpy as np
    import mockroll
    from physloc import injectors, scenarios
    from physloc.scenarios import TIERS

    sc = scenarios.get(scenario_name)
    inj = injectors.get(family)
    inj.window_frames = None
    for s in range(int(seed), int(seed) + REACHABLE_SEEDS):
        spec = sc.sample(s, TIERS["debug"], "L0")
        traj = mockroll.roll(spec, sc)
        plan = inj.plan(spec, traj, np.random.RandomState(s + 7919), severity)
        if plan is not None:
            return spec, traj, inj, plan
    return None


def reachable_ladder(scenario_name, family, seed, severities):
    """Plans for every severity in `severities`, all from ONE seed.

    The severity ladder is only meaningful within a single scene: weak, medium
    and strong are the same violation turned up, and their magnitudes are
    compared to each other. Falling back to a fresh seed per bin -- which the
    first version of this did -- silently compares a weak plan on seed 4242
    against a strong one on 4244, two different scenes, and the ordering
    assertion stops meaning anything.

    So a seed counts only if it can build the WHOLE ladder.
    """
    import numpy as np
    import mockroll
    from physloc import injectors, scenarios
    from physloc.scenarios import TIERS

    sc = scenarios.get(scenario_name)
    inj = injectors.get(family)
    inj.window_frames = None
    for s in range(int(seed), int(seed) + REACHABLE_SEEDS):
        spec = sc.sample(s, TIERS["debug"], "L0")
        traj = mockroll.roll(spec, sc)
        plans = [inj.plan(spec, traj, np.random.RandomState(s + 7919), sev)
                 for sev in severities]
        if all(p is not None for p in plans):
            return plans
    return None

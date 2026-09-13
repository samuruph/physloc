"""`generate` runs its most expensive jobs first, without changing any job."""
from physloc import cli


def _job(scenario, level, n_families=10, seed=1, variant=0):
    return (seed, scenario, ["f%d" % i for i in range(n_families)], variant,
            level, 3)


def test_expensive_jobs_come_first():
    jobs = [_job("drop", "L0"), _job("drop", "L2"), _job("pour", "L0"),
            _job("pour", "L3")]
    order = [(j[1], j[4]) for j in cli._longest_first(jobs, "release", 3)]
    assert order[0] == ("pour", "L3")
    assert order[-1] == ("drop", "L0")


def test_the_same_jobs_come_out():
    jobs = [_job("drop", "L0", seed=s) for s in range(5)] + [_job("pour", "L2")]
    assert sorted(cli._longest_first(jobs, "release", 3)) == sorted(jobs)


def test_equal_cost_keeps_emission_order():
    jobs = [_job("drop", "L0", seed=s) for s in (7, 3, 9)]
    assert [j[0] for j in cli._longest_first(jobs, "release", 3)] == [7, 3, 9]

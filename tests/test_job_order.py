"""`generate` runs its most expensive jobs first, without changing any job."""
from physloc import cli


def _job(scenario, level, n_families=10, seed=1, variant=0):
    return (seed, scenario, ["f%d" % i for i in range(n_families)], variant,
            level, 3)


def test_expensive_jobs_come_first_among_the_light_ones():
    jobs = [_job("drop", "L0"), _job("drop", "L2"),
            _job("drop", "L0", n_families=20)]
    order = [(j[1], j[4]) for j in cli._longest_first(jobs, "release", 3)]
    assert order[0] == ("drop", "L2")
    assert order[-1] == ("drop", "L0")


def test_memory_heavy_jobs_run_last():
    """Two L3 pour jobs at ~47 GB each once held half the machine for hours."""
    assert cli.job_memory_gb("pour", "release", "L3") >= cli.HEAVY_JOB_GB
    jobs = [_job("pour", "L3"), _job("drop", "L0"), _job("drop", "L2")]
    order = [(j[1], j[4]) for j in cli._longest_first(jobs, "release", 3)]
    assert order[-1] == ("pour", "L3")
    assert order[0] == ("drop", "L2")


def test_the_same_jobs_come_out():
    jobs = [_job("drop", "L0", seed=s) for s in range(5)] + [_job("pour", "L2")]
    assert sorted(cli._longest_first(jobs, "release", 3)) == sorted(jobs)


def test_equal_cost_keeps_emission_order():
    jobs = [_job("drop", "L0", seed=s) for s in (7, 3, 9)]
    assert [j[0] for j in cli._longest_first(jobs, "release", 3)] == [7, 3, 9]


def test_release_pour_reservations_are_scheduled_as_heavy():
    for level in ("L0", "L1", "L2", "L3"):
        jobs = [_job("pour", level), _job("drop", "L0")]
        assert cli._longest_first(jobs, "release", 3)[-1][1] == "pour"

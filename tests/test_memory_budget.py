"""`MemoryBudget` admits jobs by measured peak memory, first in first out."""
import threading
import time

from physloc import cli


def test_acquire_and_release_restore_the_budget():
    b = cli.MemoryBudget(10)
    got = b.acquire(4)
    assert got == 4 and b.free == 6
    b.release(got)
    assert b.free == 10


def test_a_job_larger_than_the_budget_runs_alone_rather_than_never():
    b = cli.MemoryBudget(8)
    got = b.acquire(50)
    assert got == 8 and b.free == 0
    b.release(got)
    assert b.free == 8


def test_a_big_job_is_not_starved_by_small_ones():
    """The head of the queue blocks later arrivals, so 20 GB is reached."""
    b = cli.MemoryBudget(24)
    order = []
    first = b.acquire(10)                      # a small job already running

    def big():
        g = b.acquire(20)
        order.append("big")
        b.release(g)

    def small():
        g = b.acquire(2)
        order.append("small")
        b.release(g)

    tb = threading.Thread(target=big)
    tb.start()
    time.sleep(0.05)                           # big is queued first
    ts = threading.Thread(target=small)
    ts.start()
    time.sleep(0.05)
    assert order == []                         # small may not jump the queue
    b.release(first)
    tb.join(2)
    ts.join(2)
    assert order == ["big", "small"]


def test_pour_is_the_outlier_only_at_the_scanned_level():
    """96 grains become 96 scanned meshes at L3; below it pour is ordinary-sized."""
    assert cli.job_memory_gb("pour", "debug", "L3") > 5 * cli.job_memory_gb("pour", "debug", "L0")
    assert cli.job_memory_gb("pour", "debug", "L3") > 5 * cli.job_memory_gb("drop", "debug", "L3")


def test_ordinary_jobs_are_charged_typical_not_peak_memory():
    """Charging every job its peak left most of the pool idle; see DEFAULT_JOB_MEMORY_GB."""
    for tier in ("debug", "release"):
        assert cli.job_memory_gb("drop", tier, "L0") <= 1.0


def test_release_is_never_priced_below_debug():
    for level in ("L0", "L1", "L2", "L3"):
        assert (cli.job_memory_gb("pour", "release", level)
                >= cli.job_memory_gb("pour", "debug", level))

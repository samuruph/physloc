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


def _race(budget, need_head, need_small, settle=0.05):
    """A head that does not fit, then a small job that does. Returns start order."""
    order = []

    def job(name, gb):
        g = budget.acquire(gb)
        order.append(name)
        return g

    held = {}
    th = threading.Thread(target=lambda: held.setdefault("big", job("big", need_head)))
    th.start()
    time.sleep(settle)
    ts = threading.Thread(target=lambda: held.setdefault("small", job("small", need_small)))
    ts.start()
    return order, held, th, ts


def test_a_small_job_backfills_while_the_head_is_young():
    """The first release run: 77 jobs waited behind one that did not fit."""
    b = cli.MemoryBudget(24, backfill_seconds=60)
    running = b.acquire(10)                                   # 14 GB free
    order, held, th, ts = _race(b, need_head=20, need_small=2)
    ts.join(2)
    assert order == ["small"]                                 # passed the head
    b.release(running)
    b.release(held["small"])
    th.join(2)
    assert order == ["small", "big"]


def test_backfill_stops_once_the_head_has_waited_long_enough():
    """A big job is delayed a bounded time, never starved."""
    b = cli.MemoryBudget(24, backfill_seconds=0.05)
    running = b.acquire(10)
    order, held, th, ts = _race(b, need_head=20, need_small=2, settle=0.15)
    time.sleep(0.1)
    assert order == []                                        # head too old to pass
    b.release(running)
    th.join(2)
    ts.join(2)
    assert order == ["big", "small"]


def test_ordinary_scanned_jobs_are_charged_more_than_the_baseline():
    assert cli.job_memory_gb("collision", "release", "L3") > cli.job_memory_gb("collision", "release", "L0")


def test_pour_is_the_outlier_only_at_the_scanned_level():
    """96 grains become 96 scanned meshes at L3; below it pour is ordinary-sized."""
    assert cli.job_memory_gb("pour", "debug", "L3") > 5 * cli.job_memory_gb("pour", "debug", "L0")
    assert cli.job_memory_gb("pour", "debug", "L3") > 5 * cli.job_memory_gb("drop", "debug", "L3")


def test_ordinary_jobs_have_room_for_their_peak_memory():
    assert cli.job_memory_gb("drop", "debug", "L0") >= 3.0
    assert cli.job_memory_gb("drop", "release", "L0") >= 4.0
    assert cli.job_memory_gb("pour", "release", "L0") >= 20.0


def test_release_is_never_priced_below_debug():
    for level in ("L0", "L1", "L2", "L3"):
        assert (cli.job_memory_gb("pour", "release", level)
                >= cli.job_memory_gb("pour", "debug", level))

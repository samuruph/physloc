"""`_cpu_slots` hands each parallel worker a disjoint core slice and a Blender
thread count -- and never a count of one, which hangs Cycles 2.93."""
import os

import pytest

from physloc import cli


@pytest.fixture
def cores(monkeypatch):
    def use(n):
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(n)),
                            raising=False)
    return use


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get())
    return out


def test_one_worker_keeps_blender_auto(cores):
    cores(32)
    assert _drain(cli._cpu_slots(1)) == [{}]


@pytest.mark.parametrize("n_cores,workers", [(32, 4), (32, 16), (32, 32),
                                             (96, 96), (8, 64)])
def test_never_one_render_thread(cores, n_cores, workers):
    cores(n_cores)
    slots = _drain(cli._cpu_slots(workers))
    assert len(slots) == workers
    assert all(int(s["PHYSLOC_THREADS"]) >= cli.MIN_RENDER_THREADS
               for s in slots)


def test_slices_are_disjoint_and_sized_to_the_worker_count(cores):
    cores(32)
    slots = _drain(cli._cpu_slots(8))
    sets = [set(s["PHYSLOC_CPUSET"].split(",")) for s in slots]
    assert all(len(s) == 4 for s in sets)
    assert len(set().union(*sets)) == 32
    assert all(s["PHYSLOC_THREADS"] == "4" for s in slots)


def test_more_workers_than_cores_is_not_pinned(cores):
    cores(8)
    assert all("PHYSLOC_CPUSET" not in s for s in _drain(cli._cpu_slots(64)))


@pytest.mark.parametrize("n_cores,workers", [(32, 32), (32, 20), (96, 96)])
def test_never_pinned_to_a_single_core(cores, n_cores, workers):
    """A one-core cpuset hangs Cycles exactly like a one-thread render does."""
    cores(n_cores)
    for s in _drain(cli._cpu_slots(workers)):
        if "PHYSLOC_CPUSET" in s:
            assert len(s["PHYSLOC_CPUSET"].split(",")) >= cli.MIN_RENDER_THREADS
        else:
            assert s["PHYSLOC_THREADS"] == str(cli.MIN_RENDER_THREADS)


def test_workers_auto_is_the_core_count(cores):
    cores(96)
    assert cli._workers("auto") == 96
    assert cli._workers("12") == 12

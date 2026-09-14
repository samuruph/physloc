"""Ctrl-C stops a run: the first interrupt stops new jobs and kills the run's
containers without exiting, so the run can report what it kept."""
import os
import signal
import threading
import time

from physloc import cli


def test_first_interrupt_stops_new_jobs_and_kills_the_run_containers(monkeypatch):
    killed = []
    monkeypatch.setattr(cli, "_kill_run_containers",
                        lambda run_id: killed.append(run_id) or 0)
    stopping = threading.Event()
    restore = cli._install_stop_handler(stopping, "test-run")
    try:
        os.kill(os.getpid(), signal.SIGINT)
        deadline = time.monotonic() + 2
        while not stopping.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert stopping.is_set()
        assert killed == ["test-run"]
    finally:
        restore()
    assert signal.getsignal(signal.SIGINT) is not None


def test_the_previous_handlers_come_back():
    before = signal.getsignal(signal.SIGTERM)
    restore = cli._install_stop_handler(threading.Event(), "test-run")
    assert signal.getsignal(signal.SIGTERM) is not before
    restore()
    assert signal.getsignal(signal.SIGTERM) is before


def test_containers_are_labelled_with_their_run():
    script = open(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "docker", "kubric.sh")).read()
    assert "--label physloc=1" in script
    assert "physloc.run=$PHYSLOC_RUN_ID" in script

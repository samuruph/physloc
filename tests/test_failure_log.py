"""A failed job leaves its reason on disk, not only in the terminal.

The first v0 release run lost every failure: the reason went to the terminal,
the terminal went with the instance, and the run's directory held a progress log
saying FAILED 176 times and nothing about why.
"""
from __future__ import annotations

import contextlib
import glob
import io
import os
import tempfile

from physloc import cli


def _generate(monkeypatch, fake_worker, extra_args=(), base_dir=None):
    monkeypatch.setattr(cli, "_run_worker", fake_worker)
    monkeypatch.setattr(cli, "_annotate", lambda *a, **k: iter(()))
    ap, _ = cli._build()
    tmp = base_dir or tempfile.mkdtemp()
    a = ap.parse_args(["generate", "--scenario", "drop", "--family", "solidity",
                       "--seed", "777", "--workdir", os.path.join(tmp, "work"),
                       "--outdir", os.path.join(tmp, "release"),
                       "--no-overlay", "--keep-going", *extra_args])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        a.fn(a)
    return os.path.join(tmp, "release"), buf.getvalue()


def test_a_crashed_worker_writes_its_stderr_to_failures(monkeypatch):
    def crashed(*args, **kwargs):
        return 137, {"stderr": "Traceback (most recent call last):\n"
                               "MemoryError: the whole reason\n"}

    rel, text = _generate(monkeypatch, crashed)
    logs = glob.glob(os.path.join(rel, "failures", "*.log"))
    assert len(logs) == 1, logs
    body = open(logs[0]).read()
    assert "MemoryError: the whole reason" in body
    assert "exit code: 137" in body and "scenario=drop" in body
    assert logs[0] in text, "the summary should point at the log"


def test_a_worker_that_reported_failure_writes_its_report(monkeypatch):
    def reported(*args, **kwargs):
        return 3, {"ok": False, "variants": [
            {"family": "solidity", "ok": False, "error": "plan raised: boom"}]}

    rel, _ = _generate(monkeypatch, reported)
    logs = glob.glob(os.path.join(rel, "failures", "*.log"))
    assert logs and "plan raised: boom" in open(logs[0]).read()


def test_a_clean_run_writes_no_failures(monkeypatch):
    def fine(scenario, seed, tier, family, severity, workdir, **kwargs):
        return 0, {"outdir": workdir, "ok": True,
                   "variants": [{"family": "solidity", "ok": True,
                                 "dir": "/dev/null"}]}

    rel, _ = _generate(monkeypatch, fine)
    assert not glob.glob(os.path.join(rel, "failures", "*.log"))


def test_worker_limit_matches_its_memory_reservation(monkeypatch):
    limits = []

    def worker(*args, **kwargs):
        limits.append(kwargs["env"]["PHYSLOC_MEMORY"])
        return 137, {"stderr": "simulated worker failure"}

    _generate(monkeypatch, worker, ("--tier", "release"))
    assert limits == ["4g"]


def test_successful_resume_clears_the_previous_failure(monkeypatch, tmp_path):
    base = str(tmp_path)
    _generate(monkeypatch, lambda *a, **k: (137, {"stderr": "OOM"}),
              base_dir=base)
    assert glob.glob(os.path.join(base, "release", "failures", "*.log"))

    def success(*args, **kwargs):
        return 0, {"outdir": args[5], "ok": True, "variants": []}

    _generate(monkeypatch, success, ("--resume",), base_dir=base)
    assert not glob.glob(os.path.join(base, "release", "failures", "*.log"))

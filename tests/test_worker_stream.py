"""`_run_worker` streams the container's output instead of collecting it at exit.

The worker announces every render as it finishes, and the progress bar only
moves render by render if those lines reach the host while the container is
still running. The container is replaced here by a stand-in `docker/kubric.sh`
that prints what a worker prints, so no docker is needed.
"""
import json
import os
import stat
import time

from physloc import cli

FAKE_KUBRIC = """#!/usr/bin/env bash
echo "Blender noise on stdout"
echo "PHYSLOC_RENDERED valid"
sleep 0.4
echo "PHYSLOC_NOT_RENDERED colour_shift/strong"
echo "some stderr chatter" >&2
echo "PHYSLOC_RENDERED solidity/strong"
echo 'PHASE0 %s'
"""


def _fake_repo(tmp_path, phase0):
    docker = tmp_path / "docker"
    docker.mkdir()
    script = docker / "kubric.sh"
    script.write_text(FAKE_KUBRIC % json.dumps(phase0))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(tmp_path)


def test_render_lines_arrive_while_the_worker_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "REPO", _fake_repo(tmp_path, {"ok": True, "outdir": "x"}))
    seen = []
    rc, info = cli._run_worker("drop", 777, "debug", "solidity", "strong",
                               str(tmp_path / "work"),
                               on_line=lambda line: seen.append((time.monotonic(), line)))
    assert rc == 0 and info["outdir"] == "x"
    assert [line for _, line in seen] == [
        "PHYSLOC_RENDERED valid",
        "PHYSLOC_NOT_RENDERED colour_shift/strong",
        "PHYSLOC_RENDERED solidity/strong"]
    # The first announcement landed before the worker's 0.4 s pause ended, so it
    # was delivered live rather than read back when the process exited.
    assert seen[1][0] - seen[0][0] > 0.3


def test_a_failed_worker_still_returns_its_stderr(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    docker = repo / "docker"
    docker.mkdir()
    script = docker / "kubric.sh"
    script.write_text("#!/usr/bin/env bash\necho 'Traceback: boom' >&2\nexit 7\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(cli, "REPO", str(repo))
    rc, info = cli._run_worker("drop", 777, "debug", "solidity", "strong",
                               str(tmp_path / "work"), on_line=lambda line: None)
    assert rc == 7 and "Traceback: boom" in info["stderr"]


def test_a_broken_progress_callback_does_not_fail_the_job(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "REPO", _fake_repo(tmp_path, {"ok": True}))

    def explode(line):
        raise RuntimeError("progress bug")

    rc, _ = cli._run_worker("drop", 777, "debug", "solidity", "strong",
                            str(tmp_path / "work"), on_line=explode)
    assert rc == 0

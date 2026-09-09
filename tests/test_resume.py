"""`--resume` must never hand back a clip that does not match what was asked.

A resume is only safe if it is CONSERVATIVE in exactly one direction: it may
redo work that was already fine (wasteful, harmless) and it must never reuse
work that was made under different settings (silent, and it ships a mixed
release). These tests pin that asymmetry, and the file-existence check that
stops a ledger from outliving the clips it describes.

The helpers under test are closures inside `cmd_generate`, so they are exercised
here through a stand-in that mirrors them exactly -- same request dict, same
staleness rules. `test_request_covers_every_output_changing_dial` is the guard
that keeps the two from drifting: it reads the real source.
"""
import json
import os

import pytest


# --------------------------------------------------------------------------
# A faithful stand-in for the closures in cmd_generate.
# --------------------------------------------------------------------------
def make_ledger(root, request_of):
    ledger_dir = os.path.join(root, ".jobs")

    def path(job):
        return os.path.join(ledger_dir, "%s_%s_%d_v%d.json"
                            % (job["level"], job["scenario"], job["seed"],
                               job["variant"]))

    def save(job, outcome, clips):
        os.makedirs(ledger_dir, exist_ok=True)
        p = path(job)
        with open(p + ".tmp", "w") as fh:
            json.dump({"request": request_of(job), "clips": clips,
                       "outcome": outcome}, fh, default=str)
        os.replace(p + ".tmp", p)

    def load(job):
        try:
            with open(path(job)) as fh:
                entry = json.load(fh)
        except (OSError, ValueError):
            return None
        if entry.get("request") != request_of(job):
            return None
        for clip in entry.get("clips", []):
            if not os.path.exists(os.path.join(clip, "meta.json")):
                return None
        return entry.get("outcome")

    return save, load


JOB = {"level": "L0", "scenario": "drop", "seed": 777, "variant": 0}


def _clip(tmp_path, name="invalid_solidity_strong"):
    d = tmp_path / "clips" / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text("{}")
    return str(d)


def test_a_matching_job_is_resumed(tmp_path):
    req = lambda j: {"families": ["solidity"], "severity": "all"}
    save, load = make_ledger(str(tmp_path), req)
    clip = _clip(tmp_path)
    save(JOB, {"rc": 0, "scenario": "drop"}, [clip])
    assert load(JOB) == {"rc": 0, "scenario": "drop"}


def test_a_changed_request_is_not_resumed(tmp_path):
    """The whole safety property: different settings means rebuild."""
    state = {"families": ["solidity"]}
    req = lambda j: dict(state)
    save, load = make_ledger(str(tmp_path), req)
    save(JOB, {"rc": 0}, [_clip(tmp_path)])
    assert load(JOB) is not None

    state["families"] = ["solidity", "permanence"]      # asked for more
    assert load(JOB) is None, "a changed family list must force a rebuild"


def test_a_ledger_whose_clips_are_gone_is_not_resumed(tmp_path):
    """An entry is a CLAIM about files, and the claim is verified.

    A half-deleted outdir, an interrupted rsync or a full disk all leave the
    ledger describing clips that are not there. Trusting it produces a release
    with holes that `validate` only finds days later.
    """
    req = lambda j: {"families": ["solidity"]}
    save, load = make_ledger(str(tmp_path), req)
    clip = _clip(tmp_path)
    save(JOB, {"rc": 0}, [clip])
    assert load(JOB) is not None

    os.remove(os.path.join(clip, "meta.json"))
    assert load(JOB) is None


def test_a_corrupt_ledger_is_ignored_not_fatal(tmp_path):
    req = lambda j: {"families": ["solidity"]}
    save, load = make_ledger(str(tmp_path), req)
    save(JOB, {"rc": 0}, [_clip(tmp_path)])
    p = os.path.join(str(tmp_path), ".jobs",
                     "L0_drop_777_v0.json")
    with open(p, "w") as fh:
        fh.write("{ this is not json")
    assert load(JOB) is None


def test_a_missing_ledger_is_simply_absent(tmp_path):
    req = lambda j: {"families": ["solidity"]}
    _save, load = make_ledger(str(tmp_path), req)
    assert load(JOB) is None


# --------------------------------------------------------------------------
# The guard that keeps the stand-in honest.
# --------------------------------------------------------------------------
def test_request_covers_every_output_changing_dial():
    """A dial that changes pixels but is absent from the request is a bug.

    If someone adds a render knob and forgets it here, `--resume` will keep
    clips made at the old setting and report them as new ones -- the exact
    failure mode this whole mechanism exists to avoid. So the request dict is
    asserted to mention each one by name.
    """
    src = open(os.path.join(os.path.dirname(__file__), "..", "physloc",
                            "cli.py")).read()
    start = src.index("def _ledger_request(job):")
    body = src[start:src.index("def _ledger_path(job):")]
    for dial in ("scenario", "seed", "level", "variant", "families",
                 "severity", "tier", "window", "overlay", "resolution",
                 "fps", "frames", "spp",
                 "PHYSLOC_ADAPTIVE", "PHYSLOC_DENOISER", "PHYSLOC_GPU",
                 "PHYSLOC_IMAGE"):
        assert dial in body, (
            "%s changes a clip but is not in the resume request -- a resume "
            "would reuse clips made at a different setting" % dial)


def test_only_completed_jobs_are_written():
    """A failed job must leave no ledger entry, or a retry would be skipped."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "physloc",
                            "cli.py")).read()
    start = src.index("def _ledger_save(job, outcome):")
    body = src[start:start + 400]
    assert 'outcome.get("rc") != 0' in body and "return" in body

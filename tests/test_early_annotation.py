"""A clip is annotated as soon as its render finishes, not when its job ends.

A release job renders about fifty clips over fifteen to twenty hours; videos
used to appear only at the end of it. `generate` is driven here with the
container stubbed out: the fake worker announces each render the way the real
one does, and `_annotate` records what it was asked to do.
"""
import contextlib
import io
import os
import tempfile
import threading

from physloc import cli

FAMILIES = ("continuity", "solidity")


def _generate(monkeypatch, fail_first=False):
    calls = []                       # (variant dirs, thread name)
    lock = threading.Lock()
    failed = []

    def fake_annotate(workdir, outroot, overlay=True, only=None):
        assert only, "an empty `only` would annotate every variant"
        with lock:
            calls.append((tuple(only), threading.current_thread().name))
            if fail_first and not failed:
                failed.append(only[0])
                raise RuntimeError("transient")
        # The record shape `annotate_pair` returns: `generate`'s end-of-run
        # summary reads every one of these fields.
        out = []
        for d in only:
            family, severity = os.path.basename(d).rsplit("_", 1)
            out.append({"samples": {"invalid": os.path.join(outroot, os.path.basename(d))},
                        "family": family, "severity": severity, "t_event": 8,
                        "observability_lag": 0, "violation_windows": [[8, 12]],
                        "peak_severity": 1.0, "peak_score": 1.0})
        return out

    def fake_worker(scenario, seed, tier, family, severity, workdir,
                    complexity="L0", window=None, dials=None, variant=0,
                    n_variants=None, params_path=None, env=None, on_line=None):
        job_dir = os.path.join(workdir, scenario, "%04d" % seed)
        on_line("PHYSLOC_RENDERED valid")
        variants = []
        for fam in family.split(","):
            on_line("PHYSLOC_RENDERED %s/strong" % fam)
            variants.append({"family": fam, "severity": "strong", "ok": True,
                             "dir": os.path.join(job_dir, "variants", "%s_strong" % fam)})
        return 0, {"outdir": job_dir, "variants": variants}

    monkeypatch.setattr(cli, "_run_worker", fake_worker)
    monkeypatch.setattr(cli, "_annotate", fake_annotate)
    ap, _ = cli._build()
    with tempfile.TemporaryDirectory() as tmp:
        a = ap.parse_args(["generate", "--scenario", "drop", "--family", ",".join(FAMILIES),
                           "--severity", "strong", "--variants", "1", "--workers", "1",
                           "--workdir", os.path.join(tmp, "work"),
                           "--outdir", os.path.join(tmp, "release"), "--keep-going"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = a.fn(a)
    return rc, calls


def test_each_clip_is_annotated_once_as_it_renders(monkeypatch):
    rc, calls = _generate(monkeypatch)
    assert rc == 0
    annotated = [os.path.basename(d) for only, _ in calls for d in only]
    assert sorted(annotated) == sorted("%s_strong" % f for f in FAMILIES)
    # One clip per call, on the annotator's own thread -- not a batch at job end.
    assert all(len(only) == 1 for only, _ in calls)
    assert all(name == "sample-annotator" for _, name in calls)


def test_a_clip_that_fails_early_is_annotated_when_its_job_ends(monkeypatch):
    rc, calls = _generate(monkeypatch, fail_first=True)
    assert rc == 0
    first = os.path.basename(calls[0][0][0])
    retried = [only for only, name in calls if name != "sample-annotator"]
    assert [os.path.basename(d) for only in retried for d in only] == [first]

"""The SLURM path: one array task per `generate` job (slurm/README.md).

Nothing here needs a cluster, docker or Singularity. It pins the pieces the
array run is built from:

* `generate --list-jobs` lists exactly the jobs a run would execute, by the
  names `--only` and the ledger use, and writes nothing else;
* `--no-finalize` keeps a task from rewriting the shared release, and
  `finalize` does the same work once;
* `PHYSLOC_LAUNCHER` swaps docker/kubric.sh for slurm/kubric_singularity.sh,
  which builds the right Singularity command; docker/kubric.sh is untouched;
* the asset mirror rewrites a manifest and nothing else;
* `slurm/make_job_lists.py` puts every job in a render script that fits it,
  and `slurm/status.py` reads SLURM's verdicts and prescribes the re-send.
"""
import csv
import importlib.util
import os
import shutil
import stat
import subprocess
import sys

import pytest

from physloc import cli

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _list_jobs(tmp_path, *extra, name="jobs.tsv"):
    path = tmp_path / name
    rc = cli.main(["generate", "--config", "v0_release", "--list-jobs", str(path),
                   "--workdir", str(tmp_path / "work"),
                   "--outdir", str(tmp_path / "rel")] + list(extra))
    assert rc == 0
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# ------------------------------------------------------------------ list-jobs
def test_list_jobs_names_every_job_once_and_writes_nothing_else(tmp_path):
    rows = _list_jobs(tmp_path)
    names = [r["name"] for r in rows]
    assert names and len(names) == len(set(names))
    assert list(rows[0]) == list(cli.JOB_LIST_COLUMNS)
    for r in rows:
        assert r["name"] == "%s_%s_%s_v%s" % (r["level"], r["scenario"],
                                              r["seed"], r["variant"])
        assert int(r["renders"]) >= 1 + int(r["families"])
        assert float(r["memory_gb"]) > 0
    # A listing is not a run: no params.json, no workdir, no release root.
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "rel").exists()


def test_list_jobs_is_the_same_at_any_worker_count(tmp_path):
    one = _list_jobs(tmp_path, "--workers", "1", name="a.tsv")
    many = _list_jobs(tmp_path, "--workers", "8", name="b.tsv")
    assert one == many


def test_every_listed_name_is_accepted_by_only(tmp_path):
    rows = _list_jobs(tmp_path)
    for r in (rows[0], rows[len(rows) // 2], rows[-1]):
        only = _list_jobs(tmp_path, "--only", r["name"], name="one.tsv")
        assert [o["name"] for o in only] == [r["name"]]


def test_job_name_is_the_ledger_name():
    job = (20260824, "drop", ["solidity"], 3, "L1", 5)
    assert cli.job_name(job) == "L1_drop_20260824_v3"


# --------------------------------------------------------------- no-finalize
def _failing_run(tmp_path, monkeypatch, *extra):
    """One real `generate --only` whose worker fails at once, so the run goes
    straight to its end -- where finalizing does or does not happen."""
    calls = []
    monkeypatch.setattr(cli, "_run_worker",
                        lambda *a, **k: (4, {"stderr": "stand-in failure"}))
    monkeypatch.setattr(cli, "_finish_release", lambda rel: calls.append(rel))
    name = _list_jobs(tmp_path)[-1]["name"]
    rc = cli.main(["generate", "--config", "v0_release", "--only", name,
                   "--workers", "1", "--workdir", str(tmp_path / "work"),
                   "--outdir", str(tmp_path / "rel")] + list(extra))
    assert rc == 0                      # keep_going: a failed job is reported
    return calls


def test_a_run_finalizes_its_release_by_default(tmp_path, monkeypatch):
    assert _failing_run(tmp_path, monkeypatch) == [str(tmp_path / "rel")]


def test_no_finalize_leaves_the_shared_release_alone(tmp_path, monkeypatch):
    assert _failing_run(tmp_path, monkeypatch, "--no-finalize") == []


def test_finalize_command_refuses_a_root_without_samples(tmp_path):
    assert cli.main(["finalize", str(tmp_path)]) == 2


# ------------------------------------------------------------------ launcher
def _fake_bin(tmp_path, name):
    """A stand-in `name` on PATH that records its argv, one arg per line."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / ("%s.argv" % name)
    exe = bindir / name
    exe.write_text('#!/usr/bin/env bash\nprintf "%%s\\n" "$@" > %s\n' % log)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return bindir, log


def test_generate_runs_the_launcher_slurm_env_names(tmp_path, monkeypatch):
    """PHYSLOC_LAUNCHER, relative to the repo, replaces docker/kubric.sh."""
    script = tmp_path / "slurm" / "kubric_singularity.sh"
    script.parent.mkdir()
    script.write_text('#!/usr/bin/env bash\necho \'PHASE0 {"ok": true, "who": "singularity"}\'\n')
    monkeypatch.setattr(cli, "REPO", str(tmp_path))
    monkeypatch.setenv("PHYSLOC_LAUNCHER", "slurm/kubric_singularity.sh")
    rc, info = cli._run_worker("drop", 7, "debug", "solidity", "strong",
                               str(tmp_path / "work"))
    assert rc == 0 and info["who"] == "singularity"


def _singularity(tmp_path, env):
    bindir, log = _fake_bin(tmp_path, "singularity")
    full = {"PATH": "%s:%s" % (bindir, os.environ["PATH"]), "HOME": str(tmp_path)}
    full.update(env)
    proc = subprocess.run(["bash", os.path.join(REPO, "slurm", "kubric_singularity.sh"),
                           "physloc/render/worker.py", "--seed", "7"],
                          env=full, capture_output=True, text=True)
    argv = log.read_text().splitlines() if log.exists() else []
    return proc, argv


def test_singularity_launcher_binds_forwards_and_isolates(tmp_path):
    sif = tmp_path / "kubric.sif"
    sif.write_text("")
    for d in ("data", "assets"):
        (tmp_path / d).mkdir()
    scratch = tmp_path / "scratch"
    proc, argv = _singularity(tmp_path, {
        "PHYSLOC_SIF": str(sif), "PHYSLOC_THREADS": "4",
        "PHYSLOC_DATA": str(tmp_path / "data"),
        "PHYSLOC_ASSETS": str(tmp_path / "assets"),
        "PHYSLOC_SCRATCH": str(scratch), "PHYSLOC_MEMORY": "40g"})
    assert proc.returncode == 0, proc.stderr
    pairs = list(zip(argv, argv[1:]))
    assert argv[0] == "exec" and "--cleanenv" in argv
    assert ("--pwd", "/kubric") in pairs
    assert ("--bind", "%s:/kubric" % REPO) in pairs
    for d in (tmp_path / "data", tmp_path / "assets", scratch):
        assert ("--bind", str(d)) in pairs
    assert ("--env", "PHYSLOC_THREADS=4") in pairs
    assert ("--env", "PHYSLOC_ASSETS=%s" % (tmp_path / "assets")) in pairs
    assert ("--env", "TMPDIR=%s/tmp" % scratch) in pairs
    assert ("--home", "%s/home" % scratch) in pairs
    # An unset dial is not invented; docker-only knobs do not leak in.
    assert not any(a.startswith("PHYSLOC_DEBUG") for a in argv)
    assert not any("40g" in a for a in argv)
    assert argv[-5:] == [str(sif), "python3", "physloc/render/worker.py",
                         "--seed", "7"]


def test_singularity_launcher_fails_loudly_without_an_image(tmp_path):
    proc, argv = _singularity(tmp_path, {"PHYSLOC_SIF": str(tmp_path / "nope.sif")})
    assert proc.returncode == 66 and argv == []
    assert "PHYSLOC_SIF" in proc.stderr


def test_docker_launcher_knows_nothing_about_the_cluster():
    """The docker path stays exactly as it was: the cluster lives in slurm/."""
    text = open(os.path.join(REPO, "docker", "kubric.sh")).read()
    assert "singularity" not in text.lower()
    assert "docker run" in text


# --------------------------------------------------------------- asset mirror
def _mirror_module():
    path = os.path.join(REPO, "slurm", "setup", "mirror_assets.py")
    spec = importlib.util.spec_from_file_location("mirror_assets", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MANIFEST = {"name": "GSO", "version": "1", "data_dir": "gs://kubric-public/assets/GSO",
            "assets": {"a": {"asset_type": "FileBasedObject",
                             "kwargs": {"render_filename": "{asset_dir}/v.obj"}},
                       "b": {"asset_type": "Texture", "path": "b.tar.gz"},
                       "c": {"asset_type": "Texture", "path": None}}}


def test_mirror_rewrites_data_dir_and_keeps_entries():
    m = _mirror_module()
    out = m.rewrite(MANIFEST, "/work/assets/GSO", keep=["a", "b"])
    assert out["data_dir"] == "/work/assets/GSO"
    assert out["assets"] == {k: MANIFEST["assets"][k] for k in ("a", "b")}
    assert MANIFEST["data_dir"].startswith("gs://")          # input untouched
    assert m.rewrite(MANIFEST, "/x", keep=None)["assets"] == MANIFEST["assets"]


def test_mirror_refuses_ids_the_manifest_lacks():
    with pytest.raises(KeyError):
        _mirror_module().rewrite(MANIFEST, "/x", keep=["a", "zzz"])


def test_mirror_resolves_files_as_kubric_does():
    m = _mirror_module()
    assert m.asset_file("a", MANIFEST["assets"]["a"]) == "a.tar.gz"
    assert m.asset_file("b", MANIFEST["assets"]["b"]) == "b.tar.gz"
    assert m.asset_file("c", MANIFEST["assets"]["c"]) is None


def test_mirror_covers_every_id_the_code_samples():
    ids = _mirror_module().wanted_ids()
    assert ids["KuBasic"] is None and ids["GSO"] and ids["HDRI_haven"]


# ------------------------------------------------------------- job lists
def _slurm_module(name):
    path = os.path.join(REPO, "slurm", name + ".py")
    sys.path.insert(0, os.path.dirname(path))
    try:
        spec = importlib.util.spec_from_file_location("slurm_" + name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def test_make_job_lists_sorts_every_job_into_a_render_script(tmp_path):
    """Run slurm/make_job_lists.py in a scratch copy of the repo layout, so a
    real slurm/jobs/ (on the cluster) is never overwritten by a test."""
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("physloc", "configs"):
        os.symlink(os.path.join(REPO, name), root / name)
    shutil.copytree(os.path.join(REPO, "slurm"), root / "slurm",
                    ignore=shutil.ignore_patterns("jobs", "logs", "__pycache__"))
    env = dict(os.environ, PHYSLOC_CONFIG="v0_release",
               PHYSLOC_OUTDIR=str(tmp_path / "physloc_v0"))
    proc = subprocess.run([sys.executable, str(root / "slurm" / "make_job_lists.py")],
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    jobs = root / "slurm" / "jobs"
    rows = {r["name"]: r for r in
            csv.DictReader(open(jobs / "all.tsv"), delimiter="\t")}
    seen = []
    for list_file in jobs.glob("*.txt"):
        header = open(root / "slurm" / ("render_%s.slurm" % list_file.stem)).read()
        cores = int(header.split("--cpus-per-task=")[1].split()[0])
        t4 = "--qos=qos_cpu-t4" in header
        for name in list_file.read_text().split():
            r = rows[name]
            if cores * 4 < float(r["memory_gb"]) + 2:        # charged more than given:
                assert name in proc.stderr                    # ...said so, not hidden
            long_hdri = (r["scenario"] == "shadow_track" and r["level"] in ("L2", "L3"))
            assert t4 == (r["scenario"] == "pour" or long_hdri)   # only these > 20 h
            assert (cores >= 8) == (r["level"] in ("L2", "L3")
                                    or r["scenario"] == "pour")
            seen.append(name)
    assert sorted(seen) == sorted(rows)                        # every job, once
    assert "sbatch --array=0 slurm/render_4cores.slurm" in proc.stdout


def test_make_job_lists_needs_env_sh():
    env = {k: v for k, v in os.environ.items() if not k.startswith("PHYSLOC_")}
    proc = subprocess.run([sys.executable, os.path.join(REPO, "slurm", "make_job_lists.py")],
                          env=env, capture_output=True, text=True)
    assert proc.returncode != 0 and "source slurm/env.sh" in proc.stderr


# ------------------------------------------------------------------ status
def test_status_reads_squeue_and_sacct_lines():
    st = _slurm_module("status")
    tasks = st.parse_tasks("\n".join([
        "101_3|render-4cores|TIMEOUT",
        "101_[4-9]|render-4cores|PENDING",        # a pending range: skipped
        "102_3|render-4cores|CANCELLED by 42",
        "103|finalize|COMPLETED",                 # not an array task: skipped
        "104_0|render-pour_L3|OUT_OF_MEMORY"]))
    assert tasks[("4cores", 3)] == [(101, "TIMEOUT"), (102, "CANCELLED")]
    assert tasks[("pour_L3", 0)] == [(104, "OUT_OF_MEMORY")]
    assert len(tasks) == 2


@pytest.mark.parametrize("attempts, reason", [
    ([], "never ran"),
    ([(1, "TIMEOUT")], "timeout"),
    ([(1, "FAILED"), (2, "TIMEOUT")], "timeout"),           # the LAST attempt decides
    ([(1, "OUT_OF_MEMORY")], "out of memory"),
    ([(1, "NODE_FAIL")], "interrupted"),
    ([(1, "FAILED")], "failed once"),
    ([(1, "FAILED"), (2, "FAILED")], "failed twice"),
])
def test_status_diagnoses_each_outcome(attempts, reason):
    assert _slurm_module("status").diagnose(attempts) == reason


def test_status_remedies():
    st = _slurm_module("status")
    assert st.remedy("never ran", "4cores") == ""
    assert st.remedy("interrupted", "4cores") == ""
    assert st.remedy("timeout", "4cores") == "--qos=qos_cpu-t4 --time=60:00:00"
    assert st.remedy("timeout", "pour") == "--time=100:00:00"   # already on t4
    assert st.remedy("out of memory", "4cores") == "--cpus-per-task=8"
    l3 = int(st.script_setting("pour_L3", "cpus-per-task"))
    assert st.remedy("out of memory", "pour_L3") == "--cpus-per-task=%d" % min(40, 2 * l3)
    st_cap = st.MAX_CORES
    assert st_cap == 40                                         # one cpu_p1 node
    assert st.remedy("failed twice", "4cores") is None          # investigate instead


def test_status_writes_compact_array_ranges():
    spec = _slurm_module("status").array_spec
    assert spec([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert spec([3]) == "3"
    assert spec(range(180)) == "0-179"


def test_status_reads_slurm_memory_figures():
    gb = _slurm_module("status").parse_kib
    assert abs(gb("33449380K") - 31.9) < 0.1          # sstat's usual form
    assert gb("512M") == 0.5 and gb("2G") == 2.0
    assert gb("") is None


def test_status_reads_durations():
    h = _slurm_module("status").parse_duration
    assert abs(h("16:11:35") - 16.193) < 0.001
    assert h("1-02:00:00") == 26.0
    assert abs(h("05:30.500") - 0.0918) < 0.001        # TotalCPU's short form
    assert h("") is None


def test_status_advises_from_the_pilot():
    st = _slurm_module("status")
    rows = st.finished_tasks_table("\n".join([
        "614405_0|render-pour|COMPLETED|16:11:35|5-00:00:00|",
        "614405_0.batch|batch|COMPLETED|16:11:35|5-00:00:00|6257480K",
        "614406_0|render-8cores|COMPLETED|13:48:34|3-12:00:00|",
        "614406_0.batch|batch|COMPLETED|13:48:34|3-12:00:00|3405788K",
        "614407_0|render-4cores|RUNNING|02:00:00||",          # not finished: skipped
        "608373|setup-image|COMPLETED|00:02:46||"]))          # not a render: skipped
    assert sorted(r["list"] for r in rows) == ["8cores", "pour"]
    pour = st.advise("pour", [r for r in rows if r["list"] == "pour"])
    assert pour[0].endswith("--time=25:00:00")                # 16.2 h x 1.5, rounded up
    eight = st.advise("8cores", [r for r in rows if r["list"] == "8cores"])
    assert "margin 1.45x" in eight[0] and "OK" in eight[0]    # 13.8 h of 20 h
    assert not any("!!" in line for line in eight)


# ------------------------------------------------------------------ archive
def _release(tmp_path):
    rel = tmp_path / "physloc_v0"
    (rel / "samples" / "physloc_v0" / "L0" / "a").mkdir(parents=True)
    (rel / "samples" / "physloc_v0" / "L0" / "a" / "data.h5").write_bytes(b"x" * 100)
    (rel / ".jobs").mkdir()
    (rel / ".jobs" / "L0_a.json").write_text("{}")
    (rel / "empty").mkdir()
    os.symlink("samples", rel / "link")
    return rel


def _tar(path, root, members):
    import tarfile
    with tarfile.open(path, "w") as tf:
        for m in members:
            tf.add(os.path.join(os.path.dirname(root), m), arcname=m)


def test_check_archive_accepts_an_exact_archive(tmp_path):
    chk = _slurm_module("check_archive")
    rel = _release(tmp_path)
    _tar(tmp_path / "all.tar", str(rel), ["physloc_v0"])
    problems, n = chk.compare(str(rel), [str(tmp_path / "all.tar")])
    assert problems == [] and n == 2


def test_check_archive_catches_missing_changed_and_extra(tmp_path):
    chk = _slurm_module("check_archive")
    rel = _release(tmp_path)
    _tar(tmp_path / "all.tar", str(rel), ["physloc_v0"])
    (rel / "new.txt").write_text("added after archiving")                 # missing
    (rel / "samples/physloc_v0/L0/a/data.h5").write_bytes(b"x" * 99)      # size differs
    (rel / ".jobs" / "L0_a.json").unlink()                                 # extra in tar
    problems, _ = chk.compare(str(rel), [str(tmp_path / "all.tar")])
    text = "\n".join(problems)
    assert "missing from the tars: physloc_v0/new.txt" in text
    assert "differs: physloc_v0/samples/physloc_v0/L0/a/data.h5" in text
    assert "in a tar but not in the folder: physloc_v0/.jobs/L0_a.json" in text


def _ledger(level, variant, scenario, seed, families, made, bad=()):
    return {"request": {"level": level, "variant": variant, "scenario": scenario,
                        "seed": seed, "n_variants": 10, "families": families,
                        "severity": "all"},
            "outcome": {"results": [{"family": f, "severity": b} for f, b in made],
                        "bad": [{"family": f, "severity": b, "error": e}
                                for f, b, e in bad]}}


def test_status_counts_missing_samples_from_the_ledgers():
    st = _slurm_module("status")
    every = lambda f: [(f, b) for b in ("weak", "medium", "strong")]
    crash = ("annotate failed: RuntimeError('annotation worker exited 1: Traceback"
             "\\n  File x\\nValueError: affected clock disagrees\\n')")
    ledgers = {
        # L0 variant 8 is `multi`: fission is never owed there, so its decline
        # is not a gap; solidity's strong bin left the frame, then a retry
        # seed built superelastic, which the main job had declined.
        "L0_drop_v8": _ledger("L0", 8, "drop", 8, ["fission", "solidity", "superelastic"],
                              every("solidity")[:2],
                              [("fission", b, "multi clip needs two or more") for b in
                               ("weak", "medium", "strong")]
                              + [("solidity", "strong", "violator leaves the frame after")]
                              + [("superelastic", b, "injector produced no plan")
                                 for b in ("weak", "medium", "strong")]),
        "L0_drop_v8_retry": _ledger("L0", 8, "drop", 18, ["superelastic"],
                                    every("superelastic")),
        "L0_drop_v0": _ledger("L0", 0, "drop", 0, ["solidity"], every("solidity")[1:],
                              [("solidity", "weak", crash)]),
    }
    gaps = st.sample_gaps(ledgers)
    assert gaps[("L0", 8, "drop")]["expected"] == 6           # fission left out
    assert gaps[("L0", 8, "drop")]["missing"] == {
        ("solidity", "strong"): "violator leaves the frame"}
    assert gaps[("L0", 0, "drop")]["missing"] == {
        ("solidity", "weak"): "BUG ValueError: affected clock disagrees"}
    text = "\n".join(st.sample_report(gaps, details=True))
    assert "7 of 9 expected, 2 missing" in text
    assert "L0_drop_v0" in text and "re-send" in text


@pytest.mark.parametrize("attempts, reason", [
    ([(True, False)] * 6, "a body falls through the floor"),
    ([(False, True)] * 6, "violator leaves the frame"),
    ([(False, True), (True, False)], "leaves the frame / falls through floor"),
])
def test_a_decline_names_the_check_it_failed(attempts, reason):
    """The worker's two visibility checks used to share one message, so a body
    sunk into the floor in full view was counted as leaving the frame."""
    from physloc.injectors import _geom
    st = _slurm_module("status")
    made = [{"ok": True, "in_frame": f, "above_floor": a, "visible": f and a}
            for f, a in attempts]
    assert st.reason_class(_geom.unseen_error(made)) == reason

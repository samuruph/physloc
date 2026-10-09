"""How far the release has got, why unfinished jobs did not finish, and the
exact sbatch line to re-send them. Submits nothing itself.

    source slurm/env.sh
    python slurm/status.py            # summary + what to re-send
    python slurm/status.py --details  # ...and one line per unfinished job, and
                                      #    per job with missing samples
    python slurm/status.py --running  # running tasks: time so far, peak memory
    python slurm/status.py --finished # finished tasks, and a --time / cores advice per list

Below the table, the SAMPLES of the done jobs: how many of those the taxonomy
expects are present, and the missing ones counted by reason -- a done job can
still have declined or crashed on some of its cells.

A job is DONE when its ledger entry `$PHYSLOC_OUTDIR/.jobs/<name>.json` exists:
it is written only once the whole job has landed. For the rest, SLURM says
what happened -- `squeue` for what is still queued or running, `sacct` for how
the last attempt ended -- and each outcome has its own remedy:

    never ran / interrupted   (NODE_FAIL, CANCELLED, ...)  re-send as is
    TIMEOUT                   ran out of time              re-send on qos_cpu-t4, longer
    OUT_OF_MEMORY             needed more memory           re-send with more cores
    FAILED once               crashed (see failures/)      re-send once as is
    FAILED twice or more      it is not bad luck           do NOT re-send: read
                                                           $PHYSLOC_OUTDIR/failures/<job>.log
"""
import argparse
import collections
import getpass
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from make_job_lists import (ANNOTATION_GB, JOBS_DIR, LISTS, env,  # noqa: E402
                            script_setting)

INTERRUPTED = {"NODE_FAIL", "CANCELLED", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
               "REQUEUED", "SUSPENDED"}
MAX_CORES = 40                      # one cpu_p1 node
T4_MAX_TIME = "100:00:00"


def _run(cmd):
    """stdout of a SLURM command, or "" where SLURM is not available."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def parse_tasks(text):
    """`<array>_<index>|<job name>|<STATE ...>` lines -> {(list, index): [(array, state)]}.

    Pending ranges (`123_[4-9]`) and anything not from a render script are
    skipped; a state like `CANCELLED by 123` keeps its first word.
    """
    out = collections.defaultdict(list)
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        m = re.match(r"^(\d+)_(\d+)$", parts[0])
        if not m or not parts[1].startswith("render-"):
            continue
        key = (parts[1][len("render-"):], int(m.group(2)))
        out[key].append((int(m.group(1)), parts[2].split()[0]))
    return out


def queued_tasks():
    return parse_tasks(_run(["squeue", "-h", "-r", "-u", getpass.getuser(),
                             "-o", "%i|%j|%T"]))


def finished_tasks(days=30):
    return parse_tasks(_run(["sacct", "-n", "-X", "-P", "-u", getpass.getuser(),
                             "-S", "now-%ddays" % days, "-o", "JobID,JobName,State"]))


def parse_kib(value):
    """A SLURM memory figure (`2433616K`, `33.4G`, `512M`) in GB, or None."""
    m = re.match(r"^([\d.]+)([KMGT]?)$", value.strip())
    if not m:
        return None
    scale = {"": 1 / 1024 ** 3, "K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1.0,
             "T": 1024.0}[m.group(2)]
    return float(m.group(1)) * scale


def parse_duration(text):
    """SLURM durations (`1-02:03:04`, `16:11:35`, `05:12.345`) in hours, or None."""
    text = text.strip()
    if not text:
        return None
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [float(x) for x in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts
    return days * 24 + h + m / 60 + sec / 3600


def finished_tasks_table(text):
    """sacct -P lines (job + its .batch step) -> one dict per finished render task."""
    tasks, batch = {}, {}
    for line in text.splitlines():
        p = (line.strip().split("|") + [""] * 6)[:6]
        job_id, name, state, elapsed, total_cpu, rss = p
        if job_id.endswith(".batch"):
            batch[job_id[:-len(".batch")]] = rss
            continue
        if not name.startswith("render-") or "." in job_id \
                or not re.match(r"^\d+_\d+$", job_id):
            continue
        tasks[job_id] = {"list": name[len("render-"):], "task": job_id,
                         "state": state.split()[0] if state else "",
                         "hours": parse_duration(elapsed),
                         "cpu_hours": parse_duration(total_cpu)}
    for job_id, t in tasks.items():
        t["peak_gb"] = parse_kib(batch.get(job_id, ""))
    return [t for t in tasks.values() if t["state"] not in ("RUNNING", "PENDING")]


def advise(list_name, rows):
    """What the finished tasks of one list say about its #SBATCH header."""
    cores = int(script_setting(list_name, "cpus-per-task"))
    qos = script_setting(list_name, "qos")
    done = [r for r in rows if r["state"] == "COMPLETED" and r["hours"]]
    if not done:
        return ["no completed task yet"]
    longest = max(r["hours"] for r in done)
    time_h = int(longest * 1.5) + 1                      # +50%, rounded up
    peaks = [r["peak_gb"] for r in done if r["peak_gb"] is not None]
    if qos == "qos_cpu-t3":
        # t3 caps at 20 h. The pilot is the list's longest job, so a 1.2x margin
        # is enough in practice; a rare TIMEOUT is re-sent on t4 by this script.
        margin = 20 / longest
        if margin >= 1.5:
            note = "comfortable"
        elif margin >= 1.2:
            note = "OK; a rare TIMEOUT gets re-sent on qos_cpu-t4"
        else:
            note = "!! too tight: use qos_cpu-t4, or more cores"
        out = ["longest %.1f h of qos_cpu-t3's 20 h (margin %.2fx): %s"
               % (longest, margin, note)]
    else:
        out = ["longest %.1f h  ->  #SBATCH --time=%d:00:00" % (longest, time_h)]
    if peaks:
        need = int((max(peaks) * 1.5 + ANNOTATION_GB) / 4) + 1   # +50% headroom
        out.append("peak memory %.1f GB of %d GB  ->  memory alone needs >= %d cores (has %d)"
                   % (max(peaks), cores * 4, need, cores))
    eff = [r["cpu_hours"] / (r["hours"] * cores) for r in done
           if r["cpu_hours"] and r["hours"]]
    if eff:
        out.append("CPU efficiency %.0f%% (busy share of the %d cores)"
                   % (100 * sum(eff) / len(eff), cores))
    return out


def finished_report(days=30):
    rows = finished_tasks_table(_run([
        "sacct", "-n", "-P", "-u", getpass.getuser(), "-S", "now-%ddays" % days,
        "-o", "JobID,JobName,State,Elapsed,TotalCPU,MaxRSS"]))
    if not rows:
        print("no finished render task in the last %d days" % days)
        return
    print("%-9s %-12s %-14s %8s %10s" % ("list", "task", "state", "hours", "peak mem"))
    for r in sorted(rows, key=lambda r: (LISTS.index(r["list"])
                                         if r["list"] in LISTS else 99, r["task"])):
        print("%-9s %-12s %-14s %8s %10s" % (
            r["list"], r["task"], r["state"],
            "%.1f" % r["hours"] if r["hours"] is not None else "?",
            "%.1f GB" % r["peak_gb"] if r["peak_gb"] is not None else "?"))
    print()
    for list_name in LISTS:
        mine = [r for r in rows if r["list"] == list_name]
        if mine:
            print("render_%s.slurm:" % list_name)
            for line in advise(list_name, mine):
                print("  " + line)


def running_report():
    """One line per running render task: list, task, time so far, peak memory
    against what its cores give it. sacct only reports MaxRSS once a task has
    ended; sstat is the live figure."""
    rows = []
    for line in _run(["squeue", "-h", "-u", getpass.getuser(), "-t", "R",
                      "-o", "%i|%j|%M"]).splitlines():
        task, name, elapsed = (line.split("|") + ["", "", ""])[:3]
        if not name.startswith("render-"):
            continue
        list_name = name[len("render-"):]
        rss = _run(["sstat", "-n", "-P", "-j", task + ".batch", "-o", "MaxRSS"])
        peak = parse_kib(rss.splitlines()[0]) if rss.strip() else None
        limit = int(script_setting(list_name, "cpus-per-task")) * 4
        rows.append((list_name, task, elapsed, peak, limit))
    if not rows:
        print("no render task is running")
        return
    print("%-9s %-12s %11s %10s %8s %5s" % ("list", "task", "elapsed", "peak mem",
                                           "limit", "used"))
    for list_name, task, elapsed, peak, limit in rows:
        if peak is None:
            print("%-9s %-12s %11s %10s %6d GB" % (list_name, task, elapsed, "?", limit))
        else:
            print("%-9s %-12s %11s %7.1f GB %5d GB %4.0f%%" % (
                list_name, task, elapsed, peak, limit, 100 * peak / limit))


def diagnose(attempts):
    """What to do about a job that is not done, from its finished attempts."""
    if not attempts:
        return "never ran"
    states = [s for _, s in sorted(attempts)]
    last = states[-1]
    if last == "TIMEOUT":
        return "timeout"
    if last == "OUT_OF_MEMORY":
        return "out of memory"
    if last in INTERRUPTED:
        return "interrupted"
    failed = sum(1 for s in states if s in ("FAILED", "COMPLETED"))
    return "failed twice" if failed >= 2 else "failed once"


def remedy(reason, list_name):
    """Extra sbatch options for a re-send, or None when it must not be re-sent."""
    if reason == "timeout":
        if script_setting(list_name, "qos") == "qos_cpu-t4":
            return "--time=%s" % T4_MAX_TIME
        return "--qos=qos_cpu-t4 --time=60:00:00"
    if reason == "out of memory":
        cores = int(script_setting(list_name, "cpus-per-task"))
        return "--cpus-per-task=%d" % min(MAX_CORES, cores * 2)
    if reason == "failed twice":
        return None
    return ""


def array_spec(indices):
    """[0, 1, 2, 5, 7, 8] -> "0-2,5,7-8", the form `sbatch --array` takes."""
    parts, run = [], []
    for i in sorted(indices):
        if run and i != run[-1] + 1:
            parts.append(run)
            run = []
        run.append(i)
    if run:
        parts.append(run)
    return ",".join(str(r[0]) if len(r) == 1 else "%d-%d" % (r[0], r[-1])
                    for r in parts)


def last_error(outdir, job):
    """The most telling line of a failed job's log in failures/, if any."""
    path = os.path.join(outdir, "failures", job + ".log")
    try:
        with open(path, errors="replace") as fh:
            lines = [l.strip() for l in fh if l.strip()]
    except OSError:
        return ""
    errors = [l for l in lines if re.search(r"Error|Exception|Killed|exit code", l)]
    return (errors or lines or [""])[-1][:160]


# ------------------------------------------------------------------ samples
# A DONE JOB IS NOT A FULL ONE. A job lands once it has run, and inside it a
# cell can still decline (its violator leaves the shot) or crash (one clip's
# annotation) while the rest of the job is written -- so the table above can
# say "all done" over hundreds of missing samples. Each ledger entry records
# what the job asked for and what came of every cell, so the gap is counted
# here from the ledgers, against what the taxonomy says the job's level and
# condition can hold (`taxonomy.expected_cells`).

def reason_class(error):
    """A decline or failure message -> one short, countable reason."""
    error = str(error or "")
    if error.startswith("multi clip needs"):
        return "multi: fewer than two violators"
    if "leaves the frame" in error:
        return "violator leaves the frame"
    if "never observable" in error:
        return "violation never observable"
    if error.startswith("injector produced no plan"):
        return "no plan, after every retry seed"
    if error.startswith(("annotate failed", "plan raised", "trajectory prefix")):
        # The exception itself, not the traceback around it.
        last = re.findall(r"(?:\\n|\n)\s*(\w+(?:Error|Exception)\b[^\\\n']*)", error)
        return "BUG " + (last[-1] if last else error.split(":")[0])[:70]
    return error[:60] or "no record"


def sample_gaps(ledgers):
    """{(level, variant, scenario): {"job", "expected", "missing": {(family, bin): reason}}}

    `ledgers` maps a ledger name to its entry. A job and the retries that
    rebuilt its declined cells on fresh seeds share (level, variant, scenario),
    so a cell a retry built is not missing.
    """
    from physloc.scenarios.base import condition_for
    from physloc.taxonomy import SEVERITY_BINS, why_absent

    groups = collections.defaultdict(list)
    for name, entry in ledgers.items():
        req = entry.get("request") or {}
        groups[(req.get("level"), req.get("variant"), req.get("scenario"))].append(
            (int(req.get("seed") or 0), name, entry))
    out = {}
    for (level, variant, scenario), runs in groups.items():
        runs.sort(key=lambda r: r[0])                 # the main job, then retries
        req = runs[0][2]["request"]
        bins = (SEVERITY_BINS if req.get("severity") == "all"
                else [b for b in str(req.get("severity")).split(",") if b])
        condition = condition_for(int(variant), req.get("n_variants"), level)
        expected = {(f, b) for f in req.get("families") or [] for b in bins
                    if why_absent(scenario, f, level, condition) is None}
        made, reasons = set(), {}
        for _seed, _name, entry in runs:              # the LATEST attempt's reason wins
            outcome = entry.get("outcome") or {}
            made |= {(r.get("family"), r.get("severity"))
                     for r in outcome.get("results") or []}
            for bad in outcome.get("bad") or []:
                reasons[(bad.get("family"), bad.get("severity"))] = reason_class(
                    bad.get("error"))
        out[(level, variant, scenario)] = {
            "job": runs[0][1], "expected": len(expected),
            "missing": {cell: reasons.get(cell) or reasons.get((cell[0], None))
                        or "no record" for cell in expected - made}}
    return out


def sample_report(gaps, details=False):
    """A few lines: expected, present, missing by reason; per job with --details."""
    expected = sum(g["expected"] for g in gaps.values())
    missing = sum(len(g["missing"]) for g in gaps.values())
    if not expected:
        return []
    lines = ["Samples of the done jobs: %d of %d expected, %d missing (%.1f%%)"
             % (expected - missing, expected, missing, 100.0 * missing / expected)]
    by_reason = collections.defaultdict(collections.Counter)
    bug_jobs = collections.defaultdict(set)
    for g in gaps.values():
        for (family, _bin), reason in g["missing"].items():
            by_reason[reason][family] += 1
            if reason.startswith("BUG"):
                bug_jobs[reason].add(g["job"])
    for reason, fams in sorted(by_reason.items(), key=lambda kv: -sum(kv[1].values())):
        top = ", ".join("%s %d" % kv for kv in fams.most_common(4))
        lines.append("  %5d  %-34s %s%s" % (sum(fams.values()), reason[:34], top,
                                           ", ..." if len(fams) > 4 else ""))
        if reason in bug_jobs:
            lines.append("         %s  in: %s" % (reason[34:70], " ".join(sorted(bug_jobs[reason]))))
    if bug_jobs:
        lines.append("  BUG rows are failures, not decisions: fix, then delete those "
                     "jobs' .jobs/<job>.json and re-send them.")
    if details:
        for key, g in sorted(gaps.items(), key=lambda kv: kv[1]["job"]):
            if not g["missing"]:
                continue
            fams = collections.defaultdict(list)
            for (family, b), reason in g["missing"].items():
                fams[(family, reason)].append(b[0])
            lines.append("  %-34s %s" % (g["job"], "; ".join(
                "%s[%s] %s" % (f, "".join(sorted(bs)), r[:28])
                for (f, r), bs in sorted(fams.items()))))
    return lines


def read_ledgers(outdir):
    out = {}
    folder = os.path.join(outdir, ".jobs")
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if name.endswith(".json"):
            try:
                with open(os.path.join(folder, name)) as fh:
                    out[name[:-len(".json")]] = json.load(fh)
            except (OSError, ValueError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--details", action="store_true",
                    help="one line per unfinished job, with its error, and one per "
                         "job with missing samples")
    ap.add_argument("--running", action="store_true",
                    help="only the running tasks: time so far and peak memory")
    ap.add_argument("--finished", action="store_true",
                    help="finished tasks, and --time / cores advice per render script")
    a = ap.parse_args()
    if a.running:
        running_report()
        return 0
    if a.finished:
        finished_report()
        return 0
    outdir = env("PHYSLOC_OUTDIR")
    queued, finished = queued_tasks(), finished_tasks()

    total = done_total = 0
    resend = collections.defaultdict(list)       # (list, options) -> indices
    investigate, details = [], []
    print("%-9s %6s %6s %8s %10s" % ("list", "jobs", "done", "queued", "not done"))
    for list_name in LISTS:
        path = os.path.join(JOBS_DIR, list_name + ".txt")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            names = [l.strip() for l in fh if l.strip()]
        n_done = n_queued = 0
        for i, job in enumerate(names):
            if os.path.exists(os.path.join(outdir, ".jobs", job + ".json")):
                n_done += 1
                continue
            if (list_name, i) in queued:
                n_queued += 1
                continue
            reason = diagnose(finished.get((list_name, i), []))
            options = remedy(reason, list_name)
            if options is None:
                investigate.append(job)
            else:
                resend[(list_name, options)].append(i)
            details.append("  %-9s %4d  %-34s %-14s %s" % (
                list_name, i, job, reason, last_error(outdir, job)))
        print("%-9s %6d %6d %8d %10d" % (list_name, len(names), n_done, n_queued,
                                         len(names) - n_done - n_queued))
        total += len(names)
        done_total += n_done
    print("%-9s %6d %6d\n" % ("total", total, done_total))
    report = sample_report(sample_gaps(read_ledgers(outdir)), details=a.details)
    if report:
        print("\n".join(report) + "\n")

    if a.details and details:
        print("Unfinished jobs (list, task, job, reason, last error line):")
        print("\n".join(details) + "\n")
    if resend:
        print("Re-send (a task that lands in the meantime is skipped in seconds):")
        for (list_name, options), indices in sorted(resend.items()):
            print("  sbatch --array=%s %sslurm/render_%s.slurm" % (
                array_spec(indices), options + " " if options else "",
                list_name))
        print()
    if investigate:
        print("Failed twice -- a bug, not bad luck. Do not re-send; read:")
        for job in investigate:
            print("  %s/failures/%s.log" % (outdir, job))
        print("Reproduce on the docker box with:  python -m physloc.cli generate "
              "--config $PHYSLOC_CONFIG --only <job> --workers 1\n")
    if done_total == total and total:
        print("All done. Next:  sbatch slurm/finalize.slurm")
    elif not resend and not investigate:
        print("Nothing to do yet: the rest is still queued or running.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""How far the release has got, why unfinished jobs did not finish, and the
exact sbatch line to re-send them. Submits nothing itself.

    source slurm/env.sh
    python slurm/status.py            # summary + what to re-send
    python slurm/status.py --details  # ...and one line per unfinished job

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
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_job_lists import JOBS_DIR, LISTS, env, script_setting  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--details", action="store_true",
                    help="one line per unfinished job, with its error")
    a = ap.parse_args()
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

    if a.details and details:
        print("Unfinished jobs (list, task, job, reason, last error line):")
        print("\n".join(details) + "\n")
    if resend:
        print("Re-send (a task that lands in the meantime is skipped in seconds):")
        for (list_name, options), indices in sorted(resend.items()):
            print("  sbatch --array=%s %sslurm/render_%s.slurm" % (
                ",".join(map(str, indices)), options + " " if options else "",
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

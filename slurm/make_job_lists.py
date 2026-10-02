"""Write the job lists the render_*.slurm scripts read, and print the sbatch
commands to submit them. Submits nothing itself.

    source slurm/env.sh
    python slurm/make_job_lists.py

A job list is one job name per line (e.g. `L0_drop_20260824_v3`); array task i
of a render script runs line i+1 of its list. Every job goes to exactly one
list, by what it needs:

    pour_L3   pour at L3               -> render_pour_L3.slurm
    pour      pour at L0-L2            -> render_pour.slurm
    8cores    everything else, L2/L3   -> render_8cores.slurm
    4cores    everything else, L0/L1   -> render_4cores.slurm

Each list is sorted longest job first, so `--array=0` (the pilot) is the job
most likely to hit a limit. Before writing anything, every job is checked to
fit the memory its script gives it: on cpu_p1 that is 4 GB per core, and the
cores are read from the script's own `#SBATCH --cpus-per-task` line.
"""
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
JOBS_DIR = os.path.join(HERE, "jobs")

#: The lists, in the order to submit them: the longest jobs first.
LISTS = ("pour_L3", "pour", "8cores", "4cores")
GB_PER_CORE = 4
#: Memory the annotation needs beside the render, inside the same task.
ANNOTATION_GB = 2


def list_for(level, scenario):
    """Which render script a job belongs to."""
    if scenario == "pour":
        return "pour_L3" if level == "L3" else "pour"
    return "8cores" if level in ("L2", "L3") else "4cores"


def script_setting(list_name, option):
    """A value from a render script's `#SBATCH --<option>=...` line."""
    path = os.path.join(HERE, "render_%s.slurm" % list_name)
    with open(path) as fh:
        for line in fh:
            m = re.match(r"#SBATCH --%s=(\S+)" % re.escape(option), line)
            if m:
                return m.group(1)
    raise KeyError("%s has no #SBATCH --%s" % (path, option))


def env(name):
    value = os.environ.get(name)
    if not value:
        sys.exit("%s is not set -- run `source slurm/env.sh` first" % name)
    return value


def main():
    config, outdir = env("PHYSLOC_CONFIG"), env("PHYSLOC_OUTDIR")
    os.makedirs(JOBS_DIR, exist_ok=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    table = os.path.join(JOBS_DIR, "all.tsv")

    # The jobs, exactly as a local `generate` would build them.
    subprocess.run([sys.executable, "-m", "physloc.cli", "generate",
                    "--config", config, "--list-jobs", table],
                   cwd=REPO, check=True)
    with open(table) as fh:
        jobs = list(csv.DictReader(fh, delimiter="\t"))

    lists = {name: [] for name in LISTS}
    too_big = []
    for job in jobs:
        name = list_for(job["level"], job["scenario"])
        cores = int(script_setting(name, "cpus-per-task"))
        if float(job["memory_gb"]) + ANNOTATION_GB > cores * GB_PER_CORE:
            too_big.append("%s needs %s GB; render_%s.slurm gives %d cores = %d GB"
                           % (job["name"], job["memory_gb"], name, cores,
                              cores * GB_PER_CORE))
        lists[name].append(job["name"])
    if too_big:
        sys.exit("raise --cpus-per-task in these scripts first:\n  "
                 + "\n  ".join(too_big))

    for old in os.listdir(JOBS_DIR):
        if old.endswith(".txt"):
            os.remove(os.path.join(JOBS_DIR, old))
    for name, names in lists.items():
        if names:
            with open(os.path.join(JOBS_DIR, name + ".txt"), "w") as fh:
                fh.write("\n".join(names) + "\n")

    used = [n for n in LISTS if lists[n]]
    print("\nJob lists for %s -> %s" % (config, outdir))
    for n in used:
        print("  slurm/jobs/%-12s %4d jobs   %s cores, %s, %s" % (
            n + ".txt", len(lists[n]), script_setting(n, "cpus-per-task"),
            script_setting(n, "qos"), script_setting(n, "time")))
    print("\nPILOT -- the longest job of each list:")
    for n in used:
        print("  sbatch --array=0 slurm/render_%s.slurm" % n)
    print("\nEVERYTHING -- after the pilot looked right:")
    for n in used:
        print("  sbatch --array=0-%d slurm/render_%s.slurm" % (len(lists[n]) - 1, n))
    print("\nThen: python slurm/status.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

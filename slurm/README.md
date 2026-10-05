# Running PhysLoc on Jean Zay (SLURM)

Everything for the cluster is in this folder. Your local runs (`bash scripts/run.sh ...`) are not
affected.

## The idea in one picture

A release is a list of **jobs**. One job is one level × scenario × variant: its valid clip plus every
family at every severity, about 40 renders that share one scene. Each job becomes **one array task**:

```
sbatch --array=0-179 slurm/render_4cores.slurm
          │
          ├─ task 0  ─ reads line 1 of slurm/jobs/4cores.txt  → L0_collision_20260824_v0
          ├─ task 1  ─ reads line 2                           → L0_rolling_ramp_20260824_v0
          └─ ...
               each task runs:
               python -m physloc.cli generate --only <that job>     (your normal code)
                 ├─ slurm/kubric_singularity.sh → kubric.sif        (render)
                 └─ annotation                                      (writes samples/)
```

This is the IDRIS job-array example, with `./mon_exe < fichier${SLURM_ARRAY_TASK_ID}.in` replaced by
"read line N of a list, run that job".

## The files

| file | what it does | you run it |
|---|---|---|
| `env.sh` | your account and paths: **the only file to edit** | `source slurm/env.sh` in every new shell |
| `setup/1_conda.slurm` | creates the Python env in `$WORK` | once |
| `setup/2_image.slurm` | builds `kubric.sif` from the same docker image as local runs, then `idrcontmgr` | once |
| `setup/3_assets.slurm` | downloads the Kubric assets (~12 GB): compute nodes have no internet | once |
| `make_job_lists.py` | writes `slurm/jobs/*.txt` and **prints the sbatch commands to run** | once per release |
| `render_4cores.slurm` | L0/L1 jobs (except `pour`): 4 cores, 20 h | `sbatch --array=...` |
| `render_8cores.slurm` | L2/L3 jobs (except `pour`): 8 cores, 20 h | `sbatch --array=...` |
| `render_pour.slurm` | `pour` L0–L2: 10 cores, long queue | `sbatch --array=...` |
| `render_pour_L3.slurm` | `pour` L3: 32 cores (it needs ~120 GB), long queue | `sbatch --array=...` |
| `status.py` | what is done, why the rest is not, and the exact line to re-send it | any time |
| `finalize.slurm` | stats, index, validate, audit, coverage video | once, at the end |
| `archive.slurm` | tar the release into `$STORE` | once, at the end |
| `kubric_singularity.sh` | how a render starts inside `kubric.sif` (used by `generate`; you never call it) | never |

The four render scripts are identical except for their `#SBATCH` header and the job list they read.
Why four: on Jean Zay memory comes with cores (about 4 GB per core), and `pour` needs the most of both.

## Step by step

Run everything **from the repo root** (`cd $WORK/physloc`): logs and job lists are relative paths.

### 0. Once: get the code, edit one file

```bash
cd $WORK && git clone https://github.com/samuruph/physloc.git && cd physloc && git checkout dev
nano slurm/env.sh          # set SBATCH_ACCOUNT (run `idrproj` to see yours) and PHYSLOC_CONDA_MODULE
source slurm/env.sh
```

### 1. Once: setup (three jobs on prepost, which has internet and is not charged)

```bash
sbatch slurm/setup/1_conda.slurm
sbatch slurm/setup/2_image.slurm
sbatch slurm/setup/3_assets.slurm
squeue -u $USER                     # wait until all three are gone
tail slurm/logs/setup/setup-*.out         # conda ends "host env OK"; image lists kubric.sif
```

If `idrcontmgr` refuses the image (see the `setup-image` log), stop there and write to
assist@idris.fr.

### 2. Quick test on one node (qos_cpu-dev: up to 2 h, starts fast)

```bash
srun --qos=qos_cpu-dev --hint=nomultithread --cpus-per-task=4 --time=01:00:00 --pty bash
module load singularity
source slurm/env.sh
export PHYSLOC_THREADS=4 PHYSLOC_SCRATCH=${JOBSCRATCH:-$SCRATCH/smoke_tmp}
bash slurm/kubric_singularity.sh scripts/probes/worker_smoke.py --frames 4      # renders?
python -m physloc.cli generate --config review --scenario drop --family solidity \
    --workers 1 --outdir $PHYSLOC_DATA/smoke --workdir $PHYSLOC_SCRATCH/w      # render + annotate
python -m physloc.cli validate $PHYSLOC_DATA/smoke                              # must end "ok": true
exit
```

### 3. Make the job lists

```bash
python slurm/make_job_lists.py
```

It prints how many jobs each list has, then the exact commands for steps 4 and 5.

### 4. Pilot: one job per render script

```bash
sbatch --array=0 slurm/render_pour_L3.slurm
sbatch --array=0 slurm/render_pour.slurm
sbatch --array=0 slurm/render_8cores.slurm
sbatch --array=0 slurm/render_4cores.slurm
```

Line 1 of each list is its longest job. When all four have finished:

```bash
sacct -u $USER --starttime today --format=JobID,JobName%16,Elapsed,MaxRSS,State
```

- **Elapsed:** `render-4cores` and `render-8cores` must be well under 20 h. In `render_pour*.slurm`,
  set `#SBATCH --time` to the `pour` time plus about 50% (`qos_cpu-t4` allows up to 100 h).
- **MaxRSS:** must be below cores × 4 GB.
- **State:** `COMPLETED`. Otherwise read `slurm/logs/render-*_<id>_0.out`.
- **Look at a few clips** (`overlay.mp4` under `$PHYSLOC_OUTDIR/samples/`) before going further.

### 5. Everything

Paste the four "EVERYTHING" lines `make_job_lists.py` printed, e.g.:

```bash
sbatch --array=0-1   slurm/render_pour_L3.slurm
sbatch --array=0-17  slurm/render_pour.slurm
sbatch --array=0-59  slurm/render_8cores.slurm
sbatch --array=0-179 slurm/render_4cores.slurm
```

The pilot's jobs are already done, so their tasks finish in seconds. To follow progress:

```bash
squeue -u $USER          # what is running / waiting (and why, e.g. QOSMaxCpuPerUserLimit)
python slurm/status.py   # what is done, and what to re-send
```

### 6. If a job fails

Nothing is lost: every other job carries on. A job only counts as done once its file
`$PHYSLOC_OUTDIR/.jobs/<job>.json` exists, so a failed job is redone from scratch when re-sent. Its
clips come out the same, because the seed comes from the job's name.

When `squeue -u $USER` shows nothing left, run:

```bash
python slurm/status.py --details
```

For every job that isn't done, it asks SLURM how the last attempt ended and prints the line to re-send
it:

| what SLURM says | what it means | what `status.py` prints |
|---|---|---|
| never ran, `NODE_FAIL`, `CANCELLED` | not the job's fault | the same `sbatch` line again |
| `TIMEOUT` | needed more than its `--time` | the line with `--qos=qos_cpu-t4 --time=60:00:00` |
| `OUT_OF_MEMORY` | needed more than cores × 4 GB | the line with twice the `--cpus-per-task` |
| `FAILED` once | the program crashed | the same line again (often a one-off) |
| `FAILED` twice | a bug, not bad luck | **no re-send**: the log to read |

Run the lines it prints, for example `sbatch --array=17,42 slurm/render_4cores.slurm`. Options given on
the `sbatch` line override the script's `#SBATCH` header. Re-sending is always safe: a job that
finished in the meantime is skipped in seconds.

**A job that failed twice.** Read `$PHYSLOC_OUTDIR/failures/<job>.log` (the full error) and the task
log `slurm/logs/render-<list>_<array>_<task>.out`. Then reproduce it on the docker box, where you can
debug:

```bash
python -m physloc.cli generate --config v0_release --only <job> --workers 1
```

Fix it, commit, `git pull` on Jean Zay, and re-send that job.

Only fix bugs this way if the fix doesn't change how other clips look. If it does, the jobs already
done were made with the old code, and the release would mix the two.

**Not a failure:** a log line saying a family "declined" a scene and was retried on another seed.
That's normal, and handled inside the job.

### 7. Finish

```bash
sbatch slurm/finalize.slurm       # when status.py says "All done"
```

Then:
- Look at `$PHYSLOC_OUTDIR/audit.txt`, `validate.json` and `coverage_strong.mp4`.
- `sbatch slurm/archive.slurm` copies the release to `$STORE`. Do this soon: `$SCRATCH` deletes files
  nobody has read for 30 days.
- Publishing is by hand, on prepost:

  ```bash
  srun --partition=prepost --time=10:00:00 --pty bash
  source slurm/env.sh
  python -m physloc.cli export $PHYSLOC_OUTDIR --outdir $SCRATCH/hf/$PHYSLOC_RELEASE --push-to <user>/physloc
  ```

## Good to know

- **Re-running is always safe.** A job that already finished has a file in `$PHYSLOC_OUTDIR/.jobs/`
  and is skipped in seconds.
- **Don't change the code during a release.** Run every task from the same commit.
- **Useful commands:**
  - `scancel <array id>` stops a whole array; `scancel <array id>_<task>` stops one task.
  - `sacct -j <id> --format=JobID,Elapsed,MaxRSS,State` shows what happened to a job.
- **Where things are:**
  - code and env: `$WORK`
  - release: `$SCRATCH/physloc/<release>`
  - per-task scratch: `$JOBSCRATCH` (deleted when the task ends)
  - archive: `$STORE/physloc`

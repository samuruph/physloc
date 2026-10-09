# Running PhysLoc on Jean Zay (SLURM)

Everything for the cluster is in this folder. Your local runs (`bash scripts/run.sh ...`) are not
affected.

## The idea in one picture

A release is a list of **jobs**. One job is one level × scenario × variant: its valid clip plus every
family at every severity, about 40 renders that share one scene. Each job becomes **one array task**:

```
sbatch --array=0-179 slurm/render/render_4cores.slurm
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

One folder per stage of a release, in the order you use them. Everything runs from the repo root.

```
slurm/
  env.sh                  your account and paths: the ONLY file to edit
  kubric_singularity.sh   how a render starts inside kubric.sif (generate calls it; you never do)
  setup/    once:          conda env, container image, asset mirror
  render/   the release:   job lists + one render script per resource size
  check/    any time:      what is done, what is missing and why, is what is there sound
  refill/   after a fix:   re-render one family on finished jobs, merge it in
  finish/   at the end:    finalize, archive to $STORE, publish
  jobs/  logs/             written by the scripts (gitignored)
```

| file | what it does | you run it |
|---|---|---|
| `env.sh` | your account and paths: **the only file to edit** | `source slurm/env.sh` in every new shell |
| `setup/1_conda.slurm` | creates the Python env in `$WORK` | once |
| `setup/2_image.slurm` | builds `kubric.sif` from the same docker image as local runs, then `idrcontmgr` | once |
| `setup/3_assets.slurm` | downloads the Kubric assets (~12 GB): compute nodes have no internet | once |
| `render/make_job_lists.py` | writes `slurm/jobs/*.txt` and **prints the sbatch commands to run** | once per release |
| `render/render_4cores.slurm` | L0/L1 jobs (except `pour`): 4 cores, 20 h | `sbatch --array=...` |
| `render/render_8cores.slurm` | L2/L3 jobs (except `pour` and `shadow_track`): 8 cores, 20 h | `sbatch --array=...` |
| `render/render_8cores_long.slurm` | `shadow_track` L2/L3 (each clip rendered 2–3 times): 8 cores, long queue | `sbatch --array=...` |
| `render/render_pour.slurm` | `pour` L0–L2: 10 cores, long queue | `sbatch --array=...` |
| `render/render_pour_L3.slurm` | `pour` L3: 16 cores (~64 GB), long queue | `sbatch --array=...` |
| `check/status.py` | jobs done or not (with the line to re-send), then samples missing, by reason | any time |
| `check/check_release.py` | empty sample folders, unverified prefixes, bodies falling out of valid clips | before finalize |
| `check/compare_valid.py` | after a refill: its valid clips equal the release's, frame by frame | after a refill |
| `check/fetch_for_inspection.sh` | copies a release (minus `data.h5`) to your workstation | on your workstation |
| `refill/render_refill.slurm` | re-renders ONE family on finished jobs, into its own folder | after a fix (6b) |
| `refill/merge_refill.py` | swaps that family's samples into the release | after the refill |
| `finish/finalize.slurm` | stats, index, split, validate, audit, coverage video, showcase | once, at the end |
| `finish/archive.slurm` | tar the release into `$STORE`, verified by `finish/check_archive.py` | once, at the end |
| `finish/publish.slurm` | export + upload to HuggingFace | by hand, when happy |

The five render scripts are identical except for their `#SBATCH` header and the job list they read.
Why five: on Jean Zay memory comes with cores (about 4 GB per core), and `pour` needs the most of both.

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
python slurm/render/make_job_lists.py
```

It prints how many jobs each list has, then the exact commands for steps 4 and 5.

### 4. Pilot: one job per render script

```bash
sbatch --array=0 slurm/render/render_pour_L3.slurm
sbatch --array=0 slurm/render/render_pour.slurm
sbatch --array=0 slurm/render/render_8cores.slurm
sbatch --array=0 slurm/render/render_4cores.slurm
```

Line 1 of each list is its longest job. When all four have finished, `python slurm/check/status.py --finished` does the arithmetic below for you. By hand:

```bash
sacct -u $USER --starttime today --format=JobID,JobName%16,Elapsed,MaxRSS,State
```

- **Elapsed:** `render-4cores` and `render-8cores` must be well under 20 h. In `render_pour*.slurm`,
  set `#SBATCH --time` to the `pour` time plus about 50% (`qos_cpu-t4` allows up to 100 h).
- **MaxRSS:** must be below cores × 4 GB.
- **State:** `COMPLETED`. Otherwise read `slurm/logs/render-*_<id>_0.out`.
- **Look at a few clips** (`overlay.mp4` under `$PHYSLOC_OUTDIR/samples/`) before going further.

### 5. Everything

Paste the "EVERYTHING" lines `make_job_lists.py` printed (one per render script), e.g.:

```bash
sbatch --array=0-1   slurm/render/render_pour_L3.slurm
sbatch --array=0-17  slurm/render/render_pour.slurm
sbatch --array=0-59  slurm/render/render_8cores.slurm
sbatch --array=0-179 slurm/render/render_4cores.slurm
```

The pilot's jobs are already done, so their tasks finish in seconds. To follow progress:

```bash
squeue -u $USER          # what is running / waiting (and why, e.g. QOSMaxCpuPerUserLimit)
python slurm/check/status.py   # what is done, and what to re-send
python slurm/check/status.py --running   # each running task: time so far, peak memory vs its limit
python slurm/check/status.py --finished  # finished tasks + a --time / cores advice per render script
```

### 6. If a job fails

Nothing is lost: every other job carries on. A job only counts as done once its file
`$PHYSLOC_OUTDIR/.jobs/<job>.json` exists, so a failed job is redone from scratch when re-sent. Its
clips come out the same, because the seed comes from the job's name.

When `squeue -u $USER` shows nothing left, run:

```bash
python slurm/check/status.py --details
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

Run the lines it prints, for example `sbatch --array=17,42 slurm/render/render_4cores.slurm`. Options given on
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

**A done job can still be missing samples.** Below the table, `status.py` counts the samples of the
done jobs against what the taxonomy expects, and groups the missing ones by reason:

```
Samples of the done jobs: 9295 of 9672 expected, 377 missing (3.9%)
    221  violator leaves the frame          solidity 60, immutability 58, superelastic 36, ...
     20  BUG ValueError: affected clock dis shadow_track ...
         agrees with the violator records  in: <job> <job>
```

Cells a level or condition can never hold (`colour_shift` at L3, `fission` in a `multi` clip, ...)
are not expected, so they are not counted. `physloc taxonomy -v` lists them with the reason for each.
A decline (the violator leaves the shot) is a decision, and the cell stays missing. A `BUG` row is a
crash: fix it, delete those jobs' `.jobs/<job>.json` and re-send them. With `--details`, there's one
line per job with what it is missing.

### 6b. Refill: re-render one family after a fix

When a fix changes how ONE family renders (e.g. `solidity` on ramps), re-render only that family on
the jobs it touches, into its own folder, then swap its samples into the release:

```bash
# the jobs: every job of these scenarios that runs this family
python -m physloc.cli generate --config $PHYSLOC_CONFIG --scenario ramp_slide,rolling_ramp \
    --family solidity --list-jobs slurm/jobs/refill.tsv
tail -n +2 slurm/jobs/refill.tsv | cut -f1 > slurm/jobs/refill.txt; wc -l slurm/jobs/refill.txt
PHYSLOC_REFILL_FAMILY=solidity sbatch --array=0-<N-1> slurm/refill/render_refill.slurm
# when they are done:
python slurm/refill/merge_refill.py --family solidity           # preview
python slurm/refill/merge_refill.py --family solidity --apply   # swap the family's samples in
python slurm/check/status.py                                   # the refill now counts for that family
```

The refill writes to `$PHYSLOC_DATA/refill_<family>/$PHYSLOC_RELEASE`, so the release is untouched until
the merge. The merge removes every old sample of the family on each refilled scene slot before copying
the new ones in, so a bin the fixed code declines doesn't keep its old clip. A slot whose refill job
failed has no ledger and is left as it was. Then run `finalize`.

### 7. Finish

```bash
sbatch slurm/finish/finalize.slurm       # when status.py says "All done"
```

Then:
- Look at `$PHYSLOC_OUTDIR/audit.txt` and `validate.json`, then the videos, all in
  `$PHYSLOC_OUTDIR/viz/`:
  - `coverage_strong.mp4`: every invalid clip in one video. Open this first.
  - `compare/`: one cell's levels, variants and conditions side by side.
  - `showcase/`: the presentation videos and `picks.json`. To curate, edit the file and re-run
    `python -m physloc.cli showcase $PHYSLOC_OUTDIR --picks $PHYSLOC_OUTDIR/viz/showcase/picks.json --only <video>`.

### 8. Archive to $STORE

```bash
sbatch slurm/finish/archive.slurm
```

Do this soon after finalize: `$SCRATCH` deletes files nobody has read for 30 days. It writes 5 tar
files to `$STORE/physloc/`:

- `physloc_v0_meta.tar`: everything except the level folders (index, manifest, stats, `viz/`, the
  `.jobs` ledger, `failures/`, ...);
- `physloc_v0_L0.tar` … `physloc_v0_L3.tar`: one per level.

Then the job **checks** that the tars together hold exactly the release: every path, size, empty
folder and link. If anything differs, the job fails and its log lists what. The last lines of
`slurm/logs/archive_<id>.out` should say `archive OK`.

To restore, extract every tar in the same folder; you get back the identical `physloc_v0/` tree:

```bash
cd $SCRATCH/physloc
for t in $STORE/physloc/physloc_v0_*.tar; do tar -xf $t; done
```

### 9. Publish to HuggingFace (by hand, when you are happy with the clips)

Once, on a login node, log in to HuggingFace, then check `PHYSLOC_HF_REPO` and `PHYSLOC_HF_PRIVATE` in
`slurm/env.sh`:

```bash
hf auth login
```

Then:

```bash
sbatch slurm/finish/publish.slurm
```

It runs on prepost (internet, not charged):

1. It packages the release into `$SCRATCH/hf/<release>`. This is a full copy (about the release's
   size again), and it's skipped if a previous run already finished it.
2. It uploads in pieces, each its own commit: first the card, index and splits, then one level at a
   time.

If it runs out of its 20 h, `sbatch` it again: pieces already on the Hub aren't sent again.

**Try it on something small first:** point `PHYSLOC_HF_REPO` at a throwaway private repo and run
it once. A 1.3 TB upload is not something to find problems in for the first time.

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

#!/bin/bash
# REFILL -- re-render ONE family on jobs that already ran, after a fix to it.
# One array task = one job of slurm/jobs/refill.txt (line i+1), run with
# `--family $PHYSLOC_REFILL_FAMILY` only: its valid clip and that family at every
# severity. Written to its own release folder,
#   $PHYSLOC_DATA/refill_<family>/$PHYSLOC_RELEASE
# so the release's own ledger and samples are untouched until
# `slurm/merge_refill.py` swaps the family's samples in (slurm/README.md, "Refill").
#
# Submit from the repo root:
#   PHYSLOC_REFILL_FAMILY=solidity sbatch --array=0-<N-1> slurm/render_refill.slurm
#SBATCH --job-name=render-refill
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3              # up to 20 h
#SBATCH --ntasks=1                    # one process: generate (it starts the render)
#SBATCH --cpus-per-task=8             # = ~32 GB of memory on cpu_p1 (4 GB per core)
#SBATCH --hint=nomultithread          # physical cores, no hyperthreading
#SBATCH --time=10:00:00               # 4 renders per job; the L3 ones are the long ones
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.out

cd ${SLURM_SUBMIT_DIR}
module purge
module load singularity
source slurm/env.sh
set -x

FAMILY=${PHYSLOC_REFILL_FAMILY:?set PHYSLOC_REFILL_FAMILY, e.g. solidity}
# The folder's NAME is the release's: it is what every sample's uid starts with.
OUT=$PHYSLOC_DATA/refill_$FAMILY/$PHYSLOC_RELEASE

# This task's job, e.g. L0_ramp_slide_20260824_v0
JOB=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" slurm/jobs/refill.txt)

export PHYSLOC_THREADS=$SLURM_CPUS_PER_TASK
export PHYSLOC_SCRATCH=$JOBSCRATCH

python -m physloc.cli generate --config $PHYSLOC_CONFIG --only $JOB \
    --family $FAMILY --workers 1 --resume --keep-going --no-finalize \
    --outdir $OUT --workdir $JOBSCRATCH/work

# Done only if the job's ledger entry was written; otherwise FAILED in sacct.
test -f $OUT/.jobs/$JOB.json

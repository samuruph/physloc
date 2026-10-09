#!/bin/bash
# RENDER -- L2 and L3 jobs (all scenarios except pour); HDRI levels render slower.
# One array task = one job (one level x scenario x variant: its valid clip and
# every family at every severity). Task i runs line i+1 of slurm/jobs/8cores.txt.
#
# Submit from the repo root, after make_job_lists.py (it prints the exact line):
#   sbatch --array=0-<N-1> slurm/render/render_8cores.slurm
#SBATCH --job-name=render-8cores
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3              # up to 20 h
#SBATCH --ntasks=1                    # one process: generate (it starts the render)
#SBATCH --cpus-per-task=8             # = ~32 GB of memory on cpu_p1 (4 GB per core)
#SBATCH --hint=nomultithread          # physical cores, no hyperthreading
#SBATCH --time=20:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.out

cd ${SLURM_SUBMIT_DIR}
module purge
module load singularity
source slurm/env.sh
set -x

# This task's job, e.g. L0_drop_20260824_v3
JOB=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" slurm/jobs/8cores.txt)

# Blender uses exactly the cores we were given; scratch is this task's own
# folder, deleted by SLURM when the task ends.
export PHYSLOC_THREADS=$SLURM_CPUS_PER_TASK
export PHYSLOC_SCRATCH=$JOBSCRATCH

python -m physloc.cli generate --config $PHYSLOC_CONFIG --only $JOB \
    --workers 1 --resume --keep-going --no-finalize \
    --outdir $PHYSLOC_OUTDIR --workdir $JOBSCRATCH/work

# Done only if the job's ledger entry was written; otherwise FAILED in sacct.
test -f $PHYSLOC_OUTDIR/.jobs/$JOB.json

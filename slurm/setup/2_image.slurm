#!/bin/bash
# SETUP 2/3 -- the render image: kubric.sif, from the SAME pinned docker digest
# the local runs use (docker/IMAGE_DIGEST), then authorised with idrcontmgr.
# Run once, from the repo root:   sbatch slurm/setup/2_image.slurm
#SBATCH --job-name=setup-image
#SBATCH --partition=prepost           # has internet, not charged to your hours
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/setup/%x_%j.out
#SBATCH --error=slurm/logs/setup/%x_%j.out

cd ${SLURM_SUBMIT_DIR}
module purge
module load singularity
source slurm/env.sh
set -ex

DIGEST=$(grep -m1 '^kubricdockerhub/kubruntu@sha256:' docker/IMAGE_DIGEST)
singularity build --force $SCRATCH/kubric.sif docker://$DIGEST
idrcontmgr cp $SCRATCH/kubric.sif     # checked by IDRIS, copied to $SINGULARITY_ALLOWED_DIR
idrcontmgr ls
# rm -f $SCRATCH/kubric.sif

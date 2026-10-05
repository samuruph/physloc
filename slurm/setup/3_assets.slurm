#!/bin/bash
# SETUP 3/3 -- the Kubric asset mirror (~12 GB) into $PHYSLOC_ASSETS. Compute
# nodes have no internet; renders read the assets from here instead.
# Run once, from the repo root:   sbatch slurm/setup/3_assets.slurm
# Safe to re-run: only missing or incomplete files are downloaded.
#SBATCH --job-name=setup-assets
#SBATCH --partition=prepost           # has internet, not charged to your hours
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/setup/%x_%j.out
#SBATCH --error=slurm/logs/setup/%x_%j.out

cd ${SLURM_SUBMIT_DIR}
module purge
source slurm/env.sh
set -ex

python slurm/setup/mirror_assets.py --dest $PHYSLOC_ASSETS --jobs 8

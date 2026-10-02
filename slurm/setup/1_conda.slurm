#!/bin/bash
# SETUP 1/3 -- the host Python environment (environment.yml), into $WORK.
# Run once, from the repo root:   sbatch slurm/setup/1_conda.slurm
#SBATCH --job-name=setup-conda
#SBATCH --partition=prepost           # has internet, not charged to your hours
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out

cd ${SLURM_SUBMIT_DIR}
module purge
source slurm/env.sh
module load $PHYSLOC_CONDA_MODULE
set -ex

export CONDA_PKGS_DIRS=$(dirname $PHYSLOC_CONDA_ENV)/pkgs   # package cache off $HOME
conda env create -p $PHYSLOC_CONDA_ENV -f environment.yml || \
    conda env update -p $PHYSLOC_CONDA_ENV -f environment.yml --prune

$PHYSLOC_CONDA_ENV/bin/python -c "import numpy, h5py, cv2, imageio_ffmpeg, yaml; print('host env OK')"

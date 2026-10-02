#!/bin/bash
# FINALIZE -- once every render job is done (python slurm/status.py says so).
# What a single local `generate` does at its end, done once for the whole
# release: stats + dataset manifest, then validate, audit and a coverage video.
#   sbatch slurm/finalize.slurm
# Uploading is NOT here: look at the clips first, then export by hand (README).
#SBATCH --job-name=finalize
#SBATCH --partition=prepost           # not charged to your hours
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out

cd ${SLURM_SUBMIT_DIR}
module purge
source slurm/env.sh
set -x

python -m physloc.cli finalize $PHYSLOC_OUTDIR                               # stats/ + index + manifest
python -m physloc.cli validate $PHYSLOC_OUTDIR > $PHYSLOC_OUTDIR/validate.json  # must say "ok": true
python -m physloc.cli audit    $PHYSLOC_OUTDIR > $PHYSLOC_OUTDIR/audit.txt 2>&1 # cells whose violation is not visible
python -m physloc.cli coverage $PHYSLOC_OUTDIR --severity strong --out $PHYSLOC_OUTDIR/coverage_strong.mp4
grep '"ok"' $PHYSLOC_OUTDIR/validate.json

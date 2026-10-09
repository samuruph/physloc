#!/bin/bash
# FINALIZE -- once every render job is done (python slurm/check/status.py says so).
# What a single local `generate` does at its end, done once for the whole
# release: stats + dataset manifest, then validate and audit, then the videos,
# all under $PHYSLOC_OUTDIR/viz/: coverage, compare (levels / variants /
# conditions) and the slide-ready showcase.
#   sbatch slurm/finish/finalize.slurm
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

# Every video goes under viz/:
VIZ=$PHYSLOC_OUTDIR/viz
python -m physloc.cli coverage $PHYSLOC_OUTDIR --severity strong --out $VIZ/coverage_strong.mp4  # open this first
for LEVEL in L0 L1 L2 L3; do                                                  # structure, 12 cells per kind
  python -m physloc.cli compare $PHYSLOC_OUTDIR --kind variants   --level $LEVEL --limit 12 --outdir $VIZ/compare/$LEVEL
  python -m physloc.cli compare $PHYSLOC_OUTDIR --kind conditions --level $LEVEL --limit 12 --outdir $VIZ/compare/$LEVEL
done
python -m physloc.cli compare  $PHYSLOC_OUTDIR --kind levels --limit 12 --outdir $VIZ/compare   # L0 | L1 | L2 | L3
python -m physloc.cli showcase $PHYSLOC_OUTDIR --outdir $VIZ/showcase          # slide-ready 1920x1080 videos
grep '"ok"' $PHYSLOC_OUTDIR/validate.json

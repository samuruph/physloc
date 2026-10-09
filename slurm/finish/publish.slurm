#!/bin/bash
# PUBLISH -- package the release for HuggingFace and upload it, after you have
# looked at the clips and finalize.slurm passed. Never runs by itself.
#   hf auth login                    # once, on a login node (stores your token in $HOME)
#   sbatch slurm/finish/publish.slurm
#
# 1. export: copy the samples into a clean folder ($SCRATCH/hf/<release>) with the
#    dataset card, index and splits. Skipped if a previous run finished it
#    (dataset.json is its last file), so a re-run does not copy 1.3 TB again.
# 2. upload in pieces -- everything but the samples, then one level at a time --
#    each its own commit. If the job runs out of time, sbatch it again: pieces
#    already on the Hub are skipped (HuggingFace does not re-send a file it has).
#
# Repo and visibility: PHYSLOC_HF_REPO / PHYSLOC_HF_PRIVATE in slurm/env.sh.
#SBATCH --job-name=publish
#SBATCH --partition=prepost           # has internet, not charged to your hours
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out

cd ${SLURM_SUBMIT_DIR}
module purge
source slurm/env.sh
set -ex

HF_DIR=$SCRATCH/hf/$PHYSLOC_RELEASE
VISIBILITY=$([ "$PHYSLOC_HF_PRIVATE" = 1 ] && echo --private || echo --no-private)

# 1. package (once)
if [ ! -f $HF_DIR/dataset.json ]; then
  python -m physloc.cli export $PHYSLOC_OUTDIR --outdir $HF_DIR
fi

# 2. upload, piece by piece
hf upload $PHYSLOC_HF_REPO $HF_DIR . --repo-type dataset $VISIBILITY \
    --exclude "samples/*" --commit-message "card, index, splits, stats"
for level in $(cd $HF_DIR/samples/$PHYSLOC_RELEASE && ls -d L[0-9]); do
  hf upload $PHYSLOC_HF_REPO $HF_DIR . --repo-type dataset $VISIBILITY \
      --include "samples/$PHYSLOC_RELEASE/$level/*" --commit-message "samples $level"
done
echo "published: https://huggingface.co/datasets/$PHYSLOC_HF_REPO"

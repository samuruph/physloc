#!/bin/bash
# ARCHIVE -- copy the finished release from $SCRATCH (deleted after 30 days
# unread) to $STORE (permanent) as a few tar files: one per level, plus one for
# everything that is not a sample (index, manifest, stats, audit, ...).
# $STORE allows only 100k files per project, hence tar.
#   sbatch slurm/archive.slurm
#SBATCH --job-name=archive
#SBATCH --partition=archive           # for copying data; not charged
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out

cd ${SLURM_SUBMIT_DIR}
module purge
source slurm/env.sh
set -ex

mkdir -p $PHYSLOC_STORE
cd $PHYSLOC_DATA
tar --create --file $PHYSLOC_STORE/${PHYSLOC_RELEASE}_meta.tar --exclude "$PHYSLOC_RELEASE/samples" $PHYSLOC_RELEASE
for level in L0 L1 L2 L3; do
  dir=$PHYSLOC_RELEASE/samples/$PHYSLOC_RELEASE/$level
  if [ -d $dir ]; then
    tar --create --file $PHYSLOC_STORE/${PHYSLOC_RELEASE}_$level.tar $dir
  fi
done
ls -lh $PHYSLOC_STORE

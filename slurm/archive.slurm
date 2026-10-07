#!/bin/bash
# ARCHIVE -- copy the finished release from $SCRATCH (deleted after 30 days
# unread) to $STORE (permanent) as a few tar files: one per level, plus one for
# everything else (index, manifest, stats, audit, videos, ledger, ...), then
# checks the tars hold exactly the release. $STORE allows only 100k files per
# project, hence tar.
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
REPO=$PWD
REL=$PHYSLOC_RELEASE

mkdir -p $PHYSLOC_STORE
cd $PHYSLOC_DATA                       # tar paths start with the release name
LEVELS=$(cd $REL/samples/$REL && ls -d L[0-9])

# 1. meta: the whole release EXCEPT the level folders -- so anything else that
#    is ever added (inside samples/ or not) is archived too, never dropped.
EXCLUDE=()
for level in $LEVELS; do EXCLUDE+=(--exclude="$REL/samples/$REL/$level"); done
tar --create --file $PHYSLOC_STORE/${REL}_meta.tar "${EXCLUDE[@]}" $REL

# 2. one tar per level folder.
for level in $LEVELS; do
  tar --create --file $PHYSLOC_STORE/${REL}_$level.tar $REL/samples/$REL/$level
done

# 3. check: the tars together must hold exactly the release -- every path,
#    size, empty folder and link. A mismatch fails this job.
python3 $REPO/slurm/check_archive.py $PHYSLOC_DATA/$REL \
    $PHYSLOC_STORE/${REL}_meta.tar $(for l in $LEVELS; do echo $PHYSLOC_STORE/${REL}_$l.tar; done)
ls -lh $PHYSLOC_STORE

# To restore, extract every tar in the same folder:
#   cd $SCRATCH/physloc && for t in $STORE/physloc/physloc_v0_*.tar; do tar -xf $t; done

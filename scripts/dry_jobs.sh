#!/usr/bin/env bash
# Run the worker's full decision path (no rendering) on named release jobs,
# writing trajectories + plans for scripts/check_multi.py.
#   bash scripts/dry_jobs.sh <outroot> L0_barrier_pass_20260832_v8 ...
# The family list is the job's own, from the v0 ledger, in the worker's order.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=$1; shift
LEDGER=/mnt/physloc/physloc_v0/.jobs
PARAMS=/mnt/physloc/work_v0/params.json
declare -A NV=([L0]=10 [L1]=5 [L2]=3 [L3]=2)
for job in "$@"; do
  level=${job%%_*}; rest=${job#*_}; v=${rest##*_v}; rest=${rest%_v*}
  seed=${rest##*_}; scen=${rest%_*}
  fams=$(python3 -c "import json,sys;print(','.join(json.load(open('$LEDGER/$job.json'))['request']['families']))" 2>/dev/null \
         || conda run -n physloc python -c "
from physloc import taxonomy; from physloc.scenarios import available; from physloc import injectors
print(','.join(sorted(f for s,f in taxonomy.build_cells() if s=='$scen' and f in set(injectors.available()))))")
  ( PHYSLOC_SKIP_RENDER=1 PHYSLOC_MEMORY=24g bash docker/kubric.sh physloc/render/worker.py \
      --scenario $scen --seed $seed --tier release --frames 61 --family $fams \
      --severity all --complexity $level --variant $v --n-variants ${NV[$level]} \
      --params $PARAMS --outdir $OUT/$level > $OUT/$job.log 2>&1
    echo "$job rc=$?" ) &
done
wait

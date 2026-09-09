#!/usr/bin/env bash
# Every review config, back to back, with a log apiece. The overnight sweep:
# start it, and in the morning there is one directory and one HuggingFace
# dataset per thing worth checking.
#
#   bash scripts/run_reviews.sh                    # the standard set
#   bash scripts/run_reviews.sh review_L2 review   # or just these
#
# `PHYSLOC_PUSH_OWNER` makes each run publish to <owner>/physloc-<config>;
# leave it unset to keep everything local.
set -u
cd "$(dirname "$0")/.."

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  # Cheapest first, so a mistake in the common path shows up in minutes rather
  # than after the two-hour full review.
  CONFIGS=(review_L0 review_L1 review_L2 review_L3
           review_conditions review_severity review_ladder review)
fi

mkdir -p out/logs
for C in "${CONFIGS[@]}"; do
  echo "=== $C  $(date) ==="
  # NOT `set -e`: one config failing is a reason to look at that config, not a
  # reason to lose the six that would have run after it overnight.
  bash scripts/run.sh "$C" 2>&1 | tee "out/logs/$C.txt"
  echo "=== $C exit ${PIPESTATUS[0]}  $(date) ==="
done
echo "ALL DONE $(date)"

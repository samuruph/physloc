#!/usr/bin/env bash
# Quick shadow sanity pass: every family staged on the shadow scenario, one
# variant, strong only.
#
#   bash scripts/run_review_shadow.sh
#
# Extra flags are forwarded to `generate`, so you can still override geometry:
#
#   bash scripts/run_review_shadow.sh --frames 37
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

bash scripts/run.sh review \
  --scenario shadow_track \
  --variants 1 \
  --severity strong \
  --outdir out/review_shadow \
  --workdir out/work_review_shadow \
  "$@"

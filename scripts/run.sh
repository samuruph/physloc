#!/usr/bin/env bash
# Generate a release from a config, validate it, and build everything worth
# looking at.
#
#   bash scripts/run.sh                       # the review sweep (configs/review.yaml)
#   bash scripts/run.sh review_random         # every cell FOUR times -- does it vary?
#   bash scripts/run.sh v0_release            # the published dataset
#   bash scripts/run.sh review --tier release # extra flags pass straight through
#
# Tiers are `debug` and `release` -- two geometries, nothing more. Difficulty is
# the complexity ladder (L0..L5, README section 8), and `v0`/`v1` are what a
# published dataset is CALLED, set by the config's outdir.
#
# The config decides tier, complexity, severity, seed and variants; see
# configs/*.yaml, which document every key. Anything after the config name is
# forwarded to `generate` and overrides the file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${1:-review}"
shift || true
PV="conda run --no-capture-output -n physloc python -m physloc.cli"

echo "== generate: --config $CONFIG $* =="
$PV generate --config "$CONFIG" "$@"

REL=$($PV config-path --config "$CONFIG" 2>/dev/null || echo "out/release")

echo "== validate =="
$PV validate "$REL" || true

# Which severity bins this release actually contains.
BINS=$(find "$REL/clips" -mindepth 4 -maxdepth 4 -type d -name 'invalid_*' \
       | sed -n 's/.*_\(weak\|medium\|strong\)$/\1/p' | sort -u)
BINS=${BINS:-strong}

echo "== coverage: scenario x family lattice, one per severity =="
for BIN in $BINS; do
  $PV coverage "$REL" --severity "$BIN" \
      --out "$REL/coverage_$BIN.mp4" || true
done

echo "== sheets: every family of a scenario, every annotation, in one frame =="
for PAIR in $(find "$REL/clips" -mindepth 3 -maxdepth 3 -type d | sort); do
  for BIN in $BINS; do
    $PV sheet "$PAIR" --severity "$BIN" || true
  done
done

echo "== grids: the valid clip beside every severity of each family =="
for PAIR in $(find "$REL/clips" -mindepth 3 -maxdepth 3 -type d | sort); do
  FAMS=$(ls "$PAIR" | sed -n 's/^invalid_\(.*\)_\(weak\|medium\|strong\)$/\1/p' | sort -u)
  for FAM in $FAMS; do
    $PV grid "$PAIR" --family "$FAM" || true
  done
done

echo "== randomisation: is the sampler actually varying? (renders nothing) =="
$PV randomisation --seeds 24 || true

echo "== export: package it as a dataset -- shards, index, card, splits =="
$PV export "$REL" --outdir "out/hf/$(basename "$REL")" || true

echo
echo "done -> $REL"
echo "  coverage_strong.mp4            scenario x family lattice -- open this first"
echo "  out/hf/$(basename "$REL")/       packaged dataset: index.parquet plays the videos"
echo "  clips/*/*/*/sheet_strong.mp4   one scenario: every family x every annotation"
echo "  clips/*/*/*/grid_<family>.mp4  one family: every severity x every annotation"

#!/usr/bin/env bash
# Generate a release from a config, validate it, and build everything worth
# looking at.
#
#   bash scripts/run.sh                       # the review sweep (configs/review.yaml)
#   bash scripts/run.sh review_random         # every cell FOUR times -- does it vary?
#   bash scripts/run.sh v0_release            # the published dataset
#   bash scripts/run.sh review --tier release # extra flags pass straight through
#
#   # generate, package AND publish in one go:
#   PHYSLOC_PUSH_TO=samueleruf/physloc bash scripts/run.sh review ...
#
#   # a small run that still shows everything -- ~13 min:
#   bash scripts/run.sh review -n 45 --variants 5 \
#        --outdir out/physloc_mini --workdir out/work_mini
#
# ONE LEVEL AT A TIME. Each adds exactly one thing to the level below, so
# rendering them separately is how you find out WHICH thing broke. ~4 min each
# for one scenario at five variants; L3 is about twice L0 because six extra
# bodies is more geometry to shade.
#
#   bash scripts/run.sh review --complexity L0 --variants 5 --outdir out/L0 --workdir out/work_L0     # baseline: flat, static
#   bash scripts/run.sh review --complexity L1 ... # + camera moves on variant 4
#   bash scripts/run.sh review --complexity L2 ... # + materials, varied mass
#   bash scripts/run.sh review --complexity L3 ... # + 6 distractors
#
# Or all of them in one go:
#
#   for L in L0 L1 L2 L3; do
#     bash scripts/run.sh review --complexity $L --scenario drop --variants 5 \
#          --outdir out/$L --workdir out/work_$L
#   done
#
# PUBLISHING. Packaging always happens; uploading only when PHYSLOC_PUSH_TO is
# set, because packaging is local and repeatable and uploading is neither.
#
#   PHYSLOC_PUSH_TO=<user>/physloc-l3 \
#     bash scripts/run.sh review --complexity L3 --scenario drop --variants 5 \
#          --outdir out/L3 --workdir out/work_L3
#
#   PHYSLOC_PUSH_TO=<user>/physloc-mini PHYSLOC_PUSH_PRIVATE=1 \
#     bash scripts/run.sh review -n 45 --variants 5 \
#          --outdir out/physloc_mini --workdir out/work_mini
#
# A push REPLACES the card and index at that repo id, so give each artefact its
# own name rather than overwriting a release with a sample.
#
# 5 variants is the MINIMUM that shows camera motion (the moving variant is
# index 4 of each block) and the minimum that fills all three splits.
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

# Where the run will write. An `--outdir` typed on the command line beats the
# config, so it has to reach `config-path` too -- otherwise this script
# generates into the directory you asked for and then validates, films and
# packages a DIFFERENT one, silently. Only that flag is forwarded: the
# `config-path` subcommand does not accept `-n` or `--variants`.
OUTDIR_ARG=()
for ((i = 1; i <= $#; i++)); do
  if [ "${!i}" = "--outdir" ]; then
    j=$((i + 1)); OUTDIR_ARG=(--outdir "${!j}")
  fi
done
REL=$($PV config-path --config "$CONFIG" "${OUTDIR_ARG[@]}" 2>/dev/null \
      || echo "out/release")

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
# Publishing is opt-in and env-driven, not a flag, because everything after the
# config name is forwarded to `generate` and `generate` has no idea what a
# HuggingFace repo is. It is deliberately a separate switch from generating:
# packaging is local and repeatable, uploading is neither -- it puts the clips
# somewhere other people can fetch, index and cache them.
#
#   PHYSLOC_PUSH_TO=<user>/<dataset> bash scripts/run.sh review ...
#
# Add PHYSLOC_PUSH_PRIVATE=1 to create the repo private.
PUSH=()
if [ -n "${PHYSLOC_PUSH_TO:-}" ]; then
  PUSH=(--push-to "$PHYSLOC_PUSH_TO")
  [ -n "${PHYSLOC_PUSH_PRIVATE:-}" ] && PUSH+=(--private)
  echo "   -> will upload to $PHYSLOC_PUSH_TO"
fi
$PV export "$REL" --outdir "out/hf/$(basename "$REL")" "${PUSH[@]}" || true

echo
echo "done -> $REL"
echo "  coverage_strong.mp4            scenario x family lattice -- open this first"
echo "  out/hf/$(basename "$REL")/       packaged dataset: index.parquet plays the videos"
echo "  clips/*/*/*/sheet_strong.mp4   one scenario: every family x every annotation"
echo "  clips/*/*/*/grid_<family>.mp4  one family: every severity x every annotation"

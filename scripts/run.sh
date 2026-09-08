#!/usr/bin/env bash
# Generate a release from a config, validate it, and build everything worth
# looking at.
#
#   bash scripts/run.sh                       # the review sweep (configs/review.yaml)
#   bash scripts/run.sh review_severity       # every family, all three bins  ~8 min
#   bash scripts/run.sh review_conditions     # every difficulty condition    ~5 min
#
# PUBLISH EVERY RUN. Set the owner once and each sweep lands on the hub under
# its own name, so a review is a link rather than a directory on this box:
#
#   export PHYSLOC_PUSH_OWNER=samueleruf
#   bash scripts/run.sh review_severity       # -> samueleruf/physloc-review_severity
#   bash scripts/run.sh review_conditions     # -> samueleruf/physloc-review_conditions
#   bash scripts/run.sh review_ladder         # -> samueleruf/physloc-review_ladder
#   bash scripts/run.sh review_ladder         # the whole ladder in proportion
#   bash scripts/run.sh v0_release            # the published dataset
#   bash scripts/run.sh v0_L2                 # ...or one level of it at a time
#
# v0_L0..v0_L3 PARTITION v0_release: each carries the variants that level would
# get in a full run, so the four together produce exactly what the whole-ladder
# config produces. Generate them on separate machines and merge the trees --
# nothing collides, because the clip path is keyed by level and each level draws
# from its own seed block.
#   bash scripts/run.sh review --tier release # extra flags pass straight through
#
#   # generate, package AND publish in one go:
#   PHYSLOC_PUSH_TO=samueleruf/physloc bash scripts/run.sh review ...
#
# EACH CONFIG ANSWERS ONE QUESTION, which is why they are all minutes rather
# than hours. `review` covers every CELL; `review_severity` every FAMILY at
# every strength; `review_conditions` every CONDITION; `review_L*` every LEVEL.
# Price any of them first:
#
#   python -m physloc.cli taxonomy --config review_ladder
#
# ONE LEVEL AT A TIME. The ladder is scene realism -- L0 baseline, L1 materials,
# L2 HDRI, L3 GSO -- and each level adds exactly one thing, so rendering them
# separately is how you find out WHICH thing broke:
#
#   bash scripts/run.sh review_L0    # baseline: flat colours, one density
#   bash scripts/run.sh review_L1    # + materials, so mass is visible
#   bash scripts/run.sh review_L2    # + HDRI environment
#   bash scripts/run.sh review_L3    # + GSO objects, real 3D scans
#
# DIFFICULTY CONDITIONS ARE NOT LEVELS. Every clip carries one of five, applied
# inside every level, on a ten-variant cycle: six `standard`, then one each of
# `camera`, `distractors`, `multi` and `camera+multi`. So use TEN variants when
# you want to see them -- the plain clips come first, so a shorter run is
# entirely `standard`, which is the intended behaviour and useless for checking
# this axis. `review_conditions` is exactly that run.
#
# `distractors` and `multi` both add 3-10 extra objects, some moving and some
# still. They differ in how many bodies get INVALID PHYSICS: exactly one under
# `distractors`, and 2..N-1 under `multi`. Only the second asks "which of these
# is wrong".
#
#   PHYSLOC_CAMERA_MOTION=orbit bash scripts/run.sh review --scenario drop
#
# PUBLISHING. Packaging always happens; uploading only when PHYSLOC_PUSH_TO is
# set, because packaging is local and repeatable and uploading is neither.
#
#   PHYSLOC_PUSH_TO=<user>/physloc-l1 \
#     bash scripts/run.sh review --complexity L1 --scenario drop --variants 10 \
#          --outdir out/L1 --workdir out/work_L1
#
#   PHYSLOC_PUSH_TO=<user>/physloc-mini PHYSLOC_PUSH_PRIVATE=1 \
#     bash scripts/run.sh review -n 45 --variants 10 \
#          --outdir out/physloc_mini --workdir out/work_mini
#
# A push REPLACES the card and index at that repo id, so give each artefact its
# own name rather than overwriting a release with a sample.
#
# Tiers are `debug` and `release` -- two geometries, nothing more. Difficulty is
# the complexity ladder (README section 8), and `v0`/`v1` are what a published
# dataset is CALLED, set by the config's outdir.
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
# clips/<release>/<level>/<scenario>/<seed>/<clip> -- the level joined the key
# when one run started producing several levels, so these depths went up by one.
BINS=$(find "$REL/clips" -mindepth 5 -maxdepth 5 -type d -name 'invalid_*' \
       | sed -n 's/.*_\(weak\|medium\|strong\)$/\1/p' | sort -u)
BINS=${BINS:-strong}

echo "== coverage: scenario x family lattice, one per severity =="
for BIN in $BINS; do
  $PV coverage "$REL" --severity "$BIN" \
      --out "$REL/coverage_$BIN.mp4" || true
done

# Grids and sheets, collected into ONE folder rather than written beside the
# clips they came from. A twenty-pair run scatters them four levels deep across
# twenty directories, which puts the videos you most want to compare furthest
# apart. `viz` names them `<level>_<scenario>_<seed>_<family>.mp4` so the sort
# order is the reading order.
#
# Re-runnable on its own, against a finished run, without re-rendering:
#   python -m physloc.cli viz out/review_conditions --outdir out/inspect
echo "== grids and sheets -> $REL/viz =="
$PV viz "$REL" || true

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
#
# PHYSLOC_PUSH_OWNER is the shortcut: set it once and every run publishes to
# `<owner>/physloc-<config>`, so a review sweep lands somewhere you can open in
# a browser without naming a repo each time. An explicit PHYSLOC_PUSH_TO still
# wins, because a one-off artefact should not be able to overwrite a release
# just because the owner was exported in your shell.
#
#   export PHYSLOC_PUSH_OWNER=samueleruf
#   bash scripts/run.sh review_conditions      # -> samueleruf/physloc-review_conditions
PUSH=()
TARGET="${PHYSLOC_PUSH_TO:-}"
if [ -z "$TARGET" ] && [ -n "${PHYSLOC_PUSH_OWNER:-}" ]; then
  TARGET="$PHYSLOC_PUSH_OWNER/physloc-$(basename "$REL")"
fi
if [ -n "$TARGET" ]; then
  PUSH=(--push-to "$TARGET")
  [ -n "${PHYSLOC_PUSH_PRIVATE:-}" ] && PUSH+=(--private)
  echo "   -> will upload to $TARGET"
fi
$PV export "$REL" --outdir "out/hf/$(basename "$REL")" "${PUSH[@]}" || true

echo
echo "done -> $REL"
echo "  coverage_strong.mp4            scenario x family lattice -- open this first"
echo "  out/hf/$(basename "$REL")/       packaged dataset: index.parquet plays the videos"
echo "  clips/*/*/*/sheet_strong.mp4   one scenario: every family x every annotation"
echo "  clips/*/*/*/grid_<family>.mp4  one family: every severity x every annotation"

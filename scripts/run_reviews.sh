#!/usr/bin/env bash
# Every review config, back to back, with a log apiece. The overnight sweep:
# start it, and in the morning there is one directory and one HuggingFace
# dataset per thing worth checking.
#
#   bash scripts/run_reviews.sh                    # the standard set
#   bash scripts/run_reviews.sh review_L2 review   # or just these
#   bash scripts/run_reviews.sh --frames 37        # every config, full length
#
# `--frames N` / `--fps N` override the render geometry of every config --
# e.g. v0's 2.97 s at 12 fps is `--frames 37` (frames must be 4k+1) -- and
# send each run to its own directories (`out/review_L0_f37`,
# `out/work_review_L0_f37`), so a long sweep never resumes into, or overwrites,
# the short one.
#
# `PHYSLOC_PUSH_OWNER` makes each run publish to <owner>/physloc-<config>;
# leave it unset to keep everything local.
set -u
cd "$(dirname "$0")/.."

DIALS=()
SUFFIX=""
POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --frames) DIALS+=(--frames "$2"); SUFFIX="${SUFFIX}_f$2"; shift 2 ;;
    --fps)    DIALS+=(--fps "$2");    SUFFIX="${SUFFIX}_fps$2"; shift 2 ;;
    *)        POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

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
  EXTRA=()
  if [ -n "$SUFFIX" ]; then
    BASE=$(conda run --no-capture-output -n physloc python -m physloc.cli \
           config-path --config "$C" 2>/dev/null || echo "out/$C")
    NAME="$(basename "$BASE")$SUFFIX"
    EXTRA=("${DIALS[@]}" --outdir "out/$NAME" --workdir "out/work_$NAME")
  fi
  bash scripts/run.sh "$C" "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 \
    | tee "out/logs/$C$SUFFIX.txt"
  echo "=== $C exit ${PIPESTATUS[0]}  $(date) ==="
done

# ACROSS the sweep. Each run.sh compared a run with itself; the level ladder only
# exists across review_L0..L3, and the conditions are richest in
# review_conditions, so draw the structure videos once more over every root
# this sweep produced. Renders nothing new -- it reads finished clips.
ROOTS=()
for C in "${CONFIGS[@]}"; do
  BASE=$(conda run --no-capture-output -n physloc python -m physloc.cli \
         config-path --config "$C" 2>/dev/null || echo "out/$C")
  R="out/$(basename "$BASE")$SUFFIX"
  [ -d "$R/clips" ] && ROOTS+=("$R")
done
if [ ${#ROOTS[@]} -gt 1 ]; then
  OUT="out/compare$SUFFIX"
  echo "=== compare across ${ROOTS[*]} -> $OUT  $(date) ==="
  conda run --no-capture-output -n physloc python -m physloc.cli compare \
      "${ROOTS[@]}" --limit "${PHYSLOC_COMPARE_LIMIT:-20}" --outdir "$OUT" \
      2>&1 | tee "out/logs/compare$SUFFIX.txt" || true
fi
echo "ALL DONE $(date)"

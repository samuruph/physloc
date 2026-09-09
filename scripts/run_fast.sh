#!/usr/bin/env bash
# scripts/run.sh with the measured fast render backend pre-set.
#
#   bash scripts/run_fast.sh review_L0
#   bash scripts/run_fast.sh v0_release --resume
#
# Exactly equivalent to exporting two variables yourself and calling run.sh --
# it exists so the pair is one thing to remember rather than two to spell:
#
#   PHYSLOC_ADAPTIVE=1 PHYSLOC_DENOISER=off bash scripts/run.sh review_L0
#
# WHAT IT BUYS, AND WHAT IT COSTS. Measured at release geometry on an idle box,
# against a 512-spp reference (README section 13, "Render backend"):
#
#   NLM, no adaptive  -- today's default   7.80 s/frame   RMSE 0.15
#   adaptive, no denoiser, spp 128         3.72 s/frame   RMSE 0.37   2.10x
#
# Both errors are far below one 8-bit level, so this is a small realism cost
# and not a visible one. It is still a real one.
#
# THIS CHANGES THE PIXELS, so it is a decision to make BEFORE a run and never
# during: a release must not mix backends. Every clip records the backend it
# was rendered on at `spec.notes.render_backend` in plan.json, so a mixed tree
# can be detected after the fact -- but not repaired without re-rendering.
#
# Prefix identity is unaffected. Both twins render in one process under one
# setting, so the frames before `t_event` are bit-identical here exactly as
# they are on the default backend -- verified on a real pair, all seven passes.
#
# THE DOME LEVELS GAIN LESS: L2 measures 1.53x where L0 measures 2.10x, because
# a dome fills the frame with surface that never fully converges and adaptive
# sampling has less it can skip. L2 and L3 are 46% of a release, so a
# whole-ladder run comes out nearer 1.8x than 2.1x.
#
# For better quality at a smaller saving, override the denoiser:
#
#   PHYSLOC_DENOISER=OPENIMAGEDENOISE bash scripts/run_fast.sh review_L0
#
# `scripts/probe_backend.sh` reproduces every number above on your own box.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Adaptive sampling stops sampling a pixel once it has converged. `1` means
# "on, at Cycles' own threshold".
export PHYSLOC_ADAPTIVE="${PHYSLOC_ADAPTIVE:-1}"

# NLM is the legacy CPU denoiser kubric hardcodes, and it costs 2.4 s of every
# 7.8 s frame. Overridable, so `PHYSLOC_DENOISER=OPENIMAGEDENOISE
# bash scripts/run_fast.sh ...` picks the middle row of the table above.
export PHYSLOC_DENOISER="${PHYSLOC_DENOISER:-off}"

# SPP 128 RATHER THAN THE TIER'S 64, and it is nearly free: once sampling is
# adaptive the extra budget is only spent where the image is still noisy, so
# 64 -> 128 costs +0.15 s a frame and halves the worst-pixel error (22 -> 11).
# Anything the caller passes wins, including their own --spp.
SPP=(--spp 128)
for arg in "$@"; do
  [ "$arg" = "--spp" ] && SPP=()
done

echo "== fast backend: adaptive=$PHYSLOC_ADAPTIVE denoiser=$PHYSLOC_DENOISER ${SPP[*]} =="
exec bash scripts/run.sh "$@" "${SPP[@]}"

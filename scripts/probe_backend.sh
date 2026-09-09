#!/usr/bin/env bash
# What does a frame actually cost, and which part of it can we buy back?
#
# `probe_cost.py` already fits `T = a + b*spp` and finds only ~26% of a frame is
# SAMPLING. That was read as "there is no room to speed a render up". It really
# says "the room is not in sampling" -- 74% of every frame is something else,
# and nobody has ever measured what. This sweep names it.
#
#   bash scripts/probe_backend.sh              # L0 at release geometry
#   bash scripts/probe_backend.sh L2 512 64    # level, resolution, spp
#
# THE BOX MUST BE IDLE. Blender takes every core it can get, so a probe run
# beside a generation job measures the contention, not the render. Check first:
#
#   docker ps          # must be empty
#   uptime             # load average must be near zero
#
# Each line is one scene, four frames, printed by probe_cost.py as
#
#   COST L0 floor=as-is res=512 spp=64 ... per_frame=7.690 adaptive=... denoiser=...
#
# Read the `per_frame` column down the block. What each row is worth knowing:
#
#   baseline        what SECONDS_PER_CLIP is built on -- should reproduce 7.69
#                   at L0/512/64 on an idle box. If it does not, stop: nothing
#                   below this line means anything until it does.
#   denoise-off     NLM is the legacy CPU denoiser and kubric hardcodes it.
#                   baseline minus this is what denoising costs. If it is
#                   large, OIDN or no denoiser at higher spp is the cheapest
#                   win available and needs no new image.
#   oidn            OpenImageDenoise instead of NLM. Usually both faster AND
#                   better, but it is a different image, so check a frame.
#   adaptive        Cycles stops sampling a pixel once it has converged. These
#                   scenes -- a flat slab, a few primitives -- are what it is
#                   for. Expect the largest single win here.
#   adaptive+oidn   the two together, which is the configuration to actually
#                   consider adopting.
#
# WHAT TO DO WITH THE ANSWER. Anything adopted here changes the pixels, so it
# is a re-pin of the render, not a tweak: decide it BEFORE a release run and
# not during one. Prefix identity is unaffected either way -- both twins of a
# pair are rendered by one process with one setting.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

LEVEL="${1:-L0}"
RES="${2:-512}"
SPP="${3:-64}"
FRAMES="${4:-4}"

if [ -n "$(docker ps -q)" ]; then
  echo "REFUSING: $(docker ps -q | wc -l) container(s) are running." >&2
  echo "A render probe on a busy box measures contention, not the render." >&2
  exit 1
fi

echo "# level=$LEVEL res=$RES spp=$SPP frames=$FRAMES  load:$(uptime | sed 's/.*average://')"
echo "# each row is one scene; compare the per_frame column"

run() {
  local label="$1"; shift
  printf '%-16s ' "$label"
  # `env -u` so an unset dial is genuinely unset inside the container rather
  # than an empty string -- docker/kubric.sh forwards by name, and "" and unset
  # mean different things to _render_backend.
  env "$@" bash docker/kubric.sh physloc/render/probe_cost.py \
      --complexity "$LEVEL" --resolution "$RES" --spp "$SPP" \
      --frames "$FRAMES" 2>/dev/null | grep '^COST' || echo "FAILED"
}

run baseline        PHYSLOC_X=1
run denoise-off     PHYSLOC_DENOISER=off
run oidn            PHYSLOC_DENOISER=OPENIMAGEDENOISE
run adaptive        PHYSLOC_ADAPTIVE=1
run adaptive+oidn   PHYSLOC_ADAPTIVE=1 PHYSLOC_DENOISER=OPENIMAGEDENOISE

echo
echo "# baseline minus denoise-off = what NLM costs."
echo "# baseline minus adaptive    = what sampling every pixel to the end costs."
echo "# If adaptive wins big, check a FRAME before adopting it: the threshold"
echo "# trades noise for time, and a noisy severity map is worse than a slow one."

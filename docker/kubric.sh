#!/usr/bin/env bash
# Run a worker script inside the pinned Kubric image.
#
#   bash docker/kubric.sh physloc/render/worker_smoke.py --resolution 512
#
# The pattern is "your script, their container": the image already contains a
# complete Kubric + Blender install, so we mount this repo at /kubric and run our
# own file against it. Nothing is vendored. See docs/PLAN.md Part 0.
#
#   --user   makes rendered output land owned by you, not root
#   --volume is both how the script gets in and how the frames get out
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIGEST_FILE="$REPO_ROOT/docker/IMAGE_DIGEST"
IMAGE=""

if [ $# -lt 1 ]; then
  echo "usage: docker/kubric.sh <script.py> [args...]" >&2
  exit 64
fi

# Prefer the pinned digest; fall back to the tag with a warning so a fresh
# clone still works before the first pull.
if [ -f "$DIGEST_FILE" ]; then
  IMAGE="$(grep -m1 '^kubricdockerhub/kubruntu@sha256:' "$DIGEST_FILE" || true)"
fi
# An explicit override wins over the pin. This is how the GPU image
# (docker/Dockerfile.gpu) gets used WITHOUT unpinning the CPU one: the digest
# in docker/IMAGE_DIGEST stays exactly as it is, and a GPU experiment is one
# environment variable rather than an edit to a file every run reads.
if [ -n "${PHYSLOC_IMAGE:-}" ]; then
  IMAGE="$PHYSLOC_IMAGE"
fi

if [ -z "$IMAGE" ]; then
  echo "warn: no pinned digest in docker/IMAGE_DIGEST, using :latest (not reproducible)" >&2
  IMAGE="kubricdockerhub/kubruntu"
fi

# Debug dials the host sets and the container has to see. `--env NAME` with no
# value forwards the host's value, and forwards nothing when it is unset, so an
# unset dial cannot silently become an empty string inside the container.
GPU_ARGS=()
# Only ask docker for the card when a GPU render was actually requested.
# The pinned CPU image ignores it, but `--gpus` fails outright on a host
# without the nvidia runtime, and that must not break a CPU run.
if [ -n "${PHYSLOC_GPU:-}" ] && [ "${PHYSLOC_GPU}" != "0" ]; then
  GPU_ARGS=(--gpus all)
fi

exec docker run --rm --interactive \
  "${GPU_ARGS[@]}" \
  --user "$(id -u):$(id -g)" \
  --env PHYSLOC_CAMERA_MOTION \
  --env PHYSLOC_ADAPTIVE \
  --env PHYSLOC_DENOISER \
  --env PHYSLOC_GPU \
  --env PHYSLOC_GPU_BACKEND \
  --volume "$REPO_ROOT:/kubric" \
  --workdir /kubric \
  "$IMAGE" \
  python3 "$@"

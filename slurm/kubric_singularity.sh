#!/usr/bin/env bash
# The cluster's version of docker/kubric.sh: run a worker script inside the
# pinned Kubric image, as a Singularity .sif instead of a docker container.
#
#   bash slurm/kubric_singularity.sh scripts/probes/worker_smoke.py --frames 4
#
# `generate` calls this instead of docker/kubric.sh when PHYSLOC_LAUNCHER
# points here (slurm/env.sh sets it). Same image, same digest, same pixels.
#
# What docker needed extra flags for, SLURM already does: the task's cores and
# memory are its own, the process runs as you, and scancel reaches it.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${PHYSLOC_SIF:-${SINGULARITY_ALLOWED_DIR:-}/kubric.sif}"
if [ ! -f "$SIF" ]; then
  echo "kubric_singularity.sh: no image at '$SIF' (set PHYSLOC_SIF)" >&2
  exit 66
fi

# The repo is mounted at /kubric, as with docker. --cleanenv starts the
# container with an empty environment (so the host's conda Python cannot leak
# in), and the dials below are passed in by name -- only those that are set.
ARGS=(--cleanenv --pwd /kubric --bind "$REPO_ROOT:/kubric")
for v in PHYSLOC_THREADS PHYSLOC_ASSETS PHYSLOC_CAMERA_MOTION PHYSLOC_ADAPTIVE \
         PHYSLOC_DENOISER PHYSLOC_SKIP_RENDER PHYSLOC_DEBUG; do
  if [ -n "${!v:-}" ]; then ARGS+=(--env "$v=${!v}"); fi
done

# Folders the worker reads or writes, mounted at their own paths so a path
# means the same file inside and outside: the asset mirror, the data folder,
# and the task's scratch. The scratch also gives the container its TMPDIR
# (Kubric unpacks every asset there) and its HOME (Blender writes config).
for d in "${PHYSLOC_ASSETS:-}" "${PHYSLOC_DATA:-}"; do
  if [ -n "$d" ] && [ -d "$d" ]; then ARGS+=(--bind "$d"); fi
done
if [ -n "${PHYSLOC_SCRATCH:-}" ]; then
  mkdir -p "$PHYSLOC_SCRATCH/tmp" "$PHYSLOC_SCRATCH/home"
  ARGS+=(--bind "$PHYSLOC_SCRATCH" --env "TMPDIR=$PHYSLOC_SCRATCH/tmp"
         --home "$PHYSLOC_SCRATCH/home")
fi

exec singularity exec "${ARGS[@]}" "$SIF" python3 "$@"

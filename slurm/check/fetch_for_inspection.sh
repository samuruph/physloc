#!/usr/bin/env bash
# Copy a release from Jean Zay to YOUR WORKSTATION for inspection -- everything
# except the data.h5 files (the per-frame arrays, ~all of the 1.2 TB), so you get
# every video, sample.json, stats figure, audit and log at a small fraction of
# the size, with the folder structure unchanged.
#
# RUN IT ON YOUR WORKSTATION, not on Jean Zay: Jean Zay cannot connect out to
# your machine, your machine can connect in (as for ssh). This file does not
# need the repo -- copy it anywhere and run:
#
#   bash fetch_for_inspection.sh                 # copy (resumes if interrupted)
#   bash fetch_for_inspection.sh --dry-run       # only print what it would copy, and how much
#   bash fetch_for_inspection.sh --with-h5 <sample folder under the release>
#                                                # also fetch one sample's data.h5, for the GUI
#
# Re-running copies only what is new or changed, so it also keeps a copy up to
# date while a release is still being generated.
set -euo pipefail

# ---- your values: set them in your shell, never in this file --------------------
# They identify you and your machines, so they stay out of the repository. Put
# them in your workstation's ~/.bashrc (or type them before running):
#
#   export JZ_LOGIN=<you>@jean-zay.idris.fr       # how you ssh to Jean Zay
#   export JZ_RELEASE=<release folder on Jean Zay> # $PHYSLOC_OUTDIR there (echo it after
#                                                  #   `source slurm/env.sh`)
#   export INSPECT_DEST=<folder on this machine>   # where the copy goes
for v in JZ_LOGIN JZ_RELEASE INSPECT_DEST; do
  if [ -z "${!v:-}" ]; then
    echo "$v is not set -- see the top of this file" >&2
    exit 2
  fi
done
REMOTE="$JZ_RELEASE"
DEST="$INSPECT_DEST"
# ---------------------------------------------------------------------------------

# ONE PROGRESS LINE for the whole copy where rsync can (3.1+): a release is ~30 000
# files, and listing each one buries the progress. macOS ships an older rsync, which
# falls back to per-file progress. --partial resumes a file the copy broke off in.
if rsync --help 2>&1 | grep -q -- "--info"; then
  PROGRESS=(--info=progress2 --no-inc-recursive --partial)
else
  PROGRESS=(-P)
fi

case "${1:-}" in
  --dry-run)
    echo "Would copy from $JZ_LOGIN:$REMOTE to $DEST (data.h5 excluded):"
    rsync -a --dry-run --stats --exclude='data.h5' "$JZ_LOGIN:$REMOTE/" "$DEST/" \
      | grep -E "Number of (regular )?files|Total file size|Total transferred file size"
    ;;
  --with-h5)
    sample="${2:?usage: --with-h5 <sample folder, relative to the release>}"
    mkdir -p "$DEST/$sample"
    rsync -a "${PROGRESS[@]}" "$JZ_LOGIN:$REMOTE/$sample/data.h5" "$DEST/$sample/"
    echo "Open it with:  python test_dataset_loader.py $DEST/$sample --gui"
    ;;
  "")
    mkdir -p "$DEST"
    # -a keeps the structure; --exclude skips every data.h5, wherever it is.
    rsync -a "${PROGRESS[@]}" --exclude='data.h5' "$JZ_LOGIN:$REMOTE/" "$DEST/"
    echo
    echo "Done -> $DEST   ($(du -sh "$DEST" | cut -f1) on disk)"
    echo "Start with: audit.txt, validate.json, stats/, viz/coverage_strong.mp4;"
    echo "every sample's rgb.mp4, overlay.mp4 and sample.json are under samples/."
    ;;
  *)
    echo "unknown option: $1  (use --dry-run or --with-h5 <sample folder>)" >&2
    exit 64
    ;;
esac

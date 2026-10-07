# PhysLoc on Jean Zay -- the ONE file to edit. Only variables, nothing runs.
#
# Source it in your login shell before any sbatch:   source slurm/env.sh
# Every .slurm script here sources it too.

# ---- 1. edit these -----------------------------------------------------------
export SBATCH_ACCOUNT="${IDRPROJ}@cpu"      # your hours (`idrproj`); sbatch reads this
export SALLOC_ACCOUNT="$SBATCH_ACCOUNT"   # ...salloc reads this one
export SLURM_ACCOUNT="$SBATCH_ACCOUNT"    # ...and srun this one
export PHYSLOC_CONFIG="v0_release"        # which config to generate
export PHYSLOC_RELEASE="physloc_v0"       # its outdir name (configs/<config>.yaml)
export PHYSLOC_CONDA_MODULE="miniforge"   # `module avail conda` -- only for setup
export PHYSLOC_HF_REPO="samueleruf/physloc_v0"  # where publish.slurm uploads (HuggingFace dataset)
export PHYSLOC_HF_PRIVATE=1                 # 1 = create the repo private; 0 = public
# ------------------------------------------------------------------------------

# ---- 2. where things live (usually fine as is) -------------------------------
export PHYSLOC_CONDA_ENV="${WORK:-$HOME}/conda/physloc"   # host Python env
export PHYSLOC_ASSETS="${WORK:-$HOME}/kubric_assets"      # asset mirror (no internet on nodes)
export PHYSLOC_DATA="${SCRATCH:-$HOME}/physloc"           # where releases are written
export PHYSLOC_OUTDIR="$PHYSLOC_DATA/$PHYSLOC_RELEASE"    # this release
export PHYSLOC_STORE="${STORE:-$HOME}/physloc"            # final archive (tar files)
export PHYSLOC_SIF="${SINGULARITY_ALLOWED_DIR:-}/kubric.sif"

# ---- 3. plumbing (do not edit) -----------------------------------------------
# Render with Singularity instead of docker, and use the host env's Python.
export PHYSLOC_LAUNCHER="slurm/kubric_singularity.sh"
export PATH="$PHYSLOC_CONDA_ENV/bin:$PATH"

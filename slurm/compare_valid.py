"""Are a refill's valid clips the release's own, frame for frame?

A refill (`render_refill.slurm`) renders each scene's valid twin again and puts
only the new INVALID clips into the release, next to the release's old valid
one. That pairing is a twin only if the two valid renders agree -- prefix
identity, non-negotiable 1. Comparing `rgb.mp4` bytes cannot answer it (an
encode can differ where the frames do not), so this compares what the clips
contain, in three groups, which also says WHERE a difference comes from:

  physics  object positions and orientations, camera     -> the simulation
  passes   depth, segmentation, flow, normals, coords    -> the renderer
  rgb      decoded frames                                -> renderer, denoiser, encoder

    source slurm/env.sh
    python slurm/compare_valid.py --family solidity
    python slurm/compare_valid.py --family solidity --newer-than <file>   # only those
"""
import argparse
import glob
import os
import sys

import numpy as np

GROUPS = {
    "physics": ["objects/positions", "objects/quaternions", "camera/positions",
                "camera/quaternions"],
    "passes": ["observations/depth", "observations/segmentation",
               "observations/forward_flow", "observations/normal",
               "observations/object_coordinates"],
}


def _frames(path):
    import imageio.v2 as imageio

    with imageio.get_reader(path) as r:
        return np.stack([np.asarray(f) for f in r])


def _first_and_max(a, b, t_axis=0):
    """(first differing frame, max |a - b|), or None if equal."""
    if a.shape != b.shape:
        return ("-", "shape %s vs %s" % (a.shape, b.shape))
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if not diff.any():
        return None
    per = np.moveaxis(diff, t_axis, 0).reshape(diff.shape[t_axis], -1).max(axis=1)
    return (int(np.flatnonzero(per)[0]), float(diff.max()))


def compare(a_dir, b_dir):
    import h5py

    out = {}
    with h5py.File(os.path.join(a_dir, "data.h5"), "r") as fa, \
            h5py.File(os.path.join(b_dir, "data.h5"), "r") as fb:
        for group, names in GROUPS.items():
            worst = []
            for name in names:
                if name in fa and name in fb:
                    # per-object arrays are [object, frame, ...]
                    d = _first_and_max(fa[name][()], fb[name][()],
                                       1 if name.startswith("objects/") else 0)
                    if d is not None:
                        worst.append("%s (from frame %s, max %s)" % (
                            name.split("/")[-1], d[0], d[1] if isinstance(d[1], str)
                            else "%.3g" % d[1]))
            out[group] = worst
    d = _first_and_max(_frames(os.path.join(a_dir, "rgb.mp4")),
                       _frames(os.path.join(b_dir, "rgb.mp4")))
    out["rgb"] = [] if d is None else ["frames (from frame %s, max %s of 255)" % d]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", required=True)
    ap.add_argument("--release", default=os.environ.get("PHYSLOC_OUTDIR"))
    ap.add_argument("--refill")
    ap.add_argument("--newer-than", help="only refill clips written after this file")
    a = ap.parse_args()
    name = os.path.basename(os.path.normpath(a.release))
    refill = a.refill or os.path.join(os.environ.get("PHYSLOC_DATA", ""),
                                      "refill_" + a.family, name)
    since = os.path.getmtime(a.newer_than) if a.newer_than else None
    n_same = 0
    for new in sorted(glob.glob(os.path.join(refill, "samples", name, "*", "*", "*", "valid"))):
        if since is not None and os.path.getmtime(new) <= since:
            continue
        old = os.path.join(a.release, os.path.relpath(new, refill))
        scene = os.path.relpath(os.path.dirname(new), os.path.join(refill, "samples", name))
        if not os.path.isdir(old):
            print("NEW   %s (not in the release)" % scene)
            continue
        res = compare(old, new)
        if not any(res.values()):
            n_same += 1
            print("same  %s" % scene)
            continue
        print("DIFF  %s" % scene)
        for group in ("physics", "passes", "rgb"):
            if res[group]:
                print("        %-7s %s" % (group, "; ".join(res[group])))
    print("\n%d identical" % n_same)
    return 0


if __name__ == "__main__":
    sys.exit(main())

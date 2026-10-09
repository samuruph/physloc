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
    python slurm/check/compare_valid.py --family solidity
    python slurm/check/compare_valid.py --family solidity --newer-than <file>   # only those
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


def _first(a, b, t_axis, n_frames):
    """(first differing frame, max |a - b| on that frame), or None if equal.

    FRAME BY FRAME, from the HDF5 datasets or video readers, never whole
    arrays: a 512^2 release clip's flow and normal passes are hundreds of MB
    each, and loading two of them with their difference cost GBs per scene --
    enough to be killed on a login node, silently under `| tail`.
    """
    for t in range(n_frames):
        x, y = a(t), b(t)
        if x.shape != y.shape:
            return (t, "shape %s vs %s" % (x.shape, y.shape))
        if not _same(x, y):
            diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
            where = np.isnan(x) != np.isnan(y) if x.dtype.kind == "f" else None
            if where is not None and where.any():
                return (t, "NaN in one but not the other")
            return (t, "%.3g" % float(np.nanmax(diff)))
    return None


def _same(x, y):
    """Equal, with NaN equal to NaN: a body absent from a frame has a NaN pose
    in BOTH twins, and plain `array_equal` called every crowded scene different."""
    if x.dtype.kind == "f":
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def compare(a_dir, b_dir):
    import h5py
    import imageio.v2 as imageio

    out = {}
    with h5py.File(os.path.join(a_dir, "data.h5"), "r") as fa, \
            h5py.File(os.path.join(b_dir, "data.h5"), "r") as fb:
        for group, names in GROUPS.items():
            worst = []
            for name in names:
                if name not in fa or name not in fb:
                    continue
                da, db = fa[name], fb[name]
                if da.shape != db.shape:
                    worst.append("%s (shape %s vs %s)" % (name.split("/")[-1],
                                                          da.shape, db.shape))
                    continue
                if name.startswith("objects/"):           # [object, frame, ...]
                    d = _first(lambda t: da[:, t], lambda t: db[:, t], 1, da.shape[1])
                else:
                    d = _first(lambda t: da[t], lambda t: db[t], 0, da.shape[0])
                if d is not None:
                    worst.append("%s (from frame %s, max %s)"
                                 % (name.split("/")[-1], d[0], d[1]))
            out[group] = worst
    with imageio.get_reader(os.path.join(a_dir, "rgb.mp4")) as ra, \
            imageio.get_reader(os.path.join(b_dir, "rgb.mp4")) as rb:
        fa_, fb_ = iter(ra), iter(rb)
        out["rgb"] = []
        t = 0
        while True:
            x, y = next(fa_, None), next(fb_, None)
            if x is None and y is None:
                break
            if x is None or y is None:
                out["rgb"] = ["frame count differs from frame %d" % t]
                break
            x, y = np.asarray(x), np.asarray(y)
            if not np.array_equal(x, y):
                out["rgb"] = ["frames (from frame %d, max %d of 255)"
                              % (t, int(np.abs(x.astype(int) - y.astype(int)).max()))]
                break
            t += 1
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
            print("same  %s" % scene, flush=True)
            continue
        print("DIFF  %s" % scene, flush=True)
        for group in ("physics", "passes", "rgb"):
            if res[group]:
                print("        %-7s %s" % (group, "; ".join(res[group])), flush=True)
    print("\n%d identical" % n_same)
    return 0


if __name__ == "__main__":
    sys.exit(main())

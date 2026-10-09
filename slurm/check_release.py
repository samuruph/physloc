"""Checks over a whole release that `validate` and `status.py` do not make.

`validate` checks each sample's schema; `status.py` counts what is missing.
Neither looks at whether what IS there is sound:

  empty    sample folders with no sample.json -- left by a clip that crashed
           while being written
  prefix   invalid samples whose prefix was NOT verified identical to their
           valid twin (`provenance.prefix_identical_verified`) -- non-negotiable 1
  fell     valid clips in which a moving body ends up well below the floor:
           it left the slab and fell out of the world in the LAWFUL clip, so
           every family on that scene inherits it

Read-only. Prints counts and the first few of each; --list prints them all.

    source slurm/env.sh
    python slurm/check_release.py                 # $PHYSLOC_OUTDIR
    python slurm/check_release.py --list > check_release.txt
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

#: How far below the floor's top a body must get to count as fallen, in m.
#: A lawful body rests ON the floor; even a body sunk by `solidity` belongs to
#: an invalid clip, and only valid clips are checked.
FELL_BELOW = 0.5


def fallen(sample_dir, doc):
    """[(name, lowest z, floor top)] for moving bodies below the floor."""
    import h5py

    objects = doc.get("objects") or []
    floor = [k for k, o in enumerate(objects) if o.get("role") == "floor"]
    with h5py.File(os.path.join(sample_dir, "data.h5"), "r") as h5:
        pos = h5["objects/positions"][()]                    # [N, T, 3]
    if floor:
        k = floor[0]
        top = float(pos[k, 0, 2]) + float((objects[k].get("render") or {})
                                          .get("scale", [0, 0, 0])[2])
    else:
        top = 0.0
    out = []
    for k, o in enumerate(objects):
        phys = o.get("physics") or {}
        if phys.get("static") or phys.get("dormant") or k >= pos.shape[0]:
            continue
        low = float(np.nanmin(pos[k, :, 2]))
        if low < top - FELL_BELOW:
            out.append((str(o.get("name")), low, top))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", nargs="?", default=os.environ.get("PHYSLOC_OUTDIR"))
    ap.add_argument("--list", action="store_true", help="print every finding")
    a = ap.parse_args()
    if not a.root:
        sys.exit("no release folder: source slurm/env.sh, or pass it")
    show = None if a.list else 5
    samples = os.path.join(a.root, "samples")
    empty, unverified, fell = [], [], []
    n_valid = n_invalid = 0
    for d in sorted(glob.glob(os.path.join(samples, "*", "L*", "*", "*", "*"))):
        if not os.path.isdir(d):
            continue
        rel = os.path.relpath(d, samples)
        meta = os.path.join(d, "sample.json")
        if not os.path.exists(meta):
            empty.append(rel)
            continue
        with open(meta) as fh:
            doc = json.load(fh)
        if os.path.basename(d) == "valid":
            n_valid += 1
            try:
                for name, low, top in fallen(d, doc):
                    fell.append("%s  %s reaches z=%.2f (floor top %.2f)"
                                % (os.path.dirname(rel), name, low, top))
            except (OSError, KeyError) as exc:
                fell.append("%s  unreadable: %s" % (rel, exc))
        else:
            n_invalid += 1
            if not (doc.get("provenance") or {}).get("prefix_identical_verified", False):
                unverified.append(rel)

    print("release %s: %d valid, %d invalid samples\n" % (a.root, n_valid, n_invalid))
    for title, items in (
            ("empty sample folders (no sample.json)", empty),
            ("invalid samples with an UNVERIFIED prefix", unverified),
            ("bodies falling out of the world in a VALID clip "
             "(lines are scene, body)", fell)):
        print("%5d  %s" % (len(items), title))
        for line in items[:show]:
            print("         %s" % line)
        if show is not None and len(items) > show:
            print("         ... %d more (--list)" % (len(items) - show))
    return 0


if __name__ == "__main__":
    sys.exit(main())

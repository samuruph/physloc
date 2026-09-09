"""Where do `annotate.difficulty`'s thresholds come from? Here, once.

Reads every invalid clip in a run, computes the seven factors exactly as the
annotator will, and prints the tertiles of each. Thresholds set at the tertiles
give each factor roughly a third of the corpus in each band; the worst-factor
rule then makes `easy` the rarest label overall, which is the intent -- an easy
clip should be easy on EVERY axis.

    conda activate physloc
    python scripts/fit_difficulty.py out/review_L0 out/review_conditions ...

Run it on a corpus that spans the conditions, or the clutter and camera columns
are degenerate: `review_conditions` is the one config that contains all five.

**The output is not applied automatically.** Paste the numbers into `FACTORS`
and say in the commit what corpus they came from. A benchmark whose difficulty
labels move between releases cannot be compared with itself, so these are
frozen once published -- refitting is a new release, not a bug fix.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physloc.annotate import difficulty as D  # noqa: E402


def load(root):
    """Every invalid clip under `root`, as (meta, vmask, seg)."""
    for meta_path in sorted(glob.glob(
            os.path.join(root, "clips", "**", "meta.json"), recursive=True)):
        with open(meta_path) as fh:
            meta = json.load(fh)
        if not meta.get("violation"):
            continue
        cdir = os.path.dirname(meta_path)
        vmask = seg = None
        p = os.path.join(cdir, "violation_mask.npz")
        if os.path.exists(p):
            with np.load(p) as z:
                vmask = z[list(z.keys())[0]]
        p = os.path.join(cdir, "seg.npz")
        if os.path.exists(p):
            with np.load(p) as z:
                seg = z[list(z.keys())[0]]
        yield meta, vmask, seg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    a = ap.parse_args()

    rows = []
    for root in a.roots:
        for meta, vmask, seg in load(root):
            rows.append(D.measure(meta, vmask, seg))
    if not rows:
        print("no invalid clips found", file=sys.stderr)
        return 1

    print("%d invalid clips from %s\n" % (len(rows), ", ".join(a.roots)))
    print("%-11s %-6s %8s %8s %8s %8s %8s  %s"
          % ("factor", "easier", "min", "33%", "50%", "67%", "max", "suggested"))
    for f in D.FACTORS:
        vals = np.asarray([r[f.name] for r in rows
                           if r[f.name] is not None], np.float64)
        if not vals.size:
            print("%-11s  no values" % f.name)
            continue
        lo, mid, hi = np.percentile(vals, [33.3, 50.0, 66.7])
        # The EASY boundary is the tertile on the easy side, and the moderate
        # boundary the one on the hard side -- which end that is depends on the
        # factor, so read it off `easier` rather than assuming.
        easy, moderate = (hi, lo) if f.easier == "high" else (lo, hi)
        print("%-11s %-6s %8.4f %8.4f %8.4f %8.4f %8.4f  easy=%.4g moderate=%.4g"
              % (f.name, f.easier, vals.min(), lo, mid, hi, vals.max(),
                 easy, moderate))

    # What the CURRENT table does to this corpus, so a refit can be compared
    # against what it replaces rather than judged on its own.
    print()
    counts = {lv: 0 for lv in D.LEVELS}
    binding = {}
    for meta, vmask, seg in (x for root in a.roots for x in load(root)):
        got = D.assess(meta, vmask, seg)
        counts[got["level"]] += 1
        for name in got["binding_factors"]:
            binding[name] = binding.get(name, 0) + 1
    total = max(1, sum(counts.values()))
    print("under the thresholds in FACTORS today:")
    for lv in D.LEVELS:
        print("   %-9s %4d  %5.1f%%" % (lv, counts[lv],
                                        100.0 * counts[lv] / total))
    print("   binding factors: %s" % ", ".join(
        "%s %d" % kv for kv in sorted(binding.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

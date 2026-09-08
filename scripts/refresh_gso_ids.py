"""Bake a curated GSO id list -- WITH BOUNDS -- into `physloc/scenarios/_gso.py`.

The HDRI list needs only ids, because an environment map has no geometry. A GSO
object does, and the host sampler needs it: where a body stands, how far a
distractor must clear it, what its support surface is and how big it is drawn
are all decided on the host, with no container to ask.

So this dumps `id -> (bounds, category)` for a curated subset, and the sampler
scales each asset the way MOVi does -- `scale / max(bounds[1] - bounds[0])`,
normalising the longest axis to a chosen size in metres
(`movi_c_worker.py:167-170`).

    bash docker/kubric.sh scripts/refresh_gso_ids.py --report
    bash docker/kubric.sh scripts/refresh_gso_ids.py --write --count 48
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "physloc", "scenarios", "_gso.py")
GSO = "gs://kubric-public/assets/GSO/GSO.json"

#: Squat, convex-ish, roughly isotropic assets. A physics-violation dataset
#: needs bodies whose lawful motion a viewer can predict -- something that
#: topples in a way nobody expects reads as the violation.
MAX_ASPECT = 2.6

#: At most this many assets from any one group. Without a cap the selection
#: collapses onto whichever family happens to be most isotropic.
PER_GROUP = 2


def _pyliteral(obj) -> str:
    """A Python literal, not JSON.

    `json.dumps` writes `false`, and the file it lands in is imported -- the
    first version raised `NameError: name 'false' is not defined` on import.
    """
    import pprint

    return pprint.pformat(obj, indent=4, width=78, sort_dicts=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=48)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    import kubric as kb

    src = kb.AssetSource.from_manifest(GSO)
    ids = sorted(getattr(src, "_assets", {}) or {})
    print("manifest %s\nassets: %d" % (GSO, len(ids)))
    if not ids:
        print("no assets -- the manifest shape changed", file=sys.stderr)
        return 1

    train, test = src.get_test_split(fraction=0.1)
    print("kubric's own split: %d train / %d held out" % (len(train), len(test)))

    # STRAIGHT OFF THE MANIFEST. `src.create(asset_id=...)` DOWNLOADS the
    # asset, and doing that for all 1030 to read a bounding box is a quarter of
    # an hour of network for six numbers apiece. The manifest already carries
    # `kwargs.bounds`, `kwargs.mass`, `metadata.category` and the licence --
    # which is also the one this repo is required to record.
    rows = []
    for aid in ids:
        entry = src._assets[aid]
        kw = entry.get("kwargs") or {}
        md = entry.get("metadata") or {}
        b = kw.get("bounds")
        if not b:
            continue
        lo, hi = np.asarray(b, np.float64)
        ext = hi - lo
        if float(ext.min()) <= 1e-6:
            continue
        rows.append((aid, float(ext.max() / ext.min()), lo.tolist(), hi.tolist(),
                     aid in set(test), str(md.get("category") or "unknown"),
                     str(entry.get("license") or "unknown"),
                     float(kw.get("mass") or 0.0)))

    # ROUND-ROBIN ACROSS CATEGORIES, and cap each. Sorting by aspect alone and
    # taking the top 48 gave THIRTEEN near-identical Ecoforms plant pots -- the
    # most isotropic objects in the set are all the same object -- which is one
    # asset repeated, not a shape axis. The name is the fallback grouping
    # because a third of GSO carries no category at all, and its first token
    # ("Ecoforms", "Nescafe") separates the families that matter here.
    eligible = sorted((r for r in rows if r[1] <= MAX_ASPECT),
                      key=lambda r: r[1])
    buckets = {}
    for r in eligible:
        cat = r[5] if r[5] not in ("None", "unknown") else r[0].split("_")[0]
        buckets.setdefault(cat, []).append(r)
    keep, round_no = [], 0
    while len(keep) < a.count and round_no < PER_GROUP:
        added = False
        for cat in sorted(buckets):
            if len(buckets[cat]) > round_no and len(keep) < a.count:
                keep.append(buckets[cat][round_no])
                added = True
        if not added:
            break
        round_no += 1
    keep.sort(key=lambda r: r[0])
    print("usable: %d, within aspect %.1f: %d, groups: %d, keeping %d"
          % (len(rows), MAX_ASPECT, len(eligible), len(buckets), len(keep)))
    if a.report:
        import collections
        cats = collections.Counter(r[5] for r in keep)
        print("categories kept: %s" % dict(cats.most_common(12)))
        print("licences kept  : %s"
              % dict(collections.Counter(r[6] for r in keep)))
        for aid, asp, lo, hi, held, cat, lic, mass in keep[:16]:
            print("   %-32s %-14s aspect %.2f  extent %s%s"
                  % (aid, cat[:14], asp,
                     [round(h - l, 3) for l, h in zip(lo, hi)],
                     "  [held-out]" if held else ""))
    if not a.write:
        return 0

    body = {aid: {"bounds": [[round(x, 6) for x in lo],
                             [round(x, 6) for x in hi]],
                  "category": cat, "license": lic, "held_out": bool(held)}
            for aid, _, lo, hi, held, cat, lic, _m in keep}
    text = '''"""Deterministic GSO choice, with the geometry the HOST needs.

Regenerate with `bash docker/kubric.sh scripts/refresh_gso_ids.py --write`.

The manifest lives in the container, but scenario sampling runs on the host --
where a body stands, how far a distractor must clear it and what its support
surface is are all decided before anything is rendered. An environment map has
no geometry so `_hdri.py` bakes ids alone; a GSO object does, so this bakes its
BOUNDS as well.

Curated: squat and roughly isotropic (longest axis at most %.1f x the
shortest). A physics dataset needs bodies whose lawful motion a viewer can
predict -- something that topples unexpectedly reads as the violation.

`held_out` marks the assets in Kubric's own 10%% test split, kept so a release
can align its held-out clips with objects the community also treats as unseen.
`license` is carried because non-negotiable 7 requires every asset to ship one,
and `validate` enforces it -- recorded when the source is enabled, not at
release.
"""
from __future__ import annotations

from typing import Dict, List

GSO_ASSETS: Dict[str, dict] = %s

GSO_IDS: List[str] = sorted(GSO_ASSETS)
''' % (MAX_ASPECT, _pyliteral(body))
    open(OUT, "w").write(text)
    print("wrote %s with %d assets" % (OUT, len(body)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Every (scenario, family) cell's weak/medium/strong severity, one line each.

Reads `sample.json` under a release root -- no loader, no container, seconds
over a full release -- and prints one row per cell with its three
peak_severity values and whether they climb weak -> medium -> strong. This is
the same question `physloc audit`'s ladder check asks, but for every cell,
not just the ones that fail it: a full listing to check by eye rather than a
list of exceptions to trust.

    python scripts/severity_report.py out/_final
    python scripts/severity_report.py out/_final --csv out/_final/severity.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

BINS = ("weak", "medium", "strong")
SLACK = 0.02


def collect(root: str):
    rows = {}
    pattern = os.path.join(root, "samples", "**", "invalid_*", "sample.json")
    for path in glob.glob(pattern, recursive=True):
        with open(path) as fh:
            doc = json.load(fh)
        scene = doc.get("scene", {})
        violation = doc.get("violation", {})
        scenario = scene.get("scenario")
        family = scene.get("family")
        severity_bin = violation.get("severity_bin")
        if not (scenario and family and severity_bin in BINS):
            continue
        violators = violation.get("violators") or []
        peak = max((float(v.get("peak_severity") or 0.0) for v in violators),
                   default=0.0)
        key = (scenario, family)
        rows.setdefault(key, {})[severity_bin] = peak
    return rows


def verdict(vals) -> str:
    if any(b not in vals for b in BINS):
        return "incomplete"
    w, m, s = (vals[b] for b in BINS)
    if w > m + SLACK or m > s + SLACK:
        return "OUT OF ORDER"
    if min(w, m, s) >= 0.99:
        return "all saturated"
    if max(w, m, s) < 0.03:
        return "all ~0"
    return "ok"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--csv", help="also write a CSV to this path")
    a = ap.parse_args(argv)

    rows = collect(a.root)
    if not rows:
        print("no invalid samples under %s" % a.root, file=sys.stderr)
        return 2

    ordered = sorted(rows.items())
    w_scn = max(len(s) for s, _ in rows)
    w_fam = max(len(f) for _, f in rows)
    print("%-3s %-*s %-*s %6s %6s %6s  %s"
          % ("", w_scn, "scenario", w_fam, "family", "weak", "medium",
             "strong", "verdict"))
    bad = 0
    for (scenario, family), vals in ordered:
        v = verdict(vals)
        if v not in ("ok",):
            bad += 1
        cells = ["%.2f" % vals[b] if b in vals else "  -" for b in BINS]
        flag = "!!" if v not in ("ok",) else "  "
        print("%-3s %-*s %-*s %6s %6s %6s  %s"
              % (flag, w_scn, scenario, w_fam, family, cells[0], cells[1],
                 cells[2], v))

    print("\n%d cells, %d flagged (not cleanly ordered / saturated / all-zero)"
          % (len(ordered), bad))

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["scenario", "family", "weak", "medium", "strong",
                             "verdict"])
            for (scenario, family), vals in ordered:
                writer.writerow([scenario, family] +
                                [vals.get(b, "") for b in BINS] +
                                [verdict(vals)])
        print("wrote %s" % a.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

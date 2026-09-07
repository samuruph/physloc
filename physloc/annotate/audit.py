"""Which cells produce a violation you can actually see.

A `(scenario, family)` pair can be perfectly well-formed and still not be worth
generating. Friction applied to a body that has already come to rest changes
nothing; a shove delivered while the actor is off screen changes nothing you can
look at. Those clips carry a full set of labels describing a violation the video
does not contain, which is worse than not having the cell at all.

This module measures that, per cell, from a generated release -- so the decision
to drop a cell is a number rather than an opinion. Three independent signals,
because each of them misses cases the others catch:

  severity    the annotation's own claim about how badly wrong it is
  observable  frames on which the twins differ AT ALL
  evidence    peak pixel divergence inside the violation mask, which is the
              only one that asks whether a viewer could see it
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np

#: A cell is flagged when it fails ALL of these. Any one of them passing means
#: there is something there, and the cell stays.
MIN_SEVERITY = 0.02
MIN_OBSERVABLE_FRAMES = 1
MIN_EVIDENCE = 0.02          # peak |valid - invalid| inside the mask, 0..1


def measure_clip(cdir: str) -> Dict[str, object]:
    """Severity, observability and pixel evidence for one invalid clip."""
    meta = json.load(open(os.path.join(cdir, "meta.json")))
    v = meta.get("violation") or {}
    out = {
        "scenario": meta.get("scenario"), "family": meta.get("family"),
        "severity_bin": (v.get("intervention") or {}).get("severity_bin"),
        "peak_severity": 0.0, "observable_frames": 0, "evidence": 0.0,
    }
    tl_path = os.path.join(cdir, "timelines.npz")
    if os.path.exists(tl_path):
        tl = np.load(tl_path)
        out["peak_severity"] = float(np.asarray(tl["severity_t"]).max())
        out["observable_frames"] = int(np.asarray(tl["observable"]).sum())

    mp = os.path.join(cdir, "violation_mask.npz")
    dp = os.path.join(cdir, "divergence_map.npz")
    if os.path.exists(mp) and os.path.exists(dp):
        mask = np.load(mp)["mask"].astype(bool)
        div = np.asarray(np.load(dp)["divergence"], np.float32)
        if mask.any():
            out["evidence"] = float(div[mask].max())
    return out


def is_invisible(row: Dict[str, object]) -> bool:
    """Nothing happened that a viewer could see."""
    return (float(row["peak_severity"]) < MIN_SEVERITY
            and int(row["observable_frames"]) < MIN_OBSERVABLE_FRAMES
            and float(row["evidence"]) < MIN_EVIDENCE)


def is_unscored(row: Dict[str, object]) -> bool:
    """Something visibly happened and the annotation reports nothing.

    The more dangerous of the two, and the one the first audit actually found.
    An invisible cell ships a label with no picture; an unscored one ships a
    picture with a severity of zero, which trains a model that a clearly wrong
    clip is fine. Both belong in NOT_MEANINGFUL unless the residual can be made
    to see the violation.
    """
    return (float(row["evidence"]) >= MIN_EVIDENCE
            and float(row["peak_severity"]) < MIN_SEVERITY)


def audit(release_root: str) -> List[Dict[str, object]]:
    rows = []
    for mp in sorted(glob.glob(os.path.join(release_root, "clips", "**",
                                            "meta.json"), recursive=True)):
        meta = json.load(open(mp))
        if meta.get("label") != "invalid":
            continue
        rows.append(measure_clip(os.path.dirname(mp)))
    return rows

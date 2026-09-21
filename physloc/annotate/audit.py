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

from typing import Dict, List

import numpy as np

from .. import loader

#: A cell is flagged when it fails ALL of these. Any one of them passing means
#: there is something there, and the cell stays.
MIN_SEVERITY = 0.02
MIN_OBSERVABLE_FRAMES = 1
MIN_EVIDENCE = 0.02          # peak |valid - invalid| inside the mask, 0..1

#: WHAT A VIEWER CAN SEE, which `evidence` does not measure. It is the PEAK
#: difference of a single pixel, so one anti-aliased edge flickering reads as
#: much evidence as a body jumping across the frame: `global_gravity` weak on
#: `resting_table` scored 0.43 with nothing visibly moving. So also count the
#: pixels that change by at least `VISIBLE_STEP` of full intensity, and ask
#: for a patch of them -- `MIN_VISIBLE_SHARE` of the frame -- on at least
#: `MIN_VISIBLE_FRAMES` frames. 0.1% is a 4x4 patch at the debug tier's 128^2
#: and 16x16 at 512^2.
VISIBLE_STEP = 25            # of 255
MIN_VISIBLE_SHARE = 0.001
MIN_VISIBLE_FRAMES = 2
#: The weakest bin must still be a violation: small, but seen and scored.
MIN_WEAK_SEVERITY = 0.03


def measure_sample(sample_dir: str, twin=None) -> Dict[str, object]:
    """Severity, observability and pixel evidence for one invalid sample."""
    sample = loader.Sample(sample_dir, twin=twin)
    out = {
        "scenario": sample.scene.scenario, "family": sample.scene.family,
        "severity_bin": sample.violation.severity_bin,
        "pair_uid": sample.info.pair_uid,
        "peak_severity": 0.0, "observable_frames": 0, "evidence": 0.0,
    }
    tl = sample.violation.timeline
    out["peak_severity"] = float(np.asarray(tl["severity"]).max())
    out["observable_frames"] = int(np.asarray(tl["observable"]).sum())

    # Divergence is no longer shipped; it is computed from the two videos, which
    # is what it always was -- so this needs the valid twin beside the clip.
    mask = sample.violation.mask
    if mask.any() and sample.twin is not None:
        out["evidence"] = float(sample.divergence[mask].max())
    out["visible_share"], out["visible_frames"] = 0.0, 0
    if sample.twin is not None:
        a = sample.video.rgb.astype(np.int16)
        b = sample.twin.video.rgb.astype(np.int16)
        changed = (np.abs(a - b).max(axis=-1) >= VISIBLE_STEP)
        share = changed.reshape(changed.shape[0], -1).mean(axis=1)
        out["visible_share"] = float(share.max())
        out["visible_frames"] = int((share >= MIN_VISIBLE_SHARE).sum())
    sample.release()
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


def is_too_faint(row: Dict[str, object]) -> bool:
    """A viewer could not see it: too few pixels change, on too few frames."""
    return (float(row.get("visible_share", 0.0)) < MIN_VISIBLE_SHARE
            or int(row.get("visible_frames", 0)) < MIN_VISIBLE_FRAMES)


def weak_failure(row: Dict[str, object]) -> str:
    """Why a clip fails the weakest-bin rule, or '' if it passes.

    Every bin is a violation, including the weakest: it may be small, never
    absent. Two ways to fail, and they call for different fixes -- a clip too
    faint to see needs a stronger ladder; one that is seen but scores zero
    needs its residual law to see what the picture shows.
    """
    if is_too_faint(row):
        return "too faint"
    if float(row["peak_severity"]) < MIN_WEAK_SEVERITY:
        return "unscored"
    return ""


#: How far a stronger bin may fall below a weaker one, in severity, before the
#: ladder counts as out of order -- measurement noise, not a real reversal.
LADDER_SLACK = 0.02
BINS = ("weak", "medium", "strong")


def ladder_failures(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Scenes whose three bins do not climb.

    One scene -- one `pair_uid`, one family -- rendered at every bin must come
    out weak <= medium <= strong in measured severity, and not all three
    pinned at the ceiling, which is one bin wearing three labels. Scenes that
    lack a bin are skipped: there is nothing to order.
    """
    by = {}
    for r in rows:
        by.setdefault((r.get("pair_uid"), r["family"]), {})[r["severity_bin"]] = r
    out = []
    for (pair, family), bins in sorted(by.items(), key=lambda kv: str(kv[0])):
        if not all(b in bins for b in BINS):
            continue
        s = [float(bins[b]["peak_severity"]) for b in BINS]
        why = ""
        if s[0] > s[1] + LADDER_SLACK or s[1] > s[2] + LADDER_SLACK:
            why = "out of order"
        elif min(s) >= 0.99:
            why = "all saturated"
        if why:
            out.append({"pair_uid": pair, "family": family,
                        "scenario": bins["weak"]["scenario"],
                        "severities": s, "why": why})
    return out


def audit(release_root: str) -> List[Dict[str, object]]:
    dataset = loader.PhysLocDataset(release_root)
    rows = []
    for sample in dataset.samples:
        if sample.info.is_valid:
            continue
        rows.append(measure_sample(sample.path, sample.twin))
    return rows

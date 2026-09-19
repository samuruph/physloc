"""Dataset-structure videos: one scenario x family, side by side along one axis.

A release has three axes a viewer should be able to SEE, and each has a video:

* `levels`     -- L0, L1, L2 and L3 of one scenario x family: the complexity
  ladder, which is realism and nothing else. **Not one scene redressed**:
  every level draws its own scenes from its own seed block, so the tiles are
  four different scenes of the same cell. The video says so in its header,
  because "twins" is the obvious misreading.
* `variants`   -- several variants of one cell at one level: what the sampler
  varies from seed to seed.
* `conditions` -- standard / camera / distractors / multi / camera+multi of
  one cell: the difficulty axis, one named change per clip.

Every tile is the INVALID clip with its violation mask in red and the
reference outline in green -- the `mask` panel of `overlay.mp4` -- labelled,
with a timeline under it: violation windows red, observable windows amber, the
event a white tick, the playhead below, and a red dot while the violation is
active. Clips of different lengths play frame by frame; a shorter one holds its
last frame. A tile with nothing generated is a dark square that says so.

The clips can come from several release roots at once (`review_L0` and
`review_L3`, say), since a review run usually holds one level.

mp4 only -- no image files are written.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import loader
from ..annotate import layout
from . import overlay as ov
from . import video as vid

LEVELS = ("L0", "L1", "L2", "L3")
CONDITIONS = ("standard", "camera", "distractors", "multi", "camera+multi")
KINDS = ("levels", "variants", "conditions")

CELL = 192
PAD = 8
HEADER = 48
LABEL = 34
BAR = 18

AXIS_NOTE = {
    "levels": "complexity ladder = realism only. Each level draws its OWN "
              "scenes: four scenes of one cell, not twins",
    "variants": "independent variants of one cell at %s: what the sampler "
                "varies between seeds",
    "conditions": "difficulty conditions of one cell at %s: one named change "
                  "per clip against standard",
}

Tile = Tuple[str, str, Optional[Dict[str, object]]]


# ------------------------------------------------------------------ index
def index(roots: Sequence[str]) -> List[Dict[str, object]]:
    """Every invalid sample under `roots`, with the fields selection needs.

    Reads `sample.json` and nothing else; arrays are loaded per tile only for
    the samples a video actually draws.
    """
    out = []
    for root in roots:
        for mp in layout.find(root):
            try:
                sample = loader.Sample(os.path.dirname(mp))
            except (OSError, ValueError):
                continue
            info, scene, v = sample.info, sample.scene, sample.violation
            if info.is_valid:
                continue
            windows = v.windows
            out.append({
                "dir": sample.path, "root": root,
                "uid": info.uid, "valid_uid": info.valid_uid,
                "release": info.release,
                "scenario": str(scene.scenario), "family": str(scene.family),
                "level": str(scene.level or "?"),
                "condition": str(scene.condition or "standard"),
                "seed": int(info.seed or 0), "variant": info.variant,
                "severity": str(v.severity_bin or "strong"),
                "num_frames": sample.video.num_frames,
                "fps": int(sample.video.fps or 12),
                "t_event": v.t_event,
                "vwin": [tuple(w) for w in windows["active"]],
                "owin": [tuple(w) for w in windows["observable"]],
            })
    return out


def _order(r) -> tuple:
    cond = r["condition"]
    return (CONDITIONS.index(cond) if cond in CONDITIONS else len(CONDITIONS),
            r["variant"], r["seed"], r["root"], r["dir"])


def _cell(recs, scenario: str, family: str, severity: str):
    return [r for r in recs if r["scenario"] == scenario
            and r["family"] == family and r["severity"] == severity]


def _first(rows):
    rows = sorted(rows, key=_order)
    return rows[0] if rows else None


def pick_levels(recs, scenario, family, severity="strong") -> List[Tile]:
    """One clip per level, preferring the standard condition."""
    rows = _cell(recs, scenario, family, severity)
    out = []
    for lv in LEVELS:
        r = _first([x for x in rows if x["level"] == lv])
        sub = ("" if r is None else "seed %d  %s  event f%s"
               % (r["seed"], r["condition"], r["t_event"]))
        out.append((lv, sub, r))
    return out


def pick_variants(recs, scenario, family, level="L0", n=5,
                  severity="strong") -> List[Tile]:
    """Up to `n` distinct variants of the cell at `level`, in variant order."""
    rows = sorted((r for r in _cell(recs, scenario, family, severity)
                   if r["level"] == level),
                  key=lambda r: (r["variant"], r["seed"], r["root"], r["dir"]))
    seen, out = set(), []
    for r in rows:
        # The SCENE, not the directory: two review roots often hold the same
        # L0 scene (`review_L0` and `review_conditions` both draw seed 777,
        # variant 0), and keying on the root drew it twice.
        key = (r["seed"], r["variant"], r["condition"])
        if key in seen:
            continue
        seen.add(key)
        out.append(("variant %d" % r["variant"],
                    "seed %d  %s  event f%s" % (r["seed"], r["condition"],
                                                r["t_event"]), r))
        if len(out) >= int(n):
            break
    return out


def pick_conditions(recs, scenario, family, level="L0",
                    severity="strong") -> List[Tile]:
    """One clip per difficulty condition at `level`, in the declared order."""
    rows = [r for r in _cell(recs, scenario, family, severity)
            if r["level"] == level]
    out = []
    for cond in CONDITIONS:
        r = _first([x for x in rows if x["condition"] == cond])
        sub = ("" if r is None else "variant %d  seed %d  event f%s"
               % (r["variant"], r["seed"], r["t_event"]))
        out.append((cond, sub, r))
    return out


# ------------------------------------------------------------------ drawing
def _fit(s: str, width: int, scale: float) -> str:
    """`s`, cut with '..' until it fits `width` pixels at `scale`."""
    if ov._w(s, scale) <= width:
        return s
    while len(s) > 1 and ov._w(s + "..", scale) > width:
        s = s[:-1]
    return s + ".."


def _load(rec) -> Dict[str, object]:
    """A tile's arrays, through the loader: the reference outline needs the
    clip's valid twin, and is left out when that twin is not on disk."""
    sample = loader.Sample.from_dir(rec["dir"])
    valid_uid = rec.get("valid_uid")
    if valid_uid and valid_uid != sample.uid:
        valid_dir = os.path.join(rec["root"], "samples", *str(valid_uid).split("/"))
        if os.path.exists(os.path.join(valid_dir, loader.SAMPLE_METADATA)):
            sample._twin = loader.Sample.from_dir(valid_dir)
    T = int(rec["num_frames"])
    v = sample.violation
    return {"rgb": sample.video.rgb[:T],
            "mask": v.mask,
            "ref": v.reference_mask if sample.twin is not None else None,
            "active": v.timeline["active"]}


def _bar(f, rec, t: int, T: int, x: int, y: int, width: int) -> None:
    """Violation windows (red) over observable ones (amber), event tick, playhead."""
    import cv2

    for row, (wins, colour) in enumerate(((rec["vwin"], ov.C_MASK),
                                          (rec["owin"], ov.C_OBS))):
        ry = y + row * 5
        cv2.rectangle(f, (x, ry), (x + width - 1, ry + 3), (40, 40, 48), -1)
        for s, e in wins:
            x0 = x + int(s / T * width)
            x1 = min(x + width - 1, max(x + int((e + 1) / T * width) - 1, x0 + 1))
            cv2.rectangle(f, (x0, ry), (x1, ry + 3), colour, -1)
    if rec.get("t_event") is not None:
        ex = x + min(width - 1, int((int(rec["t_event"]) + 0.5) / T * width))
        cv2.line(f, (ex, y - 2), (ex, y + 9), (255, 255, 255), 1)
    px = x + min(width - 1, int((t + 0.5) / T * width))
    cv2.rectangle(f, (px - 1, y + 11), (px + 1, y + 14), ov.C_TEXT, -1)


def render(title: str, note: str, tiles: Sequence[Tile], out_path: str,
           cell: int = CELL) -> str:
    """Draw `tiles` side by side into `out_path` and return it."""
    import cv2

    loaded = [(label, sub, rec, None if rec is None else _load(rec))
              for label, sub, rec in tiles]
    lengths = [d["rgb"].shape[0] for *_, d in loaded if d is not None]
    if not lengths:
        raise ValueError("nothing generated to draw for: %s" % title)
    T = max(lengths)
    fps = next(int(rec["fps"]) for _, _, rec, d in loaded if d is not None)
    n = len(loaded)
    W = PAD + n * (cell + PAD)
    H = HEADER + LABEL + cell + BAR + PAD
    frames = np.zeros((T, H, W, 3), np.uint8)

    for t in range(T):
        f = np.full((H, W, 3), ov.C_BG, np.uint8)
        counter = "f %d/%d" % (t, T - 1)
        ov._text(f, _fit(title, W - 3 * PAD - ov._w(counter, 0.42), 0.52),
                 (PAD, 20), ov.C_TEXT, 0.52, 1)
        ov._text(f, counter, (W - PAD - ov._w(counter, 0.42), 20),
                 ov.C_TEXT, 0.42, 1)
        ov._text(f, _fit(note, W - 2 * PAD, 0.36), (PAD, 38), ov.C_DIM, 0.36, 1)
        for i, (label, sub, rec, d) in enumerate(loaded):
            x = PAD + i * (cell + PAD)
            y = HEADER + LABEL
            if label:
                ov._text(f, _fit(label, cell, 0.44), (x, HEADER + 12),
                         (255, 255, 255), 0.44, 1)
            if sub:       # an empty string still draws its backing box
                ov._text(f, _fit(sub, cell, 0.33), (x, HEADER + 28),
                         ov.C_DIM, 0.33, 1)
            if d is None:
                cv2.rectangle(f, (x, y), (x + cell - 1, y + cell - 1),
                              (10, 10, 13), -1)
                cv2.rectangle(f, (x, y), (x + cell - 1, y + cell - 1),
                              (38, 38, 46), 1)
                msg = "not generated"
                ov._text(f, msg, (x + (cell - ov._w(msg, 0.4)) // 2,
                                  y + cell // 2), ov.C_DIM, 0.4, 1,
                         backing=False)
                continue
            Ti = int(d["rgb"].shape[0])
            ti = min(t, Ti - 1)
            f[y:y + cell, x:x + cell] = ov._panel(
                "mask", ti, d["rgb"], d["mask"], None, None, None, cell, d["ref"])
            cv2.rectangle(f, (x, y), (x + cell - 1, y + cell - 1), (60, 60, 70), 1)
            act = d["active"]
            if act is not None and ti < len(act) and bool(act[ti]):
                cv2.circle(f, (x + cell - 10, y + 10), 5, ov.C_ACTIVE, -1)
                cv2.circle(f, (x + cell - 10, y + 10), 5, (255, 255, 255), 1)
            _bar(f, rec, ti, Ti, x, y + cell + 5, cell)
        frames[t] = f

    vid.write(frames, out_path, fps=fps)
    return out_path


# ------------------------------------------------------------------ videos
def one(kind: str, recs, scenario: str, family: str, out_path: str,
        severity: str = "strong", level: str = "L0", n: int = 5) -> str:
    """One structure video of `kind` for one scenario x family."""
    if kind == "levels":
        tiles = pick_levels(recs, scenario, family, severity)
        title = "%s x %s  --  complexity levels  (%s)" % (scenario, family, severity)
        note = AXIS_NOTE["levels"]
    elif kind == "variants":
        tiles = pick_variants(recs, scenario, family, level, n, severity)
        title = "%s x %s  --  %d variants at %s  (%s)" % (
            scenario, family, len(tiles), level, severity)
        note = AXIS_NOTE["variants"] % level
    elif kind == "conditions":
        tiles = pick_conditions(recs, scenario, family, level, severity)
        title = "%s x %s  --  difficulty conditions at %s  (%s)" % (
            scenario, family, level, severity)
        note = AXIS_NOTE["conditions"] % level
    else:
        raise ValueError("unknown kind %r; expected one of %s" % (kind, KINDS))
    return render(title, note, tiles, out_path)


def candidates(recs, kind: str, severity: str = "strong",
               level: str = "L0") -> List[Tuple[str, str]]:
    """Scenario x family pairs with at least two tiles of `kind` to compare."""
    keys = defaultdict(set)
    for r in recs:
        if r["severity"] != severity:
            continue
        if kind == "levels":
            key = r["level"]
        elif r["level"] != level:
            continue
        elif kind == "variants":
            key = (r["root"], r["seed"], r["variant"])
        else:
            key = r["condition"]
        keys[(r["scenario"], r["family"])].add(key)
    return sorted(pair for pair, got in keys.items() if len(got) >= 2)


def spread(pairs: Sequence[Tuple[str, str]], limit: Optional[int],
           seed: int = 0) -> List[Tuple[str, str]]:
    """At most `limit` pairs, taken a scenario at a time.

    Round-robin over scenarios, each scenario's families in a fixed shuffled
    order, so a subset still covers every scenario and is not just the
    alphabetically first family of each.
    """
    pairs = list(pairs)
    if not limit or limit >= len(pairs):
        return pairs
    per: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for s, fam in pairs:
        per[s].append((s, fam))
    rng = np.random.RandomState(seed)
    for s in per:
        per[s] = [per[s][i] for i in rng.permutation(len(per[s]))]
    out: List[Tuple[str, str]] = []
    while len(out) < limit and any(per.values()):
        for s in sorted(per):
            if per[s] and len(out) < limit:
                out.append(per[s].pop(0))
    return sorted(out)


def batch(roots: Sequence[str], outdir: str, kinds: Sequence[str] = KINDS,
          limit: Optional[int] = None, severity: str = "strong",
          level: str = "L0", n: int = 5) -> Dict[str, object]:
    """Every (or `limit` per kind) structure video the roots can support.

    Written to `<outdir>/<kind>/<scenario>__<family>.mp4`. Returns what was
    made and how long each took, so a caller can see whether the full set is
    affordable before asking for it.
    """
    recs = index(roots)
    made: Dict[str, List[str]] = {}
    seconds: List[float] = []
    for kind in kinds:
        made[kind] = []
        for scenario, family in spread(candidates(recs, kind, severity, level),
                                       limit):
            path = os.path.join(outdir, kind, "%s__%s.mp4" % (scenario, family))
            t0 = time.perf_counter()
            one(kind, recs, scenario, family, path, severity, level, n)
            seconds.append(time.perf_counter() - t0)
            made[kind].append(path)
    return {"outdir": outdir, "clips_indexed": len(recs),
            "made": {k: len(v) for k, v in made.items()}, "files": made,
            "seconds_per_video": (round(float(np.mean(seconds)), 2)
                                  if seconds else None),
            "seconds_total": round(float(np.sum(seconds)), 1)}

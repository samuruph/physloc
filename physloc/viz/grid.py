"""Two comparison videos over a generated pair directory.

`build` -- **one family, every severity.** Columns are variants (valid, then
each bin in order); rows are the same annotation views the single-clip overlay
shows, so a grid cell and an overlay panel are directly comparable. Answers "how
do weak, medium and strong differ, and is each one annotated correctly".

    +--------------------------------------------------------------------+
    | drop / solidity                    seed 777  tier debug     f 7/24  |
    +---------+----------------+----------------+----------------+-------+
    |         | VALID          | weak     [o]   | medium   [o]   | strong|
    | RGB     |                |                |                |       |
    +---------+----------------+----------------+----------------+-------+
    | MASK    | (should-be)    | + MASK         | + MASK         |       |
    +---------+----------------+----------------+----------------+-------+
    | SEVERITY| n/a            | heat + scale   | heat + scale   |       |
    +---------+----------------+----------------+----------------+-------+
    | CAUSAL  | ...            |                |                |       |
    +---------+----------------+----------------+----------------+-------+
    | DIVERG. | ...            |                |                |       |
    +---------+----------------+----------------+----------------+-------+

`sheet` -- **every family at once**, for one scenario and seed. Rows are
families, columns are valid plus each severity bin, one annotation view
throughout (`view=`). Answers "do all of these violations look like what they
claim, side by side" -- which is the question a per-family video cannot answer
because you have to remember the last one.

mp4 only -- no image files are written.
"""
from __future__ import annotations

import glob
import math
import json
import os
from typing import Dict, List, Optional

import numpy as np

from . import overlay as ov
from . import video as vid

CELL = 224
PAD = 4
HEADER = 30
COLHEAD = 20
BAR = 40
#: Left-hand gutter for row labels. Labelling rows in the gutter rather than
#: inside the panels keeps the imagery unobstructed -- the panels are the thing
#: being judged, and text drawn over a mask is text drawn over the evidence.
GUTTER = 104

ORDER = {"valid": 0, "weak": 1, "medium": 2, "strong": 3}

#: (row label, `overlay._panel` kind, which array must exist). The order and the
#: labels match the single-clip overlay's panels exactly, so a grid cell, a
#: sheet cell and an overlay panel can be read against each other without
#: translating. It runs evidence first -- what the renderer saw -- then the
#: annotation derived from it.
VIEWS = [
    ("RGB", "rgb", None),
    ("ENERGY", "energy", "energy"),
    ("SEGMENT", "seg", "seg"),
    ("DEPTH", "depth", "depth"),
    ("OPTICAL FLOW", "flow", "flow"),
    ("MASK", "mask", "mask"),
    ("SEVERITY", "sev", "sev"),
    ("CAUSAL", "causal", "causal"),
    ("DIVERGENCE", "div", "div"),
]
VIEW_NOTE = {
    "mask": "red = violation   green = should-be",
    "sev": "0..1, inferno, gamma for display",
    "causal": "1 = culprit   2 = affected",
    "div": "|valid - invalid|  NOT ground truth",
    "energy": "per-body E, fraction of E0",
    "seg": "instance ids = tracks",
    "depth": "metres, near -> far",
    "flow": "hue=direction val=speed",
}


def build(pair_dir: str, family: Optional[str] = None,
          out_path: Optional[str] = None, cell: int = CELL,
          views: Optional[List[str]] = None) -> Dict[str, object]:
    """One family: a row per severity, a column per annotation view.

    **Rows are severities, columns are views** -- the transpose of the first
    version, which stacked five views vertically and produced a 1232x576 tower
    that does not fit on a screen. A row now reads like the single-clip overlay,
    left to right, and the severities line up under each other so the ladder is
    a vertical scan.
    """
    import cv2

    cols = _collect(pair_dir, family)
    if len(cols) < 2:
        raise ValueError("need a valid clip and at least one variant in %s" % pair_dir)

    views_avail = _rows_for(cols, views)
    T = int(cols[0]["meta"]["num_frames"])
    ncol, nrow = len(views_avail), len(cols)
    W = GUTTER + ncol * cell + (ncol + 1) * PAD
    H = HEADER + COLHEAD + nrow * (cell + PAD) + PAD + BAR

    meta0 = cols[0]["meta"]
    fam = family or next((c["meta"].get("family") for c in cols[1:]), "-")
    out = np.zeros((T, H, W, 3), np.uint8)

    for t in range(T):
        f = np.full((H, W, 3), ov.C_BG, np.uint8)
        title = "%s / %s   |   seed %s  %s  %s" % (
            meta0.get("scenario"), fam, meta0.get("seed"), meta0.get("tier"),
            (meta0.get("complexity") or {}).get("name", ""))
        ov._text(f, title, (PAD + 2, 20), ov.C_TEXT, 0.50, 1)
        fs = "f %d/%d" % (t, T - 1)
        ov._text(f, fs, (W - PAD - ov._w(fs, 0.46), 20), ov.C_TEXT, 0.46, 1)

        for j, (label, kind, _need) in enumerate(views_avail):
            x = GUTTER + PAD + j * (cell + PAD)
            ov._text(f, label, (x + 4, HEADER + 14), ov.C_TEXT, 0.44, 1)
            note = VIEW_NOTE.get(kind)
            if note:
                ov._text(f, note, (x + 6 + ov._w(label, 0.44), HEADER + 14),
                         ov.C_DIM, 0.32, 1)

        for i, c in enumerate(cols):
            y = HEADER + COLHEAD + i * (cell + PAD)
            active = bool(c["tl"]["active"][t]) if c["tl"] is not None else False
            ov._text(f, c["label"], (PAD + 2, y + 15), (255, 255, 255), 0.44, 1)
            if c["mag"] is not None:
                ov._text(f, "%.3g" % c["mag"], (PAD + 2, y + 30), ov.C_DIM, 0.36, 1)
                ov._text(f, c["mag_unit"].replace("m_", "")[:13],
                         (PAD + 2, y + 43), ov.C_DIM, 0.30, 1)
            if c["lag"] is not None:
                ov._text(f, "lag %d" % c["lag"], (PAD + 2, y + 58), ov.C_DIM, 0.32, 1)
            if active:
                cv2.circle(f, (PAD + 12, y + cell - 12), 6, ov.C_ACTIVE, -1)
                cv2.circle(f, (PAD + 12, y + cell - 12), 6, (255, 255, 255), 1)

            for j, (_label, kind, need) in enumerate(views_avail):
                x = GUTTER + PAD + j * (cell + PAD)
                _draw_cell(f, c, kind, need, t, x, y, cell)

            by = H - BAR + 4
            _window_bar(f, c, t, T, GUTTER + PAD, by, W - GUTTER - 2 * PAD)

        # The same legend the single-clip overlay carries, and for the same
        # reason: three coloured bars with no key is a puzzle, and none of the
        # three colours means anything by convention.
        _window_legend(f, PAD + 2, H - 5)
        out[t] = f

    out_path = out_path or os.path.join(pair_dir, "grid_%s.mp4" % (fam or "all"))
    vid.write(out, out_path, fps=int(meta0.get("fps", 12)))
    return {"path": out_path, "rows": [c["label"] for c in cols],
            "columns": [v[0] for v in views_avail], "frames": T}


def sheet(pair_dir: str, out_path: Optional[str] = None,
          view: str = "mask", cell: Optional[int] = None,
          severity: str = "strong") -> Dict[str, object]:
    """Every family of one scenario, with every annotation, in one frame.

    **Columns are clips, rows are annotation views** -- so each column is the
    single-clip overlay turned on its side, and reading across a row compares
    the same annotation over every violation at the same instant. The leftmost
    column is the valid twin, which has an RGB frame and nothing else to show:
    a valid clip has no mask, no severity and no causal map, so those cells are
    blank and say so rather than being left ambiguous.

    `view` is kept for callers that want a single-view sheet; the default now
    shows every view the clips carry.
    """
    import cv2

    cols = _collect(pair_dir, None)
    valid = next((c for c in cols if c["is_valid"]), None)
    if valid is None:
        raise ValueError("no valid clip in %s" % pair_dir)
    picked = [c for c in cols
              if not c["is_valid"] and c["label"] == severity]
    if not picked:
        picked = [c for c in cols if not c["is_valid"]]
    picked.sort(key=lambda c: c["meta"].get("family", ""))
    if not picked:
        raise ValueError("no invalid clips in %s" % pair_dir)

    rows = _rows_for([valid] + picked, None)
    entries = [valid] + picked
    ncol, nrow = len(entries), len(rows)
    if cell is None:
        cell = 168 if ncol <= 10 else (140 if ncol <= 16 else 116)

    T = int(valid["meta"]["num_frames"])
    W = GUTTER + ncol * cell + (ncol + 1) * PAD
    H = HEADER + COLHEAD + nrow * (cell + PAD) + PAD
    out = np.zeros((T, H, W, 3), np.uint8)
    meta0 = valid["meta"]

    for t in range(T):
        f = np.full((H, W, 3), ov.C_BG, np.uint8)
        title = "%s  seed %s  %s  %s  |  severity %s" % (
            meta0.get("scenario"), meta0.get("seed"), meta0.get("tier"),
            (meta0.get("complexity") or {}).get("name", ""), severity)
        ov._text(f, title, (PAD + 2, 20), ov.C_TEXT, 0.50, 1)
        fs = "f %d/%d" % (t, T - 1)
        ov._text(f, fs, (W - PAD - ov._w(fs, 0.46), 20), ov.C_TEXT, 0.46, 1)

        for i, (label, kind, _need) in enumerate(rows):
            y = HEADER + COLHEAD + i * (cell + PAD)
            ov._text(f, label, (PAD + 2, y + 15), (255, 255, 255), 0.40, 1)
            note = VIEW_NOTE.get(kind)
            if note:
                for k, part in enumerate(_wrap(note, 14)):
                    ov._text(f, part, (PAD + 2, y + 29 + 11 * k), ov.C_DIM, 0.29, 1)

        for j, c in enumerate(entries):
            x = GUTTER + PAD + j * (cell + PAD)
            lab = "VALID" if c["is_valid"] else c["meta"].get("family", "?")
            ov._text(f, lab[:18], (x + 3, HEADER + 14),
                     ov.C_TEXT if c["is_valid"] else (210, 210, 235), 0.40, 1)
            for i, (_label, kind, need) in enumerate(rows):
                y = HEADER + COLHEAD + i * (cell + PAD)
                _draw_cell(f, c, kind, need, t, x, y, cell, compact=True)
            if not c["is_valid"] and c["tl"] is not None \
                    and bool(c["tl"]["active"][t]):
                cv2.circle(f, (x + cell - 9, HEADER + COLHEAD + 9), 5,
                           ov.C_ACTIVE, -1)
                cv2.circle(f, (x + cell - 9, HEADER + COLHEAD + 9), 5,
                           (255, 255, 255), 1)
        out[t] = f

    out_path = out_path or os.path.join(pair_dir, "sheet_%s.mp4" % severity)
    vid.write(out, out_path, fps=int(meta0.get("fps", 12)))
    return {"path": out_path, "columns": [
        "valid" if c["is_valid"] else c["meta"].get("family") for c in entries],
        "rows": [r[0] for r in rows], "severity": severity, "frames": T}


# ------------------------------------------------------------------ shared
def _rows_for(cols, views: Optional[List[str]]):
    """Which annotation views this pair can actually fill.

    Driven by what is on disk, not by a fixed list: a release generated without
    divergence maps should lose the view rather than draw an empty box captioned
    with a promise the files do not keep.
    """
    have = set()
    for c in cols:
        for key in ("mask", "sev", "causal", "div", "energy", "seg",
                    "depth", "flow"):
            if c.get(key) is not None:
                have.add(key)
    rows = [v for v in VIEWS if v[2] is None or v[2] in have]
    if views:
        rows = [v for v in rows if v[1] in views]
    return rows


def _draw_cell(f, c, kind, need, t, x, y, cell, compact: bool = False) -> None:
    import cv2
    if need is not None and c.get(need) is None and not c["is_valid"]:
        cv2.rectangle(f, (x, y), (x + cell - 1, y + cell - 1), (44, 44, 52), 1)
        ov._text(f, "no %s" % need, (x + 6, y + cell // 2), ov.C_DIM, 0.34, 1)
        return
    # A valid clip has no violation to shade, so severity, causal and
    # divergence are blank by definition rather than missing. Saying which keeps
    # the column honest -- an empty panel with no caption reads as a failed
    # render. Energy is the exception: the valid twin's trace is the baseline
    # every anomaly is measured against, so it belongs there.
    # seg, depth and flow are properties of the render rather than of the
    # violation, so the valid twin carries them just as the invalid one does.
    if c["is_valid"] and kind in ("sev", "causal", "div"):
        cv2.rectangle(f, (x, y), (x + cell - 1, y + cell - 1), (44, 44, 52), 1)
        ov._text(f, "n/a", (x + 6, y + cell // 2), ov.C_DIM, 0.36, 1)
        return

    img = ov._panel(kind, t, c["rgb"],
                    None if c["is_valid"] else c["mask"],
                    c["sev"], c["causal"], c["div"], cell, c["ref"],
                    c.get("energy"), c.get("seg"), c.get("depth"),
                    c.get("flow"), c.get("normals"))
    f[y:y + cell, x:x + cell] = img
    cv2.rectangle(f, (x, y), (x + cell - 1, y + cell - 1), (60, 60, 70), 1)

    if kind == "mask":
        src = c["ref"] if c["is_valid"] else c["mask"]
        if src is not None:
            ov._text(f, "%s %d px" % ("should-be" if c["is_valid"] else "viol",
                                      int(src[t].sum())),
                     (x + 4, y + cell - 6), ov.C_DIM, 0.33, 1)
    elif kind == "energy" and c.get("etrace") is not None:
        # The full curve, both lines and the colour key -- the same panel the
        # single-clip overlay draws. The compact one-line form was dropping the
        # comparison against the valid twin, which is the whole point: a number
        # says the clip holds 77 J, the curve says it climbed there from 26.
        tr = c["etrace"]
        if c["is_valid"]:
            ov._text(f, "E %.1fJ  (baseline)" % float(tr["total"][t]),
                     (x + 4, y + cell - 6), ov.C_REF, 0.35, 1)
        else:
            ov._energy_scale(f, x, y, cell,
                             float(np.abs(c["energy"][t]).max())
                             if c.get("energy") is not None else None)
            ov._energy_curve(f, x, y, cell, t, tr, c.get("twin_etrace"))
    elif kind == "sev" and c["tl"] is not None:
        sv = float(c["tl"]["severity_t"][t])
        if compact:
            ov._text(f, "s=%.3f" % sv, (x + 4, y + cell - 6), (255, 255, 255),
                     0.35, 1)
        else:
            ov._sev_scale(f, x, y, cell, sv)


def _window_legend(f, x: int, y: int) -> None:
    """Swatch + name + what the window actually means."""
    import cv2

    for label, gloss, colour in (
            ("intervening", "being changed", ov.C_INTERVENE),
            ("consequence", "still wrong", ov.C_MASK),
            ("observable", "a viewer could tell", ov.C_OBS)):
        cv2.rectangle(f, (x, y - 8), (x + 9, y), colour, -1)
        cv2.rectangle(f, (x, y - 8), (x + 9, y), (20, 20, 26), 1)
        x += 13
        ov._text(f, label, (x, y), ov.C_TEXT, 0.36, 1)
        x += ov._w(label, 0.36) + 4
        gl = "(%s)" % gloss
        ov._text(f, gl, (x, y), ov.C_DIM, 0.31, 1)
        x += ov._w(gl, 0.31) + 14


def _window_bar(f, c, t, T, x, by, width) -> None:
    """All three clocks, in the overlay's colours and the overlay's order.

    intervening (blue) is when we are changing something, consequence (red) is
    when the scene is wrong as a result, observable (amber) is when a viewer
    could tell. Drawing only the last two -- which is what this did -- makes a
    `colour_shift` that finishes turning at frame 14 look like it goes on to the
    end of the clip, because the union does.
    """
    import cv2
    rows = ((c.get("iwin") or c["vwin"], ov.C_INTERVENE),
            (c.get("cwin") or c["vwin"], ov.C_MASK),
            (c["owin"], ov.C_OBS))
    for row, (wins, colour) in enumerate(rows):
        ry = by + row * 5
        cv2.rectangle(f, (x, ry), (x + width - 1, ry + 4), (40, 40, 48), -1)
        for s, e in wins:
            cv2.rectangle(f, (x + int(s / T * width), ry),
                          (x + max(int((e + 1) / T * width) - 1,
                                   int(s / T * width) + 1), ry + 4), colour, -1)
    px = x + min(width - 1, int((t + 0.5) / T * width))
    cv2.line(f, (px, by - 2), (px, by + 15), (255, 255, 255), 1)


def _wrap(s: str, width: int) -> List[str]:
    lines, cur = [], ""
    for word in s.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    if cur:
        lines.append(cur)
    return lines[:3]


def _collect(pair_dir: str, family: Optional[str]) -> List[Dict]:
    cols = []
    for mp in sorted(glob.glob(os.path.join(pair_dir, "*", "meta.json"))):
        cdir = os.path.dirname(mp)
        with open(mp) as fh:
            meta = json.load(fh)
        is_valid = meta.get("label") == "valid"
        if not is_valid and family and meta.get("family") != family:
            continue
        v = meta.get("violation") or {}
        sev = (v.get("intervention") or {}).get("severity_bin", "valid")
        tlp = os.path.join(cdir, "timelines.npz")
        cols.append({
            "dir": cdir, "meta": meta, "is_valid": is_valid,
            "label": "VALID" if is_valid else sev,
            "sort": ORDER.get("valid" if is_valid else sev, 9),
            "rgb": ov._rgb(cdir, int(meta["num_frames"])),
            "mask": ov._npz(cdir, "violation_mask.npz", "mask"),
            "ref": ov._npz(cdir, "reference_mask.npz", "mask"),
            "sev": (lambda a: None if a is None else a.astype(np.float32))(
                ov._npz(cdir, "severity_map.npz", "severity")),
            "causal": ov._npz(cdir, "causal_mask.npz", "mask"),
            "energy": (lambda a: None if a is None else a.astype(np.float32))(
                ov._npz(cdir, "energy_map.npz", "energy")),
            "etrace": ov._energy_trace(cdir),
            "twin_etrace": ov._energy_trace(
                os.path.join(os.path.dirname(cdir), "valid")),
            "seg": ov._npz(cdir, "seg.npz", "seg"),
            "depth": ov._npz(cdir, "depth.npz", "depth"),
            "flow": ov._npz(cdir, "flow_fwd.npz", "flow_fwd"),
            "normals": ov._npz(cdir, "normals.npz", "normals"),
            "div": (lambda a: None if a is None else a.astype(np.float32))(
                ov._npz(cdir, "divergence_map.npz", "divergence")),
            "tl": np.load(tlp) if os.path.exists(tlp) else None,
            "vwin": [tuple(w) for w in v.get("violation_windows", [])],
            "owin": [tuple(w) for w in v.get("observable_windows", [])],
            "iwin": [tuple(w) for w in v.get("intervention_windows", [])],
            "cwin": [tuple(w) for w in v.get("consequence_windows", [])],
            "lag": v.get("observability_lag_frames"),
            "mag": (v.get("intervention") or {}).get("magnitude"),
            "mag_unit": (v.get("intervention") or {}).get("magnitude_unit", ""),
        })
    return sorted(cols, key=lambda c: c["sort"])


def coverage(release_root: str, out_path: Optional[str] = None,
             cell: Optional[int] = None, cols: Optional[int] = None,
             severity: str = "strong") -> Dict[str, object]:
    """The whole release at once: **a row per scenario, a column per family**.

    The "is all of this working" view. It used to reflow every invalid clip into
    a roughly square block, which put unrelated cells next to each other and
    made a missing one invisible -- the tiles simply closed up around the gap.
    On a fixed scenario x family lattice a cell that was never built is a black
    square in a known place, so the shape of the coverage is legible rather than
    something you count.
    """
    import cv2

    clips: Dict[str, Dict[str, Dict]] = {}
    for mp in sorted(glob.glob(os.path.join(release_root, "clips", "**",
                                            "meta.json"), recursive=True)):
        cdir = os.path.dirname(mp)
        with open(mp) as fh:
            meta = json.load(fh)
        if meta.get("label") != "invalid":
            continue
        v = meta.get("violation") or {}
        bin_ = (v.get("intervention") or {}).get("severity_bin", "strong")
        if bin_ != severity:
            continue
        tlp = os.path.join(cdir, "timelines.npz")
        clips.setdefault(meta.get("scenario", "?"), {})[
            meta.get("family", "?")] = {
                "meta": meta,
                "rgb": ov._rgb(cdir, int(meta["num_frames"])),
                "mask": ov._npz(cdir, "violation_mask.npz", "mask"),
                "ref": ov._npz(cdir, "reference_mask.npz", "mask"),
                "tl": np.load(tlp) if os.path.exists(tlp) else None,
                "lag": v.get("observability_lag_frames", 0)}
    if not clips:
        raise ValueError("no invalid %s clips under %s" % (severity, release_root))

    scenarios = sorted(clips)
    families = sorted({f for row in clips.values() for f in row})
    nrow, ncol = len(scenarios), len(families)
    if cell is None:
        cell = 128 if ncol <= 14 else (104 if ncol <= 20 else 84)
    T = min(int(c["meta"]["num_frames"])
            for row in clips.values() for c in row.values())

    head = 46
    W = GUTTER + ncol * cell + (ncol + 1) * PAD
    H = HEADER + head + nrow * (cell + PAD) + PAD
    out = np.zeros((T, H, W, 3), np.uint8)
    built = sum(len(r) for r in clips.values())

    for t in range(T):
        f = np.full((H, W, 3), ov.C_BG, np.uint8)
        ov._text(f, "coverage: %d cells over %d scenarios x %d families  "
                    "(severity %s)   black = not built"
                 % (built, nrow, ncol, severity), (PAD + 2, 20),
                 ov.C_TEXT, 0.48, 1)
        fs = "f %d/%d" % (t, T - 1)
        ov._text(f, fs, (W - PAD - ov._w(fs, 0.46), 20), ov.C_TEXT, 0.46, 1)

        for j, fam in enumerate(families):
            x = GUTTER + PAD + j * (cell + PAD)
            for k, part in enumerate(_wrap(fam.replace("_", " "), 11)):
                ov._text(f, part, (x + 2, HEADER + 13 + 12 * k),
                         ov.C_TEXT, 0.33, 1)

        for i, scen in enumerate(scenarios):
            y = HEADER + head + i * (cell + PAD)
            ov._text(f, scen[:15], (PAD + 2, y + 15), (255, 255, 255), 0.38, 1)
            ov._text(f, "%d/%d" % (len(clips[scen]), ncol),
                     (PAD + 2, y + 29), ov.C_DIM, 0.32, 1)
            for j, fam in enumerate(families):
                x = GUTTER + PAD + j * (cell + PAD)
                c = clips[scen].get(fam)
                if c is None:
                    cv2.rectangle(f, (x, y), (x + cell - 1, y + cell - 1),
                                  (10, 10, 13), -1)
                    cv2.rectangle(f, (x, y), (x + cell - 1, y + cell - 1),
                                  (38, 38, 46), 1)
                    continue
                img = ov._panel("mask", t, c["rgb"], c["mask"], None, None,
                                None, cell, c["ref"])
                f[y:y + cell, x:x + cell] = img
                cv2.rectangle(f, (x, y), (x + cell - 1, y + cell - 1),
                              (60, 60, 70), 1)
                if c["tl"] is not None and bool(c["tl"]["active"][t]):
                    cv2.circle(f, (x + cell - 9, y + 9), 5, ov.C_ACTIVE, -1)
                    cv2.circle(f, (x + cell - 9, y + 9), 5, (255, 255, 255), 1)
        out[t] = f

    out_path = out_path or os.path.join(release_root, "coverage.mp4")
    fps = next(iter(next(iter(clips.values())).values()))["meta"].get("fps", 12)
    vid.write(out, out_path, fps=int(fps))
    return {"path": out_path, "cells": built, "scenarios": nrow,
            "families": ncol, "severity": severity, "frames": T}

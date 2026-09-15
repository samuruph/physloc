"""What a generated release actually contains, as figures and one JSON.

`validate` says a release is well formed and `audit` says every cell depicts
something. Neither says what the DISTRIBUTIONS look like, and the distributions
are what a benchmark is judged on: whether the conditions came out at their
declared shares, whether the severity ladder is monotone in practice and not
just in the plan, where the difficulty thresholds fall against the data they
were fitted on, and which axis is making clips hard.

Every one of those questions has been answered by eye at some point in this
project, and eye-answers have been wrong every time -- a "varied" sampler with
three constants in it, a severity ladder whose bins were V-shaped, thirty-four
clips whose severity was zero. So they get plotted, once per release, from the
shipped `metadata.json` files and nothing else.

    python -m physloc.cli stats out/physloc_v0

Writes `<root>/stats/`: the six figures in `FIGURES` and `stats.json`, the
numbers behind them -- so a regression can be diffed rather than squinted at,
and so the dataset card can quote them without re-deriving anything. `generate`
writes them at the end of every run and `export` ships them with the release.

**Reads `metadata.json` and nothing else** on a current release -- no arrays, no
renders -- so it runs in seconds over a full release and can be re-run after
any annotation change. The one exception is a clip generated before the
difficulty label existed: `load` back-fills it, and reads that clip's masks to
do so.

The look follows benchmark reports (IntPhys 2, LikePhys and the like) rather
than a dashboard: part-to-whole as labelled donuts, distributions as smooth
densities, coverage as a lattice, one quiet palette. Colours are the validated
reference palette of the data-viz method -- categorical slots in a fixed order
for identity, one blue ramp for anything ordered (levels, difficulty,
severity), grey for the neutral -- so a colour means the same thing in every
panel it appears in.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from ..annotate import difficulty as D

# ------------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
NEUTRAL = "#d9d8d3"
#: Categorical slots, IN ORDER -- never cycled, never generated.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948")
#: One hue for everything ordered, light -> dark = low -> high.
ORDINAL3 = ("#86b6ef", "#2a78d6", "#104281")
ORDINAL4 = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
ACCENT = CATEGORICAL[0]
LEVELS = ("L0", "L1", "L2", "L3")
BINS = ("weak", "medium", "strong")
# Kept for callers of the previous report.
EASY, MODERATE, HARD = ORDINAL3
LEVEL_COLOUR = dict(zip(D.LEVELS, ORDINAL3))

#: The figures `report` writes, in reading order, with the caption the dataset
#: card prints under each.
FIGURES: Tuple[Tuple[str, str], ...] = (
    ("composition.png",
     "What the release contains: violated clips by domain and family, clips by "
     "complexity level, and detection difficulty within each level."),
    ("coverage.png",
     "Where the clips come from: clips per scenario, and the scenario x family "
     "lattice -- a pale cell was never built, a grey one is missing."),
    ("difficulty.png",
     "How hard the violations are to detect, and which factor set each label."),
    ("difficulty_factors.png",
     "Each difficulty factor's distribution against its two cuts: easy, "
     "moderate, and hard past the second cut."),
    ("distributions.png",
     "When violations fire, how long until they are visible, and how strong "
     "they measure at each severity bin."),
    ("structure.png",
     "Difficulty conditions against their declared shares, and how the "
     "violators of multi-object clips are timed."),
)


# ------------------------------------------------------------------- reading
def _array(path: str):
    """One array out of an `.npz`, or None. The files here hold exactly one."""
    if not os.path.exists(path):
        return None
    import numpy as np

    with np.load(path) as z:
        keys = list(z.keys())
        return z[keys[0]] if keys else None


def _violation_mask(cdir: str):
    """The clip's violation mask, or None when it has no v2 annotations."""
    from .. import loader

    clip = loader.Clip.from_dir(cdir)
    return clip.violation_mask if clip.has(loader.MASKS) else None


def load(root: str) -> List[Dict[str, object]]:
    """Every clip's `metadata.json` under a release root.

    BACK-FILLS `difficulty` when a clip predates it, so this works on runs
    generated before the label existed -- of which there are several on disk,
    and they are the corpus the thresholds were fitted to. The masks are read
    only in that case, and only when they are sitting beside the metadata;
    a current release carries the label already and this touches no arrays.
    """
    from ..annotate import layout

    out = []
    for path in layout.find(root):
        with open(path) as fh:
            meta = json.load(fh)
        if meta.get("violation") and not meta.get("difficulty"):
            cdir = os.path.dirname(path)
            meta["difficulty"] = D.assess(
                meta,
                _violation_mask(cdir),
                _array(os.path.join(cdir, layout.SEGMENTATIONS)))
        out.append(meta)
    return out


def _md(meta) -> Dict[str, object]:
    return meta.get("metadata") or {}


def _level(meta) -> str:
    c = _md(meta).get("complexity")
    return str((c.get("name") if isinstance(c, dict) else c) or "?")


def summarise(metas: List[Dict[str, object]]) -> Dict[str, object]:
    """The numbers behind every figure, and the file a regression diffs."""
    from .. import taxonomy as T

    invalid = [m for m in metas if m.get("violation")]
    valid = [m for m in metas if not m.get("violation")]

    levels = Counter(_level(m) for m in metas)
    conditions = Counter(str(_md(m).get("condition") or "?") for m in metas)
    bins = Counter(
        str(((m["violation"].get("intervention") or {}).get("severity_bin")))
        for m in invalid)
    diff = Counter(str((m.get("difficulty") or {}).get("level") or "unlabelled")
                   for m in invalid)
    # Through `canonical`: a clip labelled before three factors were renamed
    # carries `footprint`, `clutter` and `camera`, and counts under the new names.
    binding = Counter(
        D.canonical(name) for m in invalid
        for name in ((m.get("difficulty") or {}).get("binding_factors") or ()))

    # Difficulty against the complexity level -- the grid the difficulty module
    # exists to keep measurable, and the reason the level is not a factor in it.
    grid: Dict[str, Counter] = defaultdict(Counter)
    for m in invalid:
        grid[_level(m)][str((m.get("difficulty") or {}).get("level")
                            or "unlabelled")] += 1

    factors: Dict[str, List[float]] = {f.name: [] for f in D.FACTORS}
    for m in invalid:
        got = {D.canonical(k): v for k, v in
               ((m.get("difficulty") or {}).get("factors") or {}).items()}
        for name in factors:
            v = (got.get(name) or {}).get("value")
            if v is not None:
                factors[name].append(float(v))

    severity = defaultdict(list)
    for m in invalid:
        sb = str((m["violation"].get("intervention") or {}).get("severity_bin"))
        pr = (m["violation"].get("peak_residual") or {}).get("score")
        if pr is not None:
            severity[sb].append(float(pr))

    level_cond: Dict[str, Counter] = defaultdict(Counter)
    for m in invalid:
        level_cond[_level(m)][str(_md(m).get("condition") or "?")] += 1
    domains = Counter(str(_md(m).get("domain") or "?") for m in invalid)
    timing = Counter(str(m["violation"].get("violator_timing")
                         or m["violation"].get("culprit_timing") or "shared")
                     for m in invalid)

    # WHEN violations fire: as a share of the clip (is the event band used, or
    # does everything cluster?), in seconds (the same at every clip length),
    # and how long until a viewer could tell.
    share, seconds, lag = [], [], []
    for m in invalid:
        t = m["violation"].get("t_event_frame")
        frames = _md(m).get("num_frames")
        fps = float(_md(m).get("frame_rate") or 12)
        if t is None or not frames:
            continue
        share.append(float(t) / float(frames))
        seconds.append(float(t) / fps)
        lg = m["violation"].get("observability_lag_frames")
        if lg is not None:
            lag.append(float(lg) / fps)

    families = Counter(str(_md(m).get("family")) for m in invalid)
    scenarios = Counter(str(_md(m).get("scenario")) for m in metas)
    cells: Dict[str, Counter] = defaultdict(Counter)
    for m in invalid:
        cells[str(_md(m).get("scenario"))][str(_md(m).get("family"))] += 1
    present = set(scenarios)
    built = sorted([s, f] for s, f in T.build_cells() if s in present)

    try:
        from ..scenarios.base import CONDITION_CYCLE, condition_share
        declared = {c: condition_share(c) for c in dict.fromkeys(CONDITION_CYCLE)}
    except Exception:                                          # noqa: BLE001
        declared = {}

    return {
        "clips": len(metas), "invalid": len(invalid), "valid": len(valid),
        "levels": dict(levels), "conditions": dict(conditions),
        "declared_condition_shares": declared,
        "severity_bins": dict(bins), "difficulty": dict(diff),
        "binding_factors": dict(binding),
        "difficulty_by_level": {k: dict(v) for k, v in grid.items()},
        "level_by_condition": {k: dict(v) for k, v in level_cond.items()},
        "factor_values": factors,
        "peak_score_by_bin": {k: v for k, v in severity.items()},
        "domains": dict(domains), "violator_timing": dict(timing),
        "event_time_share": share, "event_time_seconds": seconds,
        "observability_lag_seconds": lag,
        "families": dict(families), "scenarios": dict(scenarios),
        "family_domain": {f: getattr(T.FAMILIES.get(f), "domain", "?")
                          for f in families},
        "scenario_medium": {s: getattr(T.SCENARIOS.get(s), "physics_medium", "?")
                            for s in scenarios},
        "cells": {k: dict(v) for k, v in cells.items()},
        "built_cells": built,
        "thresholds": {f.name: {"easier": f.easier, "easy": f.easy,
                                "moderate": f.moderate} for f in D.FACTORS},
    }


# ------------------------------------------------------------------- drawing
def _rc() -> Dict[str, object]:
    return {"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE, "font.size": 9,
            "font.family": "DejaVu Sans", "axes.edgecolor": GRID,
            "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
            "text.color": INK, "legend.frameon": False}


def _tint(hex_colour: str, t: float) -> str:
    """`hex_colour` moved `t` of the way towards the surface."""
    c = [int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]
    s = [int(SURFACE[i:i + 2], 16) for i in (1, 3, 5)]
    mix = [round(a + (b - a) * float(t)) for a, b in zip(c, s)]
    return "#%02x%02x%02x" % tuple(mix)


def _ink_on(hex_colour: str) -> str:
    """White or ink, whichever reads on this fill."""
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else INK


def _heading(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.975, title, fontsize=14, fontweight="bold",
             color=INK, va="top")
    fig.text(0.012, 0.915, subtitle, fontsize=9, color=INK2, va="top")


def _panel_title(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold",
                 color=INK, pad=(18 + 11 * subtitle.count("\n")) if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=8,
                color=INK2, va="bottom")


def _quiet(ax, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    if grid_axis:
        getattr(ax, "%saxis" % grid_axis).grid(True, color=GRID, linewidth=0.8)


def _empty(ax, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED,
            fontsize=9, transform=ax.transAxes)


def _donut(ax, names: Sequence[str], values: Sequence[float],
           colours: Sequence[str], centre: str, caption: str,
           label_min: float = 0.025, radius: float = 1.0, width: float = 0.34,
           legend: bool = True, legend_cols: int = 1, span: float = 1.95) -> None:
    """A labelled donut: 2 px surface gaps, direct labels on the slices big
    enough to carry one, a legend for every slice, the total in the middle."""
    import numpy as np

    pairs = [(n, float(v), c) for n, v, c in zip(names, values, colours) if v > 0]
    if not pairs:
        _empty(ax, "nothing in this release")
        return
    names, values, colours = zip(*pairs)
    total = float(sum(values))
    wedges, _ = ax.pie(values, radius=radius, colors=colours, startangle=90,
                       counterclock=False,
                       wedgeprops={"width": width, "edgecolor": SURFACE,
                                   "linewidth": 2.0})
    ax.text(0, 0.07, centre, ha="center", va="center", fontsize=17,
            fontweight="bold", color=INK)
    ax.text(0, -0.2, caption, ha="center", va="center", fontsize=8, color=INK2)
    _slice_labels(ax, wedges, ["%s  %d%%" % (n, round(100 * v / total))
                               for n, v in zip(names, values)],
                  [v / total for v in values], radius, label_min)
    ax.set_aspect("equal")
    ax.set_xlim(-span, span)
    ax.set_ylim(-1.5, 1.4)
    ax.axis("off")
    if legend:
        handles = [__import__("matplotlib").patches.Patch(color=c) for c in colours]
        ax.legend(handles, ["%s (%d)" % (n, v) for n, v in zip(names, values)],
                  loc="upper center", bbox_to_anchor=(0.5, 0.02),
                  ncol=legend_cols, fontsize=7.5, handlelength=0.9,
                  handleheight=0.9, columnspacing=1.0, labelcolor=INK2)
    del np


def _slice_labels(ax, wedges, texts, shares, radius, label_min) -> None:
    """Direct labels outside the ring with elbow leaders, de-overlapped per side."""
    import numpy as np

    sides = {1: [], -1: []}
    for w, text, share in zip(wedges, texts, shares):
        if share < label_min:
            continue
        ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        x, y = float(np.cos(ang)), float(np.sin(ang))
        sides[1 if x >= 0 else -1].append([y, x, text])
    step = 0.15
    for sign, rows in sides.items():
        rows.sort(key=lambda r: -r[0])
        ys = [r[0] * (radius + 0.16) for r in rows]
        for i in range(1, len(ys)):
            ys[i] = min(ys[i], ys[i - 1] - step)
        floor = -(radius + 0.32)
        if ys and ys[-1] < floor:                 # pushed off the bottom: lift
            lift = floor - ys[-1]
            ys = [y + lift for y in ys]
        for (y0, x0, text), yl in zip(rows, ys):
            px, py = x0 * radius, y0 * radius
            ex = x0 * (radius + 0.08)
            lx = sign * (radius + 0.3)
            ax.plot([px, ex, lx - sign * 0.03], [py, yl, yl], color=MUTED,
                    linewidth=0.7, solid_capstyle="round")
            ax.text(lx, yl, text, ha="left" if sign > 0 else "right",
                    va="center", fontsize=7.6, color=INK2)


def _kde(values, lo: float, hi: float, bounds=(None, None), n: int = 256):
    """(x, density): a Gaussian KDE with Scott's bandwidth, reflected at any
    hard bound so a quantity that cannot go below 0 does not leak past it."""
    import numpy as np

    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    xs = np.linspace(lo, hi, n)
    if v.size == 0:
        return xs, np.zeros_like(xs)
    spread = float(v.std())
    if spread <= 0:
        spread = max((hi - lo) * 0.02, 1e-6)
    # A floor on the bandwidth, so a quantity that comes in whole frames (a lag
    # of 0, 1/12, 2/12 s) reads as a distribution rather than a comb.
    bw = max(1.06 * spread * v.size ** (-0.2), (hi - lo) * 0.035)
    pts = [v]
    if bounds[0] is not None:
        pts.append(2.0 * bounds[0] - v)
    if bounds[1] is not None:
        pts.append(2.0 * bounds[1] - v)
    allp = np.concatenate(pts)
    dens = np.exp(-0.5 * ((xs[:, None] - allp[None, :]) / bw) ** 2).sum(1)
    dens /= v.size * bw * np.sqrt(2.0 * np.pi)
    return xs, dens


def _density(ax, values, colour: str, lo: float, hi: float, bounds=(None, None),
             label: Optional[str] = None, wash: bool = True):
    import numpy as np

    xs, ys = _kde(values, lo, hi, bounds)
    if wash:
        ax.fill_between(xs, ys, color=colour, alpha=0.10, linewidth=0)
    ax.plot(xs, ys, color=colour, linewidth=2.0, solid_capstyle="round",
            label=label)
    return xs, ys, float(np.median(values)) if len(values) else None


# ------------------------------------------------------------------- figures
def _fig_composition(plt, s, path) -> None:
    """Taxonomy sunburst, complexity levels, difficulty within each level."""
    from .. import taxonomy as T

    fig = plt.figure(figsize=(15.5, 6.2))
    _heading(fig, "Composition",
             "%d clips: %d with a violation, %d lawful twins"
             % (s["clips"], s["invalid"], s["valid"]))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.15], left=0.02,
                          right=0.985, top=0.8, bottom=0.2, wspace=0.18)

    # (a) taxonomy: domains inside, their families outside in tints.
    ax = fig.add_subplot(gs[0, 0])
    _panel_title(ax, "Taxonomy", "violated clips: domain (inner) and family (outer)")
    fams = s.get("families") or {}
    fd = s.get("family_domain") or {}
    order = [d for d in T.DOMAINS if any(fd.get(f) == d for f in fams)]
    order += sorted({fd.get(f, "?") for f in fams} - set(order))
    dom_val = [sum(v for f, v in fams.items() if fd.get(f) == d) for d in order]
    if sum(dom_val):
        dom_col = [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(order))]
        fam_names, fam_vals, fam_cols = [], [], []
        for d, colour in zip(order, dom_col):
            members = sorted((f for f in fams if fd.get(f) == d),
                             key=lambda f: -fams[f])
            for k, f in enumerate(members):
                fam_names.append(f)
                fam_vals.append(fams[f])
                fam_cols.append(_tint(colour, 0.12 + 0.5 * k / max(len(members), 1)))
        total = float(sum(fam_vals))
        ax.pie(dom_val, radius=0.64, colors=dom_col, startangle=90,
               counterclock=False,
               wedgeprops={"width": 0.26, "edgecolor": SURFACE, "linewidth": 2.0})
        outer, _ = ax.pie(fam_vals, radius=1.0, colors=fam_cols, startangle=90,
                          counterclock=False,
                          wedgeprops={"width": 0.33, "edgecolor": SURFACE,
                                      "linewidth": 1.5})
        ax.text(0, 0.06, "%d" % s["invalid"], ha="center", va="center",
                fontsize=16, fontweight="bold")
        ax.text(0, -0.17, "violated", ha="center", va="center", fontsize=8,
                color=INK2)
        _slice_labels(ax, outer, [f.replace("_", " ") for f in fam_names],
                      [v / total for v in fam_vals], 1.0, 0.03)
        ax.set_aspect("equal")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.5, 1.4)
        ax.axis("off")
        handles = [__import__("matplotlib").patches.Patch(color=c) for c in dom_col]
        ax.legend(handles, ["%s (%d)" % (d, v) for d, v in zip(order, dom_val)],
                  loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4,
                  fontsize=7.5, handlelength=0.9, handleheight=0.9,
                  columnspacing=1.0, labelcolor=INK2)
    else:
        _empty(ax, "no violated clips")

    # (b) complexity levels.
    ax = fig.add_subplot(gs[0, 1])
    _panel_title(ax, "Complexity level", "all clips, realism ladder L0 -> L3")
    lv = [k for k in LEVELS if k in s["levels"]] + sorted(
        k for k in s["levels"] if k not in LEVELS)
    _donut(ax, lv, [s["levels"][k] for k in lv],
           [ORDINAL4[LEVELS.index(k)] if k in LEVELS else NEUTRAL for k in lv],
           "%d" % s["clips"], "clips", legend_cols=4)

    # (c) difficulty within each level, 100% bars.
    ax = fig.add_subplot(gs[0, 2])
    _panel_title(ax, "Difficulty x complexity",
                 "share of violated clips per level; the two axes kept apart")
    grid = s.get("difficulty_by_level") or {}
    rows = [k for k in LEVELS if k in grid] + sorted(k for k in grid if k not in LEVELS)
    if rows:
        _stacked_share(ax, rows, [[grid[r].get(n, 0) for n in D.LEVELS] for r in rows],
                       list(D.LEVELS), list(ORDINAL3))
    else:
        _empty(ax, "no violated clips")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _stacked_share(ax, rows, counts, names, colours) -> None:
    """Horizontal 100% bars with 2 px gaps, in-segment % only where it fits."""
    import numpy as np

    counts = np.asarray(counts, float)
    totals = counts.sum(1)
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for j, (name, colour) in enumerate(zip(names, colours)):
        share = np.where(totals > 0, counts[:, j] / np.maximum(totals, 1), 0)
        ax.barh(y, share, left=left, height=0.56, color=colour,
                edgecolor=SURFACE, linewidth=2.0, label=name)
        for yi, l, w in zip(y, left, share):
            if w >= 0.11:
                ax.text(l + w / 2, yi, "%d%%" % round(100 * w), ha="center",
                        va="center", fontsize=7.5, color=_ink_on(colour))
        left += share
    for yi, t in zip(y, totals):
        ax.text(1.015, yi, "n=%d" % t, va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(rows)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    _quiet(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=len(names),
              fontsize=7.5, handlelength=0.9, labelcolor=INK2)


def _fig_coverage(plt, s, path) -> None:
    """Clips per scenario, and the scenario x family lattice."""
    import numpy as np
    from .. import taxonomy as T

    scen = s.get("scenarios") or {}
    medium = s.get("scenario_medium") or {}
    media_order = [m for m in T.MEDIA if m in set(medium.values())]
    rows = sorted(scen, key=lambda k: (media_order.index(medium.get(k))
                                       if medium.get(k) in media_order else 99, k))
    built = {(a, b) for a, b in (s.get("built_cells") or [])}
    fd = s.get("family_domain") or {}
    fams_built = {b for _, b in built}
    for f in fd:
        fams_built.add(f)
    doms = list(T.DOMAINS)
    cols = sorted(fams_built, key=lambda f: (
        doms.index(getattr(T.FAMILIES.get(f), "domain", "")) if
        getattr(T.FAMILIES.get(f), "domain", "") in doms else 99, f))

    fig = plt.figure(figsize=(16, 6.8))
    _heading(fig, "Coverage",
             "%d scenarios x %d families; %d buildable cells in this release"
             % (len(rows), len(cols), len(built)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 2.3], left=0.03,
                          right=0.99, top=0.8, bottom=0.2, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    _panel_title(ax, "Clips per scenario", "every clip, lawful and violated")
    if rows:
        _donut(ax, rows, [scen[r] for r in rows], [ACCENT] * len(rows),
               "%d" % s["clips"], "clips", label_min=1.01, legend_cols=3)
        # Media as a thin inner ring, in the neutral: identity is the label.
        med_vals = [sum(scen[r] for r in rows if medium.get(r) == m)
                    for m in media_order]
        if len(media_order) > 1:
            inner, _ = ax.pie(med_vals, radius=0.62, startangle=90,
                              counterclock=False,
                              colors=[_tint(INK2, 0.35 + 0.4 * i / len(media_order))
                                      for i in range(len(media_order))],
                              wedgeprops={"width": 0.07, "edgecolor": SURFACE,
                                          "linewidth": 2.0})
            for w, m in zip(inner, media_order):
                if (w.theta1 - w.theta2) == 0:
                    continue
                ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
                ax.text(0.44 * np.cos(ang), 0.44 * np.sin(ang), m, fontsize=6.5,
                        color=INK2, ha="center", va="center")
    else:
        _empty(ax, "no clips")

    ax = fig.add_subplot(gs[0, 1])
    _panel_title(ax, "Scenario x family",
                 "violated clips per cell; pale = not a meaningful cell, "
                 "grey = buildable but missing")
    cells = s.get("cells") or {}
    if rows and cols:
        counts = np.array([[cells.get(r, {}).get(c, 0) for c in cols] for r in rows],
                          float)
        top = max(1.0, counts.max())
        rgb = np.ones(counts.shape + (3,))
        from matplotlib.colors import LinearSegmentedColormap, to_rgb
        cmap = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#0d366b"])
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                if counts[i, j] > 0:
                    rgb[i, j] = cmap(0.08 + 0.92 * counts[i, j] / top)[:3]
                elif (r, c) in built:
                    rgb[i, j] = to_rgb(NEUTRAL)
                else:
                    rgb[i, j] = to_rgb("#f3f2ef")
        ax.imshow(rgb, aspect="auto", interpolation="nearest")
        for i in range(len(rows)):
            for j in range(len(cols)):
                if counts[i, j] > 0:
                    fill = "#%02x%02x%02x" % tuple(int(255 * x) for x in rgb[i, j])
                    ax.text(j, i, "%d" % counts[i, j], ha="center", va="center",
                            fontsize=6.3, color=_ink_on(fill))
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels([c.replace("_", " ") for c in cols], rotation=55,
                           ha="right", fontsize=7.5)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(["%s  (%s)" % (r, medium.get(r, "?")) for r in rows],
                           fontsize=7.8)
        ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.6)
        ax.tick_params(which="both", length=0, colors=INK2)
        for side in ax.spines.values():
            side.set_visible(False)
        # Domain strip over the columns, in the taxonomy's colours.
        for j, c in enumerate(cols):
            d = getattr(T.FAMILIES.get(c), "domain", None)
            if d in doms:
                ax.add_patch(__import__("matplotlib").patches.Rectangle(
                    (j - 0.5, -1.05), 1.0, 0.35, clip_on=False,
                    color=CATEGORICAL[doms.index(d) % len(CATEGORICAL)],
                    linewidth=0))
        ax.set_ylim(len(rows) - 0.5, -1.1)
    else:
        _empty(ax, "no cells")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _fig_difficulty(plt, s, path) -> None:
    """The label split, and which factor set it."""
    import numpy as np

    fig = plt.figure(figsize=(12.5, 5.4))
    _heading(fig, "Detection difficulty",
             "every violated clip is easy, moderate or hard; its label is its "
             "worst factor")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], left=0.02,
                          right=0.97, top=0.78, bottom=0.14, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    _panel_title(ax, "Difficulty label")
    _donut(ax, list(D.LEVELS), [s["difficulty"].get(k, 0) for k in D.LEVELS],
           list(ORDINAL3), "%d" % s["invalid"], "violated clips", legend_cols=3)

    ax = fig.add_subplot(gs[0, 1])
    _panel_title(ax, "Which factor set the label",
                 "share of violated clips; a clip can be bound by several")
    bind = s.get("binding_factors") or {}
    order = sorted(bind, key=lambda k: -bind[k])
    if order:
        total = max(1, s["invalid"])
        vals = [bind[k] / total for k in order]
        y = np.arange(len(order))[::-1]
        ax.barh(y, vals, height=0.56, color=ACCENT)
        for yi, v, k in zip(y, vals, order):
            ax.text(v + 0.012, yi, "%d%%  (%d)" % (round(100 * v), bind[k]),
                    va="center", fontsize=7.8, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_xlim(0, min(1.15, max(vals) * 1.3 + 0.05))
        ax.xaxis.set_major_formatter(
            __import__("matplotlib").ticker.PercentFormatter(1.0, decimals=0))
        _quiet(ax, "x")
        ax.spines["left"].set_visible(False)
    else:
        _empty(ax, "no labels")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _fig_factors(plt, s, path) -> None:
    """Each factor against its three zones."""
    import numpy as np

    fs = list(D.FACTORS)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6))
    fig.subplots_adjust(left=0.04, right=0.985, top=0.8, bottom=0.08,
                        wspace=0.28, hspace=0.62)
    _heading(fig, "Difficulty factors",
             "each factor has two cuts, so three zones: easy | moderate | hard "
             "past the second cut. The label is the worst zone a clip reaches.")
    for ax, f in zip(axes.ravel(), fs):
        vals = np.asarray(s["factor_values"].get(f.name) or [], float)
        _factor_panel(ax, f, vals)
    legend_ax = axes.ravel()[len(fs)]
    legend_ax.axis("off")
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=_tint(c, 0.55)) for c in ORDINAL3]
    legend_ax.legend(handles, ["easy", "moderate", "hard"], loc="center left",
                     fontsize=9, handlelength=1.2, handleheight=1.2,
                     labelcolor=INK2, title="zone", title_fontsize=9)
    legend_ax.text(0.02, 0.12, "violation_area is the violation's AREA:\n"
                   "the fraction of the frame it covers\nat its largest.",
                   transform=legend_ax.transAxes, fontsize=8, color=INK2)
    for ax in axes.ravel()[len(fs) + 1:]:
        ax.axis("off")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _factor_panel(ax, f, vals) -> None:
    import numpy as np

    shares = [0, 0, 0]
    for v in vals:
        shares[f.level(float(v))] += 1
    n = max(1, int(vals.size))
    # The zone shares on ONE line under the title. Written inside the zones
    # they collided and were clipped wherever a zone was narrow.
    line = ("   ".join("%s %d%%" % (D.LEVELS[k], round(100 * shares[k] / n))
                       for k in range(3))
            if vals.size else "not measured in this release")
    _panel_title(ax, f.name, "%s, %s is easier\n%s" % (
        f.unit, "higher" if f.easier == "high" else "lower", line))

    cuts = sorted([f.easy, f.moderate])
    # A quantity bounded to 0-1 is drawn on 0-1: `severity` reached 1.4 on an
    # axis whose value stops at 1.0.
    bounded01 = "0-1" in f.unit or f.unit.startswith("fraction")
    if vals.size:
        lo = min(0.0, float(vals.min()))
        # A long tail is cut at the 95th percentile, so a handful of pour clips
        # with twenty violators does not squeeze everything else into a sliver.
        hi = max(float(np.percentile(vals, 95)), cuts[1] * 1.6, cuts[1] + 1e-3)
    else:
        lo, hi = 0.0, cuts[1] * 1.6 + 1e-3
    if bounded01:
        hi = min(max(hi, cuts[1] * 1.2), 1.0)
    # Zones, easier side first.
    if f.easier == "high":
        zones = ((f.easy, hi, 0), (f.moderate, f.easy, 1), (lo, f.moderate, 2))
    else:
        zones = ((lo, f.easy, 0), (f.easy, f.moderate, 1), (f.moderate, hi, 2))
    for a, b, k in zones:
        if b > a:
            ax.axvspan(a, b, color=_tint(ORDINAL3[k], 0.78), linewidth=0, zorder=0)
    for c in cuts:
        ax.axvline(c, color=INK2, linewidth=0.9, zorder=3)
    integer = bool(vals.size) and bool(np.all(np.equal(np.mod(vals, 1), 0)))
    if integer:
        ks, cs = np.unique(vals, return_counts=True)
        ax.bar(ks, cs / n, width=0.6, color=ACCENT, zorder=2)
        ax.set_ylabel("share of clips", fontsize=7.5)
        ax.set_ylim(0, float((cs / n).max()) * 1.2)
    elif vals.size:
        bounds = (0.0 if lo >= 0 else None, 1.0 if bounded01 else None)
        _xs, ys, _med = _density(ax, vals, ACCENT, lo, hi, bounds)
        ax.set_yticks([])
        ax.set_ylim(0, float(ys.max()) * 1.12 if ys.max() > 0 else 1.0)
    else:
        ax.set_yticks([])
    ax.set_xlim(lo, hi)
    _quiet(ax, "")
    ax.spines["left"].set_visible(integer)


def _fig_distributions(plt, s, path) -> None:
    """Event moment, observability lag, and measured severity by bin."""
    import numpy as np

    from ..injectors._geom import EVENT_BAND

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.8))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.08,
                        wspace=0.16, hspace=0.78)
    _heading(fig, "Distributions",
             "smooth densities over the violated clips; a dot marks the median")

    def median_mark(ax, xs, ys, med, unit):
        if med is None:
            return
        y = float(np.interp(med, xs, ys))
        ax.plot([med], [y], "o", ms=6.5, color=ACCENT, markeredgecolor=SURFACE,
                markeredgewidth=2.0, zorder=5)
        # Below and to the right of the dot, and with headroom above the peak,
        # so the label never runs into the panel's subtitle.
        ax.annotate("median %s" % unit % med, (med, y), xytext=(10, -16),
                    textcoords="offset points", fontsize=7.8, color=INK2)
        ax.set_ylim(0, max(ax.get_ylim()[1], float(np.max(ys)) * 1.18))

    ax = axes[0, 0]
    _panel_title(ax, "When violations fire", "seconds into the clip")
    vals = s.get("event_time_seconds") or []
    if vals:
        hi = max(vals) * 1.1 + 0.05
        xs, ys, med = _density(ax, vals, ACCENT, 0.0, hi, (0.0, None))
        median_mark(ax, xs, ys, med, "%.2f s")
        ax.set_xlim(0, hi)
    else:
        _empty(ax, "no events")
    _quiet(ax, "")
    ax.set_yticks([])

    ax = axes[0, 1]
    _panel_title(ax, "...as a share of the clip",
                 "shaded: the band a moment is drawn from when physics does "
                 "not set it")
    vals = s.get("event_time_share") or []
    ax.axvspan(EVENT_BAND[0], EVENT_BAND[1], color=_tint(NEUTRAL, 0.65),
               linewidth=0, zorder=0)
    if vals:
        xs, ys, med = _density(ax, vals, ACCENT, 0.0, 1.0, (0.0, 1.0))
        median_mark(ax, xs, ys, med, "%.2f")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(
        __import__("matplotlib").ticker.PercentFormatter(1.0, decimals=0))
    _quiet(ax, "")
    ax.set_yticks([])

    ax = axes[1, 0]
    vals = s.get("observability_lag_seconds") or []
    at_once = (100.0 * sum(1 for v in vals if v <= 0) / len(vals)) if vals else 0
    _panel_title(ax, "Observability lag",
                 "seconds from the event until a viewer could tell; "
                 "%d%% visible at once" % round(at_once))
    if vals:
        hi = max(max(vals) * 1.15, 0.25)
        xs, ys, med = _density(ax, vals, ACCENT, 0.0, hi, (0.0, None))
        median_mark(ax, xs, ys, med, "%.2f s")
        ax.set_xlim(0, hi)
    else:
        _empty(ax, "no events")
    _quiet(ax, "")
    ax.set_yticks([])

    ax = axes[1, 1]
    _panel_title(ax, "Measured severity by bin",
                 "peak bounded residual (0 = lawful, 1 = saturated); the "
                 "ladder should climb")
    pk = s.get("peak_score_by_bin") or {}
    drew = False
    for b, colour in zip(BINS, ORDINAL3):
        vals = pk.get(b) or []
        if len(vals) >= 2:
            _density(ax, vals, colour, 0.0, 1.0, (0.0, 1.0),
                     label="%s  n=%d, median %.2f" % (b, len(vals), float(np.median(vals))),
                     wash=False)
            drew = True
    if drew:
        ax.legend(loc="upper left", fontsize=7.8, labelcolor=INK2,
                  handlelength=1.4)
        ax.set_xlim(0, 1)
        _quiet(ax, "")
        ax.set_yticks([])
    else:
        _empty(ax, "fewer than two clips per bin")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _fig_structure(plt, s, path) -> None:
    """Conditions against declared shares; how multi-object violators are timed."""
    fig = plt.figure(figsize=(12.5, 5.6))
    _heading(fig, "Structure",
             "every clip carries exactly one difficulty condition")
    gs = fig.add_gridspec(1, 2, left=0.02, right=0.98, top=0.78, bottom=0.14,
                          wspace=0.2)
    ax = fig.add_subplot(gs[0, 0])
    declared = s.get("declared_condition_shares") or {}
    conds = [c for c in declared if c in s["conditions"]] + sorted(
        c for c in s["conditions"] if c not in declared)
    _panel_title(ax, "Difficulty condition",
                 "all clips; legend gives the declared share in brackets")
    total = max(1, sum(s["conditions"].values()))
    _donut(ax, conds, [s["conditions"][c] for c in conds],
           [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(conds))],
           "%d" % total, "clips", legend=False)
    import matplotlib.patches as mpatches
    ax.legend([mpatches.Patch(color=CATEGORICAL[i % len(CATEGORICAL)])
               for i in range(len(conds))],
              ["%s %d%% [%d%%]" % (c, round(100 * s["conditions"][c] / total),
                                   round(100 * declared.get(c, 0)))
               for c in conds],
              loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3,
              fontsize=7.5, handlelength=0.9, labelcolor=INK2)

    ax = fig.add_subplot(gs[0, 1])
    _panel_title(ax, "Violator timing",
                 "violated clips: one moment, a forced shared moment, "
                 "or each violator its own")
    timing = s.get("violator_timing") or {}
    names = [k for k in ("shared", "sync", "independent") if k in timing]
    names += sorted(k for k in timing if k not in names)
    colours = {"shared": NEUTRAL, "sync": CATEGORICAL[5], "independent": CATEGORICAL[6]}
    _donut(ax, names, [timing[k] for k in names],
           [colours.get(k, MUTED) for k in names],
           "%d" % s["invalid"], "violated clips", legend_cols=3)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def report(root: str, outdir: Optional[str] = None,
           metas: Optional[List[Dict[str, object]]] = None) -> Dict[str, object]:
    """Write `FIGURES` and `stats.json`; return the summary.

    `metas` lets a caller pass clips from several roots at once.
    """
    import matplotlib
    matplotlib.use("Agg")                       # headless box, no display
    import matplotlib.pyplot as plt

    metas = load(root) if metas is None else metas
    if not metas:
        raise SystemExit("no clips under %s" % root)
    s = summarise(metas)
    out = outdir or os.path.join(root, "stats")
    os.makedirs(out, exist_ok=True)
    figures = {"composition.png": _fig_composition,
               "coverage.png": _fig_coverage,
               "difficulty.png": _fig_difficulty,
               "difficulty_factors.png": _fig_factors,
               "distributions.png": _fig_distributions,
               "structure.png": _fig_structure}
    with plt.rc_context(_rc()):
        for name, _caption in FIGURES:
            figures[name](plt, s, os.path.join(out, name))
    # Figures an older report wrote, gone from this one: remove them, or a
    # re-run leaves a stale severity.png beside the new distributions.png.
    for stale in ("severity.png", "timing.png"):
        path = os.path.join(out, stale)
        if os.path.exists(path):
            os.remove(path)
    with open(os.path.join(out, "stats.json"), "w") as fh:
        json.dump(s, fh, indent=2, sort_keys=True)
    s["outdir"] = out
    return s

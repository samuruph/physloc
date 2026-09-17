"""What a generated schema-v3 release contains, as figures and one JSON.

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
shipped `sample.json` files and nothing else.

    python -m physloc.cli stats out/physloc_v0

Writes `<root>/stats/`: the six figures in `FIGURES` and `stats.json`, the
numbers behind them -- so a regression can be diffed rather than squinted at,
and so the dataset card can quote them without re-deriving anything. `generate`
writes them at the end of every run and `export` ships them with the release.

**Reads `sample.json` and nothing else** -- no dense arrays or renders -- so it
runs in seconds over a full release and can be re-run after metadata changes.

The look follows benchmark reports (IntPhys 2, LikePhys and the like) rather
than a dashboard: part-to-whole as labelled donuts, distributions as histograms,
coverage as a lattice, one quiet palette. Colours are the validated
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
#: Conditions in a fixed order, so each keeps one colour whatever order the
#: summary's keys arrive in (`stats.json` is written with sorted keys).
CONDITIONS = ("standard", "camera", "distractors", "multi", "camera+multi")
# Kept for callers of the previous report.
EASY, MODERATE, HARD = ORDINAL3
LEVEL_COLOUR = dict(zip(D.LEVELS, ORDINAL3))

#: The figures `report` writes, with the caption the dataset card prints under
#: each. Numbered so a directory listing is the reading order; one question each.
FIGURES: Tuple[Tuple[str, str], ...] = (
    ("1_overview.png",
     "What the release is made of: complexity levels, difficulty conditions "
     "against their declared shares, severity bins, and violator timing."),
    ("2_taxonomy.png",
     "Which physics is violated -- domain (inner ring) and family (outer "
     "ring) -- and how many clips each scenario contributes."),
    ("3_coverage.png",
     "The scenario x family lattice: violated clips per cell, families grouped "
     "under their domain, with row and column totals."),
    ("4_difficulty.png",
     "Detection difficulty: the label split, the split per complexity level, "
     "which factor set each label, and the rule table labels are computed from."),
    ("5_difficulty_factors.png",
     "Every difficulty factor as a histogram of the release, against the two "
     "cuts that make its easy, moderate and hard zones."),
    ("6_timing_and_severity.png",
     "When violations fire, how long until they are visible, and how strong "
     "they measure at each severity bin."),
)


# ------------------------------------------------------------------- reading
def load(root: str) -> List[Dict[str, object]]:
    """Return every sample as the compact report record expected below."""
    from ..annotate import layout

    out = []
    for path in layout.find(root):
        document = layout.read(os.path.dirname(path))
        info = document["metadata"]["sample_info"]
        scene = document["metadata"]["scene"]["info"]
        world = document["metadata"]["scene"]["world"]
        counts = world.get("objects_summary") or {}
        annotations = document.get("annotations") or {}
        summary = annotations.get("violation_summary")
        record = {
            "metadata": {
                "sample_uid": info.get("sample_uid"),
                "pair_uid": info.get("pair_uid"),
                "label": info.get("label"),
                "release": info.get("dataset_version"),
                "seed": info.get("seed"),
                "variant": info.get("variant"),
                "num_frames": info.get("num_frames"),
                "frame_rate": info.get("fps"),
                "resolution": info.get("resolution"),
                "scenario": scene.get("type"),
                "family": scene.get("family"),
                "domain": scene.get("domain"),
                "physics_medium": scene.get("physics_medium"),
                "condition": scene.get("condition"),
                "complexity": {"name": scene.get("complexity")},
                "n_actors": counts.get("n_actors", counts.get("n_subjects", 0)),
                "n_distractors": counts.get("n_distractors", counts.get("n_context", 0)),
                "n_violators": counts.get("n_violators", 0),
            },
            "camera": world.get("camera") or {},
            "violation": summary,
            "difficulty": scene.get("difficulty_analysis"),
        }
        # Early schema-v3 samples retained the label but accidentally dropped
        # the measured factor block. All inputs needed to reproduce it remain
        # in sample.json, so reports repair those samples without opening HDF5.
        if summary and not record["difficulty"]:
            record["difficulty"] = D.assess(record)
        out.append(record)
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
    binding = Counter(
        name for m in invalid
        for name in ((m.get("difficulty") or {}).get("binding_factors") or ()))

    # Difficulty against the complexity level -- the grid the difficulty module
    # exists to keep measurable, and the reason the level is not a factor in it.
    grid: Dict[str, Counter] = defaultdict(Counter)
    for m in invalid:
        grid[_level(m)][str((m.get("difficulty") or {}).get("level")
                            or "unlabelled")] += 1

    factors: Dict[str, List[float]] = {f.name: [] for f in D.FACTORS}
    for m in invalid:
        got = (m.get("difficulty") or {}).get("factors") or {}
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
        "samples": len(metas), "invalid": len(invalid), "valid": len(valid),
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
                                "moderate": f.moderate, "basis": f.basis}
                       for f in D.FACTORS},
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


def _shade(hex_colour: str, k: float) -> str:
    """`hex_colour` scaled towards black -- k < 1 darkens."""
    c = [int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(v * float(k)))) for v in c)


def _luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _readable_fill(hex_colour: str) -> str:
    """The same hue, dark enough to carry WHITE text.

    Every label inside a ring is white, in every figure -- a label whose colour
    flips with the slice it sits on reads as two different kinds of label. So
    the fill moves to meet the text rather than the other way round.
    """
    out = hex_colour
    while _luminance(out) > 0.45:
        out = _shade(out, 0.88)
    return out


def _ink_on(hex_colour: str) -> str:
    """White or ink, whichever reads on this fill."""
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else INK


def _heading(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.975, title, fontsize=14, fontweight="bold",
             color=INK, va="top")
    fig.text(0.012, 0.925, subtitle, fontsize=9, color=INK2, va="top")


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


def _patch(colour: str):
    import matplotlib.patches as mpatches
    return mpatches.Patch(color=colour)


# ----------------------------------------------------- labels inside a ring
def _points_per_unit(ax) -> float:
    """Points per data unit along x -- rings are drawn with an equal aspect.

    The aspect is applied first: until then the axes fill their whole grid
    cell, and a ring measured in that box is wider than the one drawn, which
    passed labels that then spilled over their slice."""
    ax.apply_aspect()
    a = ax.transData.transform((0.0, 0.0))
    b = ax.transData.transform((1.0, 0.0))
    return float(b[0] - a[0]) * 72.0 / ax.figure.dpi


def _text_size(ax, text: str, fontsize: float, weight: str = "normal"):
    """(width, height) in points of `text`, unrotated."""
    fig = ax.figure
    t = ax.text(0, 0, text, fontsize=fontsize, fontweight=weight)
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    return bb.width * 72.0 / fig.dpi, bb.height * 72.0 / fig.dpi


def _upright(deg: float) -> float:
    d = (deg + 180.0) % 360.0 - 180.0
    if d > 90.0:
        d -= 180.0
    elif d < -90.0:
        d += 180.0
    return d


def _ring_labels(ax, wedges, candidates, r_in: float, r_out: float,
                 colours: Sequence[str], fontsize: float = 7.5,
                 weight: str = "normal",
                 text_colour: Optional[str] = None) -> List[int]:
    """Write each slice's label INSIDE the slice, where one of its candidate
    strings fits -- along the ring first, then across it. Returns the indices
    of the slices where nothing fitted, for the caller to label outside.

    Measured rather than guessed: the rendered text box against the slice's
    chord and the ring's thickness, both in points, so the same call is right
    at any figure size.
    """
    import numpy as np

    k = _points_per_unit(ax)
    thick = (r_out - r_in) * k
    r_mid = (r_in + r_out) / 2.0
    pad = 4.0
    missed = []
    for i, (w, options) in enumerate(zip(wedges, candidates)):
        span = np.deg2rad(abs(w.theta2 - w.theta1))
        ang = (w.theta1 + w.theta2) / 2.0
        # Chords: at the middle of the ring for text along it, and near the
        # inner edge (the narrow end) for text across it.
        along = 2.0 * r_mid * np.sin(min(span, np.pi) / 2.0) * k
        across = 2.0 * (r_in + 0.45 * (r_out - r_in)) * np.sin(min(span, np.pi) / 2.0) * k
        placed = False
        for text in options:
            tw, th = _text_size(ax, text, fontsize, weight)
            if span >= np.pi and tw + pad <= thick * 2.2 and th + pad <= thick:
                rot = 0.0                     # a big slice: just write it level
            elif (tw + pad <= along * 0.92
                  and th + pad + tw * tw / (8.0 * r_mid * k) <= thick):
                rot = _upright(ang - 90.0)
            elif tw + 2.0 * pad <= thick * 0.85 and th + pad <= across * 0.85:
                rot = _upright(ang)
            else:
                continue
            x, y = r_mid * np.cos(np.deg2rad(ang)), r_mid * np.sin(np.deg2rad(ang))
            ax.text(x, y, text, rotation=rot, rotation_mode="anchor",
                    ha="center", va="center", fontsize=fontsize,
                    fontweight=weight, linespacing=1.05,
                    color=text_colour or _ink_on(colours[i]))
            placed = True
            break
        if not placed:
            missed.append(i)
    return missed


def _slice_labels(ax, wedges, texts, radius: float) -> None:
    """Labels outside the ring with elbow leaders, de-overlapped per side --
    the fallback for slices too small to hold their own text."""
    import numpy as np

    sides = {1: [], -1: []}
    for w, text in zip(wedges, texts):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        x, y = float(np.cos(ang)), float(np.sin(ang))
        sides[1 if x >= 0 else -1].append([y, x, text])
    step = 0.13
    for sign, rows in sides.items():
        rows.sort(key=lambda r: -r[0])
        ys = [r[0] * (radius + 0.16) for r in rows]
        for i in range(1, len(ys)):
            ys[i] = min(ys[i], ys[i - 1] - step)
        floor = -(radius + 0.3)
        if ys and ys[-1] < floor:
            lift = floor - ys[-1]
            ys = [y + lift for y in ys]
        for (y0, x0, text), yl in zip(rows, ys):
            px, py = x0 * radius, y0 * radius
            ex = x0 * (radius + 0.08)
            lx = sign * (radius + 0.28)
            ax.plot([px, ex, lx - sign * 0.03], [py, yl, yl], color=MUTED,
                    linewidth=0.7, solid_capstyle="round")
            ax.text(lx, yl, text, ha="left" if sign > 0 else "right",
                    va="center", fontsize=7.4, color=INK2)


def _donut(ax, names: Sequence[str], values: Sequence[float],
           colours: Sequence[str], centre: str, caption: str,
           legend_labels: Optional[Sequence[str]] = None,
           legend_cols: int = 2, width: float = 0.44) -> None:
    """A donut whose slices carry their own name and share, the total in the
    middle, and a legend with the counts. Slices too thin for text are labelled
    outside with a leader."""
    pairs = [(n, float(v), c, l) for n, v, c, l in zip(
        names, values, colours, legend_labels or [None] * len(names)) if v > 0]
    if not pairs:
        _empty(ax, "nothing in this release")
        return
    names, values, colours, legend_labels = zip(*pairs)
    total = float(sum(values))
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.42, 1.32)
    ax.axis("off")
    wedges, _ = ax.pie(values, radius=1.0, colors=colours, startangle=90,
                       counterclock=False,
                       wedgeprops={"width": width, "edgecolor": SURFACE,
                                   "linewidth": 2.0})
    ax.text(0, 0.08, centre, ha="center", va="center", fontsize=16,
            fontweight="bold", color=INK)
    ax.text(0, -0.2, caption, ha="center", va="center", fontsize=8, color=INK2)
    share = ["%d%%" % round(100 * v / total) for v in values]
    missed = _ring_labels(
        ax, wedges, [("%s\n%s" % (n, s), "%s %s" % (n, s))
                     for n, s in zip(names, share)],
        1.0 - width, 1.0, colours, fontsize=8)
    if missed:
        _slice_labels(ax, [wedges[i] for i in missed],
                      ["%s %s" % (names[i], share[i]) for i in missed], 1.0)
    labels = [l or "%s (%d)" % (n, v)
              for n, v, l in zip(names, values, legend_labels)]
    ax.legend([_patch(c) for c in colours], labels, loc="upper center",
              bbox_to_anchor=(0.5, 0.02), ncol=legend_cols, fontsize=7.5,
              handlelength=0.9, handleheight=0.9, columnspacing=1.2,
              labelcolor=INK2)


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
            if w >= 0.08:
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


# ------------------------------------------------------------- distributions
#: Bars, not a smooth curve, and no scattered dots.
#:
#: The first version of this report drew a Gaussian density with one dot per
#: clip underneath it, the dots' HEIGHT being jitter so they would not overlap.
#: Nobody could tell what the vertical axis meant -- reasonably, since it meant
#: nothing -- and dots landing on one vertical line looked like a bug rather
#: than like several clips sharing a value. A histogram answers the question
#: people actually asked of it: how many clips are in this range.
HIST_BINS = 28


def _density(values, lo: float, hi: float, points: int = 256):
    """A Gaussian kernel density of `values` on [lo, hi], or None.

    Silverman's bandwidth, never narrower than a sixtieth of the axis so a
    release where most clips share one value draws a peak rather than a
    spike. REFLECTED at 0 and, for a quantity that cannot exceed 1, at 1:
    every value plotted here is a share, a duration or a score, and a kernel
    that leaks past a hard bound draws mass where no clip can be.
    Every value counts, including those past a trimmed axis, so the tail
    the axis hides still shapes the curve up to its edge.
    """
    import numpy as np

    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 3 or hi <= lo:
        return None
    spread = min(float(np.std(v)),
                 float(np.subtract(*np.percentile(v, [75, 25]))) / 1.34) \
        or float(np.std(v))
    bw = max(0.9 * spread * v.size ** -0.2, (hi - lo) / 60.0)
    mirrored = [v]
    if float(v.min()) >= 0.0:
        mirrored.append(-v)
    if float(v.max()) <= 1.0 and hi >= 1.0:
        mirrored.append(2.0 - v)
    pool = np.concatenate(mirrored)
    x = np.linspace(lo, hi, points)
    z = (x[:, None] - pool[None, :]) / bw
    y = np.exp(-0.5 * z * z).sum(axis=1) / (v.size * bw * np.sqrt(2.0 * np.pi))
    return x, y


def _hist(ax, values, lo: float, hi: float, colour: str = ACCENT,
          bins: int = HIST_BINS, share: bool = False, step: bool = False,
          label: Optional[str] = None, curve: bool = True):
    """Draw a histogram with its density on top; return (tallest mark,
    median, how many fell past `hi`).

    The curve is in the bars' own units -- clips per bin, or a share per bin
    -- so it sits on them rather than on an axis of its own."""
    import numpy as np

    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    counts, edges = np.histogram(v, bins=bins, range=(float(lo), float(hi)))
    y = counts / max(1, v.size) if share else counts
    if step:
        ax.step(np.append(edges, edges[-1]), np.append(np.append(y, y[-1]), 0),
                where="post", color=_tint(colour, 0.35), linewidth=1.1,
                zorder=3)
    else:
        ax.bar((edges[:-1] + edges[1:]) / 2.0, y, width=(edges[1] - edges[0]) * 0.92,
               color=_tint(colour, 0.25) if curve else colour, linewidth=0,
               zorder=2, label=None if step else label)
    top = float(y.max()) if len(y) else 0.0
    dens = _density(v, lo, hi) if curve else None
    if dens is not None:
        x, d = dens
        d = d * (edges[1] - edges[0]) * (1.0 if share else v.size)
        ax.plot(x, d, color=_shade(colour, 0.75) if not step else colour,
                linewidth=2.0, zorder=5, label=label if step else None,
                solid_capstyle="round")
        top = max(top, float(d.max()))
    elif step:
        ax.plot([], [], color=colour, linewidth=2.0, label=label)
    return (top,
            float(np.median(v)) if v.size else None,
            int((v > hi).sum()))


def _median_line(ax, med: Optional[float], fmt: str, _y: float = 0.0,
                 y_axes: float = 0.62) -> None:
    """A dashed rule at the median, labelled -- the one summary a bar chart
    cannot show on its own.

    Placed in AXES coordinates, below the zone names and the cut boxes, and
    flipped to the left of its own rule in the right-hand fifth of the panel:
    a `severity` median of 1.00 sits on the axis edge and its label ran off
    the figure.
    """
    import matplotlib.transforms as mtransforms

    if med is None:
        return
    ax.axvline(med, color=INK, linewidth=1.0, linestyle=(0, (2, 2)), zorder=6)
    lo, hi = ax.get_xlim()
    at = (float(med) - lo) / (hi - lo) if hi > lo else 0.5
    right = at > 0.8
    band = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(med, y_axes, ("median " + fmt) % med, transform=band,
            ha="right" if right else "left", va="center", fontsize=7.6,
            color=INK, zorder=7,
            bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none",
                  "alpha": 0.9})


def _beyond(ax, n: int, y: float = 0.72) -> None:
    if n:
        ax.text(0.99, y, "+%d beyond the axis >" % n, transform=ax.transAxes,
                ha="right", fontsize=7, color=MUTED, zorder=7,
                bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none",
                      "alpha": 0.85})


# ------------------------------------------------------------------- figures
def _fig_overview(plt, s, path) -> None:
    """Part-to-whole of the release along its four declared axes."""
    fig = plt.figure(figsize=(17, 5.8))
    _heading(fig, "Overview",
             "%d samples  |  %d with a violation  |  %d lawful twins  |  "
             "%d scenarios  |  %d families"
             % (s["samples"], s["invalid"], s["valid"],
                len(s.get("scenarios") or {}), len(s.get("families") or {})))
    gs = fig.add_gridspec(1, 4, left=0.01, right=0.99, top=0.8, bottom=0.2,
                          wspace=0.12)

    ax = fig.add_subplot(gs[0, 0])
    _panel_title(ax, "Complexity level", "all samples; scene realism L0 -> L3")
    lv = [k for k in LEVELS if k in s["levels"]] + sorted(
        k for k in s["levels"] if k not in LEVELS)
    _donut(ax, lv, [s["levels"][k] for k in lv],
           [ORDINAL4[LEVELS.index(k)] if k in LEVELS else NEUTRAL for k in lv],
           "%d" % s["samples"], "samples", legend_cols=4)

    ax = fig.add_subplot(gs[0, 1])
    declared = s.get("declared_condition_shares") or {}
    known = list(CONDITIONS) + sorted(set(declared) - set(CONDITIONS))
    conds = [c for c in known if c in s["conditions"]] + sorted(
        c for c in s["conditions"] if c not in known)
    total = max(1, sum(s["conditions"].values()))
    _panel_title(ax, "Difficulty condition",
                 "all samples; the legend compares with the declared share")
    _donut(ax, conds, [s["conditions"][c] for c in conds],
           [CATEGORICAL[known.index(c) % len(CATEGORICAL)] if c in known
            else NEUTRAL for c in conds],
           "%d" % total, "samples",
           legend_labels=["%s %d%% (declared %d%%)"
                          % (c, round(100 * s["conditions"][c] / total),
                             round(100 * declared.get(c, 0))) for c in conds],
           legend_cols=2)

    ax = fig.add_subplot(gs[0, 2])
    _panel_title(ax, "Severity bin", "violated samples; requested magnitude")
    bins = s.get("severity_bins") or {}
    names = [b for b in BINS if b in bins] + sorted(b for b in bins if b not in BINS)
    _donut(ax, names, [bins[b] for b in names],
           [ORDINAL3[BINS.index(b)] if b in BINS else NEUTRAL for b in names],
           "%d" % s["invalid"], "violated", legend_cols=3)

    ax = fig.add_subplot(gs[0, 3])
    _panel_title(ax, "Violator timing",
                 "violated clips: one moment, forced shared, or one per violator")
    timing = s.get("violator_timing") or {}
    names = [k for k in ("shared", "sync", "independent") if k in timing]
    names += sorted(k for k in timing if k not in names)
    colours = {"shared": NEUTRAL, "sync": CATEGORICAL[5], "independent": CATEGORICAL[6]}
    _donut(ax, names, [timing[k] for k in names],
           [colours.get(k, MUTED) for k in names],
           "%d" % s["invalid"], "violated", legend_cols=3)
    fig.savefig(path, dpi=170)
    plt.close(fig)


#: The narrowest a sunburst slice is drawn, as a share of the circle: wide
#: enough for a one-word label written across the ring. Measured against
#: `_ring_labels` at the figure's size -- a bold domain name wants about 12
#: degrees on the inner ring, a two-line family name about 11 on the outer.
DOMAIN_MIN_SHARE = 0.036
FAMILY_MIN_SHARE = 0.036


def _fig_taxonomy(plt, s, path) -> None:
    """Domain -> family sunburst, and clips per scenario grouped by medium."""
    import numpy as np
    from .. import taxonomy as T

    fig = plt.figure(figsize=(16, 7.6))
    _heading(fig, "Taxonomy",
             "which physics is violated, and in which scenes")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], left=0.01,
                          right=0.97, top=0.84, bottom=0.1, wspace=0.14)

    ax = fig.add_subplot(gs[0, 0])
    _panel_title(ax, "Domain and family",
                 "violated clips: domain inside, families outside; thin slices drawn wider, counts exact")
    fams = s.get("families") or {}
    fd = s.get("family_domain") or {}
    order = [d for d in T.DOMAINS if any(fd.get(f) == d for f in fams)]
    order += sorted({fd.get(f, "?") for f in fams} - set(order))
    dom_val = [sum(v for f, v in fams.items() if fd.get(f) == d) for d in order]
    if sum(dom_val):
        dom_col = [_readable_fill(CATEGORICAL[i % len(CATEGORICAL)])
                   for i in range(len(order))]
        # EVERY LABEL INSIDE ITS RING. A slice is drawn at least a minimum
        # share of the circle -- families first, then a domain whose families
        # still add up to less is widened as a whole -- so `optical` at 1.7%
        # of the clips carries its own name the way `equilibrium` does, rather
        # than hanging off a leader line. The printed counts stay exact; only
        # the angle of a thin slice is generous.
        total = float(sum(dom_val))
        fam_names, fam_vals, fam_cols, fam_draw = [], [], [], []
        dom_draw = []
        for d, colour in zip(order, dom_col):
            members = sorted((f for f in fams if fd.get(f) == d),
                             key=lambda f: -fams[f])
            draw = [max(float(fams[f]), FAMILY_MIN_SHARE * total)
                    for f in members]
            grow = max(1.0, DOMAIN_MIN_SHARE * total / max(sum(draw), 1e-9))
            draw = [v * grow for v in draw]
            dom_draw.append(sum(draw))
            for k, (f, v) in enumerate(zip(members, draw)):
                fam_names.append(f)
                fam_vals.append(fams[f])
                fam_draw.append(v)
                fam_cols.append(_readable_fill(
                    _tint(colour, 0.1 + 0.45 * k / max(len(members), 1))))
        ax.set_aspect("equal")
        ax.axis("off")
        inner, _ = ax.pie(dom_draw, radius=0.62, colors=dom_col, startangle=90,
                          counterclock=False,
                          wedgeprops={"width": 0.38, "edgecolor": SURFACE,
                                      "linewidth": 2.0})
        outer, _ = ax.pie(fam_draw, radius=1.0, colors=fam_cols, startangle=90,
                          counterclock=False,
                          wedgeprops={"width": 0.36, "edgecolor": SURFACE,
                                      "linewidth": 1.5})
        # The ring fills the panel's height: the space a leader line needed
        # is better spent making every slice wide enough to be written in.
        # After `pie`, which sets limits of its own.
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.04, 1.04)
        ax.text(0, 0.05, "%d" % s["invalid"], ha="center", va="center",
                fontsize=14, fontweight="bold")
        ax.text(0, -0.12, "violated", ha="center", va="center", fontsize=7.5,
                color=INK2)
        inner_missed = _ring_labels(ax, inner, [(d, d[:5] + ".") for d in order],
                                    0.24, 0.62, dom_col, fontsize=7, weight="bold",
                                    text_colour="white")
        pretty = [f.replace("_", " ") for f in fam_names]
        missed = _ring_labels(
            ax, outer, [("%s\n%d" % (p, v), p, p.replace(" ", "\n"))
                        for p, v in zip(pretty, fam_vals)],
            0.64, 1.0, fam_cols, fontsize=6.8, text_colour="white")
        # Both rings' leftovers in ONE call, so the leaders de-overlap against
        # each other rather than landing on top of one another.
        out_w = [outer[i] for i in missed] + [inner[i] for i in inner_missed]
        out_t = ["%s (%d)" % (pretty[i], fam_vals[i]) for i in missed] + \
                ["%s (%d)" % (order[i], dom_val[i]) for i in inner_missed]
        if out_w:
            _slice_labels(ax, out_w, out_t, 1.0)
        ax.legend([_patch(c) for c in dom_col],
                  ["%s (%d)" % (d, v) for d, v in zip(order, dom_val)],
                  loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4,
                  fontsize=7.5, handlelength=0.9, handleheight=0.9,
                  columnspacing=1.0, labelcolor=INK2)
    else:
        _empty(ax, "no violated clips")

    ax = fig.add_subplot(gs[0, 1])
    _panel_title(ax, "Clips per scenario",
                 "violated clips and their lawful twins, grouped by physics medium")
    scen = s.get("scenarios") or {}
    medium = s.get("scenario_medium") or {}
    cells = s.get("cells") or {}
    media = [m for m in T.MEDIA if m in set(medium.values())]
    media += sorted(set(medium.values()) - set(media))
    rows: List[Tuple[str, Optional[str]]] = []
    for m in media:
        members = sorted((k for k in scen if medium.get(k) == m),
                         key=lambda k: -scen[k])
        if members:
            rows.append((m, None))
            rows += [(k, m) for k in members]
    if rows:
        y = np.arange(len(rows))[::-1]
        top = max(scen.values())
        labels = []
        for yi, (name, med) in zip(y, rows):
            if med is None:
                labels.append(name.upper())
                continue
            bad = sum((cells.get(name) or {}).values())
            good = max(0, scen[name] - bad)
            ax.barh(yi, bad, height=0.66, color=ACCENT, edgecolor=SURFACE,
                    linewidth=1.5)
            ax.barh(yi, good, left=bad, height=0.66, color=NEUTRAL,
                    edgecolor=SURFACE, linewidth=1.5)
            if bad / top > 0.08:
                ax.text(bad / 2, yi, "%d" % bad, ha="center", va="center",
                        fontsize=7, color="white")
            ax.text(scen[name] + top * 0.01, yi, "%d" % scen[name],
                    va="center", fontsize=7.5, color=INK2)
            labels.append(name.replace("_", " "))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        for tick, (_name, med) in zip(ax.get_yticklabels(), rows):
            if med is None:
                tick.set_fontweight("bold")
                tick.set_color(MUTED)
                tick.set_fontsize(7.2)
        ax.set_xlim(0, top * 1.1)
        _quiet(ax, "x")
        ax.spines["left"].set_visible(False)
        ax.legend([_patch(ACCENT), _patch(NEUTRAL)], ["violated", "lawful twin"],
                  loc="lower right", fontsize=7.5, handlelength=0.9,
                  labelcolor=INK2)
    else:
        _empty(ax, "no clips")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _fig_coverage(plt, s, path) -> None:
    """The scenario x family lattice, families grouped under a NAMED domain band."""
    import matplotlib.patches as mpatches
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from .. import taxonomy as T

    scen = s.get("scenarios") or {}
    medium = s.get("scenario_medium") or {}
    media = [m for m in T.MEDIA if m in set(medium.values())]
    rows = sorted(scen, key=lambda k: (media.index(medium.get(k))
                                       if medium.get(k) in media else 99, k))
    built = {(a, b) for a, b in (s.get("built_cells") or [])}
    doms = list(T.DOMAINS)

    def dom_of(f):
        return getattr(T.FAMILIES.get(f), "domain", None) or \
            (s.get("family_domain") or {}).get(f, "?")

    cols = sorted({b for _, b in built} | set(s.get("family_domain") or {}),
                  key=lambda f: (doms.index(dom_of(f)) if dom_of(f) in doms else 99, f))

    fig = plt.figure(figsize=(16, 8.2))
    _heading(fig, "Coverage",
             "violated clips per scenario x family cell; %d cells are buildable "
             "in this release" % len(built))
    ax = fig.add_axes([0.13, 0.2, 0.8, 0.62])
    cells = s.get("cells") or {}
    if not (rows and cols):
        _empty(ax, "no cells")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        return
    counts = np.array([[cells.get(r, {}).get(c, 0) for c in cols] for r in rows], float)
    top = max(1.0, counts.max())
    rgb = np.ones(counts.shape + (3,))
    cmap = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#0d366b"])
    missing = 0
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if counts[i, j] > 0:
                rgb[i, j] = cmap(0.08 + 0.92 * counts[i, j] / top)[:3]
            elif (r, c) in built:
                rgb[i, j] = to_rgb("#f2b8b5")
                missing += 1
            else:
                rgb[i, j] = to_rgb("#f3f2ef")
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if counts[i, j] > 0:
                fill = "#%02x%02x%02x" % tuple(int(255 * x) for x in rgb[i, j])
                ax.text(j, i, "%d" % counts[i, j], ha="center", va="center",
                        fontsize=6.5, color=_ink_on(fill))
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c.replace("_", " ") for c in cols], rotation=50,
                       ha="right", fontsize=7.8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(["%s  (%s)" % (r.replace("_", " "), medium.get(r, "?"))
                        for r in rows], fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.6)
    ax.tick_params(which="both", length=0, colors=INK2)
    for side in ax.spines.values():
        side.set_visible(False)

    # THE DOMAIN BAND, named. Families are sorted by domain, so each domain is
    # one contiguous run of columns and gets one labelled block over it.
    runs: List[List] = []
    for j, c in enumerate(cols):
        d = dom_of(c)
        if runs and runs[-1][0] == d:
            runs[-1][2] = j
        else:
            runs.append([d, j, j])
    renderer = fig.canvas.get_renderer()
    col_pts = (ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]) \
        * 72.0 / fig.dpi
    for d, a, b in runs:
        colour = CATEGORICAL[doms.index(d) % len(CATEGORICAL)] if d in doms else MUTED
        ax.add_patch(mpatches.FancyBboxPatch(
            (a - 0.46, -1.95), b - a + 0.92, 0.72, clip_on=False, linewidth=0,
            boxstyle="round,pad=0,rounding_size=0.12", color=colour, zorder=5))
        label = d
        t = ax.text((a + b) / 2, -1.59, label, ha="center", va="center",
                    fontsize=7.6, fontweight="bold", color=_ink_on(colour),
                    clip_on=False, zorder=6)
        width = t.get_window_extent(renderer=renderer).width * 72.0 / fig.dpi
        if width > (b - a + 0.9) * col_pts:
            t.set_rotation(0)
            t.set_fontsize(6.4)
            t.set_text(d[:4] + ".")
    # Column totals under the band, row totals to the right.
    for j in range(len(cols)):
        ax.text(j, -0.82, "%d" % counts[:, j].sum(), ha="center", va="center",
                fontsize=6.8, color=INK2, clip_on=False)
    for i in range(len(rows)):
        ax.text(len(cols) - 0.3, i, "%d" % counts[i].sum(), ha="left",
                va="center", fontsize=7.2, color=INK2, clip_on=False)
    ax.text(len(cols) - 0.3, -0.82, "total", ha="left", va="center",
            fontsize=6.8, color=MUTED, clip_on=False)
    ax.set_ylim(len(rows) - 0.5, -2.0)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    fig.legend([_patch(cmap(0.6)), _patch("#f2b8b5"), _patch("#f3f2ef")],
               ["has clips (darker = more)",
                "buildable but missing (%d)" % missing,
                "not a meaningful cell"],
               loc="lower left", bbox_to_anchor=(0.012, 0.015), ncol=3,
               fontsize=8, handlelength=1.1, handleheight=1.1, labelcolor=INK2)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _zone_counts(f, values) -> List[int]:
    out = [0, 0, 0]
    for v in values:
        out[f.level(float(v))] += 1
    return out


def _rule(f) -> Tuple[str, str, str]:
    """The easy / moderate / hard conditions of one factor, as a reader says them."""
    e, m = "%g" % f.easy, "%g" % f.moderate
    if f.easier == "high":
        return (">= %s" % e, "%s to %s" % (m, e), "< %s" % m)
    return ("<= %s" % e, "%s to %s" % (e, m), "> %s" % m)


def _fig_difficulty(plt, s, path) -> None:
    """The label, how it is decided, and what set it."""
    import matplotlib.patches as mpatches
    import numpy as np

    fig = plt.figure(figsize=(16.5, 11.2))
    _heading(fig, "Detection difficulty",
             "every violated clip gets ONE label: the worst zone it reaches over "
             "seven factors (KITTI's easy / moderate / hard rule)")
    gs = fig.add_gridspec(2, 3, width_ratios=[0.9, 1.05, 1.05],
                          height_ratios=[1.0, 1.05], left=0.02, right=0.97,
                          top=0.86, bottom=0.04, wspace=0.3, hspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    _panel_title(ax, "Difficulty label", "violated clips")
    _donut(ax, list(D.LEVELS), [s["difficulty"].get(k, 0) for k in D.LEVELS],
           list(ORDINAL3), "%d" % s["invalid"], "violated", legend_cols=3)

    ax = fig.add_subplot(gs[0, 1])
    _panel_title(ax, "Label by complexity level",
                 "share of violated clips per level; the two axes are measured apart")
    grid = s.get("difficulty_by_level") or {}
    rows = [k for k in LEVELS if k in grid] + sorted(k for k in grid if k not in LEVELS)
    if rows:
        _stacked_share(ax, rows, [[grid[r].get(n, 0) for n in D.LEVELS] for r in rows],
                       list(D.LEVELS), list(ORDINAL3))
    else:
        _empty(ax, "no violated clips")

    ax = fig.add_subplot(gs[0, 2])
    _panel_title(ax, "Which factor set the label",
                 "share of violated clips whose worst zone came from this factor;\n"
                 "ties count for every factor involved, so bars sum past 100%")
    bind = s.get("binding_factors") or {}
    order = sorted(bind, key=lambda k: -bind[k])
    if order:
        total = max(1, s["invalid"])
        vals = [bind[k] / total for k in order]
        y = np.arange(len(order))[::-1]
        ax.barh(y, vals, height=0.6, color=ACCENT)
        for yi, v, k in zip(y, vals, order):
            ax.text(v + 0.012, yi, "%d%%  (%d)" % (round(100 * v), bind[k]),
                    va="center", fontsize=7.8, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_xlim(0, min(1.2, max(vals) * 1.3 + 0.05))
        ax.xaxis.set_major_formatter(
            __import__("matplotlib").ticker.PercentFormatter(1.0, decimals=0))
        _quiet(ax, "x")
        ax.spines["left"].set_visible(False)
    else:
        _empty(ax, "no labels")

    # THE RULE TABLE: the answer to "how is a label decided", with the cuts.
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _panel_title(ax, "How a label is decided",
                 "each factor puts a clip in a zone by two cuts; the clip's label "
                 "is its WORST zone -- easy only if every factor is easy, hard if "
                 "any one is hard. An unmeasured factor counts as moderate.")
    fs = list(D.FACTORS)
    cols = [("factor", 0.0), ("what it asks", 0.12), ("cuts", 0.42),
            ("easy", 0.49), ("moderate", 0.6), ("hard", 0.71),
            ("zones over this release", 0.83)]
    row_h = 1.0 / (len(fs) + 1.4)
    head_y = 1.0 - row_h * 0.5
    for name, x in cols:
        colour = {"easy": ORDINAL3[0], "moderate": ORDINAL3[1],
                  "hard": ORDINAL3[2]}.get(name)
        ax.text(x + (0.05 if colour else 0.0), head_y, name,
                ha="center" if colour else "left", va="center", fontsize=8.5,
                fontweight="bold", color=colour or INK2)
    ax.plot([0, 1], [1.0 - row_h, 1.0 - row_h], color=GRID, linewidth=1.0)
    for i, f in enumerate(fs):
        yc = 1.0 - row_h * (i + 1.5)
        if i % 2 == 0:
            ax.add_patch(mpatches.Rectangle((0, yc - row_h / 2), 1.0, row_h,
                                            color="#f4f3ef", linewidth=0, zorder=0))
        ax.text(0.0 + 0.004, yc, f.name, va="center", fontsize=9, fontweight="bold")
        ax.text(0.12, yc + row_h * 0.16, f.question, va="center", fontsize=8,
                color=INK)
        ax.text(0.12, yc - row_h * 0.2, "%s  |  %s is easier" % (
            f.unit, "higher" if f.easier == "high" else "lower"),
            va="center", fontsize=7.2, color=INK2)
        ax.text(0.42, yc, f.basis or "-", va="center", fontsize=7.8, color=INK2,
                style="italic")
        for k, text in enumerate(_rule(f)):
            x = cols[3 + k][1]
            ax.add_patch(mpatches.FancyBboxPatch(
                (x + 0.004, yc - row_h * 0.3), 0.092, row_h * 0.6,
                boxstyle="round,pad=0,rounding_size=0.006",
                color=_tint(ORDINAL3[k], 0.72 - 0.12 * k), linewidth=0))
            ax.text(x + 0.05, yc, text, ha="center", va="center", fontsize=8.5,
                    color=INK, family="DejaVu Sans Mono")
        vals = (s.get("factor_values") or {}).get(f.name) or []
        zc = _zone_counts(f, vals)
        n = max(1, sum(zc))
        left = cols[6][1]
        width = 0.16
        for k in range(3):
            w = width * zc[k] / n
            if w <= 0:
                continue
            ax.add_patch(mpatches.Rectangle((left, yc - row_h * 0.28), w,
                                            row_h * 0.56, color=ORDINAL3[k],
                                            linewidth=0))
            if zc[k] / n >= 0.12:
                ax.text(left + w / 2, yc, "%d%%" % round(100 * zc[k] / n),
                        ha="center", va="center", fontsize=7.2,
                        color=_ink_on(ORDINAL3[k]))
            left += w
        if not vals:
            ax.text(cols[6][1], yc, "not measured", va="center", fontsize=7.5,
                    color=MUTED)
    ax.text(0.0, -0.02, "cuts: fitted = searched so the whole release lands near "
            "30% easy / 40% moderate / 30% hard; chosen = argued from what the "
            "sampler draws (how many objects a crowded clip holds, how many bodies "
            "a multi clip violates, what counts as a moving camera). Both are "
            "frozen once published.",
            va="top", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _fig_factors(plt, s, path) -> None:
    """Each factor's real values against its three zones."""
    fs = list(D.FACTORS)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.4))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.8, bottom=0.07,
                        wspace=0.22, hspace=0.62)
    _heading(fig, "Difficulty factors",
             "how many clips hold each value, against the factor's easy / "
             "moderate / hard zones (the cuts are printed in 4_difficulty)")
    for ax, f in zip(axes.ravel(), fs):
        _factor_panel(ax, f, s["factor_values"].get(f.name) or [])
    guide = axes.ravel()[len(fs)]
    guide.axis("off")
    guide.legend([_patch(_tint(c, 0.7)) for c in ORDINAL3],
                 ["easy zone", "moderate zone", "hard zone"],
                 loc="upper left", fontsize=9, handlelength=1.3, handleheight=1.3,
                 labelcolor=INK2, title="how to read", title_fontsize=9.5,
                 alignment="left")
    guide.text(0.02, 0.5,
               "bars    how many clips fall in that\n"
               "         range (whole-number factors\n"
               "         get one bar per value)\n"
               "curve   the same clips, smoothed\n"
               "line    a cut, its value in the box\n"
               "dashes  the median\n\n"
               "violation_area: the fraction of the\n"
               "frame the violation covers at its peak",
               transform=guide.transAxes, fontsize=8, color=INK2, va="top",
               family="DejaVu Sans Mono")
    for ax in axes.ravel()[len(fs) + 1:]:
        ax.axis("off")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _factor_panel(ax, f, values) -> None:
    """One factor: how many clips hold each value, against its three zones."""
    import matplotlib.transforms as mtransforms
    import numpy as np

    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]
    zc = _zone_counts(f, vals)
    n = max(1, int(vals.size))
    line = ("   ".join("%s %d%%" % (D.LEVELS[k], round(100 * zc[k] / n))
                       for k in range(3))
            if vals.size else "not measured in this release")
    _panel_title(ax, f.name, "%s, %s is easier   (n=%d)\n%s" % (
        f.unit, "higher" if f.easier == "high" else "lower", vals.size, line))

    cuts = sorted([f.easy, f.moderate])
    bounded01 = "0-1" in f.unit or f.unit.startswith("fraction")
    counts = f.unit == "count"
    if vals.size:
        lo = min(0.0, float(vals.min()))
        # A long tail is cut at the 95th percentile, so a handful of `pour`
        # clips with twenty violators do not squeeze everything else into a
        # sliver. What is past the axis is reported rather than dropped.
        hi = max(float(np.percentile(vals, 95)), cuts[1] * 1.6, cuts[1] + 1e-3)
    else:
        lo, hi = 0.0, cuts[1] * 1.6 + 1e-3
    if counts:
        lo, hi = lo - 0.6, hi + 0.6
    if bounded01:
        hi = min(max(hi, cuts[1] * 1.2), 1.0)

    if f.easier == "high":
        zones = ((f.easy, hi, 0), (f.moderate, f.easy, 1), (lo, f.moderate, 2))
    else:
        zones = ((lo, f.easy, 0), (f.easy, f.moderate, 1), (f.moderate, hi, 2))
    band = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    span_pts = ax.get_window_extent().width * 72.0 / ax.figure.dpi
    for a, b, k in zones:
        a, b = max(a, lo), min(b, hi)
        if b <= a:
            continue
        ax.axvspan(a, b, color=_tint(ORDINAL3[k], 0.8), linewidth=0, zorder=0)
        if (b - a) / (hi - lo) * span_pts > 7 * len(D.LEVELS[k]):
            ax.text((a + b) / 2, 0.975, D.LEVELS[k], transform=band, ha="center",
                    va="top", fontsize=7.2, fontweight="bold",
                    color=_tint(ORDINAL3[2], 0.1 if k == 2 else 0.3), zorder=6)

    top, med, over = 0.0, None, 0
    if vals.size and counts:
        ks, cs = np.unique(vals, return_counts=True)
        keep = (ks >= lo) & (ks <= hi)
        ax.bar(ks[keep], cs[keep], width=0.62, color=ACCENT, zorder=2)
        top = float(cs[keep].max()) if keep.any() else 0.0
        for k, c in zip(ks[keep], cs[keep]):
            ax.text(k, c + top * 0.02, "%d" % c, ha="center", va="bottom",
                    fontsize=7, color=INK, zorder=6)
        med, over = float(np.median(vals)), int((vals > hi).sum())
    elif vals.size:
        top, med, over = _hist(ax, vals, lo, hi)
    ax.set_ylabel("clips", fontsize=7.5)
    ax.set_ylim(0, (top or 1.0) * 1.42)

    for c in cuts:
        if lo < c < hi:
            ax.axvline(c, color=INK2, linewidth=0.9, zorder=3)
            ax.text(c, 0.84, "%g" % c, transform=band, ha="center", va="center",
                    fontsize=6.8, color=INK, zorder=7,
                    bbox={"boxstyle": "round,pad=0.18", "fc": SURFACE,
                          "ec": GRID, "lw": 0.6})
    _median_line(ax, med, "%g" if counts else "%.3g", y_axes=0.62)
    _beyond(ax, over, 0.44)
    ax.set_xlim(lo, hi)
    _quiet(ax, "y")
    ax.tick_params(labelsize=7.5)


def _fig_timing(plt, s, path) -> None:
    """Event moment, observability lag, and measured severity by bin."""
    import numpy as np

    from ..injectors._geom import EVENT_BAND

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6))
    fig.subplots_adjust(left=0.045, right=0.985, top=0.72, bottom=0.12, wspace=0.18)
    _heading(fig, "Timing and severity",
             "how many violated clips fall in each range; the dashed rule is "
             "the median")

    ax = axes[0]
    _panel_title(ax, "When violations fire",
                 "event frame as a share of the clip; grey = the band a moment\n"
                 "is drawn from when the physics does not set it")
    vals = s.get("event_time_share") or []
    ax.axvspan(EVENT_BAND[0], EVENT_BAND[1], color=_tint(NEUTRAL, 0.55),
               linewidth=0, zorder=0)
    if vals:
        top, med, _over = _hist(ax, vals, 0.0, 1.0)
        ax.set_ylim(0, top * 1.35)
        _median_line(ax, med, "%.2f", y_axes=0.78)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(
        __import__("matplotlib").ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("clips", fontsize=8)
    _quiet(ax, "y")

    ax = axes[1]
    vals = s.get("observability_lag_seconds") or []
    at_once = (100.0 * sum(1 for v in vals if v <= 0) / len(vals)) if vals else 0
    _panel_title(ax, "Observability lag",
                 "seconds from the event until a viewer could tell;\n"
                 "%d%% are visible on the event frame itself" % round(at_once))
    if vals:
        hi = max(float(np.percentile(vals, 98)) * 1.2, 0.25)
        top, med, over = _hist(ax, vals, 0.0, hi)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, top * 1.35)
        _median_line(ax, med, "%.2f s", y_axes=0.78)
        _beyond(ax, over)
    else:
        _empty(ax, "no events")
    ax.set_ylabel("clips", fontsize=8)
    _quiet(ax, "y")

    ax = axes[2]
    _panel_title(ax, "Measured severity by bin",
                 "peak bounded residual (0 = lawful, 1 = saturated); the ladder\n"
                 "should climb weak -> strong. Each bin as a share of ITS clips.")
    pk = s.get("peak_score_by_bin") or {}
    present = [b for b in BINS if len(pk.get(b) or []) >= 2]
    if present:
        top = 0.0
        for b in present:
            v = np.asarray(pk[b], float)
            t, _med, _over = _hist(ax, v, 0.0, 1.0, colour=ORDINAL3[BINS.index(b)],
                                   bins=20, share=True, step=True,
                                   label="%s  n=%d, median %.2f"
                                         % (b, v.size, float(np.median(v))))
            top = max(top, t)
        ax.legend(loc="upper left", fontsize=7.8, labelcolor=INK2, handlelength=1.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, top * 1.45)
        ax.yaxis.set_major_formatter(
            __import__("matplotlib").ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("share of that bin's clips", fontsize=8)
        _quiet(ax, "y")
    else:
        _empty(ax, "fewer than two clips per bin")
    fig.savefig(path, dpi=170)
    plt.close(fig)


_DRAW = {"1_overview.png": _fig_overview,
         "2_taxonomy.png": _fig_taxonomy,
         "3_coverage.png": _fig_coverage,
         "4_difficulty.png": _fig_difficulty,
         "5_difficulty_factors.png": _fig_factors,
         "6_timing_and_severity.png": _fig_timing}
#: Figures earlier reports wrote under other names. A re-run removes them, or
#: an old composition.png sits beside the new overview and nobody knows which
#: one is current.
STALE = ("composition.png", "coverage.png", "difficulty.png",
         "difficulty_factors.png", "distributions.png", "structure.png",
         "severity.png", "timing.png")


def draw(s: Dict[str, object], out: str) -> None:
    """Every figure from a summary -- `stats.json` is enough to redraw them."""
    import matplotlib
    matplotlib.use("Agg")                       # headless box, no display
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    with plt.rc_context(_rc()):
        for name, _caption in FIGURES:
            _DRAW[name](plt, s, os.path.join(out, name))
    for stale in STALE:
        path = os.path.join(out, stale)
        if os.path.exists(path):
            os.remove(path)


def report(root: str, outdir: Optional[str] = None,
           metas: Optional[List[Dict[str, object]]] = None) -> Dict[str, object]:
    """Write `FIGURES` and `stats.json`; return the summary.

    `metas` lets a caller pass clips from several roots at once.
    """
    metas = load(root) if metas is None else metas
    if not metas:
        raise SystemExit("no clips under %s" % root)
    s = summarise(metas)
    out = outdir or os.path.join(root, "stats")
    draw(s, out)
    with open(os.path.join(out, "stats.json"), "w") as fh:
        json.dump(s, fh, indent=2, sort_keys=True)
    s["outdir"] = out
    return s

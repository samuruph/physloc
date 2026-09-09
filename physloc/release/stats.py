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
shipped `meta.json` files and nothing else.

    python -m physloc.cli stats out/physloc_v0

Writes `<root>/stats/`: five figures and `stats.json`. The JSON is the numbers
behind the figures -- so a regression can be diffed rather than squinted at,
and so the dataset card can quote them without re-deriving anything.

**Reads `meta.json` and nothing else** on a current release -- no arrays, no
renders -- so it runs in seconds over a full release and can be re-run after
any annotation change. The one exception is a clip generated before the
difficulty label existed: `load` back-fills it, and reads that clip's masks to
do so.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from ..annotate import difficulty as D

#: Muted, colour-blind safe, and ordered so that "worse" reads darker. Three
#: hues do the whole report -- a figure that needs a legend of twelve colours
#: is a figure nobody reads.
INK = "#1b1f23"
MUTED = "#6a737d"
GRID = "#e6e8ea"
EASY, MODERATE, HARD = "#7fb3d5", "#e8a33d", "#c0504d"
LEVEL_COLOUR = {"easy": EASY, "moderate": MODERATE, "hard": HARD}
ACCENT = "#4c78a8"


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """One look for every panel: no box, no chartjunk, a light y-grid only.

    The grid sits UNDER the data (`set_axisbelow`) -- over it, a gridline
    crossing a bar reads as a division in the bar.
    """
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8)


def _categories(ax, n: int, least: int = 3, first: float = 0.0) -> None:
    """Keep a bar looking like a bar when there are only one or two of them.

    With a single category matplotlib's x-range is about [-0.5, 0.5] and a
    0.62-wide bar fills the whole panel, which reads as a filled axes rather
    than as a count. Pad the limits out to `least` slots and centre what there
    is -- a single-level run (`review_L0`, `v0_L2`) is a normal thing to plot.
    """
    pad = max(0.0, (least - n) / 2.0)
    # `first` is where the first category sits: bars start at 0, boxplots at 1.
    ax.set_xlim(first - 0.5 - pad, first + n - 0.5 + pad)


def _headroom(ax, frac: float = 0.18) -> None:
    """Space above the tallest bar, for its value label and any legend."""
    top = ax.get_ylim()[1]
    ax.set_ylim(ax.get_ylim()[0], top * (1.0 + frac) if top > 0 else 1.0)


def _bar_labels(ax, bars, fmt="%d") -> None:
    """The number on the bar. A reader should not have to measure against an
    axis to read a count."""
    for b in bars:
        h = b.get_height()
        if h <= 0:
            continue
        ax.annotate(fmt % h, (b.get_x() + b.get_width() / 2.0, h),
                    ha="center", va="bottom", fontsize=7.5, color=MUTED,
                    xytext=(0, 2), textcoords="offset points")


def _array(path: str):
    """One array out of an `.npz`, or None. The files here hold exactly one."""
    if not os.path.exists(path):
        return None
    import numpy as np

    with np.load(path) as z:
        keys = list(z.keys())
        return z[keys[0]] if keys else None


def load(root: str) -> List[Dict[str, object]]:
    """Every clip's `meta.json` under a release root.

    BACK-FILLS `difficulty` when a clip predates it, so this works on runs
    generated before the label existed -- of which there are several on disk,
    and they are the corpus the thresholds were fitted to. The masks are read
    only in that case, and only when they are sitting beside the `meta.json`;
    a current release carries the label already and this touches no arrays.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(root, "clips", "**", "meta.json"),
                                 recursive=True)):
        with open(path) as fh:
            meta = json.load(fh)
        if meta.get("violation") and not meta.get("difficulty"):
            cdir = os.path.dirname(path)
            meta["difficulty"] = D.assess(
                meta,
                _array(os.path.join(cdir, "violation_mask.npz")),
                _array(os.path.join(cdir, "seg.npz")))
        out.append(meta)
    return out


def summarise(metas: List[Dict[str, object]]) -> Dict[str, object]:
    """The numbers behind every figure, and the file a regression diffs."""
    invalid = [m for m in metas if m.get("violation")]
    valid = [m for m in metas if not m.get("violation")]

    levels = Counter(str((m.get("complexity") or {}).get("name") or "?")
                     for m in metas)
    conditions = Counter(str(m.get("condition") or "?") for m in metas)
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
        lv = str((m.get("complexity") or {}).get("name") or "?")
        grid[lv][str((m.get("difficulty") or {}).get("level") or "unlabelled")] += 1

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

    return {
        "clips": len(metas), "invalid": len(invalid), "valid": len(valid),
        "levels": dict(levels), "conditions": dict(conditions),
        "severity_bins": dict(bins), "difficulty": dict(diff),
        "binding_factors": dict(binding),
        "difficulty_by_level": {k: dict(v) for k, v in grid.items()},
        "factor_values": factors,
        "peak_score_by_bin": {k: v for k, v in severity.items()},
        "families": dict(Counter(str(m.get("family")) for m in invalid)),
        "scenarios": dict(Counter(str(m.get("scenario")) for m in metas)),
        "thresholds": {f.name: {"easier": f.easier, "easy": f.easy,
                                "moderate": f.moderate} for f in D.FACTORS},
    }


# ------------------------------------------------------------------- figures
def _fig_difficulty(plt, s, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    counts = [s["difficulty"].get(lv, 0) for lv in D.LEVELS]
    total = max(1, sum(counts))
    bars = axes[0].bar(list(D.LEVELS), counts,
                       color=[LEVEL_COLOUR[lv] for lv in D.LEVELS], width=0.62)
    _bar_labels(axes[0], bars)
    _style(axes[0], "Detection difficulty — the worst factor wins", "",
           "clips")
    # `set_xticks` before `set_xticklabels`, or matplotlib warns that the
    # labels may be attached to ticks a locator is free to move.
    axes[0].set_xticks(range(len(D.LEVELS)))
    axes[0].set_xticklabels(["%s\n%.0f%%" % (lv, 100.0 * c / total)
                             for lv, c in zip(D.LEVELS, counts)])
    _headroom(axes[0])

    order = sorted(s["binding_factors"], key=lambda k: -s["binding_factors"][k])
    bars = axes[1].bar(order, [s["binding_factors"][k] for k in order],
                       color=ACCENT, width=0.62)
    _bar_labels(axes[1], bars)
    _style(axes[1], "Which factor set the label", "", "clips")
    _headroom(axes[1])
    axes[1].tick_params(axis="x", labelrotation=35)
    for lab in axes[1].get_xticklabels():
        lab.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fig_factors(plt, s, path) -> None:
    """Each factor's distribution with its two thresholds drawn on it.

    The one figure that shows a threshold and the data it was fitted to in the
    same frame -- which is the only way to see that a cut is doing something.
    """
    import numpy as np

    fs = list(D.FACTORS)
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for ax, f in zip(axes.ravel(), fs):
        vals = np.asarray(s["factor_values"].get(f.name) or [], float)
        if vals.size:
            ax.hist(vals, bins=24, color=ACCENT, alpha=0.85,
                    edgecolor="white", linewidth=0.4)
        # Labels INSIDE the axes and on two rows: at 1.0 they collided with the
        # panel title, and on `footprint` the two cuts are close enough in data
        # units that one row put "easy" on top of "moderate".
        for cut, colour, label, y in ((f.easy, EASY, "easy", 0.97),
                                      (f.moderate, MODERATE, "moderate", 0.85)):
            ax.axvline(cut, color=colour, linewidth=1.6, zorder=5)
            ax.annotate(label, (cut, y), xycoords=("data", "axes fraction"),
                        color=colour, fontsize=7, ha="center", va="top",
                        bbox={"facecolor": "white", "edgecolor": "none",
                              "pad": 0.8, "alpha": 0.85})
        arrow = "→ easier" if f.easier == "high" else "← easier"
        _style(ax, f.name, "%s  (%s)" % (f.unit, arrow), "clips")
    for ax in axes.ravel()[len(fs):]:
        ax.axis("off")
    fig.suptitle("Difficulty factors: the distribution, and where the cuts fall",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fig_composition(plt, s, path, shares) -> None:
    """Did the release come out in the proportions it declares?"""
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    lv = [k for k in ("L0", "L1", "L2", "L3") if k in s["levels"]]
    bars = axes[0].bar(lv, [s["levels"][k] for k in lv], color=ACCENT,
                       width=0.62)
    _bar_labels(axes[0], bars)
    _style(axes[0], "Complexity level", "", "clips")
    _categories(axes[0], len(lv))
    _headroom(axes[0])

    # Conditions, against the shares `CONDITION_CYCLE` declares. Counts alone
    # cannot say whether a run is skewed; the declared line is the answer.
    names = [k for k in shares if k in s["conditions"]] or list(s["conditions"])
    got = np.asarray([s["conditions"].get(k, 0) for k in names], float)
    tot = max(1.0, got.sum())
    bars = axes[1].bar(names, 100.0 * got / tot, color=ACCENT, width=0.62)
    _bar_labels(axes[1], bars, "%.0f%%")
    for i, k in enumerate(names):
        want = 100.0 * shares.get(k, 0.0)
        axes[1].plot([i - 0.31, i + 0.31], [want, want], color=HARD,
                     linewidth=1.8, zorder=5,
                     label="declared" if i == 0 else None)
    _style(axes[1], "Difficulty condition — bars measured, line declared", "",
           "% of clips")
    axes[1].tick_params(axis="x", labelrotation=35)
    for lab in axes[1].get_xticklabels():
        lab.set_horizontalalignment("right")
    _categories(axes[1], len(names), least=5)
    _headroom(axes[1])
    if names:
        axes[1].legend(frameon=False, fontsize=7.5, labelcolor=MUTED,
                       loc="upper right")

    grid = s["difficulty_by_level"]
    lv2 = [k for k in ("L0", "L1", "L2", "L3") if k in grid]
    bottom = np.zeros(len(lv2))
    for name in D.LEVELS:
        vals = np.asarray([grid[k].get(name, 0) for k in lv2], float)
        axes[2].bar(lv2, vals, bottom=bottom, color=LEVEL_COLOUR[name],
                    width=0.62, label=name)
        bottom += vals
    _style(axes[2], "Difficulty x complexity — the two axes, kept apart", "",
           "invalid clips")
    _categories(axes[2], len(lv2))
    # Headroom BEFORE the legend, so the legend has somewhere to sit that is
    # not on top of the tallest stack.
    _headroom(axes[2], 0.28)
    axes[2].legend(frameon=False, fontsize=7.5, labelcolor=MUTED, ncol=3,
                   loc="upper center")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fig_severity(plt, s, path) -> None:
    """Is the ladder monotone in what was MEASURED, not just in the knob?"""
    import numpy as np

    order = [b for b in ("weak", "medium", "strong") if b in s["peak_score_by_bin"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    data = [s["peak_score_by_bin"][b] for b in order]
    if data:
        # `tick_labels`, not `labels`: matplotlib renamed it and 3.11 raises on
        # the old spelling rather than warning.
        bp = axes[0].boxplot(data, tick_labels=order, patch_artist=True,
                             widths=0.5, medianprops={"color": INK},
                             flierprops={"marker": ".", "markersize": 3,
                                         "markerfacecolor": MUTED,
                                         "markeredgecolor": "none"})
        for patch, colour in zip(bp["boxes"], (EASY, MODERATE, HARD)):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)
            patch.set_edgecolor(GRID)
        for whisk in bp["whiskers"] + bp["caps"]:
            whisk.set_color(GRID)
    _style(axes[0], "Measured peak score, by declared bin", "",
           "bounded residual")
    _categories(axes[0], len(order), first=1.0)

    bins = [b for b in ("weak", "medium", "strong") if b in s["severity_bins"]]
    bars = axes[1].bar(bins, [s["severity_bins"][b] for b in bins],
                       color=[EASY, MODERATE, HARD][:len(bins)], width=0.62)
    _bar_labels(axes[1], bars)
    _style(axes[1], "Clips per bin", "", "invalid clips")
    _categories(axes[1], len(bins))
    _headroom(axes[1])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fig_coverage(plt, s, path) -> None:
    """Every family and scenario, so a gap is visible rather than inferred."""
    fam = sorted(s["families"].items(), key=lambda kv: kv[1])
    sc = sorted(s["scenarios"].items(), key=lambda kv: kv[1])
    h = max(3.2, 0.22 * max(len(fam), len(sc)) + 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, h))
    for ax, rows, title in ((axes[0], fam, "Invalid clips per family"),
                            (axes[1], sc, "Clips per scenario")):
        names = [k for k, _ in rows]
        ax.barh(names, [v for _, v in rows], color=ACCENT, height=0.66)
        for i, (_, v) in enumerate(rows):
            ax.annotate("%d" % v, (v, i), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=7.5,
                        color=MUTED)
        _style(ax, title, "clips", "")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.yaxis.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def report(root: str, outdir: Optional[str] = None) -> Dict[str, object]:
    """Write the figures and `stats.json`; return the summary."""
    import matplotlib
    matplotlib.use("Agg")                       # headless box, no display
    import matplotlib.pyplot as plt

    from ..scenarios.base import condition_share, CONDITION_CYCLE

    metas = load(root)
    if not metas:
        raise SystemExit("no clips under %s" % root)
    s = summarise(metas)
    out = outdir or os.path.join(root, "stats")
    os.makedirs(out, exist_ok=True)

    shares = {name: condition_share(name)
              for name in dict.fromkeys(CONDITION_CYCLE)}
    _fig_difficulty(plt, s, os.path.join(out, "difficulty.png"))
    _fig_factors(plt, s, os.path.join(out, "difficulty_factors.png"))
    _fig_composition(plt, s, os.path.join(out, "composition.png"), shares)
    _fig_severity(plt, s, os.path.join(out, "severity.png"))
    _fig_coverage(plt, s, os.path.join(out, "coverage.png"))
    with open(os.path.join(out, "stats.json"), "w") as fh:
        json.dump(s, fh, indent=2, sort_keys=True)
    s["outdir"] = out
    return s

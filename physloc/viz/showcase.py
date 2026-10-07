"""Presentation videos: one axis of the dataset per video, for a slide.

`coverage`, `compare`, `grid`, `sheet` and `overlay` answer "is this right";
these answer "what is in it". **Each axis video shows one axis on one scene**
-- the clips side by side differ in that axis and nothing else -- and **there
is one per scenario**, in a folder per axis, so a slide can use any of them:

    <outdir>/
      teaser.mp4               square; the story of the dataset, below
      scenarios.mp4            every built scenario (the one many-scene video)
      families/<scenario>.mp4  ONE scene: valid, then every family staged on it
      severity/<scenario>.mp4  one scene: valid | weak | medium | strong
      levels/<scenario>.mp4    one cell: L0 | L1 | L2 | L3 (each its own scene)
      conditions/<scenario>.mp4  standard | camera | distractors | multi | both
      variants/<scenario>.mp4  one cell at L0: what the sampler varies
      annotations/<scenario>.mp4  one clip, a sliding line through every modality
      picks.json               which sample fills each slot

The only text is a caption in a semi-transparent box at a tile's bottom
right, in capitals. The axis videos are RGB only; annotation appears in
`annotations` and the teaser.

**The teaser** (1080x1080, the clips' own shape) tells the dataset in order:
a carousel of every scenario's valid scene stops on `TEASER_HERO`'s twin; the
hero plays clean, then a sliding line reveals its severity map; its annotation
fades off for a carousel of every family as plain video (the hero's scene
first, then as few others as cover the rest); the line brings the annotation
back with each condition and each level; and a zoom out
to a `TEASER_MOSAIC`-square mosaic, ending on `TEASER_TITLE`. Captions
crossfade, or are wiped by the line with the picture. Every count on it
is read at run time (the taxonomy, or the clips on disk); a slot the run did
not generate is a placeholder tile.

**Picks are a file.** Each run writes `picks.json` (video -> the sample in each
slot) and `--picks` replays it; `--scenario` / `--family` narrow the
per-scenario videos.

mp4 only -- no image files are written.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import taxonomy
from . import compare
from . import overlay as ov
from . import video as vid

W, H = 1920, 1080
MARGIN = 48
TOP = 48
LABEL = 0           # captions sit inside the tile
GAP = 16

BG = (12, 12, 14)
TITLE = (165, 167, 176)
LABEL_INK = (240, 240, 244)
EMPTY = (26, 26, 30)
BADGE_ALPHA = 0.58

#: The modality cycle of `annotations`: (overlay panel, caption).
MODALITIES = (("rgb", "RGB"), ("depth", "depth"), ("flow", "optical flow"),
              ("segmentation", "segmentation"), ("energy", "energy"),
              ("mask", "violation mask"), ("severity", "severity map"),
              ("causal", "causal mask"))

CONDITION_NAMES = {"standard": "standard", "camera": "camera motion",
                   "distractors": "distractors", "multi": "multi-violation",
                   "camera+multi": "camera + multi"}
LEVEL_NAMES = {"L0": "L0 baseline", "L1": "L1 materials", "L2": "L2 HDRI",
               "L3": "L3 scanned objects"}

#: Videos at the top of `outdir`, and the axes that get one video per scenario.
SINGLE = ("teaser", "scenarios")
AXES = ("families", "severity", "levels", "conditions", "variants",
        "annotations")
VIDEOS = SINGLE + AXES

Slot = Dict[str, object]

# ---------------------------------------------------------------- teaser
#: The clip the scenario carousel stops on: one ball struck by a blow that
#: never happened -- a violation anyone reads at a glance.
TEASER_HERO = ("collision", "phantom_impulse")
#: The square the teaser is drawn in, and the mosaic it ends on (per side).
TEASER_SIZE = 1080
TEASER_MOSAIC = 20
#: A carousel's speed, set by position: it holds `start` of its cruise speed
#: for the first `slow` tiles, speeds up to cruise -- one tile per `seconds` --
#: by the time tile `accel` arrives (the sixth), and slows into the stop over
#: the last `decel` tiles. `lead` loops of the first clip play before it moves.
SCROLL = {"start": 0.25, "slow": 3.0, "accel": 5.0, "seconds": 0.35,
          "decel": 3.0, "lead": 1}
#: The family carousel opens slower still -- a violation needs a whole clip
#: to be seen, so each of the first families takes about one loop to pass --
#: then cruises like the scenarios. It opens on the hero, already seen.
#: Seconds the hero's annotation takes to fade off before the families,
#: which play as plain video.
FAMILY_FADE = 0.6
FAMILY_SCROLL = {"start": 0.12, "slow": 4.0, "accel": 6.0, "lead": 0}
#: Seconds a caption takes to crossfade into the next one. Where the sliding
#: line changes the picture, the caption is wiped with it instead.
CAPTION_FADE = 0.4
#: Each condition and level: how many loops it plays, and how many of them
#: the sliding line takes to cross into it.
STEP_LOOPS = 1.5
STEP_WIPE = 0.6
#: The name the teaser ends on, its font (one of matplotlib's bundled ones,
#: so it renders the same wherever the env exists) and how long it holds.
TEASER_TITLE = "PhysLoc"
TITLE_FONT = "DejaVuSans-Bold.ttf"
TITLE_SECONDS = 3.5
ZOOM_SECONDS = 3.0
#: Clip loops the mosaic holds after the zoom.
MOSAIC_LOOPS = 2
#: Fraction of a loop the sliding line takes to cross.
WIPE = 0.5


# ================================================================ text
_FONTS: Dict[int, object] = {}
_SPRITES: Dict[tuple, np.ndarray] = {}


def _font(px: int):
    """DejaVu Sans at `px`, from matplotlib's bundled fonts; None if absent."""
    if px not in _FONTS:
        try:
            from matplotlib import font_manager
            from PIL import ImageFont
            _FONTS[px] = ImageFont.truetype(font_manager.findfont(
                font_manager.FontProperties(family="DejaVu Sans")), px)
        except Exception:                                      # noqa: BLE001
            _FONTS[px] = None
    return _FONTS[px]


def sprite(s: str, px: int, colour=LABEL_INK) -> np.ndarray:
    """`s` as an RGBA array. Cached."""
    key = (s, px, tuple(colour))
    if key in _SPRITES:
        return _SPRITES[key]
    font = _font(px)
    if font is None:                         # no PIL / font: Hershey
        scale = px / 30.0
        img = np.zeros((px + 8, ov._w(s, scale) + 4, 3), np.uint8)
        ov._text(img, s, (2, px + 2), colour, scale, 1, backing=False)
        out = np.dstack([img, (img.max(-1) > 0).astype(np.uint8) * 255])
    else:
        from PIL import Image, ImageDraw
        x0, _, x1, _ = font.getbbox(s)
        im = Image.new("RGBA", (max(x1 - x0, 1), int(px * 1.3)), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((-x0, 0), s, font=font,
                                fill=tuple(colour) + (255,))
        out = np.asarray(im)
    _SPRITES[key] = out
    return out


def blit(f: np.ndarray, spr: np.ndarray, x: int, y: int,
         opacity: float = 1.0) -> None:
    """Alpha-composite an RGBA sprite onto `f` at (x, y), clipped."""
    h, w = spr.shape[:2]
    X0, Y0 = max(x, 0), max(y, 0)
    X1, Y1 = min(x + w, f.shape[1]), min(y + h, f.shape[0])
    if X1 <= X0 or Y1 <= Y0:
        return
    s = spr[Y0 - y:Y1 - y, X0 - x:X1 - x]
    a = s[..., 3:4].astype(np.float32) * (opacity / 255.0)
    f[Y0:Y1, X0:X1] = (f[Y0:Y1, X0:X1] * (1 - a) + s[..., :3] * a).astype(np.uint8)


def text(f, s, x, y, px, colour=LABEL_INK, centre=False) -> None:
    spr = sprite(s, px, colour)
    blit(f, spr, x - spr.shape[1] // 2 if centre else x, y)


def canvas(title: str = "") -> np.ndarray:
    f = np.full((H, W, 3), BG, np.uint8)
    if title:
        text(f, title, MARGIN, 34, 30, TITLE)
    return f


def nice(name: str) -> str:
    return CONDITION_NAMES.get(name, name.replace("_", " "))


# ================================================================ layout
def layout(n: int, x0: int = MARGIN, y0: int = TOP, x1: int = W - MARGIN,
           y1: int = H - MARGIN // 2, gap: int = GAP,
           below: int = LABEL) -> Tuple[int, int, int]:
    """(cols, rows, cell): the largest square cell fitting `n` tiles, each
    with `below` pixels for its label."""
    best = (1, max(n, 1), 0)
    for cols in range(1, max(n, 1) + 1):
        rows = -(-n // cols)
        cell = min((x1 - x0 - (cols - 1) * gap) // cols,
                   (y1 - y0 - (rows - 1) * gap) // rows - below)
        if cell > best[2]:
            best = (cols, rows, cell)
    return best


def grid_positions(n, cols, rows, cell, x0=MARGIN, y0=TOP, x1=W - MARGIN,
                   y1=H - MARGIN // 2, gap=GAP, below=LABEL):
    """Top-left of each of `n` tiles: the block centred, and a short last row
    centred under the others."""
    bh = rows * (cell + below) + (rows - 1) * gap
    oy = y0 + (y1 - y0 - bh) // 2
    out = []
    for i in range(n):
        r, c = divmod(i, cols)
        in_row = min(cols, n - r * cols)
        ox = x0 + (x1 - x0 - (in_row * cell + (in_row - 1) * gap)) // 2
        out.append((ox + c * (cell + gap), oy + r * (cell + below + gap)))
    return out


# ================================================================ clips
def _resize(img: np.ndarray, s: int) -> np.ndarray:
    import cv2
    if img.shape[0] == s and img.shape[1] == s:
        return img
    interp = cv2.INTER_AREA if img.shape[0] > s else cv2.INTER_CUBIC
    return cv2.resize(img, (s, s), interpolation=interp)


class Clip:
    """One sample, opened once, rendered into [T,s,s,3] stacks at tile size.

    `kind` is an `overlay.Renderer` panel name, or "valid" for the twin's RGB.
    The decoded arrays are released by `close` once a video has its stacks.

    **Clips advance by time, not by frame.** Roots can mix tiers -- a 25-frame
    12 fps debug clip beside a 61-frame 30 fps one -- and stepping each one
    source frame per output frame played the first 2.4 times for every play
    of the second, and the second in slow motion. `at(g)` is the source frame
    showing at output frame `g` of a video written at `fps_out`; `n` is one
    play in output frames.
    """

    def __init__(self, rec: Dict[str, object], size: int,
                 fps_out: Optional[float] = None):
        self.rec = rec
        self.size = int(size)
        self.sample = compare._open(rec)
        self.T = int(self.sample.video.num_frames)
        self.fps = float(self.sample.video.fps or rec.get("fps") or 12)
        self.fps_out = float(fps_out or self.fps)
        self.n = max(1, int(round(self.T * self.fps_out / self.fps)))
        self.r = ov.Renderer(self.sample, panels=("rgb",),
                             panel=int(self.sample.video.rgb.shape[1]))
        self._stacks: Dict[str, np.ndarray] = {}

    def at(self, g: int) -> int:
        """The source frame on screen at output frame `g`, looping.

        One play is exactly `n` output frames, so a clip whose length in
        output frames is not whole (25 frames at 12 fps is 62.5 at 30) still
        restarts on a frame boundary instead of drifting across loops."""
        return min(self.T - 1, int((g % self.n) * self.T / self.n))

    def frame(self, kind: str, t: int, size: Optional[int] = None) -> np.ndarray:
        t = t % self.T
        if kind == "valid":
            twin = self.sample.twin
            img = (twin.video.rgb[min(t, twin.video.num_frames - 1)]
                   if twin is not None else self.sample.video.rgb[t])
        else:
            img = self.r._panel(kind, t)
        return _resize(np.asarray(img, np.uint8), size or self.size)

    def stack(self, kind: str) -> np.ndarray:
        if kind not in self._stacks:
            self._stacks[kind] = np.stack(
                [self.frame(kind, t) for t in range(self.T)])
        return self._stacks[kind]

    def close(self) -> None:
        if self.sample.twin is not None:
            self.sample.twin.release()
        self.sample.release()
        self.r = None


def _ease(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def wiped(a: np.ndarray, b: np.ndarray, p: float) -> np.ndarray:
    """`b` left of a line at fraction `p` across, `a` right of it, and the line."""
    s = a.shape[1]
    x = int(round(_ease(p) * s))
    if x <= 0:
        return a
    if x >= s:
        return b
    out = a.copy()
    out[:, :x] = b[:, :x]
    divider(out, x, 0, out.shape[0])
    return out


def divider(f, x: int, y0: int, y1: int) -> None:
    """The sliding line: white, with a soft glow."""
    for half, a in ((5, 0.10), (2, 0.25), (0, 1.0)):
        lo, hi = max(x - half, 0), min(x + half + 1, f.shape[1])
        band = f[y0:y1, lo:hi].astype(np.float32)
        f[y0:y1, lo:hi] = (band * (1 - a) + 255 * a).astype(np.uint8)


# ================================================================ picking
def rank(r) -> tuple:
    """Best first: strong, standard, L0, then the most visible violation --
    area against occlusion, from `sample.json` -- then the lowest seed."""
    vis = r.get("area", 0.0) * (1.0 - min(r.get("occlusion", 0.0), 1.0))
    return (r["severity"] != "strong", r["condition"] != "standard",
            r["level"] != "L0", -vis, r["seed"], r["dir"])


def _fam(rec) -> "taxonomy.Family":
    return taxonomy.FAMILIES[rec["family"]]


def _rigid(rec) -> bool:
    return rec.get("medium", "rigid") == "rigid"


def _slot(rec, label: str, **kw) -> Slot:
    return dict(kw, label=label, uid=None if rec is None else rec["uid"],
                dir=None if rec is None else rec["dir"])


def _want(recs, scenario=None, family=None):
    return [r for r in recs if (not scenario or r["scenario"] == scenario)
            and (not family or r["family"] == family)]


def _built_scenarios() -> List[str]:
    return [s for s in taxonomy.SCENARIOS if s not in taxonomy.UNBUILT]


def _lasting(rec) -> tuple:
    """Sorts first the clips whose severity map stays lit: a `state` family
    is wrong in every frame it lasts, an `event` family only across its
    change -- and the teaser's whole point is the map lighting up."""
    state = rec["family"] in taxonomy.FAMILIES and _fam(rec).detectable == "state"
    return (not state, -round(float(rec.get("duration", 0.0)), 1))


def resolve(slots: Sequence[Slot], recs) -> List[Tuple[Slot, Optional[Dict]]]:
    """Each slot's record: by directory, else by uid (a picks file moved
    between machines keeps its uids)."""
    by_dir = {r["dir"]: r for r in recs}
    by_uid = {r["uid"]: r for r in recs}
    return [(s, by_dir.get(s.get("dir")) or by_uid.get(s.get("uid")))
            for s in slots]


def pick_scenarios(recs, **_) -> List[Slot]:
    """One scene per built scenario, shown as its valid twin."""
    out = []
    for scen in _built_scenarios():
        rows = [r for r in recs if r["scenario"] == scen]
        twinned = [r for r in rows if r.get("valid_dir")] or rows
        r = min(twinned, key=rank) if twinned else None
        out.append(_slot(r, nice(scen), valid=True))
    return out


def pick_family_scene(recs, scenario=None, **_) -> List[Slot]:
    """ONE scene of `scenario`: its valid twin, then every family staged on it.

    A pair directory holds every family on the identical scene, so the video
    is one pair -- the one with the most families, standard and L0 first.
    """
    pairs: Dict[str, List[Dict]] = defaultdict(list)
    for r in _want(recs, scenario):
        if r["severity"] == "strong" and r.get("valid_dir"):
            pairs[os.path.dirname(r["dir"])].append(r)
    if not pairs:
        return []
    p = min(pairs, key=lambda p: (-len(pairs[p]),) + rank(pairs[p][0])[1:3] + (p,))
    order = list(taxonomy.FAMILIES)
    rows = sorted(pairs[p], key=lambda r: order.index(r["family"])
                  if r["family"] in order else len(order))
    return ([_slot(rows[0], "valid", valid=True)]
            + [_slot(r, nice(r["family"])) for r in rows])


def pick_severity(recs, scenario=None, family=None, **_) -> List[Slot]:
    """One scene through every bin. A severity ladder reads best on a rigid,
    graded family that is wrong in every frame it lasts."""
    groups: Dict[tuple, Dict[str, Dict]] = defaultdict(dict)
    for r in _want(recs, scenario, family):
        if r.get("valid_dir"):
            groups[(r["valid_dir"], r["family"])][r["severity"]] = r
    full = [g for g in groups.values()
            if all(b in g for b in taxonomy.SEVERITY_BINS)]
    if not full:
        return []
    g = min(full, key=lambda g: (not _rigid(g["strong"]),
                                 _fam(g["strong"]).detectable != "state",
                                 not _fam(g["strong"]).graded) + rank(g["strong"]))
    return ([_slot(g["strong"], "valid", valid=True)]
            + [_slot(g[b], b) for b in taxonomy.SEVERITY_BINS])


def _best_cell(recs, axis: str, values: Sequence[str], scenario=None,
               family=None, **fixed) -> Tuple[Optional[tuple], Dict[str, Dict]]:
    """The (scenario, family) with the most `values` of `axis` and the best
    clip of each -- rigid first."""
    cells: Dict[tuple, Dict[str, Dict]] = defaultdict(dict)
    for r in _want(recs, scenario, family):
        if r[axis] in values and all(r.get(k) == v for k, v in fixed.items()):
            cur = cells[(r["scenario"], r["family"])].get(r[axis])
            if cur is None or rank(r) < rank(cur):
                cells[(r["scenario"], r["family"])][r[axis]] = r
    if not cells:
        return None, {}
    key = min(cells, key=lambda k: (-len(cells[k]),
                                    not all(_rigid(r) for r in cells[k].values()),
                                    _lasting(min(cells[k].values(), key=rank)),
                                    min(rank(r) for r in cells[k].values())))
    return key, cells[key]


def _across(recs, axis, values, names, **kw) -> List[Slot]:
    key, got = _best_cell(recs, axis, values, **kw)
    return [] if key is None else [_slot(got.get(v), names.get(v, v))
                                   for v in values]


def pick_levels(recs, scenario=None, family=None, **_) -> List[Slot]:
    return _across(recs, "level", compare.LEVELS, LEVEL_NAMES,
                   scenario=scenario, family=family, severity="strong",
                   condition="standard")


def pick_conditions(recs, scenario=None, family=None, **_) -> List[Slot]:
    return _across(recs, "condition", compare.CONDITIONS, CONDITION_NAMES,
                   scenario=scenario, family=family, severity="strong",
                   level="L0")


def pick_variants(recs, scenario=None, family=None, n: int = 6, **_) -> List[Slot]:
    """Up to `n` distinct L0 standard scenes of the cell with the most."""
    seeds: Dict[tuple, Dict[tuple, Dict]] = defaultdict(dict)
    for r in _want(recs, scenario, family):
        if (r["level"], r["condition"], r["severity"]) == ("L0", "standard", "strong"):
            seeds[(r["scenario"], r["family"])].setdefault(
                (r["seed"], r["variant"]), r)
    if not seeds:
        return []
    key = min(seeds, key=lambda k: (-min(len(seeds[k]), n),
                                    not all(_rigid(r) for r in seeds[k].values()),
                                    min(rank(r) for r in seeds[k].values())))
    rows = [seeds[key][k] for k in sorted(seeds[key])][:n]
    return [_slot(r, "variant %d" % (i + 1)) for i, r in enumerate(rows)]


def pick_annotations(recs, scenario=None, family=None, **_) -> List[Slot]:
    """A standard clip whose violation moves another body, so the causal
    mask has its second level to show."""
    rows = _want(recs, scenario, family)
    pool = ([r for r in rows if r["affected"] and r["condition"] == "standard"]
            or [r for r in rows if r["affected"]] or rows)
    return [_slot(min(pool, key=rank), "hero")] if pool else []


def pick_teaser(recs, seed: int = 0, mosaic: int = TEASER_MOSAIC, **_) -> List[Slot]:
    """Every slot of the teaser, in playing order, each tagged with its `role`.

    scenario x12 (valid scenes), hero, family x23 (`_family_reel`),
    condition x5, level x4, mosaic x(n*n-1). The levels continue on the
    conditions' cell when it has them all; the last level is the mosaic's
    centre, where the zoom begins.
    """
    strong = [r for r in recs if r["severity"] == "strong"] or list(recs)
    if not strong:
        return []

    def best(rows, key=lambda r: _lasting(r) + rank(r)):
        return min(rows, key=key) if rows else None

    hero = (best([r for r in strong if (r["scenario"], r["family"]) == TEASER_HERO],
                 key=rank)
            or best([r for r in strong if _rigid(r) and r["condition"] == "standard"])
            or best(strong))
    out = [_slot(best([r for r in strong if r["scenario"] == s
                       and r.get("valid_dir")], key=rank),
                 nice(s), role="scenario", valid=True)
           for s in _built_scenarios() if s != hero["scenario"]]
    out.append(_slot(hero, nice(hero["scenario"]), role="hero"))

    ckey, conds = _best_cell(strong, "condition", compare.CONDITIONS, level="L0")
    lkey, levels = (ckey, {}) if ckey is None else _best_cell(
        strong, "level", compare.LEVELS, *ckey, condition="standard")
    if len(levels) < len(compare.LEVELS):
        lkey, levels = _best_cell(strong, "level", compare.LEVELS,
                                  condition="standard")

    out += [_slot(r, nice(f), role="family") for f, r in _family_reel(strong, hero)]
    out += [_slot(conds.get(c), CONDITION_NAMES[c], role="condition")
            for c in compare.CONDITIONS]
    out += [_slot(levels.get(lv), LEVEL_NAMES[lv], role="level")
            for lv in compare.LEVELS]

    # The mosaic: round-robin over every cell so it shows the breadth, each
    # cell's clips interleaving the levels, the rest placeholders.
    rng = np.random.RandomState(seed)
    centre = levels.get(compare.LEVELS[-1]) if levels else None
    cells: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in strong:
        if centre is None or r["dir"] != centre["dir"]:
            cells[(r["scenario"], r["family"])].append(r)
    queues = []
    for key in [list(cells)[i] for i in rng.permutation(len(cells))]:
        by_level = defaultdict(list)
        for r in sorted(cells[key], key=lambda r: _lasting(r) + rank(r)):
            by_level[r["level"]].append(r)
        mixed = [r for group in zip(*[by_level[lv] + [None] * len(cells[key])
                                      for lv in compare.LEVELS]) for r in group
                 if r is not None]
        queues.append(mixed)
    tiles: List[Dict] = []
    while len(tiles) < mosaic * mosaic - 1 and any(queues):
        for q in queues:
            if q and len(tiles) < mosaic * mosaic - 1:
                tiles.append(q.pop(0))
    tiles += [None] * (mosaic * mosaic - 1 - len(tiles))
    out += [_slot(r, "", role="mosaic") for r in tiles]
    return out


def _family_reel(strong, hero) -> List[Tuple[str, Optional[Dict]]]:
    """(family, clip) for every family, in the order the carousel shows them.

    The hero's own scene first -- every family staged on it, the hero first --
    then as few other scenes as cover the families it lacks (greedy set
    cover over pair directories, rigid and standard first), each showing only
    what is still missing. A family no scene carries is a placeholder.
    """
    order = list(taxonomy.FAMILIES)
    pairs: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for r in strong:
        if r.get("valid_dir"):
            pairs[os.path.dirname(r["dir"])].setdefault(r["family"], r)
    home = pairs.pop(os.path.dirname(hero["dir"]), {})
    reel = [(hero["family"], hero)] + [(f, home[f]) for f in order
                                       if f in home and f != hero["family"]]
    seen = {f for f, _ in reel}
    while True:
        def gain(p):
            return len(set(pairs[p]) - seen)
        best_pair = min(pairs, default=None, key=lambda p: (
            -gain(p), not all(_rigid(r) for r in pairs[p].values()),
            rank(next(iter(pairs[p].values())))[1:3], p))
        if best_pair is None or not gain(best_pair):
            break
        got = pairs.pop(best_pair)
        reel += [(f, got[f]) for f in order if f in got and f not in seen]
        seen |= set(got)
    return reel + [(f, None) for f in order if f not in seen]


AXIS_PICKERS: Dict[str, Callable] = {
    "families": pick_family_scene, "severity": pick_severity,
    "levels": pick_levels, "conditions": pick_conditions,
    "variants": pick_variants, "annotations": pick_annotations,
}


# ================================================================ captions
def badge(s: str, px: int) -> np.ndarray:
    """`s` in capitals on a semi-transparent dark box, as an RGBA sprite.

    A two-line caption ("5 conditions\ncamera motion") puts the first line
    small and dim above the second: the section, then the thing it is on.
    """
    key = ("badge", s, px)
    if key in _SPRITES:
        return _SPRITES[key]
    parts = s.upper().split("\n")
    lines = ([sprite(parts[0], int(px * 0.7), TITLE)] if len(parts) > 1 else []) \
        + [sprite(parts[-1], px, LABEL_INK)]
    padx, pady, lead = max(8, px // 2), max(4, px // 4), max(2, px // 8)
    w = max(l.shape[1] for l in lines) + 2 * padx
    h = sum(l.shape[0] for l in lines) + lead * (len(lines) - 1) + 2 * pady
    out = np.zeros((h, w, 4), np.uint8)
    out[..., 3] = int(255 * BADGE_ALPHA)
    y = pady
    for txt in lines:
        a = txt[..., 3:4].astype(np.float32) / 255.0
        region = out[y:y + txt.shape[0], padx:padx + txt.shape[1]]
        region[..., :3] = (txt[..., :3] * a).astype(np.uint8)
        region[..., 3] = np.maximum(region[..., 3], txt[..., 3])
        y += txt.shape[0] + lead
    _SPRITES[key] = out
    return out


def caption(f, s: str, px: int, x1: int, y1: int, margin: int = 12,
            opacity: float = 1.0) -> None:
    """`badge(s)` in the bottom-right corner of the box ending at (x1, y1)."""
    if s and opacity > 0:
        spr = badge(s, px)
        blit(f, spr, x1 - spr.shape[1] - margin, y1 - spr.shape[0] - margin,
             opacity)


def title_sprite(s: str, px: int) -> np.ndarray:
    """`s` large in `TITLE_FONT`, white, on a soft dark shadow."""
    import cv2
    key = ("title", s, px)
    if key in _SPRITES:
        return _SPRITES[key]
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    font = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts",
                                           "ttf", TITLE_FONT), px)
    x0, y0, x1, y1 = font.getbbox(s)
    pad = px // 3
    im = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - x0, pad - y0), s, font=font, fill=255)
    a = np.asarray(im).astype(np.float32) / 255.0
    shadow = cv2.GaussianBlur(a, (0, 0), px / 10.0) * 0.85
    alpha = np.maximum(a, shadow)
    rgb = (a / np.maximum(alpha, 1e-6))[..., None] * 255.0
    out = np.dstack([np.repeat(rgb, 3, -1), alpha[..., None] * 255.0]).astype(np.uint8)
    _SPRITES[key] = out
    return out


def placeholder(size: int) -> np.ndarray:
    img = np.full((size, size, 3), EMPTY, np.uint8)
    if size >= 120:
        spr = sprite("NOT GENERATED", max(12, size // 22), TITLE)
        blit(img, spr, (size - spr.shape[1]) // 2, (size - spr.shape[0]) // 2)
    return img


def put(f, img, x, y) -> None:
    """Paste `img` with its top-left at (x, y), clipped to `f`."""
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    X0, Y0 = max(x, 0), max(y, 0)
    X1, Y1 = min(x + w, f.shape[1]), min(y + h, f.shape[0])
    if X1 > X0 and Y1 > Y0:
        f[Y0:Y1, X0:X1] = img[Y0 - y:Y1 - y, X0 - x:X1 - x]


# ================================================================ videos
def _fps(pairs) -> int:
    """The output rate: the fastest clip's, so no clip loses frames."""
    return max((int(r["fps"]) for _, r in pairs if r is not None), default=12)


def side_by_side(w: "vid.Writer", pairs) -> None:
    """One scene's clips in a row (or a grid when there are many), RGB only,
    each captioned in its corner. Plays the longest clip once."""
    fps = _fps(pairs)
    cols, rows, s = layout(len(pairs))
    pos = grid_positions(len(pairs), cols, rows, s)
    px = int(min(26, max(14, s / 16)))
    bg = canvas()
    clips = []
    try:
        for (slot, rec), (x, y) in zip(pairs, pos):
            if rec is None:
                bg[y:y + s, x:x + s] = placeholder(s)
                caption(bg, str(slot.get("label", "")), px, x + s, y + s)
                clips.append(None)
                continue
            clip = Clip(rec, s, fps)
            clip.stack("valid" if slot.get("valid") else "rgb")
            clips.append(clip)
        live = [(c, xy, sl) for c, xy, (sl, _) in zip(clips, pos, pairs) if c]
        T = max(c.n for c, _, _ in live)
        for t in range(T):
            f = bg.copy()
            for clip, (x, y), slot in live:
                f[y:y + s, x:x + s] = clip.stack(
                    "valid" if slot.get("valid") else "rgb")[clip.at(t)]
                caption(f, str(slot.get("label", "")), px, x + s, y + s)
            w.append(f)
    finally:
        for c in clips:
            if c is not None:
                c.close()


def video_annotations(w: "vid.Writer", pairs) -> None:
    """One clip, large; per loop the line wipes to the next modality."""
    (_, rec), = pairs[:1]
    big = 1000
    x, y = (W - big) // 2, (H - big) // 2
    clip = Clip(rec, big, _fps(pairs))
    try:
        bg = canvas()
        for k in range(len(MODALITIES)):
            prev, cur = MODALITIES[max(k - 1, 0)], MODALITIES[k]
            for t in range(clip.n):
                f = bg.copy()
                p = t / max(1.0, WIPE * clip.n)
                src = clip.at(t)
                a = clip.frame(prev[0], src, big)
                img = a if k == 0 else wiped(a, clip.frame(cur[0], src, big), p)
                f[y:y + big, x:x + big] = img
                name = cur[1] if k == 0 or _ease(p) > 0.5 else prev[1]
                caption(f, name, 30, x + big, y + big, 20)
                w.append(f)
    finally:
        clip.close()


def scroll_offsets(n: int, step: float, fps: float,
                   profile: Optional[Dict[str, float]] = None) -> List[float]:
    """The scroll offset of each frame of a carousel of `n` tiles `step`
    apart: from 0 to (n-1)*step at the `SCROLL` speed profile.

    Speed is set by position, not time: `start` of cruise for the first
    `slow` tiles, up to cruise by tile `accel`, and down over the last `decel`
    as the square root of the distance left -- constant deceleration, so it
    settles rather than stops. A floor keeps it moving.
    """
    pr = dict(SCROLL, **(profile or {}))
    D = max(0.0, (n - 1) * step)
    cruise = step / pr["seconds"] / fps             # pixels per frame
    out, x = [0.0], 0.0
    while x < D:
        slow = pr["slow"] * step
        up = pr["start"] + (1 - pr["start"]) * _ease(
            (x - slow) / max(1e-9, (pr["accel"] - pr["slow"]) * step))
        down = min(1.0, ((D - x) / max(1e-9, pr["decel"] * step)) ** 0.5)
        x = min(D, x + cruise * max(0.03, min(up, down)))
        out.append(x)
    return out


class _Teaser:
    """The teaser's state: the output, its clock, and the clips it holds.

    Every clip keeps one global clock (`Clip.at(g)`, by time), so a clip that
    carries over from one section to the next never restarts at the seam.
    """

    def __init__(self, w: "vid.Writer", fps: int, counts: Dict[str, int]):
        self.w, self.fps, self.counts = w, fps, counts
        self.S = TEASER_SIZE
        self.g = 0
        self.px = 30
        self.cap = self.prev = ""
        self.since = 0

    def clip(self, rec, size=None) -> Optional[Clip]:
        return None if rec is None else Clip(rec, size or self.S, self.fps)

    def img(self, clip, kind, size=None) -> np.ndarray:
        size = size or self.S
        return placeholder(size) if clip is None else clip.frame(
            kind, clip.at(self.g), size)

    def emit(self, f, text_: str) -> None:
        """Write `f` with `text_` captioned; a new caption crossfades in."""
        if text_ != self.cap:
            self.prev, self.cap, self.since = self.cap, text_, 0
        # Out, then in: two boxes of different widths half-visible on top of
        # each other read as a smudge, so the old one leaves before the new
        # one arrives.
        a = min(1.0, self.since / max(1.0, CAPTION_FADE * self.fps))
        if self.prev and a < 0.5:
            caption(f, self.prev, self.px, self.S, self.S, 24, _ease(1 - 2 * a))
        elif self.prev:
            caption(f, self.cap, self.px, self.S, self.S, 24, _ease(2 * a - 1))
        else:
            caption(f, self.cap, self.px, self.S, self.S, 24, _ease(a))
        self.since += 1
        self.w.append(f)
        self.g += 1

    def emit_burned(self, f, text_: str) -> None:
        """Write `f`, whose caption is already drawn (a wipe carries it):
        `text_` becomes the caption in force, with no fade."""
        self.prev, self.cap, self.since = "", text_, 10 ** 9
        self.w.append(f)
        self.g += 1

    def captioned(self, img, text_: str) -> np.ndarray:
        img = img.copy()
        caption(img, text_, self.px, self.S, self.S, 24)
        return img

    def blank(self) -> np.ndarray:
        return np.full((self.S, self.S, 3), BG, np.uint8)

    def loop(self, clip) -> int:
        return clip.n if clip is not None else int(round(3 * self.fps))

    def carousel(self, recs, kind: str, text_: str,
                 profile: Optional[Dict[str, float]] = None) -> None:
        """Clips side by side, scrolling right to left from the first to the
        last at `scroll_offsets`' speed; at most two are ever on screen."""
        S, step = self.S, self.S + 24
        n = len(recs)
        held: Dict[int, Optional[Clip]] = {}
        pr = dict(SCROLL, **(profile or {}))
        try:
            # The first clip plays `lead` times in place before anything moves.
            if n and pr["lead"]:
                held[0] = self.clip(recs[0])
                for _ in range(int(pr["lead"] * self.loop(held[0]))):
                    self.emit(self.img(held[0], kind), text_)
            for off in scroll_offsets(n, step, self.fps, profile):
                k0 = int(off // step)
                f = self.blank()
                for k in (k0, k0 + 1):
                    x = k * step - off
                    if k >= n or x >= S or x + S <= 0:
                        continue
                    if k not in held:
                        held[k] = self.clip(recs[k])
                    put(f, self.img(held[k], kind), x, 0)
                for k in [k for k in held if k < k0]:
                    gone = held.pop(k)
                    if gone is not None:
                        gone.close()
                self.emit(f, text_)
        finally:
            for c in held.values():
                if c is not None:
                    c.close()

    def wipes(self, cur, seq, prefix: str,
              cur_kind: str = "severity") -> Optional[Clip]:
        """From `cur` (shown as `cur_kind`), the sliding line wipes to each
        clip of `seq` in turn -- picture and caption together -- and each
        plays `STEP_LOOPS` loops, annotation on. Returns the last clip, still
        open."""
        for label, rec in seq:
            nxt = self.clip(rec)
            n = self.loop(nxt)
            before, after = self.cap, "%s\n%s" % (prefix, label)
            for t in range(int(round(STEP_LOOPS * n))):
                a = self.captioned(self.img(cur, cur_kind), before)
                b = self.captioned(self.img(nxt, "severity"), after)
                self.emit_burned(wiped(a, b, t / max(1.0, STEP_WIPE * n)), after)
            if cur is not None:
                cur.close()
            cur, cur_kind = nxt, "severity"
        return cur

    def zoom(self, centre, recs, text_: str) -> None:
        """Zoom out from `centre` to the mosaic around it, then hold it."""
        import cv2
        S, M, gap = self.S, TEASER_MOSAIC, 2
        cell = (S - (M + 1) * gap) // M
        edge = (S - (M * cell + (M - 1) * gap)) // 2
        ci = cj = M // 2
        slots = [(r, c) for r in range(M) for c in range(M) if (r, c) != (ci, cj)]
        tiles: Dict[tuple, Optional[Tuple[Clip, np.ndarray, np.ndarray]]] = {}
        for (r, c), rec in zip(slots, recs):
            if rec is None:
                tiles[(r, c)] = None
                continue
            # Inner rings are seen large early in the zoom, so they keep a
            # bigger copy; every tile also keeps one at mosaic size.
            ring = max(abs(r - ci), abs(c - cj))
            size = int(min(256, max(cell, S // (2 * ring + 1))))
            clip = self.clip(rec, size)
            big = clip.stack("severity")
            small = np.stack([_resize(x, cell) for x in big])
            clip.close()
            tiles[(r, c)] = (clip, big, small)

        def xy(r, c):
            return edge + c * (cell + gap), edge + r * (cell + gap)

        cx, cy = xy(ci, cj)
        cx, cy = cx + cell / 2, cy + cell / 2
        Z = max(2, int(ZOOM_SECONDS * self.fps))
        z0 = S / cell
        for i in range(Z + 1):
            u = _ease(i / Z)
            z = z0 ** (1 - u)
            px, py = S / 2 + (cx - S / 2) * u, S / 2 + (cy - S / 2) * u
            size = max(1, int(round(cell * z)))
            f = self.blank()
            for (r, c), tile in list(tiles.items()) + [((ci, cj), "centre")]:
                x, y = xy(r, c)
                tx = px + (x + cell / 2 - cx) * z - size / 2
                ty = py + (y + cell / 2 - cy) * z - size / 2
                if tx >= S or ty >= S or tx + size <= 0 or ty + size <= 0:
                    continue
                if tile == "centre":
                    img = self.img(centre, "severity", size)
                elif tile is None:
                    img = np.full((size, size, 3), EMPTY, np.uint8)
                else:
                    clip, big, _ = tile
                    img = cv2.resize(big[clip.at(self.g)], (size, size),
                                     interpolation=cv2.INTER_LINEAR)
                put(f, img, tx, ty)
            self.emit(f, text_)
        n = max([t[0].n for t in tiles.values() if t] + [self.loop(centre)])
        held = MOSAIC_LOOPS * n
        title = int(round(TITLE_SECONDS * self.fps))
        name = title_sprite(TEASER_TITLE, S // 6)
        for k in range(held + title):
            f = self.blank()
            for (r, c), tile in tiles.items():
                x, y = xy(r, c)
                f[y:y + cell, x:x + cell] = (
                    EMPTY if tile is None else tile[2][tile[0].at(self.g)])
            x, y = xy(ci, cj)
            f[y:y + cell, x:x + cell] = self.img(centre, "severity", cell)
            if k >= held:
                # The name, last: the mosaic dims behind it as it fades in.
                a = _ease((k - held) / max(1.0, 0.3 * title))
                f = (f.astype(np.float32) * (1 - 0.55 * a)).astype(np.uint8)
                blit(f, name, (S - name.shape[1]) // 2, (S - name.shape[0]) // 2, a)
            self.emit(f, text_)


def video_teaser(w: "vid.Writer", pairs, counts: Dict[str, int]) -> None:
    """Scenarios, the hero clean then annotated, families, conditions,
    levels, and the zoom out to the mosaic -- see the module docstring."""
    by: Dict[str, List] = defaultdict(list)
    for slot, rec in pairs:
        by[str(slot.get("role"))].append((str(slot.get("label", "")), rec))
    hero_rec = next((r for _, r in by["hero"]), None)
    if hero_rec is None:
        return
    t = _Teaser(w, _fps(pairs), counts)

    # 1. Every scenario, right to left, as its valid scene; the last is the
    # hero's own twin, identical to the hero until its violation begins.
    t.carousel([r for _, r in by["scenario"]] + [hero_rec], "valid",
               "%d scenarios" % counts["scenarios"])
    # 2-3. The hero once clean, then the line reveals its severity map --
    # and wipes the caption with it.
    hero = t.clip(hero_rec)
    try:
        for _ in range(hero.n):
            t.emit(t.img(hero, "rgb"), "simulated physical violations")
        before, after = t.cap, "fine-grained violation annotations"
        for k in range(hero.n):
            a = t.captioned(t.img(hero, "rgb"), before)
            b = t.captioned(t.img(hero, "severity"), after)
            t.emit_burned(wiped(a, b, k / max(1.0, WIPE * hero.n)), after)
        # The annotation fades off: the families read best as plain video,
        # where the violation itself is the thing that catches the eye.
        families = "%d violation families" % counts["families"]
        fade = max(1, int(round(FAMILY_FADE * t.fps)))
        for k in range(fade):
            a = _ease((k + 1) / fade)
            img = (t.img(hero, "severity").astype(np.float32) * (1 - a)
                   + t.img(hero, "rgb").astype(np.float32) * a).astype(np.uint8)
            t.emit(img, families)
    finally:
        hero.close()
    # 4. Every family, right to left, as plain video: the hero's scene first.
    fams = [r for _, r in by["family"]]
    t.carousel(fams, "rgb", families, FAMILY_SCROLL)
    # 5-6. The conditions, then the levels, each brought in by the line --
    # which brings the annotation back with the first of them.
    cur = t.clip(fams[-1]) if fams else None
    cur = t.wipes(cur, by["condition"], "%d conditions" % counts["conditions"],
                  cur_kind="rgb")
    cur = t.wipes(cur, by["level"], "%d complexity levels" % counts["levels"])
    # 7. Zoom out to everything, then the name.
    try:
        t.zoom(cur, [r for _, r in by["mosaic"]],
               "{:,} videos".format(counts["videos"]))
    finally:
        if cur is not None:
            cur.close()


def render(name: str, w: "vid.Writer", pairs, counts) -> None:
    axis = name.split("/")[0]
    if axis == "teaser":
        video_teaser(w, pairs, counts)
    elif axis == "annotations":
        video_annotations(w, pairs)
    else:
        side_by_side(w, pairs)


# ================================================================ driver
def counts_of(recs) -> Dict[str, int]:
    """The teaser's numbers: the taxonomy's, and the videos actually on disk."""
    return {"scenarios": len(_built_scenarios()),
            "families": len(taxonomy.FAMILIES),
            "conditions": len(compare.CONDITIONS),
            "levels": len(compare.LEVELS),
            "videos": len(recs) + len({r["valid_dir"] for r in recs
                                       if r.get("valid_dir")})}


def build(roots: Sequence[str], outdir: str,
          only: Optional[Sequence[str]] = None, picks: Optional[str] = None,
          seed: int = 0, scenario: Optional[str] = None,
          family: Optional[str] = None) -> Dict[str, object]:
    """Every showcase video the roots can fill, plus `picks.json`.

    The axis videos are made once per built scenario (`--scenario` narrows
    that to one, `--family` picks the family where the axis allows); a
    `picks` file replays (or curates) any slot.
    """
    recs = compare.index(roots)
    if not recs:
        raise ValueError("no invalid samples under %s" % ", ".join(roots))
    for name in only or ():
        if name not in VIDEOS:
            raise ValueError("unknown video %r; expected one of %s"
                             % (name, ", ".join(VIDEOS)))
    replay: Dict[str, List[Slot]] = {}
    if picks:
        with open(picks) as fh:
            replay = json.load(fh)
    # A partial run keeps the other videos' picks -- the replayed file's, or
    # else the ones already in `outdir` -- so redoing one video never
    # forgets the rest.
    chosen: Dict[str, List[Slot]] = {}
    previous = os.path.join(outdir, "picks.json")
    if not picks and os.path.exists(previous):
        with open(previous) as fh:
            chosen = json.load(fh)
    chosen.update(replay)

    jobs: Dict[str, List[Slot]] = {}
    skipped: Dict[str, str] = {}
    for name in only or VIDEOS:
        if name in SINGLE:
            jobs[name] = replay.get(name) or (
                pick_teaser(recs, seed=seed) if name == "teaser"
                else pick_scenarios(recs))
            continue
        for scen in [scenario] if scenario else _built_scenarios():
            key = "%s/%s" % (name, scen)
            slots = replay.get(key) or AXIS_PICKERS[name](
                recs, scenario=scen, family=family)
            if slots:
                jobs[key] = slots
            else:
                skipped[key] = "nothing generated for this scenario"

    counts = counts_of(recs)
    made = {}
    for key, slots in jobs.items():
        chosen[key] = slots
        pairs = resolve(slots, recs)
        if not any(r is not None for _, r in pairs):
            skipped[key] = "nothing generated to draw"
            continue
        path = os.path.join(outdir, key + ".mp4")
        with vid.Writer(path, fps=_fps(pairs), block=2) as w:
            render(key, w, pairs, counts)
        made[key] = {"path": path, "frames": w.frames,
                     "tiles": sum(r is not None for _, r in pairs)}
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "picks.json"), "w") as fh:
        json.dump(chosen, fh, indent=1)
    return {"outdir": outdir, "clips_indexed": len(recs), "made": made,
            "skipped": skipped}
